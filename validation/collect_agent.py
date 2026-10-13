"""Run three real OpenCode user turns and retain raw client events."""
import argparse
import json
from pathlib import Path
import subprocess

from validation.run_agent import prepare_run, create_inputs, turn_command, run_process, analyze_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18180/v1")
    parser.add_argument("--model", default="replay-lab-qwen3-8b")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--binary", default="/root/.opencode/bin/opencode")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    prepared = prepare_run(args.base_url, args.model, args.run_dir)
    root = prepared["run_dir"]
    prompts = create_inputs(prepared["workspace"])
    version = subprocess.check_output([args.binary, "--version"], env=prepared["env"], text=True).strip()
    # OpenCode v2 debug config uses its managed service even with isolated XDG paths.
    # Only `run --standalone` is permitted here; keep the generated config as the audit record.
    (root / "user-turns.json").write_text(json.dumps(prompts, indent=2) + "\n")
    results, session = [], None
    for index, prompt in enumerate(prompts):
        stdout, stderr = root / f"turn-{index+1}.jsonl", root / f"turn-{index+1}.stderr"
        command = turn_command(args.binary, args.model, prompt, session)
        print(f"[agent] turn={index+1} session={session or 'new'}", flush=True)
        run_process(command, prepared, stdout, stderr, args.timeout)
        raw = stdout.read_text()
        event_sessions = {json.loads(line).get("sessionID") for line in raw.splitlines() if line.strip()}
        if len(event_sessions) != 1 or not all(isinstance(s, str) and s.startswith("ses_") for s in event_sessions):
            raise ValueError("missing or inconsistent CLI session IDs")
        current_session = event_sessions.pop()
        exported = root / f"turn-{index+1}.export.json"
        run_process([args.binary, "session", "export", "--standalone", current_session], prepared,
                    exported, root / f"turn-{index+1}.export.stderr", 30)
        result = analyze_events(raw, prepared["workspace"], expected_session=session,
            required_file=("inventory.txt", "policy.txt", None)[index],
            session_export=json.loads(exported.read_text()))
        session = result["session_id"]
        results.append(result)
        (root / "session.json").write_text(json.dumps(dict(version=version, session_id=session,
            user_turns=len(results), turns=results, base_url=args.base_url, model=args.model), indent=2) + "\n")
        print(f"[agent] completed turn={index+1} executed_tools={len(result['tool_calls'])}", flush=True)
    calls = [call for turn in results for call in turn["tool_calls"]]
    if len(calls) < 2:
        raise SystemExit("[FAILED] agent did not execute both fixture reads")
    print(f"[ok] real agent: user_turns={len(results)} executed_tool_calls={len(calls)} session={session}", flush=True)


if __name__ == "__main__":
    main()
