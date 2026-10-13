#!/usr/bin/env python3
"""Collect a real OpenCode session against an explicitly local model endpoint."""
import ipaddress
import json
import os
import signal
import subprocess
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit


def prepare_run(base_url, model, run_dir, context=8192, output=1024):
    url = urlsplit(base_url)
    try:
        local = url.hostname == "localhost" or ipaddress.ip_address(url.hostname or "").is_loopback
        port = url.port
    except ValueError:
        local = False
        port = None
    if (not local or url.scheme not in ("http", "https") or url.username is not None
            or url.password is not None or url.query or url.fragment or port == 0):
        raise ValueError("base URL must be loopback HTTP(S), without credentials, query, or fragment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]*", model):
        raise ValueError("model must be an exact nonempty model ID, without whitespace or #variant")
    if not isinstance(context, int) or not isinstance(output, int) or not (256 <= context and 0 < output < context):
        raise ValueError("token limits must satisfy context >= 256 and 0 < output < context")
    run_dir = Path(run_dir).expanduser().absolute()
    if run_dir.is_symlink() or (run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir()))):
        raise ValueError("run directory must be new or empty, and must not be a symlink")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "SHELL": "/bin/bash"}
    for key, suffix in {
        "HOME": "home",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "XDG_CACHE_HOME": "cache",
        "XDG_STATE_HOME": "state",
        "XDG_RUNTIME_DIR": "runtime",
        "TMPDIR": "tmp",
        "OPENCODE_CONFIG_DIR": "config/opencode",
    }.items():
        path = run_dir / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    workspace = run_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    config_path = Path(env["OPENCODE_CONFIG_DIR"]) / "opencode.json"
    model_ref = f"lab/{model}"
    permissions = [
        {"action": "*", "resource": "*", "effect": "deny"},
        *({"action": "read", "resource": name, "effect": "allow"}
          for name in ("inventory.txt", "policy.txt")),
    ]
    config = {
        "model": model_ref,
        "default_agent": "lab",
        "update": "disable",
        "share": "disabled",
        "snapshots": False,
        "warming": False,
        "lsp": False,
        "formatter": False,
        "compaction": {"auto": False},
        "permissions": permissions,
        "plugins": [
            "-*", "opencode.config.agent", "opencode.config.provider",
            "opencode.config.policy", "opencode.config.compaction", "opencode.tool.read",
        ],
        "agents": {
            "lab": {
                "mode": "primary", "model": model_ref, "steps": 6,
                "system": "You are a read-only fixture inspection agent. Use the read tool when asked; never guess file contents. Do not edit, run commands, delegate, browse, or access other files. Keep answers concise.",
            },
            **{name: {"model": model_ref} for name in ("title", "summary", "compaction")},
        },
        "providers": {
            "lab": {
                "name": "Local validation endpoint", "env": [],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": base_url, "apiKey": "local-only"},
                "body": {"max_tokens": output, "temperature": 0},
                "models": {
                    model: {
                        "modelID": model, "name": model,
                        "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                        "limit": {"context": context, "output": output},
                    },
                },
            },
        },
        "experimental": {"policies": [
            {"action": "provider.use", "resource": "*", "effect": "deny"},
            {"action": "provider.use", "resource": "lab", "effect": "allow"},
            {"action": "permission", "resource": "*", "effect": "deny"},
            *({"action": "permission", "resource": f"read:{name}", "effect": "allow"}
              for name in ("inventory.txt", "policy.txt")),
        ]},
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    env.update({
        "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_CONFIG_PROJECT_DISABLE": "1", "OPENCODE_DISABLE_FILEWATCHER": "1",
    })
    return {"env": env, "workspace": workspace, "config_path": config_path, "run_dir": run_dir}


def create_inputs(workspace):
    """Seed task inputs only. Assistant messages/tool results come solely from OpenCode."""
    workspace = Path(workspace)
    for name, content in {
        "inventory.txt": "Screws: 7\nWashers: 3\n",
        "policy.txt": "Minimum stock:\nScrews: 5\nWashers: 6\n",
    }.items():
        (workspace / name).write_text(content + f"fixture-marker: {secrets.token_hex(8)}\n", encoding="utf-8")
    return [
        f"Use the read tool now to read {workspace / 'inventory.txt'}. Report the items and quantities from the file. Do not guess or read other files.",
        f"Continue this same session. Use the read tool now to read {workspace / 'policy.txt'}. Report its minimum stock requirements. Do not guess or read other files.",
        "Using the two earlier tool results in this session, say which stock requirements are met and which are not. Cite inventory.txt and policy.txt. Do not make another tool call.",
    ]


def turn_command(binary, model, prompt, session_id=None):
    command = [str(binary), "run", "--standalone", "--format", "json",
               "--model", f"lab/{model}", "--agent", "lab"]
    if session_id:
        command += ["--session", session_id]
    else:
        command += ["--title", "Read-only forced-replay validation"]
    return command + [prompt]


def run_process(command, prepared, stdout_path, stderr_path, timeout):
    """No shell, no PTY, no managed/global OpenCode service; preserve exact bytes."""
    with Path(stdout_path).open("wb") as stdout, Path(stderr_path).open("wb") as stderr:
        process = subprocess.Popen(command, cwd=prepared["workspace"], env=prepared["env"],
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            raise RuntimeError(f"command interrupted or timed out after {timeout}s; see {stdout_path} and {stderr_path}") from None
        finally:
            # This new session is ours, including descendants left by an exited
            # parent. Reaping the parent does not prove its process group is gone.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
    if code:
        raise RuntimeError(f"command exited {code}; see {stdout_path} and {stderr_path}")


def analyze_events(raw, workspace, expected_session=None, required_file=None, session_export=None):
    """Reject incomplete/error traces; count executed reads, never assistant claims."""
    workspace = Path(workspace).resolve()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("missing or non-object OpenCode JSON events")
    sessions = {event.get("sessionID") for event in events}
    if len(sessions) != 1 or not all(isinstance(s, str) and s.startswith("ses_") for s in sessions):
        raise ValueError("missing or inconsistent session ID")
    session_id = sessions.pop()
    if expected_session and session_id != expected_session:
        raise ValueError("OpenCode did not resume the same session")
    calls = {}
    stopped = False
    for event in events:
        part = event.get("part", {})
        if part.get("sessionID", session_id) != session_id:
            raise ValueError("part belongs to another session")
        if event.get("type") == "error":
            raise ValueError(f"OpenCode error: {event.get('error')}")
        if event.get("type") == "step_finish":
            reason = part.get("reason")
            if reason == "length":
                raise ValueError("model exhausted its output limit")
            stopped = stopped or reason == "stop"
        if event.get("type") != "tool_use":
            continue
        state = part.get("state", {})
        if part.get("tool") != "read" or state.get("status") != "completed":
            raise ValueError("trace contains a failed or non-read tool call")
        path = state.get("input", {}).get("path")
        if not isinstance(path, str):
            raise ValueError("read event is missing its input path")
        path = (workspace / path).resolve()
        if path not in {workspace / "inventory.txt", workspace / "policy.txt"}:
            raise ValueError("read event accessed a non-fixture file")
        output = state.get("output", "")
        if not isinstance(output, str) or not all(line in output for line in path.read_text(encoding="utf-8").splitlines()):
            raise ValueError("completed read output does not contain the fixture contents")
        identity = (part.get("messageID"), part.get("id"))
        if not all(identity):
            raise ValueError("tool event is missing call/message identifiers")
        calls[identity] = {"message_id": identity[0], "call_id": identity[1], "tool": "read", "path": str(path)}
    completion_source = "cli_step_finish"
    if not stopped and session_export is not None:
        # v2 run may close its event stream before the final step_finish arrives.
        # Require persisted completion for this exact session and latest CLI message.
        info = session_export.get("info", {})
        assistants = [m for m in session_export.get("messages", []) if m.get("type") == "assistant"]
        final = assistants[-1] if assistants else {}
        latest_id = events[-1].get("part", {}).get("messageID")
        stopped = (info.get("id") == session_id and info.get("outcome") == "succeeded"
                   and final.get("id") == latest_id and final.get("finish") == "stop"
                   and bool(final.get("time", {}).get("completed")))
        completion_source = "session_export"
    if not stopped:
        raise ValueError("turn has no successful final step_finish or matching persisted completion")
    if required_file and not any(call["path"] == str(workspace / required_file) for call in calls.values()):
        raise ValueError(f"turn did not execute a completed read of {required_file}")
    return {"session_id": session_id, "tool_calls": list(calls.values()), "event_count": len(events),
            "completion_source": completion_source}
