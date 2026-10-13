"""Offline tests; never start a server or synthesize agent/model output."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import sys
import socket
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("run_agent.py")


def runner_module():
    if not SOURCE.exists():
        raise AssertionError("run_agent.py must implement the isolated runner")
    spec = importlib.util.spec_from_file_location("isolated_run_agent", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    def test_fixture_inputs_and_three_prompts_do_not_reveal_file_contents(self):
        runner = runner_module()
        self.assertTrue(hasattr(runner, "create_inputs"), "fixture/user-turn inputs are missing")
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            prompts = runner.create_inputs(workspace)
            self.assertEqual(len(prompts), 3)
            for name in ("inventory.txt", "policy.txt"):
                self.assertTrue((workspace / name).is_file())
                content = (workspace / name).read_text(encoding="utf-8")
                self.assertIn("fixture-marker:", content)
                self.assertNotIn(content, "\n".join(prompts))
            self.assertIn(str(workspace / "inventory.txt"), prompts[0])
            self.assertIn(str(workspace / "policy.txt"), prompts[1])
            self.assertIn("earlier", prompts[2])

    def test_real_subprocess_output_is_saved(self):
        runner = runner_module()
        self.assertTrue(hasattr(runner, "run_process"), "bounded subprocess runner is missing")
        with tempfile.TemporaryDirectory() as root:
            prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "replay-lab-qwen3-8b", Path(root) / "run")
            stdout = prepared["run_dir"] / "process.stdout"
            stderr = prepared["run_dir"] / "process.stderr"
            runner.run_process([sys.executable, "-c", "import sys; print('CPU subprocess'); print('diagnostic', file=sys.stderr)"], prepared, stdout, stderr, 5)
            self.assertEqual(stdout.read_bytes(), b"CPU subprocess\n")
            self.assertEqual(stderr.read_bytes(), b"diagnostic\n")

    def test_real_subprocess_timeout_is_bounded(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", Path(root) / "run")
            stdout = prepared["run_dir"] / "process.stdout"
            stderr = prepared["run_dir"] / "process.stderr"
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                runner.run_process([sys.executable, "-c", "import time; time.sleep(60)"], prepared, stdout, stderr, 0.1)
            self.assertLess(time.monotonic() - started, 10)
            # Output is checked separately; startup need not finish before timeout.

    def test_real_subprocess_nonzero_exit_is_reported(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", Path(root) / "run")
            stdout = prepared["run_dir"] / "process.stdout"
            stderr = prepared["run_dir"] / "process.stderr"
            with self.assertRaisesRegex(RuntimeError, "command exited 3"):
                runner.run_process([sys.executable, "-c", "raise SystemExit(3)"], prepared, stdout, stderr, 5)

    def test_cleanup_tolerates_process_group_disappearance_races(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", Path(root) / "run")
            stdout = prepared["run_dir"] / "process.stdout"
            stderr = prepared["run_dir"] / "process.stderr"
            original_killpg = os.killpg

            def disappearing_group(pgid, sig):
                self.assertNotEqual(pgid, os.getpgrp())
                original_killpg(pgid, sig)
                # Inject only the signal/exit race; process and output remain real.
                raise ProcessLookupError("owned process group just exited")

            with patch.object(runner.os, "killpg", side_effect=disappearing_group):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    runner.run_process([sys.executable, "-c", "import time; time.sleep(60)"], prepared, stdout, stderr, 0.1)

    def test_successful_parent_exit_cleans_owned_descendant_group(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", Path(root) / "run")
            stdout = prepared["run_dir"] / "process.stdout"
            stderr = prepared["run_dir"] / "process.stderr"
            lock_path = prepared["workspace"] / "descendant.lock"
            # The pipe guarantees the child holds its lock and ignores SIGTERM
            # before the parent exits. A released lock proves it stopped running,
            # even on hosts whose PID 1 does not immediately reap orphan zombies.
            script = """
import fcntl, os, signal, time
read_fd, write_fd = os.pipe()
if os.fork() == 0:
    os.close(read_fd)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open('descendant.lock', 'wb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.write(write_fd, b'1')
        os.close(write_fd)
        time.sleep(60)
    os._exit(0)
os.close(write_fd)
assert os.read(read_fd, 1) == b'1'
os.close(read_fd)
print(os.getpid(), flush=True)
"""
            def lock_is_released():
                with lock_path.open("rb") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return True
                    except BlockingIOError:
                        return False

            try:
                started = time.monotonic()
                runner.run_process([sys.executable, "-c", script], prepared, stdout, stderr, 5)
                self.assertLess(time.monotonic() - started, 10, "group cleanup must be bounded")
                deadline = time.monotonic() + 2
                while not lock_is_released() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(lock_is_released(), "successful parent exit left a live descendant")
            finally:
                # Only the new session created by this test; never user services.
                if stdout.exists() and stdout.read_bytes().strip():
                    try:
                        os.killpg(int(stdout.read_bytes()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 5
                    while not lock_is_released() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(lock_is_released(), "test descendant cleanup failed")

    def test_analyze_events_counts_only_completed_unique_fixture_reads(self):
        runner = runner_module()
        self.assertTrue(hasattr(runner, "analyze_events"), "real JSON event validator is missing")
        # Schema-only unit inputs, not collection output or assistant responses.
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            fixture = workspace / "inventory.txt"
            fixture.write_text("fixture input\n", encoding="utf-8")
            event = {"type": "tool_use", "sessionID": "ses_unit", "part": {
                "id": "tool_unit", "messageID": "msg_unit", "sessionID": "ses_unit", "tool": "read",
                "state": {"status": "completed", "input": {"path": str(fixture)}, "output": "1: fixture input\n"},
            }}
            finish = {"type": "step_finish", "sessionID": "ses_unit", "part": {"reason": "stop"}}
            raw = "\n".join(json.dumps(e) for e in (event, event, finish)) + "\n"
            result = runner.analyze_events(raw, workspace, required_file="inventory.txt")
            self.assertEqual(result["session_id"], "ses_unit")
            self.assertEqual(len(result["tool_calls"]), 1)
            self.assertEqual(result["tool_calls"][0]["path"], str(fixture))
            event["part"]["state"]["status"] = "error"
            with self.assertRaises(ValueError):
                runner.analyze_events(json.dumps(event), workspace, required_file="inventory.txt")

    def test_analyze_events_reads_utf8_fixtures_without_locale_default(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            content = "库存：螺丝 7\n"
            fixture = workspace / "inventory.txt"
            fixture.write_bytes(content.encode("utf-8"))
            # Schema-only validator inputs, never collected agent output.
            events = [
                {"type": "tool_use", "sessionID": "ses_unit", "part": {
                    "id": "tool_unit", "messageID": "msg_unit", "tool": "read",
                    "state": {"status": "completed", "input": {"path": str(fixture)}, "output": content}}},
                {"type": "step_finish", "sessionID": "ses_unit", "part": {"reason": "stop"}},
            ]
            original_open = Path.open

            def ascii_default(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
                if "b" not in mode and encoding in (None, "locale"):
                    encoding = "ascii"
                return original_open(path, mode, buffering, encoding, errors, newline)

            with patch.object(Path, "open", ascii_default):
                result = runner.analyze_events("\n".join(json.dumps(e) for e in events), workspace)
            self.assertEqual(len(result["tool_calls"]), 1)

    def test_v2_cli_missing_final_step_uses_verified_session_export(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            (workspace / "inventory.txt").write_text("unit input\n", encoding="utf-8")
            tool = {"type": "tool_use", "sessionID": "ses_unit", "part": {
                "id": "tool_unit", "messageID": "msg_tool", "tool": "read",
                "state": {"status": "completed", "input": {"path": str(workspace / "inventory.txt")}, "output": "unit input"}}}
            text = {"type": "text", "sessionID": "ses_unit", "part": {"messageID": "msg_final", "text": "unit response"}}
            raw = "\n".join(json.dumps(row) for row in (tool, text))
            export = {"info": {"id": "ses_unit", "outcome": "succeeded"}, "messages": [
                {"id": "msg_final", "type": "assistant", "finish": "stop", "time": {"completed": 1}}]}
            result = runner.analyze_events(raw, workspace, required_file="inventory.txt", session_export=export)
            self.assertEqual(result["completion_source"], "session_export")
            export["messages"][0]["id"] = "msg_old"
            with self.assertRaises(ValueError):
                runner.analyze_events(raw, workspace, session_export=export)

    def test_analyze_events_rejects_missing_tools_bad_json_errors_and_session_drift(self):
        runner = runner_module()
        self.assertTrue(hasattr(runner, "analyze_events"), "real JSON event validator is missing")
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            finish = {"type": "step_finish", "sessionID": "ses_unit", "part": {"reason": "stop"}}
            invalid = ["", "not JSON", "[]", json.dumps({"type": "error", "sessionID": "ses_unit", "error": {"message": "unit error"}}), json.dumps(finish)]
            for raw in invalid:
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    runner.analyze_events(raw, workspace, required_file="inventory.txt")
            with self.assertRaises(ValueError):
                runner.analyze_events(json.dumps(finish), workspace, expected_session="ses_other")
            finish["part"]["reason"] = "length"
            with self.assertRaises(ValueError):
                runner.analyze_events(json.dumps(finish), workspace)
            shell = {"type": "tool_use", "sessionID": "ses_unit", "part": {"tool": "shell", "state": {"status": "completed"}}}
            with self.assertRaises(ValueError):
                runner.analyze_events(json.dumps(shell), workspace)

    def test_turn_commands_resume_one_explicit_session_without_autoapprove(self):
        runner = runner_module()
        self.assertTrue(hasattr(runner, "turn_command"), "three-turn command builder is missing")
        for index in range(3):
            session = None if index == 0 else "ses_unit_test"
            command = runner.turn_command("/root/.opencode/bin/opencode", "replay-lab-qwen3-8b", f"user turn {index + 1}", session)
            self.assertEqual(command[:5], ["/root/.opencode/bin/opencode", "run", "--standalone", "--format", "json"])
            self.assertEqual(command[-1], f"user turn {index + 1}")
            self.assertIn("lab/replay-lab-qwen3-8b", command)
            self.assertIn("--agent", command)
            self.assertNotIn("--auto", command)
            self.assertNotIn("--continue", command)
            self.assertNotIn("--fork", command)
            if session:
                self.assertEqual(command[command.index("--session") + 1], session)
            else:
                self.assertNotIn("--session", command)

    def test_fixture_permissions_use_v2_workspace_relative_resources(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "replay-lab-qwen3-8b", Path(root) / "run")
            config = json.loads(prepared["config_path"].read_text(encoding="utf-8"))
            self.assertEqual([p["resource"] for p in config["permissions"][1:]], ["inventory.txt", "policy.txt"])
            self.assertEqual([p["resource"] for p in config["experimental"]["policies"][3:]], ["read:inventory.txt", "read:policy.txt"])

    def test_prepare_run_rejects_remote_urls_before_writing(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            for index, url in enumerate(("https://api.openai.com/v1", "http://192.168.1.2/v1", "http://localhost.evil/v1", "http://user:pass@127.0.0.1/v1", "http://127.0.0.1/v1?key=oops", "http://127.0.0.1:70000/v1")):
                with self.subTest(url=url):
                    target = Path(root) / str(index)
                    with self.assertRaises(ValueError):
                        runner.prepare_run(url, "Qwen3-8B", target)
                    self.assertFalse(target.exists())

    def test_prepare_run_refuses_contaminated_or_symlinked_run_dirs(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            existing = Path(root) / "existing"
            existing.mkdir()
            (existing / "opencode.json").write_text("user-config", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", existing)
            self.assertEqual((existing / "opencode.json").read_text(encoding="utf-8"), "user-config")
            link = Path(root) / "link"
            link.symlink_to(existing, target_is_directory=True)
            with self.assertRaises(ValueError):
                runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", link)

    def test_prepare_run_rejects_bad_model_and_token_limits(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            for index, values in enumerate((("", 8192, 1024), ("bad#variant", 8192, 1024), ("bad model", 8192, 1024), ("Qwen3-8B", 100, 1024), ("Qwen3-8B", 8192, 0))):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    runner.prepare_run("http://127.0.0.1:18180/v1", values[0], Path(root) / str(index), context=values[1], output=values[2])

    def test_config_exposes_only_local_provider_and_read_tools(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", Path(root) / "run")
            config = json.loads(prepared["config_path"].read_text(encoding="utf-8"))
            self.assertIn("model", config, "local-only OpenCode config is not implemented")
            self.assertEqual(config["model"], "lab/Qwen3-8B")
            self.assertEqual(set(config["providers"]), {"lab"})
            provider = config["providers"]["lab"]
            self.assertEqual(provider["settings"]["baseURL"], "http://127.0.0.1:18180/v1")
            self.assertEqual(provider["env"], [])
            self.assertEqual(provider["package"], "@opencode/ai/providers/openai-compatible")
            self.assertEqual(provider["models"]["Qwen3-8B"]["limit"], {"context": 8192, "output": 1024})
            self.assertEqual(provider["body"]["max_tokens"], 1024)
            self.assertEqual(config["permissions"][0], {"action": "*", "resource": "*", "effect": "deny"})
            self.assertEqual([p["action"] for p in config["permissions"] if p["effect"] == "allow"], ["read", "read"])
            self.assertEqual(config["plugins"][0], "-*")
            self.assertEqual([p for p in config["plugins"] if p.startswith("opencode.tool.")], ["opencode.tool.read"])
            self.assertEqual(config["experimental"]["policies"][:2], [
                {"action": "provider.use", "resource": "*", "effect": "deny"},
                {"action": "provider.use", "resource": "lab", "effect": "allow"},
            ])
            self.assertEqual(config["agents"]["lab"]["model"], "lab/Qwen3-8B")
            for agent in ("title", "summary", "compaction"):
                self.assertEqual(config["agents"][agent]["model"], "lab/Qwen3-8B")
            for key in ("OPENCODE_DISABLE_MODELS_FETCH", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_CONFIG_PROJECT_DISABLE", "OPENCODE_DISABLE_FILEWATCHER"):
                self.assertEqual(prepared["env"][key], "1")

    def test_prepare_run_isolates_paths_and_drops_inherited_credentials(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "run"
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "do-not-inherit",
                "OPENCODE_CONFIG": "/root/user-config.json",
                "OPENCODE_CONFIG_CONTENT": "malicious",
                "HTTP_PROXY": "http://remote.example",
                "AWS_ACCESS_KEY_ID": "do-not-inherit",
            }):
                prepared = runner.prepare_run("http://127.0.0.1:18180/v1", "Qwen3-8B", run_dir)
            env = prepared["env"]
            for key in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "TMPDIR", "OPENCODE_CONFIG_DIR"):
                self.assertTrue(Path(env[key]).is_relative_to(run_dir), key)
                self.assertTrue(Path(env[key]).is_dir(), key)
            for key in ("OPENAI_API_KEY", "OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT", "HTTP_PROXY", "AWS_ACCESS_KEY_ID"):
                self.assertNotIn(key, env)
            self.assertTrue(prepared["workspace"].is_relative_to(run_dir))
            self.assertTrue(prepared["config_path"].is_file())


if __name__ == "__main__":
    unittest.main()
