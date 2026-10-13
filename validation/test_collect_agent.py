"""CPU test for the collection entry point, not a fabricated model run."""
from pathlib import Path
import importlib.util
import pytest


def test_all_agent_processes_use_private_server(monkeypatch, tmp_path):
    from validation import collect_agent
    monkeypatch.setattr("sys.argv", ["collect_agent", "--run-dir", str(tmp_path / "agent")])
    monkeypatch.setattr(collect_agent.subprocess, "check_output", lambda *a, **k: "unit-version")
    class ReachedPrivateProcess(Exception):
        pass
    def verify(command, *args):
        assert "--standalone" in command, "must never start or contact the user's managed OpenCode service"
        raise ReachedPrivateProcess
    monkeypatch.setattr(collect_agent, "run_process", verify)
    with pytest.raises(ReachedPrivateProcess):
        collect_agent.main()


def test_collection_entry_point_has_a_runnable_main():
    path = Path(__file__).with_name("collect_agent.py")
    assert path.exists(), "real-agent collection CLI is missing"
    spec = importlib.util.spec_from_file_location("collect_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
