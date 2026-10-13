import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_processor_reference_loads_with_the_installed_vllm_loader():
    from validation import lab, capture
    from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns
    assert _load_logitsprocs_by_fqcns([lab.PROCESSOR]) == [capture.ReplayProcessor]


def implementation():
    assert Path(__file__).with_name("lab.py").exists(), "validation launcher is missing"
    return importlib.import_module("validation.lab")


def test_parent_api_key_cannot_block_isolated_lab_requests(monkeypatch, tmp_path):
    import os
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware
    module = implementation()
    parent_key = "unit-parent-key-not-a-real-credential"
    monkeypatch.setenv("VLLM_API_KEY", parent_key)
    env = module.lab_environment(tmp_path, 0)
    app = FastAPI()
    async def probe():
        return {"ok": True}
    app.add_api_route("/v1/probe", probe, methods=["POST"])
    # Same env-driven auth branch as installed vLLM api_server.build_app.
    if env.get("VLLM_API_KEY"):
        app.add_middleware(AuthenticationMiddleware, tokens=[env["VLLM_API_KEY"]])
    with TestClient(app) as client:
        response = client.post("/v1/probe", headers={"Authorization": "Bearer local-only"})
    assert response.status_code == 200
    assert "VLLM_API_KEY" not in env
    assert os.environ["VLLM_API_KEY"] == parent_key


def test_replay_restores_resolved_alias_and_reasoning_configuration():
    module = implementation()
    manifest = {"engine_kwargs": {"model": "/model", "enable_prefix_caching": True}}
    resolved = {"model_config": {"served_model_name": "local-alias"},
                "structured_outputs_config": {"reasoning_parser": "qwen3"}}
    options = module.replay_options(manifest, resolved, "session")
    assert options["served_model_name"] == "local-alias"
    assert options["structured_outputs_config"]["reasoning_parser"] == "qwen3"
    assert module.configuration_differences(resolved, resolved) == {}
    changed = dict(resolved, structured_outputs_config={"reasoning_parser": ""})
    assert "structured_outputs_config" in module.configuration_differences(resolved, changed)


@pytest.fixture
def unit_model(tmp_path):
    """CPU unit fixture only: dummy bytes, NOT a loadable model or GPU evidence."""
    model = tmp_path / "fake-model-unit-fixture"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json",
                 "model.safetensors.index.json"):
        (model / name).write_text('{"unit_fixture": "测试"}\n', encoding="utf-8")
    (model / "model-00001.safetensors").write_bytes(b"unit-fixture-not-real-weights")
    return model


def capture_unit_manifest(module, monkeypatch, run, model):
    """Exercise manifest creation without querying GPUs or executing vLLM."""
    captured = {}
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda *a, **kw: b"CPU unit fixture; no GPU queried")
    monkeypatch.setattr(module.os, "chdir", lambda *a: None)
    def no_server(executable, command, env):
        captured.update(env=env)
        raise SystemExit("unit fixture: server execution disabled")
    monkeypatch.setattr(module.os, "execve", no_server)
    args = SimpleNamespace(run_dir=run, model=model, gpu_util=0.3,
                           enforce_eager=True, gpu=0, port=18180, name="unit-fixture")
    with pytest.raises(SystemExit, match="unit fixture"):
        module.serve(args)
    return json.loads((run / "manifest.json").read_text(encoding="utf-8")), captured["env"]


def test_manifest_restores_controlled_runtime_environment_without_secrets(monkeypatch, tmp_path, unit_model):
    module = implementation()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "unit-secret-do-not-record")
    monkeypatch.setenv("VLLM_SECRET_TOKEN", "unit-other-secret-do-not-record")
    monkeypatch.setenv("VLLM_UNCONTROLLED_KERNEL", "unit-unknown-setting")
    monkeypatch.setenv("FORCED_REPLAY_MODE", "replay")
    run = tmp_path / "run"
    manifest, env = capture_unit_manifest(module, monkeypatch, run, unit_model)
    recorded = manifest.get("runtime_environment")
    assert recorded is not None, "manifest must capture the controlled vLLM runtime environment"
    assert recorded["VLLM_BATCH_INVARIANT"] == "1"
    assert recorded["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert recorded["VLLM_USE_DEEP_GEMM"] is None
    assert env["VLLM_BATCH_INVARIANT"] == "1"
    assert env["FORCED_REPLAY_MODE"] == "capture"
    assert all(name not in env for name in ("VLLM_API_KEY", "VLLM_SECRET_TOKEN", "VLLM_UNCONTROLLED_KERNEL"))
    assert "unit-secret" not in (run / "manifest.json").read_text(encoding="utf-8")
    assert "unit-other-secret" not in (run / "manifest.json").read_text(encoding="utf-8")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    restored = module.lab_environment(tmp_path / "replay", 3, recorded)
    assert restored["VLLM_BATCH_INVARIANT"] == "1"
    assert restored["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert "VLLM_USE_DEEP_GEMM" not in restored
    assert "VLLM_API_KEY" not in restored
    assert restored["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["VLLM_API_KEY"] == "unit-secret-do-not-record"


@pytest.fixture
def unit_run(monkeypatch, tmp_path, unit_model):
    """Manifest/trace are CPU unit fixtures, never acceptance evidence."""
    module = implementation()
    monkeypatch.setattr(module.os, "environ", dict(os.environ))
    run = tmp_path / "run"
    manifest, _ = capture_unit_manifest(module, monkeypatch, run, unit_model)
    events = [
        {"kind": "engine", "config": {"model_config": {"served_model_name": "unit-fixture"},
                                     "structured_outputs_config": {}}},
        {"kind": "request", "request_id": "unit-request", "prompt_ids": [1, 2],
         "sampling": {"n": 1, "extra_args": {"trace_id": "unit-测试"}}},
        {"kind": "schedule", "request_id": "unit-request", "start": 0, "end": 2,
         "cached_tokens": 0, "preemptions": 0, "batch_size": 1},
        {"kind": "output", "request_id": "unit-request", "ids": [3],
         "logprobs": [{"3": -0.1}], "finish_reason": "length"},
    ]
    write_unit_events(run / "online" / "engine.jsonl", events)
    return run, manifest, events


def write_unit_events(path, events):
    path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
                    encoding="utf-8")


def install_unit_llm(monkeypatch, events, on_load=None, measurement=None):
    """In-process replay stub: does not import or initialize GPU runtime."""
    import sys
    module = implementation()
    class UnitLLM:
        def __init__(self, **options):
            if on_load:
                on_load()
        def generate(self, prompt, params, use_tqdm):
            target = Path(os.environ["FORCED_REPLAY_TRACE_DIR"])
            write_unit_events(target / "engine.jsonl", events)
            row = dict(trace_id="unit-测试", step=0, forced_id=3, forced_logprob=-0.1,
                       forced_rank=1, entropy=0.5, topk_ids=[3], topk_logprobs=[-0.1])
            if measurement:
                row.update(measurement)
            write_unit_events(target / "measurements.jsonl", [row])
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[3])], num_cached_tokens=0)]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=UnitLLM, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=lambda **kwargs: kwargs))


def replay_unit_args(run, mode="session"):
    return SimpleNamespace(run_dir=run, gpu=0, mode=mode, tolerance=1e-5)


def test_replay_applies_recorded_environment_before_llm_load(monkeypatch, unit_run):
    module = implementation()
    run, manifest, events = unit_run
    module.os.environ.update(VLLM_BATCH_INVARIANT="1", VLLM_API_KEY="unit-new-secret",
                             VLLM_UNCONTROLLED_KERNEL="new-shell-knob")
    module.os.environ["FORCED_REPLAY_MODE"] = "capture"
    for name in manifest["runtime_environment"]:
        if manifest["runtime_environment"][name] is None:
            module.os.environ[name] = "1"
    def check_environment():
        assert os.environ["FORCED_REPLAY_MODE"] == "replay"
        assert {name: os.environ.get(name) for name in manifest["runtime_environment"]} == manifest["runtime_environment"]
        assert "VLLM_API_KEY" not in os.environ
        assert "VLLM_UNCONTROLLED_KERNEL" not in os.environ
        raise RuntimeError("unit fixture: stopped before GPU load")
    install_unit_llm(monkeypatch, events, on_load=check_environment)
    with pytest.raises(RuntimeError, match="stopped before GPU"):
        module.replay(replay_unit_args(run))


@pytest.mark.parametrize("mode,target_name", [("session", "replay"), ("rebuild-prefix", "rebuild_prefix")])
def test_existing_replay_target_never_overwrites_case(monkeypatch, unit_run, mode, target_name):
    module = implementation()
    run, _, _ = unit_run
    target = run / target_name
    target.mkdir()
    (target / "report.json").write_bytes(b"unit fixture existing report")
    original = b"unit fixture existing case must not be overwritten\n"
    (run / "case.json").write_bytes(original)
    with pytest.raises(FileExistsError):
        module.replay(replay_unit_args(run, mode))
    assert (run / "case.json").read_bytes() == original
    assert sorted(p.name for p in target.iterdir()) == ["report.json"]


def test_changed_existing_case_is_rejected_before_any_evidence_write(monkeypatch, unit_run):
    module = implementation()
    run, _, events = unit_run
    (run / "case.json").write_text('{"unit_fixture": "different evidence"}\n', encoding="utf-8")
    original = (run / "case.json").read_bytes()
    install_unit_llm(monkeypatch, events, on_load=lambda: pytest.fail("must reject before LLM load"))
    with pytest.raises(ValueError, match="case.json"):
        module.replay(replay_unit_args(run, "rebuild-prefix"))
    assert (run / "case.json").read_bytes() == original
    assert not (run / "rebuild_prefix").exists()


@pytest.mark.parametrize("mode", ["session", "rebuild-prefix"])
def test_matching_existing_case_is_reused_without_rewriting(monkeypatch, unit_run, mode):
    module = implementation()
    run, manifest, events = unit_run
    case = dict(manifest=manifest, requests=module.assemble(events))
    (run / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=7), encoding="utf-8")
    original = (run / "case.json").read_bytes()
    def stop_load():
        raise RuntimeError("unit fixture: preflight complete")
    install_unit_llm(monkeypatch, events, on_load=stop_load)
    with pytest.raises(RuntimeError, match="preflight complete"):
        module.replay(replay_unit_args(run, mode))
    assert (run / "case.json").read_bytes() == original


@pytest.mark.parametrize("api_rows,expected_count,expected_status", [
    (None, None, "missing"), ([], 0, "available"),
    ([{"kind": "api_request"}, {"kind": "api_response"}, {"kind": "api_request"}], 2, "available"),
])
def test_api_metadata_is_optional_and_reported(monkeypatch, unit_run, api_rows, expected_count, expected_status):
    module = implementation()
    run, _, events = unit_run
    if api_rows is not None:
        write_unit_events(run / "online" / "api.jsonl", api_rows)
    install_unit_llm(monkeypatch, events)
    module.replay(replay_unit_args(run))
    report = json.loads((run / "replay" / "report.json").read_text(encoding="utf-8"))
    assert report["online_api_requests"] == expected_count
    assert report["online_api_metadata_status"] == expected_status
    assert (run / "replay" / "requests.json").is_file()
    assert (run / "replay" / "measurements.npz").is_file()
    case_bytes = (run / "case.json").read_bytes()
    module.replay(replay_unit_args(run, "rebuild-prefix"))
    assert (run / "case.json").read_bytes() == case_bytes


def test_malformed_optional_api_metadata_fails_preflight_not_after_llm_load(monkeypatch, unit_run):
    module = implementation()
    run, _, events = unit_run
    (run / "online" / "api.jsonl").write_text("unit fixture: invalid JSON\n", encoding="utf-8")
    install_unit_llm(monkeypatch, events, on_load=lambda: pytest.fail("must preflight API metadata"))
    with pytest.raises(json.JSONDecodeError):
        module.replay(replay_unit_args(run))
    assert not (run / "case.json").exists()
    assert not (run / "replay").exists()


@pytest.mark.parametrize("change,expected_error", [
    ("config.json", "fingerprint"), ("tokenizer_config.json", "fingerprint"),
    ("tokenizer.json", "fingerprint"), ("model.safetensors.index.json", "fingerprint"),
    ("missing-tokenizer", "fingerprint"), ("extra-weight", "weight"),
    ("missing-weight", "weight"), ("resized-weight", "weight"), ("missing-model", "model"),
])
def test_model_drift_is_rejected_before_llm_load_or_evidence_writes(monkeypatch, unit_run, change, expected_error):
    module = implementation()
    run, manifest, events = unit_run
    model = Path(manifest["engine_kwargs"]["model"])
    if change.endswith(".json"):
        (model / change).write_text('{"unit_fixture": "changed"}', encoding="utf-8")
    elif change == "missing-tokenizer":
        (model / "tokenizer.json").unlink()
    elif change == "extra-weight":
        (model / "unexpected.safetensors").write_bytes(b"unit-fixture-extra-weight")
    elif change == "missing-weight":
        (model / "model-00001.safetensors").unlink()
    elif change == "resized-weight":
        (model / "model-00001.safetensors").write_bytes(b"unit-fixture-resized")
    else:
        model.rename(model.with_name("moved-unit-fixture"))
    install_unit_llm(monkeypatch, events, on_load=lambda: pytest.fail("model drift must fail before LLM load"))
    with pytest.raises(ValueError, match=expected_error):
        module.replay(replay_unit_args(run))
    assert not (run / "case.json").exists()
    assert not (run / "replay").exists()


def test_weight_verification_is_inventory_and_size_not_content_checksum(monkeypatch, unit_run):
    module = implementation()
    run, manifest, events = unit_run
    weight = Path(manifest["engine_kwargs"]["model"]) / "model-00001.safetensors"
    weight.write_bytes(b"X" * weight.stat().st_size)
    install_unit_llm(monkeypatch, events)
    module.replay(replay_unit_args(run))
    report = json.loads((run / "replay" / "report.json").read_text(encoding="utf-8"))
    assert report["model_identity_verified"] is True
    assert "no weight-content checksum" in report["model_identity_scope"]


@pytest.mark.parametrize("mode", ["session", "rebuild-prefix"])
def test_legacy_environment_metadata_is_unknown_not_certified(monkeypatch, unit_run, mode):
    module = implementation()
    run, manifest, events = unit_run
    manifest.pop("runtime_environment")
    manifest.pop("runtime_environment_policy")
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    module.os.environ["VLLM_BATCH_INVARIANT"] = "1"
    install_unit_llm(monkeypatch, events)
    if mode == "session":
        with pytest.raises(SystemExit, match="verification failed"):
            module.replay(replay_unit_args(run, mode))
    else:
        module.replay(replay_unit_args(run, mode))
    target = run / ("replay" if mode == "session" else "rebuild_prefix")
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert report["runtime_environment_status"] == "unknown"
    assert report["runtime_environment_match"] is None
    assert report["configuration_match"] is False
    assert report["numerical_match"] is True
    assert "VLLM_BATCH_INVARIANT" not in os.environ


def test_recorded_runtime_environment_is_reported_as_restored(monkeypatch, unit_run):
    module = implementation()
    run, _, events = unit_run
    install_unit_llm(monkeypatch, events)
    module.replay(replay_unit_args(run))
    report = json.loads((run / "replay" / "report.json").read_text(encoding="utf-8"))
    assert report["runtime_environment_status"] == "restored"
    assert report["runtime_environment_match"] is True
    assert report["runtime_environment_policy"] == module.RUNTIME_ENVIRONMENT_POLICY


@pytest.fixture
def ascii_default_text_io(monkeypatch):
    """Simulate a non-UTF-8 locale for lab JSON I/O; case owns JSONL I/O."""
    original = Path.open
    def ascii_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if path.suffix != ".jsonl" and "b" not in mode and encoding in (None, "locale"):
            encoding = "ascii"
        return original(path, mode, buffering, encoding, errors, newline)
    monkeypatch.setattr(Path, "open", ascii_open)


@pytest.mark.parametrize("exclusive", [False, True])
def test_dump_uses_utf8_independent_of_locale(tmp_path, ascii_default_text_io, exclusive):
    module = implementation()
    path = tmp_path / "unit-测试.json"
    module.dump(path, {"unit_fixture": "测试 ✓"}, exclusive=exclusive)
    assert json.loads(path.read_bytes().decode("utf-8")) == {"unit_fixture": "测试 ✓"}


def test_replay_reads_manifest_and_existing_case_as_utf8(monkeypatch, unit_run, ascii_default_text_io):
    module = implementation()
    run, manifest, events = unit_run
    manifest["model_name"] = "unit-fixture-测试"
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    case = dict(manifest=manifest, requests=module.assemble(events))
    (run / "case.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    original = (run / "case.json").read_bytes()
    install_unit_llm(monkeypatch, events)
    module.replay(replay_unit_args(run))
    assert (run / "case.json").read_bytes() == original
    assert "unit-测试" in (run / "replay" / "requests.json").read_bytes().decode("utf-8")


def test_nonfinite_measurement_markers_are_numeric_in_npz(monkeypatch, unit_run):
    import numpy as np
    module = implementation()
    run, _, events = unit_run
    install_unit_llm(monkeypatch, events, measurement=dict(
        forced_logprob="nan", entropy="nan", forced_rank=-1,
        topk_ids=[3, 4, 5], topk_logprobs=["-inf", "inf", "nan"]))
    module.replay(replay_unit_args(run, "rebuild-prefix"))
    with np.load(run / "rebuild_prefix" / "measurements.npz", allow_pickle=False) as data:
        for name in ("forced_logprob", "entropy", "topk_logprobs"):
            assert data[name].dtype.kind == "f", f"{name} must contain numeric IEEE values, not marker strings"
        assert np.isnan(data["forced_logprob"][0])
        assert np.isnan(data["entropy"][0])
        assert np.isneginf(data["topk_logprobs"][0, 0])
        assert np.isposinf(data["topk_logprobs"][0, 1])
        assert np.isnan(data["topk_logprobs"][0, 2])
        assert data["forced_rank"].tolist() == [-1]
    report = json.loads((run / "rebuild_prefix" / "report.json").read_text(encoding="utf-8"))
    assert report["numerical_match"] is False


@pytest.mark.parametrize("evidence_name", ["manifest.json", "case.json"])
def test_serve_preflights_existing_evidence_before_any_write(monkeypatch, tmp_path, unit_model, evidence_name):
    module = implementation()
    run = tmp_path / "run"
    run.mkdir()
    original = b"unit fixture: original evidence\n"
    (run / evidence_name).write_bytes(original)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: pytest.fail("must preflight before GPU query"))
    args = SimpleNamespace(run_dir=run, model=unit_model, gpu_util=0.3,
                           enforce_eager=True, gpu=0, port=18180, name="unit-fixture")
    with pytest.raises(FileExistsError):
        module.serve(args)
    assert (run / evidence_name).read_bytes() == original
    assert not (run / "online").exists()


def test_forced_sampling_preserves_penalties_but_disables_early_stops():
    module = implementation()
    params = module.forced_sampling({"n": 1, "temperature": 0.2, "repetition_penalty": 1.1,
        "stop_token_ids": [4], "stop": ["END"], "extra_args": {"trace_id": "t"},
        "output_kind": 1, "_all_stop_token_ids": [4]}, [3, 4])
    assert params["max_tokens"] == 2
    assert params["ignore_eos"] is True and params["stop_token_ids"] == []
    assert params["repetition_penalty"] == 1.1
    assert params["extra_args"]["forced_ids"] == [3, 4]
    assert "output_kind" not in params and "_all_stop_token_ids" not in params


def test_unsupported_sampling_is_rejected_instead_of_silently_changed():
    module = implementation()
    with pytest.raises(ValueError, match="structured"):
        module.forced_sampling({"structured_outputs": {"json_object": True}}, [1])


def test_measured_steps_must_be_complete_unique_and_match_forced_ids():
    module = implementation()
    request = {"sampling": {"extra_args": {"trace_id": "t"}}, "output_ids": [3, 4]}
    good = [{"trace_id": "t", "step": 0, "forced_id": 3, "forced_logprob": -0.1},
            {"trace_id": "t", "step": 1, "forced_id": 4, "forced_logprob": -0.2}]
    assert module.measured_probabilities(request, good) == [-0.1, -0.2]
    with pytest.raises(ValueError, match="measurement"):
        module.measured_probabilities(request, good + [good[0]])
