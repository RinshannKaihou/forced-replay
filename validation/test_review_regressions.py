"""PR #1 regressions. Synthetic unit inputs are not collection evidence."""
import json
import math
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate

from validation import capture, case


def test_nonfinite_measurements_are_strict_json_and_roundtrip_numeric_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    capture.append("numeric.jsonl", {"negative": float("-inf"), "positive": float("inf"),
                                     "invalid": float("nan"), "text": "nan", "finite": -0.5})
    raw = (tmp_path / "numeric.jsonl").read_text(encoding="utf-8")
    value = json.loads(raw, parse_constant=lambda token: pytest.fail(f"nonstandard JSON: {token}"))
    assert value["negative"] == "-inf" and value["positive"] == "inf"
    assert math.isnan(float(value["invalid"])) and value["text"] == "nan"
    assert value["finite"] == -0.5


def test_nonfinite_reference_is_preserved_but_cannot_pass_numerical_verification():
    events = [
        {"kind": "request", "request_id": "r", "prompt_ids": [1], "sampling": {}},
        {"kind": "schedule", "request_id": "r", "start": 0, "end": 1,
         "cached_tokens": 0, "preemptions": 0, "batch_size": 1},
        {"kind": "output", "request_id": "r", "ids": [2],
         "logprobs": [{"2": "nan", "3": "-inf"}], "finish_reason": "stop"},
    ]
    requests = case.assemble(events)
    assert requests[0]["online_logprobs"] == ["nan"]
    result = case.compare(requests, requests)
    assert result["numerical_match"] is False
    assert result["nonfinite_logprob_count"] == 1
    assert result["max_abs_logprob_error"] is None
    json.dumps(result, allow_nan=False)


def test_sparse_or_nan_logits_do_not_crash_measurement_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    proc = capture.ReplayProcessor(None, torch.device("cpu"), False)
    params = SamplingParams(extra_args={"trace_id": "unit", "forced_ids": [1, 1]})
    outputs = []
    proc.update_state(BatchUpdate(1, [], [(0, params, [7], outputs)], []))
    logits = torch.full((1, 24), float("-inf")); logits[0, 1] = 0
    assert proc.apply(logits).argmax(-1).tolist() == [1]
    outputs.append(1)
    assert proc.apply(torch.full((1, 24), float("nan"))).argmax(-1).tolist() == [1]
    rows = case.read_events(tmp_path / "measurements.jsonl")
    assert "-inf" in rows[0]["topk_logprobs"]
    assert rows[1]["forced_logprob"] == "nan" and rows[1]["forced_rank"] == -1


@pytest.mark.parametrize("path,payload", [
    ("/v1/completions", {"prompt": "hello"}),
    ("/v1/responses", {"input": "hello"}),
    ("/v1/embeddings", {"input": "hello"}),
    ("/v1/chat/completions", {"vllm_xargs": {"forced_ids": [1]}}),
    ("/v1/chat/completions", {"vllm_xargs": {"trace_id": "../../outside"}}),
    ("/v1/chat/completions", {"vllm_xargs": [1]}),
    ("/v1/chat/completions", {"n": 2}),
    ("/v1/chat/completions", {"response_format": {"type": "json_object"}}),
    ("/v1/chat/completions", {"tool_choice": "required"}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
])
def test_unsupported_online_input_is_rejected_before_application(path, payload, tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    app, calls = FastAPI(), []
    async def accepted(request: Request):
        calls.append(await request.json())
        return {"unexpected": "accepted"}
    app.add_api_route(path, accepted, methods=["POST"])
    app.add_middleware(capture.CaptureMiddleware)
    with TestClient(app) as client:
        result = client.post(path, json=payload)
    assert result.status_code == 400
    assert not calls and not (tmp_path / "engine.jsonl").exists()
    assert result.json()["error"]["type"] == "capture_scope_error"


def test_malformed_json_is_client_error_not_server_exception(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    app = FastAPI(); app.add_middleware(capture.CaptureMiddleware)
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", content=b"{bad json").status_code == 400
        assert client.post("/v1/chat/completions", json=[]).status_code == 400
        assert client.post("/v1/chat/completions", content=b'{"temperature":NaN}').status_code == 400
        assert client.post("/v1/chat/completions", json={"messages": 42}).status_code == 400
        assert client.get("/health").status_code == 404  # passes through unchanged


def test_online_worker_defensively_ignores_forcing_and_frontend_rejects_it(monkeypatch):
    monkeypatch.setenv("FORCED_REPLAY_MODE", "capture")
    proc = capture.ReplayProcessor(SimpleNamespace(), torch.device("cpu"), False)
    params = SamplingParams(extra_args={"forced_ids": [1]})
    proc.update_state(BatchUpdate(1, [], [(0, params, [7], [])], []))
    assert not proc.rows
    logits = torch.tensor([[2., 1., -1.]])
    assert torch.equal(proc.apply(logits.clone()), logits)
    with pytest.raises(ValueError, match="offline replay"):
        capture.ReplayProcessor.validate_params(params)


def test_offline_forcing_params_are_validated_before_scheduler(monkeypatch):
    monkeypatch.setenv("FORCED_REPLAY_MODE", "replay")
    for args in ({"forced_ids": [1]}, {"forced_ids": [1], "trace_id": "../../outside"},
                 {"forced_ids": [True], "trace_id": "r"}, {"forced_ids": [], "trace_id": "r"}):
        with pytest.raises(ValueError):
            capture.ReplayProcessor.validate_params(SamplingParams(extra_args=args))
    capture.ReplayProcessor.validate_params(SamplingParams(extra_args={"forced_ids": [0], "trace_id": "valid-1"}))


def mock_scheduler(monkeypatch):
    scheduler = object.__new__(capture.TraceScheduler)
    scheduler.requests = {}
    scheduler._trace_cache = {}
    scheduler._trace_hash_counts = {}
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)
    return scheduler


def test_finished_abort_is_logged_and_cached_request_is_freed(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    scheduler = mock_scheduler(monkeypatch)
    request = SimpleNamespace(request_id="r", sampling_params=SamplingParams(), stop_reason=None,
                              num_output_tokens=0, get_finished_reason=lambda: "abort")
    scheduler._trace_cache["r"] = 0; scheduler._trace_hash_counts["r"] = 4
    monkeypatch.setattr(capture.Scheduler, "finish_requests", lambda self, *a, **k: [request])
    from vllm.v1.request import RequestStatus
    result = scheduler.finish_requests(["r"], RequestStatus.FINISHED_ABORTED)
    assert result == [request]
    assert not scheduler._trace_cache and not scheduler._trace_hash_counts
    event = case.read_events(tmp_path / "engine.jsonl")[0]
    assert event["kind"] == "finish" and event["finish_reason"] == "abort"
    assert event["termination_source"] == "scheduler.finish_requests"


def test_aborted_case_has_explicit_reason_not_ambiguous_unfinished_error():
    events = [
        {"kind": "request", "request_id": "r", "prompt_ids": [1], "sampling": {}},
        {"kind": "finish", "request_id": "r", "finish_reason": "abort",
         "termination_source": "scheduler.finish_requests"},
    ]
    with pytest.raises(ValueError, match="unsupported termination.*abort"):
        case.assemble(events)


def test_schedule_stores_only_new_hashes_and_clears_state_after_output(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    scheduler = mock_scheduler(monkeypatch)
    req = SimpleNamespace(sampling_params=SamplingParams(), num_computed_tokens=1,
                          num_prompt_tokens=1, num_preemptions=0, block_hashes=[b"a"])
    scheduler.requests["r"] = req
    monkeypatch.setattr(capture.Scheduler, "schedule", lambda self: SimpleNamespace(num_scheduled_tokens={"r": 1}))
    scheduler.schedule(); req.num_computed_tokens = 2; scheduler.schedule()
    req.block_hashes.extend([b"b", b"c"]); req.num_computed_tokens = 3; scheduler.schedule()
    rows = case.read_events(tmp_path / "engine.jsonl")
    assert [row["new_block_hashes"] for row in rows] == [[b"a".hex()], [], [b"b".hex(), b"c".hex()]]
    assert [row["block_hash_start"] for row in rows] == [0, 1, 1]
    assert all("block_hashes" not in row for row in rows)
    out = SimpleNamespace(request_id="r", new_token_ids=[2], new_logprobs=None,
                          finish_reason="stop", stop_reason=None, num_nans_in_logits=0)
    monkeypatch.setattr(capture.Scheduler, "update_from_output", lambda self: {0: SimpleNamespace(outputs=[out])})
    scheduler.update_from_output()
    assert not scheduler._trace_cache and not scheduler._trace_hash_counts


def test_config_sets_are_canonical_without_reordering_sequences():
    values = {f"v_{i:02d}" for i in range(32)}
    assert capture.json_config({"ops": values})["ops"] == sorted(values)
    assert capture.json_config(("b", "a")) == ["b", "a"]


def test_online_engine_output_sink_keeps_nonfinite_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    scheduler = mock_scheduler(monkeypatch)
    probs = SimpleNamespace(logprob_token_ids=torch.tensor([[2, 3]]),
                            logprobs=torch.tensor([[float("nan"), float("-inf")]]))
    out = SimpleNamespace(request_id="r", new_token_ids=[2], new_logprobs=probs,
                          finish_reason="length", stop_reason=None, num_nans_in_logits=1)
    monkeypatch.setattr(capture.Scheduler, "update_from_output", lambda self: {0: SimpleNamespace(outputs=[out])})
    scheduler.update_from_output()
    row = case.read_events(tmp_path / "engine.jsonl")[0]
    assert row["logprobs"] == [{"2": "nan", "3": "-inf"}] and row["num_nans"] == 1


def test_trace_reader_uses_utf8_even_with_ascii_process_locale(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    source = tmp_path / "unicode.jsonl"
    source.write_text(json.dumps({"label": "中文"}, ensure_ascii=False) + "\n", encoding="utf-8")
    env = dict(os.environ, LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    command = [sys.executable, "-X", "utf8=0", "-c",
        "import sys; from validation.case import read_events; "
        "assert read_events(sys.argv[1])[0]['label'] == '\\u4e2d\\u6587'", str(source)]
    subprocess.run(command, env=env, cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)


@pytest.mark.parametrize("value", ["true", "false", "True", "False"])
def test_runtime_environment_rejects_bool_aliases_not_supported_by_all_pinned_parsers(tmp_path, value):
    from validation import lab
    runtime = {name: None for name in lab.RUNTIME_ENVIRONMENT_KEYS}
    runtime["VLLM_BATCH_INVARIANT"] = value
    with pytest.raises(ValueError, match="runtime environment"):
        lab.lab_environment(tmp_path, 0, runtime)


def test_decoder_excessive_nesting_is_logged_400_before_application(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    app, calls = FastAPI(), []
    async def accepted():
        calls.append(True)
        return {"unexpected": "accepted"}
    app.add_api_route("/v1/chat/completions", accepted, methods=["POST"])
    app.add_middleware(capture.CaptureMiddleware)
    body = "[" * 1200 + "0" + "]" * 1200
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=body)
    assert response.status_code == 400
    assert not calls and not (tmp_path / "engine.jsonl").exists()
    rows = case.read_events(tmp_path / "api.jsonl")
    assert [row["kind"] for row in rows] == ["api_rejection"]
    assert "nesting" in rows[0]["reason"]
    assert response.json()["error"]["message"] == rows[0]["reason"]


@pytest.mark.parametrize("array_count", [63, 600])
def test_decoded_diagnostic_nesting_is_rejected_before_normalization_or_app(array_count, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    app, calls = FastAPI(), []
    async def accepted():
        calls.append(True)
        return {"unexpected": "accepted"}
    app.add_api_route("/v1/chat/completions", accepted, methods=["POST"])
    app.add_middleware(capture.CaptureMiddleware)
    body = '{"messages":[],"vllm_xargs":{"diagnostic":' + "[" * array_count + "0" + "]" * array_count + "}}"
    # Both the decoder and encoder can handle the 600-array reviewer repro.
    json.dumps(json.loads(body), allow_nan=False)
    original_dumps = capture.json.dumps
    def no_request_encoding(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("return_token_ids") is True:
            pytest.fail("excessive nesting must be rejected before request encoding")
        return original_dumps(value, *args, **kwargs)
    monkeypatch.setattr(capture.json, "dumps", no_request_encoding)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=body)
    assert response.status_code == 400
    assert not calls and not (tmp_path / "engine.jsonl").exists()
    rows = case.read_events(tmp_path / "api.jsonl")
    assert [row["kind"] for row in rows] == ["api_rejection"]
    assert rows[0]["reason"] == f"JSON nesting exceeds maximum depth of {capture.MAX_JSON_DEPTH}"
    assert response.json()["error"]["message"] == rows[0]["reason"]


@pytest.mark.parametrize("stage", ["encode", "normalization"])
def test_remaining_request_recursion_is_logged_400_not_forwarded(stage, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    app, calls = FastAPI(), []
    async def accepted():
        calls.append(True)
        return {"unexpected": "accepted"}
    app.add_api_route("/v1/chat/completions", accepted, methods=["POST"])
    app.add_middleware(capture.CaptureMiddleware)
    if stage == "encode":
        original_dumps = capture.json.dumps
        def exhausted_encoder(value, *args, **kwargs):
            if isinstance(value, dict) and value.get("return_token_ids") is True:
                raise RecursionError("CPU fixture: encoder recursion exhausted")
            return original_dumps(value, *args, **kwargs)
        monkeypatch.setattr(capture.json, "dumps", exhausted_encoder)
    else:
        original_safe = capture.json_safe
        def exhausted_normalizer(value):
            if isinstance(value, dict) and value.get("kind") == "api_request":
                raise RecursionError("CPU fixture: normalizer recursion exhausted")
            return original_safe(value)
        monkeypatch.setattr(capture, "json_safe", exhausted_normalizer)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=b'{"messages":[]}')
    assert response.status_code == 400
    assert not calls and not (tmp_path / "engine.jsonl").exists()
    rows = case.read_events(tmp_path / "api.jsonl")
    assert [row["kind"] for row in rows] == ["api_rejection"]
    assert "nesting" in rows[0]["reason"]
    assert response.json()["error"]["message"] == rows[0]["reason"]


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf"), -1e-5])
def test_compare_rejects_nonfinite_or_negative_tolerance(tolerance):
    requests = [{"request_id": "cpu-unit", "prompt_ids": [1], "output_ids": [2],
                 "cached_tokens": 0, "schedule": [], "online_logprobs": [-0.5]}]
    with pytest.raises(ValueError, match="tolerance must be a finite nonnegative number"):
        case.compare(requests, requests, tolerance=tolerance)


@pytest.mark.parametrize("mode", ["session", "rebuild-prefix"])
@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf"), -1e-5])
def test_replay_tolerance_preflight_precedes_manifest_read_and_evidence_writes(mode, tolerance, tmp_path):
    from validation import lab
    run = tmp_path / "cpu-unit-run"
    run.mkdir()
    original = b"CPU unit fixture: invalid manifest must not be read before tolerance preflight\n"
    (run / "manifest.json").write_bytes(original)
    environment = dict(lab.os.environ)
    args = SimpleNamespace(run_dir=run, gpu=0, mode=mode, tolerance=tolerance)
    # Calls the actual replay preflight; no model stub or GPU initialization.
    with pytest.raises(ValueError, match="tolerance must be a finite nonnegative number"):
        lab.replay(args)
    assert (run / "manifest.json").read_bytes() == original
    assert sorted(path.name for path in run.iterdir()) == ["manifest.json"]
    assert dict(lab.os.environ) == environment


@pytest.mark.parametrize("tolerance", ["nan", "inf", "-inf", "-0.001"])
def test_replay_cli_rejects_invalid_tolerance_without_traceback_or_files(tolerance, tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    run = tmp_path / "must-not-be-created"
    result = subprocess.run([sys.executable, "-m", "validation.lab", "replay",
                             "--run-dir", str(run), f"--tolerance={tolerance}"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, check=False, timeout=30)
    assert result.returncode == 2
    assert "tolerance must be a finite nonnegative number" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout and not run.exists()


@pytest.mark.parametrize("array_count", [0, 30, 62])
def test_normal_requests_up_to_depth_limit_are_recorded_and_forwarded(array_count, tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    app, calls = FastAPI(), []
    async def accepted(request: Request):
        calls.append(await request.json())
        return {"ok": True}
    app.add_api_route("/v1/chat/completions", accepted, methods=["POST"])
    app.add_middleware(capture.CaptureMiddleware)
    body = '{"messages":[{"role":"user","content":"hello"}],"vllm_xargs":{"diagnostic":' + "[" * array_count + "0" + "]" * array_count + "}}"
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=body)
    assert response.status_code == 200 and len(calls) == 1
    rows = case.read_events(tmp_path / "api.jsonl")
    assert rows[0]["kind"] == "api_request"
    assert rows[0]["original"] == json.loads(body) and rows[0]["effective"] == calls[0]
    assert not any(row["kind"] == "api_rejection" for row in rows)


@pytest.mark.parametrize("body", [b'{"messages":[]}', b"[" * 1200 + b"0" + b"]" * 1200])
def test_capture_does_not_swallow_logging_io_errors(body, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"CPU unit fixture: cannot create a trace directory here")
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(blocked))
    app, calls = FastAPI(), []
    async def accepted():
        calls.append(True)
        return {"unexpected": "accepted"}
    app.add_api_route("/v1/chat/completions", accepted, methods=["POST"])
    app.add_middleware(capture.CaptureMiddleware)
    with TestClient(app) as client, pytest.raises(OSError):
        client.post("/v1/chat/completions", content=body)
    assert not calls


@pytest.mark.parametrize("tolerance", [0, 1e-5, 1])
def test_compare_accepts_finite_nonnegative_tolerance(tolerance):
    requests = [{"request_id": "cpu-unit", "prompt_ids": [1], "output_ids": [2],
                 "cached_tokens": 0, "schedule": [], "online_logprobs": [-0.5]}]
    report = case.compare(requests, requests, tolerance=tolerance)
    assert report["numerical_match"] is True and report["tolerance"] == tolerance
    json.dumps(report, allow_nan=False)
