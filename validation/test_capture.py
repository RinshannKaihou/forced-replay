"""CPU tests; use real torch tensors and vLLM batch-update semantics."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality


def implementation():
    assert Path(__file__).with_name("capture.py").exists(), "engine capture implementation is missing"
    return importlib.import_module("validation.capture")


def test_forcing_records_premask_distribution_and_preserves_terminal_ids(tmp_path, monkeypatch):
    module = implementation()
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    proc = module.ReplayProcessor(None, torch.device("cpu"), False)
    outputs = []
    params = SamplingParams(extra_args={"trace_id": "test", "forced_ids": [2, 0]})
    proc.update_state(BatchUpdate(1, [], [(0, params, [7], outputs)], []))
    logits = torch.tensor([[2., 1., -1.]])
    expected = float(torch.log_softmax(logits, dim=-1)[0, 2])
    result = proc.apply(logits.clone())
    assert result.argmax(-1).tolist() == [2]
    outputs.append(2)
    assert proc.apply(logits.clone()).argmax(-1).tolist() == [0]
    import json
    rows = [json.loads(line) for line in (tmp_path / "measurements.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1]
    assert rows[0]["forced_logprob"] == pytest.approx(expected)
    assert rows[0]["forced_rank"] == 2


def test_one_way_row_move_does_not_exchange_stale_destination():
    module = implementation()
    proc = module.ReplayProcessor(None, torch.device("cpu"), False)
    a = SamplingParams(extra_args={"forced_ids": [1], "trace_id": "a"})
    b = SamplingParams(extra_args={"forced_ids": [2], "trace_id": "b"})
    proc.update_state(BatchUpdate(2, [], [(0, a, [7], []), (1, b, [8], [])], []))
    proc.update_state(BatchUpdate(1, [], [], [(0, 1, MoveDirectionality.UNIDIRECTIONAL)]))
    assert 0 not in proc.rows
    assert proc.rows[1][0] == [1]


def test_effective_config_handles_tuple_keys_without_deepcopying_runtime_objects():
    module = implementation()
    from dataclasses import dataclass
    class RuntimeObject:
        def __deepcopy__(self, memo):
            raise AssertionError("must not deepcopy model/cache objects")
        def __str__(self):
            return "runtime-object"
    @dataclass
    class Config:
        shapes: dict
    assert module.json_config(Config({(1, 2): RuntimeObject()})) == {"shapes": {"(1, 2)": "runtime-object"}}


def test_api_capture_preserves_original_and_passes_effective_body(tmp_path, monkeypatch):
    module = implementation()
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    import json
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    app = FastAPI()
    # No postponed annotations: FastAPI resolves this locally imported type.
    async def echo(request: Request):
        return await request.json()
    app.add_api_route("/v1/chat/completions", echo, methods=["POST"])
    app.add_middleware(module.CaptureMiddleware)
    original = {"messages": [{"role": "user", "content": "hello"}], "stream": False}
    with TestClient(app) as client:
        result = client.post("/v1/chat/completions", json=original)
    assert result.status_code == 200
    effective = result.json()
    assert effective["messages"] == original["messages"]
    assert effective["logprobs"] is True and effective["return_token_ids"] is True
    rows = [json.loads(line) for line in (tmp_path / "api.jsonl").read_text().splitlines()]
    assert rows[0]["original"] == original
    assert rows[0]["effective"] == effective
    assert any(row.get("status") == 200 for row in rows)


def test_partial_prefill_dummy_sampling_is_not_measured_or_forced(tmp_path, monkeypatch):
    module = implementation()
    import json
    monkeypatch.setenv("FORCED_REPLAY_TRACE_DIR", str(tmp_path))
    active = tmp_path / "active"
    active.mkdir()
    state = active / "chunked.json"
    state.write_text(json.dumps({"end": 2, "prompt_length": 4}))
    proc = module.ReplayProcessor(SimpleNamespace(), torch.device("cpu"), False)
    params = SamplingParams(extra_args={"trace_id": "chunked", "forced_ids": [2]})
    proc.update_state(BatchUpdate(1, [], [(0, params, [7, 8, 9, 10], [])], []))
    logits = torch.tensor([[2., 1., -1.]])
    assert torch.equal(proc.apply(logits.clone()), logits)
    assert not (tmp_path / "measurements.jsonl").exists()
    state.write_text(json.dumps({"end": 4, "prompt_length": 4}))
    assert proc.apply(logits.clone()).argmax(-1).tolist() == [2]
    assert len((tmp_path / "measurements.jsonl").read_text().splitlines()) == 1


def test_online_observer_does_not_modify_logits():
    module = implementation()
    proc = module.ReplayProcessor(None, torch.device("cpu"), False)
    proc.update_state(BatchUpdate(1, [], [(0, SamplingParams(), [7], [])], []))
    logits = torch.tensor([[1., -2., 3.]])
    assert torch.equal(proc.apply(logits.clone()), logits)
