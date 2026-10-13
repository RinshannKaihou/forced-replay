import importlib
from pathlib import Path

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
