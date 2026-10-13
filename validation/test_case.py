"""CPU checks for the trace contract; fixtures are explicitly synthetic."""
import importlib.util
from pathlib import Path
import sys

import pytest


def implementation():
    path = Path(__file__).with_name("case.py")
    assert path.exists(), "session trace implementation is missing"
    spec = importlib.util.spec_from_file_location("replay_case", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def events():
    return [
        {"kind": "request", "request_id": "r1", "prompt_ids": [1, 2], "sampling": {}},
        {"kind": "schedule", "request_id": "r1", "start": 0, "end": 2, "cached_tokens": 0, "preemptions": 0, "batch_size": 1},
        {"kind": "output", "request_id": "r1", "ids": [3], "logprobs": [{"3": -0.2}], "finish_reason": None},
        {"kind": "schedule", "request_id": "r1", "start": 2, "end": 3, "cached_tokens": 0, "preemptions": 0, "batch_size": 1},
        {"kind": "output", "request_id": "r1", "ids": [4], "logprobs": [{"4": -0.3}], "finish_reason": "stop"},
        {"kind": "request", "request_id": "r2", "prompt_ids": [1, 2, 3, 4, 9], "sampling": {}},
        {"kind": "schedule", "request_id": "r2", "start": 2, "end": 5, "cached_tokens": 2, "preemptions": 0, "batch_size": 1},
        {"kind": "output", "request_id": "r2", "ids": [8], "logprobs": [{"8": -0.4}], "finish_reason": "length"},
    ]


def test_collects_engine_tokens_including_terminal_token_and_exact_new_prompt():
    case = implementation().assemble(events())
    assert len(case) == 2
    assert case[0]["output_ids"] == [3, 4]
    assert case[0]["online_logprobs"] == [-0.2, -0.3]
    assert case[1]["prompt_ids"] == [1, 2, 3, 4, 9]
    assert case[1]["cached_tokens"] == 2


def test_cache_path_difference_is_not_hidden_by_identical_tokens():
    module = implementation()
    original = module.assemble(events())
    replay = module.assemble(events())
    replay[1]["schedule"][0]["start"] = 0
    replay[1]["cached_tokens"] = 0
    report = module.compare(original, replay)
    assert report["token_match"] is True
    assert report["cache_path_match"] is False
    assert report["requests"][1]["cache_path_match"] is False


def test_incomplete_or_missing_probability_trace_is_rejected():
    module = implementation()
    with pytest.raises(ValueError, match="unfinished"):
        module.assemble(events()[:-1])
    bad = events()
    bad[2] = dict(bad[2], logprobs=[{}])
    with pytest.raises(ValueError, match="sampled token"):
        module.assemble(bad)


def test_request_cache_salt_is_preserved_for_support_checks():
    module = implementation()
    source = events()
    source[0] = dict(source[0], cache_salt="tenant-a")
    assert module.assemble(source)[0]["cache_salt"] == "tenant-a"


def test_probability_alignment_and_tolerance_are_checked():
    module = implementation()
    original = module.assemble(events())
    replay = module.assemble(events())
    replay[0]["online_logprobs"][1] = -0.32
    result = module.compare(original, replay, tolerance=0.001)
    assert result["numerical_match"] is False
    assert result["max_abs_logprob_error"] == pytest.approx(0.02)
    assert result["total_output_tokens"] == 3
