"""Version-pinned vLLM engine trace hooks and an identity/forcing processor.

No model forward hooks: sampling and scheduling observers work with CUDA graphs.
Trace writes intentionally synchronize measurements: a validation lab, not a
production performance monitor. Online mode never changes the model logits.
"""
import dataclasses
import fcntl
import json
import os
import re
from pathlib import Path
import time
import uuid

import msgspec
import torch
import vllm
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.interface import MoveDirectionality


def append(filename, event):
    root = os.environ.get("FORCED_REPLAY_TRACE_DIR")
    if not root:
        raise RuntimeError("FORCED_REPLAY_TRACE_DIR must be configured")
    path = Path(root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event, timestamp_ns=time.time_ns(), pid=os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def active_path(trace_id):
    if not isinstance(trace_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace_id):
        raise ValueError("invalid trace ID for scheduler sidecar")
    return Path(os.environ["FORCED_REPLAY_TRACE_DIR"]) / "active" / (trace_id + ".json")


def json_config(config):
    # Audit runtime configs without copying GPU modules or caches.
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return {field.name: json_config(getattr(config, field.name)) for field in dataclasses.fields(config)}
    if isinstance(config, dict):
        return {str(key): json_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple, set)):
        return [json_config(value) for value in config]
    if config is None or isinstance(config, (str, int, float, bool)):
        return config
    if hasattr(config, "to_dict"):
        return json_config(config.to_dict())
    return str(config)


class TraceScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        if vllm.__version__ != "0.26.0":
            raise RuntimeError(f"validation adapter pins vLLM 0.26.0, got {vllm.__version__}")
        super().__init__(*args, **kwargs)
        self._trace_cache = {}
        config = args[0] if args else kwargs["vllm_config"]
        append("engine.jsonl", {"kind": "engine", "vllm_version": vllm.__version__,
                                "torch_version": torch.__version__, "config": json_config(config)})

    def add_request(self, request):
        if request.prompt_token_ids is None or request.sampling_params is None:
            raise ValueError("validation supports text generation only")
        if request.resumable or request.mm_features or request.lora_request:
            raise ValueError("validation excludes streaming-input, multimodal, and LoRA requests")
        append("engine.jsonl", dict(kind="request", request_id=request.request_id,
                prompt_ids=list(request.prompt_token_ids),
                sampling=msgspec.to_builtins(request.sampling_params), cache_salt=request.cache_salt))
        return super().add_request(request)

    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        batch_size = len(output.num_scheduled_tokens)
        for rid, count in output.num_scheduled_tokens.items():
            req = self.requests[rid]
            end = req.num_computed_tokens
            start = end - count  # super.schedule already advances computed-token count.
            self._trace_cache.setdefault(rid, start)
            extra = req.sampling_params.extra_args or {}
            if extra.get("forced_ids") is not None:
                if batch_size != 1 or self.scheduler_config.async_scheduling:
                    raise ValueError("measurement sidecar requires synchronous serial scheduling")
                path = active_path(extra["trace_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(request_id=rid, start=start, end=end,
                                                 prompt_length=req.num_prompt_tokens)))
            append("engine.jsonl", dict(kind="schedule", request_id=rid, start=start, end=end,
                prompt_length=req.num_prompt_tokens, cached_tokens=self._trace_cache[rid],
                preemptions=req.num_preemptions, batch_size=batch_size,
                block_hashes=[bytes(h).hex() for h in req.block_hashes]))
        return output

    def update_from_output(self, *args, **kwargs):
        result = super().update_from_output(*args, **kwargs)
        for client in result.values():
            for output in client.outputs:
                rows = []
                if output.new_logprobs is not None:
                    probs = output.new_logprobs
                    for ids, values in zip(probs.logprob_token_ids.tolist(), probs.logprobs.tolist()):
                        rows.append({str(int(t)): float(p) for t, p in zip(ids, values)})
                append("engine.jsonl", dict(kind="output", request_id=output.request_id,
                    ids=list(output.new_token_ids), logprobs=rows,
                    finish_reason=str(output.finish_reason) if output.finish_reason is not None else None,
                    stop_reason=output.stop_reason, num_nans=output.num_nans_in_logits))
                if output.finish_reason is not None:
                    self._trace_cache.pop(output.request_id, None)
        return result


class ReplayProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.rows = {}
        self.require_schedule = vllm_config is not None

    def is_argmax_invariant(self):
        return False  # Identity online, deliberately non-invariant when forcing offline.

    def update_state(self, update):
        if update is None:
            return
        for idx in update.removed:
            self.rows.pop(idx, None)
        for idx, params, _prompt, output_ref in update.added:
            extra = params.extra_args or {}
            forced = extra.get("forced_ids")
            if forced is None:
                self.rows.pop(idx, None)
            else:
                self.rows[idx] = (forced, output_ref, extra["trace_id"])
        for a, b, direction in update.moved:
            left = self.rows.pop(a, None)
            if direction == MoveDirectionality.SWAP:
                right = self.rows.pop(b, None)
                if right is not None:
                    self.rows[a] = right
            else:
                self.rows.pop(b, None)
            if left is not None:
                self.rows[b] = left

    def apply(self, logits):
        for idx, (forced, output_ref, trace_id) in self.rows.items():
            if self.require_schedule:
                scheduled = json.loads(active_path(trace_id).read_text())
                if scheduled["end"] < scheduled["prompt_length"]:
                    # vLLM samples dummy logits for partial prefill and discards them.
                    # They are not output-token probabilities and must not be forced.
                    append("discarded_prefill.jsonl", dict(kind="discarded_prefill_sample", trace_id=trace_id,
                                                           start=scheduled["start"] if "start" in scheduled else None,
                                                           end=scheduled["end"]))
                    continue
            step = len(output_ref)
            if step >= len(forced):
                raise ValueError("generation exceeded the recorded trajectory")
            token = forced[step]
            if type(token) is not int or not 0 <= token < logits.shape[-1]:
                raise ValueError("forced token outside vocabulary")
            logp = torch.log_softmax(logits[idx].float(), -1)
            k = min(20, logp.numel())
            top = torch.topk(logp, k)
            entropy = -(logp.exp() * torch.where(torch.isfinite(logp), logp, 0)).sum()
            append("measurements.jsonl", dict(kind="measurement", trace_id=trace_id, step=step,
                forced_id=token, forced_logprob=float(logp[token]),
                forced_rank=int((logp > logp[token]).sum()), entropy=float(entropy),
                topk_ids=top.indices.tolist(), topk_logprobs=top.values.tolist()))
            logits[idx].fill_(-1e30)
            logits[idx, token] = 0
        return logits


class CaptureMiddleware:
    """ASGI interception: record original/effective API requests and streamed bytes."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        original = json.loads(body)
        effective = dict(original)
        trace_id = uuid.uuid4().hex
        effective["vllm_xargs"] = dict(original.get("vllm_xargs") or {}, trace_id=trace_id)
        effective.update(logprobs=True, top_logprobs=20, return_token_ids=True)
        if effective.get("prompt_logprobs") is not None:
            raise ValueError("prompt_logprobs would change prefix-cache reads")
        append("api.jsonl", dict(kind="api_request", trace_id=trace_id,
                                 original=original, effective=effective))
        encoded = json.dumps(effective).encode()
        scope = dict(scope)
        scope["headers"] = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-length"]
        scope["headers"].append((b"content-length", str(len(encoded)).encode()))
        delivered = False

        async def replacement_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": encoded, "more_body": False}
            return await receive()

        async def recording_send(message):
            if message["type"] == "http.response.start":
                append("api.jsonl", dict(kind="api_status", trace_id=trace_id, status=message["status"]))
            elif message["type"] == "http.response.body":
                append("api.jsonl", dict(kind="api_response", trace_id=trace_id,
                    body=message.get("body", b"").decode(), more_body=message.get("more_body", False)))
            await send(message)

        return await self.app(scope, replacement_receive, recording_send)
