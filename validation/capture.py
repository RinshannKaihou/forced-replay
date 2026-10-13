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
from validation.serialization import json_safe


def append(filename, event):
    root = os.environ.get("FORCED_REPLAY_TRACE_DIR")
    if not root:
        raise RuntimeError("FORCED_REPLAY_TRACE_DIR must be configured")
    path = Path(root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event, timestamp_ns=time.time_ns(), pid=os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
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
    if isinstance(config, (set, frozenset)):
        values = [json_config(value) for value in config]
        return sorted(values, key=lambda value: json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False))
    if isinstance(config, (list, tuple)):
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
        self._trace_hash_counts = {}
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
            if os.environ.get("FORCED_REPLAY_MODE") == "replay" and extra.get("forced_ids") is not None:
                if batch_size != 1 or self.scheduler_config.async_scheduling:
                    raise ValueError("measurement sidecar requires synchronous serial scheduling")
                path = active_path(extra["trace_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(request_id=rid, start=start, end=end,
                                                 prompt_length=req.num_prompt_tokens)), encoding="utf-8")
            previous = self._trace_hash_counts.get(rid, 0)
            reset = len(req.block_hashes) < previous
            if reset:
                previous = 0
            hashes = [bytes(h).hex() for h in req.block_hashes[previous:]]
            self._trace_hash_counts[rid] = len(req.block_hashes)
            append("engine.jsonl", dict(kind="schedule", request_id=rid, start=start, end=end,
                prompt_length=req.num_prompt_tokens, cached_tokens=self._trace_cache[rid],
                preemptions=req.num_preemptions, batch_size=batch_size,
                block_hash_start=previous, new_block_hashes=hashes, block_hash_reset=reset))
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
                    self._trace_hash_counts.pop(output.request_id, None)
        return result

    def finish_requests(self, request_ids, finished_status):
        finished = super().finish_requests(request_ids, finished_status)
        for request in finished:
            reason = request.get_finished_reason()
            append("engine.jsonl", dict(kind="finish", request_id=request.request_id,
                finish_reason=str(reason) if reason is not None else None,
                stop_reason=request.stop_reason, finished_status=finished_status.name,
                n_output=request.num_output_tokens, termination_source="scheduler.finish_requests"))
            self._trace_cache.pop(request.request_id, None)
            self._trace_hash_counts.pop(request.request_id, None)
        return finished


class ReplayProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.rows = {}
        self.require_schedule = vllm_config is not None
        # None is used only by CPU unit fixtures; real vLLM supplies its config.
        self.allow_forcing = vllm_config is None or os.environ.get("FORCED_REPLAY_MODE") == "replay"

    @classmethod
    def validate_params(cls, params):
        extra = params.extra_args or {}
        if "forced_ids" not in extra:
            return
        if os.environ.get("FORCED_REPLAY_MODE") != "replay":
            raise ValueError("internal forcing is permitted only in offline replay")
        trace_id, forced = extra.get("trace_id"), extra["forced_ids"]
        if not isinstance(trace_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace_id):
            raise ValueError("offline forcing requires a safe server trace_id")
        if not isinstance(forced, list) or not forced or any(type(t) is not int or t < 0 for t in forced):
            raise ValueError("forced_ids must be a nonempty list of nonnegative token IDs")

    def is_argmax_invariant(self):
        return False  # Identity online, deliberately non-invariant when forcing offline.

    def update_state(self, update):
        if update is None:
            return
        for idx in update.removed:
            self.rows.pop(idx, None)
        for idx, params, _prompt, output_ref in update.added:
            extra = params.extra_args or {}
            forced = extra.get("forced_ids") if self.allow_forcing else None
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
                scheduled = json.loads(active_path(trace_id).read_text(encoding="utf-8"))
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
            invalid_distribution = bool(torch.isnan(logp).any() | torch.isposinf(logp).any())
            append("measurements.jsonl", dict(kind="measurement", trace_id=trace_id, step=step,
                forced_id=token, forced_logprob=float(logp[token]),
                forced_rank=-1 if invalid_distribution else int((logp > logp[token]).sum()),
                distribution_valid=not invalid_distribution, entropy=float(entropy),
                topk_ids=top.indices.tolist(), topk_logprobs=top.values.tolist()))
            logits[idx].fill_(-1e30)
            logits[idx, token] = 0
        return logits


# Count the root container as depth 1; leave recursion headroom for trace wrappers
# and json_safe's recursive normalization while allowing ordinary tool schemas.
MAX_JSON_DEPTH = 64


def reject_non_json_constant(token):
    raise ValueError(f"nonstandard JSON number: {token}")


def validate_json_depth(value):
    """Bound decoded object/array nesting without recursive validation."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if not isinstance(item, (dict, list)):
            continue
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON nesting exceeds maximum depth of {MAX_JSON_DEPTH}")
        children = item.values() if isinstance(item, dict) else item
        stack.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))


class CaptureMiddleware:
    """ASGI interception: record original/effective API requests and streamed bytes."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        if scope["method"] != "POST" or scope["path"] != "/v1/chat/completions":
            return await self.reject(scope, receive, send, "capture supports only POST /v1/chat/completions")
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            original = json.loads(body, parse_constant=reject_non_json_constant)
        except RecursionError:
            return await self.reject(scope, receive, send, f"JSON nesting exceeds maximum depth of {MAX_JSON_DEPTH}")
        except (ValueError, UnicodeDecodeError):
            return await self.reject(scope, receive, send, "invalid JSON request body")
        try:
            validate_json_depth(original)
        except ValueError as exc:
            return await self.reject(scope, receive, send, str(exc))
        if not isinstance(original, dict):
            return await self.reject(scope, receive, send, "request must be a JSON object")
        extra = original.get("vllm_xargs")
        if extra is not None and not isinstance(extra, dict):
            return await self.reject(scope, receive, send, "vllm_xargs must be an object")
        if {"trace_id", "forced_ids"}.intersection(extra or {}):
            return await self.reject(scope, receive, send, "trace_id and forced_ids are reserved server fields")
        if original.get("n", 1) != 1 or original.get("prompt_logprobs") is not None:
            return await self.reject(scope, receive, send, "capture requires one sequence and no prompt_logprobs")
        if original.get("stop"):
            return await self.reject(scope, receive, send, "stop strings are outside stage-1 capture scope")
        if original.get("response_format") is not None or original.get("structured_outputs") is not None:
            return await self.reject(scope, receive, send, "structured generation is outside stage-1 capture scope")
        if original.get("tool_choice", "auto") not in (None, "auto", "none"):
            return await self.reject(scope, receive, send, "constrained tool_choice is outside stage-1 capture scope")
        if not isinstance(original.get("messages", []), list):
            return await self.reject(scope, receive, send, "messages must be an array")
        for message in original.get("messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list) and any(not isinstance(item, dict) or item.get("type") != "text" for item in content):
                return await self.reject(scope, receive, send, "multimodal input is outside stage-1 capture scope")
        if original.get("prompt_embeds") is not None:
            return await self.reject(scope, receive, send, "embedding input is outside stage-1 capture scope")
        effective = dict(original)
        trace_id = uuid.uuid4().hex
        effective["vllm_xargs"] = dict(extra or {}, trace_id=trace_id)
        effective.update(logprobs=True, top_logprobs=20, return_token_ids=True)
        try:
            encoded = json.dumps(effective, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except RecursionError:
            return await self.reject(scope, receive, send, f"JSON nesting exceeds maximum depth of {MAX_JSON_DEPTH}")
        except (ValueError, UnicodeError):
            return await self.reject(scope, receive, send, "request must contain valid UTF-8 and finite JSON numbers")
        try:
            append("api.jsonl", dict(kind="api_request", trace_id=trace_id,
                                     original=original, effective=effective))
        except RecursionError:
            return await self.reject(scope, receive, send, f"JSON nesting exceeds maximum depth of {MAX_JSON_DEPTH}")
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

    async def reject(self, scope, receive, send, message):
        from starlette.responses import JSONResponse
        append("api.jsonl", dict(kind="api_rejection", path=scope["path"], reason=message))
        response = JSONResponse({"error": {"type": "capture_scope_error", "message": message}}, status_code=400)
        await response(scope, receive, send)
