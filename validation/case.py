"""Build/verify request traces, without importing the inference stack."""
import json
import math
from pathlib import Path
from validation.serialization import json_safe


def read_events(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assemble(events, *, require_logprobs=True):
    requests = {}
    for event in events:
        kind, rid = event.get("kind"), event.get("request_id")
        if kind == "request":
            if rid in requests:
                raise ValueError(f"duplicate request: {rid}")
            ids = event["prompt_ids"]
            if not ids or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError(f"invalid prompt IDs: {rid}")
            requests[rid] = dict(request_id=rid, prompt_ids=ids, sampling=event["sampling"],
                                 cache_salt=event.get("cache_salt"),
                                 output_ids=[], online_logprobs=[], online_topk=[],
                                 schedule=[], finish_reason=None, cached_tokens=None)
        elif kind in ("schedule", "output", "finish"):
            if rid not in requests:
                raise ValueError(f"event without request: {rid}")
            item = requests[rid]
            if kind == "finish":
                item["finish_reason"] = event.get("finish_reason")
                item["termination_source"] = event.get("termination_source")
            elif kind == "schedule":
                if event["end"] <= event["start"]:
                    raise ValueError("invalid scheduled token range")
                item["schedule"].append({k: event[k] for k in
                    ("start", "end", "cached_tokens", "preemptions", "batch_size")})
                if item["cached_tokens"] is None:
                    item["cached_tokens"] = event["start"]
            else:
                ids, rows = event["ids"], event.get("logprobs", [])
                if require_logprobs and len(ids) != len(rows):
                    raise ValueError(f"logprob count differs from token count: {rid}")
                for i, token in enumerate(ids):
                    if type(token) is not int or token < 0:
                        raise ValueError("invalid output ID")
                    if require_logprobs:
                        row = rows[i]
                        if str(token) not in row:
                            raise ValueError(f"sampled token {token} absent from logprobs: {rid}")
                        value = float(row[str(token)])
                        item["online_logprobs"].append(json_safe(value))
                        item["online_topk"].append(row)
                item["output_ids"].extend(ids)
                if event.get("finish_reason") is not None:
                    item["finish_reason"] = event["finish_reason"]
    for item in requests.values():
        if item["finish_reason"] is None:
            raise ValueError(f"unfinished request: {item['request_id']}")
        if item["finish_reason"] not in ("stop", "length"):
            raise ValueError(f"unsupported termination: {item['finish_reason']} for {item['request_id']} "
                             f"({item.get('termination_source', 'engine output')})")
        if not item["output_ids"] or not item["schedule"]:
            raise ValueError("request has no output/schedule")
        if any(step["preemptions"] or step["batch_size"] != 1 for step in item["schedule"]):
            raise ValueError("stage-1 supports non-preempted serial single-sequence traces only")
    if not requests:
        raise ValueError("empty trace")
    return list(requests.values())


def validate_tolerance(tolerance):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")


def compare(original, replay, *, tolerance=1e-5):
    validate_tolerance(tolerance)
    if len(original) != len(replay):
        raise ValueError("request count mismatch")
    errors, reports = [], []
    nonfinite_count, total_tokens = 0, 0
    for index, (left, right) in enumerate(zip(original, replay)):
        token_match = left["prompt_ids"] == right["prompt_ids"] and left["output_ids"] == right["output_ids"]
        cache_match = left["cached_tokens"] == right["cached_tokens"] and left["schedule"] == right["schedule"]
        if len(left["online_logprobs"]) != len(right["online_logprobs"]):
            raise ValueError("probability alignment mismatch")
        pairs = [(float(a), float(b)) for a, b in zip(left["online_logprobs"], right["online_logprobs"])]
        if not pairs:
            raise ValueError("missing reference probabilities")
        bad = [step for step, (a, b) in enumerate(pairs) if not (math.isfinite(a) and math.isfinite(b))]
        diff = [abs(a-b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
        total_tokens += len(pairs)
        nonfinite_count += len(bad)
        errors.extend(diff)
        reports.append(dict(index=index, request_id=left["request_id"], n_prompt=len(left["prompt_ids"]),
                            n_output=len(left["output_ids"]), cached_tokens=left["cached_tokens"],
                            replay_cached_tokens=right["cached_tokens"], token_match=token_match,
                            cache_path_match=cache_match, nonfinite_steps=bad,
                            max_abs_logprob_error=max(diff) if diff else None))
    return dict(request_count=len(reports), total_output_tokens=total_tokens,
                token_match=all(r["token_match"] for r in reports),
                cache_path_match=all(r["cache_path_match"] for r in reports),
                numerical_match=bool(errors) and nonfinite_count == 0 and max(errors) <= tolerance,
                nonfinite_logprob_count=nonfinite_count, finite_comparison_tokens=len(errors), tolerance=tolerance,
                mean_abs_logprob_error=sum(errors)/len(errors) if errors else None,
                max_abs_logprob_error=max(errors) if errors else None, requests=reports)
