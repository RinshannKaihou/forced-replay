"""Reusable local API/agent/session-replay laboratory (vLLM 0.26.0)."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from validation.case import assemble, compare, read_events

ROOT = Path(__file__).resolve().parents[1]
PROCESSOR = "validation.capture:ReplayProcessor"


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def forced_sampling(original, ids):
    if not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("invalid forced output IDs")
    if original.get("n", 1) != 1:
        raise ValueError("stage-1 requires n=1")
    for name in ("structured_outputs", "allowed_token_ids", "bad_words", "logit_bias"):
        if original.get(name):
            raise ValueError(f"stage-1 does not support {name}")
    values = {k: v for k, v in original.items() if not k.startswith("_") and k not in
              {"output_kind", "routed_experts_prompt_start"}}
    values.update(temperature=0.0, max_tokens=len(ids), min_tokens=0, stop=[], stop_token_ids=[],
                  ignore_eos=True, detokenize=False, logprobs=None, prompt_logprobs=None)
    extra = dict(original.get("extra_args") or {})
    extra["forced_ids"] = ids
    values["extra_args"] = extra
    return values


def measured_probabilities(request, measurements):
    trace_id = request["sampling"]["extra_args"]["trace_id"]
    rows = sorted((m for m in measurements if m["trace_id"] == trace_id), key=lambda m: m["step"])
    if [m["step"] for m in rows] != list(range(len(request["output_ids"]))) or \
            [m["forced_id"] for m in rows] != request["output_ids"]:
        raise ValueError(f"measurement count/step/token mismatch: {trace_id}")
    return [m["forced_logprob"] for m in rows]


def lab_environment(trace_dir, gpu):
    env = os.environ.copy()
    env.pop('VLLM_API_KEY', None)
    # Do not inherit the unrelated fault-injection entry-point plugin or auth.
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_PLUGINS="", HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
        FORCED_REPLAY_TRACE_DIR=str(trace_dir), PYTHONPATH=str(ROOT))
    return env


def serve(args):
    run = Path(args.run_dir).resolve()
    online = run / "online"
    online.mkdir(parents=True, exist_ok=False)
    options = dict(model=str(Path(args.model).resolve()), dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_util, max_model_len=32768, max_num_seqs=1,
        max_num_batched_tokens=1024, enable_prefix_caching=True, enable_chunked_prefill=True,
        async_scheduling=False, seed=0, generation_config="vllm", max_logprobs=20,
        enforce_eager=args.enforce_eager, scheduler_cls="validation.capture.TraceScheduler",
        logits_processors=[PROCESSOR])
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if key == "enforce_eager":
            if value:
                command.append(flag)
        elif isinstance(value, bool):
            command.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            command.extend([flag] + value)
        else:
            command.extend([flag, str(value)])
    command += ["--host", "127.0.0.1", "--port", str(args.port), "--served-model-name", args.name,
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", '{"enable_thinking": false}',
                "--middleware", "validation.capture.CaptureMiddleware"]
    files = []
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = Path(options["model"]) / name
        if path.exists():
            files.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    gpu_info = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
        "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"]).decode().strip()
    manifest = dict(engine_kwargs=options, api_command=command, model_name=args.name,
        port=args.port, gpu=args.gpu, gpu_info=gpu_info, python=sys.executable,
        packages={p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers")},
        fingerprint=files, weight_files=[dict(name=p.name, bytes=p.stat().st_size)
            for p in sorted(Path(options["model"]).glob("*.safetensors"))],
        stage1_scope="CUDA TP1 serial, text/tool-calling, APC, no speculation; reasoning disabled")
    dump(run / "manifest.json", manifest)
    print("[serve] " + " ".join(command), flush=True)
    os.chdir(ROOT)
    os.execve(sys.executable, command, lab_environment(online, args.gpu))


def replay_options(manifest, resolved, mode):
    options = dict(manifest["engine_kwargs"])
    options["served_model_name"] = resolved["model_config"]["served_model_name"]
    options["structured_outputs_config"] = dict(resolved["structured_outputs_config"])
    if mode == "rebuild-prefix":
        options["enable_prefix_caching"] = False
    return options


def configuration_differences(online, offline):
    differences = {}
    for group in ("model_config", "cache_config", "parallel_config", "scheduler_config",
                  "compilation_config", "structured_outputs_config", "kernel_config"):
        left, right = online.get(group, {}), offline.get(group, {})
        changed = {key: {"online": left.get(key), "replay": right.get(key)}
                   for key in left.keys() | right.keys() if key != "compilation_time" and left.get(key) != right.get(key)}
        if changed:
            differences[group] = changed
    return differences


def replay(args):
    run = Path(args.run_dir).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    online_events = read_events(run / "online" / "engine.jsonl")
    online = assemble(online_events)
    online_config = next(event["config"] for event in online_events if event["kind"] == "engine")
    dump(run / "case.json", dict(manifest=manifest, requests=online))
    target = run / ("replay" if args.mode == "session" else "rebuild_prefix")
    target.mkdir(exist_ok=False)
    os.environ.update(lab_environment(target, args.gpu))
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    options = replay_options(manifest, online_config, args.mode)
    llm = LLM(**options)
    generated = []
    for index, request in enumerate(online):
        if request.get("cache_salt"):
            raise ValueError("stage-1 currently requires un-salted local APC")
        params = SamplingParams(**forced_sampling(request["sampling"], request["output_ids"]))
        result = llm.generate(TokensPrompt(prompt_token_ids=request["prompt_ids"]), params,
                              use_tqdm=False)[0]
        ids = list(result.outputs[0].token_ids)
        if ids != request["output_ids"]:
            raise ValueError(f"forced output mismatch at request {index}")
        generated.append(dict(index=index, cached_tokens=result.num_cached_tokens, n_output=len(ids)))
        print(f"[replay] request={index} output={len(ids)} cached={result.num_cached_tokens}", flush=True)
    replay_events = read_events(target / "engine.jsonl")
    actual = assemble(replay_events, require_logprobs=False)
    replay_config = next(event["config"] for event in replay_events if event["kind"] == "engine")
    config_differences = configuration_differences(online_config, replay_config)
    measurements = read_events(target / "measurements.jsonl")
    for request in actual:
        request["online_logprobs"] = measured_probabilities(request, measurements)
    report = compare(online, actual, tolerance=args.tolerance)
    report.update(mode=args.mode, backend="CUDA", generated=generated,
        configuration_match=not config_differences, configuration_differences=config_differences,
        configuration_scope="model/cache/parallel/scheduler/compilation/structured-output/kernel configs; exclude compilation wall time",
        verification_scope="exact tokens + serial logical compute ranges/cache-hit lengths + sampled-token raw logprobs; no KV-byte comparison",
        online_api_requests=sum(e.get("kind") == "api_request" for e in read_events(run / "online" / "api.jsonl")))
    dump(target / "report.json", report)
    dump(target / "requests.json", actual)
    import numpy as np
    np.savez_compressed(target / "measurements.npz",
        trace_id=np.array([m["trace_id"] for m in measurements]),
        step=np.array([m["step"] for m in measurements]),
        forced_id=np.array([m["forced_id"] for m in measurements]),
        forced_logprob=np.array([m["forced_logprob"] for m in measurements]),
        forced_rank=np.array([m["forced_rank"] for m in measurements]),
        entropy=np.array([m["entropy"] for m in measurements]),
        topk_ids=np.array([m["topk_ids"] for m in measurements]),
        topk_logprobs=np.array([m["topk_logprobs"] for m in measurements]))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "session" and not all(report[k] for k in ("token_match", "cache_path_match", "numerical_match", "configuration_match")):
        raise SystemExit("[FAILED] stage-1 verification failed; see report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("serve")
    start.add_argument("--run-dir", required=True)
    start.add_argument("--model", default="/workspace/models/Qwen3-8B")
    start.add_argument("--name", default="replay-lab-qwen3-8b")
    start.add_argument("--port", type=int, default=18180)
    start.add_argument("--gpu", type=int, default=0)
    start.add_argument("--gpu-util", type=float, default=0.3)
    start.add_argument("--enforce-eager", action="store_true")
    start.set_defaults(function=serve)
    again = commands.add_parser("replay")
    again.add_argument("--run-dir", required=True)
    again.add_argument("--gpu", type=int, default=0)
    again.add_argument("--mode", choices=["session", "rebuild-prefix"], default="session")
    again.add_argument("--tolerance", type=float, default=1e-5)
    again.set_defaults(function=replay)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
