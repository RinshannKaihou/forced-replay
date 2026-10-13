"""Reusable local API/agent/session-replay laboratory (vLLM 0.26.0)."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from validation.case import assemble, compare, read_events, validate_tolerance

ROOT = Path(__file__).resolve().parents[1]
PROCESSOR = "validation.capture:ReplayProcessor"
# Controlled vLLM 0.26 CUDA runtime toggles. All other inherited VLLM_* names
# (including auth, plugins, paths and unlisted kernel overrides) are removed in
# BOTH serve and replay; their pinned-version defaults/engine kwargs apply.
# Record unset toggles as null so a later shell cannot supply a different value.
RUNTIME_ENVIRONMENT_KEYS = (
    "VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_TRITON_USE_TD",
    "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0",
    "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", "VLLM_USE_TRITON_AWQ",
    "VLLM_MARLIN_USE_ATOMIC_ADD", "VLLM_USE_BREAKABLE_CUDAGRAPH",
)
RUNTIME_ENVIRONMENT_POLICY = "controlled-vllm-whitelist-v1"
MODEL_IDENTITY_SCOPE = "config/tokenizer/index SHA-256 + safetensors filename/size inventory; no weight-content checksum"


def dump(path, value, *, exclusive=False):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        handle.write(text)


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


def lab_environment(trace_dir, gpu, runtime_environment=None, *, mode="capture"):
    runtime = ({name: os.environ.get(name) for name in RUNTIME_ENVIRONMENT_KEYS}
               if runtime_environment is None else runtime_environment)
    if set(runtime) != set(RUNTIME_ENVIRONMENT_KEYS) or any(
            value is not None and value not in ("0", "1")
            for value in runtime.values()):
        raise ValueError("invalid controlled vLLM runtime environment (literal 0/1 toggles only)")
    env = {name: value for name, value in os.environ.items() if not name.startswith("VLLM_")}
    env.update({name: value for name, value in runtime.items() if value is not None})
    # Do not inherit the unrelated fault-injection entry-point plugin or auth.
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_PLUGINS="", HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
        FORCED_REPLAY_TRACE_DIR=str(trace_dir), FORCED_REPLAY_MODE=mode, PYTHONPATH=str(ROOT))
    return env


def model_identity(model):
    model = Path(model)
    if not model.is_dir():
        raise ValueError(f"local model directory is missing: {model}")
    files = []
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = model / name
        if path.exists():
            files.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    weights = [dict(name=p.name, bytes=p.stat().st_size) for p in sorted(model.glob("*.safetensors"))]
    return files, weights


def verify_model_identity(manifest):
    files, weights = model_identity(manifest["engine_kwargs"]["model"])
    if files != manifest.get("fingerprint"):
        raise ValueError("model fingerprint drift: config/tokenizer/index SHA-256 or file inventory differs")
    if weights != manifest.get("weight_files"):
        raise ValueError("model weight-file inventory/size drift (no weight-content checksum)")


def serve(args):
    run = Path(args.run_dir).resolve()
    if any((run / name).exists() for name in ("manifest.json", "case.json")):
        raise FileExistsError(f"capture evidence already exists: {run}")
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
    files, weights = model_identity(options["model"])
    gpu_info = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
        "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"]).decode("utf-8").strip()
    env = lab_environment(online, args.gpu)
    manifest = dict(engine_kwargs=options, api_command=command, model_name=args.name,
        runtime_environment={name: env.get(name) for name in RUNTIME_ENVIRONMENT_KEYS},
        runtime_environment_policy=RUNTIME_ENVIRONMENT_POLICY,
        port=args.port, gpu=args.gpu, gpu_info=gpu_info, python=sys.executable,
        packages={p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers")},
        fingerprint=files, weight_files=weights,
        stage1_scope="CUDA TP1 serial, text/tool-calling, APC, no speculation; reasoning disabled")
    dump(run / "manifest.json", manifest, exclusive=True)
    print("[serve] " + " ".join(command), flush=True)
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


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
    validate_tolerance(args.tolerance)
    run = Path(args.run_dir).resolve()
    target = run / ("replay" if args.mode == "session" else "rebuild_prefix")
    if target.exists():
        raise FileExistsError(f"replay output already exists: {target}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    online_events = read_events(run / "online" / "engine.jsonl")
    online = assemble(online_events)
    online_config = next(event["config"] for event in online_events if event["kind"] == "engine")
    case_path = run / "case.json"
    case = dict(manifest=manifest, requests=online)
    if case_path.exists() and json.loads(case_path.read_text(encoding="utf-8")) != case:
        raise ValueError("existing case.json differs from manifest/engine trace; refusing to overwrite evidence")
    api_path = run / "online" / "api.jsonl"
    api_requests = (sum(e.get("kind") == "api_request" for e in read_events(api_path))
                    if api_path.exists() else None)
    verify_model_identity(manifest)
    runtime = manifest.get("runtime_environment")
    runtime_known = runtime is not None
    if runtime is None:
        # Legacy evidence cannot certify which VLLM_* runtime was used online.
        runtime = {name: None for name in RUNTIME_ENVIRONMENT_KEYS}
    elif manifest.get("runtime_environment_policy") != RUNTIME_ENVIRONMENT_POLICY:
        raise ValueError("unsupported vLLM runtime environment policy")
    env = lab_environment(target, args.gpu, runtime, mode="replay")
    target.mkdir(exist_ok=False)  # Re-check before creating any evidence files.
    if not case_path.exists():
        dump(case_path, case, exclusive=True)
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            del os.environ[name]
    os.environ.update(env)
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
    runtime_match = ({name: os.environ.get(name) for name in RUNTIME_ENVIRONMENT_KEYS} == runtime
                     if runtime_known else None)
    report.update(mode=args.mode, backend="CUDA", generated=generated,
        model_identity_verified=True, model_identity_scope=MODEL_IDENTITY_SCOPE,
        runtime_environment_match=runtime_match,
        runtime_environment_status="restored" if runtime_known else "unknown",
        runtime_environment_policy=manifest.get("runtime_environment_policy"),
        configuration_match=not config_differences and runtime_match is True,
        configuration_differences=config_differences,
        configuration_scope="model/cache/parallel/scheduler/compilation/structured-output/kernel configs + controlled VLLM runtime environment; exclude compilation wall time",
        verification_scope="exact tokens + serial logical compute ranges/cache-hit lengths + sampled-token raw logprobs; no KV-byte comparison",
        online_api_requests=api_requests,
        online_api_metadata_status="missing" if api_requests is None else "available")
    dump(target / "report.json", report)
    dump(target / "requests.json", actual)
    import numpy as np
    np.savez_compressed(target / "measurements.npz",
        trace_id=np.array([m["trace_id"] for m in measurements]),
        step=np.array([m["step"] for m in measurements]),
        forced_id=np.array([m["forced_id"] for m in measurements]),
        forced_logprob=np.array([m["forced_logprob"] for m in measurements], dtype=float),
        forced_rank=np.array([m["forced_rank"] for m in measurements]),
        entropy=np.array([m["entropy"] for m in measurements], dtype=float),
        topk_ids=np.array([m["topk_ids"] for m in measurements]),
        topk_logprobs=np.array([m["topk_logprobs"] for m in measurements], dtype=float))
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
    if args.command == "replay":
        try:
            validate_tolerance(args.tolerance)
        except ValueError as exc:
            parser.error(str(exc))
    args.function(args)


if __name__ == "__main__":
    main()
