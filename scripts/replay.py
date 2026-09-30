#!/usr/bin/env python3
"""Forced decode-replay: recover the TRUE runtime logprobs of a saved trajectory.

Prefill [x_1..x_n] with the production engine (CUDA graphs on by default), then
at each decode step mask all logits except the recorded token x_{n'} and
snapshot the PRE-MASK distribution. Decode runs the runtime kernels, so the
recorded logprobs are what the serving engine actually computed. Validated
bit-exact against online logprobs on vLLM 0.26 / Qwen3-30B-A3B-FP8.

Input : JSON {"prompt_ids": [...], "output_ids": [...]}
Output: <out>.npz  per-step logprob/rank/top1/entropy/top-k
        <out>.meta.json  engine config actually used (audit sidecar)

FIDELITY: engine args must MATCH the deployment that produced the trajectory
(--quantization / --dtype / --enforce-eager / --tp / --max-model-len /
--max-num-batched-tokens). Mismatch fails silently with wrong numbers.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

# ---------------------------------------------------------------- CLI
def _parse_args():
    p = argparse.ArgumentParser(
        description="Forced decode-replay of a saved trajectory (runtime-faithful logprobs)")
    p.add_argument("--model", required=True, help="model path (must match deployment)")
    p.add_argument("--traj", required=True,
                   help='trajectory JSON {"prompt_ids": [...], "output_ids": [...]}')
    p.add_argument("--out", default=None,
                   help="output npz path (default: <traj>_replay.npz beside the traj file)")
    # ---- fidelity knobs: must match the deployment that produced the traj ----
    p.add_argument("--quantization", default=None,
                   help='e.g. "fp8" if deployed with dynamic FP8 (default: checkpoint default)')
    p.add_argument("--dtype", default="auto", help="deployment dtype (default auto)")
    p.add_argument("--enforce-eager", action="store_true",
                   help="ONLY if the deployment served with eager mode (no CUDA graphs)")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size (match deployment)")
    p.add_argument("--max-model-len", type=int, default=None, help="match deployment")
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="chunked-prefill size; set only to match deployment (matters when"
                        " the prompt is longer than the deployment chunk size)")
    # ---- measurement knobs ----
    p.add_argument("--topk", type=int, default=20, help="top-k snapshot size per step")
    p.add_argument("--max-num-seqs", type=int, default=8,
                   help="vLLM batch slots (bs=1 experiment; small keeps graph memory low)")
    # ---- GPU placement ----
    p.add_argument("--no-gpu-auto", action="store_true",
                   help="don't pick a GPU; honor CUDA_VISIBLE_DEVICES and use --gpu-util")
    p.add_argument("--gpu-util", type=float, default=0.85, help="used only with --no-gpu-auto")
    p.add_argument("--gpu-mem-gib", type=float, default=0,
                   help="our VRAM budget in GiB for auto mode (0 = estimate from weights)")
    p.add_argument("--gpus", default="all", help='allowed GPU indices, e.g. "0,1,2,3"')
    return p.parse_args()


ARGS = _parse_args()
TOPK_RECORD = ARGS.topk
NEG = -1e30


# ---------------------------------------------------------------- GPU
def _weights_gib(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                total += os.path.getsize(os.path.join(root, f))
    return total / 2**30


def _budget_gib(args):
    if args.gpu_mem_gib > 0:
        return args.gpu_mem_gib
    disk = _weights_gib(args.model)
    if disk == 0:
        sys.exit(f"[error] no weight files found under {args.model}")
    # dynamic fp8 keeps bf16 on disk (~0.55x at runtime); otherwise ~1x + overhead
    factor = 0.6 if args.quantization == "fp8" else 1.1
    return disk * factor + 8


def _setup_gpu(args):
    """Pick the freest allowed GPU and set CUDA_VISIBLE_DEVICES before torch loads.

    vLLM: requested = gpu_memory_utilization * total is OUR OWN budget
    (weights + activations + graphs + KV) and must be <= free memory.
    """
    if args.no_gpu_auto:
        return {"gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
                "util": args.gpu_util, "budget_gib": None}
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"]).decode()
    allowed = (None if args.gpus == "all"
               else {int(x) for x in args.gpus.split(",")})
    cands = []
    for line in out.strip().splitlines():
        idx, used, total = (int(x) for x in line.split(","))
        if allowed is None or idx in allowed:
            cands.append((total - used, idx, total))   # sort by free
    cands.sort(reverse=True)
    budget = _budget_gib(args)
    margin = 6
    for free, idx, total in cands:
        if free >= (budget + margin) * 1024:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return {"gpu": idx, "util": budget * 1024 / total, "budget_gib": budget}
    sys.exit(f"[error] no GPU in {args.gpus} has {budget:.0f}+{margin} GiB free: "
             f"{[(i, f // 1024) for f, i, _ in cands]}")


GPU_INFO = _setup_gpu(ARGS)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # in-process RECORDS

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import vllm                                                       # noqa: E402
from vllm import LLM, SamplingParams                              # noqa: E402
from vllm.inputs import TokensPrompt                              # noqa: E402
from vllm.v1.sample.logits_processor import LogitsProcessor       # noqa: E402

RECORDS = []   # engine constructs the LP internally; collect via global (TP=1, in-process)


# ---------------------------------------------------------------- forcing LP
class ForcedTrajectoryLP(LogitsProcessor):
    """Masks all logits except the recorded token; snapshots pre-mask stats."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.rows = {}          # batch row index -> (forced_ids, live output_tok_ids ref)

    def is_argmax_invariant(self) -> bool:
        return False            # we deliberately change the argmax

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for entry in batch_update.added:
            idx, params, _prompt_ids, out_ids = entry[0], entry[1], entry[2], entry[3]
            extra = getattr(params, "extra_args", None) or {}
            forced = extra.get("forced_ids")
            if forced is not None:
                # out_ids is a LIVE reference; len(out_ids) == current decode step
                self.rows[idx] = (list(forced), out_ids)
            else:
                self.rows.pop(idx, None)
        for idx in batch_update.removed:
            self.rows.pop(idx, None)
        for mv in batch_update.moved:                    # (a, b, direction)
            a, b = mv[0], mv[1]
            va, vb = self.rows.pop(a, None), self.rows.pop(b, None)
            if va is not None:
                self.rows[b] = va
            if vb is not None:
                self.rows[a] = vb

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.rows:
            return logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for idx, (forced, out_ids) in self.rows.items():
            pos = len(out_ids)
            if pos >= len(forced):
                continue
            t = int(forced[pos])
            row = logp[idx]
            tk = torch.topk(row, TOPK_RECORD)
            RECORDS.append(dict(
                step           = pos,
                forced_id      = t,
                forced_logprob = float(row[t]),
                forced_rank    = int((row > row[t]).sum()),
                top1_id        = int(tk.indices[0]),
                top1_logprob   = float(tk.values[0]),
                entropy        = float(-(row.exp() * row).sum()),
                topk_ids       = tk.indices.tolist(),
                topk_logprobs  = tk.values.tolist(),
            ))
            logits[idx].fill_(NEG)          # finite, not -inf: avoids NaN downstream
            logits[idx, t] = 0.0
        return logits


# ---------------------------------------------------------------- main
def main():
    traj = json.load(open(ARGS.traj))
    prompt_ids = list(traj["prompt_ids"])
    output_ids = list(traj["output_ids"])
    if not output_ids:
        sys.exit("[error] trajectory has empty output_ids")
    out_path = ARGS.out or os.path.join(
        os.path.dirname(os.path.abspath(ARGS.traj)),
        os.path.splitext(os.path.basename(ARGS.traj))[0] + "_replay.npz")

    print(f"[traj] n_prefill={len(prompt_ids)}  n_decode={len(output_ids)}")
    print(f"[fidelity] model={ARGS.model}  quantization={ARGS.quantization}  "
          f"dtype={ARGS.dtype}  enforce_eager={ARGS.enforce_eager}  tp={ARGS.tp}  "
          f"max_model_len={ARGS.max_model_len}  "
          f"max_num_batched_tokens={ARGS.max_num_batched_tokens}  "
          f"prefix_caching=False(always)")
    print(f"[gpu] {GPU_INFO}")

    llm = LLM(
        model                     = ARGS.model,
        tensor_parallel_size      = ARGS.tp,
        max_model_len             = ARGS.max_model_len,
        max_num_seqs              = ARGS.max_num_seqs,
        max_num_batched_tokens    = ARGS.max_num_batched_tokens,
        gpu_memory_utilization    = GPU_INFO["util"],
        enforce_eager             = ARGS.enforce_eager,
        enable_prefix_caching     = False,   # match production explicitly; never cache here
        quantization              = ARGS.quantization,
        dtype                     = ARGS.dtype,
        logits_processors         = [ForcedTrajectoryLP],
    )

    sp = SamplingParams(
        temperature = 0.0,          # greedy: the mask alone fully determines the token
        max_tokens  = len(output_ids),
        logprobs    = None,         # engine-returned logprobs are POST-mask; useless
        detokenize  = False,
        extra_args  = {"forced_ids": output_ids},
    )

    out = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), sp)[0]
    got = list(out.outputs[0].token_ids)

    if got != output_ids:
        bad = next(i for i, (a, b) in enumerate(zip(got, output_ids)) if a != b)
        sys.exit(f"[error] forcing failed at step {bad}: engine {got[bad]} != "
                 f"recorded {output_ids[bad]} (engine internals diverged; report this)")
    if len(RECORDS) != len(output_ids) or \
            sorted(r["step"] for r in RECORDS) != list(range(len(output_ids))):
        sys.exit(f"[error] record count mismatch: {len(RECORDS)} records for "
                 f"{len(output_ids)} steps")
    print(f"[ok] trajectory reproduced exactly ({len(got)} tokens)")

    RECORDS.sort(key=lambda r: r["step"])
    np.savez_compressed(
        out_path,
        step           = np.array([r["step"] for r in RECORDS]),
        forced_id      = np.array([r["forced_id"] for r in RECORDS]),
        forced_logprob = np.array([r["forced_logprob"] for r in RECORDS]),
        forced_rank    = np.array([r["forced_rank"] for r in RECORDS]),
        top1_id        = np.array([r["top1_id"] for r in RECORDS]),
        top1_logprob   = np.array([r["top1_logprob"] for r in RECORDS]),
        entropy        = np.array([r["entropy"] for r in RECORDS]),
        topk_ids       = np.array([r["topk_ids"] for r in RECORDS]),
        topk_logprobs  = np.array([r["topk_logprobs"] for r in RECORDS]),
    )
    meta = {
        "model": ARGS.model, "quantization": ARGS.quantization, "dtype": ARGS.dtype,
        "enforce_eager": ARGS.enforce_eager, "tp": ARGS.tp,
        "max_model_len": ARGS.max_model_len,
        "max_num_batched_tokens": ARGS.max_num_batched_tokens,
        "prefix_caching": False, "max_num_seqs": ARGS.max_num_seqs,
        "topk": TOPK_RECORD, "vllm_version": vllm.__version__,
        "gpu": GPU_INFO, "traj": os.path.abspath(ARGS.traj),
        "cmdline": shlex.join(sys.argv),
        "n_prefill": len(prompt_ids), "n_decode": len(output_ids),
        "token_match": True, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[saved] {out_path}\n[saved] {meta_path}")

    lp = np.array([r["forced_logprob"] for r in RECORDS])
    rank = np.array([r["forced_rank"] for r in RECORDS])
    ent = np.array([r["entropy"] for r in RECORDS])
    worst = np.argsort(lp)[:5]
    print(f"[summary] mean={lp.mean():.3f}  min={lp.min():.3f} @step={int(lp.argmin())}  "
          f"n(<-5)={int((lp < -5).sum())}  top1-match={100 * (rank == 0).mean():.1f}%  "
          f"min-entropy={ent.min():.3f} @step={int(ent.argmin())}")
    print(f"[worst ] steps/token/logprob: "
          + "  ".join(f"{int(RECORDS[s]['step'])}/{RECORDS[s]['forced_id']}/{lp[s]:.2f}"
                      for s in worst))


if __name__ == "__main__":
    main()
