---
name: forced-replay
description: >-
  Force-replay a saved LLM generation trajectory under runtime decode kernels to
  recover its TRUE per-token logprobs / rank / entropy / top-k. Use when the user
  has a model plus a saved (ill/anomalous) trajectory — repetition loop, gibberish,
  divergence — and asks for faithful logprobs, trajectory rescoring, replay
  diagnosis: "force replay", "forced decode replay", "重放轨迹", "拿真实 logprobs",
  "这条输出当时每步的概率", "为什么模型在这里卡住/循环". Not for sampling new
  outputs or live serving.
---

# Forced decode-replay (runtime-faithful trajectory logprobs)

## What & why

Re-scoring a trajectory by feeding `[prompt+output]` through ONE prefill pass is
**unfaithful**: prefill and decode use different kernels, and measured deviation
reaches ~1 nat in the tail (mean 0.045, p95 0.21 on Qwen3-30B-A3B-FP8/vLLM 0.26).
This skill replays the trajectory the right way: prefill the original prompt only,
then force each recorded token at decode time via a masked LogitsProcessor and
snapshot the PRE-MASK distribution — decode runs the runtime kernels, so the
numbers are what the serving engine actually computed (validated bit-exact
against online logprobs). Evidence and method details: `references/fidelity.md`.

## Workflow

1. **Collect inputs** (ask the user; never guess silently):
   - model path (same weights as deployment);
   - trajectory: JSON `{"prompt_ids": [...], "output_ids": [...]}` (token ids).
     If the user only has TEXT prompt+output: tokenize yourself with the model's
     tokenizer — encode the prompt normally, encode the output with
     `add_special_tokens=False`, and NEVER encode the concatenated string
     (runtime decode appends ids directly). Save as ids JSON, then proceed.
   - the deployment serving config that produced the trajectory (serve command /
     config file). Required for step 2.
2. **Map deployment config → replay flags** (each mismatch is a SILENT wrongness):
   | deployment | replay flag |
   |---|---|
   | `--quantization fp8` / quant config | `--quantization fp8` |
   | dtype | `--dtype` |
   | `--enforce-eager` (no CUDA graphs) | `--enforce-eager` |
   | `--tensor-parallel-size N` | `--tp N` |
   | `--max-model-len` | `--max-model-len` |
   | `--max-num-batched-tokens` (chunked prefill) | same flag (only matters if the prompt exceeds the chunk) |
   If the user does not know a value, say which values you assumed and record
   them in the final report — do not bury assumptions.
3. **Check the environment**: vLLM 0.26.x + CUDA GPU required (this skill's
   LogitsProcessor protocol is version-pinned — on any other version, read
   `references/fidelity.md` §"vLLM API surface" and adapt `scripts/replay.py`
   BEFORE running). On the analysis box use `/root/llm-venv/bin/python`.
   Known good on this box: `/workspace/models/Qwen3-30B-A3B` with
   `--quantization fp8`.
4. **Run** (shared GPUs: auto-picks the freest card):
   ```bash
   python scripts/replay.py --model <path> --traj <traj.json> \
       [--quantization fp8] [--tp N] [--enforce-eager] ...
   ```
5. **Verify before interpreting**: stdout must show
   `[ok] trajectory reproduced exactly` — if forcing failed or record count
   mismatched, the run is invalid; see `references/fidelity.md` §"Failure modes".
6. **Report**: summary line (mean/min logprob, n(<-5), top1-match, min entropy),
   worst steps, and the diagnosis per `references/interpretation.md`. Attach the
   `.meta.json` sidecar (or its contents) so the result is auditable.

## Hard rules

- Engine args MUST match the deployment (step 2). A mismatch produces confident
  wrong numbers, not an error.
- Never enable prefix caching or speculative decoding in the replay.
- Run one trajectory per invocation (bs=1; preemption would silently poison the
  numbers). Many trajectories = loop invocations (engine reload each time).
- CUDA only in this version. NPU/Ascend is out of scope (notes in fidelity.md).
- The recorded logprobs are RAW (pre-temperature/top-p, full-vocab log_softmax);
  the online sampler saw a modified distribution — keep this straight when
  comparing with serving-returned logprobs.

## Output glossary (`<traj>_replay.npz`)

Per decode step: `forced_logprob` (logprob the runtime kernels gave the actual
token), `forced_rank` (0 = it was the argmax), `top1_id/top1_logprob` (what the
model actually preferred), `entropy`, `topk_ids/topk_logprobs`, `forced_id`,
`step`. Sidecar `<out>.meta.json` records the exact engine config used.
