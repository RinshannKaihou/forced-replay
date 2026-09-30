# Fidelity reference

## Why one-prefill rescoring is wrong (measured evidence)

Prefill (chunked GEMM, large-batch matmul) and decode (memory-bound, CUDA-graph
captured) execute different kernel implementations; with dynamic FP8 the
quantization/reduction order also differs. Experiment on this repo
(Qwen3-30B-A3B, FP8, vLLM 0.26, 256-step trajectory, online top-20 as reference):

- one-prefill rescoring vs online decode logprobs: mean |Δ| = 0.045, median
  0.002, p95 = 0.21, **max = 1.03 nats**, corr = 0.986. Small on average,
  badly wrong exactly at the low-logprob steps that matter for diagnosis.
- forced-replay (this skill) vs online: **bit-exact** (max |Δ| = 0, corr = 1.0),
  trajectory reproduced token-for-token.

## Config mapping rationale

Every flag below changes which kernels run or how the prompt is chunked, so a
mismatch changes the numbers without any error:

| replay flag | why it must match deployment |
|---|---|
| `--quantization` | selects quant kernels (fp8 dynamic vs bf16 vs W4A8 …) |
| `--dtype` | activation dtype changes reductions |
| `--enforce-eager` | eager vs CUDA-graph capture is a different execution path; default here is graphs ON (production-like). Only pass it if the deployment served eager. |
| `--tp` | TP>1 splits reductions across ranks |
| `--max-model-len` | interacts with scheduler and graph capture shapes |
| `--max-num-batched-tokens` | chunked-prefill boundary; only numerically relevant when the prompt is longer than the deployment chunk (then the prefill-side KV differs) |
| prefix caching | hardcoded OFF in the replay. Decode-side measurements are unaffected either way (all decode steps are fresh); off keeps runs deterministic. |

Decode-side steps are the measurement target; prompt-side chunking only matters
through the KV the decode steps attend to.

## Failure modes (all silent — check proactively)

1. **Config mismatch** (above). Mitigation: print/compare the `.meta.json`
   against the deployment serve args before believing any number.
2. **Preemption**: if the sequence is preempted, vLLM recomputes it via PREFILL
   kernels — logprobs silently change while the token-match check still passes
   (tokens are forced). bs=1 with the auto GPU budget makes this effectively
   impossible; never raise `--max-num-seqs` traffic alongside a replay.
3. **Speculative decoding**: changes the decode path entirely. Must be off
   (the offline `LLM` here never enables it — do not add it).
4. **Out-of-process engine**: without `VLLM_ENABLE_V1_MULTIPROCESSING=0` the
   LogitsProcessor runs in a child process and the RECORDS global stays empty.
   The script sets this itself; keep it if you refactor.
5. **Batch-faithfulness caveat**: the original service may have decoded at
   batch>1; batched kernels have different reduction orders. This replay is
   *decode-kernel-faithful*, not *batch-faithful*. Residual noise is expected to
   be far below the ~1-nat prefill error, but do not claim bit-exactness for
   replayed production trajectories — validate against any online logprobs the
   user has (the `.npz` `topk_*` fields make a top-20 overlap check possible).
6. **Tokenizer mismatch** (text inputs): the boundary between prompt ids and
   output ids must equal the serving-time boundary. Encode prompt and output
   separately (`add_special_tokens=False` on the output).

## vLLM API surface used (pinned to 0.26.x, V1 engine)

- `LLM(..., logits_processors=[ForcedTrajectoryLP])` — class, not instance.
- `SamplingParams(extra_args={"forced_ids": [...]})` — per-sequence state
  reaching the processor.
- `LogitsProcessor` protocol (`vllm.v1.sample.logits_processor`):
  `__init__(vllm_config, device, is_pin_memory)`, `is_argmax_invariant()`,
  `update_state(batch_update)` where `batch_update.added` entries are
  `(row_idx, params, prompt_ids, output_ids_ref)` tuples and `output_ids_ref`
  is a LIVE list (its `len()` is the current decode step), `batch_update.moved`
  entries are `(a, b, direction)`; `apply(logits)` returns modified logits.
- Masking uses finite `-1e30` (not `-inf`, which NaNs downstream softmaxes) and
  the processor reports `is_argmax_invariant() -> False`.

On a different vLLM version, verify each item above against the installed source
(`venv .../vllm/v1/sample/logits_processor/`) and adapt `scripts/replay.py`
before running; the numerical conclusion (prefill ≠ decode) is version-independent.

## Out of scope / future

- **Hidden states**: forward hooks do NOT fire during CUDA-graph replay (only at
  capture time, with dummy data) — capturing hidden states together with graphs
  silently records garbage. Protocol when needed: run A = graphs (faithful
  logprobs, this script), run B = `--enforce-eager` + hook, cross-check that A/B
  logprobs agree before trusting B's hidden states.
- **One-prefill comparison mode** (quantify how wrong a prefill rescoring would
  be for this exact trajectory): rerun the trajectory as a single prompt with
  `prompt_logprobs=1` and diff against the replay npz — see the parent repo's
  `step2_prefill_logprobs.py` / `step4_compare.py` for the validated pattern.
- **NPU/Ascend**: needs `torch.npu.synchronize()` in any hook, the Ascend
  quantization config, and vllm-ascend's graph mode is a different mechanism.
  Deliberately excluded from this version.
