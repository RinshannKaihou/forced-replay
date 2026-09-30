# Interpreting replay results

All fields are the RAW full-vocab distribution the runtime decode kernels
produced, BEFORE temperature / top-p / the forcing mask. The online sampler saw
a temperature- and top-p-modified version of this same distribution.

## Headline stats

- `mean / min forced_logprob`, `n(<-5)`: overall how improbable the actual
  trajectory was under the runtime kernels.
- `top1-match` (= fraction of steps with `forced_rank == 0`): how often the model
  was actually rooting for the token that got emitted.
- `min entropy` and its step: confidence collapse location.

## Reading a repetition loop that starts at step r

Look at the window around r and inside the loop:

| Pattern in the replay data | Reading |
|---|---|
| `forced_logprob` of the repeated token climbs toward ~0 and `entropy` collapses toward ~0 as the loop proceeds | the model STATE collapsed into the loop — a genuine model pathology (attention trap / KV degradation). Not a sampling fluke. |
| repeated token keeps LOW logprob / high `forced_rank`, yet it kept being sampled | sampling-side cause (temperature/top_p/seed) — the model wanted something else every step. |
| `entropy` stays high, `top1_id` differs from `forced_id`, and the `top1_id` sequence itself is coherent | the model repeatedly proposed an "escape" it never took; compare `top1_logprob` vs `forced_logprob` to see how expensive each forced step was. |
| logprobs fine until step k then suddenly degrade | something changed AT k: look at the token at k-1/k (weird unicode, truncation boundary, special token) — often the actual trigger precedes the visible symptom r. |

Report the pattern, the worst steps (token ids + logprobs), and the step where
entropy/logprob inflects. Detokenize the flagged ids for the user.

## General ill trajectories

- Isolated low-logprob steps (`forced_logprob < -5`, rank high): single-token
  anomalies — check the surrounding top-k for what the model expected.
- Sustained moderate degradation (logprob drifting down over many steps):
  gradual divergence, often triggered much earlier — scan backwards for the
  inflection.
- `top1-match` dropping over time: the model increasingly disagreeing with the
  emitted stream.

## Caveats in the report

- State whether the replay is batch-faithful (only if the original decoded at
  bs=1 — usually unknown) or only decode-kernel-faithful.
- List any engine args you ASSUMED (from `meta.json`) — a wrong assumption can
  manufacture or mask a finding.
- Do not compare absolute logprobs across models or quantizations.
- The replay proves what the model believed at runtime; it does not by itself
  assign fault (weights vs kernels vs sampling). Pair the numbers with the
  deployment details before concluding.
