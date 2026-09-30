# forced-replay (Claude Code skill)

给定一个模型和它的一条已保存输出轨迹，在 runtime decode 内核下强制重放该轨迹，
拿到**逐 token 的真实 logprobs / rank / entropy / top-k**——而不是用一次 prefill
重打分（后者与 runtime 有最多 ~1 nat 的静默偏差）。

## 安装（团队仓库）

把 `forced-replay/` 整个目录放进团队仓库的 `.claude/skills/`：

```
your-repo/.claude/skills/forced-replay/{SKILL.md, scripts/replay.py, references/}
```

克隆该仓库的同学即可在 Claude Code 里直接说“用 forced-replay 重放这条轨迹”。

## 使用

对 Claude 说（示例）：

> 模型在 /workspace/models/Qwen3-30B-A3B，轨迹在 ill_traj.json，线上是 fp8 部署。
> force-replay 一下，拿真实 logprobs，看看第 97 步附近怎么回事。

Claude 会按 SKILL.md 收集部署配置 → 映射引擎参数 → 选卡 → 运行 → 校验 → 出诊断。

手动直跑（不经 Claude）：

```bash
python scripts/replay.py --model /path/to/model --traj traj.json --quantization fp8
```

产物：`<traj>_replay.npz`（逐步测量）+ `<traj>_replay.meta.json`（实际引擎配置，审计用）。

## 要求

- vLLM 0.26.x（LogitsProcessor 协议强绑定；其他版本需按 `references/fidelity.md` 适配）
- CUDA GPU；共享卡自动选最空闲的
- 轨迹格式：`{"prompt_ids": [...], "output_ids": [...]}`（只有文本时让 Claude 先分词转 ids）

验证背景：本目录上级的实验（step1–4）已证明 forced-replay 与在线 decode logprobs
逐位一致，one-prefill 重打分偏差 p95≈0.21 / max≈1.03 nats。
