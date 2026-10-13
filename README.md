# forced-replay：基础版与高级／完整会话版

在匹配的运行条件下，沿已保存的 LLM 输出轨迹进行强制重放，测量 **forcing 前的逐 token logprob、rank、entropy 与 top-k**。不仅固定输出文本，还核验实际计算路径与在线数值参考。

## 两种模式

| 模式 | 输入与重放范围 | 历史状态处理 | 入口 |
|---|---|---|---|
| **基础版：单请求重放** | 一次请求的 `prompt_ids` / `output_ids` | 对原 prompt 做 prefill，再强制生成原 output；关闭 APC | `scripts/replay.py` |
| **高级版：完整会话重放** | 按顺序记录的全部模型请求、实际 token IDs、引擎配置与在线参考 | 同一个离线引擎逐请求推进，保持 APC，复现并校验缓存／计算边界 | `python -m validation.lab replay` |

**“完整”指完整请求序列，不是历史 KV 永不重算，也不是所有硬件与部署特性均已支持。** 高级版不重新运行 agent 或执行工具；每轮提交记录下来的完整 prompt，并沿原 output 强制生成，让引擎复用实际可匹配的前缀。缓存不命中的部分仍按正常路径 prefill。

当前高级版验证范围：**CUDA、vLLM 0.26.0、V1 model runner、TP1、同步串行、普通自回归、CUDA graphs 与 APC**。Ascend、V2 runner、并发批次、speculative/MTP、缓存淘汰／抢占、PD 分离、LoRA、多模态、thinking 模式以及 KV／activation 原始张量保真尚未验收。不要将这次零误差外推到任意生产部署。

## 高级版：采集、重放和消融

完整操作及边界说明见 [`validation/README.md`](validation/README.md)，真实验收结果见 [`validation/REPORT.md`](validation/REPORT.md)。在项目根目录执行，并为每次实验使用新的 run-dir。

```bash
# 终端 A：本地服务；只监听 loopback。
/root/llm-venv/bin/python -m validation.lab serve \
  --run-dir validation/runs/example-001 --gpu 0 --port 18180

# 终端 B：真实 OpenCode 三轮用户交互和工具读取。
/root/llm-venv/bin/python -m validation.collect_agent \
  --run-dir validation/runs/example-001/agent

# 高级／完整会话重放：一个引擎、按原请求顺序、保留 APC。
/root/llm-venv/bin/python -m validation.lab replay \
  --run-dir validation/runs/example-001 --gpu 0

# 消融：每轮重建历史前缀，不复用 APC。
/root/llm-venv/bin/python -m validation.lab replay \
  --run-dir validation/runs/example-001 --gpu 0 --mode rebuild-prefix
```

正式模式必须同时通过 **token、缓存／计算区间、有效配置、原轨迹 token 的 raw logprob** 四项校验。输出 token 一致只是强制生成成功，不足以证明数值保真。

已实测 Qwen3-8B BF16 + OpenCode：3 轮用户交互、2 次真实工具读取、5 次模型请求、328 个引擎输出 token。完整会话重放 sampled-token logprob 最大误差为 **0**；禁用 APC 的消融最大误差为 **0.184628 nats**。这些是单条受控轨迹的结果，不是异常检测准确率或普遍误差上界。

## 基础版：单请求使用

轨迹格式：

```json
{"prompt_ids": [1, 2], "output_ids": [3, 4]}
```

上面的 IDs 仅示意格式，不是可直接用于任何模型的有效轨迹。实际 IDs 必须来自所用模型与原请求。

```bash
python scripts/replay.py --model /path/to/model --traj traj.json --quantization fp8
```

产物：`<traj>_replay.npz`（逐步测量）和 `<traj>_replay.meta.json`（引擎配置）。模型、quantization、dtype、TP、图执行与 chunked-prefill 等参数必须与原部署匹配。

基础版要求 vLLM 0.26.x、CUDA GPU；LogitsProcessor 协议强绑定，其他版本需先按 [`references/fidelity.md`](references/fidelity.md) 适配。共享 GPU 可自动选卡。原基础版脚本未随本次高级版提交修改，不应假设高级版的 partial-prefill 测量防护也已应用于旧脚本。

## 作为 Claude Code skill 安装

把源码目录放进团队仓库的 `.claude/skills/`：

```text
your-repo/.claude/skills/forced-replay/
├── SKILL.md
├── scripts/
├── references/
└── validation/          # 高级模式、测试与文档；不含 runs/
```

对 Claude 说明模型路径、原部署配置和轨迹位置；多轮会话应选择高级模式，单条请求可选择基础模式。具体输入收集、版本适配与诊断规则见 `SKILL.md`。

## Git 管理

- **入 Git**：基础／高级模式源码、测试、说明、验收摘要。
- **不入 Git**：`validation/runs/` 下的原始 API／agent 轨迹、日志、缓存相关执行记录和测量数组；该目录已忽略。
- 原始数据保留在本地，不能把被忽略的运行产物误当成可随源码克隆的测试证据；新增实验请独立保存其 case bundle。
- 此次高级／完整会话版本在功能分支管理，基础版入口保持兼容；不会自动推送、创建远端发布或合并远端主分支。

CPU 回归：

```bash
VLLM_PLUGINS='' /root/llm-venv/bin/python -m pytest validation -q
```

背景实验（本目录上级 step1–4）在 Qwen3-30B-A3B FP8 上观察到一次 prefill 重打分与在线 decode 的偏差 p95≈0.21、max≈1.03 nats；详见 `references/fidelity.md`。它与本次会话级 APC 消融是不同实验，不应混用。
