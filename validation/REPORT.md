# 第一阶段实际验收报告

生成时间（UTC）：2026-10-13T08:50:43.409836+00:00

## 实测环境与采集

- 模型：`/workspace/models/Qwen3-8B`，BF16；CUDA H20；vLLM 0.26.0；torch 2.11.0；transformers 5.17.0。
- Agent：opencode v2.0.20；本地唯一配置 provider；隔离 HOME/XDG/config/auth；仅允许读取两份受控文件。
- 会话：`ses_ecbb46145ffe3SouOB3EF446Gr`；**3 轮真实用户交互、2 次成功的真实工具读取**。
- 实际模型请求：**5**；生成 token：**328**，包含引擎控制/停止 token。
- CUDA graphs 实际启用（FULL_AND_PIECEWISE）；线上/离线均为 V1 runner、TP1、同步串行、chunked prefill=1024、APC。
- 两个前置工具读取分别触发一次生成 tool call 和一次基于工具结果的回答，第三轮利用历史工具结果直接回答。工具没有在离线重放中再次运行。
- 测试对象是受控输入的真实模型输出，**未注入故障，不宣称自动检测器已发现真实推理故障或任务语义完全正确**。

## 正式会话重放结果

- `token_match = true`
- `cache_path_match = true`
- `configuration_match = true`；模型/cache/parallel/scheduler/compilation/structured-output/kernel 配置一致，排除编译耗时。
- `numerical_match = true`；默认容差 1e-05 nats。
- 原轨迹 token 的 raw logprob：mean |Δ| = **0**，max |Δ| = **0**。

## 对照：每轮重新 prefill 历史，不复用 APC

- 输出 token 仍被正确强制，`token_match = true`，但缓存/计算区间不匹配。
- 唯一被核验的有效引擎配置差异：`enable_prefix_caching: true → false`。
- 全部 token：mean |Δ| = **0.00187040344**，max |Δ| = **0.184628367 nats**。
- 最后一轮回答：max |Δ| = **0.0530662835 nats**。
- 这证明本 case 的历史前缀重建会影响被测概率；没有直接保存原始 K/V 张量，因此不把它扩大解释为逐层 KV 误差或普遍误差上界。

| 请求（按引擎顺序） | prompt token | output token | 在线缓存命中 | 会话重放命中 | 会话重放最大误差 | 重建前缀最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 612 | 53 | 0 | 0 | 0 | 0 |
| 2 | 760 | 59 | 608 | 608 | 0 | 0.184628367 |
| 3 | 892 | 53 | 752 | 752 | 0 | 1.43050329e-06 |
| 4 | 1042 | 57 | 880 | 880 | 0 | 0.00374046573 |
| 5 | 1145 | 106 | 1024 | 1024 | 0 | 0.0530662835 |

## 代码验证及实际修正

- CPU 测试：**31 passed，16 subtests passed**；15 项第三方弃用告警，没有测试失败。提交前回归详见运行产物 `tests-post-review.log`；初次 GPU 验收时的测试日志仍保留为 `tests.log`。
- 安全静态检查：无 `shell=True`、`os.system`、`eval`、`pickle.load(s)`；子进程使用参数列表，原始运行数据不进入版本库。
- 独立复核发现继承 `VLLM_API_KEY` 会使本地采集遭遇 401：现从复制的子进程环境中移除该变量，不修改父环境；使用已安装 vLLM 的真实鉴权中间件覆盖失败到通过的 HTTP 回归。
- OpenCode v2 CLI 的最终完成事件存在缺失：通过真实 session export 的会话 ID、最新 assistant message ID、finish 与完成时间校验，不补造 CLI 事件。
- 关闭 APC 的对照真实触发部分 prefill 的 dummy sampling：之前导致 step=0 重复记录，现通过同步调度 sidecar 排除，并覆盖 CPU 回归与 GPU 再运行。
- 完整调度区间包括最后输出 token 尚未 forward 的边界；只按实际引擎事件采集，不把显示文本或客户端反序列化结果当作 token 真值。

## 证据路径与边界

原始和处理后的证据均在 `validation/runs/qwen3-8b-stage1/`：

- `manifest.json`、`online/api.jsonl`、`online/engine.jsonl`
- `agent/turn-*.jsonl`、`agent/turn-*.export.json`、`agent/session.json`
- `case.json`、`replay/report.json`、`replay/measurements.npz`
- `rebuild_prefix/report.json`、`rebuild_prefix/discarded_prefill.jsonl`
- `session-replay.log`、`rebuild-prefix.log`、`tests.log`、`tests-post-review.log`

本报告证明的是该受控配置和 case 上的 token、逻辑计算/缓存路径、有效配置、sampled-token raw logprob 保真。
**不证明 KV 字节一致、全词表分布一致、activation 保真，也不覆盖 Ascend、V2 runner、并发批次、speculation/MTP、缓存淘汰/抢占、PD 分离、LoRA、多模态或 thinking 模式。**

原 `scripts/replay.py` 单请求代码未被修改；多轮验证入口与复用命令见 `validation/README.md`。

后续 PR #1 评审整改（最终 121 项测试及独立真实 GPU case）的结果见 `PR1_REVIEW_FIXES.md`；本报告与上面的原始数字保持历史含义，不改写为新 case 的结果。
