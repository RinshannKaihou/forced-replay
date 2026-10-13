# Changelog

## full-session-v1 — 高级／完整会话重放模式

### Added

- 在同一个离线引擎内按原始请求顺序重放完整 agent 会话，保留 APC，并提交每轮真实的完整 prompt token IDs。
- 服务端 API 与引擎事件采集：原始／有效请求、接受的输出 token、在线 raw logprob、有效配置、缓存命中、计算区间。
- forcing 前的 logprob、rank、entropy、top-k 测量，以及 token／缓存路径／有效配置／数值参考四项验收。
- 隔离配置的真实 OpenCode 三轮交互与受控工具读取；离线不重新执行工具。
- 关闭 APC、每轮重建历史前缀的消融入口及实际结果报告。
- 对 partial-prefill dummy sampling、OpenCode v2 完成事件缺失、配置序列化及继承鉴权环境的回归检查。

### Fixed

- 显式移除本地服务子进程继承的 `VLLM_API_KEY`，避免与隔离 agent 的占位鉴权冲突；保持父进程环境不变，并覆盖真实鉴权中间件的 HTTP 回归。

### Evidence

- Qwen3-8B BF16、CUDA H20、vLLM 0.26.0、V1 runner、TP1、同步串行、CUDA graphs 开启。
- 3 轮用户交互、2 次真实工具读取、5 次模型请求、328 个接受的输出 token。
- 会话重放的 sampled-token raw logprob 最大误差为 0；禁用 APC 的消融最大误差为 0.184628 nats。
- 详细证据与复用命令：`validation/REPORT.md`、`validation/README.md`。原始运行数据保留在被 Git 忽略的 `validation/runs/`。

### Compatibility and scope

- 基础版 `scripts/replay.py` 保持不变。
- “完整”表示完整请求序列，不代表所有部署特性均已实现或 KV／activation 字节保真。
- 尚未验收 Ascend、V2 runner、并发批次、speculative/MTP、缓存淘汰／抢占、PD 分离、LoRA、多模态和 thinking 模式。
- 不把强制输出匹配视作数值证据，也不把单条轨迹的零误差外推为任意生产故障可复现。
