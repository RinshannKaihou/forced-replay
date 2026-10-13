# forced-replay 高级版：完整会话重放与本地验收

这是 `forced-replay` 的高级／完整会话模式，以及可复用的多请求验证实验室；原 `scripts/replay.py` 继续作为基础版单请求入口。
已实际打通：本地 vLLM API → OpenCode 三轮用户交互/真实工具读取 → 引擎 token/调度轨迹 → 同一离线引擎逐请求 forced replay → 在线参考校验。

## 已验证配置与边界

- `/root/llm-venv/bin/python`，vLLM **0.26.0**，CUDA H20，Qwen3-8B BF16。
- TP=1、单序列串行、APC 开启、chunked prefill=1024、max model len=32768。
- CUDA graphs 开启，实际 `FULL_AND_PIECEWISE`，FlashAttention 3；不是 eager 代替线上路径。
- 自定义 logits processor 使当前 vLLM 使用 **V1 model runner**。线上与离线均如此；不宣称覆盖默认 V2 runner。
- 普通自回归、文本/工具调用；此次关闭 thinking。未验证 Ascend、TP>1、并发批次、speculative/MTP、PD 分离、LoRA、多模态、缓存淘汰/抢占。
- 当前只核验逻辑计算区间、缓存命中长度和原轨迹 token 的 raw logprob。**没有逐字节核验 KV、采集原始 activation 或注入模型故障。**

## 复用：使用新的 run-dir，不覆盖原始证据

在项目根目录执行。先确认 GPU 和端口未被其他任务占用；服务只绑定 loopback。

终端 A（前台服务，验收结束用 Ctrl+C 停止）：

```bash
cd /workspace/inference_error_debugging_practice/prefill_decode_fidelity/forced-replay
/root/llm-venv/bin/python -m validation.lab serve \
  --run-dir validation/runs/example-001 --gpu 0 --port 18180
```

等到服务的 `/health` 返回 200。终端 B：

```bash
cd /workspace/inference_error_debugging_practice/prefill_decode_fidelity/forced-replay
/root/llm-venv/bin/python -m validation.collect_agent \
  --run-dir validation/runs/example-001/agent \
  --base-url http://127.0.0.1:18180/v1 --model replay-lab-qwen3-8b

/root/llm-venv/bin/python -m validation.lab replay \
  --run-dir validation/runs/example-001 --gpu 0

# 对照：每个请求重新 prefill 历史，不复用 APC。
/root/llm-venv/bin/python -m validation.lab replay \
  --run-dir validation/runs/example-001 --gpu 0 --mode rebuild-prefix
```

默认在线与离线各使用约 30% 的单卡显存预算；在本机允许服务空闲时同时加载离线引擎。显存不足时，完成采集后先停止本实验的服务，再离线运行。不要停其他用户的服务。

`serve` 拒绝已有的 `online/`、manifest 或 case，采集拒绝非空 agent 目录；重放在任何证据写入之前拒绝已有目标输出目录。两个模式可以重用完全匹配的 `case.json` 且不改它的字节；与 manifest/engine 不一致的既有 case 会被拒绝，不覆盖。重做请换 run-dir，或将旧结果明确归档后再执行；不要在 warm cache 的服务上直接把日志清空后冒充 cold-start case。

## 文件

- `manifest.json`：启动参数、包版本、GPU 信息、模型配置/tokenizer/index SHA-256、safetensors 文件名/大小库存，以及受控 VLLM 运行环境。重放在 GPU 加载前核验已记录文件指纹与库存；**不是完整权重内容校验和，同大小权重内容替换仍可能漏检。**
- `online/api.jsonl`：原始/实际 API 请求、HTTP 状态、未经改写的响应流；可选诊断文件，缺失时请求统计为 `null`，不阻止重放。
- `online/engine.jsonl`：模板处理后的真实 prompt IDs、包含停止/控制 token 的引擎输出 IDs、raw logprobs、实际调度区间、缓存命中和有效配置。新增 block hash 以 `block_hash_start` / `new_block_hashes` / `block_hash_reset` 增量保存，不在每个 decode step 重写整个前缀。`scheduler.finish_requests` 的中止/错误也写入 `finish` 事件并释放采集器索引；这类轨迹会明确报 unsupported termination，而不是不明原因的 unfinished request。
- JSON 诊断数值中的 `-inf` / `inf` / NaN 分别保存为字符串 `"-inf"` / `"inf"` / `"nan"`，有限值保持数值类型；只在已知数值字段用 `float(...)` 还原。无效分布的 rank 为 -1。非有限 selected-token logprob 保留证据，但不会通过 `numerical_match`；NPZ 中仍是浮点 IEEE 数值。
- `agent/turn-*.jsonl`：原始 OpenCode CLI 事件；`turn-*.export.json`：真实持久化会话导出；`session.json`：三轮用户交互和执行成功的工具调用审计。
- `case.json`：组装后的按请求输入/输出轨迹。
- `replay/report.json`：正式验收；`replay/measurements.npz`：逐 token forcing 前的 logprob/rank/entropy/top-k。
- `rebuild_prefix/report.json`：主动禁用 APC 的对照，不要求它通过保真校验。

## 在线输入隔离

实验室仅支持 `POST /v1/chat/completions` 文本生成；GET/HEAD/OPTIONS 原样透传（含 health/model 查询），其它修改状态的 HTTP 路由及 WebSocket 不进入引擎。其它生成接口、`n>1`、prompt logprobs、结构化生成/受约束 tool_choice 和多模态输入会返回明确的 HTTP 400（WebSocket 1008），不应拿该实验服务代替通用生产 API。正常 `tool_choice=auto/none` 工具调用仍受支持。

JSON 容器最大嵌套深度为 64（根容器算 1）；超限/递归错误在解析、编码和记录之前返回 HTTP 400，记录拒绝原因，不转交引擎。正常工具 schema 不受影响。

客户端的 `vllm_xargs.forced_ids` / `trace_id` 是保留字段，HTTP 400 拒绝；在线真实 worker 即使遇到这类参数也不会 forcing。离线仅由 driver 的 `FORCED_REPLAY_MODE=replay` 开启 forcing，参数先在前端校验。采集模式是 identity；原始与注入 trace ID 后的有效请求分别保存。

## 校验规则

正式重放必须同时通过 `token_match`、`cache_path_match`、`numerical_match`、`configuration_match`；默认 sampled-token logprob 容差 1e-5 nats。容差必须有限且非负，CLI 在读取 case/初始化 GPU 前拒绝 NaN、inf 和负数（exit=2）。配置核验覆盖 model/cache/parallel/scheduler/compilation/structured-output/kernel 配置，以及 `runtime_environment_match`，忽略编译耗时这一非计算字段。它不等同于所有硬件、内存字节、调度外部条件完全相同。

环境策略 `controlled-vllm-whitelist-v1`：serve 和 replay 都清除继承的所有 `VLLM_*`，只恢复 `validation.lab.RUNTIME_ENVIRONMENT_KEYS` 列出的布尔开关，另显式设置禁用插件/usage stats。白名单值仅接受原样的 `"0"` / `"1"` 或未设置；不把 `true/false` 文本猜测转换，因为固定版本的部分开关用 `int()` 解析。白名单涵盖 batch-invariant、runner、Triton TD、FlashInfer sampler、DeepGEMM/E8M0/TMA、AWQ、Marlin atomic-add 和 breakable-cudagraph；未设置的值以 `null` 记录并在重放时保持未设置。**非白名单设置（包括 `VLLM_ATTENTION_BACKEND` 和任何鉴权/路径变量）在这套受控实验室中不继承**，采用固定版本默认值或显式 engine kwargs，不宣称复刻任意生产 shell。日志不能保存 API key。旧 manifest 没有环境证据时标记 `unknown`，仍产生诊断报告，但正式模式不通过配置验收。

离线每轮提交记录下来的完整 prompt IDs，不从客户端显示文本拼接历史；只强制模型输出，不重新执行工具。`stop`/EOS 在离线显式改为按已记录长度结束，避免少掉最后一个引擎 token；原终止原因仍保留在 case 中。

### 实际遇到并覆盖的边界

- OpenCode v2 的 `debug config` 会尝试管理后台服务，隔离 XDG 目录不隔离它的固定端口。采集仅使用 `run --standalone` 和 `session export --standalone`，不触碰用户正在运行的 OpenCode 服务。
- v2 CLI 有时在最后一个 `step_finish` 事件到达前退出。不能据此伪造事件，也不能凭退出码放行：要对照真实 session export 中同一会话、同一最新 assistant message 的 `finish=stop` 和持久化完成时间。
- vLLM 部分 prefill chunk 也会执行 dummy sampling，随后丢弃该结果。`ReplayProcessor` 用同步调度 sidecar 排除这些无效测量；否则会重复记录 step=0。本实验的 rebuild-prefix 对照实际触发了这个边界。
- 当前 `scripts/replay.py` 仍是旧单请求实现：长 prompt 超过原单次 prefill budget 会产生重复 step=0 并报 record count mismatch；UNIDIRECTIONAL row move 仍未正确处理。其可用范围限定 TP1、固定单请求 slot、无 partial-prefill/抢占；不应改变原 chunk size 来绕过保真要求。长 prompt 使用已采集 case 的高级入口。
- 保留同步调度 sidecar、串行限制与重复 step 的严格检查。本次没有采用“每个 step 保留最后一次测量”的设计建议：不能在未独立验证重复来源和计算路径的情况下把证据折叠掉。该取舍不是宣称此优化永远不可行。
- 采集器在线强制启用 `logprobs=20` 与 token-ID 返回，原请求和有效请求分别保存。这是诊断验证配置，不是已测量零开销的生产监控插件。
- OpenCode 的 HOME/XDG/config/auth 路径都隔离在 agent run-dir；仅配置本地 provider，模型工具仅允许读取两份测试文件。随机 marker 用于核验工具确实读到了文件，不用模型的自述代替工具执行记录。

## 测试与验收证据

```bash
VLLM_PLUGINS='' /root/llm-venv/bin/python -m pytest validation -q
```

本次 GPU 实测与误差对照见 `REPORT.md`。运行产物在 `validation/runs/qwen3-8b-stage1/`，该目录被 Git 忽略，避免把流式输出与实验数组误提交到代码库。真实生产轨迹可能含敏感 prompt/工具结果，接入生产前需要独立的访问控制和保留策略。
