# PR #1 评审整改与复验

这是原始 `REPORT.md` 之后的一次独立整改验收；原始 case、测量数组与验收报告不被改写。

## 逐条处理

| 总评编号 | 处理 | 结果与边界 |
|---:|---|---|
| 1 | 代码修复 | 非有限值以数值字段字符串保存；NaN rank=-1；保留异常证据但 numerical_match=false。 |
| 2 | 代码修复 | HTTP 路由与保留参数前端隔离；真实在线 worker 不启用 forcing，离线参数先校验。 |
| 3 | 代码修复 | 测试使用 tempfile/TMPDIR，不再依赖个人 Hermes 目录。 |
| 4 | 文档限定 | 旧 scripts/replay.py 未改；明确长 prompt partial-prefill 重复 step=0 与 UNIDIRECTIONAL bug，限定基础入口支持范围。 |
| 5 | 代码修复 | 记录 finish_requests 终态并清理采集索引；中止轨迹明确拒绝验收，未新增中止请求重放能力。 |
| 6 | 代码修复 | 受控布尔环境白名单记录与恢复；其它 VLLM_* 在两个模式均清除，API key 不进入 manifest。 |
| 7 | 代码修复 | set/frozenset 规范排序；list/tuple 原顺序不变。 |
| 8 | 代码修复 | 写入前预检目标目录；既有 case 仅允许语义匹配且不重写，冲突拒绝。 |
| 9 | 代码修复 | api.jsonl 可选，缺失统计 null；已在缺该文件的真实 case 副本上完成 GPU 重放。 |
| 10 | 代码修复 | GPU 初始化前核验记录的 config/tokenizer/index SHA-256 与 safetensors 名称/大小，不称完整权重内容校验。 |
| 11 | 代码修复 | block hashes 增量存储及终态清理；不逐 decode step 重写全部前缀。 |
| 12 | 保留方案 | 未采用 last-wins 去重。保留同步 sidecar、串行边界与重复步骤失败检查；优化建议需独立验证后再引入。 |
| 13 | 代码修复 | finally 清理自有进程组，含父进程正常退出后的 SIGTERM-resistant 子进程及信号竞争。 |
| 14 | 代码修复 | 真实输出与超时检查分开；子进程回归以 pipe 同步，不依赖 100ms 内解释器打印。 |
| 15 | 代码修复 | 文本读写显式 UTF-8，ASCII locale 下中文轨迹和 agent 文件回归。 |

共 15 项：13 项代码修复、1 项文档限定、1 项明确保留现有设计。后两项不冒称旧基础代码或替代去重方案已获修复/验证。

## 终审加固后的最终复验

独立终审发现并修正深层 JSON 解码/规范化的 RecursionError，以及布尔环境拼写不兼容；并补充有限、非负容差预检。

- 最终 CPU：**121 passed，16 subtests passed**，15 项第三方弃用告警。之前 92 项/88 项日志保留其历史含义。
- 独立复核通过：无阻断安全/逻辑问题；另一个只读 reviewer 重跑 121 项测试及 16 subtests，并验证深层 JSON、编码/数字溢出、120 个固定版本环境开关输入、I/O 传播和容差预检。它是独立本地复核，不是 GitHub hosted CI。
- JSON 容器最大深度 **64**（根容器计 1），迭代检查；解析、编码与日志规范化的 RecursionError 都明确返回并记录 HTTP 400。不吞掉磁盘/文件 I/O 错误。
- 真实服务拒绝 1200 层数组、600 层 diagnostic 及内部参数注入三次请求；EngineCore trace 未变化，随后健康检查与真实 agent 全流程仍完成。
- 容差 NaN/±inf/负数在比较器、driver 和 CLI 前置拒绝；CLI exit=2，无 GPU 初始化或证据写入。
- 最终正常 case：**3 个用户轮、2 次真实工具读取、5 请求、317 引擎输出 token**。四项验收全部 true，mean/max |Δ|=0；环境/记录的模型身份范围验证通过。
- 最终 APC 消融：mean |Δ|=0.00272267553941，max |Δ|=**0.134929776192 nats**；最后请求 max |Δ|=0.134929776192 nats。
- 分离的真实中止实验：收到模型 content 后关闭流连接，记录 **FINISHED_ABORTED/abort** 终态（引擎已接受 25 个 token），健康检查仍 200；组装明确拒绝 unsupported termination: abort。它不是完整会话重放的成功案例。
- 最终实验服务已关闭；原始 57 个证据文件 SHA-256 仍未变化。没有故障权重/NaN logits 注入，不把单元边界测试冒称真实故障模型复现。

最终主 case：`validation/runs/pr1-final-hardening-20261013/`（`deep-json-live-check.json`、完整 agent/online/replay/rebuild_prefix、`tests-final.log`）。
独立中止证据：`validation/runs/pr1-abort-hook-20261013/abort-live-check.json` 与真实引擎 trace。两者均不入 Git，未覆盖之前的 case。

| 请求 | prompt tokens | output tokens | 在线缓存命中 | 重放缓存命中 | 会话 max | APC 消融 max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 630 | 59 | 0 | 0 | 0 | 0 |
| 2 | 789 | 43 | 624 | 624 | 0 | 0.0999628305435 |
| 3 | 911 | 59 | 784 | 784 | 0 | 1.19202995847e-05 |
| 4 | 1076 | 44 | 896 | 896 | 0 | 0.0480382591486 |
| 5 | 1166 | 112 | 1072 | 1072 | 0 | 0.134929776192 |

## 首次整改的执行结果（保留历史数字）

- CPU 全套：**92 passed，16 subtests passed**；15 项第三方弃用告警，无失败。
- Agent 进程回归：17 项测试连续运行 20 轮均通过；ASCII locale + EncodingWarning 作为错误也通过。
- 真实服务拒绝 4 个非法请求（其它生成路由与内部字段注入）；EngineCore trace 无新增请求，之后 `/health` 仍为 200。
- 新采集：Qwen3-8B BF16、CUDA H20、vLLM 0.26.0、V1、TP1、同步串行、APC、FULL_AND_PIECEWISE CUDA graphs。
- OpenCode opencode v2.0.20：**3 轮真实用户交互、2 次真实工具读取、5 次模型请求、357 个引擎输出 token**。
- 正式会话重放：token/cache-path/configuration/numerical 四项均 true；`runtime_environment_match=true`、`model_identity_verified=true`；mean/max sampled-token |Δ| 均 **0**。
- APC 消融（同一 case）：token_match=true，缓存/配置/数值不匹配符合对照设计；mean |Δ| **0.00237087596936**，max |Δ| **0.132792234421 nats**；最后请求 max |Δ| **0.117207765579 nats**。
- 缺失 API 元数据的真实 case 副本：四项验收均 true，max |Δ| **0**，`online_api_requests=null`、`online_api_metadata_status=missing`。
- 重跑正式模式被前置拒绝；case 和两份 report 的 SHA-256 均不变。原始 stage1 的 57 个 engine/case/NPZ/日志证据文件也逐项 SHA-256 核验未变化。
- 所有实验服务均已停止；没有修改全局 agent 配置、没有合并 PR、没有发布版本 tag。

## 首次整改 case 的逐请求对照（357 tokens）

| 请求 | prompt tokens | output tokens | 在线命中 | 会话重放命中 | 会话 max | 重建前缀 max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 630 | 59 | 0 | 0 | 0 | 0 |
| 2 | 788 | 59 | 624 | 624 | 0 | 0.132792234421 |
| 3 | 926 | 59 | 784 | 784 | 0 | 2.2648673621e-06 |
| 4 | 1091 | 61 | 912 | 912 | 0 | 0.0133802853525 |
| 5 | 1198 | 119 | 1072 | 1072 | 0 | 0.117207765579 |

## 证据与复现

新的主 case：`validation/runs/pr1-review-fixes-20261013/`。
- `server.log`、`agent-collection.log`、`manifest.json`、`online/`、`agent/`、`case.json`
- `online-isolation-check.json`、`rerun-evidence-check.json`、`original-evidence-check.json`
- `replay/`、`rebuild_prefix/`、`session-replay.log`、`rebuild-prefix.log`
- `tests-pr1-final.log`（92 项）、`tests-pr1-review-fixes.log`（先前 88 项）和 `agent-regressions-repeat.log`、`agent-ascii-locale.log`

API 文件省略的副本：`validation/runs/pr1-review-fixes-api-optional-20261013/`，包含真实 manifest/engine 的逐字节副本、`provenance.json` 和独立的重放产物。没有伪造模型输出。

以上目录均不入 Git。运行环境与复用操作见 `README.md`；重做必须使用新目录，不能复制/修改 warm-cache 日志来冒充新采集。

非有限值边界使用明确的 CPU/Torch 单元夹具；中止路径除单元测试外还完成上述独立真实断流实验。本次真实模型采集未注入推理故障，不称真实故障检测或故障归因验收。仍不覆盖 Ascend、V2、并发、抢占/淘汰、MTP/speculation、KV 字节或 activation 保真。
