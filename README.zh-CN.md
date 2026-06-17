# EvoInfer Dream MCP

[English](README.md) | [中文](README.zh-CN.md)

EvoInfer Dream 是一个面向推理优化 agent 的开盒 MCP 记忆管理器。它给
code agent 提供一组可调用工具，用来检索、写入、验证和维护带证据的经验记忆。
这些记忆来自推理优化任务和环境部署调试任务。

EvoInfer Dream 不是聊天界面，也不替代 Claude Code、Codex 这类 agent
运行时。它的定位是 agent 旁边的记忆和控制组件：当 agent 需要优化经验、
部署调试经验、从实验产物里抽取记忆，或者记录后续反馈时，通过 MCP 调用
EvoInfer Dream。

## 本地共享记忆库

EvoInfer 不应该让每个 session 都冷启动。包内自带一份可版本化的 seed memory
store：

```text
src/evoinfer_mcp/dream/seed_memories.json
```

当用户启动 EvoInfer 管理的 session，或者生成 MCP 配置时，EvoInfer 会把缺失的
seed memories 合并到用户本地持久化记忆库：

```text
~/.evoinfer/dream-share/dream/memories.json
```

这个合并是幂等的，不会覆盖本地已有同 ID 记忆。因此用户后续产生的反馈计数、
promoted memories 和手动编辑仍然以本地库为准。通过 `evoinfer` 启动的
Claude/Codex session 默认共享这一个本地库；只有显式传 `--share-dir` 时才会使用
其他库。

如果某个客户端绕过 CLI 直接启动 MCP server，第一次 `dream_search_memories`
调用也会把缺失的包内 seed memories lazy 合并到共享库。这样即使用户没有先执行
setup 命令，agent 的首次检索也不会冷启动；后续包升级带来的新内置经验也会自动
补进本地库，但不会覆盖用户已有记忆。

当前包内经验包括 CUDA softmax/RMSNorm 基础算子经验、FlashInfer attention
baseline 经验、FLA route-policy campaign 经验、limx 上 FLA parallel attention
相对 PyTorch SDPA 的边界/负迁移经验，limx 上 vLLM、SGLang、TensorRT 的
backend attention 路线验证经验，以及 FlashInfer JIT、TensorRT NVIDIA PyPI
安装、语音 STT CPU fallback、vLLM/SGLang/TensorRT 大 wheel 下载卡死等环境调试经验。

也可以手动初始化或查看本地库：

```bash
evoinfer memory-seed --json
evoinfer memory-export --json
```

## 记忆内容

EvoInfer 目前管理两类记忆：

- 优化记忆：硬件、模型或算子、推理后端、精度、workload、baseline 指标、
  优化后指标、正确性证据、profiler 或源码证据、迁移约束。
- 环境调试记忆：机器环境、组件、症状、根因、修复步骤、验证证据，以及可复用的
  部署说明。

成功经验只被当作带证据的假设，而不是无条件规则。失败经验或负迁移经验会作为
约束，避免 agent 把某个优化错误迁移到不兼容的 workload、backend、dtype 或
算子语义上。

## 安装

```bash
git clone https://github.com/binichallein/DreamAgent.git
cd DreamAgent
uv tool install --force --editable .
evoinfer doctor --json
evoinfer lifecycle-smoke --json
```

`doctor` 会检查包导入、记忆目录写权限、MCP 工具列表和 stdio 启动能力。
`lifecycle-smoke` 会跑一个本地端到端记忆流程：协议读取、检索、stuck 检索、
候选记忆暂存、产物抽取、写入、提升、反馈、列表查询和协议验证。

显式初始化本地共享记忆库：

```bash
evoinfer memory-seed --json
```

安装完成后，用户侧命令统一使用裸命令：

```bash
evoinfer <command> [args...]
```

## Claude Code 接入

生成 Claude Code 的 MCP 注册命令：

```bash
evoinfer mcp-config \
  --client claude \
  --format claude-add-json \
  --share-dir ~/.evoinfer/dream-share \
  --scope user
```

执行打印出来的命令。它大致长这样：

```bash
claude mcp add-json evoinfer-dream '{"type":"stdio","command":"/abs/path/to/python","args":["-m","evoinfer_mcp.dream.mcp_server"],"env":{"EVOINFER_SHARE_DIR":"/home/user/.evoinfer/dream-share"}}' --scope user
```

验证注册结果：

```bash
claude mcp list
claude mcp get evoinfer-dream
claude
```

进入 Claude Code 后运行：

```text
/mcp
```

应该能看到 `evoinfer-dream` 作为已连接的 stdio MCP server。

## Codex 接入

生成 Codex 的 MCP 配置块：

```bash
evoinfer mcp-config \
  --client codex \
  --format codex-toml \
  --share-dir ~/.evoinfer/dream-share
```

简写形式：

```bash
evoinfer mcp-config --client codex --format codex-toml
```

把打印出来的 `[mcp_servers.evoinfer-dream]` 配置块加入 Codex 的 MCP 配置。

## 通用 MCP 接入

任何支持 stdio MCP 的客户端都可以启动：

```bash
python -m evoinfer_mcp.dream.mcp_server
```

推荐设置：

```bash
EVOINFER_SHARE_DIR=/abs/path/to/evoinfer-share
```

持久化记忆库会写到：

```text
$EVOINFER_SHARE_DIR/dream/memories.json
```

## 强制 Dream Session

对于验证实验或高风险优化任务，建议使用独立的强制 Dream session，而不是只做
全局可选 MCP 注册。该模式会创建一个隔离 session 目录，包含：

- `mcp.json`：当前 session 的 stdio MCP 配置。
- `mcp_calls.jsonl`：当前 session 的 Dream 工具调用审计日志。
- `work/AGENTS.md` 和 `work/CLAUDE.md`：强制 Dream 协议指令。
- `share/`：当前 session 使用的 Dream 记忆库。

创建 bundle：

```bash
evoinfer force-session \
  --session-dir /tmp/evoinfer-dream-session \
  --share-dir ~/.evoinfer/dream-share \
  --workdir /tmp/evoinfer-dream-session/work
```

打印出来的命令可以在这个强制 Dream session 中启动 Claude Code、Codex 或
Kimi CLI。MCP server 还会收到：

```bash
EVOINFER_DREAM_MANDATORY=1
EVOINFER_DREAM_SESSION_ID=<session-name>
EVOINFER_MCP_CALL_LOG=<session-dir>/mcp_calls.jsonl
```

需要明确的是，MCP server 不能主动控制 agent，也不能凭空发起工具调用。
这里的强制性来自 session 范围：agent 启动时就带着该 session 的配置和指令，
同时调用日志会让漏掉 Dream 协议步骤的问题变得可审计。

## 带 Hook 的 Agent Session

日常使用时，可以通过 EvoInfer 启动 Claude Code 或 Codex，让 Dream 不只在开头
检查一次，而是在运行过程中周期性 checkpoint：

```bash
evoinfer --client codex --hook-every-steps 10
evoinfer --client claude --hook-every-steps 10
```

直接运行 `evoinfer` 会打开一个简短的终端选择界面，选择客户端和 checkpoint
步数。

Hook 模式会写入一个专用 session bundle，并启动所选 agent：

- Claude Code 使用 `.claude/settings.local.json`，配置 `SessionStart`、
  `PostToolBatch` 和 `Stop` hooks。
- Codex 使用 `.codex/hooks.json`，配置 `SessionStart`、`PostToolUse` 和
  `Stop` hooks，并在该生成 session 中用 `--dangerously-bypass-hook-trust`
  启动。
- 每隔 N 个工具 checkpoint，hook 会运行
  `evoinfer_mcp.hooks.dream_checkpoint`，搜索 Dream，写入 `dream_context.md`，
  并把 checkpoint 结果作为 hook context 注入给 agent。

Kimi CLI 目前还没有接入 hook 模式。当前稳定的 hook-backed 路径是 Claude Code
和 Codex。

## 给 Agent 的软协议

MCP 工具只有在 agent 决定调用时才会执行。为了让 agent 更稳定地主动使用
EvoInfer，可以把下面的指令加入 system prompt 或项目指令：

```markdown
You are an EvoInfer agent. For inference optimization or environment-debug tasks:

1. At task start, call `dream_get_agent_protocol`, then call
   `dream_search_memories` with the current hardware, backend, dtype, workload,
   model/operator, and failure or optimization goal.
2. Treat successful memories as evidence-backed hypotheses. Treat negative
   memories as transfer constraints.
3. Do not transfer a memory across mismatched dtype, workload, backend, or
   operator semantics without a new benchmark and correctness check.
4. If stuck, changing route, or choosing between CUDA/Triton/library/backend
   approaches, call `dream_search_memories` again before continuing.
5. If the task produces benchmark, correctness, profiler, verifier, source, or
   environment artifacts, call `dream_extract_and_write_memories` before the
   final report.
6. In the final report, mention the Dream memory IDs retrieved or written and
   the artifact paths that justify them.
```

期望的 agent 工作流：

1. 调用 `dream_get_agent_protocol`。
2. 在本地自由探索前调用 `dream_search_memories`。
3. 执行真实 benchmark、正确性验证、profiling 和环境检查。
4. 在卡住或切换优化路线时再次调用 `dream_search_memories`。
5. 写出 `benchmark_raw.json`、`correctness_raw.json`、
   `profiler_summary.json`、`environment.json`、`verifier_result.json` 等产物。
6. 调用 `dream_extract_and_write_memories`。
7. 后续证据充足时调用 `dream_promote_memory`、`dream_reject_memory` 或
   `dream_record_feedback`。

自动写入会先创建候选记忆。正式提升为可复用记忆需要显式证据审查。

## 主要 MCP 工具

- `dream_get_agent_protocol`：返回当前 EvoInfer 协议。
- `dream_search_memories`：混合结构化和词法检索，带 schema filter、证据感知打分、
  迁移检查和可选 embedding 打分。
- `dream_stage_memory_candidate`：把候选记忆写入工作目录，不修改持久化记忆库。
- `dream_extract_memory_candidates`：从 campaign artifacts 中抽取候选记忆。
- `dream_extract_and_write_memories`：基于产物门控的抽取和候选写入。
- `dream_promote_memory`：证据审查后提升候选记忆。
- `dream_reject_memory`：拒绝候选记忆，或记录负迁移约束。
- `dream_record_feedback`：根据后续复用情况更新 chosen/useful 计数。
- `dream_list_memories`、`dream_get_memory`、`dream_export_memory_store`、
  `dream_import_memory_store`：记忆管理工具。

## 基于实验产物的记忆写入

Dream 记忆应该优先从证据产物中抽取，而不是直接从聊天总结中写入。抽取器会在任务
工作目录中寻找这些标准文件：

- `benchmark_raw.json`
- `correctness_raw.json`
- `profiler_summary.json`
- `environment.json`
- `verifier_result.json`
- `dream_write_candidates.json`

Verifier 会检查候选记忆是否有足够证据。对于优化记忆，benchmark 和正确性证据是
必要条件；高质量提升还需要 profiler summary 或源码级瓶颈证据。对于环境调试记忆，
需要症状、根因、解决步骤和验证证据。

## 可选 CPU Embedding

默认情况下，检索不依赖 embedding 模型。它会使用结构化字段、标签、workload hint、
词法匹配、负记忆和证据感知 reranking。

如果希望增强语义召回，可以安装可选 embedding 依赖，并在生成 MCP 配置时启用本地
CPU 后端：

```bash
uv tool install --force --editable ".[embedding]"
evoinfer mcp-config \
  --client codex \
  --format codex-toml \
  --share-dir ~/.evoinfer/dream-share \
  --enable-embedding \
  --embedding-model BAAI/bge-small-zh-v1.5
```

生成的 MCP server 环境会包含：

```bash
EVOINFER_EMBEDDING_BACKEND=local
EVOINFER_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EVOINFER_EMBEDDING_DEVICE=cpu
```

agent 客户端不需要原生支持 embedding。它只需要调用 EvoInfer Dream MCP 工具；
embedding 模型由 MCP server 在 CPU 上配置和运行。Embedding 分数只是检索信号之一，
不是最终裁决依据。Schema filter、workload 相似度、正确性证据、profiler 证据和
负迁移约束仍然会参与最终门控。

## 运行检查

```bash
evoinfer memory-seed --json
evoinfer doctor --json
evoinfer schema --json
evoinfer lifecycle-smoke --json
```
