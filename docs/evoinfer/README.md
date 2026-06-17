# EvoInfer Dream MCP

EvoInfer Dream is an open-box MCP memory manager for inference optimization
agents. It gives a code agent tools to retrieve, write, verify, and maintain
evidence-backed memories from optimization and environment-debug work.

EvoInfer Dream is not a chat UI and it does not replace the agent runtime. It is
the memory/control component that an agent can call through MCP when it needs
optimization experience, deployment-debug experience, artifact-based memory
extraction, or feedback updates.

## Shared Local Memory

EvoInfer does not cold-start every session. The package ships with a versioned
seed memory store at `src/evoinfer_mcp/dream/seed_memories.json`. When you start
an EvoInfer-managed session or generate an MCP config, EvoInfer merges any
missing seed memories into your local durable store:

```text
~/.evoinfer/dream-share/dream/memories.json
```

The merge is idempotent and never overwrites local memories with the same ID, so
your feedback counters, promoted memories, and edits remain authoritative.
Claude/Codex sessions launched through `evoinfer` share this same local store
unless you explicitly pass `--share-dir`.

If a client starts the MCP server directly, the first `dream_search_memories`
call also lazily merges missing packaged seeds into the shared store. This
prevents retrieval cold-starts and lets package upgrades add new built-in
experience without overwriting local user memories.

Current packaged experience includes CUDA softmax/RMSNorm operator memories,
FlashInfer attention baseline evidence, FLA route-policy campaign evidence, a
limx FLA parallel-attention boundary memory against PyTorch SDPA, verified
limx backend-attention route memories for vLLM, SGLang, and TensorRT, and
reusable environment-debug memories for FlashInfer JIT, TensorRT NVIDIA PyPI
setup, voice STT CPU fallback, and vLLM/SGLang/TensorRT wheel-download stalls.

You can seed or inspect the local store manually:

```bash
evoinfer memory-seed --json
evoinfer memory-export --json
```

## What It Stores

EvoInfer currently manages two memory categories:

- Optimization memories: hardware, model/operator, backend, dtype, workload,
  baseline metrics, optimized metrics, correctness evidence, profiler/source
  evidence, and transfer constraints.
- Environment-debug memories: machine environment, component, symptoms, root
  cause, fix steps, verification evidence, and reusable deployment notes.

Successful memories are treated as hypotheses with evidence, not as rules.
Negative memories are used as transfer constraints so an agent does not reuse an
optimization under incompatible workload, backend, dtype, or operator semantics.

## Install

```bash
git clone https://github.com/binichallein/DreamAgent.git
cd DreamAgent
uv tool install --force --editable .
evoinfer doctor --json
evoinfer lifecycle-smoke --json
```

`doctor` checks package imports, share-directory writability, MCP tool surface,
and stdio startup. `lifecycle-smoke` runs a local end-to-end memory flow:
protocol, search, stuck search, stage, extract, write, promote, feedback, list,
and protocol verification.

To explicitly initialize the shared local memory store:

```bash
evoinfer memory-seed --json
```

## Claude Code Setup

Generate a Claude Code MCP registration command:

```bash
evoinfer mcp-config \
  --client claude \
  --format claude-add-json \
  --share-dir ~/.evoinfer/dream-share \
  --scope user
```

Run the printed command once. It will look like:

```bash
claude mcp add-json evoinfer-dream '{"type":"stdio","command":"/abs/path/to/python","args":["-m","evoinfer_mcp.dream.mcp_server"],"env":{"EVOINFER_SHARE_DIR":"/home/user/.evoinfer/dream-share"}}' --scope user
```

Verify the registration:

```bash
claude mcp list
claude mcp get evoinfer-dream
claude
```

Inside Claude Code, run:

```text
/mcp
```

`evoinfer-dream` should appear as a connected stdio MCP server.

## Codex Setup

Generate a Codex MCP config block:

```bash
evoinfer mcp-config \
  --client codex \
  --format codex-toml \
  --share-dir ~/.evoinfer/dream-share
```

Short form: `evoinfer mcp-config --client codex --format codex-toml`.

Add the printed `[mcp_servers.evoinfer-dream]` block to the Codex MCP config.

## Generic MCP Setup

Any stdio-compatible MCP client can launch:

```bash
python -m evoinfer_mcp.dream.mcp_server
```

Recommended environment:

```bash
EVOINFER_SHARE_DIR=/abs/path/to/evoinfer-share
```

The durable memory store is written under:

```text
$EVOINFER_SHARE_DIR/dream/memories.json
```

## Mandatory Session Mode

For validation or high-stakes optimization tasks, prefer a dedicated mandatory
Dream session instead of a global optional MCP registration. This creates an
isolated session directory with:

- `mcp.json`: per-session stdio MCP config.
- `mcp_calls.jsonl`: per-session Dream tool call audit log.
- `work/AGENTS.md` and `work/CLAUDE.md`: mandatory Dream protocol instructions.
- `share/`: the Dream memory store used by this session.

Create the bundle:

```bash
evoinfer force-session \
  --session-dir /tmp/evoinfer-dream-session \
  --share-dir ~/.evoinfer/dream-share \
  --workdir /tmp/evoinfer-dream-session/work
```

The printed commands run Claude Code, Codex, or Kimi CLI inside that dedicated
mandatory Dream session. The MCP server also receives:

```bash
EVOINFER_DREAM_MANDATORY=1
EVOINFER_DREAM_SESSION_ID=<session-name>
EVOINFER_MCP_CALL_LOG=<session-dir>/mcp_calls.jsonl
```

This does not give the MCP server magical control over an agent. MCP servers
cannot initiate tool calls by themselves. The enforcement comes from session
scoping: the agent starts with only this session's config and instructions, and
the call log makes missed Dream protocol steps auditable.

## Hooked Agent Sessions

For day-to-day use, start a Claude Code or Codex session through EvoInfer so
Dream is checked during the run, not only at the beginning:

```bash
evoinfer --client codex --hook-every-steps 10
evoinfer --client claude --hook-every-steps 10
```

Running `evoinfer` without options opens a small terminal prompt that asks
which client to use and how often to checkpoint Dream.

Hook mode writes a dedicated session bundle and launches the chosen agent:

- Claude Code uses `.claude/settings.local.json` with `SessionStart`,
  `PostToolBatch`, and `Stop` hooks.
- Codex uses `.codex/hooks.json` with `SessionStart`, `PostToolUse`, and `Stop`
  hooks, and launches with `--dangerously-bypass-hook-trust` for that generated
  session.
- Every N tool checkpoints, the hook runs `evoinfer_mcp.hooks.dream_checkpoint`,
  searches Dream, writes `dream_context.md`, and injects the checkpoint result
  back as hook context before the agent continues.

Kimi CLI is not wired into hook mode yet. The current stable hook-backed path is
Claude Code and Codex only.

## Soft Protocol For Agents

MCP tools are available only when the agent decides to call them. For reliable
autonomous use, add this instruction to the agent's system prompt or project
instructions:

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

Expected agent loop:

1. `dream_get_agent_protocol`
2. `dream_search_memories` before local exploration
3. Real benchmark, correctness, profiling, and environment inspection
4. `dream_search_memories` again when stuck or switching optimization route
5. Write artifacts such as `benchmark_raw.json`, `correctness_raw.json`,
   `profiler_summary.json`, `environment.json`, and `verifier_result.json`
6. `dream_extract_and_write_memories`
7. `dream_promote_memory`, `dream_reject_memory`, or `dream_record_feedback`
   when later evidence is available

Auto-write creates candidate memories first. Promotion requires explicit
evidence review.

## Main MCP Tools

- `dream_get_agent_protocol`: returns the current EvoInfer protocol.
- `dream_search_memories`: hybrid structured and lexical retrieval with schema
  filters, evidence-aware scoring, transfer checks, and optional embedding
  scoring.
- `dream_stage_memory_candidate`: writes a candidate file into a work directory
  without changing the durable memory store.
- `dream_extract_memory_candidates`: extracts candidate memories from campaign
  artifacts.
- `dream_extract_and_write_memories`: artifact-gated extraction and candidate
  write.
- `dream_promote_memory`: promotes a candidate after evidence review.
- `dream_reject_memory`: rejects a candidate or records a negative transfer
  constraint.
- `dream_record_feedback`: updates chosen/useful counters after later reuse.
- `dream_list_memories`, `dream_get_memory`, `dream_export_memory_store`,
  `dream_import_memory_store`: memory administration.

## Artifact-Gated Memory Writing

Dream memory should be extracted from evidence artifacts before it is trusted.
The extractor looks for standard files in a task work directory:

- `benchmark_raw.json`
- `correctness_raw.json`
- `profiler_summary.json`
- `environment.json`
- `verifier_result.json`
- `dream_write_candidates.json`

The verifier checks that a memory candidate has enough evidence before writing.
For optimization memories, benchmark and correctness evidence are required; a
profiler summary or source-level bottleneck evidence is expected for high-quality
promotion. For environment-debug memories, symptoms, root cause, solution, and
verification evidence are expected.

## Optional CPU Embeddings

Retrieval works without an embedding model by default. It uses structured schema
fields, tags, workload hints, lexical matches, negative memories, and
evidence-aware reranking.

For stronger semantic recall, install the optional embedding dependency and
enable the local CPU backend in the generated MCP config:

```bash
uv tool install --force --editable ".[embedding]"
evoinfer mcp-config \
  --client codex \
  --format codex-toml \
  --share-dir ~/.evoinfer/dream-share \
  --enable-embedding \
  --embedding-model BAAI/bge-small-zh-v1.5
```

The generated MCP server environment includes:

```bash
EVOINFER_EMBEDDING_BACKEND=local
EVOINFER_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EVOINFER_EMBEDDING_DEVICE=cpu
```

The agent client does not need native embedding support. It only calls EvoInfer
Dream MCP tools; the embedding model is configured and run inside the MCP server
on CPU. Embedding scores are one retrieval signal, not the final authority.
Schema filters, workload closeness, correctness evidence, profiler evidence, and
negative-memory constraints still gate the final result.

## Runtime Checks

```bash
evoinfer memory-seed --json
evoinfer doctor --json
evoinfer schema --json
evoinfer lifecycle-smoke --json
```
