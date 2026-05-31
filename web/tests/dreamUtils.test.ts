import assert from "node:assert/strict";
import test from "node:test";

import {
  filterDreamMemories,
  formatUsefulRate,
  groupDreamMemories,
} from "../src/features/dream/dream-utils";
import type { DreamMemory } from "../src/features/dream/types";

const memories: DreamMemory[] = [
  {
    id: "opt",
    category: "optimization",
    title: "Chunked prefill improved TTFT",
    environment: "A100",
    inference_backend: "vLLM",
    useful_rate: 0.714,
  },
  {
    id: "debug",
    category: "environment_debug",
    title: "CUDA wheel mismatch",
    root_cause: "Torch wheel was built for a different CUDA runtime.",
  },
];

test("groupDreamMemories separates optimization and environment debug memories", () => {
  const grouped = groupDreamMemories(memories);

  assert.deepEqual(
    grouped.optimization.map((memory) => memory.id),
    ["opt"],
  );
  assert.deepEqual(
    grouped.environment_debug.map((memory) => memory.id),
    ["debug"],
  );
});

test("filterDreamMemories filters by category and searchable fields", () => {
  const result = filterDreamMemories(memories, "cuda runtime", "environment_debug");

  assert.deepEqual(
    result.map((memory) => memory.id),
    ["debug"],
  );
});

test("formatUsefulRate renders percentage labels", () => {
  assert.equal(formatUsefulRate(0.714), "71%");
  assert.equal(formatUsefulRate(undefined), "0%");
});
