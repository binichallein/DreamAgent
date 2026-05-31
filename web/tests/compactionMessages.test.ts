import assert from "node:assert/strict";
import test from "node:test";

import { removeCompactionIndicator } from "../src/hooks/compactionMessages";

test("removeCompactionIndicator preserves chat history and removes only the transient compaction status", () => {
  const messages = [
    {
      id: "old-user",
      role: "user",
      variant: "text",
      content: "hello",
    },
    {
      id: "old-assistant",
      role: "assistant",
      variant: "text",
      content: "hi",
    },
    {
      id: "compact-user",
      role: "user",
      variant: "text",
      content: "/compact",
    },
    {
      id: "compact-status",
      role: "assistant",
      variant: "status",
      content: "Compacting conversation history...",
      isStreaming: true,
    },
  ] as const;

  const result = removeCompactionIndicator([...messages], "compact-status");

  assert.deepEqual(
    result.map((message) => message.id),
    ["old-user", "old-assistant", "compact-user"],
  );
});
