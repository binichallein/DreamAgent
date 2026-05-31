import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const NON_MEDIA_ATTACHMENT_CLASS_REGEX =
  /<PromptInputHoverCard>[\s\S]*?<div\s+className={cn\(\s*"([^"]+)"/;
const ATTACHMENT_MAX_WIDTH_REGEX = /(?:^|\s)max-w-\[200px\](?:\s|$)/;

test("non-media prompt input attachments cap their width so long filenames truncate", () => {
  const source = readFileSync(
    resolve(__dirname, "../src/components/ai-elements/prompt-input.tsx"),
    "utf8",
  );
  const match = source.match(NON_MEDIA_ATTACHMENT_CLASS_REGEX);

  assert.ok(match, "expected to find the non-media attachment container");
  assert.match(match[1], ATTACHMENT_MAX_WIDTH_REGEX);
});
