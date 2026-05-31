import assert from "node:assert/strict";
import test from "node:test";
import { File } from "node:buffer";

import {
  createPromptInputAttachment,
  getPromptInputUploadFile,
  preparePromptInputFiles,
} from "../src/components/ai-elements/prompt-input-files";

test("preparePromptInputFiles preserves blob URLs for direct backend upload", () => {
  const sourceFile = new File(["pdf"], "Build a Reasoning Model (From Scratch).pdf", {
    type: "application/pdf",
  });
  const files = [
    {
      id: "attachment-1",
      type: "file",
      url: "blob:http://localhost:5494/large-pdf",
      mediaType: "application/pdf",
      filename: "Build a Reasoning Model (From Scratch).pdf",
      sourceFile,
    },
  ] as const;

  const result = preparePromptInputFiles([...files]);

  assert.deepEqual(result, [
    {
      type: "file",
      url: "blob:http://localhost:5494/large-pdf",
      mediaType: "application/pdf",
      filename: "Build a Reasoning Model (From Scratch).pdf",
      sourceFile,
    },
  ]);
});

test("createPromptInputAttachment stores the original File alongside the preview URL", () => {
  const originalCreateObjectUrl = URL.createObjectURL;
  URL.createObjectURL = () => "blob:http://localhost:5494/source-file";
  try {
    const sourceFile = new File(["pdf"], "large.pdf", { type: "application/pdf" });

    const attachment = createPromptInputAttachment(sourceFile, "attachment-1");

    assert.equal(attachment.id, "attachment-1");
    assert.equal(attachment.url, "blob:http://localhost:5494/source-file");
    assert.equal(attachment.sourceFile, sourceFile);
  } finally {
    URL.createObjectURL = originalCreateObjectUrl;
  }
});

test("getPromptInputUploadFile uses sourceFile without fetching the blob URL", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (() => {
    throw new Error("fetch should not be called");
  }) as typeof fetch;
  try {
    const sourceFile = new File(["pdf"], "large.pdf", { type: "application/pdf" });

    const uploadFile = await getPromptInputUploadFile({
      type: "file",
      url: "blob:http://localhost:5494/source-file",
      mediaType: "application/pdf",
      filename: "large.pdf",
      sourceFile,
    });

    assert.equal(uploadFile, sourceFile);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
