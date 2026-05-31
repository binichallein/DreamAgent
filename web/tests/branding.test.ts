import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TITLE_REGEX = /<title>EvoInfer<\/title>/;
const KIMI_FAVICON_REGEX = /<link\s+rel="icon"[^>]*href="\/logo\.png"[^>]*>/;
const SIDEBAR_BRAND_REGEX = />\s*EvoInfer\s*</;
const OLD_SIDEBAR_BRAND_REGEX = />\s*Kimi Code\s*</;

test("browser tab title uses EvoInfer branding", () => {
  const source = readFileSync(resolve(__dirname, "../index.html"), "utf8");

  assert.match(source, TITLE_REGEX);
});

test("browser tab does not use the Kimi favicon", () => {
  const source = readFileSync(resolve(__dirname, "../index.html"), "utf8");

  assert.doesNotMatch(source, KIMI_FAVICON_REGEX);
});

test("sidebar brand label uses EvoInfer branding", () => {
  const source = readFileSync(
    resolve(__dirname, "../src/features/sessions/sessions.tsx"),
    "utf8",
  );

  assert.match(source, SIDEBAR_BRAND_REGEX);
  assert.doesNotMatch(source, OLD_SIDEBAR_BRAND_REGEX);
});
