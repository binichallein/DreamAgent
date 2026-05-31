import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REQUIRED_PAGE_PHRASES = [
  "自进化",
  "推理优化",
  "全局理解",
  "记忆库",
  "优化创新点",
] as const;
const SIDEBAR_EVOINFER_HANDLER_REGEX = /onSelectEvoInfer/;
const EVOINFER_ACTIVE_VIEW_REGEX = /activeView === "evoinfer"/;
const EVOINFER_PAGE_IMPORT_REGEX = /import \{ EvoInferPage \}/;
const EVOINFER_ACTIVE_VIEW_TYPE_REGEX =
  /type ActiveView = "chat" \| "dream" \| "evoinfer"/;
const EVOINFER_CENTERED_HERO_SECTION_REGEX =
  /<section className="space-y-6 border-b border-border pb-8">/;
const EVOINFER_CENTERED_IMAGE_FIGURE_REGEX =
  /<figure className="mx-auto w-full max-w-5xl overflow-hidden rounded-lg border border-border bg-muted\/25">/;
const EVOINFER_RIGHT_COLUMN_HERO_REGEX = /lg:grid-cols-\[1\.15fr_0\.85fr\]/;
const PRINCIPLE_IMAGE_PATH = "/evoinfer-principle.png";
const PRINCIPLE_IMAGE_ALT = "EvoInfer 自进化推理优化原理图";

test("EvoInfer project page explains the self-evolving inference optimization principle", () => {
  const source = readFileSync(
    resolve(__dirname, "../src/features/evoinfer/evoinfer-page.tsx"),
    "utf8",
  );

  for (const phrase of REQUIRED_PAGE_PHRASES) {
    assert.ok(source.includes(phrase), `expected page to mention ${phrase}`);
  }
});

test("EvoInfer project page includes the generated principle diagram", () => {
  const source = readFileSync(
    resolve(__dirname, "../src/features/evoinfer/evoinfer-page.tsx"),
    "utf8",
  );

  assert.ok(source.includes(PRINCIPLE_IMAGE_PATH));
  assert.ok(source.includes(PRINCIPLE_IMAGE_ALT));
  assert.ok(
    existsSync(resolve(__dirname, "../public/evoinfer-principle.png")),
    "expected generated EvoInfer principle image to exist in public assets",
  );
});

test("EvoInfer principle diagram is centered in the hero layout", () => {
  const source = readFileSync(
    resolve(__dirname, "../src/features/evoinfer/evoinfer-page.tsx"),
    "utf8",
  );

  assert.match(
    source,
    EVOINFER_CENTERED_HERO_SECTION_REGEX,
  );
  assert.match(
    source,
    EVOINFER_CENTERED_IMAGE_FIGURE_REGEX,
  );
  assert.doesNotMatch(source, EVOINFER_RIGHT_COLUMN_HERO_REGEX);
});

test("sidebar exposes EvoInfer as a navigable page", () => {
  const source = readFileSync(
    resolve(__dirname, "../src/features/sessions/sessions.tsx"),
    "utf8",
  );

  assert.match(source, SIDEBAR_EVOINFER_HANDLER_REGEX);
  assert.match(source, EVOINFER_ACTIVE_VIEW_REGEX);
});

test("app routes the EvoInfer sidebar entry to the project page", () => {
  const source = readFileSync(resolve(__dirname, "../src/App.tsx"), "utf8");

  assert.match(source, EVOINFER_PAGE_IMPORT_REGEX);
  assert.match(source, EVOINFER_ACTIVE_VIEW_TYPE_REGEX);
  assert.match(source, EVOINFER_ACTIVE_VIEW_REGEX);
});
