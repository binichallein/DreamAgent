import assert from "node:assert/strict";
import test from "node:test";

import {
  shouldCancelSessionTitleEdit,
  shouldCommitSessionTitleEdit,
} from "../src/features/sessions/session-title-edit";

test("Enter commits a session title edit", () => {
  assert.equal(shouldCommitSessionTitleEdit("Enter"), true);
  assert.equal(shouldCommitSessionTitleEdit("a"), false);
});

test("IME composition Enter does not prematurely commit a title edit", () => {
  assert.equal(shouldCommitSessionTitleEdit("Enter", true), false);
});

test("Escape cancels a session title edit", () => {
  assert.equal(shouldCancelSessionTitleEdit("Escape"), true);
  assert.equal(shouldCancelSessionTitleEdit("Enter"), false);
});
