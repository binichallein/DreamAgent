import assert from "node:assert/strict";
import test from "node:test";

import { clearAuthToken, getAuthToken, setAuthToken } from "../src/lib/auth";

class FakeStorage {
  private values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }

  clear(): void {
    this.values.clear();
  }
}

test("stored web auth token remains valid client-side until explicitly replaced or cleared", () => {
  const originalLocalStorage = globalThis.localStorage;
  const fakeStorage = new FakeStorage();
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: fakeStorage,
  });

  try {
    setAuthToken("server-token");
    fakeStorage.setItem("kimi_auth_token_ts", "0");

    assert.equal(getAuthToken(), "server-token");

    clearAuthToken();
    assert.equal(getAuthToken(), null);
  } finally {
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: originalLocalStorage,
    });
  }
});
