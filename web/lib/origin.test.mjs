import assert from "node:assert/strict";
import { test } from "node:test";
import { validLocalOrigin } from "./origin.ts";

test("accepts same-origin loopback browsers and local CLI", () => {
  assert.equal(validLocalOrigin("http://127.0.0.1:3000", "127.0.0.1:3000"), true);
  assert.equal(validLocalOrigin("http://localhost:3000", "localhost:3000"), true);
  assert.equal(validLocalOrigin(null, "127.0.0.1:3000"), true);
});

test("rejects cross-origin, malformed, mismatched port and remote hosts", () => {
  for (const [origin, host] of [["https://attacker.example", "127.0.0.1:3000"],
    ["http://localhost:3001", "localhost:3000"], ["http://127.0.0.1:3000", "localhost:3000"],
    ["http://attacker.example", "attacker.example"], ["null", "localhost:3000"],
    ["http://localhost:3000", null]]) assert.equal(validLocalOrigin(origin, host), false);
});
