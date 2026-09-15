"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { formatDms } = require("../src/telemetry_ocr/web_static/format.js");

test("DMS carries seconds and minutes instead of printing sixty", () => {
  // 14.9999999 degrees is 14°59'59.99964", which rounds to sixty seconds. It
  // has to carry all the way up to the degrees.
  assert.equal(formatDms(14.9999999, 2, 2).text, `15°00'00.00"`);
  assert.equal(formatDms(14.5, 2, 2).text, `14°30'00.00"`);
  assert.equal(formatDms(101.99999999, 3, 2).text, `102°00'00.00"`);
  const carried = formatDms(14.9999999, 2, 2);
  assert.equal(carried.degrees, 15);
  assert.equal(carried.minutes, 0);
  assert.equal(carried.seconds, 0);
});

test("DMS pads to the configured width", () => {
  assert.equal(formatDms(5.5, 3, 2).text, `005°30'00.00"`);
  assert.equal(formatDms(5.5, 2, 2).text, `05°30'00.00"`);
  assert.equal(formatDms(-14.5, 2, 2).text, `14°30'00.00"`, "sign is carried elsewhere");
});

test("DMS keeps the configured decimal places", () => {
  assert.equal(formatDms(14.8647, 2, 1).text, `14°51'52.9"`);
  assert.equal(formatDms(14.8647, 2, 3).text, `14°51'52.920"`);
});
