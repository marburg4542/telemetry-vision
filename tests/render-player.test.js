"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const player = require("../src/telemetry_ocr/web_static/render-player.js");
const { flattenSegments, processedUntil, segmentAt, nextSegment, staleRanges, clampSeek } = player;

// What the server sends for a render that was stopped after six seconds and
// continued: two runs, the second starting where the first left off.
const INDEX = {
  complete: false,
  duration_ms: 60000,
  runs: [
    {
      run: "run-000",
      start_ms: 0,
      segments: [
        { file: "seg_00000.m4s", start_ms: 0, duration_ms: 2000 },
        { file: "seg_00001.m4s", start_ms: 2000, duration_ms: 2000 },
        { file: "seg_00002.m4s", start_ms: 4000, duration_ms: 2000 },
      ],
    },
    {
      run: "run-001",
      start_ms: 6000,
      segments: [
        { file: "seg_00000.m4s", start_ms: 6000, duration_ms: 2000 },
        { file: "seg_00001.m4s", start_ms: 8000, duration_ms: 1500 },
      ],
    },
  ],
};

test("segments are flattened onto the source timeline, run by run", () => {
  const segments = flattenSegments(INDEX);
  assert.equal(segments.length, 5);
  assert.deepEqual(
    segments.map((item) => [item.key, item.start, item.end]),
    [
      ["run-000/seg_00000.m4s", 0, 2],
      ["run-000/seg_00001.m4s", 2, 4],
      ["run-000/seg_00002.m4s", 4, 6],
      ["run-001/seg_00000.m4s", 6, 8],
      ["run-001/seg_00001.m4s", 8, 9.5],
    ],
  );
  // The offset each run's own timestamps are shifted by when appended.
  assert.equal(segments[0].runStart, 0);
  assert.equal(segments[3].runStart, 6);
  assert.equal(processedUntil(segments), 9.5);
  assert.equal(processedUntil([]), 0);
});

test("a segment is found by the moment it holds", () => {
  const segments = flattenSegments(INDEX);
  assert.equal(segmentAt(segments, 0).file, "seg_00000.m4s");
  assert.equal(segmentAt(segments, 6.5).key, "run-001/seg_00000.m4s");
  assert.equal(segmentAt(segments, 9.4).key, "run-001/seg_00001.m4s");
  assert.equal(segmentAt(segments, 9.5), null, "past the last frame rendered");
});

test("nothing is fetched twice, and a seek jumps the queue", () => {
  const segments = flattenSegments(INDEX);
  const loaded = new Set();
  assert.equal(nextSegment(segments, 0, loaded).key, "run-000/seg_00000.m4s");
  loaded.add("run-000/seg_00000.m4s");
  assert.equal(nextSegment(segments, 0, loaded).key, "run-000/seg_00001.m4s");
  // Seeking to the second run asks for what covers that moment, not the gap.
  assert.equal(nextSegment(segments, 7, loaded).key, "run-001/seg_00000.m4s");
  for (const item of segments) loaded.add(item.key);
  assert.equal(nextSegment(segments, 0, loaded), null, "everything is loaded");
});

test("only what is within the look-ahead is fetched", () => {
  const far = flattenSegments({
    runs: [
      {
        run: "run-000",
        start_ms: 0,
        segments: Array.from({ length: 60 }, (_, index) => ({
          file: `seg_${String(index).padStart(5, "0")}.m4s`,
          start_ms: index * 2000,
          duration_ms: 2000,
        })),
      },
    ],
  });
  const loaded = new Set();
  const ahead = 10;
  const due = [];
  for (let n = 0; n < 20; n += 1) {
    const segment = nextSegment(far, 0, loaded, ahead);
    if (!segment) break;
    loaded.add(segment.key);
    due.push(segment.start);
  }
  assert.deepEqual(due, [0, 2, 4, 6, 8, 10], "fetched past the look-ahead");
});

test("what is far from the playhead is dropped, and only that", () => {
  const segments = flattenSegments(INDEX);
  assert.deepEqual(staleRanges(segments, 3), [], "a short render is kept whole");
  assert.deepEqual(staleRanges(segments, 120, 60, 30), [[0, 60]], "an hour in, the start goes");
  // After seeking back, what was fetched far ahead is dropped as well.
  const long = flattenSegments({
    runs: [
      {
        run: "run-000",
        start_ms: 0,
        segments: [{ file: "seg_00000.m4s", start_ms: 0, duration_ms: 600000 }],
      },
    ],
  });
  assert.deepEqual(staleRanges(long, 100, 60, 30), [[0, 40], [140, 660]]);
});

test("a seek past what is rendered lands at the front", () => {
  assert.equal(clampSeek(5, 9.5, false), 5);
  assert.equal(clampSeek(30, 9.5, false), 9, "dragged into what does not exist yet");
  assert.equal(clampSeek(30, 60, true), 30, "a finished render seeks anywhere");
  assert.equal(clampSeek(5, 0, false), 0, "nothing rendered yet");
  assert.equal(clampSeek(9.4, 9.5, false), 9, "right at the front counts as past it");
});

test("a malformed index yields no segments rather than throwing", () => {
  assert.deepEqual(flattenSegments(null), []);
  assert.deepEqual(flattenSegments({}), []);
  assert.deepEqual(flattenSegments({ runs: [{ run: "run-000" }] }), []);
  const broken = { runs: [{ run: "run-000", start_ms: 0, segments: [
    { file: "seg_00000.m4s", start_ms: "x", duration_ms: 2000 },
    { file: "seg_00001.m4s", start_ms: 0, duration_ms: 0 },
  ] }] };
  assert.deepEqual(flattenSegments(broken), []);
});
