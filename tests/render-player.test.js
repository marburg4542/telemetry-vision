"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const player = require("../src/telemetry_ocr/web_static/render-player.js");
const {
  flattenSegments, processedUntil, segmentAt, nextSegment, staleRanges, clampSeek,
  bufferTarget, readyToPlay, estimateWait, estimatePlayback, START_BUFFER_S, HOLD_BUFFER_S,
} = player;

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

function evenSegments(count, seconds = 2) {
  return flattenSegments({
    runs: [
      {
        run: "run-000",
        start_ms: 0,
        segments: Array.from({ length: count }, (_, index) => ({
          file: `seg_${String(index).padStart(5, "0")}.m4s`,
          start_ms: index * seconds * 1000,
          duration_ms: seconds * 1000,
        })),
      },
    ],
  });
}

test("what is far from the playhead is dropped, and only that", () => {
  assert.deepEqual(staleRanges(flattenSegments(INDEX), 3), [], "a short render is kept whole");
  const long = evenSegments(300); // ten minutes
  assert.deepEqual(staleRanges(long, 120, 60, 30), [[0, 60], [160, 660]]);
  assert.deepEqual(staleRanges(long, 20, 60, 30), [[60, 660]], "nothing behind yet");
});

test("a segment is dropped whole or not at all", () => {
  // Removing from the middle of a segment takes the rest of it with it, as
  // everything after its keyframe depends on that keyframe. A cut at 37.3 would
  // leave 36-38 empty yet counted as fetched: the stall this guards against.
  const long = evenSegments(300);
  const [behind, ahead] = staleRanges(long, 97.3, 60, 30);
  assert.deepEqual(behind, [0, 36], "the segment the cut falls in is kept");
  assert.deepEqual(ahead, [138, 660], "the one it falls in ahead is kept as well");
  for (const [from, to] of [behind, ahead]) {
    for (const item of long) {
      const overlaps = item.end > from && item.start < to;
      const inside = item.start >= from && item.end <= to;
      assert.equal(overlaps, inside, `${item.key} would be cut in half`);
    }
  }
});

test("a fetched segment the buffer no longer holds is fetched again", () => {
  const segments = evenSegments(10);
  const loaded = new Set(segments.map((item) => item.key));
  // 6-8 went missing; 12-14 is only partly there; frame timing puts the ends
  // a hair off the segment boundaries, which does not count as missing.
  const buffered = [[0, 5.98], [8.02, 12.9], [14, 20]];
  assert.deepEqual(
    player.missingSegments(segments, loaded, buffered).map((item) => item.start),
    [6, 12],
  );
  loaded.delete("run-000/seg_00003.m4s");
  assert.deepEqual(
    player.missingSegments(segments, loaded, buffered).map((item) => item.start),
    [12],
    "only what was fetched can be missing",
  );
  for (const item of player.missingSegments(segments, loaded, buffered)) loaded.delete(item.key);
  assert.equal(nextSegment(segments, 11, loaded).start, 12, "the hole is filled first");
});

test("a seek past what is rendered lands at the front", () => {
  assert.equal(clampSeek(5, 9.5, false), 5);
  assert.equal(clampSeek(30, 9.5, false), 9, "dragged into what does not exist yet");
  assert.equal(clampSeek(30, 60, true), 30, "a finished render seeks anywhere");
  assert.equal(clampSeek(5, 0, false), 0, "nothing rendered yet");
  assert.equal(clampSeek(9.4, 9.5, false), 9, "right at the front counts as past it");
});

test("playback waits for a stretch worth playing, not for the first segment", () => {
  // The renderer is slower than playback, so releasing at the first segment
  // means stopping again two seconds later, over and over.
  assert.equal(bufferTarget(false), START_BUFFER_S, "the first play waits for the start buffer");
  assert.equal(bufferTarget(true), HOLD_BUFFER_S, "catching up waits for more, not less");
  assert.equal(readyToPlay(2, HOLD_BUFFER_S, false), false);
  assert.equal(readyToPlay(HOLD_BUFFER_S, HOLD_BUFFER_S, false), true);
  // A finished render never waits: all of it is there.
  assert.equal(readyToPlay(0, HOLD_BUFFER_S, true), true);
});

test("the wait and the stretch that follows are estimated from the real rate", () => {
  // Six frames a second of a thirty fps video is a fifth of real time: fifteen
  // seconds of video takes seventy-five seconds to make, and then plays for
  // about nineteen before the playhead catches the renderer again.
  assert.equal(estimateWait(0, 15, 0.2), 75);
  assert.equal(estimateWait(10, 15, 0.2), 25);
  assert.equal(estimateWait(20, 15, 0.2), 0, "already past the target");
  assert.equal(estimateWait(0, 15, 0), null, "nothing known about the rate yet");
  assert.equal(estimatePlayback(15, 0.2), 18.75);
  assert.equal(estimatePlayback(15, 1.2), Infinity, "a render keeping up never stops");
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
