"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const overlay = require("../src/telemetry_ocr/web_static/track-overlay.js");
const {
  boxesAt, frameAt, classColour, boxLabel, fitRect, clampSeek, bufferTarget, aheadNeeded,
  readyToPlay, estimateWait, estimatePlayback, START_BUFFER_S, HOLD_BUFFER_S, RATE_MARGIN,
} = overlay;

// A box as the server sends it: [x1, y1, x2, y2, confidence, class_id, track_id, label].
function box(trackId, x, label = "car") {
  return [x, 10, x + 20, 30, 0.8, 0, trackId, label];
}

// The same render test_render_export.BoxPlacementTests uses, so the page and the
// saved MP4 put a box in the same place. Every 3rd frame: car 1 moving right,
// car 2 arriving at 3, car 3 leaving after 0, and a box not yet numbered at 3.
function render() {
  return {
    frames: [0, 3, 6],
    boxes: new Map([
      [0, [box(1, 0), box(3, 100)]],
      [3, [box(1, 30), box(2, 200), box(null, 300)]],
      [6, [box(1, 60), box(2, 200)]],
    ]),
  };
}

function at(frame, { frames, boxes } = render(), stride = 3) {
  return new Set(boxesAt(frame, frames, boxes, stride).map((item) => `${item[6]}@${+item[0].toFixed(3)}`));
}

test("a processed frame shows its own boxes", () => {
  assert.deepEqual(at(3), new Set(["1@30", "2@200", "null@300"]));
});

test("a vehicle on both sides is placed between", () => {
  assert.ok(at(1).has("1@10"));
  assert.ok(at(2).has("1@20"));
});

test("one seen on one side only shows for the nearer half", () => {
  assert.ok(at(1).has("3@100"), "leaving: still there just after");
  assert.ok(![...at(2)].some((item) => item.startsWith("3@")), "gone before the next frame");
  assert.ok(![...at(1)].some((item) => item.startsWith("2@")), "arriving: not yet");
  assert.ok(at(2).has("2@200"));
  assert.ok(at(2).has("null@300"), "an unnumbered box, from its side");
  assert.ok(![...at(1)].some((item) => item.startsWith("null@")));
});

test("past the last frame the boxes are held one stride", () => {
  assert.deepEqual(at(8), new Set(["1@60", "2@200"]));
  assert.deepEqual(at(9), new Set());
  assert.deepEqual(boxesAt(0, [], new Map(), 3), []);
});

test("frames not processed are not bridged", () => {
  const gap = render();
  gap.frames = [0, 9];
  gap.boxes.set(9, [box(1, 90)]);
  assert.deepEqual(at(2, gap), new Set(["1@0", "3@100"]));
  assert.deepEqual(at(4, gap), new Set());
});

test("the frame on screen is the one whose time has started", () => {
  assert.equal(frameAt(0, 30), 0);
  assert.equal(frameAt(1 / 30, 30), 1);
  assert.equal(frameAt(0.999 / 30, 30), 0);
  assert.equal(frameAt(10, 29.97), 299);
});

test("boxes look like the captures' marked-up copies", () => {
  // annotate.class_colour: hue (id * 67 + 88) % 360.
  assert.equal(classColour(0), "hsl(88, 88%, 62%)");
  assert.equal(classColour(2), "hsl(222, 88%, 62%)");
  assert.equal(boxLabel([0, 0, 1, 1, 0.876, 0, 12, "car"]), "car #12 0.88");
  assert.equal(boxLabel([0, 0, 1, 1, 0.5, 0, null, "truck"]), "truck 0.50");
});

test("the picture is found inside a letterboxed video element", () => {
  const wide = fitRect(1000, 1000, 1920, 1080);
  assert.ok(Math.abs(wide.x) < 1e-9);
  assert.ok(Math.abs(wide.width - 1000) < 1e-9);
  assert.ok(Math.abs(wide.y - (1000 - 562.5) / 2) < 1e-9);
  assert.equal(wide.scale, 1000 / 1920);
  assert.equal(fitRect(0, 100, 1920, 1080), null, "no size yet");
  assert.equal(fitRect(100, 100, 0, 0), null, "no video yet");
});

test("a seek past what is tracked lands at the front", () => {
  assert.equal(clampSeek(5, 9.5, false), 5);
  assert.equal(clampSeek(30, 9.5, false), 9);
  assert.equal(clampSeek(30, 60, true), 30, "a finished render seeks anywhere");
  assert.equal(clampSeek(5, 0, false), 0, "nothing tracked yet");
});

test("playback waits for a stretch worth playing", () => {
  assert.equal(bufferTarget(false), START_BUFFER_S);
  assert.equal(bufferTarget(true), HOLD_BUFFER_S);
  assert.equal(readyToPlay(2, HOLD_BUFFER_S, false), false);
  assert.equal(readyToPlay(HOLD_BUFFER_S, HOLD_BUFFER_S, false), true);
  // Nothing more is coming, so nothing is waited for.
  assert.equal(readyToPlay(0, HOLD_BUFFER_S, true), true);
  assert.equal(readyToPlay(0, null, false), false, "no rate yet, no start");
});

test("a slow render is waited for once, long enough to play to the end", () => {
  const needed = aheadNeeded(600, 0.5, false);
  assert.equal(needed, 600 * (1 - 0.5 * RATE_MARGIN));
  for (let t = 0; t <= 600; t += 1) {
    assert.ok(t <= needed + 0.5 * t + 1e-9, `caught the tracker up at ${t} s`);
  }
  assert.equal(aheadNeeded(8, 0.5, false), 8);
  assert.equal(aheadNeeded(600, 1.5, false), START_BUFFER_S);
  assert.equal(aheadNeeded(600, 1.5, true), HOLD_BUFFER_S);
  assert.equal(aheadNeeded(5, 1.5, true), 5, "never more than there is to play");
  assert.equal(aheadNeeded(600, 0, false), null, "rate not known yet");
  assert.ok(estimatePlayback(needed, 0.5) >= 600);
});

test("the wait is estimated from the real rate", () => {
  assert.equal(estimateWait(0, 15, 0.2), 75);
  assert.equal(estimateWait(20, 15, 0.2), 0);
  assert.equal(estimateWait(0, 15, 0), null);
  assert.equal(estimatePlayback(15, 0.2), 18.75);
  assert.equal(estimatePlayback(15, 1.2), Infinity);
});
