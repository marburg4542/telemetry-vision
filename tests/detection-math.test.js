"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  boxCentre,
  clampSpeed,
  median,
  smooth,
  usableGap,
  letterbox,
  extrapolationReach,
  autoIntervalMs,
  projectMediaTime,
  formatDms,
} = require("../src/telemetry_ocr/web_static/detection-math.js");

// The constants app.js is currently tuned to. Duplicated on purpose: a test
// that imported them would silently follow a bad retune, and several of these
// assertions exist to catch exactly that.
const MIN_SAMPLE_GAP_MS = 30;
const MAX_SAMPLE_GAP_MS = 600;
const AUTO_MIN_INTERVAL_MS = 1000 / 20;
const AUTO_MAX_INTERVAL_MS = 400;
const AUTO_HEADROOM = 1.1;

test("median of an empty list is null, not NaN", () => {
  // NaN here would survive every Number.isFinite guard downstream as "not
  // finite" and quietly zero the camera velocity for the rest of the session.
  assert.equal(median([]), null);
  assert.equal(median([5]), 5);
  assert.equal(median([1, 3]), 2);
  assert.equal(median([9, 1, 5]), 5);
});

test("median follows the majority, not the outlier", () => {
  // Parked cars under a panning camera all share one apparent velocity; one car
  // genuinely driving must not drag the estimate toward itself.
  assert.equal(median([0.12, 0.12, 0.12, 0.9]), 0.12);
});

test("an unusable speed is null rather than a fabricated zero", () => {
  assert.equal(clampSpeed(Number.NaN, 1.5), null);
  assert.equal(clampSpeed(undefined, 1.5), null);
  assert.equal(clampSpeed(0.4, 1.5), 0.4);
  assert.equal(clampSpeed(9, 1.5), 1.5);
  assert.equal(clampSpeed(-9, 1.5), -1.5);
});

test("smoothing starts from the first real sample and ignores rubbish", () => {
  assert.equal(smooth(null, 0.2, 0.45), 0.2);
  assert.equal(smooth(undefined, 0.2, 0.45), 0.2);
  assert.equal(smooth(0.2, Number.NaN, 0.45), 0.2);
  assert.equal(smooth(null, Number.NaN, 0.45), null);
  assert.ok(Math.abs(smooth(0, 1, 0.45) - 0.45) < 1e-12);
});

test("smoothing converges on a steady value", () => {
  let value = null;
  for (let i = 0; i < 40; i += 1) value = smooth(value, 0.12, 0.45);
  assert.ok(Math.abs(value - 0.12) < 1e-6);
});

test("the sample window excludes gaps that cannot support an estimate", () => {
  assert.equal(usableGap(3, MIN_SAMPLE_GAP_MS, MAX_SAMPLE_GAP_MS), false);
  assert.equal(usableGap(100, MIN_SAMPLE_GAP_MS, MAX_SAMPLE_GAP_MS), true);
  assert.equal(usableGap(5000, MIN_SAMPLE_GAP_MS, MAX_SAMPLE_GAP_MS), false);
  assert.equal(usableGap(Number.NaN, MIN_SAMPLE_GAP_MS, MAX_SAMPLE_GAP_MS), false);
});

test("the sample window still admits the slowest rate auto pacing can choose", () => {
  // These two bounds were once the same number, so auto pacing at its ceiling
  // put every gap exactly on the rejection boundary and threw away per-box
  // velocity on precisely the machines slow enough to need it.
  assert.ok(
    MAX_SAMPLE_GAP_MS > AUTO_MAX_INTERVAL_MS,
    "the velocity window must outlast the slowest submission interval",
  );
  assert.equal(usableGap(AUTO_MAX_INTERVAL_MS, MIN_SAMPLE_GAP_MS, MAX_SAMPLE_GAP_MS), true);
});

test("auto pacing stays under the worker and inside its bounds", () => {
  // Submitting faster than the worker drains its one-deep queue makes results
  // later, because each inference then starts on a frame that already waited.
  assert.ok(autoIntervalMs(117, AUTO_HEADROOM, AUTO_MIN_INTERVAL_MS, AUTO_MAX_INTERVAL_MS) > 117);
  // A free GPU should reach the ceiling rate, not sit at some cautious default.
  assert.equal(
    autoIntervalMs(10, AUTO_HEADROOM, AUTO_MIN_INTERVAL_MS, AUTO_MAX_INTERVAL_MS),
    AUTO_MIN_INTERVAL_MS,
  );
  // And a very slow worker must not push the gap past what the tracker survives.
  assert.equal(
    autoIntervalMs(5000, AUTO_HEADROOM, AUTO_MIN_INTERVAL_MS, AUTO_MAX_INTERVAL_MS),
    AUTO_MAX_INTERVAL_MS,
  );
  assert.equal(autoIntervalMs(null, AUTO_HEADROOM, AUTO_MIN_INTERVAL_MS, AUTO_MAX_INTERVAL_MS), null);
  assert.equal(autoIntervalMs(0, AUTO_HEADROOM, AUTO_MIN_INTERVAL_MS, AUTO_MAX_INTERVAL_MS), null);
});

test("the reach follows the cadence and respects both limits", () => {
  assert.equal(extrapolationReach(0, 250, 600), 250);
  assert.equal(extrapolationReach(100, 250, 600), 250);
  assert.equal(extrapolationReach(200, 250, 600), 400);
  assert.equal(extrapolationReach(9000, 250, 600), 600);
});

test("a full-frame box maps onto the whole picture at any stage shape", () => {
  // The stage stopped being 16:9 when the layout was made to fit one screen, so
  // the overlay cannot assume it. Pillarboxed, letterboxed and exact all have to
  // place a full-frame detection exactly over the video.
  for (const [stageW, stageH] of [[1100, 619], [1100, 420], [1100, 800], [640, 360]]) {
    const box = letterbox(stageW, stageH, 640, 360);
    const right = box.offsetX + 640 * box.scaleX;
    const bottom = box.offsetY + 360 * box.scaleY;
    assert.ok(Math.abs(right - (box.offsetX + box.displayWidth)) < 1e-9);
    assert.ok(Math.abs(bottom - (box.offsetY + box.displayHeight)) < 1e-9);
    // Centred, and never spilling outside the stage.
    assert.ok(box.offsetX >= -1e-9 && box.offsetY >= -1e-9);
    assert.ok(box.displayWidth <= stageW + 1e-9 && box.displayHeight <= stageH + 1e-9);
  }
});

test("letterboxing preserves the source aspect ratio", () => {
  const box = letterbox(1100, 420, 1920, 1080);
  assert.ok(Math.abs(box.displayWidth / box.displayHeight - 1920 / 1080) < 1e-9);
  assert.ok(Math.abs(box.scaleX - box.scaleY) < 1e-12);
});

test("the media clock leads currentTime while playing", () => {
  const base = {
    clockMs: 1000, clockWallMs: 0, playbackRate: 1, paused: false,
    currentTimeMs: 1000, leadMs: 16, stallMs: 250,
  };
  // currentTime is quantised to the frame boundary; the projection is not, so
  // staleness stops being under-reported.
  assert.equal(projectMediaTime({ ...base, nowWallMs: 20 }), 1036);
  assert.equal(projectMediaTime({ ...base, nowWallMs: 20, playbackRate: 2 }), 1072);
});

test("the media clock falls back rather than running away from the picture", () => {
  const base = {
    clockMs: 1000, clockWallMs: 0, playbackRate: 1, paused: false,
    currentTimeMs: 1000, leadMs: 16, stallMs: 250,
  };
  assert.equal(projectMediaTime({ ...base, nowWallMs: 400 }), 1000, "stalled");
  assert.equal(projectMediaTime({ ...base, nowWallMs: 20, paused: true }), 1000, "paused");
  assert.equal(projectMediaTime({ ...base, nowWallMs: 20, clockMs: null }), 1000, "no frame yet");
});

test("box centres come from the corners", () => {
  const box = { x1: 10, y1: 20, x2: 30, y2: 60 };
  assert.equal(boxCentre(box, "x"), 20);
  assert.equal(boxCentre(box, "y"), 40);
});

test("DMS carries seconds and minutes instead of printing sixty", () => {
  // Rounding can land exactly on sixty, and carrying can do it again, so a
  // coordinate reads 14°59'60.00" unless both steps run.
  assert.equal(formatDms(14.9999999, 2, 2).text, `15°00'00.00"`);
  assert.equal(formatDms(14.5, 2, 2).text, `14°30'00.00"`);
  assert.equal(formatDms(101.99999999, 3, 2).text, `102°00'00.00"`);
  const carried = formatDms(14.9999999, 2, 2);
  assert.equal(carried.minutes, 0);
  assert.ok(carried.seconds < 60);
});

test("DMS pads to the configured width", () => {
  // Longitude carries three degree digits, latitude two; the panel columns line
  // up only if the padding is honoured.
  assert.equal(formatDms(5.5, 3, 2).text, `005°30'00.00"`);
  assert.equal(formatDms(5.5, 2, 2).text, `05°30'00.00"`);
  assert.equal(formatDms(-14.5, 2, 2).text, `14°30'00.00"`, "sign is carried elsewhere");
});
