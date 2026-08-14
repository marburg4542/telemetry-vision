"use strict";

// The arithmetic behind the detection overlay, kept free of the DOM, the page's
// state object and the network so it can be exercised directly.
//
// This is where the silent failures live: an off-by-one in the sixty-carry, a
// sample window whose bounds collide with the pacing ceiling, a median over an
// empty list. None of it is visible from a screenshot -- a box lands a few
// pixels out and nobody can say why. Everything here takes its limits as
// arguments rather than reading module constants, so a test states the regime
// it is checking instead of inheriting whatever the page happens to be tuned to.
//
// Loaded as a plain script before app.js, and required directly by the tests.
(function (root) {
  function boxCentre(box, axis) {
    return axis === "x"
      ? (Number(box.x1) + Number(box.x2)) / 2
      : (Number(box.y1) + Number(box.y2)) / 2;
  }

  function clampSpeed(value, limit) {
    // NaN is not slow, it is unknown. Mapping it to zero would push a fabricated
    // "stationary" reading into the camera median and drag the estimate down.
    if (!Number.isFinite(value)) return null;
    return Math.max(-limit, Math.min(limit, value));
  }

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function smooth(previousValue, nextValue, weight) {
    if (!Number.isFinite(nextValue)) return Number.isFinite(previousValue) ? previousValue : null;
    return Number.isFinite(previousValue)
      ? previousValue + (nextValue - previousValue) * weight
      : nextValue;
  }

  // Whether a pair of samples can support a velocity estimate at all. Too close
  // together and the division amplifies a pixel of jitter into a wild speed; too
  // far apart and the earlier sample says nothing about motion now.
  function usableGap(gapMs, minGapMs, maxGapMs) {
    return Number.isFinite(gapMs) && gapMs >= minGapMs && gapMs <= maxGapMs;
  }

  // Reimplements `object-fit: contain` with a centred position, which is how the
  // video sits inside its stage. Independent of the stage's aspect ratio, so the
  // overlay follows the video through any layout or fullscreen change.
  function letterbox(stageWidth, stageHeight, sourceWidth, sourceHeight) {
    const videoRatio = sourceWidth / sourceHeight;
    const stageRatio = stageWidth / stageHeight;
    const displayWidth = videoRatio > stageRatio ? stageWidth : stageHeight * videoRatio;
    const displayHeight = videoRatio > stageRatio ? stageWidth / videoRatio : stageHeight;
    return {
      displayWidth,
      displayHeight,
      offsetX: (stageWidth - displayWidth) / 2,
      offsetY: (stageHeight - displayHeight) / 2,
      scaleX: displayWidth / sourceWidth,
      scaleY: displayHeight / sourceHeight,
    };
  }

  // How far ahead the prediction may reach. Tied to the observed result cadence
  // rather than to the measured lag: a straight line drawn from two samples
  // holds for about two intervals, and past that it overshoots every time the
  // subject turns, which reads worse than trailing does.
  function extrapolationReach(resultGapMs, minReachMs, maxReachMs) {
    return Math.min(maxReachMs, Math.max(minReachMs, 2 * (resultGapMs || 0)));
  }

  // Submitting faster than the worker drains its one-deep queue makes results
  // later, not sooner, because each inference then starts on a frame that has
  // already been waiting. Pace just under the measured round trip instead.
  function autoIntervalMs(workerCycleMs, headroom, minIntervalMs, maxIntervalMs) {
    if (!Number.isFinite(workerCycleMs) || workerCycleMs <= 0) return null;
    return Math.min(maxIntervalMs, Math.max(minIntervalMs, workerCycleMs * headroom));
  }

  // video.currentTime is quantised to frame boundaries and reads behind the
  // frame on screen, so measuring staleness against it under-reports it. The
  // presentation timestamp of the last displayed frame, carried forward, does
  // not -- until playback stalls, where the carried clock would run away from
  // the picture and the quantised reading becomes the safer one.
  function projectMediaTime(options) {
    const {
      clockMs, clockWallMs, nowWallMs, playbackRate, paused, currentTimeMs, leadMs, stallMs,
    } = options;
    if (clockMs === null || clockMs === undefined || paused) return currentTimeMs;
    const elapsed = nowWallMs - clockWallMs;
    if (!Number.isFinite(elapsed) || elapsed > stallMs) return currentTimeMs;
    return clockMs + (elapsed + leadMs) * (playbackRate || 1);
  }

  function formatDms(value, degreeDigits, decimalPlaces) {
    const absolute = Math.abs(value);
    let degrees = Math.floor(absolute);
    const minuteValue = (absolute - degrees) * 60;
    let minutes = Math.floor(minuteValue);
    let seconds = (minuteValue - minutes) * 60;
    const scale = 10 ** decimalPlaces;
    seconds = Math.round(seconds * scale) / scale;
    // Rounding seconds can land exactly on sixty, and carrying can do it again
    // to the minutes. Without both steps a coordinate reads 14°59'60.00".
    if (seconds >= 60) {
      seconds = 0;
      minutes += 1;
    }
    if (minutes >= 60) {
      minutes = 0;
      degrees += 1;
    }
    return {
      degrees,
      minutes,
      seconds,
      text:
        `${String(degrees).padStart(degreeDigits, "0")}°`
        + `${String(minutes).padStart(2, "0")}'`
        + `${seconds.toFixed(decimalPlaces).padStart(3 + decimalPlaces, "0")}"`,
    };
  }

  const api = {
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
  };

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DetectionMath = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
