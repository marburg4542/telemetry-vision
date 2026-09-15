"use strict";

// Display formatting, kept free of the DOM and the page's state so it can be
// exercised directly. What lives here is arithmetic that fails silently: a
// coordinate that rounds its seconds up to sixty reads 14°59'60.00" and nothing
// on the page says anything is wrong.
//
// Loaded as a plain script before app.js, and required directly by the tests.
(function (root) {
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

  const api = { formatDms };

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TelemetryFormat = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
