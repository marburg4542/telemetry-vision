"use strict";

// Draws a render's boxes over the original video as it plays.
//
// The server tracks the video ahead of the playhead and writes each processed
// frame's boxes; nothing is drawn into the video and nothing is encoded. Here
// the boxes of the frame on screen are painted onto a canvas laid over the
// <video>, once per presented frame, so the picture keeps the source's full
// frame rate at every stride and the boxes sit on the frame they were found on.
// Between two processed frames a vehicle is placed along the line between where
// it was on either side -- the same rule the MP4 export draws with
// (render_export.boxes_at), and tested against the same cases.
//
// While the render is still being made, playback stops where the tracker has
// got to and waits, long enough that it then plays to the end without catching
// the tracker up again -- or until "play now" is pressed.
//
// The arithmetic is exported on its own and tested under node. Only the canvas
// and the <video> plumbing need a browser.
(function (root) {
  const AHEAD_S = 30; // boxes fetched past the playhead
  const KEEP_BEHIND_S = 10; // and kept behind it, for a small seek back
  const EDGE_S = 0.3; // this close to the front counts as waiting at it
  // How much has to be ready before playback starts, and before it starts again
  // after catching the front. A render slower than playback waits longer than
  // that: long enough that, once playing, the picture never catches the tracker
  // up before the end. RATE_MARGIN plans for it running a tenth slower than
  // measured so far.
  const START_BUFFER_S = 10;
  const HOLD_BUFFER_S = 15;
  const RATE_MARGIN = 0.9;
  const POLL_MS = 2000;

  // -- arithmetic -----------------------------------------------------------

  // A box as the server sends it: [x1, y1, x2, y2, confidence, class_id,
  // track_id, label], in the source video's pixels.
  const X1 = 0, Y1 = 1, X2 = 2, Y2 = 3, TRACK = 6;

  // The boxes to draw on one source frame. `frames` is the processed frame
  // indices in order; `boxes` maps each to its boxes. See render_export.boxes_at.
  function boxesAt(frame, frames, boxes, stride) {
    const at = lastAtOrBefore(frames, frame);
    if (at < 0) return [];
    const before = frames[at];
    if (before === frame) return boxes.get(before) || [];
    const after = at + 1 < frames.length ? frames[at + 1] : null;
    if (after === null || after - before > stride) {
      return frame - before < stride ? boxes.get(before) || [] : [];
    }
    const t = (frame - before) / (after - before);
    const later = new Map();
    for (const box of boxes.get(after) || []) {
      if (box[TRACK] !== null && box[TRACK] !== undefined) later.set(box[TRACK], box);
    }
    const placed = [];
    const matched = new Set();
    for (const box of boxes.get(before) || []) {
      const partner = box[TRACK] === null || box[TRACK] === undefined ? null : later.get(box[TRACK]);
      if (partner) {
        matched.add(box[TRACK]);
        const shown = (t < 0.5 ? box : partner).slice();
        for (const key of [X1, Y1, X2, Y2]) shown[key] = box[key] + (partner[key] - box[key]) * t;
        placed.push(shown);
      } else if (t < 0.5) {
        placed.push(box);
      }
    }
    if (t >= 0.5) {
      for (const box of boxes.get(after) || []) {
        if (box[TRACK] === null || box[TRACK] === undefined || !matched.has(box[TRACK])) {
          placed.push(box);
        }
      }
    }
    return placed;
  }

  function lastAtOrBefore(sorted, value) {
    let low = 0;
    let high = sorted.length - 1;
    let found = -1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (sorted[mid] <= value) {
        found = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    return found;
  }

  // The source frame a presentation time shows. Frame n is on screen from n/fps.
  function frameAt(seconds, fps) {
    return Math.max(0, Math.floor(seconds * fps + 1e-4));
  }

  // One hue per class, the same as the captures' marked-up copies
  // (annotate.class_colour: HLS lightness 0.62, saturation 0.88).
  function classColour(classId) {
    const hue = ((Number(classId) || 0) * 67 + 88) % 360;
    return `hsl(${hue}, 88%, 62%)`;
  }

  // "car #12 0.88": the order the saved MP4 and the captures print it in.
  function boxLabel(box) {
    const confidence = Number(box[4]);
    let text = String(box[7] ?? "?");
    if (box[TRACK] !== null && box[TRACK] !== undefined) text += ` #${box[TRACK]}`;
    if (Number.isFinite(confidence)) text += ` ${confidence.toFixed(2)}`;
    return text;
  }

  // Where the picture sits inside a <video> element of a given size: the
  // element letterboxes the video to its own aspect ratio.
  function fitRect(elementWidth, elementHeight, videoWidth, videoHeight) {
    if (!(videoWidth > 0 && videoHeight > 0 && elementWidth > 0 && elementHeight > 0)) return null;
    const scale = Math.min(elementWidth / videoWidth, elementHeight / videoHeight);
    const width = videoWidth * scale;
    const height = videoHeight * scale;
    return {
      x: (elementWidth - width) / 2,
      y: (elementHeight - height) / 2,
      width,
      height,
      scale,
    };
  }

  // A seek past what has been tracked lands at the front instead.
  function clampSeek(seconds, processed, complete) {
    if (complete || seconds <= processed - EDGE_S) return seconds;
    return Math.max(0, processed - 0.5);
  }

  function bufferTarget(started) {
    return started ? HOLD_BUFFER_S : START_BUFFER_S;
  }

  // How far ahead of the playhead the tracker has to be before playing, so that
  // playback then runs to the end without stopping. Playing from here takes
  // `remaining` seconds, in which a render going at `rate` adds rate x remaining;
  // the rest has to be there already. null while the rate is not known yet.
  function aheadNeeded(remaining, rate, started) {
    if (!(rate > 0)) return null;
    const floor = Math.min(remaining, bufferTarget(started));
    const planned = rate * RATE_MARGIN;
    if (planned >= 1) return floor;
    return Math.min(remaining, Math.max(floor, remaining * (1 - planned)));
  }

  function readyToPlay(ahead, needed, settled) {
    if (settled) return true;
    return needed !== null && ahead >= needed - 0.01;
  }

  // Seconds of waiting before that much is ready, at the rate the render is
  // going -- video-seconds tracked per second of waiting. null when unknown.
  function estimateWait(ahead, target, rate) {
    if (!(rate > 0)) return null;
    return Math.max(0, (target - ahead) / rate);
  }

  // How long playback will then run before it reaches the front again.
  function estimatePlayback(ahead, rate) {
    if (!(rate >= 0)) return null;
    return rate >= 1 ? Infinity : ahead / (1 - rate);
  }

  const api = {
    AHEAD_S,
    EDGE_S,
    START_BUFFER_S,
    HOLD_BUFFER_S,
    RATE_MARGIN,
    boxesAt,
    frameAt,
    classColour,
    boxLabel,
    fitRect,
    clampSeek,
    bufferTarget,
    aheadNeeded,
    readyToPlay,
    estimateWait,
    estimatePlayback,
  };

  // -- the overlay ------------------------------------------------------------

  class TrackOverlay {
    constructor(video, canvas, options) {
      this.video = video;
      this.canvas = canvas;
      this.fetchTracks = options.fetchTracks;
      this.onState = options.onState || (() => {});
      this.closed = true;
      this.renderId = null;
      this.state = null;
      this.frames = [];
      this.boxes = new Map();
      this.pollTimer = null;
      this.fetching = false;
      this.started = false;
      this.resumeAtFront = false;
      this.rateSamples = [];
      this.frameCallback = null;
      this.listeners = [
        ["timeupdate", () => this.holdAtFront()],
        ["seeking", () => this.onSeek()],
        ["seeked", () => this.draw()],
        ["play", () => this.onPlay()],
        ["pause", () => this.draw()],
        ["loadedmetadata", () => this.resize()],
      ];
      this.resizeObserver = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => this.resize())
        : null;
    }

    open(renderId, { startAt = 0, autoplay = false, durationMs = 0 } = {}) {
      this.close();
      this.closed = false;
      this.renderId = renderId;
      this.startAt = Math.max(0, startAt || 0);
      this.autoplay = autoplay;
      this.duration = Number(durationMs) / 1000 || 0;
      this.canvas.hidden = false;
      for (const [name, handler] of this.listeners) this.video.addEventListener(name, handler);
      if (this.resizeObserver) this.resizeObserver.observe(this.video);
      // Held until enough is tracked; a finished render starts at once below.
      this.video.pause();
      this.resize();
      this.watchFrames();
      this.refresh();
    }

    close() {
      this.closed = true;
      window.clearTimeout(this.pollTimer);
      this.pollTimer = null;
      for (const [name, handler] of this.listeners) this.video.removeEventListener(name, handler);
      if (this.resizeObserver) this.resizeObserver.disconnect();
      if (this.frameCallback !== null && this.video.cancelVideoFrameCallback) {
        this.video.cancelVideoFrameCallback(this.frameCallback);
      }
      this.frameCallback = null;
      this.state = null;
      this.frames = [];
      this.boxes = new Map();
      this.started = false;
      this.resumeAtFront = false;
      this.rateSamples = [];
      const context = this.canvas.getContext("2d");
      if (context) context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.canvas.hidden = true;
    }

    // -- fetching -------------------------------------------------------------

    fps() {
      return Number(this.state && this.state.source_fps) || 30;
    }

    stride() {
      return Number(this.state && this.state.stride) || 1;
    }

    processed() {
      return Number(this.state && this.state.processed_ms) / 1000 || 0;
    }

    async refresh() {
      if (this.closed) return;
      try {
        await this.fetchWindow();
        this.dropFarFrames();
        this.begin();
        this.resumeIfWaiting();
        this.report();
        this.draw();
      } catch (error) {
        if (!this.closed) this.onState({ error: error.message });
      }
      // On after the render is finished too: the boxes are fetched half a minute
      // at a time, so playing on needs the next stretch whether or not the
      // tracker is still going.
      if (!this.closed) {
        this.pollTimer = window.setTimeout(() => this.refresh(), POLL_MS);
      }
    }

    // The boxes from the last frame held (or the playhead, after a jump) to
    // AHEAD_S past the playhead. Always asked, even when none are due: the
    // answer also says how far the render has got.
    async fetchWindow() {
      if (this.fetching) return;
      this.fetching = true;
      try {
        const here = this.position();
        const fps = this.fps();
        const last = this.frames.length ? this.frames[this.frames.length - 1] : null;
        const heldTo = last === null ? null : (last + 1) / fps;
        const covered = heldTo !== null && this.frames[0] / fps <= here + 0.5 && heldTo >= here - 1;
        const from = covered ? heldTo : Math.max(0, here - 1);
        if (!covered) {
          this.frames = [];
          this.boxes = new Map();
        }
        const to = Math.max(from, here + AHEAD_S);
        const answer = await this.fetchTracks(this.renderId, from * 1000, to * 1000);
        if (this.closed) return;
        this.state = answer;
        if (!this.duration && Number(answer.duration_ms) > 0) {
          this.duration = Number(answer.duration_ms) / 1000;
        }
        this.noteRate(answer.processed_ms);
        for (const [frame, boxes] of answer.frames || []) {
          if (!this.boxes.has(frame)) this.frames.push(frame);
          this.boxes.set(frame, boxes);
        }
        this.frames.sort((a, b) => a - b);
      } finally {
        this.fetching = false;
      }
    }

    dropFarFrames() {
      const fps = this.fps();
      const floor = (this.position() - KEEP_BEHIND_S) * fps;
      if (!this.frames.length || this.frames[0] >= floor) return;
      const keep = this.frames.filter((frame) => frame >= floor);
      for (const frame of this.frames) if (frame < floor) this.boxes.delete(frame);
      this.frames = keep;
    }

    // -- drawing --------------------------------------------------------------

    // Once per frame the video presents, so a box is never a frame behind the
    // picture it belongs to; a browser without the callback gets animation
    // frames instead, which is the same thing at the display's rate.
    watchFrames() {
      if (this.closed) return;
      if (this.video.requestVideoFrameCallback) {
        this.frameCallback = this.video.requestVideoFrameCallback((_, meta) => {
          this.frameCallback = null;
          this.onFrame(meta.mediaTime);
          this.watchFrames();
        });
      } else {
        window.requestAnimationFrame(() => {
          this.onFrame(this.video.currentTime);
          this.watchFrames();
        });
      }
    }

    onFrame(seconds) {
      if (this.closed) return;
      this.holdAtFront(seconds);
      this.draw(seconds);
    }

    resize() {
      const width = this.video.clientWidth;
      const height = this.video.clientHeight;
      const ratio = window.devicePixelRatio || 1;
      this.canvas.style.left = `${this.video.offsetLeft}px`;
      this.canvas.style.top = `${this.video.offsetTop}px`;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
      this.draw();
    }

    draw(seconds = this.video.currentTime) {
      if (this.closed) return;
      const context = this.canvas.getContext("2d");
      if (!context) return;
      context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (!this.state || !this.frames.length) return;
      const ratio = window.devicePixelRatio || 1;
      const rect = fitRect(
        this.canvas.width, this.canvas.height, this.video.videoWidth, this.video.videoHeight,
      );
      if (!rect) return;
      const frame = frameAt(seconds, this.fps());
      const placed = boxesAt(frame, this.frames, this.boxes, this.stride());
      const line = Math.max(2 * ratio, rect.width / 640);
      const fontSize = Math.max(11 * ratio, rect.width / 90);
      context.lineWidth = line;
      context.font = `600 ${fontSize}px system-ui, sans-serif`;
      context.textBaseline = "top";
      for (const box of placed) {
        const x = rect.x + box[X1] * rect.scale;
        const y = rect.y + box[Y1] * rect.scale;
        const w = (box[X2] - box[X1]) * rect.scale;
        const h = (box[Y2] - box[Y1]) * rect.scale;
        const colour = classColour(box[5]);
        context.strokeStyle = colour;
        context.strokeRect(x, y, w, h);
        const text = boxLabel(box);
        const textWidth = context.measureText(text).width + 6 * ratio;
        const textHeight = fontSize + 4 * ratio;
        // Above the box, unless that would be off the top of the picture.
        const top = y - textHeight >= rect.y ? y - textHeight : y + h;
        // Pulled back in when the box is at the right edge, rather than cut off.
        const left = Math.max(rect.x, Math.min(x, rect.x + rect.width - textWidth));
        context.fillStyle = colour;
        context.fillRect(left, top, textWidth, textHeight);
        context.fillStyle = "#101010";
        context.fillText(text, left + 3 * ratio, top + 2 * ratio);
      }
    }

    // -- waiting for the tracker -------------------------------------------------

    position() {
      if (this.started) return this.video.currentTime || 0;
      return this.startAt;
    }

    complete() {
      return Boolean(this.state && this.state.complete);
    }

    // Nothing more is coming: finished, or stopped and not being continued.
    settled() {
      return Boolean(this.state && (this.state.complete || this.state.status !== "running"));
    }

    ahead() {
      return Math.max(0, this.processed() - this.position());
    }

    remaining() {
      const end = this.duration || this.processed();
      return Math.max(0, end - this.position());
    }

    needed() {
      return aheadNeeded(this.remaining(), this.rate(), this.started);
    }

    // Playback starts once enough is tracked to play to the end, at the moment
    // asked for -- or at once when asked to.
    begin(now = false) {
      if (this.started || this.closed || !this.state) return;
      if (!now && !readyToPlay(this.ahead(), this.needed(), this.settled())) return;
      this.started = true;
      const wanted = clampSeek(this.startAt, this.processed(), this.complete());
      if (Math.abs((this.video.currentTime || 0) - wanted) > 0.05) this.video.currentTime = wanted;
      if (this.autoplay || now) this.video.play().catch(() => {});
    }

    playNow() {
      if (this.closed || !this.state) return;
      if (!this.started) {
        this.begin(true);
      } else {
        this.resumeAtFront = false;
        this.video.play().catch(() => {});
      }
      this.report();
    }

    onPlay() {
      // Pressing play on the video before it was released is "play now".
      if (!this.started && this.state) {
        this.begin(true);
        this.report();
      }
    }

    // The original plays on past what has been tracked; with the boxes on it
    // stops there instead, and says why.
    holdAtFront(seconds = this.video.currentTime) {
      if (this.closed || !this.started || this.complete()) return;
      const front = this.processed();
      if (seconds < front - EDGE_S) return;
      this.resumeAtFront = this.resumeAtFront || !this.video.paused;
      this.video.pause();
      if (this.video.currentTime > front) this.video.currentTime = Math.max(0, front - 0.05);
      this.report();
    }

    resumeIfWaiting() {
      if (!this.resumeAtFront) return;
      // Stopped partway and not being continued: its front is where it ends.
      if (this.settled() && !this.complete()) return;
      if (!readyToPlay(this.ahead(), this.needed(), this.settled())) return;
      this.resumeAtFront = false;
      this.video.play().catch(() => {});
    }

    onSeek() {
      if (this.closed || !this.state) return;
      if (!this.started) {
        this.startAt = this.video.currentTime || 0;
        return;
      }
      const processed = this.processed();
      const target = this.video.currentTime || 0;
      const allowed = clampSeek(target, processed, this.complete());
      if (Math.abs(allowed - target) > 0.05) {
        this.video.currentTime = allowed;
        this.onState({ clamped: true, processedS: processed, complete: this.complete() });
        return;
      }
      window.clearTimeout(this.pollTimer);
      this.refresh();
    }

    noteRate(processedMs) {
      const processed = Number(processedMs) / 1000;
      if (!Number.isFinite(processed)) return;
      this.rateSamples.push([Date.now() / 1000, processed]);
      if (this.rateSamples.length > 30) this.rateSamples.shift();
    }

    // The server's average over the run when it has one; the last minute of
    // polls otherwise.
    rate() {
      const reported = Number(this.state && this.state.rate);
      if (reported > 0) return reported;
      if (this.rateSamples.length < 2) return 0;
      const [firstAt, firstProcessed] = this.rateSamples[0];
      const [lastAt, lastProcessed] = this.rateSamples[this.rateSamples.length - 1];
      const elapsed = lastAt - firstAt;
      return elapsed > 0 ? Math.max(0, (lastProcessed - firstProcessed) / elapsed) : 0;
    }

    report() {
      if (this.closed || !this.state) return;
      const processed = this.processed();
      const complete = this.complete();
      const settled = this.settled();
      const ahead = this.ahead();
      const rate = this.rate();
      const needed = this.needed();
      const atFront = this.started && this.position() >= processed - EDGE_S;
      const waiting = !settled && (!this.started || this.resumeAtFront || atFront);
      this.onState({
        processedS: processed,
        complete,
        started: this.started,
        waiting,
        stopped: !complete && settled && atFront,
        aheadS: ahead,
        rate,
        remainingS: this.remaining(),
        waitS: waiting && needed !== null ? estimateWait(ahead, needed, rate) : null,
        playS: waiting && needed !== null ? estimatePlayback(needed, rate) : null,
      });
    }
  }

  api.create = (video, canvas, options) => new TrackOverlay(video, canvas, options);

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TelemetryTrackOverlay = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
