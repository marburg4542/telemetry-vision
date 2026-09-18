"use strict";

// Plays a render that may still be being made.
//
// Chrome and Edge both refuse an HLS playlist with no #EXT-X-ENDLIST -- the
// demuxer fails to parse it outright -- so a render in progress cannot be handed
// to <video src> and waited on. Its segments are ordinary fragmented MP4 though,
// so they are fetched and fed to Media Source as the server finishes them:
// playback stops where the processing has got to and carries on by itself when
// the next segment lands, and the operator watches the model work rather than
// waiting twice the length of the video to see anything at all.
//
// The arithmetic -- which segment holds a moment, what to fetch next, what to
// drop, where a seek may land -- is exported on its own and tested under node.
// Only the Media Source plumbing needs a browser.
(function (root) {
  const AHEAD_S = 30; // fetched past the playhead
  const BEHIND_S = 60; // kept behind it; a 25-minute render will not fit whole
  const EDGE_S = 0.3; // this close to the front counts as waiting at it
  // How much has to be ready before playback starts, and before it starts again
  // after catching the front. The renderer is slower than playback -- a heavy
  // model at every frame renders about six frames a second against the thirty a
  // second the picture is played at -- so the choice is not whether to wait but
  // whether to wait a moment every two seconds or once for a stretch that then
  // plays without stopping. These are the second.
  //
  // A render slower than playback waits longer than that: long enough that,
  // once playing, the picture never catches the renderer up before the end.
  // RATE_MARGIN plans for it running a tenth slower than measured so far.
  const START_BUFFER_S = 10;
  const HOLD_BUFFER_S = 15;
  const RATE_MARGIN = 0.9;
  const POLL_MS = 2000;
  const MIME = 'video/mp4; codecs="avc1.640029"';
  const INIT_FILE = "init.mp4";

  // -- arithmetic -----------------------------------------------------------

  // Every finished segment, in order, with its place on the source video's
  // timeline. A run is one pass of the renderer: the timestamps inside its
  // segments start at zero, so playing them needs the run's own start too.
  function flattenSegments(index) {
    const flat = [];
    for (const run of (index && index.runs) || []) {
      for (const segment of run.segments || []) {
        const start = Number(segment.start_ms) / 1000;
        const duration = Number(segment.duration_ms) / 1000;
        if (!Number.isFinite(start) || !Number.isFinite(duration) || duration <= 0) continue;
        flat.push({
          key: `${run.run}/${segment.file}`,
          run: run.run,
          file: segment.file,
          runStart: Number(run.start_ms) / 1000 || 0,
          start,
          end: start + duration,
        });
      }
    }
    flat.sort((a, b) => a.start - b.start);
    return flat;
  }

  function processedUntil(segments) {
    return segments.length ? segments[segments.length - 1].end : 0;
  }

  function segmentAt(segments, seconds) {
    return segments.find((item) => seconds >= item.start - 0.001 && seconds < item.end) || null;
  }

  // The next segment to fetch: the earliest one from the playhead onwards that
  // is not loaded yet and starts within the look-ahead. Null when none is due,
  // which is either everything fetched or the playhead at the front.
  function nextSegment(segments, seconds, loaded, ahead = AHEAD_S) {
    return (
      segments.find(
        (item) =>
          item.end > seconds - 0.001
          && item.start <= seconds + ahead
          && !loaded.has(item.key),
      ) || null
    );
  }

  // What to drop. Media Source is given a budget of a couple of hundred
  // megabytes, and a render is 55 MB a minute, so what is far from the playhead
  // has to go or appending stops with QuotaExceededError partway through.
  //
  // Only whole segments go. Removing from the middle of one takes the rest of
  // it too -- everything after the keyframe depends on it -- so a cut at 37.3
  // empties 36-38 while the segment still counts as fetched, and a seek back
  // there waits on a hole forever.
  function staleRanges(segments, seconds, behind = BEHIND_S, ahead = AHEAD_S) {
    const ranges = [];
    const last = processedUntil(segments);
    const before = segments.filter((item) => item.end <= seconds - behind);
    if (before.length) ranges.push([0, before[before.length - 1].end]);
    const after = segments.find((item) => item.start >= seconds + ahead + 10);
    if (after) ranges.push([after.start, last + 60]);
    return ranges;
  }

  // The fetched segments Media Source no longer holds in full: dropped by the
  // browser under memory pressure, or cut by a removal that ran over. They are
  // fetched again rather than waited on.
  function missingSegments(segments, loaded, buffered, tolerance = 0.25) {
    return segments.filter(
      (item) =>
        loaded.has(item.key)
        && !buffered.some(
          ([from, to]) => from <= item.start + tolerance && to >= item.end - tolerance,
        ),
    );
  }

  // A seek past what has been rendered lands at the front instead: the timeline
  // is the whole video's, so the bar can be dragged into what does not exist yet.
  function clampSeek(seconds, processed, complete) {
    if (complete || seconds <= processed - EDGE_S) return seconds;
    return Math.max(0, processed - 0.5);
  }

  // Playback waits for this much to be ready: less at the start, where the
  // first picture matters more, than when it catches up mid-video.
  function bufferTarget(started) {
    return started ? HOLD_BUFFER_S : START_BUFFER_S;
  }

  // How far ahead of the playhead the render has to be before playing, so that
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
  // going -- video-seconds produced per second of waiting. null when unknown.
  function estimateWait(ahead, target, rate) {
    if (!(rate > 0)) return null;
    return Math.max(0, (target - ahead) / rate);
  }

  // How long playback will then run before it reaches the front again. A render
  // producing faster than one video-second per second never does.
  function estimatePlayback(ahead, rate) {
    if (!(rate >= 0)) return null;
    return rate >= 1 ? Infinity : ahead / (1 - rate);
  }

  const api = {
    AHEAD_S,
    BEHIND_S,
    EDGE_S,
    START_BUFFER_S,
    HOLD_BUFFER_S,
    RATE_MARGIN,
    bufferTarget,
    aheadNeeded,
    readyToPlay,
    estimateWait,
    estimatePlayback,
    flattenSegments,
    processedUntil,
    segmentAt,
    nextSegment,
    staleRanges,
    missingSegments,
    clampSeek,
  };

  // -- the player -----------------------------------------------------------

  class RenderPlayer {
    constructor(video, options) {
      this.video = video;
      this.fetchIndex = options.fetchIndex;
      this.segmentUrl = options.segmentUrl;
      this.onState = options.onState || (() => {});
      this.closed = true;
      this.renderId = null;
      this.index = null;
      this.segments = [];
      this.loaded = new Set();
      this.source = null;
      this.buffer = null;
      this.objectUrl = "";
      this.initRun = null;
      this.pollTimer = null;
      this.pumping = false;
      this.started = false;
      this.quotaKey = "";
      this.resumeAtFront = false;
      this.rateSamples = [];
      this.listeners = [
        ["timeupdate", () => {
          this.holdAtFront();
          this.pump();
        }],
        ["waiting", () => {
          this.holdAtFront();
          this.pump();
        }],
        ["seeking", () => this.onSeek()],
      ];
    }

    open(renderId, { startAt = 0, autoplay = false, durationMs = 0 } = {}) {
      this.close();
      this.closed = false;
      this.renderId = renderId;
      this.startAt = Math.max(0, startAt || 0);
      this.autoplay = autoplay;
      this.duration = Number(durationMs) / 1000 || 0;
      this.source = new MediaSource();
      this.objectUrl = URL.createObjectURL(this.source);
      this.source.addEventListener("sourceopen", () => this.onSourceOpen(), { once: true });
      for (const [name, handler] of this.listeners) this.video.addEventListener(name, handler);
      this.video.src = this.objectUrl;
    }

    close() {
      this.closed = true;
      window.clearTimeout(this.pollTimer);
      this.pollTimer = null;
      for (const [name, handler] of this.listeners) this.video.removeEventListener(name, handler);
      if (this.objectUrl) {
        URL.revokeObjectURL(this.objectUrl);
        if (this.video.src === this.objectUrl) {
          this.video.removeAttribute("src");
          this.video.load();
        }
      }
      this.objectUrl = "";
      this.source = null;
      this.buffer = null;
      this.index = null;
      this.segments = [];
      this.loaded = new Set();
      this.initRun = null;
      this.started = false;
      this.resumeAtFront = false;
      this.rateSamples = [];
    }

    onSourceOpen() {
      if (this.closed || !this.source) return;
      try {
        this.buffer = this.source.addSourceBuffer(MIME);
        this.buffer.mode = "segments";
      } catch (error) {
        this.onState({ error: `เบราว์เซอร์นี้เล่นวิดีโอ AI ไม่ได้: ${error.message}` });
        return;
      }
      this.setDuration(this.duration);
      this.refresh();
    }

    // The timeline is the source video's, so switching between the original and
    // the render keeps the moment even where the render has not reached.
    setDuration(seconds) {
      if (!this.source || this.source.readyState !== "open" || !(seconds > 0)) return;
      try {
        if (this.source.duration !== seconds) this.source.duration = seconds;
      } catch (error) {
        // Set again after the first append, when it is allowed.
      }
    }

    async refresh() {
      if (this.closed) return;
      try {
        const index = await this.fetchIndex(this.renderId);
        if (this.closed) return;
        this.index = index;
        this.segments = flattenSegments(index);
        this.noteRate(index.processed_ms);
        if (!this.duration && Number(index.duration_ms) > 0) {
          this.duration = Number(index.duration_ms) / 1000;
        }
        this.setDuration(this.duration);
        await this.pump();
      } catch (error) {
        if (!this.closed) this.onState({ error: error.message });
      }
      if (!this.closed && !(this.index && this.index.complete)) {
        this.pollTimer = window.setTimeout(() => this.refresh(), POLL_MS);
      }
    }

    async pump() {
      if (this.pumping || this.closed || !this.buffer) return;
      this.pumping = true;
      try {
        while (!this.closed) {
          const seconds = this.position();
          await this.evict(seconds);
          this.forgetMissing();
          const segment = nextSegment(this.segments, seconds, this.loaded);
          if (!segment) break;
          const appended = await this.appendSegment(segment);
          if (!appended) break;
          this.begin();
          this.resumeIfWaiting();
        }
        // Waiting minutes for a slow render, nothing new is fetched for most of
        // them -- the look-ahead is already full -- so readiness is checked on
        // every poll, not only when a segment arrives.
        this.begin();
        this.resumeIfWaiting();
        this.finish();
      } catch (error) {
        if (!this.closed) this.onState({ error: error.message });
      } finally {
        this.pumping = false;
        this.report();
      }
    }

    position() {
      if (this.started) return this.video.currentTime || 0;
      // Before playback has been placed, fetch around where it will start.
      return this.startAt;
    }

    async appendSegment(segment) {
      if (segment.run !== this.initRun) {
        const init = await this.fetchBytes(this.segmentUrl(this.renderId, segment.run, INIT_FILE));
        if (!(await this.append(init, null))) return false;
        this.initRun = segment.run;
      }
      const url = this.segmentUrl(this.renderId, segment.run, segment.file);
      const appended = await this.append(await this.fetchBytes(url), segment.runStart);
      if (appended) {
        this.loaded.add(segment.key);
        this.quotaKey = "";
        return true;
      }
      // Out of room: what was dropped makes space, unless dropping it did not
      // help, in which case stop rather than fetch the same segment forever.
      if (this.quotaKey === segment.key) return false;
      this.quotaKey = segment.key;
      await this.evict(this.position(), true);
      return false;
    }

    async fetchBytes(url) {
      const response = await fetch(url, { headers: { "X-Telemetry-OCR": "1" } });
      if (!response.ok) throw new Error(`อ่านวิดีโอ AI ไม่ได้ (${response.status})`);
      return response.arrayBuffer();
    }

    async append(bytes, timestampOffset) {
      if (this.closed || !this.buffer) return false;
      await this.idle();
      if (this.closed || !this.buffer) return false;
      try {
        if (timestampOffset !== null && this.buffer.timestampOffset !== timestampOffset) {
          this.buffer.timestampOffset = timestampOffset;
        }
        this.buffer.appendBuffer(bytes);
      } catch (error) {
        if (error.name === "QuotaExceededError") return false;
        throw error;
      }
      await this.idle();
      return true;
    }

    idle() {
      return new Promise((resolve) => {
        if (!this.buffer || !this.buffer.updating) {
          resolve();
          return;
        }
        this.buffer.addEventListener("updateend", () => resolve(), { once: true });
        this.buffer.addEventListener("error", () => resolve(), { once: true });
      });
    }

    async evict(seconds, hard = false) {
      if (!this.buffer || !this.segments.length) return;
      const ranges = hard
        ? [[0, Math.max(0, seconds - 10)], [seconds + 10, processedUntil(this.segments) + 60]]
        : staleRanges(this.segments, seconds);
      for (const [from, to] of ranges) {
        if (to - from < 1) continue;
        await this.idle();
        if (this.closed || !this.buffer) return;
        try {
          this.buffer.remove(from, to);
        } catch (error) {
          return;
        }
        await this.idle();
        // Touched at all is gone: see staleRanges.
        for (const segment of this.segments) {
          if (segment.end > from && segment.start < to) this.loaded.delete(segment.key);
        }
      }
    }

    // What the buffer really holds decides what is fetched, not what was
    // appended once.
    forgetMissing() {
      if (!this.buffer || this.buffer.updating) return;
      let buffered;
      try {
        buffered = [];
        const ranges = this.buffer.buffered;
        for (let i = 0; i < ranges.length; i += 1) buffered.push([ranges.start(i), ranges.end(i)]);
      } catch (error) {
        return; // the source buffer was removed from a source being closed
      }
      for (const segment of missingSegments(this.segments, this.loaded, buffered)) {
        this.loaded.delete(segment.key);
      }
    }

    // Playback starts once enough is ready to play to the end, at the moment
    // asked for -- or at once when asked to. A finished render, or one that is
    // not being worked on, is as ready as it will get.
    begin(now = false) {
      if (this.started || this.closed || !this.segments.length) return;
      if (!now && !readyToPlay(this.ahead(), this.needed(), this.settled())) return;
      this.started = true;
      const wanted = clampSeek(this.startAt, processedUntil(this.segments), false);
      if (wanted > 0.05) this.video.currentTime = wanted;
      if (this.autoplay || now) this.video.play().catch(() => {});
    }

    // "Play now": what is there plays, and the wait comes back at the front.
    playNow() {
      if (this.closed || !this.segments.length) return;
      if (!this.started) {
        this.begin(true);
      } else {
        this.resumeAtFront = false;
        this.video.play().catch(() => {});
      }
      this.report();
    }

    settled() {
      return Boolean(this.index && (this.index.complete || this.index.status !== "running"));
    }

    remaining() {
      const end = this.duration || processedUntil(this.segments);
      return Math.max(0, end - this.position());
    }

    needed() {
      return aheadNeeded(this.remaining(), this.rate(), this.started);
    }

    finish() {
      const last = this.segments[this.segments.length - 1];
      if (!this.index || !this.index.complete || !last || !this.loaded.has(last.key)) return;
      if (!this.source || this.source.readyState !== "open" || this.buffer.updating) return;
      try {
        this.source.endOfStream();
      } catch (error) {
        // Another append is in flight; the next pump ends it instead.
      }
    }

    // How fast the render is producing video, in video-seconds per second of
    // waiting, from the last few polls rather than from the whole run: a render
    // that was competing with something else for the GPU should not hold that
    // against its estimate for the next wait.
    noteRate(processedMs) {
      const processed = Number(processedMs) / 1000;
      if (!Number.isFinite(processed)) return;
      this.rateSamples.push([Date.now() / 1000, processed]);
      if (this.rateSamples.length > 30) this.rateSamples.shift();
    }

    // The server's average over the run when it has one; the last minute of
    // polls otherwise.
    rate() {
      const reported = Number(this.index && this.index.rate);
      if (reported > 0) return reported;
      if (this.rateSamples.length < 2) return 0;
      const [firstAt, firstProcessed] = this.rateSamples[0];
      const [lastAt, lastProcessed] = this.rateSamples[this.rateSamples.length - 1];
      const elapsed = lastAt - firstAt;
      return elapsed > 0 ? Math.max(0, (lastProcessed - firstProcessed) / elapsed) : 0;
    }

    ahead() {
      const from = this.started ? this.video.currentTime || 0 : this.startAt;
      return Math.max(0, processedUntil(this.segments) - from);
    }

    // Stop when the playhead reaches the frame the renderer has got to.
    //
    // Left alone, Chrome does one of two unhelpful things there: it plays each
    // new segment the instant it lands, which is a second of video every three
    // seconds of watching, or it runs the clock on past the end of the buffer
    // while the picture stands still. Stopping deliberately and letting a
    // stretch build up plays that stretch in one piece instead.
    holdAtFront() {
      if (this.closed || !this.started || (this.index && this.index.complete)) return;
      const front = processedUntil(this.segments);
      if (this.video.currentTime < front - EDGE_S) return;
      this.resumeAtFront = this.resumeAtFront || !this.video.paused;
      this.video.pause();
      if (this.video.currentTime > front) {
        this.video.currentTime = Math.max(0, front - 0.05);
      }
    }

    resumeIfWaiting() {
      if (!this.resumeAtFront) return;
      // Stopped partway and not being continued: its front is where it ends,
      // so resuming there would only replay the last half-second over and over.
      const complete = Boolean(this.index && this.index.complete);
      if (this.settled() && !complete) return;
      if (!readyToPlay(this.ahead(), this.needed(), this.settled())) return;
      this.resumeAtFront = false;
      this.video.play().catch(() => {});
    }

    onSeek() {
      if (this.closed || !this.started) return;
      const processed = processedUntil(this.segments);
      const complete = Boolean(this.index && this.index.complete);
      const target = this.video.currentTime || 0;
      const allowed = clampSeek(target, processed, complete);
      if (Math.abs(allowed - target) > 0.05) {
        this.video.currentTime = allowed;
        this.onState({ clamped: true, processedS: processed, complete });
        return;
      }
      this.pump();
    }

    report() {
      if (this.closed) return;
      const processed = processedUntil(this.segments);
      const complete = Boolean(this.index && this.index.complete);
      const settled = this.settled();
      const ahead = this.ahead();
      const rate = this.rate();
      const needed = this.needed();
      const atFront = this.started && this.position() >= processed - EDGE_S;
      // Waiting means the picture is stopped on purpose: it has not started
      // because too little is ready, or it caught the renderer up. A render
      // nobody is working on just ends where it got to.
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

  api.create = (video, options) => new RenderPlayer(video, options);

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TelemetryRenderPlayer = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
