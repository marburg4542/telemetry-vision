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
  function staleRanges(segments, seconds, behind = BEHIND_S, ahead = AHEAD_S) {
    const ranges = [];
    const last = processedUntil(segments);
    if (seconds - behind > 0) ranges.push([0, seconds - behind]);
    if (last > seconds + ahead + 10) ranges.push([seconds + ahead + 10, last + 60]);
    return ranges;
  }

  // A seek past what has been rendered lands at the front instead: the timeline
  // is the whole video's, so the bar can be dragged into what does not exist yet.
  function clampSeek(seconds, processed, complete) {
    if (complete || seconds <= processed - EDGE_S) return seconds;
    return Math.max(0, processed - 0.5);
  }

  const api = {
    AHEAD_S,
    BEHIND_S,
    EDGE_S,
    flattenSegments,
    processedUntil,
    segmentAt,
    nextSegment,
    staleRanges,
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
      this.listeners = [
        ["timeupdate", () => {
          this.holdAtFront();
          this.pump();
        }],
        ["waiting", () => this.pump()],
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
          const segment = nextSegment(this.segments, seconds, this.loaded);
          if (!segment) break;
          const appended = await this.appendSegment(segment);
          if (!appended) break;
          this.begin();
          this.resumeIfWaiting();
        }
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
        for (const segment of this.segments) {
          if (segment.start >= from && segment.end <= to) this.loaded.delete(segment.key);
        }
      }
    }

    // Playback starts once there is something to show, at the moment asked for.
    begin() {
      if (this.started || this.closed) return;
      this.started = true;
      const wanted = clampSeek(this.startAt, processedUntil(this.segments), false);
      if (wanted > 0.05) this.video.currentTime = wanted;
      if (this.autoplay) this.video.play().catch(() => {});
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

    // Chrome runs the clock on past the end of what is buffered instead of
    // stopping there, so the picture freezes at the frame the model has reached
    // while the time reads on to the end of the video. Playback is held at the
    // front instead, and let go again when the next segments arrive.
    holdAtFront() {
      if (this.closed || !this.started || (this.index && this.index.complete)) return;
      const front = Math.max(0, processedUntil(this.segments) - 0.05);
      if (this.video.currentTime <= front + EDGE_S) return;
      this.resumeAtFront = this.resumeAtFront || !this.video.paused;
      this.video.pause();
      this.video.currentTime = front;
    }

    resumeIfWaiting() {
      if (!this.resumeAtFront) return;
      if (processedUntil(this.segments) < this.video.currentTime + 0.5) return;
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
      this.onState({
        processedS: processed,
        complete,
        // At the front with more still to come: the picture stops here until
        // the renderer gets further, which is a wait, not a fault.
        waiting: !complete && this.started && this.position() >= processed - EDGE_S,
      });
    }
  }

  api.create = (video, options) => new RenderPlayer(video, options);

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TelemetryRenderPlayer = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
