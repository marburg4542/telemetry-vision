"use strict";

const $ = (id) => document.getElementById(id);

const ui = {
  videoInput: $("videoInput"),
  chooseVideoButton: $("chooseVideoButton"),
  videoStage: $("videoStage"),
  video: $("sourceVideo"),
  streamImage: $("streamImage"),
  streamUrl: $("streamUrl"),
  streamButton: $("streamButton"),
  streamBadge: $("streamBadge"),
  captureButton: $("captureButton"),
  captureButtonText: $("captureButtonText"),
  captureStatus: $("captureStatus"),
  captureFlash: $("captureFlash"),
  lineSendButton: $("lineSendButton"),
  lineSendButtonText: $("lineSendButtonText"),
  lineQuotaLabel: $("lineQuotaLabel"),
  streamBadgeText: $("streamBadgeText"),
  dropState: $("dropState"),
  detectionOverlay: $("detectionOverlay"),
  detectionCanvas: $("detectionCaptureCanvas"),
  modelInput: $("modelInput"),
  detectionToggle: $("detectionToggle"),
  detectionRate: $("detectionRate"),
  modelConfidence: $("modelConfidence"),
  modelImageSize: $("modelImageSize"),
  modelName: $("modelName"),
  modelStatus: $("modelStatus"),
  detectionFps: $("detectionFps"),
  detectionLatency: $("detectionLatency"),
  trackIdToggle: $("trackIdToggle"),
  detectionLag: $("detectionLag"),
  detectionCapture: $("detectionCapture"),
  detectionHarvest: $("detectionHarvest"),
  detectionDropped: $("detectionDropped"),
  restartWorkerButton: $("restartWorkerButton"),
  clearModelButton: $("clearModelButton"),
  fileName: $("fileName"),
  fileDetails: $("fileDetails"),
  currentTime: $("currentTime"),
  durationTime: $("durationTime"),
  installAppButton: $("installAppButton"),
  resetButton: $("resetButton"),
  engineStatus: $("engineStatus"),
  engineStatusText: $("engineStatusText"),
  telemetryList: $("telemetryList"),
  recordCount: $("recordCount"),
  shotStrip: $("shotStrip"),
  shotProgress: $("shotProgress"),
  shotTotal: $("shotTotal"),
  shotWorking: $("shotWorking"),
  shotWorkingText: $("shotWorkingText"),
  shotReady: $("shotReady"),
  shotUnprocessed: $("shotUnprocessed"),
  shotFailed: $("shotFailed"),
  shotSelectBar: $("shotSelectBar"),
  shotSelectedCount: $("shotSelectedCount"),
  shotClearSelection: $("shotClearSelection"),
  shotDeleteSelection: $("shotDeleteSelection"),
  shotOverlay: $("shotOverlay"),
  shotOverlayImage: $("shotOverlayImage"),
  shotOverlayLabel: $("shotOverlayLabel"),
  shotOverlayClose: $("shotOverlayClose"),
  shotPreview: $("shotPreview"),
  shotImage: $("shotImage"),
  shotAnnotated: $("shotAnnotated"),
  shotDetections: $("shotDetections"),
  downloadJsonButton: $("downloadJsonButton"),
  downloadCsvButton: $("downloadCsvButton"),
  clearLogButton: $("clearLogButton"),
  canvas: $("captureCanvas"),
  toast: $("toast"),
};

const TELEMETRY_GROUPS = [
  { id: "sensor", title: "EO / IR" },
  { id: "camera", title: "CAMERA" },
  { id: "tag", title: "TAG" },
  { id: "acft", title: "ACFT" },
];

const FIELD_META = {
  eo_detail: { label: "EO", unit: "", group: "sensor" },
  ir_detail: { label: "IR", unit: "", group: "sensor" },
  lrf_m: { label: "LRF", unit: "m", group: "sensor" },
  sensor_angle_deg: { label: "Camera yaw left/right", unit: "°", group: "camera" },
  elevation_deg: { label: "Camera pitch up/down", unit: "°", group: "camera" },
  tag_lat_dms: { label: "N", unit: "", group: "tag" },
  tag_lon_dms: { label: "E", unit: "", group: "tag" },
  tag_alt_m: { label: "ALT", unit: "m", group: "tag" },
  acft_lat_dms: { label: "N", unit: "", group: "acft" },
  acft_lon_dms: { label: "E", unit: "", group: "acft" },
  acft_alt_m: { label: "ALT", unit: "m", group: "acft" },
};

const state = {
  config: null,
  sessionId: null,
  objectUrl: null,
  file: null,
  generation: 0,
  records: [],
  rowElements: new Map(),
  loopStarted: false,
  fallbackTimer: null,
  toastTimer: null,
  installPrompt: null,
  detectorLoaded: false,
  detectionInFlight: false,
  detectionGeneration: 0,
  detectionModelGeneration: 0,
  detectionSequence: 0,
  detectionSourceId: "",
  lastDetectionMs: -Infinity,
  detectionTracks: new Map(),
  detectionResultSequence: -1,
  detectionCompletions: [],
  detectionPollTimer: null,
  detectionPollToken: 0,
  detectionRevision: 0,
  detectionModelPollTimer: null,
  detectionRenderFrame: null,
  detectionSourceWidth: 0,
  detectionSourceHeight: 0,
  detectionCaptureMs: null,
  detectionPostMs: null,
  detectionTrainedImageSize: null,
  detectionDevice: "",
  detectionImageSizeTouched: false,
  detectionEncoder: undefined,
  detectionEncodeJobs: new Map(),
  detectionEncodeSequence: 0,
  detectionCameraVelocity: null,
  detectionResultGapMs: 0,
  detectionLastResultMs: null,
  mediaClockMs: null,
  mediaClockWall: 0,
  // Stream mode: the server holds the video and this page is a viewer. The
  // clock is the server's elapsed source time, carried forward locally between
  // updates the same way the video clock is between presented frames.
  streamMode: false,
  streamRunning: false,
  streamRevision: 0,
  streamPollToken: 0,
  streamClockMs: null,
  streamClockWall: 0,
  streamRenderFrame: null,
  // Capture: one session per page load unless the operator is holding the
  // button down, in which case shots keep appending to the same one.
  captureSession: null,
  captureInFlight: false,
  captureCount: 0,
  captureHolding: false,
  captureTimer: null,
  lineConfigured: false,
  lineTo: "",
  lineHow: "",
  // What LINE says is left of the month's allowance, or null when it could not
  // be read. Refreshed on load and after every send.
  lineQuota: null,
  lineSending: false,
  deleting: false,
  // The captured shots this session holds, and which one the panel is showing.
  shots: [],
  selectedShot: null,
  // Ticked for sending. A set of frame indexes rather than an array so that
  // ticking the same thumbnail twice cannot put it in the message twice.
  chosen: new Set(),
  // Captures the server has been asked for and has not answered yet. They get a
  // placeholder in the strip so a press is never a press with nothing visible
  // behind it -- the frame is saved, run through best.pt and re-read before the
  // response comes back, which takes about six tenths of a second and can take
  // far longer on a cold model.
  pendingCaptures: 0,
  captureQueued: 0,
  captureFailures: 0,
  detectionWorkerCycleMs: null,
};

// Compared against the server's version at startup. A service worker serving a
// stale app.js looks exactly like a bug in the current code, so the page says
// which build it is actually running rather than leaving it to be deduced.
const APP_BUILD = "0.5.0";

const DETECTION_MAX_EXTRAPOLATION_MS = 600;
const DETECTION_MIN_EXTRAPOLATION_MS = 250;
const DETECTION_MIN_SAMPLE_GAP_MS = 30;
// Must stay above DETECTION_AUTO_MAX_INTERVAL_MS. When the two were equal, auto
// pacing at its ceiling put every sample gap exactly on this boundary, so the
// per-box velocity was thrown away on precisely the slow machines that need it.
const DETECTION_MAX_SAMPLE_GAP_MS = 600;
// Weight of each new velocity sample. Detection boxes jitter by a few pixels
// between frames; without smoothing that jitter is divided by the gap and
// multiplied by the extrapolation window, which throws the box around.
const DETECTION_VELOCITY_SMOOTHING = 0.45;
// Source pixels per millisecond. Nothing real crosses a 640 px frame in under
// half a second, so anything faster is a tracker mistake, not motion.
const DETECTION_MAX_SPEED = 1.5;
// What is drawn this frame is shown at the next repaint, one 60 Hz interval
// later, by which time the video has moved on again.
const DETECTION_PRESENTATION_LEAD_MS = 16;
// No new video frame for this long means stalled, seeking or a background
// tab, where carrying the clock forward would run away from the picture.
const DETECTION_STALL_MS = 250;
// The camera estimate is a median across every box, so it is already denoised
// spatially and can follow a changing pan faster than a single box's velocity
// can be trusted to.
const DETECTION_CAMERA_SMOOTHING = 0.7;
// Auto pacing bounds. The headroom keeps the queue empty without letting the
// gap between results grow enough to starve the tracker.
const DETECTION_AUTO_HEADROOM = 1.1;
const DETECTION_AUTO_MIN_INTERVAL_MS = 1000 / 20;
const DETECTION_AUTO_MAX_INTERVAL_MS = 400;
// The server holds a poll open until it has something new, so these are only
// the floor and the retry gap, never the normal cadence.
const DETECTION_POLL_FLOOR_MS = 16;
const DETECTION_POLL_BACKOFF_MS = 500;
// Generous: a slow machine encoding 960 px still finishes far inside this.
// It exists to break a wedge, not to police normal timing.
const DETECTION_ENCODE_TIMEOUT_MS = 4000;
// Mirrors the bounds the frame endpoint enforces.
const DETECTION_MIN_IMAGE_SIZE = 320;
const DETECTION_MAX_IMAGE_SIZE = 1920;
const DETECTION_STALE_MS = 750;
const DETECTION_FADE_MS = 150;

async function verifyBuildMatchesServer() {
  try {
    const health = await jsonFetch("/api/health");
    const serverBuild = String(health.version || "");
    if (serverBuild && serverBuild !== APP_BUILD) {
      showToast(
        `หน้าเว็บเป็นเวอร์ชัน ${APP_BUILD} แต่ server เป็น ${serverBuild}`
        + " — กด Ctrl+Shift+R เพื่อโหลดโค้ดใหม่",
      );
      console.warn(`Stale page: app ${APP_BUILD} against server ${serverBuild}`);
    }
  } catch (error) {
    console.warn("Could not verify the page build", error);
  }
}

function initializePwa() {
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/service-worker.js").catch((error) => {
        console.warn("Service worker registration failed", error);
      });
    });
  }
  verifyBuildMatchesServer();
  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    state.installPrompt = event;
    ui.installAppButton.hidden = false;
  });
  window.addEventListener("appinstalled", () => {
    state.installPrompt = null;
    ui.installAppButton.hidden = true;
    showToast("ติดตั้ง Telemetry OCR แล้ว");
  });
  ui.installAppButton.addEventListener("click", async () => {
    if (!state.installPrompt) return;
    const prompt = state.installPrompt;
    state.installPrompt = null;
    ui.installAppButton.hidden = true;
    await prompt.prompt();
    await prompt.userChoice;
  });
}

function setEngineStatus(kind, text) {
  ui.engineStatus.classList.remove("is-loading", "is-ready", "is-error");
  ui.engineStatus.classList.add(`is-${kind}`);
  ui.engineStatusText.textContent = text;
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  ui.toast.textContent = message;
  ui.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => { ui.toast.hidden = true; }, 5000);
}

// The server refuses mutating /api/ calls without this header, so that a
// cross-origin request can never reach a handler as a CORS "simple" request.
const API_GUARD_HEADER = "X-Telemetry-OCR";

async function jsonFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set(API_GUARD_HEADER, "1");
  const response = await fetch(url, { ...options, headers });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* response may be empty */ }
  if (!response.ok) {
    const error = new Error(payload.error || `HTTP ${response.status}`);
    // Carried along rather than thrown away: a refusal says more than its
    // message. A 429 from the quota guard brings the quota that caused it, and
    // the button needs that to stop offering a send that cannot happen.
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setModelStatus(text, isError = false) {
  ui.modelStatus.textContent = text;
  ui.modelStatus.classList.toggle("is-error", isError);
}

async function initializeDetectionModel() {
  try {
    const payload = await jsonFetch("/api/detection/model");
    const wasReady = state.detectorLoaded;
    applyDetectionModelState(payload);
    if (!wasReady && state.detectorLoaded && ui.video.readyState >= 2 && ui.detectionToggle.checked) {
      await resetDetectionTracker();
      submitDetectionFrame(true);
    }
  } catch (error) {
    setModelStatus(`ระบบ AI annotation ไม่พร้อม: ${error.message}`, true);
  }
}

function applyDetectionModelState(payload) {
  const workerStatus = payload.status || (payload.loaded ? "ready" : "unloaded");
  state.detectorLoaded = workerStatus === "ready";
  if (state.detectorLoaded) startDetectionResultPolling();
  else stopDetectionResultPolling();
  const hasModel = Boolean(payload.filename);
  ui.detectionToggle.disabled = !state.detectorLoaded;
  ui.detectionRate.disabled = !state.detectorLoaded;
  ui.modelConfidence.disabled = !state.detectorLoaded;
  ui.modelImageSize.disabled = !state.detectorLoaded;
  ui.trackIdToggle.disabled = !state.detectorLoaded;
  ui.clearModelButton.disabled = !hasModel;
  ui.restartWorkerButton.hidden = !hasModel || !["error", "stopped"].includes(workerStatus);
  ui.modelName.textContent = hasModel ? payload.filename : "ยังไม่ได้เลือกโมเดล";
  ui.detectionDropped.textContent = String(payload.dropped_frames || 0);

  if (workerStatus === "ready") {
    stopModelStatusPolling();
    state.detectionTrainedImageSize = Number(payload.trained_image_size) || null;
    state.detectionDevice = String(payload.device || "");
    adoptTrainedImageSize();
    const classes = Array.isArray(payload.classes) ? payload.classes.join(", ") : "";
    const training = [
      payload.architecture,
      payload.epochs ? `${payload.epochs} epochs` : "",
      payload.trained_image_size ? `trained ${payload.trained_image_size}px` : "",
    ].filter(Boolean).join(" · ");
    const mismatch = imageSizeMismatchWarning();
    setModelStatus(
      `BoT-SORT พร้อมใช้งาน · ${classes || "โหลด class labels แล้ว"}${training ? ` · ${training}` : ""}`
      + (mismatch ? ` · ${mismatch}` : ""),
      Boolean(mismatch),
    );
  } else if (workerStatus === "loading") {
    // Spawning the worker takes seconds. Without this the boxes from the
    // previous model stay on screen the whole time, over a video that has moved
    // on, attributed to a model that is no longer loaded.
    resetDetectionFrame();
    setModelStatus(`กำลังเริ่ม Detection worker สำหรับ ${payload.filename || "โมเดล"}…`);
    scheduleModelStatusPolling();
  } else if (workerStatus === "error") {
    stopModelStatusPolling();
    setModelStatus(`Detection worker หยุดทำงาน: ${payload.error || "ไม่ทราบสาเหตุ"}`, true);
    resetDetectionFrame();
  } else if (workerStatus === "stopped" && hasModel) {
    stopModelStatusPolling();
    setModelStatus("Detection worker หยุดอยู่ กด Restart worker เพื่อเริ่มใหม่", true);
    resetDetectionFrame();
  } else {
    stopModelStatusPolling();
    setModelStatus("เลือกไฟล์ best.pt เพื่อวาดกรอบตรวจจับทับวิดีโอ");
    resetDetectionFrame();
  }
}

async function applyImageSizeChange() {
  ++state.detectionGeneration;
  state.detectionSequence = 0;
  await resetDetectionTracker();
  if (state.detectorLoaded && ui.video.readyState >= 2) submitDetectionFrame(true);
}

function adoptTrainedImageSize() {
  const trained = state.detectionTrainedImageSize;
  if (!trained) return;

  // Always make the trained size selectable, even when it is not one of the
  // three the page ships with.
  // Anything added for a previous checkpoint is removed first, or the list grows
  // by one entry per model loaded and most of them mean nothing for the current
  // one.
  for (const option of [...ui.modelImageSize.options]) {
    if (option.dataset.trained === "1") option.remove();
  }
  const options = [...ui.modelImageSize.options];
  if (!options.some((option) => Number(option.value) === trained)) {
    const option = document.createElement("option");
    option.value = String(trained);
    option.textContent = `${trained} px`;
    option.dataset.trained = "1";
    const insertBefore = options.find((candidate) => Number(candidate.value) > trained);
    ui.modelImageSize.insertBefore(option, insertBefore || null);
  }

  if (state.detectionImageSizeTouched) return;

  // Running below the trained size costs small-object accuracy, but running a
  // 960 px model on the CPU costs well over a second per frame. Cap the
  // automatic choice where the README's own benchmark stops being usable.
  const onCpu = state.detectionDevice.toLowerCase().startsWith("cpu");
  // The frame endpoint rejects anything outside 320-1920, so adopting a trained
  // size beyond that would turn every submission into a 400 with no clue why.
  const withinApi = Math.min(DETECTION_MAX_IMAGE_SIZE, Math.max(DETECTION_MIN_IMAGE_SIZE, trained));
  const target = onCpu ? Math.min(withinApi, 640) : withinApi;
  if (Number(ui.modelImageSize.value) === target) return;

  ui.modelImageSize.value = String(target);
  // Assigning .value does not fire "change", so the reset the handler would
  // have done has to happen here. Not awaited, matching handleFile: the
  // generation bump already invalidates anything in flight.
  applyImageSizeChange();
}

function imageSizeMismatchWarning() {
  const trained = state.detectionTrainedImageSize;
  const selected = Number(ui.modelImageSize.value);
  if (!trained || !selected || trained === selected) return "";
  return `โมเดลเทรนที่ ${trained}px แต่กำลังรันที่ ${selected}px — ตำแหน่งกรอบจะคลาดเคลื่อน`;
}

function scheduleModelStatusPolling() {
  window.clearTimeout(state.detectionModelPollTimer);
  const generation = state.detectionModelGeneration;
  state.detectionModelPollTimer = window.setTimeout(async () => {
    try {
      const payload = await jsonFetch("/api/detection/model");
      if (generation !== state.detectionModelGeneration) return;
      const wasReady = state.detectorLoaded;
      applyDetectionModelState(payload);
      if (!wasReady && state.detectorLoaded && ui.video.readyState >= 2) {
        await resetDetectionTracker();
        submitDetectionFrame(true);
      }
    } catch (error) {
      if (generation === state.detectionModelGeneration) {
        setModelStatus(`อ่านสถานะ Detection worker ไม่ได้: ${error.message}`, true);
        scheduleModelStatusPolling();
      }
    }
  }, 500);
}

function stopModelStatusPolling() {
  window.clearTimeout(state.detectionModelPollTimer);
  state.detectionModelPollTimer = null;
}

async function uploadDetectionModel(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pt")) {
    setModelStatus("รองรับเฉพาะโมเดล Ultralytics YOLO นามสกุล .pt", true);
    return;
  }
  if (file.size > 256 * 1024 * 1024) {
    setModelStatus("ไฟล์โมเดลต้องมีขนาดไม่เกิน 256 MB", true);
    return;
  }
  const generation = ++state.detectionModelGeneration;
  state.detectorLoaded = false;
  // An image size chosen for the outgoing model says nothing about the incoming
  // one, which may well have been trained at a different size. Clearing this
  // lets the new model's trained size be adopted the way a first load would.
  state.detectionImageSizeTouched = false;
  ui.detectionToggle.disabled = true;
  ui.detectionRate.disabled = true;
  ui.modelConfidence.disabled = true;
  ui.modelImageSize.disabled = true;
  ui.clearModelButton.disabled = true;
  setModelStatus(`กำลังอัปโหลดและโหลด ${file.name}…`);
  try {
    const form = new FormData();
    form.append("model", file, file.name);
    const payload = await jsonFetch("/api/detection/model", { method: "POST", body: form });
    if (generation !== state.detectionModelGeneration) return;
    applyDetectionModelState(payload);
  } catch (error) {
    if (generation === state.detectionModelGeneration) {
      applyDetectionModelState({ loaded: false });
      setModelStatus(`โหลดโมเดลไม่สำเร็จ: ${error.message}`, true);
    }
  } finally {
    ui.modelInput.value = "";
  }
}

async function clearDetectionModel() {
  ++state.detectionModelGeneration;
  // The encoder is per-page, not per-model, but a wedged one would otherwise
  // survive the clear and greet the next model already broken.
  failEncoder("model cleared");
  stopModelStatusPolling();
  try {
    applyDetectionModelState(await jsonFetch("/api/detection/model", { method: "DELETE" }));
  } catch (error) {
    setModelStatus(`ล้างโมเดลไม่สำเร็จ: ${error.message}`, true);
  }
}

async function restartDetectionWorker() {
  ++state.detectionModelGeneration;
  stopModelStatusPolling();
  try {
    applyDetectionModelState(await jsonFetch("/api/detection/restart", { method: "POST" }));
  } catch (error) {
    setModelStatus(`Restart Detection worker ไม่สำเร็จ: ${error.message}`, true);
  }
}

async function initialize() {
  try {
    state.config = await jsonFetch("/api/config");
    const session = await jsonFetch("/api/session", { method: "POST" });
    state.sessionId = session.session_id;
    renderTelemetryRows();
    setEngineStatus("ready", `${state.config.model} พร้อมใช้งาน`);
    initializeDetectionModel();
    adoptRunningStream();
    checkLineConfigured();
  } catch (error) {
    setEngineStatus("error", "เชื่อมต่อ OCR ไม่สำเร็จ");
    // The page is a shell the service worker can serve on its own, so reaching
    // it says nothing about the server being up. Almost every time this fires
    // the answer is that the local server is not running, so say so instead of
    // reporting a fetch failure the reader has to interpret.
    showToast(
      `เชื่อมต่อ OCR server ที่ ${location.host} ไม่ได้ — ตรวจว่า server เปิดอยู่หรือไม่`
      + ` (รัน scripts\\start_windows_ocr_web.ps1 หรือดับเบิลคลิก install-ocr-web-autostart.cmd`
      + ` เพื่อให้เปิดเองหลัง login) · รายละเอียด: ${error.message}`,
    );
  }
}

function renderTelemetryRows() {
  ui.telemetryList.replaceChildren();
  state.rowElements.clear();
  const groupElements = new Map();
  for (const group of TELEMETRY_GROUPS) {
    const section = document.createElement("section");
    section.className = "telemetry-group";
    section.dataset.group = group.id;
    const heading = document.createElement("h3");
    heading.textContent = group.title;
    const rows = document.createElement("div");
    rows.className = "telemetry-group-rows";
    section.append(heading, rows);
    ui.telemetryList.append(section);
    groupElements.set(group.id, rows);
  }
  for (const field of state.config.fields.filter((item) => item.field_id !== "timestamp")) {
    const meta = FIELD_META[field.field_id] || { label: field.field_id, unit: "", group: "sensor" };
    const row = document.createElement("div");
    row.className = "telemetry-row is-pending";
    row.dataset.fieldId = field.field_id;

    const labelNode = document.createElement("div");
    labelNode.className = "telemetry-label";
    const indicator = document.createElement("span");
    indicator.className = "field-indicator";
    const labelText = document.createElement("span");
    labelText.textContent = meta.label;
    labelNode.append(indicator, labelText);

    const valueNode = document.createElement("div");
    valueNode.className = "telemetry-value";
    const value = document.createElement("strong");
    value.textContent = "—";
    const detail = document.createElement("small");
    detail.textContent = field.profile === "stable" ? "รอยืนยัน 3/5 เฟรม" : "รอยืนยัน 2/3 เฟรม";
    valueNode.append(value, detail);
    row.append(labelNode, valueNode);
    (groupElements.get(meta.group) || ui.telemetryList).append(row);
    state.rowElements.set(field.field_id, { row, value, detail, unit: meta.unit });
  }
}

function handleFile(file) {
  if (!file) return;
  if (state.streamMode) stopStream();
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|avi|mkv|webm)$/i.test(file.name)) {
    showToast("กรุณาเลือกไฟล์วิดีโอ");
    return;
  }
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  ui.video.src = state.objectUrl;
  ui.video.load();
  ui.fileName.textContent = file.name;
  ui.fileDetails.textContent = `${formatBytes(file.size)} · กำลังอ่านข้อมูลวิดีโอ`;
  ui.dropState.hidden = true;
  // Reset the outgoing source before adopting the new id. Minting it first made
  // the reset name a source the server had never seen, while the tracker state
  // for the video being replaced was never cleared at all.
  resetDetectionTracker();
  state.detectionSourceId = globalThis.crypto?.randomUUID?.()
    || `video-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  ++state.detectionGeneration;
  state.detectionSequence = 0;
  resetDetectionFrame();
  resetSession(true);
}

async function resetSession(clearRecords = true) {
  const generation = ++state.generation;
  if (clearRecords) clearLog();
  resetCurrentValues();
  if (!state.sessionId) return;
  try {
    await jsonFetch(`/api/session/${state.sessionId}/reset`, { method: "POST" });
    if (generation !== state.generation) return;
    setEngineStatus("ready", `${state.config.model} พร้อมใช้งาน`);
    // No frame read on reset: reading happens when a capture is taken.
  } catch (error) {
    showToast(`รีเซ็ต OCR ไม่สำเร็จ: ${error.message}`);
  }
}

function resetCurrentValues() {
  for (const { row, value, detail } of state.rowElements.values()) {
    row.classList.remove("is-confirmed");
    row.classList.add("is-pending");
    value.textContent = "—";
    detail.textContent = "รอข้อมูล OCR";
  }
}

function clearLog() {
  state.records = [];
  ui.recordCount.textContent = "0 records";
  ui.downloadJsonButton.disabled = true;
  ui.downloadCsvButton.disabled = true;
  ui.clearLogButton.disabled = true;
}

function startFrameLoop() {
  if (state.loopStarted) return;
  state.loopStarted = true;
  if (typeof ui.video.requestVideoFrameCallback === "function") {
    // Detection goes first. Its result is drawn on top of the moving picture,
    // so every millisecond it waits shows up as a box in the wrong place, while
    // OCR reads fixed regions and does not care. Behind maybeSubmit it paid for
    // the full-resolution ROI encode before it could start its own.
    const tick = (_, metadata) => {
      // The reference point the overlay's media clock is carried forward from.
      state.mediaClockMs = metadata.mediaTime * 1000;
      state.mediaClockWall = performance.now();
      maybeSubmitDetection(metadata.mediaTime * 1000);
      updatePlaybackTime();
      ui.video.requestVideoFrameCallback(tick);
    };
    ui.video.requestVideoFrameCallback(tick);
  } else {
    state.fallbackTimer = window.setInterval(() => {
      if (!ui.video.paused) {
        const sourceTimeMs = ui.video.currentTime * 1000;
        maybeSubmitDetection(sourceTimeMs);
        updatePlaybackTime();
      }
    }, 35);
  }
}

// Two possible pictures, one set of consumers. In stream mode the server owns
// the video and the page shows an <img> fed by MJPEG, because no browser can
// open RTSP. Everything that used to read the <video> directly asks here
// instead, so detection, the overlay and the capture path stay identical
// whichever source is on screen.
function sourceElement() {
  return state.streamMode ? ui.streamImage : ui.video;
}

function sourceWidth() {
  return state.streamMode ? ui.streamImage.naturalWidth : ui.video.videoWidth;
}

function sourceHeight() {
  return state.streamMode ? ui.streamImage.naturalHeight : ui.video.videoHeight;
}

// "There is a picture to read." An <img> reports naturalWidth only once it has
// decoded something, which is the same guarantee readyState >= 2 gives.
function sourceReady() {
  return state.streamMode ? ui.streamImage.naturalWidth > 0 : ui.video.readyState >= 2;
}

// "The picture is moving." A stream has no pause; it is either live or it is
// not, and a stopped stream must stop detection exactly as a paused video does.
function sourceLive() {
  return state.streamMode ? state.streamRunning : !(ui.video.paused || ui.video.ended);
}

function maybeSubmitDetection(sourceTimeMs) {
  if (
    !sourceLive()
    || !state.detectorLoaded
    || !ui.detectionToggle.checked
    || state.detectionInFlight
  ) return;
  // detectionIntervalMs is a wall-clock budget while this delta is media time,
  // so at 2x playback an unscaled comparison submitted twice as often and
  // oversubscribed the very worker the pacing exists to protect.
  const mediaInterval = detectionIntervalMs()
    * (state.streamMode ? 1 : (ui.video.playbackRate || 1));
  if (sourceTimeMs - state.lastDetectionMs < mediaInterval) return;
  submitDetectionFrame(false, sourceTimeMs);
}

// Submitting faster than the worker can keep up makes the boxes worse, not
// better. When it is oversubscribed it always finishes onto a frame that has
// been waiting in the queue, so that wait is added to every result; when it is
// not, each frame goes straight in and the staleness is just the inference. The
// wasted frames also cost a browser encode and an upload for nothing.
//
// Auto therefore paces submissions just under the measured round trip, which is
// the fastest rate at which nothing has to queue.
function detectionIntervalMs() {
  const choice = ui.detectionRate.value;
  if (choice !== "auto") return 1000 / Number(choice);
  const paced = DetectionMath.autoIntervalMs(
    state.detectionWorkerCycleMs,
    DETECTION_AUTO_HEADROOM,
    DETECTION_AUTO_MIN_INTERVAL_MS,
    DETECTION_AUTO_MAX_INTERVAL_MS,
  );
  // Nothing measured yet, so fall back to a rate that suits a typical machine.
  return paced === null ? 1000 / 10 : paced;
}

async function submitDetectionFrame(force = false, sourceTimeMs = currentMediaTime()) {
  if (!state.detectorLoaded || state.detectionInFlight || !sourceReady()) return;
  if (!force && (!sourceLive() || !ui.detectionToggle.checked)) return;
  const generation = state.detectionGeneration;
  const sequence = state.detectionSequence++;
  // Everything describing this frame is read before the first await. Dropping a
  // new video mid-encode reassigns detectionSourceId and bumps the generation,
  // and reading these afterwards posted old pixels under the new source id --
  // which the server takes as a source change and uses to reset the tracker
  // generation the client had only just installed. The image size has the same
  // problem: the bitmap was already sized from the previous value.
  const sourceId = state.detectionSourceId;
  const imageSize = ui.modelImageSize.value;
  const confidence = Math.min(1, Math.max(0.05, Number(ui.modelConfidence.value) || 0.45));
  state.detectionInFlight = true;
  state.lastDetectionMs = sourceTimeMs;
  try {
    const captureStarted = performance.now();
    let blob;
    try {
      blob = await captureDetectionFrame();
    } catch (error) {
      // A capture failure is local — a dead encoder, a frame the browser cannot
      // read. Reporting it as a model fault and re-initialising the model would
      // be wrong, and would start another polling loop for good measure.
      if (generation === state.detectionGeneration) {
        setModelStatus(`สร้างเฟรมสำหรับ AI ไม่สำเร็จ: ${error.message}`, true);
      }
      return;
    }
    state.detectionCaptureMs = performance.now() - captureStarted;
    const params = new URLSearchParams({
      source_id: sourceId,
      generation: String(generation),
      sequence: String(sequence),
      source_time_ms: sourceTimeMs.toFixed(3),
      confidence: confidence.toFixed(2),
      iou: "0.45",
      image_size: imageSize,
    });
    const postStarted = performance.now();
    const payload = await jsonFetch(`/api/detection/frame?${params}`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    state.detectionPostMs = performance.now() - postStarted;
    if (generation !== state.detectionGeneration) return;
    applyDetectionPayload(payload);
    // The polling loop delivers results now, whether or not playback is running.
  } catch (error) {
    if (generation === state.detectionGeneration) {
      setModelStatus(`AI detection ขัดข้อง: ${error.message}`, true);
      initializeDetectionModel();
    }
  } finally {
    state.detectionInFlight = false;
  }
}

// Encoding the detection frame used to cost ~78 ms on the main thread, all of
// it added to how stale the frame was before inference even started. The
// downscale goes through createImageBitmap, which resizes without the main
// thread touching pixels, and the JPEG encode moves to a worker. The worker is
// built from a blob rather than shipped as a file so it cannot be served from a
// stale cache independently of app.js.
const ENCODER_SOURCE = `
self.onmessage = async (event) => {
  const { id, bitmap, quality } = event.data;
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(bitmap, 0, 0);
    const blob = await canvas.convertToBlob({ type: "image/jpeg", quality });
    self.postMessage({ id, blob });
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  } finally {
    // Transferred, so the main thread cannot release it; without this a failed
    // encode leaks a full-size backing store every time.
    bitmap.close();
  }
};
`;

function detectionEncoder() {
  if (state.detectionEncoder !== undefined) return state.detectionEncoder;
  state.detectionEncoder = null;
  const supported = typeof Worker === "function"
    && typeof OffscreenCanvas === "function"
    && typeof createImageBitmap === "function";
  if (supported) {
    try {
      const url = URL.createObjectURL(new Blob([ENCODER_SOURCE], { type: "text/javascript" }));
      const worker = new Worker(url);
      URL.revokeObjectURL(url);
      worker.onmessage = (event) => {
        const pending = state.detectionEncodeJobs.get(event.data.id);
        if (!pending) return;
        settleEncodeJob(event.data.id);
        if (event.data.error) pending.reject(new Error(event.data.error));
        else pending.resolve(event.data.blob);
      };
      // Without these a dead worker leaves every job unsettled. The capture is
      // awaited inside submitDetectionFrame, so its finally never runs, the
      // in-flight flag stays raised and detection stops for good with nothing
      // said. A worker that has failed is also discarded rather than reused.
      worker.onerror = (event) => failEncoder(`worker error: ${event.message || "unknown"}`);
      worker.onmessageerror = () => failEncoder("worker returned an uncloneable message");
      state.detectionEncoder = worker;
    } catch (error) {
      console.warn("Detection encoder worker unavailable, using the main thread", error);
    }
  }
  return state.detectionEncoder;
}

function settleEncodeJob(id) {
  const pending = state.detectionEncodeJobs.get(id);
  if (!pending) return null;
  window.clearTimeout(pending.timer);
  state.detectionEncodeJobs.delete(id);
  return pending;
}

// Fails every outstanding encode and drops the worker. The next capture builds
// a fresh one, or falls back to the main thread if that fails too, so a crashed
// encoder costs frames rather than the whole feature.
function failEncoder(reason) {
  for (const id of [...state.detectionEncodeJobs.keys()]) {
    settleEncodeJob(id)?.reject(new Error(reason));
  }
  if (state.detectionEncoder) state.detectionEncoder.terminate();
  state.detectionEncoder = undefined;
}

function detectionFrameSize() {
  const width = sourceWidth();
  const height = sourceHeight();
  const targetLongSide = Math.max(320, Number(ui.modelImageSize.value) || 640);
  const scale = Math.min(1, targetLongSide / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

async function captureDetectionFrame() {
  const { width, height } = detectionFrameSize();
  const worker = detectionEncoder();
  if (worker) {
    const bitmap = await createImageBitmap(sourceElement(), {
      resizeWidth: width,
      resizeHeight: height,
      resizeQuality: "medium",
    });
    const id = ++state.detectionEncodeSequence;
    return new Promise((resolve, reject) => {
      // Last line of defence: a worker that is alive but silently never answers
      // this job would otherwise wedge detection permanently.
      const timer = window.setTimeout(() => {
        settleEncodeJob(id);
        reject(new Error(`encode timed out after ${DETECTION_ENCODE_TIMEOUT_MS} ms`));
      }, DETECTION_ENCODE_TIMEOUT_MS);
      state.detectionEncodeJobs.set(id, { resolve, reject, timer });
      worker.postMessage({ id, bitmap, quality: 0.86 }, [bitmap]);
    });
  }
  return captureDetectionFrameOnMainThread(width, height);
}

function captureDetectionFrameOnMainThread(width, height) {
  ui.detectionCanvas.width = width;
  ui.detectionCanvas.height = height;
  const context = ui.detectionCanvas.getContext("2d", { alpha: false });
  context.drawImage(sourceElement(), 0, 0, width, height);
  return new Promise((resolve, reject) => {
    ui.detectionCanvas.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error("สร้างเฟรมสำหรับ AI ไม่สำเร็จ")),
      "image/jpeg",
      0.86,
    );
  });
}

function resetDetectionFrame() {
  state.lastDetectionMs = -Infinity;
  state.detectionResultSequence = -1;
  state.detectionTracks.clear();
  state.detectionCompletions = [];
  state.detectionSourceWidth = 0;
  state.detectionSourceHeight = 0;
  state.detectionCaptureMs = null;
  state.detectionPostMs = null;
  // Abandoned encodes must be failed rather than forgotten, or whatever is
  // awaiting them never resumes and never clears the in-flight flag.
  for (const id of [...state.detectionEncodeJobs.keys()]) {
    settleEncodeJob(id)?.reject(new Error("detection was reset"));
  }
  // Motion learned from the previous video, model or seek position says nothing
  // about the next one.
  state.detectionCameraVelocity = null;
  state.detectionWorkerCycleMs = null;
  state.detectionResultGapMs = 0;
  state.detectionLastResultMs = null;
  state.mediaClockMs = null;
  ui.detectionFps.textContent = "0.0";
  ui.detectionLatency.textContent = "—";
  ui.detectionLag.textContent = "—";
  ui.detectionCapture.textContent = "—";
  ui.detectionHarvest.textContent = "—";
  drawDetections();
}

async function resetDetectionTracker() {
  resetDetectionFrame();
  // The running loop is pinned to the previous generation and would exit on
  // its own; restarting it now avoids waiting out its in-flight long poll.
  restartDetectionResultPolling();
  if (!state.detectorLoaded || !state.detectionSourceId) return;
  const params = new URLSearchParams({
    source_id: state.detectionSourceId,
    generation: String(state.detectionGeneration),
  });
  try {
    await jsonFetch(`/api/detection/reset?${params}`, { method: "POST" });
  } catch (error) {
    setModelStatus(`Reset tracker ไม่สำเร็จ: ${error.message}`, true);
  }
}

function applyDetectionPayload(payload) {
  const worker = payload.worker || {};
  // Recorded before every guard below, and never reset: the server's revision
  // only climbs, so remembering it is exactly what lets the next poll block.
  // Skipping it for a result that fails a guard would leave the poll asking for
  // a revision the server already passed, which is an unthrottled loop.
  const revision = Number(payload.revision);
  if (Number.isFinite(revision)) state.detectionRevision = revision;
  ui.detectionDropped.textContent = String(payload.dropped_frames ?? worker.dropped_frames ?? 0);
  if (worker.status && worker.status !== "ready") {
    applyDetectionModelState(worker);
    return;
  }
  const result = payload.latest_result;
  if (!result || result.source_id !== state.detectionSourceId) return;
  if (Number(result.generation) !== state.detectionGeneration) return;
  if (Number(result.sequence) <= state.detectionResultSequence) return;

  state.detectionResultSequence = Number(result.sequence);
  state.detectionSourceWidth = Number(result.width) || sourceWidth();
  state.detectionSourceHeight = Number(result.height) || sourceHeight();
  const completedNow = performance.now();
  state.detectionCompletions.push(completedNow);
  state.detectionCompletions = state.detectionCompletions.filter((value) => completedNow - value <= 5000);
  const durationMs = state.detectionCompletions.length > 1
    ? completedNow - state.detectionCompletions[0]
    : 0;
  const fps = durationMs > 0
    ? (state.detectionCompletions.length - 1) * 1000 / durationMs
    : 0;
  // Measured against the same clock the overlay uses to decide staleness;
  // reading video.currentTime here under-reported it by a frame or two and
  // disagreed with the compensation actually being applied.
  const lagMs = Math.max(0, currentMediaTime() - Number(result.source_time_ms));
  const timing = payload.timing || {};
  // What one frame actually costs the worker end to end, which is what the
  // automatic pacing has to stay just under.
  const workerCycle = Number(result.inference_ms) + Number(timing.transport_ms || 0);
  if (Number.isFinite(workerCycle) && workerCycle > 0) {
    state.detectionWorkerCycleMs = smooth(state.detectionWorkerCycleMs, workerCycle);
  }
  ui.detectionFps.textContent = fps.toFixed(1);
  ui.detectionLatency.textContent = String(Math.round(Number(result.inference_ms) || 0));
  ui.detectionLag.textContent = String(Math.round(lagMs));
  ui.detectionCapture.textContent = formatMs(state.detectionCaptureMs);
  ui.detectionHarvest.textContent = formatMs(timing.harvest_ms);
  applyDetectionResult(result, completedNow);
  const objectCount = Array.isArray(result.detections) ? result.detections.length : 0;
  let detectionSummary = `ไม่พบวัตถุในเฟรมล่าสุดที่ confidence ${ui.modelConfidence.value}`;
  if (objectCount > 0 && lagMs > DETECTION_STALE_MS) {
    detectionSummary = `พบ ${objectCount} วัตถุ แต่ผลช้ากว่าวิดีโอเกิน ${DETECTION_STALE_MS} ms จึงซ่อนกรอบ`;
  } else if (objectCount > 0) {
    detectionSummary = `พบ ${objectCount} วัตถุ`;
  }
  // The breakdown has to reconcile with lag, which is the ground truth; it is
  // the only way to tell where the delay actually lives.
  const breakdown = [
    `rate ${(1000 / detectionIntervalMs()).toFixed(1)} Hz`,
    `cap ${formatMs(state.detectionCaptureMs)}`,
    `post ${formatMs(state.detectionPostMs)}`,
    `infer ${formatMs(result.inference_ms)}`,
    `transport ${formatMs(timing.transport_ms)}`,
    `wait ${formatMs(timing.harvest_ms)}`,
  ].join(" · ");
  // Re-stated on every result: this line is rewritten continuously, so a
  // warning shown once at load time would vanish the moment boxes arrive.
  const mismatch = imageSizeMismatchWarning();
  // A worker that rejects every frame stays "ready" and simply stops producing
  // boxes. Without this the overlay just froze and the reason lived only in the
  // server log.
  const frameError = worker.frame_error
    ? `เฟรมล่าสุดประมวลผลไม่สำเร็จ (${worker.frame_errors} ครั้ง): ${worker.frame_error}`
    : "";
  setModelStatus(
    `${ui.modelName.textContent} · BoT-SORT · ${detectionSummary} · ${fps.toFixed(1)} FPS`
    + ` · lag ${Math.round(lagMs)} ms · ${breakdown}`
    + (mismatch ? ` · ${mismatch}` : "")
    + (frameError ? ` · ${frameError}` : ""),
    Boolean(mismatch || frameError),
  );
}

function formatMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? String(Math.round(number)) : "—";
}

// video.currentTime is quantised to frame boundaries and reads behind the frame
// actually on screen, so measuring staleness against it under-reports it and
// the box is compensated by less than it needs. requestVideoFrameCallback gives
// the exact presentation timestamp of the frame that was just shown; the clock
// is carried forward from there at the current playback rate.
function currentMediaTime() {
  if (state.streamMode) {
    // A live feed has no seekable timeline to read; the server's elapsed source
    // time is the only clock both sides agree on, carried forward here between
    // updates so the overlay can still compensate between them.
    if (state.streamClockMs === null) return 0;
    return state.streamClockMs + (performance.now() - state.streamClockWall);
  }
  return projectMediaTime({
    clockMs: state.mediaClockMs,
    clockWallMs: state.mediaClockWall,
    nowWallMs: performance.now(),
    playbackRate: ui.video.playbackRate,
    paused: ui.video.paused,
    currentTimeMs: ui.video.currentTime * 1000,
    leadMs: DETECTION_PRESENTATION_LEAD_MS,
    stallMs: DETECTION_STALL_MS,
  });
}

// Shared with detection-math.js so the arithmetic the page runs is the
// arithmetic the tests exercise.
const {
  boxCentre, letterbox, usableGap, extrapolationReach, projectMediaTime,
} = DetectionMath;

const clampSpeed = (value) => DetectionMath.clampSpeed(value, DETECTION_MAX_SPEED);
const median = (values) => DetectionMath.median(values);
const smooth = (previousValue, nextValue, weight = DETECTION_VELOCITY_SMOOTHING) =>
  DetectionMath.smooth(previousValue, nextValue, weight);

function expireDetectionTracks(renderNow) {
  for (const [key, track] of state.detectionTracks) {
    if (track.missingSince === null) continue;
    if (renderNow - track.missingSince >= DETECTION_FADE_MS) state.detectionTracks.delete(key);
  }
}

function applyDetectionResult(result, completedNow) {
  const activeKeys = new Set();
  const sourceTimeMs = Number(result.source_time_ms);
  const sampled = [];

  // How far apart results actually land, in media time. Used to decide how far
  // forward a linear guess can be trusted.
  if (Number.isFinite(state.detectionLastResultMs)) {
    const gap = sourceTimeMs - state.detectionLastResultMs;
    if (usableGap(gap, DETECTION_MIN_SAMPLE_GAP_MS, DETECTION_MAX_SAMPLE_GAP_MS)) {
      state.detectionResultGapMs = smooth(state.detectionResultGapMs, gap);
    }
  }
  state.detectionLastResultMs = sourceTimeMs;

  (result.detections || []).forEach((detection, index) => {
    const untracked = detection.track_id === null || detection.track_id === undefined;
    const key = untracked
      ? `untracked:${result.sequence}:${index}`
      : `track:${detection.track_id}`;
    activeKeys.add(key);
    const existing = state.detectionTracks.get(key);
    const current = { ...detection, sourceTimeMs };
    const previous = existing?.current && existing.current.sourceTimeMs < current.sourceTimeMs
      ? existing.current
      : existing?.previous || null;

    let velocityX = null;
    let velocityY = null;
    const gap = previous ? current.sourceTimeMs - previous.sourceTimeMs : 0;
    if (usableGap(gap, DETECTION_MIN_SAMPLE_GAP_MS, DETECTION_MAX_SAMPLE_GAP_MS)) {
      velocityX = smooth(
        existing?.velocityX,
        clampSpeed((boxCentre(current, "x") - boxCentre(previous, "x")) / gap),
      );
      velocityY = smooth(
        existing?.velocityY,
        clampSpeed((boxCentre(current, "y") - boxCentre(previous, "y")) / gap),
      );
      // A box with a malformed coordinate yields no usable speed. Feeding that
      // into the median as a zero would invent a stationary object and pull the
      // camera estimate toward standing still.
      if (Number.isFinite(velocityX) && Number.isFinite(velocityY)) {
        sampled.push([velocityX, velocityY]);
      }
    }

    state.detectionTracks.set(key, {
      previous,
      current,
      missingSince: null,
      untracked,
      velocityX,
      velocityY,
    });
  });

  // When the aircraft moves, every object in frame translates together, so the
  // median across objects estimates that shared motion far more reliably than
  // two samples of any single box. The median rather than the mean so that a
  // genuinely moving vehicle among stationary ones does not drag the estimate.
  if (!sampled.length && state.detectionCameraVelocity) {
    // Nothing measurable this round: a single-object scene, every box on its
    // first sighting. Holding the last pan indefinitely kept sliding every box
    // off its object after the camera had already stopped, so it decays toward
    // rest instead of persisting until the next reset.
    state.detectionCameraVelocity = {
      x: smooth(state.detectionCameraVelocity.x, 0, DETECTION_CAMERA_SMOOTHING),
      y: smooth(state.detectionCameraVelocity.y, 0, DETECTION_CAMERA_SMOOTHING),
    };
  }
  if (sampled.length) {
    state.detectionCameraVelocity = {
      x: smooth(
        state.detectionCameraVelocity?.x,
        median(sampled.map((v) => v[0])),
        DETECTION_CAMERA_SMOOTHING,
      ),
      y: smooth(
        state.detectionCameraVelocity?.y,
        median(sampled.map((v) => v[1])),
        DETECTION_CAMERA_SMOOTHING,
      ),
    };
  }

  // A box seen only once has no velocity of its own and would sit still while
  // everything around it moved. That is every newly assigned track id, and every
  // detection the tracker never identified at all -- those get a fresh key each
  // result and so can never accumulate a second sample. The camera estimate is
  // the best available guess for them.
  const camera = state.detectionCameraVelocity;
  if (camera) {
    for (const key of activeKeys) {
      const track = state.detectionTracks.get(key);
      if (track && !Number.isFinite(track.velocityX)) {
        track.velocityX = camera.x;
        track.velocityY = camera.y;
      }
    }
  }

  for (const [key, track] of state.detectionTracks) {
    if (!activeKeys.has(key) && track.missingSince === null) {
      track.missingSince = completedNow;
    }
  }
}

// Results used to arrive only piggy-backed on the next frame POST, which meant
// a finished detection waited a whole submission interval before it could be
// drawn. This loop runs independently of submission and blocks on the server
// until a newer result exists, so a result reaches the canvas as soon as the
// worker publishes it.
// Each loop carries the token it was started with. A boolean could not do this
// job: stopping only flips it, while the running loop is parked inside a fetch
// and cannot observe the change until that returns — so a stop immediately
// followed by a start left the old loop alive beside the new one, and every
// reset that did not also bump the generation added another.
function startDetectionResultPolling() {
  const token = ++state.detectionPollToken;
  pollDetectionResultLoop(token);
}

function stopDetectionResultPolling() {
  // Only invalidates the token. Cancelling the pending delay would leave its
  // promise unresolved and strand the loop mid-await forever; letting it expire
  // costs one short timer and the loop then sees the token and exits.
  ++state.detectionPollToken;
}

function pollingAbandoned(token) {
  return token !== state.detectionPollToken || !state.detectorLoaded;
}

async function pollDetectionResultLoop(token) {
  while (!pollingAbandoned(token)) {
    try {
      const params = new URLSearchParams({
        source_time_ms: (ui.video.currentTime * 1000).toFixed(3),
        after_revision: String(state.detectionRevision),
        wait_ms: "250",
      });
      const payload = await jsonFetch(`/api/detection/result?${params}`);
      if (pollingAbandoned(token)) return;
      try {
        applyDetectionPayload(payload);
      } catch (error) {
        // Distinguished from a transport failure on purpose: retrying the same
        // malformed payload at 2 Hz forever, silently, is not a recovery.
        console.error("Discarding a detection payload that could not be applied", error);
      }
      // A backstop, not the pacing mechanism: the server is supposed to hold
      // the request until there is something new. If it ever returns instantly
      // — a bug, a version mismatch, an unexpected payload — this is what keeps
      // the page from turning into a flood against the loop that also runs OCR.
      await pollDelay(DETECTION_POLL_FLOOR_MS);
    } catch (error) {
      if (pollingAbandoned(token)) return;
      await pollDelay(DETECTION_POLL_BACKOFF_MS);
    }
  }
}

function pollDelay(ms) {
  return new Promise((resolve) => {
    state.detectionPollTimer = window.setTimeout(resolve, ms);
  });
}

function restartDetectionResultPolling() {
  stopDetectionResultPolling();
  if (state.detectorLoaded) startDetectionResultPolling();
}

function startDetectionRenderLoop() {
  if (state.detectionRenderFrame !== null) return;
  const render = (now) => {
    drawDetections(now);
    state.detectionRenderFrame = window.requestAnimationFrame(render);
  };
  state.detectionRenderFrame = window.requestAnimationFrame(render);
}

function drawDetections(renderNow = performance.now()) {
  const stageWidth = ui.videoStage.clientWidth;
  const stageHeight = ui.videoStage.clientHeight;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const requiredWidth = Math.max(1, Math.round(stageWidth * pixelRatio));
  const requiredHeight = Math.max(1, Math.round(stageHeight * pixelRatio));
  if (ui.detectionOverlay.width !== requiredWidth || ui.detectionOverlay.height !== requiredHeight) {
    ui.detectionOverlay.width = requiredWidth;
    ui.detectionOverlay.height = requiredHeight;
  }
  const context = ui.detectionOverlay.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, stageWidth, stageHeight);
  // Expiry runs before anything that can return early. It used to sit inside the
  // draw loop, past both guards below, so switching the overlay off and then
  // scrubbing grew the map for the life of the page — forced submissions keep
  // arriving regardless of the toggle, and every untracked detection takes a
  // fresh key it can never be given a second sample under.
  expireDetectionTracks(renderNow);
  if (!state.detectorLoaded || !ui.detectionToggle.checked || !state.detectionTracks.size) return;

  const frameWidth = state.detectionSourceWidth || sourceWidth();
  const frameHeight = state.detectionSourceHeight || sourceHeight();
  if (!frameWidth || !frameHeight) return;
  const { offsetX, offsetY, scaleX, scaleY } = letterbox(
    stageWidth, stageHeight, frameWidth, frameHeight,
  );
  context.lineWidth = 2;
  context.font = "600 12px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.textBaseline = "top";

  // prefers-reduced-motion used to disable prediction entirely. That query is
  // about decorative animation causing discomfort; a tracking box is neither
  // decorative nor calmer for being wrong. Honouring it here just froze the
  // compensation and left every box trailing its object.
  const currentMediaTimeMs = currentMediaTime();
  for (const [key, track] of state.detectionTracks) {
    let opacity = 1;
    if (track.missingSince !== null) {
      opacity = Math.max(0, 1 - (renderNow - track.missingSince) / DETECTION_FADE_MS);
      if (opacity <= 0) continue;
    }
    const detection = track.current;
    const mediaAgeMs = currentMediaTimeMs - detection.sourceTimeMs;
    if (mediaAgeMs > DETECTION_STALE_MS || mediaAgeMs < -100) continue;
    // A linear guess holds for roughly two result intervals; past that the box
    // overshoots on every turn, which reads worse than trailing. The reach is
    // therefore tied to the observed cadence rather than to the lag.
    const reach = extrapolationReach(
      state.detectionResultGapMs,
      DETECTION_MIN_EXTRAPOLATION_MS,
      DETECTION_MAX_EXTRAPOLATION_MS,
    );
    const extrapolationMs = sourceLive()
      ? Math.min(reach, Math.max(0, mediaAgeMs))
      : 0;
    // Velocity is smoothed as results arrive, and falls back to the shared
    // camera motion for boxes with only one sample. Only the centre moves:
    // BoT-SORT's jitter is mostly in the box dimensions, so predicting the
    // corners separately amplified the noisiest signal and made boxes pulse
    // while they drifted.
    const rawWidth = Number(detection.x2) - Number(detection.x1);
    const rawHeight = Number(detection.y2) - Number(detection.y1);
    const velocityX = Number.isFinite(track.velocityX) ? track.velocityX : 0;
    const velocityY = Number.isFinite(track.velocityY) ? track.velocityY : 0;
    const centreX = boxCentre(detection, "x") + velocityX * extrapolationMs;
    const centreY = boxCentre(detection, "y") + velocityY * extrapolationMs;
    const x1 = Math.min(frameWidth, Math.max(0, centreX - rawWidth / 2));
    const y1 = Math.min(frameHeight, Math.max(0, centreY - rawHeight / 2));
    const x2 = Math.min(frameWidth, Math.max(x1 + 1, centreX + rawWidth / 2));
    const y2 = Math.min(frameHeight, Math.max(y1 + 1, centreY + rawHeight / 2));
    const x = offsetX + x1 * scaleX;
    const y = offsetY + y1 * scaleY;
    const width = Math.max(1, (x2 - x1) * scaleX);
    const height = Math.max(1, (y2 - y1) * scaleY);
    const hue = (Number(detection.class_id) * 67 + 88) % 360;
    const color = `hsl(${hue} 88% 62%)`;
    // The track id is BoT-SORT's identity for this object across frames. It is
    // off by default because it is debugging information: it matters when you
    // are checking whether the tracker is holding on to an object, and is
    // noise the rest of the time.
    const trackLabel = !ui.trackIdToggle.checked
      || detection.track_id === null
      || detection.track_id === undefined
      ? ""
      : ` #${detection.track_id}`;
    const label = `${detection.label}${trackLabel} ${(Number(detection.confidence) * 100).toFixed(0)}%`;
    const metrics = context.measureText(label);
    const labelWidth = metrics.width + 10;
    const labelHeight = 20;
    const labelY = Math.max(offsetY, y - labelHeight);
    context.globalAlpha = opacity;
    context.strokeStyle = color;
    context.strokeRect(x, y, width, height);
    context.fillStyle = "rgba(4,8,10,.82)";
    context.fillRect(x, labelY, labelWidth, labelHeight);
    context.fillStyle = color;
    context.fillText(label, x + 5, labelY + 3);
    context.globalAlpha = 1;
  }
}


// ---------------------------------------------------------------------------
// Stream mode
//
// The server decodes the camera and runs OCR; this page shows the picture and
// the numbers it is sent. Detection is unchanged -- frames are captured from
// the <img> and posted to the endpoint the file path already uses, so YOLO
// reaches a stream through code that is already tested.
// ---------------------------------------------------------------------------

async function toggleStream() {
  if (state.streamMode) {
    await stopStream();
    return;
  }
  const url = ui.streamUrl.value.trim();
  if (!url) {
    showToast("ใส่ URL ของสตรีมก่อน เช่น rtsp://host:port/path");
    ui.streamUrl.focus();
    return;
  }
  await startStream(url);
}

async function startStream(url) {
  ui.streamButton.disabled = true;
  try {
    await jsonFetch("/api/stream/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  } catch (error) {
    showToast(`เปิดสตรีมไม่สำเร็จ: ${error.message}`);
    return;
  } finally {
    ui.streamButton.disabled = false;
  }
  enterStreamMode(url);
}

// Adopt a stream this page did not start.
//
// The server keeps reading its camera with no browser attached -- that is the
// point of the auto-start -- so a page opened or reloaded afterwards finds a
// stream already running. Without this it showed the file UI instead, the
// picture never appeared, and pressing the stream button only produced "already
// running". The capture button was unreachable for exactly the setup that
// feature was built for.
async function adoptRunningStream() {
  try {
    const status = await jsonFetch("/api/stream/status");
    if (!status.running || !status.url) return;
    ui.streamUrl.value = status.url;
    enterStreamMode(status.url);
    showToast(`กำลังแสดงสตรีมที่ server เปิดค้างไว้: ${status.url}`);
  } catch {
    // A server too old or too busy to answer is not a reason to hold up the
    // rest of the page; the file UI still works.
  }
}

function enterStreamMode(url) {
  state.streamMode = true;
  state.streamRunning = true;
  state.streamRevision = 0;
  state.streamClockMs = null;
  // The file source has to stop before the stream takes over, or a video left
  // playing underneath keeps posting frames to the OCR the server now owns.
  ui.video.pause();
  ui.video.hidden = true;
  ui.streamImage.hidden = false;
  ui.dropState.hidden = true;
  ui.streamButton.textContent = "หยุดสตรีม";
  ui.streamBadge.hidden = false;
  ui.streamBadgeText.textContent = "LIVE";
  ui.fileName.textContent = url;
  ui.fileDetails.textContent = "server กำลังถอด telemetry จากสตรีมนี้";
  // Cache-busted so a reconnect is never served the previous run's response.
  ui.streamImage.src = `/api/stream/mjpeg?t=${Date.now()}`;

  // A stream is a different source: the tracker must not carry identities over
  // from whatever was on screen before it.
  resetDetectionTracker();
  state.detectionSourceId = globalThis.crypto?.randomUUID?.()
    || `stream-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  ++state.detectionGeneration;
  state.detectionSequence = 0;
  resetDetectionFrame();
  resetCurrentValues();
  setEngineStatus("ready", `${state.config.model} · stream live`);
  updateCaptureAvailability();
  startStreamTelemetryPolling();
  startStreamRenderLoop();
}

async function stopStream() {
  ui.streamButton.disabled = true;
  try {
    await jsonFetch("/api/stream/stop", { method: "POST" });
  } catch (error) {
    showToast(`หยุดสตรีมไม่สำเร็จ: ${error.message}`);
  } finally {
    ui.streamButton.disabled = false;
  }
  leaveStreamMode();
}

function leaveStreamMode() {
  state.streamMode = false;
  state.streamRunning = false;
  state.streamClockMs = null;
  ++state.streamPollToken;
  if (state.streamRenderFrame !== null) {
    window.cancelAnimationFrame(state.streamRenderFrame);
    state.streamRenderFrame = null;
  }
  // Dropping the src is what actually closes the MJPEG connection; leaving it
  // in place holds the response open and the server keeps writing to it.
  ui.streamImage.removeAttribute("src");
  ui.streamImage.hidden = true;
  ui.video.hidden = false;
  ui.streamBadge.hidden = true;
  ui.streamButton.textContent = "เปิดสตรีม";
  if (!state.file) ui.dropState.hidden = false;
  stopCaptureHold();
  updateCaptureAvailability();
  drawDetections();
}

function startStreamRenderLoop() {
  // requestVideoFrameCallback does not exist on an <img>, and MJPEG frames
  // arrive without an event worth hooking, so the overlay is driven at display
  // rate instead. That is the cadence the video path ends up at anyway.
  const token = state.streamPollToken;
  const tick = () => {
    if (!state.streamMode || token !== state.streamPollToken) return;
    maybeSubmitDetection(currentMediaTime());
    drawDetections();
    state.streamRenderFrame = window.requestAnimationFrame(tick);
  };
  state.streamRenderFrame = window.requestAnimationFrame(tick);
}

function startStreamTelemetryPolling() {
  const token = ++state.streamPollToken;
  (async () => {
    while (state.streamMode && token === state.streamPollToken) {
      let payload;
      try {
        const params = new URLSearchParams({
          after_revision: String(state.streamRevision),
          wait_ms: "5000",
        });
        payload = await jsonFetch(`/api/stream/telemetry?${params}`);
      } catch (error) {
        if (token !== state.streamPollToken) return;
        setEngineStatus("error", "สตรีมขัดข้อง");
        showToast(`อ่าน telemetry จากสตรีมไม่สำเร็จ: ${error.message}`);
        // A failed poll must still pause before retrying. Without this a server
        // that has gone away turns this into an unthrottled retry storm -- the
        // same failure the detection poll was fixed for.
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        continue;
      }
      if (token !== state.streamPollToken) return;
      applyStreamTelemetry(payload);
      if (!payload.running) {
        // The server stopped on its own -- an unreachable camera, or a feed that
        // never came back. Say why rather than leaving a frozen picture.
        showToast(payload.error
          ? `สตรีมหยุด: ${payload.error}`
          : "สตรีมหยุดแล้ว");
        setEngineStatus("error", "สตรีมหยุด");
        leaveStreamMode();
        return;
      }
    }
  })();
}

function applyStreamTelemetry(payload) {
  const revision = Number(payload.revision);
  // Recorded even when nothing else here uses the payload, for the same reason
  // the detection revision is: a poll that forgets it asks for a revision the
  // server has already passed and never blocks again.
  if (Number.isFinite(revision)) state.streamRevision = revision;
  state.streamRunning = Boolean(payload.running);
  // Here rather than in the render loop. requestAnimationFrame does not run in
  // a tab that is not being composited, which left the capture button disabled
  // for a stream that was demonstrably working. This poll is driven by the
  // server and keeps running wherever the tab is.
  updateCaptureAvailability();
  const sourceTimeMs = Number(payload.source_time_ms);
  if (Number.isFinite(sourceTimeMs)) {
    state.streamClockMs = sourceTimeMs;
    state.streamClockWall = performance.now();
  }
  const fields = payload.fields || {};
  const records = Object.entries(fields).map(([fieldId, item]) => ({
    field_id: fieldId,
    value: item.value,
    raw_text: item.raw_text,
    confidence: item.confidence,
    confirmed: item.confirmed,
    preprocess_variant: item.preprocess_variant || "stream",
    source_time_ms: sourceTimeMs,
    frame_index: Number(payload.frame_index) || 0,
    latency_ms: Number(payload.latency_ms) || 0,
  }));
  if (!records.length) return;
  // Deliberately the same path a file takes, including the log and the download
  // buttons: a stream is a recording too, it just has no end.
  applyOcrResult({
    latency_ms: Number(payload.latency_ms) || 0,
    records,
  });
  ui.streamBadgeText.textContent = `LIVE · ${Math.round(sourceTimeMs / 1000)}s`;
}


// ---------------------------------------------------------------------------
// Capture
//
// Everything else here is for watching: it answers while the video plays and
// keeps nothing. This is the one control that writes something permanent -- the
// frame on screen, and what OCR made of it, in the session format `extract`
// reads. Pressing it is how a value that matters stops being a thing that went
// past and becomes a thing that can be checked.
//
// Click takes one shot. Holding takes them until released, which is what a
// moving readout needs; a single click there almost always lands on the frame
// after the interesting one.
// ---------------------------------------------------------------------------

const CAPTURE_HOLD_INTERVAL_MS = 400;

function captureSessionName() {
  if (state.captureSession) return state.captureSession;
  // Named for when it started rather than what it contains, because what it
  // contains is not known yet and a timestamp sorts correctly in a file browser.
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
    "-",
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0"),
  ].join("");
  state.captureSession = `session-${stamp}`;
  return state.captureSession;
}

function setCaptureStatus(message, isError = false) {
  ui.captureStatus.textContent = message;
  ui.captureStatus.classList.toggle("is-error", Boolean(isError));
}

function flashCapture() {
  // Restarting the animation needs the class gone for a frame, or a second shot
  // taken quickly gives no feedback at all.
  ui.captureFlash.classList.remove("is-firing");
  void ui.captureFlash.offsetWidth;
  ui.captureFlash.classList.add("is-firing");
}

// Deepest the queue is allowed to get while the button is held.
//
// Holding asks for a frame every 400 ms and one capture takes about 600 ms of
// saving, detection and re-reading, so the requests arrive faster than they can
// be served. Without a cap, five seconds of holding would leave a backlog that
// keeps capturing long after the button is released. With it, holding settles
// at whatever rate the model can actually sustain.
const MAX_CAPTURE_QUEUE = 3;

function requestCapture() {
  if (!sourceReady()) {
    setCaptureStatus("ยังไม่มีภาพให้เก็บ เลือกวิดีโอหรือเปิดสตรีมก่อน", true);
    return false;
  }
  if (state.captureQueued >= MAX_CAPTURE_QUEUE) return false;
  state.captureQueued += 1;
  renderShotStrip();
  updateShotProgress();
  pumpCaptures();
  return true;
}

// One capture at a time, and every request answered.
//
// It used to drop any press that arrived while another was in flight, which on
// a 400 ms hold and a 600 ms capture silently threw away roughly every other
// frame: the operator held the button, watched nothing happen, and had no way
// to tell how many frames they had actually got.
async function pumpCaptures() {
  if (state.captureInFlight) return;
  while (state.captureQueued > 0) {
    state.captureQueued -= 1;
    state.pendingCaptures += 1;
    state.captureInFlight = true;
    renderShotStrip();
    updateShotProgress();
    try {
      await takeCaptureShot();
    } finally {
      state.pendingCaptures -= 1;
      state.captureInFlight = false;
    }
  }
  renderShotStrip();
  updateShotProgress();
}

async function takeCaptureShot() {
  try {
    const params = new URLSearchParams({
      session: captureSessionName(),
      source_time_ms: Math.max(0, currentMediaTime()).toFixed(3),
    });
    // In stream mode the server already holds the frame and its reading, so
    // sending one would be both wasteful and a different picture than the one
    // it read. Only the file path has to post pixels.
    let options = { method: "POST" };
    if (!state.streamMode) {
      params.set("session_id", state.sessionId);
      params.set("source", state.file ? state.file.name : "browser");
      const blob = await captureFullFrame();
      options = {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      };
    }
    // The server does not answer until the frame has been saved, put through
    // best.pt and re-read from the file, so by the time this returns the shot
    // is finished rather than merely stored.
    const shot = await jsonFetch(`/api/capture/shot?${params}`, options);
    state.captureCount += 1;
    flashCapture();
    await refreshShots(shot.frame_index);
    const stage = describeShotDetections(shot.detections);
    setCaptureStatus(
      `เก็บแล้ว ${state.shots.length} ภาพ · ล่าสุด #${shot.frame_index} · ${stage}`,
    );
  } catch (error) {
    state.captureFailures += 1;
    setCaptureStatus(`Capture ไม่สำเร็จ: ${error.message}`, true);
  }
}

// The whole frame, not the ROI-only montage the OCR path sends. A capture that
// kept only the strips it read would be unable to answer the question it exists
// for, which is what the rest of the picture looked like at that moment.
function captureFullFrame() {
  const width = sourceWidth();
  const height = sourceHeight();
  ui.canvas.width = width;
  ui.canvas.height = height;
  const context = ui.canvas.getContext("2d", { alpha: false });
  context.drawImage(sourceElement(), 0, 0, width, height);
  return new Promise((resolve, reject) => {
    ui.canvas.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error("สร้างภาพเฟรมไม่สำเร็จ")),
      "image/jpeg",
      0.95,
    );
  });
}

function startCaptureHold() {
  if (state.captureHolding) return;
  state.captureHolding = true;
  ui.captureButton.classList.add("is-recording");
  ui.captureButtonText.textContent = "Capturing";
  requestCapture();
  state.captureTimer = window.setInterval(() => {
    if (!state.captureHolding) return;
    requestCapture();
  }, CAPTURE_HOLD_INTERVAL_MS);
}

function stopCaptureHold() {
  if (!state.captureHolding) return;
  state.captureHolding = false;
  window.clearInterval(state.captureTimer);
  state.captureTimer = null;
  ui.captureButton.classList.remove("is-recording");
  ui.captureButtonText.textContent = "Capture";
}

function updateCaptureAvailability() {
  ui.captureButton.disabled = !sourceReady();
  updateLineAvailability();
}

// ---------------------------------------------------------------------------
// Sending to LINE
//
// Separate from Capture on purpose. Capture can be held down and takes a frame
// every 400 ms; tying a push to it would spend the account's monthly message
// quota by the handful. This is pressed once, deliberately, when a shot is worth
// telling someone about.
// ---------------------------------------------------------------------------

// At most four pictures fit in one push: LINE takes five messages per request
// and the first is the telemetry. A fifth picture is not a bigger message, it
// is a second request and a second message off the monthly allowance.
const MAX_SEND_SELECTION = 4;

function updateLineAvailability() {
  // Needs somewhere to send to and at least one ticked picture.
  const count = state.chosen.size || (state.selectedShot === null ? 0 : 1);
  const quota = state.lineQuota;
  const exhausted = Boolean(quota && quota.exhausted);
  ui.lineSendButton.disabled =
    state.lineSending || !state.lineConfigured || count === 0 || exhausted;

  // The button says what pressing it does, and nothing else. The allowance goes
  // in the label beside it.
  ui.lineSendButtonText.textContent = state.lineSending
    ? "กำลังส่ง..."
    : count > 1 ? `ส่ง ${count} รูปเข้า LINE` : "ส่งเข้า LINE";

  ui.lineSendButton.classList.toggle("is-warning", Boolean(quota && quota.warn) && !exhausted);
  ui.lineSendButton.classList.toggle("is-exhausted", exhausted);
  ui.lineSendButton.title = lineButtonTitle(count, quota);
  renderQuotaLabel(quota, exhausted);
}

function renderQuotaLabel(quota, exhausted) {
  const label = ui.lineQuotaLabel;
  // Nothing to say when the quota could not be read: the tooltip carries the
  // reason, and a blank number beside the button would look like zero.
  if (!quota || quota.error) {
    label.hidden = true;
    return;
  }
  // "เดือนนี้" every time, never just a bare number. The allowance resets
  // monthly and a count on its own reads as a daily one.
  label.textContent = quota.unlimited
    ? `เดือนนี้ส่งไป ${quota.used} · ไม่จำกัด`
    : `เดือนนี้เหลือ ${quota.remaining}/${quota.limit}`;
  label.classList.toggle("is-warning", Boolean(quota.warn) && !exhausted);
  label.classList.toggle("is-exhausted", Boolean(exhausted));
  label.hidden = false;
}

function lineButtonTitle(count, quota) {
  if (!state.lineConfigured) return state.lineHow || "ยังไม่ได้ตั้งค่า LINE";
  const lines = [`ส่งไปยัง ${state.lineTo || "ปลายทางที่ตั้งไว้"}`];
  if (quota && quota.error) {
    lines.push(quota.error);
  } else if (quota && quota.unlimited) {
    lines.push(`ส่งไปแล้วเดือนนี้ ${quota.used} ข้อความ (แพ็กเกจไม่จำกัด)`);
  } else if (quota && Number.isFinite(quota.remaining)) {
    lines.push(`โควต้ารายเดือน ${quota.used}/${quota.limit} · เหลือ ${quota.remaining}`);
    lines.push("นับรวมทั้งเดือน ไม่ใช่ต่อวัน และรีเซ็ตเมื่อขึ้นเดือนใหม่");
    // Worth spelling out where the button is pressed: someone about to tick a
    // fourth picture is reasonably worried it costs four.
    lines.push("การส่ง 1 ครั้งนับเป็น 1 ข้อความ ไม่ว่าจะแนบกี่รูป");
  }
  if (count === 0) lines.push("เลือกภาพที่จะส่งก่อน");
  return lines.join("\n");
}

function applyLineQuota(quota) {
  if (!quota) return;
  state.lineQuota = quota;
  if (quota.exhausted) {
    setCaptureStatus(
      `โควต้า LINE ของเดือนนี้เต็มแล้ว (${quota.used}/${quota.limit} ข้อความ)`
      + " จะได้โควต้าใหม่เมื่อขึ้นเดือนถัดไป",
      true,
    );
  } else if (quota.warn) {
    setCaptureStatus(
      `โควต้า LINE เหลือ ${quota.remaining} ข้อความสำหรับทั้งเดือนนี้`,
      true,
    );
  }
  updateLineAvailability();
}

// ---------------------------------------------------------------------------
// Capture preview
//
// This panel used to show the live reading as it streamed past. It does not any
// more, because that is no longer what gets sent: OCR and detection both run on
// the captured frame, so the thing worth looking at is the frame -- and which
// of them you are about to send.
// ---------------------------------------------------------------------------

async function refreshShots(selectIndex = null) {
  if (!state.captureSession) return;
  try {
    const params = new URLSearchParams({ session: state.captureSession });
    const body = await jsonFetch(`/api/capture/shots?${params}`);
    state.shots = body.shots || [];
  } catch (error) {
    setCaptureStatus(`อ่านรายการภาพไม่สำเร็จ: ${error.message}`, true);
    return;
  }
  // A shot that is gone cannot be sent, so it cannot stay ticked either.
  const present = new Set(state.shots.map((shot) => shot.frame_index));
  for (const index of [...state.chosen]) {
    if (!present.has(index)) state.chosen.delete(index);
  }
  renderShotStrip();
  // Newest first, so index 0 is the shot just taken.
  const wanted = selectIndex === null ? state.shots[0]?.frame_index : selectIndex;
  if (wanted !== undefined) selectShot(wanted);
  const hasShots = state.shots.length > 0;
  ui.downloadJsonButton.disabled = !hasShots;
  ui.downloadCsvButton.disabled = !hasShots;
  ui.clearLogButton.disabled = !hasShots;
  updateShotProgress();
}

// What the captures are doing, said in the panel.
//
// The endpoint does not answer until the frame has been saved, put through
// best.pt and re-read, so a shot that is listed is a shot that finished. What
// is missing from the list is what is still inside the model, and that is the
// part the operator could not see at all before.
function updateShotProgress() {
  const total = state.shots.length;
  const working = state.pendingCaptures + state.captureQueued;
  const unprocessed = state.shots.filter(
    (shot) => shot.stage && shot.stage !== "ready",
  ).length;
  ui.recordCount.textContent = `${total} ภาพ`;
  ui.shotTotal.textContent = `${total} ภาพ`;
  ui.shotWorking.hidden = working === 0;
  ui.shotWorkingText.textContent = state.captureQueued
    ? `กำลังผ่านโมเดล ${state.pendingCaptures} · รออีก ${state.captureQueued}`
    : `กำลังผ่านโมเดล ${state.pendingCaptures}`;
  const ready = total - unprocessed;
  ui.shotReady.hidden = ready === 0;
  ui.shotReady.textContent = `พร้อมส่ง ${ready}`;
  ui.shotUnprocessed.hidden = unprocessed === 0;
  ui.shotUnprocessed.textContent = `ไม่ได้ผ่านโมเดล ${unprocessed}`;
  // Counted rather than only flashed in the status line: a failure during a
  // long hold scrolls past, and the operator is left believing they have more
  // frames than they got.
  ui.shotFailed.hidden = state.captureFailures === 0;
  ui.shotFailed.textContent = `เก็บไม่สำเร็จ ${state.captureFailures}`;
  ui.shotSelectBar.hidden = total === 0;
  ui.shotSelectedCount.textContent =
    `เลือกส่ง ${state.chosen.size} รูป (สูงสุด ${MAX_SEND_SELECTION})`;
  ui.shotClearSelection.disabled = state.chosen.size === 0;
  ui.shotDeleteSelection.disabled = state.chosen.size === 0 || state.deleting;
  updateLineAvailability();
}

// ---------------------------------------------------------------------------
// Throwing a capture away
//
// Asked about first, and only once: this removes the picture, its marked-up
// copy and every telemetry line that named them. A record pointing at a file
// that is gone would be worse than no record, because `extract` would read the
// session back and report a frame nobody can look at.
// ---------------------------------------------------------------------------

async function deleteChosenShots() {
  if (state.deleting || state.chosen.size === 0) return;
  const frames = [...state.chosen].sort((a, b) => a - b);
  const asked = window.confirm(
    `ลบ ${frames.length} รูปนี้ถาวร?\n`
    + `เฟรม ${frames.map((index) => `#${index}`).join(", ")}\n\n`
    + "ทั้งรูป รูปที่ตีกรอบ และ telemetry ของรูปนั้นจะถูกลบ กู้คืนไม่ได้",
  );
  if (!asked) return;
  state.deleting = true;
  updateShotProgress();
  try {
    const params = new URLSearchParams({ session: state.captureSession });
    for (const index of frames) params.append("frame_index", String(index));
    const result = await jsonFetch(`/api/capture/shot?${params}`, { method: "DELETE" });
    state.chosen.clear();
    if (frames.includes(state.selectedShot)) {
      state.selectedShot = null;
      ui.shotPreview.hidden = true;
      closeShotOverlay();
      resetCurrentValues();
    }
    setCaptureStatus(`ลบแล้ว ${result.removed.length} รูป · เหลือ ${result.remaining} ภาพ`);
    await refreshShots(state.selectedShot);
  } catch (error) {
    setCaptureStatus(`ลบไม่สำเร็จ: ${error.message}`, true);
  } finally {
    state.deleting = false;
    updateShotProgress();
  }
}

// ---------------------------------------------------------------------------
// The enlarged view
//
// Over the video rather than beside it, because that is the only space on the
// page big enough to read an OSD in -- and reading the OSD against the rows
// below is the whole way of telling whether the extraction was right.
// ---------------------------------------------------------------------------

function openShotOverlay() {
  const shot = state.shots.find((item) => item.frame_index === state.selectedShot);
  if (!shot) return;
  ui.shotOverlayImage.src = ui.shotImage.getAttribute("src");
  ui.shotOverlayLabel.textContent =
    `เฟรม #${shot.frame_index} · ${describeShotDetections(shot.detections)}`;
  ui.shotOverlay.hidden = false;
}

function closeShotOverlay() {
  ui.shotOverlay.hidden = true;
  // Released so the browser is not holding a full frame for a hidden element.
  ui.shotOverlayImage.removeAttribute("src");
}

function renderShotStrip() {
  ui.shotStrip.replaceChildren();
  // The ones still in the model go first, where the newest shot would be, so a
  // press moves something on screen immediately.
  for (let i = 0; i < state.pendingCaptures + state.captureQueued; i += 1) {
    const card = document.createElement("div");
    card.className = "shot-thumb is-pending";
    const spinner = document.createElement("span");
    spinner.className = "shot-spinner";
    card.append(spinner);
    const label = document.createElement("span");
    label.className = "shot-thumb-label";
    label.textContent = i < state.pendingCaptures ? "กำลังผ่านโมเดล" : "รอคิว";
    card.append(label);
    ui.shotStrip.append(card);
  }
  for (const shot of state.shots) ui.shotStrip.append(buildShotThumb(shot));
}

function buildShotThumb(shot) {
  const card = document.createElement("div");
  card.className = "shot-thumb";
  card.dataset.frameIndex = String(shot.frame_index);
  card.setAttribute("role", "option");
  card.classList.toggle("is-selected", shot.frame_index === state.selectedShot);
  card.classList.toggle("is-chosen", state.chosen.has(shot.frame_index));

  // The picture is a button: pressing it shows that frame enlarged.
  const open = document.createElement("button");
  open.type = "button";
  open.className = "shot-thumb-open";
  // The thumbnail is the annotated copy when there is one, so the strip shows
  // what was found without having to open each frame, and it is the small copy
  // rather than the full frame -- a session of a hundred would otherwise have
  // the browser decoding a 1920x1080 bitmap per postage stamp.
  open.style.backgroundImage =
    `url("${shotImageUrl(shot, Boolean(shot.annotated), true)}")`;
  open.setAttribute("aria-label", `ดูเฟรม ${shot.frame_index}`);
  open.addEventListener("click", () => selectShot(shot.frame_index));
  card.append(open);

  const tick = document.createElement("label");
  tick.className = "shot-thumb-tick";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = state.chosen.has(shot.frame_index);
  box.setAttribute("aria-label", `เลือกเฟรม ${shot.frame_index} เพื่อส่ง`);
  box.addEventListener("change", () => toggleChosen(shot.frame_index, box));
  tick.append(box);
  card.append(tick);

  const label = document.createElement("span");
  label.className = "shot-thumb-label";
  label.textContent = `#${shot.frame_index}`;
  card.append(label);

  // Said on the thumbnail rather than only in the enlarged view, because a
  // frame the model never saw looks exactly like a frame it saw and cleared.
  if (shot.stage && shot.stage !== "ready") {
    const badge = document.createElement("span");
    badge.className = "shot-thumb-badge";
    badge.textContent = shot.stage === "no_model" ? "ไม่มีโมเดล" : "โมเดลไม่ตอบ";
    card.append(badge);
  }
  return card;
}

function toggleChosen(frameIndex, box) {
  if (box.checked) {
    if (state.chosen.size >= MAX_SEND_SELECTION) {
      box.checked = false;
      setCaptureStatus(
        `เลือกได้สูงสุด ${MAX_SEND_SELECTION} รูปต่อการส่ง 1 ครั้ง`
        + " (LINE รับ 5 ข้อความต่อ request และข้อความแรกคือ telemetry)",
        true,
      );
      return;
    }
    state.chosen.add(frameIndex);
  } else {
    state.chosen.delete(frameIndex);
  }
  const card = box.closest(".shot-thumb");
  if (card) card.classList.toggle("is-chosen", box.checked);
  updateShotProgress();
}

function clearChosen() {
  state.chosen.clear();
  for (const card of ui.shotStrip.children) {
    card.classList.remove("is-chosen");
    const box = card.querySelector("input[type=checkbox]");
    if (box) box.checked = false;
  }
  updateShotProgress();
}

function shotImageUrl(shot, annotated, thumb = false) {
  const params = new URLSearchParams({
    session: state.captureSession,
    frame_index: String(shot.frame_index),
  });
  if (annotated) params.set("annotated", "1");
  if (thumb) params.set("thumb", "1");
  // Cache-busted on captured_at rather than on now(): the same frame keeps one
  // URL, so switching back to it does not refetch.
  params.set("v", shot.captured_at || "");
  return `/api/capture/image?${params}`;
}

function selectShot(frameIndex) {
  const shot = state.shots.find((item) => item.frame_index === frameIndex);
  if (!shot) return;
  state.selectedShot = frameIndex;
  for (const card of ui.shotStrip.children) {
    card.classList.toggle(
      "is-selected", Number(card.dataset.frameIndex) === frameIndex,
    );
  }
  ui.shotPreview.hidden = false;
  // Only offer the boxes when there are boxes to show.
  ui.shotAnnotated.disabled = !shot.annotated;
  const wantsBoxes = ui.shotAnnotated.checked && Boolean(shot.annotated);
  ui.shotImage.src = shotImageUrl(shot, wantsBoxes);
  if (!ui.shotOverlay.hidden) openShotOverlay();
  ui.shotDetections.textContent = describeShotDetections(shot.detections);
  renderShotValues(shot);
  updateLineAvailability();
}

function describeShotDetections(detections) {
  if (!detections) return "";
  if (detections.status === "no_model") {
    return "ยังไม่ได้โหลดโมเดล — ภาพนี้ไม่ได้ผ่าน best.pt";
  }
  if (detections.status === "no_result") return "โมเดลไม่ตอบสำหรับภาพนี้";
  const counts = Object.entries(detections.counts || {});
  if (!counts.length) return "ผ่านโมเดลแล้ว · ไม่พบวัตถุ";
  return counts
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([name, count]) => {
      const stat = (detections.confidence || {})[name];
      return stat ? `${name} ${count} (${stat.mean.toFixed(2)})` : `${name} ${count}`;
    })
    .join(", ");
}

function renderShotValues(shot) {
  resetCurrentValues();
  for (const record of shot.records || []) updateTelemetryRow(record);
}

async function checkLineConfigured() {
  try {
    const status = await jsonFetch("/api/line/status");
    state.lineConfigured = Boolean(status.configured);
    state.lineTo = status.to || "";
    state.lineHow = status.how || "";
    state.lineQuota = status.quota || null;
  } catch {
    // Not configured and not reachable look the same from here, and neither is
    // a reason to hold up the rest of the page.
    state.lineConfigured = false;
    state.lineQuota = null;
  }
  updateLineAvailability();
  // Said once on load when it is already low, rather than only at the press.
  // Whoever is about to record a sortie should know before they start.
  const quota = state.lineQuota;
  if (quota && (quota.warn || quota.exhausted)) applyLineQuota(quota);
}

async function sendSelectedToLine() {
  if (state.lineSending || !state.captureSession) return;
  state.lineSending = true;
  ui.lineSendButtonText.textContent = "กำลังส่ง...";
  updateLineAvailability();
  try {
    const params = new URLSearchParams({ session: state.captureSession });
    // What is ticked, or -- when nothing is -- the one being looked at. Both are
    // a deliberate choice of frame; neither is "whatever was captured last".
    const frames = state.chosen.size
      ? [...state.chosen].sort((a, b) => a - b)
      : state.selectedShot === null ? [] : [state.selectedShot];
    for (const index of frames) params.append("frame_index", String(index));
    const result = await jsonFetch(`/api/line/send?${params}`, { method: "POST" });
    const sent = result.frame_indexes || [result.frame_index];
    const withImage = result.images ? ` พร้อมรูป ${result.images} รูป` : " (ข้อความอย่างเดียว)";
    setCaptureStatus(
      `ส่งเข้า LINE แล้ว${withImage} · เฟรม ${sent.map((i) => `#${i}`).join(", ")}`
      + ` · ${result.characters} ตัวอักษร`,
    );
    if (result.share_error) setCaptureStatus(`ส่งแล้ว แต่: ${result.share_error}`, true);
    // After the status line above, so a quota that has just crossed into the
    // warning band is the last thing said rather than the first.
    if (result.quota) state.lineQuota = result.quota;
    if (result.quota && (result.quota.warn || result.quota.exhausted)) {
      applyLineQuota(result.quota);
    }
  } catch (error) {
    // The server passes LINE's own wording through, because a wrong token, a
    // wrong destination and an exhausted quota are fixed in three different
    // places and "ส่งไม่สำเร็จ" would hide which one happened.
    setCaptureStatus(`ส่งเข้า LINE ไม่สำเร็จ: ${error.message}`, true);
    if (error.payload && error.payload.quota) state.lineQuota = error.payload.quota;
  } finally {
    state.lineSending = false;
    // The button label is not restored here. updateLineAvailability owns it and
    // has to run anyway to re-enable the button and repaint the quota beside it,
    // so writing "ส่งเข้า LINE" first would only be overwritten a line later.
    updateLineAvailability();
  }
}

// The realtime read is gone from this page entirely.
//
// It used to montage every ROI onto a canvas and POST that to
// /api/session/<id>/frame on a timer. Two things retired it: the panel it kept
// current now shows a chosen capture instead, and the reading that actually
// gets sent is taken from the saved image with the fallback recipes on, which
// the per-frame path could never afford. What remained was a function nothing
// called, spending nothing but still describing a design the page had left.
//
// The endpoint itself stays -- it is a documented part of the HTTP API and has
// its own tests; it simply has no caller in this file any more.

// Kept for the downloadable log only. The panel it used to fill now shows a
// chosen capture instead, because that is what gets sent.
function applyOcrResult(payload) {
  for (const record of payload.records) state.records.push(record);
  if (state.records.length > 200000) state.records.splice(0, state.records.length - 200000);
}

function updateTelemetryRow(record) {
  const nodes = state.rowElements.get(record.field_id);
  if (!nodes) return;
  nodes.row.classList.toggle("is-confirmed", Boolean(record.confirmed));
  nodes.row.classList.toggle("is-pending", !record.confirmed);
  const isCoordinate = record.field_id.includes("lat_dms") || record.field_id.includes("lon_dms");
  nodes.value.textContent = formatValue(record.value, record.field_id, nodes.unit);
  const confidence = Number.isFinite(record.confidence) ? `${(record.confidence * 100).toFixed(0)}%` : "—";
  const raw = record.raw_text || "no text";
  // A capture's reading has no preprocess variant: it is one pass over the file
  // that was saved, not the pick of several tries. Printing a field the reading
  // does not carry as "undefined" is the panel claiming to know something it
  // does not.
  const variant = record.preprocess_variant ? ` · ${record.preprocess_variant}` : "";
  nodes.detail.textContent = isCoordinate
    ? `conf ${confidence}${variant}`
    : `raw: ${raw} · conf ${confidence}${variant}`;
}

function formatValue(value, fieldId, unit) {
  if (value === null || value === undefined || value === "") return "—";
  if (fieldId === "timestamp") return String(value);
  if (fieldId === "eo_detail") {
    return String(value).replace("xFOV", "x FOV ");
  }
  if (fieldId === "ir_detail") return String(value);
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  if (fieldId === "sensor_angle_deg") {
    return `${numeric.toFixed(2)}°`;
  }
  if (fieldId === "elevation_deg") {
    return `${numeric.toFixed(2)}°`;
  }
  if (fieldId.includes("lat_dms") || fieldId.includes("lon_dms")) {
    const fieldConfig = state.config?.fields?.find((field) => field.field_id === fieldId);
    const degreeDigits = Number(fieldConfig?.dms_degrees_digits)
      || (fieldId.includes("lon_dms") ? 3 : 2);
    const decimalPlaces = Number.isInteger(fieldConfig?.dms_decimal_places)
      ? fieldConfig.dms_decimal_places
      : 2;
    return formatDms(numeric, degreeDigits, decimalPlaces);
  }
  if (fieldId === "lrf_m") return `${numeric.toFixed(1)} m`;
  if (fieldId.includes("alt_m")) return `${numeric.toFixed(2)} m`;
  const text = Number.isInteger(numeric) ? numeric.toString() : numeric.toFixed(2);
  return unit ? `${text} ${unit}` : text;
}

function formatDms(value, degreeDigits, decimalPlaces) {
  return DetectionMath.formatDms(value, degreeDigits, decimalPlaces).text;
}

function downloadJsonl() {
  const content = state.records.map((record) => JSON.stringify(record)).join("\n") + "\n";
  downloadBlob(content, "application/x-ndjson", buildDownloadName("jsonl"));
}

function downloadCsv() {
  const columns = ["source_time_ms", "frame_index", "field_id", "value", "raw_text", "confidence", "confirmed", "preprocess_variant", "latency_ms"];
  const lines = [columns.join(",")];
  for (const record of state.records) lines.push(columns.map((key) => csvCell(record[key])).join(","));
  downloadBlob(lines.join("\r\n") + "\r\n", "text/csv;charset=utf-8", buildDownloadName("csv"));
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function buildDownloadName(extension) {
  const base = (state.file?.name || "telemetry").replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "_");
  return `${base}_ocr.${extension}`;
}

function downloadBlob(content, type, name) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function updatePlaybackTime() {
  ui.currentTime.textContent = formatTime(ui.video.currentTime || 0);
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00.000";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds % 3600 / 60);
  const secs = Math.floor(seconds % 60);
  const millis = Math.floor((seconds % 1) * 1000);
  const prefix = hours ? `${String(hours).padStart(2, "0")}:` : "";
  return `${prefix}${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

ui.chooseVideoButton.addEventListener("click", () => ui.videoInput.click());
// The browser's own fullscreen still works on the video element itself; the
// overlay is a sibling of it rather than a child, so it does not come along.
// Redraw whenever anything leaves fullscreen so the boxes match the new size.
document.addEventListener("fullscreenchange", () => drawDetections());
ui.videoInput.addEventListener("change", () => handleFile(ui.videoInput.files[0]));
ui.streamButton.addEventListener("click", () => toggleStream());
ui.shotClearSelection.addEventListener("click", () => clearChosen());
ui.shotDeleteSelection.addEventListener("click", () => deleteChosenShots());
ui.shotImage.addEventListener("click", () => openShotOverlay());
ui.shotOverlayClose.addEventListener("click", () => closeShotOverlay());
ui.shotOverlay.addEventListener("click", (event) => {
  // Only the backdrop closes it; clicking the picture itself is how you look
  // at the picture.
  if (event.target === ui.shotOverlay) closeShotOverlay();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !ui.shotOverlay.hidden) closeShotOverlay();
});
ui.captureButton.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  startCaptureHold();
});
for (const name of ["pointerup", "pointerleave", "pointercancel"]) {
  ui.captureButton.addEventListener(name, () => stopCaptureHold());
}
// A pointer released anywhere else still ends the hold; without this, releasing
// outside the button leaves it capturing with no visible cause.
window.addEventListener("pointerup", () => stopCaptureHold());
ui.lineSendButton.addEventListener("click", () => sendSelectedToLine());
ui.shotAnnotated.addEventListener("change", () => {
  if (state.selectedShot !== null) selectShot(state.selectedShot);
});
ui.captureButton.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    requestCapture();
  }
});
ui.streamUrl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") toggleStream();
});
// The MJPEG response is one long-lived request; if it fails the picture simply
// stops, with nothing in the console to say why.
ui.streamImage.addEventListener("error", () => {
  if (state.streamMode) showToast("\u0e20\u0e32\u0e1e\u0e2a\u0e15\u0e23\u0e35\u0e21\u0e2b\u0e25\u0e38\u0e14 \u2014 \u0e01\u0e33\u0e25\u0e31\u0e07\u0e23\u0e2d\u0e15\u0e48\u0e2d\u0e43\u0e2b\u0e21\u0e48");
});
// Closing the tab deliberately does NOT stop the stream. The server is the one
// reading the camera and writing the telemetry file, and an auto-started server
// is expected to keep doing that with no browser open at all. A beacon could not
// have stopped it anyway: sendBeacon cannot set the guard header, so the request
// would be refused.
ui.video.addEventListener("loadedmetadata", () => {
  ui.durationTime.textContent = formatTime(ui.video.duration);
  ui.fileDetails.textContent = `${formatBytes(state.file.size)} · ${ui.video.videoWidth}×${ui.video.videoHeight} · ${formatTime(ui.video.duration)}`;
  updateCaptureAvailability();
  ui.resetButton.disabled = false;
  drawDetections();
  startFrameLoop();
  startDetectionRenderLoop();
});
ui.video.addEventListener("loadeddata", async () => {
  if (!state.detectorLoaded || !ui.detectionToggle.checked) return;
  await resetDetectionTracker();
  submitDetectionFrame(true);
});
ui.video.addEventListener("pause", () => {
  updatePlaybackTime();
  // Playing a video no longer starts OCR; capturing does.
  if (ui.detectionToggle.checked) submitDetectionFrame(true);
});
ui.video.addEventListener("timeupdate", updatePlaybackTime);
ui.video.addEventListener("seeked", async () => {
  ++state.detectionGeneration;
  state.detectionSequence = 0;
  await resetDetectionTracker();
  resetSession(false);
  if (state.detectorLoaded) submitDetectionFrame(true);
});
ui.video.addEventListener("error", () => showToast("Browser ไม่สามารถเล่นวิดีโอนี้ได้ ลองใช้ MP4 (H.264) หรือ WebM"));
ui.modelInput.addEventListener("change", () => uploadDetectionModel(ui.modelInput.files[0]));
ui.detectionToggle.addEventListener("change", () => {
  state.lastDetectionMs = -Infinity;
  drawDetections();
  if (ui.detectionToggle.checked && ui.video.readyState >= 2) submitDetectionFrame(true);
});
ui.detectionRate.addEventListener("change", () => { state.lastDetectionMs = -Infinity; });
ui.modelConfidence.addEventListener("change", async () => {
  const value = Math.min(1, Math.max(0.05, Number(ui.modelConfidence.value) || 0.45));
  ui.modelConfidence.value = value.toFixed(2);
  ++state.detectionGeneration;
  state.detectionSequence = 0;
  await resetDetectionTracker();
  if (state.detectorLoaded && ui.video.readyState >= 2) submitDetectionFrame(true);
});
ui.modelImageSize.addEventListener("change", async () => {
  // A deliberate choice wins over the trained-size default from then on.
  state.detectionImageSizeTouched = true;
  await applyImageSizeChange();
});
// Only changes what the overlay writes, so no tracker reset and no need to wait
// for another detection result -- the render loop picks it up on the next frame.
ui.trackIdToggle.addEventListener("change", () => drawDetections());
ui.restartWorkerButton.addEventListener("click", restartDetectionWorker);
ui.clearModelButton.addEventListener("click", clearDetectionModel);
ui.resetButton.addEventListener("click", () => resetSession(true));
ui.downloadJsonButton.addEventListener("click", downloadJsonl);
ui.downloadCsvButton.addEventListener("click", downloadCsv);
ui.clearLogButton.addEventListener("click", clearLog);

for (const eventName of ["dragenter", "dragover"]) {
  ui.videoStage.addEventListener(eventName, (event) => {
    event.preventDefault();
    ui.videoStage.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  ui.videoStage.addEventListener(eventName, (event) => {
    event.preventDefault();
    ui.videoStage.classList.remove("is-dragging");
  });
}
ui.videoStage.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));
// Wrapped, not passed directly: both callbacks supply an argument (an entries
// array, an Event), which would land in drawDetections's renderNow parameter --
// a default only applies to undefined -- and turn the fade arithmetic into NaN,
// repainting expiring boxes at full opacity instead of fading them.
if ("ResizeObserver" in window) new ResizeObserver(() => drawDetections()).observe(ui.videoStage);
else window.addEventListener("resize", () => drawDetections());
window.addEventListener("beforeunload", () => {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  if (state.fallbackTimer) window.clearInterval(state.fallbackTimer);
  stopDetectionResultPolling();
  stopModelStatusPolling();
  if (state.detectionEncoder) state.detectionEncoder.terminate();
  if (state.detectionRenderFrame !== null) window.cancelAnimationFrame(state.detectionRenderFrame);
});

initializePwa();
startDetectionRenderLoop();
initialize();
