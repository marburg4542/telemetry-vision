"use strict";

const $ = (id) => document.getElementById(id);

const ui = {
  videoInput: $("videoInput"),
  chooseVideoButton: $("chooseVideoButton"),
  videoStage: $("videoStage"),
  video: $("sourceVideo"),
  dropState: $("dropState"),
  processingBadge: $("processingBadge"),
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
  ocrToggle: $("ocrToggle"),
  rateSelect: $("rateSelect"),
  resetButton: $("resetButton"),
  engineStatus: $("engineStatus"),
  engineStatusText: $("engineStatusText"),
  telemetryList: $("telemetryList"),
  ocrRate: $("ocrRate"),
  latency: $("latency"),
  confirmedRate: $("confirmedRate"),
  recordCount: $("recordCount"),
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
  inFlight: false,
  generation: 0,
  sequence: 0,
  lastSubmittedMs: -Infinity,
  records: [],
  completions: [],
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
  detectionModelPollTimer: null,
  detectionRenderFrame: null,
  detectionSourceWidth: 0,
  detectionSourceHeight: 0,
  detectionCaptureMs: null,
  detectionPostMs: null,
  detectionTrainedImageSize: null,
  detectionDevice: "",
  detectionImageSizeTouched: false,
};

const DETECTION_MAX_EXTRAPOLATION_MS = 250;
const DETECTION_STALE_MS = 750;
const DETECTION_FADE_MS = 150;

function initializePwa() {
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/service-worker.js").catch((error) => {
        console.warn("Service worker registration failed", error);
      });
    });
  }
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
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
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
  const hasModel = Boolean(payload.filename);
  ui.detectionToggle.disabled = !state.detectorLoaded;
  ui.detectionRate.disabled = !state.detectorLoaded;
  ui.modelConfidence.disabled = !state.detectorLoaded;
  ui.modelImageSize.disabled = !state.detectorLoaded;
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
  const options = Array.from(ui.modelImageSize.options);
  if (!options.some((option) => Number(option.value) === trained)) {
    const option = document.createElement("option");
    option.value = String(trained);
    option.textContent = `${trained} px`;
    const insertBefore = options.find((candidate) => Number(candidate.value) > trained);
    ui.modelImageSize.insertBefore(option, insertBefore || null);
  }

  if (state.detectionImageSizeTouched) return;

  // Running below the trained size costs small-object accuracy, but running a
  // 960 px model on the CPU costs well over a second per frame. Cap the
  // automatic choice where the README's own benchmark stops being usable.
  const onCpu = state.detectionDevice.toLowerCase().startsWith("cpu");
  const target = onCpu ? Math.min(trained, 640) : trained;
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
    chooseNearestRate(state.config.target_hz);
    setEngineStatus("ready", `${state.config.model} พร้อมใช้งาน`);
    initializeDetectionModel();
  } catch (error) {
    setEngineStatus("error", "เชื่อมต่อ OCR ไม่สำเร็จ");
    showToast(`เริ่มระบบไม่ได้: ${error.message}`);
  }
}

function chooseNearestRate(target) {
  const options = [...ui.rateSelect.options];
  const closest = options.reduce((best, option) => (
    Math.abs(Number(option.value) - Number(target)) < Math.abs(Number(best.value) - Number(target))
      ? option : best
  ), options[0]);
  ui.rateSelect.value = closest.value;
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
  state.detectionSourceId = globalThis.crypto?.randomUUID?.()
    || `video-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  ++state.detectionGeneration;
  state.detectionSequence = 0;
  resetDetectionFrame();
  resetDetectionTracker();
  resetSession(true);
}

async function resetSession(clearRecords = true) {
  const generation = ++state.generation;
  state.sequence = 0;
  state.lastSubmittedMs = -Infinity;
  state.completions = [];
  if (clearRecords) clearLog();
  resetCurrentValues();
  if (!state.sessionId) return;
  try {
    await jsonFetch(`/api/session/${state.sessionId}/reset`, { method: "POST" });
    if (generation !== state.generation) return;
    setEngineStatus("ready", `${state.config.model} พร้อมใช้งาน`);
    if (ui.video.readyState >= 2 && ui.ocrToggle.checked) submitFrame(true);
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
  ui.ocrRate.textContent = "0.0";
  ui.latency.textContent = "—";
  ui.confirmedRate.textContent = "0";
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
    const tick = (_, metadata) => {
      maybeSubmit(metadata.mediaTime * 1000, metadata.presentedFrames);
      maybeSubmitDetection(metadata.mediaTime * 1000);
      ui.video.requestVideoFrameCallback(tick);
    };
    ui.video.requestVideoFrameCallback(tick);
  } else {
    state.fallbackTimer = window.setInterval(() => {
      if (!ui.video.paused) {
        const sourceTimeMs = ui.video.currentTime * 1000;
        maybeSubmit(sourceTimeMs, state.sequence);
        maybeSubmitDetection(sourceTimeMs);
      }
    }, 35);
  }
}

function maybeSubmitDetection(sourceTimeMs) {
  if (
    ui.video.paused
    || ui.video.ended
    || !state.detectorLoaded
    || !ui.detectionToggle.checked
    || state.detectionInFlight
  ) return;
  const intervalMs = 1000 / Number(ui.detectionRate.value);
  if (sourceTimeMs - state.lastDetectionMs < intervalMs) return;
  submitDetectionFrame(false, sourceTimeMs);
}

async function submitDetectionFrame(force = false, sourceTimeMs = ui.video.currentTime * 1000) {
  if (!state.detectorLoaded || state.detectionInFlight || ui.video.readyState < 2) return;
  if (!force && (ui.video.paused || !ui.detectionToggle.checked)) return;
  const generation = state.detectionGeneration;
  const sequence = state.detectionSequence++;
  state.detectionInFlight = true;
  state.lastDetectionMs = sourceTimeMs;
  try {
    const captureStarted = performance.now();
    const blob = await captureDetectionFrame();
    state.detectionCaptureMs = performance.now() - captureStarted;
    const confidence = Math.min(1, Math.max(0.05, Number(ui.modelConfidence.value) || 0.45));
    const params = new URLSearchParams({
      source_id: state.detectionSourceId,
      generation: String(generation),
      sequence: String(sequence),
      source_time_ms: sourceTimeMs.toFixed(3),
      confidence: confidence.toFixed(2),
      iou: "0.45",
      image_size: ui.modelImageSize.value,
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
    if (force || ui.video.paused) pollDetectionResult(sequence, generation);
  } catch (error) {
    if (generation === state.detectionGeneration) {
      setModelStatus(`AI detection ขัดข้อง: ${error.message}`, true);
      initializeDetectionModel();
    }
  } finally {
    state.detectionInFlight = false;
  }
}

function captureDetectionFrame() {
  const sourceWidth = ui.video.videoWidth;
  const sourceHeight = ui.video.videoHeight;
  const targetLongSide = Math.max(320, Number(ui.modelImageSize.value) || 640);
  const scale = Math.min(1, targetLongSide / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  ui.detectionCanvas.width = width;
  ui.detectionCanvas.height = height;
  const context = ui.detectionCanvas.getContext("2d", { alpha: false });
  context.drawImage(ui.video, 0, 0, width, height);
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
  ui.detectionFps.textContent = "0.0";
  ui.detectionLatency.textContent = "—";
  ui.detectionLag.textContent = "—";
  ui.detectionCapture.textContent = "—";
  ui.detectionHarvest.textContent = "—";
  drawDetections();
}

async function resetDetectionTracker() {
  resetDetectionFrame();
  window.clearTimeout(state.detectionPollTimer);
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
  state.detectionSourceWidth = Number(result.width) || ui.video.videoWidth;
  state.detectionSourceHeight = Number(result.height) || ui.video.videoHeight;
  const completedNow = performance.now();
  state.detectionCompletions.push(completedNow);
  state.detectionCompletions = state.detectionCompletions.filter((value) => completedNow - value <= 5000);
  const durationMs = state.detectionCompletions.length > 1
    ? completedNow - state.detectionCompletions[0]
    : 0;
  const fps = durationMs > 0
    ? (state.detectionCompletions.length - 1) * 1000 / durationMs
    : 0;
  const lagMs = Math.max(0, ui.video.currentTime * 1000 - Number(result.source_time_ms));
  const timing = payload.timing || {};
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
    `cap ${formatMs(state.detectionCaptureMs)}`,
    `post ${formatMs(state.detectionPostMs)}`,
    `infer ${formatMs(result.inference_ms)}`,
    `transport ${formatMs(timing.transport_ms)}`,
    `wait ${formatMs(timing.harvest_ms)}`,
  ].join(" · ");
  // Re-stated on every result: this line is rewritten continuously, so a
  // warning shown once at load time would vanish the moment boxes arrive.
  const mismatch = imageSizeMismatchWarning();
  setModelStatus(
    `${ui.modelName.textContent} · BoT-SORT · ${detectionSummary} · ${fps.toFixed(1)} FPS`
    + ` · lag ${Math.round(lagMs)} ms · ${breakdown}`
    + (mismatch ? ` · ${mismatch}` : ""),
    Boolean(mismatch),
  );
}

function formatMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? String(Math.round(number)) : "—";
}

function applyDetectionResult(result, completedNow) {
  const activeKeys = new Set();
  (result.detections || []).forEach((detection, index) => {
    const key = detection.track_id === null || detection.track_id === undefined
      ? `untracked:${result.sequence}:${index}`
      : `track:${detection.track_id}`;
    activeKeys.add(key);
    const existing = state.detectionTracks.get(key);
    const current = {
      ...detection,
      sourceTimeMs: Number(result.source_time_ms),
    };
    state.detectionTracks.set(key, {
      previous: existing?.current && existing.current.sourceTimeMs < current.sourceTimeMs
        ? existing.current
        : existing?.previous || null,
      current,
      missingSince: null,
      untracked: detection.track_id === null || detection.track_id === undefined,
    });
  });
  for (const [key, track] of state.detectionTracks) {
    if (!activeKeys.has(key) && track.missingSince === null) {
      track.missingSince = completedNow;
    }
  }
}

function pollDetectionResult(targetSequence, generation, attempts = 0) {
  window.clearTimeout(state.detectionPollTimer);
  if (generation !== state.detectionGeneration || !state.detectorLoaded) return;
  state.detectionPollTimer = window.setTimeout(async () => {
    try {
      const params = new URLSearchParams({
        source_time_ms: (ui.video.currentTime * 1000).toFixed(3),
      });
      const payload = await jsonFetch(`/api/detection/result?${params}`);
      if (generation !== state.detectionGeneration) return;
      applyDetectionPayload(payload);
      if (state.detectionResultSequence < targetSequence && attempts < 24) {
        pollDetectionResult(targetSequence, generation, attempts + 1);
      }
    } catch (error) {
      if (generation === state.detectionGeneration && attempts < 8) {
        pollDetectionResult(targetSequence, generation, attempts + 1);
      }
    }
  }, 120);
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
  if (!state.detectorLoaded || !ui.detectionToggle.checked || !state.detectionTracks.size) return;

  const sourceWidth = state.detectionSourceWidth || ui.video.videoWidth;
  const sourceHeight = state.detectionSourceHeight || ui.video.videoHeight;
  if (!sourceWidth || !sourceHeight) return;
  const videoRatio = sourceWidth / sourceHeight;
  const stageRatio = stageWidth / stageHeight;
  const displayWidth = videoRatio > stageRatio ? stageWidth : stageHeight * videoRatio;
  const displayHeight = videoRatio > stageRatio ? stageWidth / videoRatio : stageHeight;
  const offsetX = (stageWidth - displayWidth) / 2;
  const offsetY = (stageHeight - displayHeight) / 2;
  const scaleX = displayWidth / sourceWidth;
  const scaleY = displayHeight / sourceHeight;
  context.lineWidth = 2;
  context.font = "600 12px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.textBaseline = "top";

  const currentMediaTimeMs = ui.video.currentTime * 1000;
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  for (const [key, track] of state.detectionTracks) {
    let opacity = 1;
    if (track.missingSince !== null) {
      opacity = Math.max(0, 1 - (renderNow - track.missingSince) / DETECTION_FADE_MS);
      if (opacity <= 0) {
        state.detectionTracks.delete(key);
        continue;
      }
    }
    const detection = track.current;
    const mediaAgeMs = currentMediaTimeMs - detection.sourceTimeMs;
    if (mediaAgeMs > DETECTION_STALE_MS || mediaAgeMs < -100) continue;
    const extrapolationMs = ui.video.paused || reduceMotion
      ? 0
      : Math.min(DETECTION_MAX_EXTRAPOLATION_MS, Math.max(0, mediaAgeMs));
    const previous = track.previous;
    const sampleDeltaMs = previous ? detection.sourceTimeMs - previous.sourceTimeMs : 0;
    const predict = (field) => {
      const velocity = previous && sampleDeltaMs > 0
        ? (Number(detection[field]) - Number(previous[field])) / sampleDeltaMs
        : 0;
      return Number(detection[field]) + velocity * extrapolationMs;
    };
    const x1 = Math.min(sourceWidth, Math.max(0, predict("x1")));
    const y1 = Math.min(sourceHeight, Math.max(0, predict("y1")));
    const x2 = Math.min(sourceWidth, Math.max(x1 + 1, predict("x2")));
    const y2 = Math.min(sourceHeight, Math.max(y1 + 1, predict("y2")));
    const x = offsetX + x1 * scaleX;
    const y = offsetY + y1 * scaleY;
    const width = Math.max(1, (x2 - x1) * scaleX);
    const height = Math.max(1, (y2 - y1) * scaleY);
    const hue = (Number(detection.class_id) * 67 + 88) % 360;
    const color = `hsl(${hue} 88% 62%)`;
    const trackLabel = detection.track_id === null || detection.track_id === undefined
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

function maybeSubmit(sourceTimeMs, suggestedFrame) {
  updatePlaybackTime();
  if (ui.video.paused || ui.video.ended || !ui.ocrToggle.checked || state.inFlight) return;
  const intervalMs = 1000 / Number(ui.rateSelect.value);
  if (sourceTimeMs - state.lastSubmittedMs < intervalMs) return;
  submitFrame(false, sourceTimeMs, suggestedFrame);
}

async function submitFrame(force = false, sourceTimeMs = ui.video.currentTime * 1000, suggestedFrame = null) {
  if (!state.sessionId || !state.config || state.inFlight || ui.video.readyState < 2) return;
  if (!force && !ui.ocrToggle.checked) return;

  const generation = state.generation;
  state.inFlight = true;
  state.lastSubmittedMs = sourceTimeMs;
  ui.processingBadge.hidden = false;
  const frameIndex = Number.isInteger(suggestedFrame) && suggestedFrame >= 0
    ? Math.max(state.sequence++, suggestedFrame)
    : state.sequence++;
  try {
    const blob = await captureRois();
    const params = new URLSearchParams({
      source_time_ms: sourceTimeMs.toFixed(3),
      frame_index: String(frameIndex),
    });
    const payload = await jsonFetch(
      `/api/session/${state.sessionId}/frame?${params}`,
      { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob },
    );
    if (generation === state.generation) applyOcrResult(payload);
  } catch (error) {
    if (generation === state.generation) {
      setEngineStatus("error", "OCR ขัดข้อง — กำลังลองเฟรมถัดไป");
      showToast(`OCR ไม่สำเร็จ: ${error.message}`);
    }
  } finally {
    state.inFlight = false;
    ui.processingBadge.hidden = ui.video.paused || !ui.ocrToggle.checked;
  }
}

function captureRois() {
  const width = ui.video.videoWidth;
  const height = ui.video.videoHeight;
  ui.canvas.width = width;
  ui.canvas.height = height;
  const context = ui.canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#000";
  context.fillRect(0, 0, width, height);

  for (const field of state.config.fields) {
    const [x, y, w, h] = field.box;
    const pad = Number(field.padding || 0);
    const left = Math.max(0, (x - w * pad) * width);
    const top = Math.max(0, (y - h * pad) * height);
    const right = Math.min(width, (x + w * (1 + pad)) * width);
    const bottom = Math.min(height, (y + h * (1 + pad)) * height);
    const roiWidth = right - left;
    const roiHeight = bottom - top;
    if (roiWidth > 0 && roiHeight > 0) {
      context.drawImage(ui.video, left, top, roiWidth, roiHeight, left, top, roiWidth, roiHeight);
    }
  }

  return new Promise((resolve, reject) => {
    ui.canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("สร้างภาพเฟรมไม่สำเร็จ")), "image/jpeg", 0.92);
  });
}

function applyOcrResult(payload) {
  setEngineStatus("ready", `${state.config.model} · OCR live`);
  ui.latency.textContent = Math.round(payload.latency_ms).toString();
  const now = performance.now();
  state.completions.push(now);
  state.completions = state.completions.filter((time) => now - time <= 4000).slice(-40);
  if (state.completions.length > 1) {
    const elapsed = (state.completions.at(-1) - state.completions[0]) / 1000;
    ui.ocrRate.textContent = ((state.completions.length - 1) / Math.max(elapsed, 0.001)).toFixed(1);
  }

  let confirmed = 0;
  for (const record of payload.records) {
    updateTelemetryRow(record);
    if (record.confirmed) confirmed += 1;
    state.records.push(record);
  }
  if (state.records.length > 200000) state.records.splice(0, state.records.length - 200000);
  ui.confirmedRate.textContent = payload.records.length
    ? Math.round(confirmed * 100 / payload.records.length).toString()
    : "0";
  ui.recordCount.textContent = `${state.records.length.toLocaleString()} records`;
  const hasRecords = state.records.length > 0;
  ui.downloadJsonButton.disabled = !hasRecords;
  ui.downloadCsvButton.disabled = !hasRecords;
  ui.clearLogButton.disabled = !hasRecords;
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
  nodes.detail.textContent = isCoordinate
    ? `conf ${confidence} · ${record.preprocess_variant}`
    : `raw: ${raw} · conf ${confidence} · ${record.preprocess_variant}`;
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
  const absolute = Math.abs(value);
  let degrees = Math.floor(absolute);
  const minuteValue = (absolute - degrees) * 60;
  let minutes = Math.floor(minuteValue);
  let seconds = (minuteValue - minutes) * 60;
  const scale = 10 ** decimalPlaces;
  seconds = Math.round(seconds * scale) / scale;
  if (seconds >= 60) {
    seconds = 0;
    minutes += 1;
  }
  if (minutes >= 60) {
    minutes = 0;
    degrees += 1;
  }
  const degreeText = String(degrees).padStart(degreeDigits, "0");
  const minuteText = String(minutes).padStart(2, "0");
  const secondText = seconds.toFixed(decimalPlaces).padStart(3 + decimalPlaces, "0");
  return `${degreeText}°${minuteText}'${secondText}"`;
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
ui.videoInput.addEventListener("change", () => handleFile(ui.videoInput.files[0]));
ui.video.addEventListener("loadedmetadata", () => {
  ui.durationTime.textContent = formatTime(ui.video.duration);
  ui.fileDetails.textContent = `${formatBytes(state.file.size)} · ${ui.video.videoWidth}×${ui.video.videoHeight} · ${formatTime(ui.video.duration)}`;
  ui.ocrToggle.disabled = false;
  ui.rateSelect.disabled = false;
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
ui.video.addEventListener("play", () => { ui.processingBadge.hidden = !ui.ocrToggle.checked; });
ui.video.addEventListener("pause", () => {
  ui.processingBadge.hidden = true;
  updatePlaybackTime();
  if (ui.ocrToggle.checked) submitFrame(true);
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
ui.ocrToggle.addEventListener("change", () => {
  if (ui.ocrToggle.checked && ui.video.readyState >= 2) submitFrame(true);
  else ui.processingBadge.hidden = true;
});
ui.rateSelect.addEventListener("change", () => { state.lastSubmittedMs = -Infinity; });
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
if ("ResizeObserver" in window) new ResizeObserver(drawDetections).observe(ui.videoStage);
else window.addEventListener("resize", drawDetections);
window.addEventListener("beforeunload", () => {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  if (state.fallbackTimer) window.clearInterval(state.fallbackTimer);
  window.clearTimeout(state.detectionPollTimer);
  stopModelStatusPolling();
  if (state.detectionRenderFrame !== null) window.cancelAnimationFrame(state.detectionRenderFrame);
});

initializePwa();
startDetectionRenderLoop();
initialize();
