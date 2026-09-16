"use strict";

const $ = (id) => document.getElementById(id);

const ui = {
  videoSelect: $("videoSelect"),
  viewSwitch: $("viewSwitch"),
  viewOriginal: $("viewOriginal"),
  viewRendered: $("viewRendered"),
  videoStage: $("videoStage"),
  video: $("sourceVideo"),
  dropState: $("dropState"),
  emptyStateText: $("emptyStateText"),
  captureButton: $("captureButton"),
  captureButtonText: $("captureButtonText"),
  captureStatus: $("captureStatus"),
  captureFlash: $("captureFlash"),
  lineSendButton: $("lineSendButton"),
  lineSendButtonText: $("lineSendButtonText"),
  lineQuotaLabel: $("lineQuotaLabel"),
  modelSelect: $("modelSelect"),
  manageModelsButton: $("manageModelsButton"),
  renderConfidence: $("renderConfidence"),
  renderImageSize: $("renderImageSize"),
  renderStride: $("renderStride"),
  renderButton: $("renderButton"),
  renderProgress: $("renderProgress"),
  renderBarFill: $("renderBarFill"),
  renderProgressText: $("renderProgressText"),
  renderCancelButton: $("renderCancelButton"),
  renderStatus: $("renderStatus"),
  renderList: $("renderList"),
  renderWait: $("renderWait"),
  aiPanel: $("aiPanel"),
  aiToggle: $("aiToggle"),
  aiToggleHint: $("aiToggleHint"),
  modelDialog: $("modelDialog"),
  modelDialogClose: $("modelDialogClose"),
  registeredModels: $("registeredModels"),
  candidateModels: $("candidateModels"),
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
  generation: 0,
  rowElements: new Map(),
  toastTimer: null,
  installPrompt: null,
  // The project's videos, the registered checkpoints, and the renders the
  // server holds for the video on screen.
  videos: [],
  video: null,
  models: [],
  renders: [],
  // The render the page plays or last chose for this video. Captures take their
  // boxes from it even while the original is on screen.
  activeRender: null,
  view: "original",
  // null until the server has said; false means nothing new can be rendered.
  ffmpeg: null,
  renderJob: null,
  renderPollTimer: null,
  // Frames a second this machine managed on its most recent finished render,
  // which is what the time estimate is made from. null until there is one.
  renderSpeed: null,
  renderSpeedAt: "",
  // The Media Source player a segmented render is watched through, how far that
  // render has got, and what the <video> currently has attached.
  renderStream: null,
  renderProcessedS: 0,
  attached: "",
  // Folded away to give the video the height. Remembered per browser.
  aiCollapsed: false,
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
  // behind it.
  pendingCaptures: 0,
  captureQueued: 0,
  captureFailures: 0,
};

// Compared against the server's version at startup. A service worker serving a
// stale app.js looks exactly like a bug in the current code, so the page says
// which build it is actually running rather than leaving it to be deduced.
const APP_BUILD = "0.5.0";

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

function setRenderStatus(text, isError = false) {
  ui.renderStatus.textContent = text;
  ui.renderStatus.classList.toggle("is-error", isError);
}

// ---------------------------------------------------------------------------
// Models
//
// The registry lives on the server. The page names a checkpoint and never sends
// one: adding picks from the .pt files the server itself found under models/
// and runs/*/weights, and removing forgets a model without deleting its file.
// ---------------------------------------------------------------------------

async function loadModels(preferred = null) {
  const payload = await jsonFetch("/api/models");
  state.models = Array.isArray(payload.models) ? payload.models : [];
  renderModelSelect(preferred ?? ui.modelSelect.value);
  if (payload.error) setRenderStatus(payload.error, true);
}

function renderModelSelect(preferred) {
  const select = ui.modelSelect;
  select.replaceChildren();
  if (!state.models.length) {
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "ไม่มีโมเดลในรายการ";
    select.append(none);
  }
  for (const model of state.models) {
    const option = document.createElement("option");
    option.value = model.name;
    const notes = [];
    if (model.default) notes.push("ค่าเริ่มต้น");
    // Listed rather than hidden: the fix for a missing file is to put it back.
    if (!model.available) notes.push("ไม่พบไฟล์");
    const suffix = notes.length ? ` (${notes.join(", ")})` : "";
    option.textContent = `${model.name} · ${model.label}${suffix}`;
    option.disabled = !model.available;
    select.append(option);
  }
  const usable = state.models.filter((model) => model.available);
  const chosen = usable.find((model) => model.name === preferred)
    || usable.find((model) => model.default)
    || usable[0];
  select.value = chosen ? chosen.name : "";
  select.disabled = usable.length === 0;
}

function selectedModel() {
  return state.models.find(
    (model) => model.name === ui.modelSelect.value && model.available,
  ) || null;
}

async function openModelManager() {
  if (!ui.modelDialog.open) ui.modelDialog.showModal();
  await refreshModelManager();
}

async function refreshModelManager() {
  ui.registeredModels.replaceChildren(managerNote("กำลังอ่านรายการ…"));
  ui.candidateModels.replaceChildren();
  try {
    const [models, found] = await Promise.all([
      jsonFetch("/api/models"),
      jsonFetch("/api/models/candidates"),
    ]);
    state.models = Array.isArray(models.models) ? models.models : [];
    renderModelSelect(ui.modelSelect.value);
    renderRegisteredModels();
    renderCandidateModels(Array.isArray(found.candidates) ? found.candidates : []);
  } catch (error) {
    ui.registeredModels.replaceChildren(
      managerNote(`อ่านรายการโมเดลไม่สำเร็จ: ${error.message}`, true),
    );
  }
  onRenderChoiceChange();
}

function managerNote(text, isError = false) {
  const note = document.createElement("p");
  note.className = "model-note";
  note.classList.toggle("is-error", isError);
  note.textContent = text;
  return note;
}

function modelBadge(text, isWarning = false) {
  const badge = document.createElement("span");
  badge.className = "model-badge";
  badge.classList.toggle("is-warning", isWarning);
  badge.textContent = text;
  return badge;
}

function renderRegisteredModels() {
  ui.registeredModels.replaceChildren();
  if (!state.models.length) {
    ui.registeredModels.append(managerNote("ยังไม่มีโมเดลในรายการ — เพิ่มจากไฟล์ด้านล่าง"));
    return;
  }
  for (const model of state.models) {
    const row = document.createElement("div");
    row.className = "model-item";
    const info = document.createElement("div");
    info.className = "model-item-info";
    const name = document.createElement("strong");
    name.className = "model-item-name";
    name.textContent = model.name;
    if (model.default) name.append(modelBadge("ค่าเริ่มต้น"));
    if (!model.available) name.append(modelBadge("ไม่พบไฟล์", true));
    const meta = document.createElement("small");
    meta.className = "model-item-meta";
    meta.textContent = model.available
      ? `${model.label} · ${formatBytes(model.size_bytes)}`
      : model.label;
    info.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "model-item-actions";
    if (!model.default && model.available) {
      const makeDefault = document.createElement("button");
      makeDefault.type = "button";
      makeDefault.className = "link-button";
      makeDefault.textContent = "ตั้งเป็นค่าเริ่มต้น";
      makeDefault.addEventListener("click", () => setDefaultModel(model.name));
      actions.append(makeDefault);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "link-button is-danger";
    remove.textContent = "เอาออกจากรายการ";
    remove.addEventListener("click", () => removeModel(model));
    actions.append(remove);
    row.append(info, actions);
    ui.registeredModels.append(row);
  }
}

function renderCandidateModels(candidates) {
  ui.candidateModels.replaceChildren();
  if (!candidates.length) {
    ui.candidateModels.append(managerNote("ไม่พบไฟล์ .pt ใหม่ — ทุกไฟล์ในโฟลเดอร์อยู่ในรายการแล้ว"));
    return;
  }
  for (const candidate of candidates) {
    const row = document.createElement("div");
    row.className = "model-item";
    const info = document.createElement("div");
    info.className = "model-item-info";
    const label = document.createElement("strong");
    label.className = "model-item-name";
    label.textContent = candidate.label;
    const meta = document.createElement("small");
    meta.className = "model-item-meta";
    meta.textContent =
      `${formatBytes(candidate.size_bytes)} · แก้ไขล่าสุด ${formatDate(candidate.modified * 1000)}`;
    info.append(label, meta);

    const actions = document.createElement("div");
    actions.className = "model-item-actions";
    const nameInput = document.createElement("input");
    nameInput.className = "model-name-input";
    nameInput.type = "text";
    nameInput.maxLength = 40;
    nameInput.value = candidate.suggested_name;
    nameInput.setAttribute("aria-label", `ชื่อย่อสำหรับ ${candidate.label}`);
    const add = document.createElement("button");
    add.type = "button";
    add.className = "secondary-button";
    add.textContent = "เพิ่ม";
    add.addEventListener("click", () => addModel(candidate, nameInput.value, add));
    actions.append(nameInput, add);
    row.append(info, actions);
    ui.candidateModels.append(row);
  }
}

async function addModel(candidate, name, button) {
  button.disabled = true;
  try {
    const added = await jsonFetch("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_id: candidate.id, name: name.trim() }),
    });
    showToast(`เพิ่มโมเดล ${added.name} แล้ว`);
    await refreshModelManager();
    renderModelSelect(added.name);
    onRenderChoiceChange();
  } catch (error) {
    button.disabled = false;
    showToast(`เพิ่มโมเดลไม่สำเร็จ: ${error.message}`);
  }
}

async function removeModel(model) {
  const asked = window.confirm(
    `เอา ${model.name} ออกจากรายการ?\n`
    + `ไฟล์ ${model.label} ยังอยู่บนเครื่อง เพิ่มกลับได้จากส่วน เพิ่มโมเดล`,
  );
  if (!asked) return;
  try {
    await jsonFetch(`/api/models/${encodeURIComponent(model.name)}`, { method: "DELETE" });
    showToast(`เอา ${model.name} ออกจากรายการแล้ว (ไฟล์ยังอยู่)`);
    await refreshModelManager();
  } catch (error) {
    showToast(`เอาออกไม่สำเร็จ: ${error.message}`);
  }
}

async function setDefaultModel(name) {
  try {
    await jsonFetch("/api/models/default", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await refreshModelManager();
  } catch (error) {
    showToast(`ตั้งค่าเริ่มต้นไม่สำเร็จ: ${error.message}`);
  }
}

// ---------------------------------------------------------------------------
// Videos and AI renders
//
// The page plays files the server serves. A render is the same video run
// through a model once, every frame in order, with the boxes drawn into the
// picture -- so there is nothing to draw here, and nothing that can fall
// behind the video the way a live overlay did.
// ---------------------------------------------------------------------------

const RENDER_ACTIVE = new Set(["loading", "running", "cancelling"]);

async function initializeLibrary() {
  const results = await Promise.allSettled([loadModels(), loadVideos()]);
  const failed = results.find((result) => result.status === "rejected");
  if (failed) {
    setRenderStatus(`อ่านรายการวิดีโอหรือโมเดลไม่สำเร็จ: ${failed.reason.message}`, true);
  }
  try {
    const payload = await jsonFetch("/api/renders");
    rememberRenderSpeed(payload.renders);
    state.ffmpeg = Boolean(payload.ffmpeg);
    applyRenderJob(payload.job);
  } catch (error) {
    setRenderStatus(`อ่านสถานะการประมวลผลไม่ได้: ${error.message}`, true);
    return;
  }
  if (!failed) describeRenderChoice();
}

async function loadVideos() {
  const payload = await jsonFetch("/api/videos");
  state.videos = Array.isArray(payload.videos) ? payload.videos : [];
  const select = ui.videoSelect;
  const previous = select.value;
  select.replaceChildren();
  const prompt = document.createElement("option");
  prompt.value = "";
  prompt.textContent = state.videos.length ? "— เลือกวิดีโอ —" : "ไม่พบวิดีโอในโฟลเดอร์";
  select.append(prompt);
  for (const video of state.videos) {
    const option = document.createElement("option");
    option.value = video.id;
    option.textContent =
      `${video.id} · ${formatTime(video.duration_ms / 1000)} · ${formatBytes(video.size_bytes)}`;
    select.append(option);
  }
  select.value = state.videos.some((video) => video.id === previous) ? previous : "";
  select.disabled = state.videos.length === 0;
  ui.emptyStateText.textContent = state.videos.length
    ? "เลือกจากรายการ วิดีโอ ด้านบน"
    : `ไม่พบวิดีโอใน ${payload.directory || "โฟลเดอร์วิดีโอ"} — คัดลอกไฟล์ไปไว้ที่นั่นแล้วโหลดหน้าใหม่`;
}

function renderSettings() {
  const confidence = Math.min(1, Math.max(0.05, Number(ui.renderConfidence.value) || 0.45));
  ui.renderConfidence.value = confidence.toFixed(2);
  return {
    confidence: Number(confidence.toFixed(2)),
    image_size: Number(ui.renderImageSize.value),
    stride: Number(ui.renderStride.value),
  };
}

async function refreshRenders({ autoplay = false } = {}) {
  if (!state.video) return;
  const videoId = state.video.id;
  let payload;
  try {
    payload = await jsonFetch(`/api/renders?${new URLSearchParams({ video: videoId })}`);
  } catch (error) {
    setRenderStatus(`อ่านผลการประมวลผลไม่สำเร็จ: ${error.message}`, true);
    return;
  }
  // Another video may have been chosen while this was in flight.
  if (!state.video || state.video.id !== videoId) return;
  state.renders = Array.isArray(payload.renders) ? payload.renders : [];
  rememberRenderSpeed(state.renders);
  state.ffmpeg = Boolean(payload.ffmpeg);
  applyRenderJob(payload.job);
  const listed = state.activeRender && state.renders.find(
    (render) => render.render_id === state.activeRender.render_id,
  );
  if (listed) {
    // The listed copy carries how far it has got, which the list and the status
    // line report while it is being made.
    state.activeRender = listed;
  } else if (state.activeRender && !isRenderingNow(state.activeRender.render_id)) {
    // Deleted, or never written: a render being made is not listed until its
    // first segment exists, and dropping it then would close what is playing.
    state.activeRender = null;
    showView("original");
  }
  if (autoplay) {
    const match = matchingRender(["complete", "running"]);
    if (match) {
      state.activeRender = match;
      showView("rendered");
    }
  }
  renderRenderList();
  updateRenderControls();
  describeRenderChoice();
}

// The speed of the newest finished render. A guess from a real run on this
// machine beats a constant, which would be wrong on every other one.
function rememberRenderSpeed(renders) {
  for (const render of renders || []) {
    const fps = Number(render.frames_written) / Number(render.elapsed_s);
    const created = String(render.created_at || "");
    if (Number.isFinite(fps) && fps > 0 && (!state.renderSpeed || created > state.renderSpeedAt)) {
      state.renderSpeed = fps;
      state.renderSpeedAt = created;
    }
  }
}

// A render of these files with the model and settings chosen above, in one of
// the states asked for: finished, stopped partway, or being made right now.
function matchingRender(statuses = ["complete"]) {
  const model = selectedModel();
  if (!state.video || !model) return null;
  const wanted = renderSettings();
  return state.renders.find((render) => (
    render.current
    && statuses.includes(render.status || "complete")
    && render.model === model.name
    && Number(render.settings.confidence).toFixed(2) === wanted.confidence.toFixed(2)
    && Number(render.settings.image_size) === wanted.image_size
    && Number(render.stride) === wanted.stride
  )) || null;
}

const AI_COLLAPSED_KEY = "telemetry-ocr.ai-collapsed";

function restoreAiPanel() {
  let collapsed = false;
  try {
    collapsed = window.localStorage.getItem(AI_COLLAPSED_KEY) === "1";
  } catch (error) {
    // Private windows refuse storage; the panel simply opens each time.
  }
  setAiCollapsed(collapsed, false);
}

function setAiCollapsed(collapsed, remember = true) {
  state.aiCollapsed = collapsed;
  ui.aiPanel.classList.toggle("is-collapsed", collapsed);
  ui.aiToggle.setAttribute("aria-expanded", String(!collapsed));
  ui.aiToggle.title = collapsed ? "แสดงแผงประมวลผลด้วย AI" : "ซ่อนแผงเพื่อให้วิดีโอใหญ่ขึ้น";
  updateAiHint();
  if (!remember) return;
  try {
    window.localStorage.setItem(AI_COLLAPSED_KEY, collapsed ? "1" : "0");
  } catch (error) {
    // Nothing to do: the choice just does not survive this page.
  }
}

// The one line the header keeps when the panel is folded: what is running, or
// what would run if the button were pressed.
function updateAiHint() {
  if (!state.aiCollapsed) {
    ui.aiToggleHint.textContent = "";
    return;
  }
  const job = state.renderJob;
  if (job && RENDER_ACTIVE.has(job.status)) {
    const percent = job.total ? Math.min(100, (job.done / job.total) * 100) : 0;
    const parts = [job.model, `${percent.toFixed(1)}%`];
    if (Number.isFinite(job.eta_s)) parts.push(`เหลืออีกราว ${formatDuration(job.eta_s)}`);
    ui.aiToggleHint.textContent = parts.join(" · ");
    return;
  }
  const model = selectedModel();
  ui.aiToggleHint.textContent = model ? `${model.name} · ${strideLabel()}` : "ยังไม่มีโมเดลที่ใช้ได้";
}

function strideLabel() {
  const stride = renderSettings().stride;
  return stride > 1 ? `ทุก ${stride} เฟรม` : "ทุกเฟรม";
}

function isRenderingNow(renderId) {
  const job = state.renderJob;
  return Boolean(job && RENDER_ACTIVE.has(job.status) && job.render_id === renderId);
}

// How much of the video a render has been through, 0 to 100.
function renderPercent(render) {
  const duration = Number(render.duration_ms) || Number(state.video?.duration_ms) || 0;
  if (!duration) return 0;
  return Math.min(100, Math.round((Number(render.processed_ms) || 0) / duration * 100));
}

function renderRenderList() {
  ui.renderList.replaceChildren();
  ui.renderList.hidden = !state.video || state.renders.length === 0;
  if (ui.renderList.hidden) return;
  const title = document.createElement("div");
  title.className = "render-list-title";
  title.textContent = "ผลการประมวลผลของวิดีโอนี้";
  ui.renderList.append(title);
  for (const render of state.renders) {
    const playing = state.view === "rendered"
      && state.activeRender?.render_id === render.render_id;
    const row = document.createElement("div");
    row.className = "render-item";
    row.dataset.renderId = render.render_id;
    row.classList.toggle("is-active", playing);
    const text = document.createElement("span");
    text.className = "render-item-text";
    const settings = render.settings || {};
    const stride = Number(render.stride) > 1 ? ` · ทุก ${render.stride} เฟรม` : "";
    text.textContent =
      `${render.model} · conf ${Number(settings.confidence).toFixed(2)} · ${settings.image_size} px${stride}`;
    const meta = document.createElement("small");
    meta.className = "render-item-meta";
    meta.classList.toggle("is-stale", !render.current);
    meta.textContent = describeRender(render, playing);
    const actions = document.createElement("span");
    actions.className = "render-item-actions";
    const play = document.createElement("button");
    play.type = "button";
    play.className = "link-button";
    play.textContent = "เล่น";
    play.disabled = playing;
    play.addEventListener("click", () => {
      state.activeRender = render;
      showView("rendered");
      describeRenderChoice();
    });
    actions.append(play);
    // A render that stopped partway carries on from where it got to, which is
    // what makes cancelling safe on a video that takes half an hour.
    if (render.status === "partial" && render.current) {
      const resume = document.createElement("button");
      resume.type = "button";
      resume.className = "link-button";
      resume.textContent = "ทำต่อ";
      resume.disabled = Boolean(state.renderJob && RENDER_ACTIVE.has(state.renderJob.status));
      resume.addEventListener("click", () => startRender(render));
      actions.append(resume);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "link-button is-danger";
    remove.textContent = "ลบ";
    remove.addEventListener("click", () => deleteRender(render));
    actions.append(remove);
    row.append(text, meta, actions);
    ui.renderList.append(row);
  }
}

function describeRender(render, playing) {
  if (!render.current) return "วิดีโอหรือโมเดลเปลี่ยนไปหลังประมวลผล — ควรประมวลผลใหม่";
  const prefix = playing ? "กำลังเล่น · " : "";
  if (render.status === "running") return `${prefix}กำลังประมวลผล ${renderPercent(render)}%`;
  if (render.status === "partial") return `${prefix}ประมวลผลไว้ ${renderPercent(render)}% · หยุดไว้`;
  return `${prefix}สร้างเมื่อ ${formatDate(render.created_at)}`;
}

async function deleteRender(render) {
  const asked = window.confirm(
    `ลบผลการประมวลผล ${render.model} ของวิดีโอนี้?\n`
    + "วิดีโอต้นฉบับและไฟล์โมเดลไม่ถูกลบ ประมวลผลใหม่ได้ภายหลัง",
  );
  if (!asked) return;
  try {
    const params = new URLSearchParams({ render: render.render_id });
    await jsonFetch(`/api/renders?${params}`, { method: "DELETE" });
    if (state.activeRender?.render_id === render.render_id) {
      state.activeRender = null;
      showView("original");
    }
    await refreshRenders();
  } catch (error) {
    setRenderStatus(`ลบไม่สำเร็จ: ${error.message}`, true);
  }
}

async function onRenderButton() {
  const match = matchingRender(["complete", "running"]);
  if (match) {
    state.activeRender = match;
    showView("rendered");
    describeRenderChoice();
    return;
  }
  await startRender(matchingRender(["partial"]));
}

// Starts a render, or continues one that stopped partway, and shows it at once:
// the picture follows the model rather than waiting for the whole file.
async function startRender(resume = null) {
  const model = resume ? { name: resume.model } : selectedModel();
  if (!state.video || !model) return;
  const settings = resume
    ? {
      confidence: Number(resume.settings.confidence),
      image_size: Number(resume.settings.image_size),
      stride: Number(resume.stride),
    }
    : renderSettings();
  ui.renderButton.disabled = true;
  try {
    const payload = await jsonFetch("/api/renders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: state.video.id, model: model.name, ...settings }),
    });
    if (payload.existing) {
      await refreshRenders();
      state.activeRender = state.renders.find(
        (render) => render.render_id === payload.existing.render_id,
      ) || payload.existing;
      showView("rendered");
      describeRenderChoice();
      return;
    }
    applyRenderJob(payload.job);
    const job = payload.job || {};
    state.activeRender = {
      render_id: job.render_id,
      model: model.name,
      settings,
      stride: settings.stride,
      format: "hls",
      status: "running",
      current: true,
      duration_ms: state.video.duration_ms,
      processed_ms: resume ? Number(resume.processed_ms) || 0 : 0,
    };
    setRenderStatus(
      `${resume ? "ทำต่อ" : "เริ่มประมวลผล"} ${state.video.name} ด้วย ${model.name}`
      + " — ดูส่วนที่ประมวลผลแล้วได้เลยระหว่างรอ",
    );
    // A continued render picks up a few seconds before where it stopped, so the
    // join is watched rather than jumped over.
    showView("rendered", resume ? Math.max(0, (Number(resume.processed_ms) || 0) / 1000 - 5) : 0, true);
    await refreshRenders();
  } catch (error) {
    if (error.payload?.job) applyRenderJob(error.payload.job);
    setRenderStatus(`เริ่มประมวลผลไม่สำเร็จ: ${error.message}`, true);
  } finally {
    updateRenderControls();
  }
}

async function cancelRender() {
  ui.renderCancelButton.disabled = true;
  try {
    const payload = await jsonFetch("/api/renders/current", { method: "DELETE" });
    applyRenderJob(payload.job);
  } catch (error) {
    ui.renderCancelButton.disabled = false;
    setRenderStatus(`ยกเลิกไม่สำเร็จ: ${error.message}`, true);
  }
}

function applyRenderJob(job) {
  const previous = state.renderJob;
  state.renderJob = job && job.status && job.status !== "idle" ? job : null;
  const current = state.renderJob;
  const active = Boolean(current && RENDER_ACTIVE.has(current.status));
  ui.renderProgress.hidden = !active;
  if (active) {
    const percent = current.total ? Math.min(100, (current.done / current.total) * 100) : 0;
    ui.renderBarFill.style.width = `${percent.toFixed(1)}%`;
    const parts = [`${current.video} · ${current.model}`];
    if (current.status === "loading") {
      parts.push("กำลังโหลดโมเดล…");
    } else if (current.status === "cancelling") {
      parts.push("กำลังยกเลิก…");
    } else {
      parts.push(`${percent.toFixed(1)}%`);
      if (current.fps) parts.push(`${Number(current.fps).toFixed(1)} เฟรม/วินาที`);
      if (Number.isFinite(current.eta_s)) parts.push(`เหลืออีกราว ${formatDuration(current.eta_s)}`);
    }
    ui.renderProgressText.textContent = parts.join(" · ");
    ui.renderCancelButton.disabled = current.status === "cancelling";
    scheduleRenderPoll();
  } else {
    stopRenderPoll();
  }
  const finished = previous && RENDER_ACTIVE.has(previous.status)
    && current && !RENDER_ACTIVE.has(current.status);
  if (finished) {
    if (current.status === "complete") {
      setRenderStatus(`ประมวลผล ${current.video} ด้วย ${current.model} เสร็จแล้ว`);
      showToast("ประมวลผลวิดีโอด้วย AI เสร็จแล้ว");
    } else if (current.status === "error") {
      setRenderStatus(`ประมวลผลไม่สำเร็จ: ${current.error || "ไม่ทราบสาเหตุ"}`, true);
    } else if (current.status === "cancelled") {
      setRenderStatus(
        current.kept
          ? "ยกเลิกแล้ว — ส่วนที่ประมวลผลไว้ยังดูได้ กด ทำต่อ เพื่อทำส่วนที่เหลือ"
          : "ยกเลิกการประมวลผลแล้ว",
      );
    }
    if (state.video && state.video.id === current.video) refreshRenders();
  }
  updateRenderControls();
  updateAiHint();
}

function scheduleRenderPoll() {
  if (state.renderPollTimer !== null) return;
  state.renderPollTimer = window.setTimeout(async () => {
    state.renderPollTimer = null;
    try {
      applyRenderJob(await jsonFetch("/api/renders/current"));
    } catch (error) {
      setRenderStatus(`อ่านสถานะการประมวลผลไม่ได้: ${error.message}`, true);
      scheduleRenderPoll();
    }
  }, 1000);
}

function stopRenderPoll() {
  window.clearTimeout(state.renderPollTimer);
  state.renderPollTimer = null;
}

function updateRenderControls() {
  const busy = Boolean(state.renderJob && RENDER_ACTIVE.has(state.renderJob.status));
  const model = selectedModel();
  const match = matchingRender(["complete", "running"]);
  const partial = matchingRender(["partial"]);
  const playingMatch = Boolean(
    match && state.view === "rendered" && state.activeRender?.render_id === match.render_id,
  );
  ui.renderButton.textContent = match
    ? "เล่นวิดีโอ AI"
    : partial ? `ทำต่อจาก ${renderPercent(partial)}%` : "ประมวลผลด้วย AI";
  ui.renderButton.disabled = !state.video || !model || playingMatch
    || (!match && (busy || state.ffmpeg === false));
}

function describeRenderChoice() {
  if (state.renderJob && RENDER_ACTIVE.has(state.renderJob.status)) {
    // The bar above says how far along it is; this says what is on screen --
    // which may be a render finished earlier, not the one being made.
    if (state.view === "rendered" && state.activeRender) {
      const watching = state.activeRender.render_id === state.renderJob.render_id;
      setRenderStatus(
        watching
          ? `กำลังดูวิดีโอ AI ระหว่างประมวลผล · ประมวลผลถึง ${formatTime(state.renderProcessedS)}`
          : `กำลังแสดงวิดีโอ AI จาก ${state.activeRender.model} ที่ประมวลผลไว้แล้ว`,
      );
    }
    return;
  }
  if (!state.video) {
    setRenderStatus("เลือกวิดีโอและโมเดล แล้วกดประมวลผลด้วย AI");
    return;
  }
  const model = selectedModel();
  if (!model) {
    setRenderStatus("ยังไม่มีโมเดลที่ใช้ได้ — กด จัดการโมเดล เพื่อเพิ่ม", true);
    return;
  }
  const match = matchingRender();
  const partial = matchingRender(["partial"]);
  if (match && state.view === "rendered" && state.activeRender?.render_id === match.render_id) {
    setRenderStatus(`กำลังแสดงวิดีโอ AI จาก ${match.model} · สลับเป็นต้นฉบับได้ที่ปุ่มด้านบน`);
  } else if (state.view === "rendered" && state.activeRender) {
    const upto = state.activeRender.status === "complete"
      ? ""
      : ` · ประมวลผลไว้ถึง ${formatTime(state.renderProcessedS)}`;
    setRenderStatus(`กำลังแสดงวิดีโอ AI จาก ${state.activeRender.model}${upto}`);
  } else if (match) {
    setRenderStatus(`มีวิดีโอ AI จาก ${match.model} ที่ค่านี้แล้ว — กด เล่นวิดีโอ AI`);
  } else if (partial) {
    setRenderStatus(
      `ประมวลผล ${partial.model} ที่ค่านี้ไว้ ${renderPercent(partial)}% แล้วหยุดไว้`
      + " — กด ทำต่อ เพื่อทำส่วนที่เหลือ",
    );
  } else if (state.ffmpeg === false) {
    setRenderStatus("ไม่พบ ffmpeg บนเครื่องนี้ จึงประมวลผลวิดีโอใหม่ไม่ได้ (ยังเล่นผลที่มีอยู่ได้)", true);
  } else {
    const frames = Number(state.video.frames) / renderSettings().stride;
    const estimate = state.renderSpeed && frames
      ? `ทั้งไฟล์ใช้เวลาประมาณ ${formatDuration(frames / state.renderSpeed)}`
        + ` (จากความเร็วครั้งล่าสุด ${state.renderSpeed.toFixed(1)} เฟรม/วินาที)`
      : "ใช้เวลานานกว่าความยาววิดีโอ";
    setRenderStatus(
      `ยังไม่ได้ประมวลผล ${state.video.name} ด้วย ${model.name} ที่ค่านี้ · ${estimate}`
      + " · ดูส่วนที่ประมวลผลแล้วได้ทันทีระหว่างรอ",
    );
  }
}

function onRenderChoiceChange() {
  updateRenderControls();
  describeRenderChoice();
  updateAiHint();
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("th-TH", { dateStyle: "medium", timeStyle: "short" });
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours) return `${hours} ชม. ${minutes} นาที`;
  if (minutes) return `${minutes} นาที`;
  return `${total} วินาที`;
}

async function initialize() {
  try {
    state.config = await jsonFetch("/api/config");
    const session = await jsonFetch("/api/session", { method: "POST" });
    state.sessionId = session.session_id;
    renderTelemetryRows();
    restoreAiPanel();
    setEngineStatus("ready", `${state.config.model} พร้อมใช้งาน`);
    initializeLibrary();
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

async function selectVideo(videoId) {
  const video = state.videos.find((item) => item.id === videoId) || null;
  state.video = video;
  state.activeRender = null;
  state.renders = [];
  renderRenderList();
  if (!video) {
    state.view = "original";
    state.attached = "";
    closeRenderPlayer();
    ui.video.removeAttribute("src");
    ui.video.load();
    ui.dropState.hidden = false;
    ui.viewSwitch.hidden = true;
    ui.fileName.textContent = "ยังไม่ได้เลือกวิดีโอ";
    ui.fileDetails.textContent = "วิดีโอจากโฟลเดอร์วิดีโอของโปรเจกต์";
    ui.durationTime.textContent = formatTime(0);
    updateCaptureAvailability();
    onRenderChoiceChange();
    return;
  }
  ui.dropState.hidden = true;
  ui.fileName.textContent = video.name;
  ui.fileDetails.textContent = [
    formatBytes(video.size_bytes),
    `${video.width}×${video.height}`,
    formatTime(video.duration_ms / 1000),
    `${Number(video.fps).toFixed(2)} fps`,
  ].join(" · ");
  resetSession(true);
  showView("original", 0);
  updateRenderControls();
  await refreshRenders({ autoplay: true });
}

// A render is segments the server keeps adding to while it works, not a file
// that exists once it is finished, so it is played through Media Source instead
// of by pointing the <video> at a URL. Renders made before segments are one MP4
// and still play that way.
function attachedKey(view) {
  if (view === "rendered" && state.activeRender) return `render:${state.activeRender.render_id}`;
  return `video:${state.video.id}`;
}

function renderPlayer() {
  if (!state.renderStream) {
    state.renderStream = TelemetryRenderPlayer.create(ui.video, {
      fetchIndex: (render) => jsonFetch(`/api/renders/stream?${new URLSearchParams({ render })}`),
      segmentUrl: (render, run, file) => (
        `/api/renders/segment?${new URLSearchParams({ render, run, file })}`
      ),
      onState: applyStreamState,
    });
  }
  return state.renderStream;
}

function closeRenderPlayer() {
  if (state.renderStream) state.renderStream.close();
  state.renderProcessedS = 0;
  ui.renderWait.hidden = true;
}

function applyStreamState(streamState) {
  if (streamState.error) {
    ui.renderWait.hidden = true;
    setRenderStatus(`เล่นวิดีโอ AI ไม่ได้: ${streamState.error}`, true);
    return;
  }
  state.renderProcessedS = Number(streamState.processedS) || 0;
  // Said on the picture: stopping at the frame the model has reached is waiting,
  // and without a word for it, it looks like a video that has hung. How long the
  // wait is, and how long it will play afterwards, come from the rate the render
  // is actually going at.
  ui.renderWait.hidden = !streamState.waiting;
  if (streamState.waiting) ui.renderWait.textContent = describeWait(streamState);
  if (streamState.clamped) {
    showToast(`ยังประมวลผลไม่ถึงตรงนั้น — กลับมาที่ ${formatTime(state.renderProcessedS)}`);
  }
  if (state.view === "rendered") describeRenderChoice();
}

function describeWait(streamState) {
    const parts = [
      streamState.started ? "กำลังประมวลผลส่วนถัดไป" : "กำลังสะสมภาพก่อนเริ่มเล่น",
    ];
    if (Number.isFinite(streamState.waitS)) {
      parts.push(`พร้อมเล่นในอีกราว ${formatDuration(streamState.waitS)}`);
    }
    if (Number.isFinite(streamState.playS) && streamState.playS >= 3) {
      parts.push(`แล้วเล่นต่อเนื่องราว ${formatDuration(streamState.playS)}`);
    }
    return `${parts.join(" · ")}…`;
}

// Swaps what the <video> plays and keeps the moment on screen. The original and
// its render share one timeline, so switching is a comparison, not a restart.
function showView(view, atSeconds = null, play = null) {
  if (!state.video) return;
  const target = view === "rendered" && state.activeRender ? "rendered" : "original";
  state.view = target;
  ui.viewSwitch.hidden = !state.activeRender;
  ui.viewOriginal.classList.toggle("is-active", target === "original");
  ui.viewRendered.classList.toggle("is-active", target === "rendered");
  ui.viewOriginal.setAttribute("aria-pressed", String(target === "original"));
  ui.viewRendered.setAttribute("aria-pressed", String(target === "rendered"));
  renderRenderList();
  const key = attachedKey(target);
  if (state.attached === key && atSeconds === null) return;
  const resumeAt = atSeconds ?? (ui.video.currentTime || 0);
  const resume = play ?? (!ui.video.paused && !ui.video.ended);
  state.attached = key;
  const render = state.activeRender;
  if (target === "rendered" && render && render.format !== "mp4") {
    // Cleared before the new render reports its own: left alone, the line below
    // the panel keeps saying how far the render just left behind had got.
    state.renderProcessedS = 0;
    ui.renderWait.hidden = false;
    renderPlayer().open(render.render_id, {
      startAt: resumeAt,
      autoplay: resume,
      durationMs: state.video.duration_ms,
    });
    return;
  }
  closeRenderPlayer();
  ui.video.addEventListener("loadedmetadata", () => {
    if (resumeAt > 0) ui.video.currentTime = Math.min(resumeAt, ui.video.duration || resumeAt);
    if (resume) ui.video.play().catch(() => {});
  }, { once: true });
  ui.video.src = target === "rendered"
    ? `/api/renders/video?${new URLSearchParams({ render: render.render_id })}`
    : `/api/videos/source?${new URLSearchParams({ video: state.video.id })}`;
  ui.video.load();
}

function chooseView(view) {
  showView(view);
  updateRenderControls();
  describeRenderChoice();
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
  // Clears the readings on screen, and only those.
  //
  // It used to empty state.records and disable the two download buttons, from
  // when those buttons serialised that array. They export the captured shots
  // now, so disabling them here claimed to have thrown away something that is
  // still on disk -- and the next refreshShots re-enabled them a moment later
  // anyway. It also wrote "0 records" into the element the shot count owns,
  // which then said "0 records" until the next capture rewrote it as "N ภาพ".
  resetCurrentValues();
  ui.shotDetections.textContent = "";
}

// "There is a picture to read."
function sourceReady() {
  return Boolean(state.video) && ui.video.readyState >= 2;
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
    setCaptureStatus("ยังไม่มีภาพให้เก็บ เลือกวิดีโอก่อน", true);
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
      source_time_ms: Math.max(0, (ui.video.currentTime || 0) * 1000).toFixed(3),
      session_id: state.sessionId,
      video: state.video.id,
    });
    // Boxes come from the render being played, or the one last chosen for this
    // video. The frame itself is always read from the original on the server:
    // the rendered picture has boxes over the OSD digits OCR reads.
    if (state.activeRender) params.set("render", state.activeRender.render_id);
    // The server does not answer until the frame has been read, recognised and
    // re-read from the saved file, so by the time this returns it is finished.
    const shot = await jsonFetch(`/api/capture/shot?${params}`, { method: "POST" });
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
    ? `กำลังอ่านค่า ${state.pendingCaptures} · รออีก ${state.captureQueued}`
    : `กำลังอ่านค่า ${state.pendingCaptures}`;
  const ready = total - unprocessed;
  ui.shotReady.hidden = ready === 0;
  ui.shotReady.textContent = `พร้อมส่ง ${ready}`;
  ui.shotUnprocessed.hidden = unprocessed === 0;
  ui.shotUnprocessed.textContent = `ไม่มีกรอบ AI ${unprocessed}`;
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
    label.textContent = i < state.pendingCaptures ? "กำลังอ่านค่า" : "รอคิว";
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
    badge.textContent = {
      no_render: "ไม่มี AI", not_processed: "ยังไม่ถึง", no_model: "ไม่มีโมเดล",
    }[shot.stage]
      || "โมเดลไม่ตอบ";
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
  if (detections.status === "no_render") {
    return "ยังไม่ได้ประมวลผลด้วย AI — ภาพนี้ไม่มีกรอบ";
  }
  if (detections.status === "not_processed") {
    return "ยังประมวลผลไม่ถึงเฟรมนี้ — ภาพนี้ยังไม่มีกรอบ";
  }
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
  return TelemetryFormat.formatDms(value, degreeDigits, decimalPlaces).text;
}

// One row per field per captured shot.
//
// These used to serialise state.records, which is only ever filled by the
// stream's telemetry poll. In file mode -- load a video, hold Capture, press
// Download -- it is empty, so the JSONL was a single newline and the CSV was a
// header row, with the buttons enabled and nothing said. The buttons are
// enabled from the number of shots, so the shots are what they have to export.
//
// The stream's own reading is not included and is not missing: the server
// writes it to output/stream-telemetry.jsonl as it goes. This page is a viewer
// there, and duplicating that file into the browser's downloads would leave two
// records of one run that could disagree.
const EXPORT_COLUMNS = [
  "frame_index",
  "captured_at",
  "source_time_ms",
  "field_id",
  "value",
  "raw_text",
  "confidence",
  "confirmed",
];

function exportRows() {
  const rows = [];
  // Oldest first. The strip shows newest first because that is where the eye
  // is after a capture; a file is read from the top, in the order it happened.
  const shots = [...state.shots].sort((a, b) => a.frame_index - b.frame_index);
  for (const shot of shots) {
    for (const record of shot.records || []) {
      rows.push({
        frame_index: shot.frame_index,
        captured_at: shot.captured_at,
        source_time_ms: shot.source_time_ms,
        field_id: record.field_id,
        value: record.value,
        raw_text: record.raw_text,
        confidence: record.confidence,
        confirmed: record.confirmed,
      });
    }
  }
  return rows;
}

function downloadJsonl() {
  const rows = exportRows();
  if (!rows.length) {
    setCaptureStatus("ยังไม่มีค่าที่อ่านได้จากภาพที่เก็บไว้ จึงไม่มีอะไรให้ดาวน์โหลด", true);
    return;
  }
  downloadBlob(
    rows.map((row) => JSON.stringify(row)).join("\n") + "\n",
    "application/x-ndjson",
    buildDownloadName("jsonl"),
  );
}

function downloadCsv() {
  const rows = exportRows();
  if (!rows.length) {
    setCaptureStatus("ยังไม่มีค่าที่อ่านได้จากภาพที่เก็บไว้ จึงไม่มีอะไรให้ดาวน์โหลด", true);
    return;
  }
  const lines = [EXPORT_COLUMNS.join(",")];
  for (const row of rows) {
    lines.push(EXPORT_COLUMNS.map((key) => csvCell(row[key])).join(","));
  }
  downloadBlob(lines.join("\r\n") + "\r\n", "text/csv;charset=utf-8", buildDownloadName("csv"));
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function buildDownloadName(extension) {
  const base = (state.video?.name || "telemetry").replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "_");
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

ui.videoSelect.addEventListener("change", () => selectVideo(ui.videoSelect.value));
ui.viewOriginal.addEventListener("click", () => chooseView("original"));
ui.viewRendered.addEventListener("click", () => chooseView("rendered"));
ui.modelSelect.addEventListener("change", onRenderChoiceChange);
for (const control of [ui.renderConfidence, ui.renderImageSize, ui.renderStride]) {
  control.addEventListener("change", onRenderChoiceChange);
}
ui.aiToggle.addEventListener("click", () => setAiCollapsed(!state.aiCollapsed));
ui.renderButton.addEventListener("click", onRenderButton);
ui.renderCancelButton.addEventListener("click", cancelRender);
ui.manageModelsButton.addEventListener("click", openModelManager);
ui.modelDialogClose.addEventListener("click", () => ui.modelDialog.close());
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
ui.video.addEventListener("loadedmetadata", () => {
  ui.durationTime.textContent = formatTime(ui.video.duration);
  updateCaptureAvailability();
  ui.resetButton.disabled = false;
});
ui.video.addEventListener("pause", updatePlaybackTime);
ui.video.addEventListener("timeupdate", updatePlaybackTime);
ui.video.addEventListener("seeked", () => {
  updatePlaybackTime();
  // Temporal voting assumes consecutive frames; a jump starts it afresh.
  resetSession(false);
});
ui.video.addEventListener("error", () => {
  if (!ui.video.getAttribute("src")) return;
  showToast(
    state.view === "original"
      ? "เบราว์เซอร์เล่นวิดีโอต้นฉบับนี้ไม่ได้ — ประมวลผลด้วย AI แล้วเล่นวิดีโอ AI ซึ่งเข้ารหัสเป็น H.264 เสมอ"
      : "เล่นวิดีโอ AI ไม่ได้ — ลบผลนี้แล้วประมวลผลใหม่",
  );
});
ui.resetButton.addEventListener("click", () => resetSession(true));
ui.downloadJsonButton.addEventListener("click", downloadJsonl);
ui.downloadCsvButton.addEventListener("click", downloadCsv);
ui.clearLogButton.addEventListener("click", clearLog);
window.addEventListener("beforeunload", () => stopRenderPoll());

initializePwa();
initialize();
