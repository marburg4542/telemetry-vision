"""Capture the screenshots the user guide shows, from a running server.

Drives headless Chrome over the DevTools protocol with aiohttp, which the web
extra already installs, so no browser-automation package is needed. The page is
the real one and the video a real flight clip: a guide whose pictures were
mocked up drifts from the app the first time either changes.

Expects the server to list a video that already has a finished render with the
default settings (best50, conf 0.45, 640 px, every frame), because rendering a
whole video just to photograph it would take as long as the video.

Side effects, on purpose and nothing more: three captures are written to
output/sessions/, and a render is started with a different stride and cancelled
to photograph its progress bar. Nothing is sent to LINE.

Usage:
    .venv-cuda/Scripts/python.exe docs/user-guide/capture_screens.py [BASE_URL] [VIDEO_ID]
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_lab.report import find_browser

APP = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
VIDEO = sys.argv[2] if len(sys.argv) > 2 else "clip-60s.mp4"
OUT = Path(__file__).resolve().parent / "assets" / "screens"
PORT = 9333
DESKTOP = (1440, 900)
MOBILE = (390, 844)
# Moments in the clip where best50 found a vehicle, so the AI frames show boxes.
BOX_TIMES = (19.8, 17.6, 5.0)

# Elements the guide points at with numbered markers. Rects are recorded beside
# each screenshot, so a marker lands where the element really is.
MARKED = {
    "engine": "#engineStatus",
    "video_select": ".source-bar .video-select",
    "view_switch": "#viewSwitch",
    "stage": "#videoStage",
    "capture": "#captureButton",
    "line": "#lineSendButton",
    "quota": "#lineQuotaLabel",
    "time": ".playback-time",
    "reset": "#resetButton",
    "source_line": ".source-line",
    "ai_panel": "#aiPanel",
    "model_select": ".ai-row .model-select",
    "manage_models": "#manageModelsButton",
    "render_button": "#renderButton",
    "render_progress": "#renderProgress",
    "render_status": "#renderStatus",
    "render_list": "#renderList",
    "shot_progress": "#shotProgress",
    "shot_strip": "#shotStrip",
    "select_bar": "#shotSelectBar",
    "preview": "#shotPreview",
    "telemetry": "#telemetryList",
    "downloads": ".panel-actions",
    "capture_status": "#captureStatus",
    "dialog_registered": "#registeredModels",
    "dialog_candidates": "#candidateModels",
}


class Cdp:
    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self.ws = ws
        self.next_id = 0
        self.pending: dict[int, asyncio.Future] = {}
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for message in self.ws:
            data = json.loads(message.data)
            future = self.pending.pop(data.get("id", -1), None)
            if future is not None and not future.done():
                if "error" in data:
                    future.set_exception(RuntimeError(str(data["error"])))
                else:
                    future.set_result(data.get("result", {}))

    async def send(self, method: str, **params):
        self.next_id += 1
        future = asyncio.get_running_loop().create_future()
        self.pending[self.next_id] = future
        await self.ws.send_str(
            json.dumps({"id": self.next_id, "method": method, "params": params})
        )
        return await asyncio.wait_for(future, 120)

    async def js(self, expression: str):
        result = await self.send(
            "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True
        )
        if "exceptionDetails" in result:
            raise RuntimeError(json.dumps(result["exceptionDetails"])[:600])
        return result["result"].get("value")

    async def wait_for(self, expression: str, timeout: float = 90, label: str = "") -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.js(f"Boolean({expression})"):
                return
            await asyncio.sleep(0.4)
        raise TimeoutError(f"timed out waiting for {label or expression}")


async def metrics(cdp: Cdp, width: int, height: int, mobile: bool) -> None:
    await cdp.send(
        "Emulation.setDeviceMetricsOverride",
        width=width, height=height, deviceScaleFactor=2, mobile=mobile,
    )


async def rects(cdp: Cdp) -> dict:
    selectors = json.dumps(MARKED)
    return await cdp.js(
        f"""(() => {{
          const out = {{}};
          for (const [key, selector] of Object.entries({selectors})) {{
            const el = document.querySelector(selector);
            if (!el) continue;
            const r = el.getBoundingClientRect();
            if (!r.width || !r.height) continue;
            out[key] = [r.left, r.top, r.width, r.height];
          }}
          return {{ viewport: [innerWidth, innerHeight], rects: out }};
        }})()"""
    )


async def shot(cdp: Cdp, name: str, clip_selector: str | None = None, pad: int = 10) -> None:
    params: dict = {"format": "png"}
    if clip_selector:
        await cdp.js(
            f"document.querySelector({json.dumps(clip_selector)})"
            ".scrollIntoView({block: 'center'})"
        )
        await asyncio.sleep(0.5)
        x, y, w, h = await cdp.js(
            f"""(() => {{
              const r = document.querySelector({json.dumps(clip_selector)}).getBoundingClientRect();
              return [r.left + scrollX, r.top + scrollY, r.width, r.height];
            }})()"""
        )
        params["clip"] = {
            "x": max(0, x - pad), "y": max(0, y - pad),
            "width": w + 2 * pad, "height": h + 2 * pad, "scale": 1,
        }
        params["captureBeyondViewport"] = True
    data = await cdp.send("Page.captureScreenshot", **params)
    (OUT / f"{name}.png").write_bytes(base64.b64decode(data["data"]))
    if not clip_selector:
        (OUT / f"{name}.json").write_text(json.dumps(await rects(cdp), indent=1), "utf-8")
    print("saved", name)


async def seek(cdp: Cdp, seconds: float) -> None:
    await cdp.js(f"sourceVideo.pause(); sourceVideo.currentTime = {seconds}")
    await asyncio.sleep(2.5)


async def set_stride(cdp: Cdp, stride: int) -> None:
    await cdp.js(
        f"renderStride.value = '{stride}'; renderStride.dispatchEvent(new Event('change'))"
    )


async def capture_at(cdp: Cdp, seconds: float) -> None:
    before = await cdp.js("state.shots.length")
    await seek(cdp, seconds)
    await cdp.js("requestCapture()")
    await cdp.wait_for(
        f"state.shots.length > {before} && !state.pendingCaptures && !state.captureQueued",
        timeout=120, label="capture",
    )


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="guide-chrome-"))
    chrome = subprocess.Popen(
        [
            find_browser(), "--headless=new", f"--remote-debugging-port={PORT}",
            f"--user-data-dir={profile}", "--no-first-run", "--hide-scrollbars",
            "--autoplay-policy=no-user-gesture-required", "--window-size=1440,900",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        async with aiohttp.ClientSession() as http:
            pages: list = []
            for _ in range(60):
                try:
                    async with http.get(f"http://127.0.0.1:{PORT}/json/list") as response:
                        pages = [p for p in await response.json() if p["type"] == "page"]
                        if pages:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.5)
            async with http.ws_connect(pages[0]["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                cdp = Cdp(ws)
                for domain in ("Page", "Runtime", "DOM"):
                    await cdp.send(f"{domain}.enable")
                await metrics(cdp, *DESKTOP, mobile=False)
                await cdp.send("Page.navigate", url=APP)
                await cdp.wait_for(
                    "document.readyState === 'complete' && typeof state !== 'undefined'",
                    label="page",
                )
                await cdp.wait_for(
                    "engineStatus.classList.contains('is-ready') && !videoSelect.disabled"
                    " && !modelSelect.disabled && state.ffmpeg !== null",
                    timeout=120, label="engine, videos and models",
                )
                await asyncio.sleep(1)
                await shot(cdp, "01-start")

                # Choosing the video plays its finished render straight away.
                await cdp.js(
                    f"videoSelect.value = {json.dumps(VIDEO)};"
                    " videoSelect.dispatchEvent(new Event('change'))"
                )
                await cdp.wait_for(
                    "state.view === 'rendered' && sourceVideo.readyState >= 2",
                    timeout=60, label="the AI video",
                )
                await seek(cdp, BOX_TIMES[0])
                await shot(cdp, "03-ai-video")
                await shot(cdp, "04-ai-panel", "#aiPanel")

                await cdp.js("viewOriginal.click()")
                await cdp.wait_for(
                    "state.view === 'original' && sourceVideo.readyState >= 2", label="original"
                )
                await seek(cdp, BOX_TIMES[0])
                await shot(cdp, "02-original")

                # A render in progress: every second frame, so it is not the one
                # already on disk, then cancelled once the bar has moved.
                await set_stride(cdp, 2)
                await asyncio.sleep(1)
                await cdp.js("renderButton.click()")
                await cdp.wait_for(
                    "state.renderJob && state.renderJob.status === 'running'"
                    " && state.renderJob.done > 60",
                    timeout=120, label="render progress",
                )
                await asyncio.sleep(2)
                await shot(cdp, "05-render-progress", "#aiPanel")
                await cdp.js("renderCancelButton.click()")
                await cdp.wait_for(
                    "!state.renderJob || state.renderJob.status === 'cancelled'",
                    timeout=60, label="cancelled",
                )
                await set_stride(cdp, 1)

                await cdp.js("manageModelsButton.click()")
                await cdp.wait_for(
                    "registeredModels.querySelectorAll('.model-item').length > 0", label="models"
                )
                await asyncio.sleep(1)
                await shot(cdp, "06-model-dialog")
                await cdp.js("modelDialog.close()")

                await cdp.js("viewRendered.click()")
                await cdp.wait_for(
                    "state.view === 'rendered' && sourceVideo.readyState >= 2", label="AI video"
                )
                for seconds in BOX_TIMES:
                    await capture_at(cdp, seconds)
                await asyncio.sleep(1.5)
                # The shot with the most detections, so the preview carries boxes.
                await cdp.js(
                    """(() => {
                      const total = (shot) => Object.values((shot.detections || {}).counts || {})
                        .reduce((sum, n) => sum + n, 0);
                      const best = [...state.shots].sort((a, b) => total(b) - total(a))[0];
                      document.querySelector(
                        `.shot-thumb[data-frame-index="${best.frame_index}"] .shot-thumb-open`
                      ).click();
                    })()"""
                )
                await asyncio.sleep(2.5)
                await shot(cdp, "07-captured")

                await cdp.js(
                    "[...document.querySelectorAll('.shot-thumb-tick input')].slice(0, 2)"
                    ".forEach((box) => box.click())"
                )
                await asyncio.sleep(1)
                await shot(cdp, "08-selected")
                await shot(cdp, "10-action-bar", ".action-bar")
                await shot(cdp, "11-telemetry-panel", ".telemetry-panel", pad=0)

                await cdp.js("shotImage.click()")
                await asyncio.sleep(2)
                await shot(cdp, "09-overlay")
                await cdp.js("closeShotOverlay(); clearChosen()")

                await metrics(cdp, *MOBILE, mobile=True)
                await cdp.js("scrollTo(0, 0)")
                await asyncio.sleep(2)
                await shot(cdp, "12-mobile-top")
                await cdp.js("document.querySelector('.telemetry-panel').scrollIntoView()")
                await asyncio.sleep(1.5)
                await shot(cdp, "13-mobile-panel")
                cdp.reader.cancel()
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
