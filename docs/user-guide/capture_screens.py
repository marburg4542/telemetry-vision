"""Capture the screenshots the user guide shows, from a running server.

Drives headless Chrome over the DevTools protocol with aiohttp, which the web
extra already installs, so no browser-automation package is needed. The page is
the real one and the video a real flight clip: a guide whose pictures were
mocked up drifts from the app the first time either changes.

The moments shown with boxes are read out of the render's own tracks.jsonl, so
the pictures show the model finding something whichever checkpoint is used.

Side effects, on purpose and nothing more: three captures are written to
output/sessions/, and the video is rendered with the chosen model if it has not
been already -- cancelled once and continued, which is what the pictures of
those two states are of. A render the script starts only to photograph (when a
finished one already exists) is deleted again at the end. Nothing is sent to LINE.

Usage:
    .venv-cuda/Scripts/python.exe docs/user-guide/capture_screens.py [BASE_URL] [VIDEO_ID] [MODEL]
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
MODEL = sys.argv[3] if len(sys.argv) > 3 else ""
# "static" leaves the two pictures of a render in progress alone and takes
# only the ones a finished render can pose for, which needs no GPU -- what to
# pass when the machine is busy rendering something of its own.
STATIC = len(sys.argv) > 4 and sys.argv[4] == "static"
OUT = Path(__file__).resolve().parent / "assets" / "screens"
RENDERS = ROOT / "output" / "renders"
PORT = 9333
DESKTOP = (1440, 900)
MOBILE = (390, 844)

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


async def capture_at(cdp: Cdp, seconds: float) -> None:
    before = await cdp.js("state.shots.length")
    await seek(cdp, seconds)
    await cdp.js("requestCapture()")
    await cdp.wait_for(
        f"state.shots.length > {before} && !state.pendingCaptures && !state.captureQueued",
        timeout=120, label="capture",
    )


async def click_render_action(cdp: Cdp, render_id: str, label: str) -> None:
    """Press one button on one render's row; another render may be listed too."""
    await cdp.js(
        f"[...renderList.querySelectorAll('[data-render-id={json.dumps(render_id)}]"
        " .link-button')]"
        f".find((button) => button.textContent === {json.dumps(label)}).click()"
    )


async def export_panel_shot(cdp: Cdp, render_id: str) -> None:
    """The save-as-MP4 choices, opened and photographed, then closed. Nothing is exported."""
    await click_render_action(cdp, render_id, "บันทึก MP4")
    await cdp.wait_for("document.querySelector('.render-export')", label="the export panel")
    await asyncio.sleep(0.5)
    await shot(cdp, "05c-render-export", "#renderList")
    await click_render_action(cdp, render_id, "บันทึก MP4")


def box_times(render_id: str, fallback=(19.8, 17.6, 5.0)) -> tuple[float, ...]:
    """Three moments the render found the most in, far enough apart to differ."""
    path = RENDERS / render_id / "tracks.jsonl"
    found: list[tuple[float, float]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            area = 0.0
            for box in record.get("detections", []):
                area += (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
            if area:
                found.append((area, float(record["time_ms"]) / 1000.0))
    except (OSError, ValueError, KeyError):
        return fallback
    chosen: list[float] = []
    for _, seconds in sorted(found, reverse=True):
        if all(abs(seconds - taken) > 3 for taken in chosen):
            chosen.append(seconds)
        if len(chosen) == 3:
            break
    return tuple(chosen) if chosen else fallback


async def render_for_the_pictures(cdp: Cdp) -> str:
    if STATIC:
        render_id = await cdp.js("(matchingRender() || {}).render_id || ''")
        if not render_id:
            raise SystemExit("static mode needs a finished render for these settings")
        print("static: keeping the pictures of a render in progress as they are")
        return render_id

    """Render the clip, photographing it being made, cancelled and continued.

    Returns the render id, and whether it was made only for the pictures -- in
    which case the caller removes it again.
    """
    finished = await cdp.js("Boolean(matchingRender())")
    if finished:
        # A finished render is already there to photograph playing, so the
        # in-progress pictures are taken of a throwaway at another confidence --
        # not another stride, which the panel in those pictures would show.
        await cdp.js(
            "renderConfidence.value = '0.50';"
            " renderConfidence.dispatchEvent(new Event('change'))"
        )
    await cdp.js("renderButton.click()")
    await cdp.wait_for(
        "state.renderJob && state.renderJob.status === 'running' && state.renderProcessedS > 12",
        timeout=600, label="enough of the render to photograph",
    )
    await asyncio.sleep(2)
    await shot(cdp, "05-render-progress", "#aiPanel")
    render_id = await cdp.js("state.activeRender && state.activeRender.render_id")

    await cdp.js("renderCancelButton.click()")
    await cdp.wait_for(
        "state.renders.some((render) => render.render_id === "
        f"{json.dumps(render_id)} && render.status === 'partial')",
        timeout=120, label="the partial render",
    )
    await asyncio.sleep(1.5)
    await shot(cdp, "05b-render-partial", "#aiPanel")

    await click_render_action(cdp, render_id, "ทำต่อ")
    if not finished:
        await cdp.wait_for(
            "state.renders.some((render) => render.render_id === "
            f"{json.dumps(render_id)} && render.status === 'complete')",
            timeout=1800, label="the render to finish",
        )
        return render_id

    # The throwaway was only ever for the pictures: let it prove it continues,
    # then stop it and take it away again.
    await cdp.wait_for(
        "state.renderJob && state.renderJob.status === 'running'",
        timeout=300, label="the continued render",
    )
    await cdp.js("renderCancelButton.click()")
    await cdp.wait_for(
        "state.renderJob && state.renderJob.status === 'cancelled'",
        timeout=120, label="the throwaway to stop",
    )
    await cdp.js(
        "renderConfidence.value = '0.45'; renderConfidence.dispatchEvent(new Event('change'))"
    )
    await cdp.js(
        "(async () => {"
        f"  const params = new URLSearchParams({{ render: {json.dumps(render_id)} }});"
        "  await fetch(`/api/renders?${params}`,"
        "    { method: 'DELETE', headers: { 'X-Telemetry-OCR': '1' } });"
        "  await refreshRenders();"
        "})()"
    )
    await asyncio.sleep(1)
    return await cdp.js("(matchingRender() || {}).render_id || ''")


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

                await cdp.js(
                    f"videoSelect.value = {json.dumps(VIDEO)};"
                    " videoSelect.dispatchEvent(new Event('change'))"
                )
                if MODEL:
                    await cdp.js(
                        f"modelSelect.value = {json.dumps(MODEL)};"
                        " modelSelect.dispatchEvent(new Event('change'))"
                    )
                await cdp.wait_for("state.video && sourceVideo.readyState >= 2", label="the video")

                render_id = await render_for_the_pictures(cdp)
                times = box_times(render_id)
                print("moments with boxes:", times)

                # The button plays the finished render when nothing is on screen;
                # when the render just made is already showing, it is the switch.
                await cdp.js(
                    "if (state.activeRender) viewRendered.click(); else renderButton.click()"
                )
                await cdp.wait_for(
                    "state.view === 'rendered' && sourceVideo.readyState >= 2", label="the AI video"
                )
                await seek(cdp, times[0])
                await shot(cdp, "03-ai-video")
                await shot(cdp, "04-ai-panel", "#aiPanel")
                await export_panel_shot(cdp, render_id)

                await cdp.js("viewOriginal.click()")
                await cdp.wait_for(
                    "state.view === 'original' && sourceVideo.readyState >= 2", label="the original"
                )
                await seek(cdp, times[0])
                await shot(cdp, "02-original")
                await cdp.js("viewRendered.click()")
                await cdp.wait_for(
                    "state.view === 'rendered' && sourceVideo.readyState >= 2", label="the AI video"
                )
                await seek(cdp, times[0])

                await cdp.js("manageModelsButton.click()")
                await cdp.wait_for(
                    "registeredModels.querySelectorAll('.model-item').length > 0", label="models"
                )
                await asyncio.sleep(1)
                await shot(cdp, "06-model-dialog")
                await cdp.js("modelDialog.close()")

                for seconds in times:
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
