from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
import webbrowser
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import AppConfig, clone_config
from .detection_worker import DetectionFrame, DetectionWorker, ProcessDetectionWorker
from .errors import DependencyError
from .ocr import Recognizer
from .pipeline import FramePacket, TelemetryPipeline


@dataclass(slots=True)
class BrowserSession:
    pipeline: TelemetryPipeline
    last_access: float


@dataclass(slots=True)
class DetectionState:
    worker: DetectionWorker
    temp_dir: tempfile.TemporaryDirectory[str]
    model_path: Path | None = None
    filename: str | None = None
    size_bytes: int = 0


_LOG = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# A request carrying this header can never be a CORS-safelisted "simple" request,
# so the browser must send a preflight the guard below can reject. Without it a
# multipart POST from any site the user has open would reach /api/detection/model,
# and that model is loaded by a pickle-based torch.load in the worker process.
GUARD_HEADER = "X-Telemetry-OCR"

_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

SESSION_TTL_SECONDS = 3600.0


def create_app(
    config: AppConfig,
    recognizer: Recognizer,
    *,
    static_dir: str | Path | None = None,
    max_sessions: int = 4,
    allowed_origins: Iterable[str] = (),
    allowed_hosts: Iterable[str] = (),
    detection_worker_factory: Callable[[], DetectionWorker] | None = None,
) -> Any:
    try:
        from aiohttp import web
    except ImportError as exc:
        raise DependencyError(
            "The web UI requires aiohttp; install the project with `pip install -e .[web]`."
        ) from exc

    assets = Path(static_dir) if static_dir else Path(__file__).with_name("web_static")
    if not (assets / "index.html").is_file():
        raise FileNotFoundError(f"Web assets not found: {assets}")

    if max_sessions < 1:
        raise ValueError("max_sessions must be at least 1")

    # The browser UI deliberately omits the OSD clock. It is not telemetry and
    # recognizing it consumes one ROI from every realtime batch. The clock is
    # whichever field derives itself from source time, not whatever happens to
    # be named "timestamp".
    web_config = clone_config(config)
    web_config.fields = [
        field for field in web_config.fields if not field.derive_from_source_time
    ]
    if not web_config.fields:
        raise ValueError("The web UI needs at least one field that is not the OSD clock")

    configured_origins = {origin.rstrip("/") for origin in allowed_origins if origin}
    configured_hosts = LOOPBACK_HOSTS | {
        hostname for hostname in (_hostname(host) for host in allowed_hosts) if hostname
    }

    def is_loopback_origin(origin: str) -> bool:
        try:
            return urlsplit(origin).hostname in LOOPBACK_HOSTS
        except ValueError:
            return False

    def is_allowed_origin(origin: str) -> bool:
        return is_loopback_origin(origin) or origin.rstrip("/") in configured_origins

    def is_allowed_host(request: Any) -> bool:
        return _hostname(request.headers.get("Host", "")) in configured_hosts

    @web.middleware
    async def guard_middleware(request: Any, handler: Any) -> Any:
        """Reject requests a browser must not be able to make on the user's behalf.

        The CORS middleware below only decides what a caller may *read*. Every
        endpoint here has side effects that happen before that, so the request
        itself has to be refused rather than merely hidden.
        """
        # A request whose Host is not this loopback server was routed here by a
        # rebound DNS name, not by the user typing the address.
        if not is_allowed_host(request):
            _LOG.warning(
                "Refused %s %s carrying Host %r",
                request.method,
                request.path,
                request.headers.get("Host", ""),
            )
            return web.json_response({"error": "Host is not allowed"}, status=403)
        if not request.path.startswith("/api/"):
            return await handler(request)
        origin = request.headers.get("Origin", "")
        if request.method != "OPTIONS" and origin and not is_allowed_origin(origin):
            _LOG.warning("Refused %s %s from origin %r", request.method, request.path, origin)
            return web.json_response({"error": "Origin is not allowed"}, status=403)
        if request.method in _MUTATING_METHODS and GUARD_HEADER not in request.headers:
            _LOG.warning(
                "Refused %s %s without %s", request.method, request.path, GUARD_HEADER
            )
            return web.json_response(
                {"error": f"{GUARD_HEADER} header is required"}, status=403
            )
        return await handler(request)

    @web.middleware
    async def cors_middleware(request: Any, handler: Any) -> Any:
        origin = request.headers.get("Origin", "")
        if request.method == "OPTIONS" and request.path.startswith("/api/"):
            response = web.Response(status=204)
        else:
            response = await handler(request)
        if origin and request.path.startswith("/api/") and is_allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = f"Content-Type, {GUARD_HEADER}"
            response.headers["Vary"] = "Origin"
            if request.headers.get("Access-Control-Request-Private-Network") == "true":
                response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    @web.middleware
    async def error_middleware(request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except (DependencyError, TypeError, ValueError) as exc:
            _LOG.warning("%s %s rejected: %s", request.method, request.path, exc)
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            # The client only receives the exception type, so without this the
            # reason for a 500 is lost entirely.
            _LOG.exception("%s %s failed", request.method, request.path)
            return web.json_response(
                {"error": f"OCR request failed: {type(exc).__name__}"}, status=500
            )

    model_upload_limit = 256 * 1024**2
    detection_frame_limit = 8 * 1024**2
    sessions_key = web.AppKey("sessions", dict)
    ocr_lock_key = web.AppKey("ocr_lock", asyncio.Lock)
    max_sessions_key = web.AppKey("max_sessions", int)
    detection_state_key = web.AppKey("detection_state", DetectionState)
    app = web.Application(
        client_max_size=model_upload_limit,
        middlewares=[guard_middleware, cors_middleware, error_middleware],
    )
    app[sessions_key] = {}
    app[ocr_lock_key] = asyncio.Lock()
    app[max_sessions_key] = max_sessions
    app[detection_state_key] = DetectionState(
        worker=(detection_worker_factory or ProcessDetectionWorker)(),
        temp_dir=tempfile.TemporaryDirectory(prefix="telemetry-ocr-model-"),
    )

    async def index(_: Any) -> Any:
        return web.FileResponse(
            assets / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    async def manifest(_: Any) -> Any:
        return web.FileResponse(
            assets / "manifest.webmanifest",
            headers={
                "Cache-Control": "no-cache",
                "Content-Type": "application/manifest+json",
            },
        )

    async def service_worker(_: Any) -> Any:
        return web.FileResponse(
            assets / "service-worker.js",
            headers={
                "Cache-Control": "no-cache",
                "Content-Type": "text/javascript; charset=utf-8",
                "Service-Worker-Allowed": "/",
            },
        )

    async def health(_: Any) -> Any:
        return web.json_response(
            {"status": "ready", "model": web_config.runtime.model_name}
        )

    async def config_payload(_: Any) -> Any:
        return web.json_response(
            {
                "model": web_config.runtime.model_name,
                "target_hz": web_config.runtime.ocr_hz,
                "padding": web_config.runtime.padding,
                "fields": [
                    {
                        "field_id": field.field_id,
                        "box": list(field.box),
                        "padding": (
                            web_config.runtime.padding
                            if field.padding is None
                            else field.padding
                        ),
                        "profile": field.profile,
                        "ocr_interval": field.ocr_interval,
                        "decimals": field.decimals,
                        "dms_degrees_digits": field.dms_degrees_digits,
                        "dms_decimal_places": field.dms_decimal_places,
                    }
                    for field in web_config.fields
                ],
            }
        )

    def new_pipeline() -> TelemetryPipeline:
        return TelemetryPipeline(web_config, recognizer)

    async def create_session(_: Any) -> Any:
        sessions: dict[str, BrowserSession] = app[sessions_key]
        # A closed tab never says so, and each session pins a pipeline with its
        # own temporal history, so idle ones are dropped on their own schedule.
        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        for stale in [key for key, item in sessions.items() if item.last_access < cutoff]:
            del sessions[stale]
        if sessions and len(sessions) >= app[max_sessions_key]:
            oldest = min(sessions, key=lambda key: sessions[key].last_access)
            sessions.pop(oldest, None)
        session_id = uuid.uuid4().hex
        sessions[session_id] = BrowserSession(new_pipeline(), time.monotonic())
        return web.json_response({"session_id": session_id}, status=201)

    def get_session(request: Any) -> BrowserSession:
        session = app[sessions_key].get(request.match_info["session_id"])
        if session is None:
            raise web.HTTPNotFound(
                text='{"error":"Session not found"}', content_type="application/json"
            )
        session.last_access = time.monotonic()
        return session

    async def reset_session(request: Any) -> Any:
        session = get_session(request)
        session.pipeline = new_pipeline()
        return web.json_response({"reset": True})

    async def recognize_frame(request: Any) -> Any:
        session = get_session(request)
        source_time_ms = float(request.query.get("source_time_ms", "nan"))
        frame_index = int(request.query.get("frame_index", "-1"))
        if not 0 <= source_time_ms < 86_400_000:
            raise ValueError("source_time_ms is outside the supported range")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        received_at = time.perf_counter()
        payload = await request.read()
        if not payload:
            raise ValueError("Frame body is empty")
        frame = await asyncio.to_thread(_decode_image, payload)
        packet = FramePacket(frame, frame_index, source_time_ms, received_at)
        async with app[ocr_lock_key]:
            records, latency_ms = await asyncio.to_thread(
                session.pipeline.process_frame, packet
            )
        return web.json_response(
            {
                "source_time_ms": source_time_ms,
                "frame_index": frame_index,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "latency_ms": latency_ms,
                "records": [asdict(record) for record in records],
            }
        )

    def detection_status_payload() -> dict[str, Any]:
        state = app[detection_state_key]
        payload = state.worker.status()
        payload["filename"] = state.filename or payload.get("filename")
        payload["size_bytes"] = state.size_bytes
        return payload

    async def detection_model_status(_: Any) -> Any:
        return web.json_response(detection_status_payload())

    async def upload_detection_model(request: Any) -> Any:
        if not request.content_type.startswith("multipart/"):
            raise ValueError("Model upload must use multipart/form-data")
        reader = await request.multipart()
        model_field = None
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "model":
                model_field = field
                break
        if model_field is None or not model_field.filename:
            raise ValueError("Select an Ultralytics YOLO .pt model")
        filename = Path(model_field.filename).name
        if Path(filename).suffix.lower() != ".pt":
            raise ValueError("Only Ultralytics YOLO .pt models are supported")

        state = app[detection_state_key]
        temp_dir = Path(state.temp_dir.name)
        model_path = temp_dir / f"{uuid.uuid4().hex}.pt"
        written = 0
        try:
            with model_path.open("wb") as handle:
                while True:
                    chunk = await model_field.read_chunk(size=1024**2)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > model_upload_limit:
                        raise ValueError("Model file exceeds the 256 MB limit")
                    handle.write(chunk)
            if written == 0:
                raise ValueError("Model file is empty")
            previous_path = state.model_path
            # Spawning the worker process is slow enough to stall every other
            # request, including the OCR frames the page is still sending.
            await asyncio.to_thread(state.worker.start, model_path, filename)
            state.model_path = model_path
            state.filename = filename
            state.size_bytes = written
            if previous_path is not None and previous_path != model_path:
                previous_path.unlink(missing_ok=True)
        except Exception:
            if state.model_path != model_path:
                model_path.unlink(missing_ok=True)
            raise
        return web.json_response(detection_status_payload(), status=202)

    async def clear_detection_model(_: Any) -> Any:
        state = app[detection_state_key]
        await asyncio.to_thread(state.worker.stop)
        if state.model_path is not None:
            state.model_path.unlink(missing_ok=True)
        state.model_path = None
        state.filename = None
        state.size_bytes = 0
        return web.json_response(detection_status_payload())

    async def restart_detection_worker(_: Any) -> Any:
        state = app[detection_state_key]
        if state.model_path is None:
            return web.json_response({"error": "No YOLO model is available"}, status=409)
        await asyncio.to_thread(state.worker.restart)
        return web.json_response(detection_status_payload(), status=202)

    async def reset_detection(request: Any) -> Any:
        source_id = _source_id(request.query.get("source_id", ""))
        generation = _non_negative_int(request.query.get("generation", "0"), "generation")
        app[detection_state_key].worker.reset(source_id, generation)
        return web.json_response(
            {
                "reset": True,
                "source_id": source_id,
                "generation": generation,
            }
        )

    def detection_result_payload(source_time_ms: float) -> tuple[dict[str, Any], int]:
        state = app[detection_state_key]
        worker_status = state.worker.status()
        latest = state.worker.latest()
        result_age_ms = None
        timing = None
        if latest is not None:
            result_age_ms = max(0.0, source_time_ms - float(latest["source_time_ms"]))
            latest, timing = _split_detection_timing(latest)
        payload = {
            "accepted": False,
            "worker": worker_status,
            "latest_result": latest,
            "result_age_ms": result_age_ms,
            "timing": timing,
            "dropped_frames": int(worker_status.get("dropped_frames", 0)),
        }
        return payload, 200 if latest is not None else 202

    async def detect_frame(request: Any) -> Any:
        state = app[detection_state_key]
        worker_status = state.worker.status()
        if worker_status.get("status") != "ready":
            return web.json_response(
                {
                    "error": worker_status.get("error") or "YOLO worker is not ready",
                    "worker": worker_status,
                },
                status=409,
            )
        source_id = _source_id(request.query.get("source_id", ""))
        generation = _non_negative_int(request.query.get("generation", "0"), "generation")
        sequence = _non_negative_int(request.query.get("sequence", "-1"), "sequence")
        source_time_ms = float(request.query.get("source_time_ms", "nan"))
        confidence = float(request.query.get("confidence", "0.45"))
        iou = float(request.query.get("iou", "0.45"))
        image_size = int(request.query.get("image_size", "640"))
        if not 0 <= source_time_ms < 86_400_000:
            raise ValueError("source_time_ms is outside the supported range")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if image_size < 320 or image_size > 1920:
            raise ValueError("image_size must be between 320 and 1920")
        payload = await request.read()
        if not payload:
            raise ValueError("Detection frame body is empty")
        if len(payload) > detection_frame_limit:
            raise ValueError("Detection frame exceeds the 8 MB limit")
        # submit() pickles up to 8 MB onto a multiprocessing queue and can spin
        # while the feeder thread catches up, so it does not belong on the loop.
        accepted = await asyncio.to_thread(
            state.worker.submit,
            DetectionFrame(
                payload=payload,
                source_id=source_id,
                generation=generation,
                sequence=sequence,
                source_time_ms=source_time_ms,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
            ),
        )
        response_payload, response_status = detection_result_payload(source_time_ms)
        response_payload["accepted"] = accepted
        response_payload["accepted_sequence"] = sequence if accepted else None
        return web.json_response(response_payload, status=response_status)

    async def latest_detection_result(request: Any) -> Any:
        source_time_ms = float(request.query.get("source_time_ms", "0"))
        if not 0 <= source_time_ms < 86_400_000:
            raise ValueError("source_time_ms is outside the supported range")
        payload, status = detection_result_payload(source_time_ms)
        return web.json_response(payload, status=status)

    async def options(_: Any) -> Any:
        return web.Response(status=204)

    async def cleanup(_: Any) -> None:
        state = app[detection_state_key]
        await asyncio.to_thread(state.worker.stop)
        state.temp_dir.cleanup()

    app.router.add_get("/", index)
    app.router.add_get("/manifest.webmanifest", manifest)
    app.router.add_get("/service-worker.js", service_worker)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/config", config_payload)
    app.router.add_post("/api/session", create_session)
    app.router.add_post("/api/session/{session_id}/reset", reset_session)
    app.router.add_post("/api/session/{session_id}/frame", recognize_frame)
    app.router.add_get("/api/detection/model", detection_model_status)
    app.router.add_post("/api/detection/model", upload_detection_model)
    app.router.add_delete("/api/detection/model", clear_detection_model)
    app.router.add_post("/api/detection/restart", restart_detection_worker)
    app.router.add_post("/api/detection/reset", reset_detection)
    app.router.add_post("/api/detection/frame", detect_frame)
    app.router.add_get("/api/detection/result", latest_detection_result)
    app.router.add_route("OPTIONS", "/api/{tail:.*}", options)
    app.router.add_static("/assets/", assets, show_index=False)
    app.on_cleanup.append(cleanup)
    return app


def run_web_app(
    config: AppConfig,
    recognizer: Recognizer,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    allowed_origins: Iterable[str] = (),
) -> None:
    try:
        from aiohttp import web
    except ImportError as exc:
        raise DependencyError(
            "The web UI requires aiohttp; install the project with `pip install -e .[web]`."
        ) from exc
    # The scheduled task runs this hidden, so refusals and 500s need somewhere
    # to go besides a dropped stderr.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = create_app(
        config, recognizer, allowed_origins=allowed_origins, allowed_hosts=(host,)
    )
    url = f"http://{host}:{port}"
    if open_browser:

        async def launch_browser(_: Any) -> None:
            asyncio.get_running_loop().call_later(0.5, webbrowser.open, url)

        app.on_startup.append(launch_browser)
    print(f"Telemetry OCR web UI: {url}")
    web.run_app(app, host=host, port=port, print=None)


def _decode_image(payload: bytes) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise DependencyError("OpenCV and NumPy are required to decode browser frames") from exc
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Frame is not a supported JPEG/PNG/WebP image")
    height, width = frame.shape[:2]
    if width > 4096 or height > 2160:
        raise ValueError("Frame dimensions exceed 4096x2160")
    return frame


def _hostname(value: str) -> str:
    """Extract the hostname from a `Host` header or a bare host string."""
    candidate = value.strip()
    if not candidate:
        return ""
    try:
        return (urlsplit(f"//{candidate}").hostname or "").lower()
    except ValueError:
        return ""


def _source_id(value: str) -> str:
    source_id = value.strip()
    if not source_id:
        raise ValueError("source_id is required")
    if len(source_id) > 128:
        raise ValueError("source_id must not exceed 128 characters")
    return source_id


def _non_negative_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _split_detection_timing(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn the worker's raw spans into a breakdown, and drop the raw clocks.

    `enqueued_at` and `available_at` are readings of this machine's monotonic
    clock. They mean nothing to a browser, so they are removed rather than
    serialised; only the differences leave the process.
    """
    trimmed = dict(result)
    available_at = trimmed.pop("available_at", None)
    trimmed.pop("enqueued_at", None)
    pipeline_ms = trimmed.pop("pipeline_ms", None)
    worker_total_ms = trimmed.get("worker_total_ms")

    transport_ms = None
    if pipeline_ms is not None and worker_total_ms is not None:
        # Everything the worker did not spend on the frame itself: queue wait
        # plus both IPC hops.
        transport_ms = max(0.0, float(pipeline_ms) - float(worker_total_ms))

    timing = {
        "decode_ms": trimmed.get("decode_ms"),
        "track_ms": trimmed.get("track_ms"),
        "worker_total_ms": worker_total_ms,
        "pipeline_ms": pipeline_ms,
        "transport_ms": transport_ms,
        # How long a finished result sat here unread. This is the delivery
        # delay the background pump exists to remove.
        "harvest_ms": (
            None
            if available_at is None
            else max(0.0, (time.perf_counter() - available_at) * 1000.0)
        ),
    }
    return trimmed, timing
