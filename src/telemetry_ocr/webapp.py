from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
import webbrowser
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import __version__
from .config import AppConfig, clone_config
from .detection_worker import DetectionFrame, DetectionWorker, ProcessDetectionWorker
from .errors import DependencyError
from .ocr import Recognizer
from .pipeline import FramePacket, TelemetryPipeline
from .public_share import DEFAULT_SHARE_PORT, ShareServer, ShareStore


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

# What a capture asks the detector for. A capture is one deliberate frame, not a
# stream, so it can afford the larger input the checkpoints were trained at and
# a threshold low enough to show the marginal boxes with their confidence
# attached rather than silently dropping them.
_CAPTURE_DETECT_IMAGE_SIZE = 960
_CAPTURE_DETECT_CONFIDENCE = 0.25
# Long enough for a cold model on this card, short enough that a wedged worker
# costs the boxes rather than the capture.
_CAPTURE_DETECT_TIMEOUT_S = 20.0
# How long one submission is given before it is assumed lost and sent again.
# Inference on this card is well under a second, so anything past a few seconds
# means the frame was displaced rather than merely slow.
_CAPTURE_DETECT_ATTEMPT_S = 5.0

# Smallest gap between two LINE pushes. Every push spends one message from the
# account's monthly quota, so a wedged client or a held button must not be able
# to drain it; three seconds is far below anything a person does deliberately.
_LINE_MIN_INTERVAL_S = 3.0

# How often the supervisor re-checks a configured stream. Long enough that an
# unreachable camera costs one connection attempt every half minute rather than
# a tight loop, short enough that a camera coming up is picked up promptly.
_STREAM_SUPERVISOR_INTERVAL_S = 30.0


def create_app(
    config: AppConfig,
    recognizer: Recognizer,
    *,
    static_dir: str | Path | None = None,
    max_sessions: int = 4,
    stream_url: str | None = None,
    stream_output_dir: str | Path = "output",
    share_port: int = DEFAULT_SHARE_PORT,
    quick_tunnel: bool = False,
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
        # No exceptions. Images are served by a separate process listening on
        # its own port, which is what the tunnel is pointed at, so nothing
        # reaching this application ever needs to arrive without a guard header
        # or with a public Host.
        #
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
    async def static_cache_middleware(request: Any, handler: Any) -> Any:
        """Make the browser revalidate the app's own assets.

        add_static sends ETag and Last-Modified but no Cache-Control, and a
        response with neither a max-age nor a no-cache is left to the browser's
        heuristic freshness -- commonly a tenth of the age of the file. A page
        that had been open a while therefore kept running a previous build of
        app.js after a redeploy, silently, which is precisely what the service
        worker was made network-first to prevent. The ETag is already there, so
        revalidating costs a 304.
        """
        response = await handler(request)
        if request.path.startswith("/assets/"):
            response.headers.setdefault("Cache-Control", "no-cache")
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

    stream_output_dir = Path(stream_output_dir)
    model_upload_limit = 256 * 1024**2
    detection_frame_limit = 8 * 1024**2
    sessions_key = web.AppKey("sessions", dict)
    ocr_lock_key = web.AppKey("ocr_lock", asyncio.Lock)
    max_sessions_key = web.AppKey("max_sessions", int)
    detection_state_key = web.AppKey("detection_state", DetectionState)
    # `revision` counts results this server has published. It is deliberately
    # NOT the client's frame sequence: the page restarts that at zero on every
    # seek, model change and settings change, so a watermark built on it goes
    # stale the moment the client resets and then reports "nothing newer" to a
    # request that is in fact waiting for everything. A counter the client
    # cannot influence is always comparable.
    #
    # Kept outside app state because aiohttp deprecates mutating that after
    # startup. The event is replaced rather than cleared on each result: that
    # releases every waiter on the old object exactly once and leaves no window
    # for a waiter to arrive between a set and a clear.
    pump_state: dict[str, Any] = {
        "revision": 0,
        "seen": None,
        "ready": asyncio.Event(),
        # The task handle lives here for the same reason the event does:
        # aiohttp freezes the Application before on_startup runs, so storing
        # it in app state warns on every assignment and is slated to become
        # an error.
        "task": None,
    }
    # The server-side stream, when one is running. `ready` follows the same
    # replace-on-notify idiom as the pump above, for the same reason: it is set
    # from the service's own threads by way of the loop, and replacing it wakes
    # every waiter exactly once with no window between a set and a clear.
    # Open capture sessions, by name. Held for the life of the server so that
    # pressing the button repeatedly appends to one recording rather than
    # reopening a directory and re-reading its numbering every time.
    capture_sessions: dict[str, Any] = {}
    # When the last LINE push went out, so the rate limit survives across
    # requests without being stored on the frozen Application.
    line_state: dict[str, float] = {"last_sent": 0.0}
    # Tokens for images LINE is allowed to fetch. Created only when a send needs
    # one, and expiring on their own.
    share_store = ShareStore(Path(stream_output_dir) / "sessions")
    # Started only when something needs it: an application that never sends a
    # picture should not have a second listener open for no reason.
    share_state: dict[str, Any] = {"server": None, "tunnel": None}

    async def ensure_share_server() -> Any:
        if share_state["server"] is None:
            server = ShareServer(share_store, share_port)
            await server.start()
            share_state["server"] = server
        return share_state["server"]

    async def ensure_quick_tunnel() -> str | None:
        """The address LINE should fetch from, starting the tunnel if needed."""
        from .quick_tunnel import QuickTunnel

        tunnel = share_state["tunnel"]
        if tunnel is None:
            tunnel = QuickTunnel(share_port)
            share_state["tunnel"] = tunnel
        if tunnel.url:
            return str(tunnel.url)
        # Starting it blocks for as long as cloudflared takes to be given an
        # address, so it happens off the loop.
        return await asyncio.to_thread(tunnel.start)
    stream_state: dict[str, Any] = {
        "service": None,
        "ready": asyncio.Event(),
        "loop": None,
        "supervisor": None,
    }
    app = web.Application(
        client_max_size=model_upload_limit,
        middlewares=[
            guard_middleware,
            cors_middleware,
            static_cache_middleware,
            error_middleware,
        ],
    )
    app[sessions_key] = {}
    app[ocr_lock_key] = asyncio.Lock()
    app[max_sessions_key] = max_sessions
    app[detection_state_key] = DetectionState(
        worker=(detection_worker_factory or ProcessDetectionWorker)(),
        temp_dir=tempfile.TemporaryDirectory(prefix="telemetry-ocr-model-"),
    )
    # A capture takes the detector for itself while it runs. One at a time, and
    # the live overlay stands down for the duration -- see detect_on_frame.
    detector_claim = asyncio.Lock()
    detection_holds = {"captures": 0}

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
            {
                "status": "ready",
                "model": web_config.runtime.model_name,
                # The page compares this against its own constant. A stale
                # service-worker copy of app.js reporting an older build was
                # mistaken for a behaviour bug once; now it says so out loud.
                "version": __version__,
            }
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
        # One recognizer, one process. The stream service calls it from its own
        # thread, outside this loop's lock, so letting a second tab post frames
        # at the same time would mean two threads inside PaddleOCR at once.
        service = stream_state["service"]
        if service is not None and service.running:
            return web.json_response(
                {
                    "error": (
                        "A live stream is using OCR. Stop the stream to run OCR "
                        "on a file in this tab."
                    ),
                    "stream": service.status(),
                },
                status=409,
            )
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
            async with pump_paused():
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
        async with pump_paused():
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
        async with pump_paused():
            await asyncio.to_thread(state.worker.restart)
        return web.json_response(detection_status_payload(), status=202)

    async def reset_detection(request: Any) -> Any:
        source_id = _source_id(request.query.get("source_id", ""))
        generation = _non_negative_int(request.query.get("generation", "0"), "generation")
        # reset() drains the frame queue, which is blocking IPC with up to 8 MB
        # of pickled JPEG to unpickle and throw away. Every other worker call is
        # already off the loop for the same reason.
        await asyncio.to_thread(app[detection_state_key].worker.reset, source_id, generation)
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
            # Echoed back by the client as `after_revision`. It must record this
            # even for a result it discards, or it would ask for the same
            # revision forever and never block.
            "revision": int(pump_state["revision"]),
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
        if detection_holds["captures"]:
            # A capture has the detector. This frame is one of many and the next
            # one is 400 ms away; the capture is a deliberate act that has to
            # come back with boxes on the picture being saved. Skipped rather
            # than queued, because a frame held for half a second and then
            # inferred would draw boxes from the wrong moment.
            deferred, deferred_status = detection_result_payload(source_time_ms)
            deferred["accepted"] = False
            deferred["accepted_sequence"] = None
            deferred["deferred"] = True
            return web.json_response(deferred, status=deferred_status)
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
        after_revision = _non_negative_int(
            request.query.get("after_revision", "0"), "after_revision"
        )
        wait_ms = _non_negative_int(request.query.get("wait_ms", "0"), "wait_ms")
        # Capped so a poll cannot outlive shutdown, and so a client cannot pin a
        # connection open indefinitely.
        wait_ms = min(wait_ms, 1000)

        if wait_ms:
            deadline = time.monotonic() + wait_ms / 1000.0
            while pump_state["revision"] <= after_revision:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready = pump_state["ready"]
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(ready.wait(), remaining)

        payload, status = detection_result_payload(source_time_ms)
        return web.json_response(payload, status=status)

    async def options(_: Any) -> Any:
        return web.Response(status=204)

    async def pump_detection_results(_: Any) -> None:
        """Collect finished results instead of waiting for a request to do it.

        Without this the worker's output sits in the IPC queue until some
        handler calls status(), latest() or submit(), which adds a whole
        submission interval to every frame's latency — and the queue holds only
        four before the worker starts discarding the oldest.

        It runs on the event loop rather than in a thread on purpose: the
        worker's bookkeeping is read by handlers on this loop, so a second
        thread mutating it would reintroduce the race that moving the drain out
        of submit() removed.
        """
        state = app[detection_state_key]
        while True:
            try:
                state.worker.drain()
                latest = state.worker.latest()
                # Identity rather than ordering: a reset clears the stored
                # result and the next one may reuse a sequence already seen, so
                # "different from what was published last" is the only test that
                # survives the client renumbering itself.
                seen = None if latest is None else (
                    latest.get("source_id"),
                    latest.get("generation"),
                    latest.get("sequence"),
                )
                if seen != pump_state["seen"]:
                    pump_state["seen"] = seen
                    if seen is not None:
                        pump_state["revision"] += 1
                        pump_state["ready"].set()
                        pump_state["ready"] = asyncio.Event()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A worker that is unloaded, stopped or mid-restart must not be
                # able to kill the pump; the next tick picks it up again.
                _LOG.exception("Detection result pump failed")
            await asyncio.sleep(0.01)

    @asynccontextmanager
    async def pump_paused() -> Any:
        """Hold the pump still while the worker's queues are torn down.

        stop() closes both queues from a thread; a pump tick landing in that
        window calls get_nowait() on a closed queue and logs a stack trace every
        ten milliseconds until the teardown finishes. cleanup already ordered
        things this way — these are the paths that did not.
        """
        pump = pump_state["task"]
        if pump is not None:
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump
        try:
            yield
        finally:
            pump_state["task"] = asyncio.create_task(pump_detection_results(None))

    async def supervise_stream(_: Any) -> None:
        """Keep the configured stream running, without needing anyone present.

        The point of an auto-started server is that nobody has to be. A camera is
        rarely up at the moment Windows logs in, and a stream can drop for longer
        than the decoder's own backoff covers, so one attempt at startup would
        leave a server that looks healthy and is reading nothing.

        Deliberately not infinitely eager: the retry interval is long enough that
        an unreachable camera costs one connection attempt every half minute
        rather than a tight loop against a host that is not answering.
        """
        from .stream_service import StreamService

        assert stream_url is not None  # only scheduled when one is configured
        while True:
            service = stream_state["service"]
            if service is None or not service.running:
                if service is not None and service.status()["error"]:
                    _LOG.warning(
                        "Configured stream is down (%s); retrying",
                        service.status()["error"],
                    )
                try:
                    started = StreamService(
                        web_config,
                        recognizer,
                        stream_url,
                        jsonl_path=stream_output_dir / "stream-telemetry.jsonl",
                        csv_path=stream_output_dir / "stream-telemetry.csv",
                        on_update=notify_stream,
                    )
                    stream_state["service"] = started
                    started.start()
                    _LOG.info("Started the configured stream: %s", stream_url)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _LOG.exception("Could not start the configured stream")
            await asyncio.sleep(_STREAM_SUPERVISOR_INTERVAL_S)

    async def start_pump(_: Any) -> None:
        pump_state["task"] = asyncio.create_task(pump_detection_results(None))
        # Captured here so the stream service, which lives on its own threads,
        # has a loop to hand its notifications back to.
        stream_state["loop"] = asyncio.get_running_loop()
        if stream_url:
            stream_state["supervisor"] = asyncio.create_task(supervise_stream(None))

    async def cleanup(_: Any) -> None:
        # First: it holds a child process, and leaving cloudflared running after
        # the server it fronts has gone would leave an address answering 502.
        tunnel = share_state["tunnel"]
        if tunnel is not None:
            await asyncio.to_thread(tunnel.stop)
        server = share_state["server"]
        if server is not None:
            await server.stop()
        supervisor = stream_state["supervisor"]
        if supervisor is not None:
            # Before the stop below, or it would helpfully start a replacement
            # for the stream that is being shut down.
            supervisor.cancel()
            with suppress(asyncio.CancelledError):
                await supervisor
        service = stream_state["service"]
        if service is not None:
            # First, so the camera is released and the OCR thread is idle before
            # anything it shares is torn down underneath it.
            await asyncio.to_thread(service.stop)
            stream_state["ready"].set()
        pump = pump_state["task"]
        if pump is not None:
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump
        # Released after the pump stops so a long poll cannot outlive it.
        pump_state["ready"].set()
        state = app[detection_state_key]
        try:
            await asyncio.to_thread(state.worker.stop)
        finally:
            # Without the finally a failing stop() stranded the uploaded
            # checkpoint — up to 256 MB — and its directory for good.
            state.temp_dir.cleanup()

    def wake_stream_waiters() -> None:
        stream_state["ready"].set()
        stream_state["ready"] = asyncio.Event()

    def notify_stream() -> None:
        """Called from the service's threads, so it may not touch the loop."""
        loop = stream_state["loop"]
        if loop is None:
            return
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(wake_stream_waiters)

    def current_stream() -> Any:
        service = stream_state["service"]
        if service is None:
            raise web.HTTPNotFound(text="No stream has been started")
        return service

    async def stream_start(request: Any) -> Any:
        from .stream_service import StreamService

        body = await request.json()
        url = str(body.get("url", "")).strip()
        if not url:
            raise ValueError("A stream URL is required")
        existing = stream_state["service"]
        if existing is not None and existing.running:
            raise ValueError(
                "A stream is already running. Stop it before starting another: "
                "one camera at a time, since they share this process's OCR."
            )
        if existing is not None:
            # A finished service is not restartable, and leaving it in place
            # would report the previous run's error against the new URL.
            await asyncio.to_thread(existing.stop)
        service = StreamService(
            web_config,
            recognizer,
            url,
            jsonl_path=stream_output_dir / "stream-telemetry.jsonl",
            csv_path=stream_output_dir / "stream-telemetry.csv",
            on_update=notify_stream,
        )
        stream_state["service"] = service
        service.start()
        return web.json_response({"started": True, "status": service.status()})

    async def stream_stop(_: Any) -> Any:
        supervisor = stream_state["supervisor"]
        if supervisor is not None:
            # Stopping means stopping. Leaving the supervisor running would have
            # it restart the stream within the interval, which reads as the
            # button not working.
            supervisor.cancel()
            with suppress(asyncio.CancelledError):
                await supervisor
            stream_state["supervisor"] = None
        service = stream_state["service"]
        if service is None:
            return web.json_response({"stopped": False, "reason": "no stream"})
        # stop() waits on a decoder that may be blocked on a silent camera until
        # its read timeout expires, so it cannot run on the loop.
        await asyncio.to_thread(service.stop)
        wake_stream_waiters()
        return web.json_response({"stopped": True, "status": service.status()})

    async def stream_status(_: Any) -> Any:
        service = stream_state["service"]
        if service is None:
            return web.json_response({"running": False, "url": None, "error": None})
        return web.json_response(service.status())

    async def stream_telemetry(request: Any) -> Any:
        service = current_stream()
        after_revision = _non_negative_int(
            request.query.get("after_revision", "0"), "after_revision"
        )
        # Longer than the detection cap because this waits on an event rather
        # than spinning: an idle stream should cost one held connection, not a
        # poll every second.
        wait_ms = min(_non_negative_int(request.query.get("wait_ms", "0"), "wait_ms"), 5000)
        if wait_ms:
            deadline = time.monotonic() + wait_ms / 1000.0
            while service.snapshot().revision <= after_revision and service.running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready = stream_state["ready"]
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(ready.wait(), remaining)
        snapshot = service.snapshot()
        return web.json_response(
            {
                "running": service.running,
                "revision": snapshot.revision,
                "source_time_ms": snapshot.source_time_ms,
                "frame_index": snapshot.frame_index,
                "latency_ms": snapshot.latency_ms,
                "fields": snapshot.fields,
                "error": service.status()["error"],
            }
        )

    async def stream_frame(_: Any) -> Any:
        service = current_stream()
        payload, revision = service.preview()
        if payload is None:
            # 202: the stream exists but has not produced a picture yet, which
            # is different from there being no stream at all.
            return web.json_response({"error": "No frame yet"}, status=202)
        return web.Response(
            body=payload,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Stream-Revision": str(revision)},
        )

    async def stream_mjpeg(request: Any) -> Any:
        """The preview as one long-lived response the browser renders natively.

        An <img> pointed at this needs no polling loop, no canvas and no
        JavaScript to stay current, which is what keeps the request rate at one
        per stream instead of one per frame -- the failure this codebase has
        already been bitten by once.
        """
        service = current_stream()
        response = web.StreamResponse(
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)
        sent_revision = 0
        try:
            while True:
                service = stream_state["service"]
                if service is None or not service.running:
                    break
                payload, revision = service.preview()
                if payload is None or revision <= sent_revision:
                    ready = stream_state["ready"]
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(ready.wait(), 1.0)
                    continue
                sent_revision = revision
                await response.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(payload)).encode("ascii")
                    + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )
        except (ConnectionResetError, ConnectionAbortedError):
            # The viewer closed the tab. Normal, and not worth a stack trace.
            pass
        return response

    async def detect_on_frame(frame: Any) -> dict[str, Any] | None:
        """Run the detector on this exact frame and wait for the answer.

        The alternative -- reading whatever result happened to be latest -- is
        what put boxes on a capture that visibly did not sit on the vehicles:
        the newest result belongs to whichever frame the page last posted, and
        on a moving camera that is a different scene. A capture is a deliberate
        act with time to spare, so it pays for its own inference.

        Returns None when there is no model, or when the frame was superseded in
        the worker's newest-wins queue before it was inferred.
        """
        import uuid as _uuid

        state = app[detection_state_key]
        try:
            if state.worker.status().get("status") != "ready":
                return None
        except Exception:
            _LOG.exception("Could not read detector status for a capture")
            return None

        payload = await asyncio.to_thread(_encode_for_detection, frame)
        if payload is None:
            return None

        # Claimed for the duration, because otherwise the live overlay takes the
        # detector back before the capture's own frame has been inferred.
        #
        # The worker keeps one active source id and drops any result that does
        # not carry it, so submitting under a fresh `capture-` id and then
        # letting the page submit its next overlay frame does not merely
        # overwrite the capture's answer -- it discards it. The overlay posts a
        # frame at least every 400 ms while a video plays and inference takes
        # longer than that, so with a video running the capture lost the race
        # nearly every time: every shot came back "no_result" with no boxes
        # drawn, while the same capture on a paused page succeeded.
        async with detector_claim:
            detection_holds["captures"] += 1
            try:
                deadline = time.monotonic() + _CAPTURE_DETECT_TIMEOUT_S
                while time.monotonic() < deadline:
                    # A fresh id per attempt. Reusing one would let a stale
                    # answer from the previous attempt satisfy this one.
                    source_id = f"capture-{_uuid.uuid4().hex[:12]}"
                    accepted = await asyncio.to_thread(
                        state.worker.submit,
                        DetectionFrame(
                            payload=payload,
                            source_id=source_id,
                            generation=0,
                            sequence=0,
                            source_time_ms=0.0,
                            confidence=_CAPTURE_DETECT_CONFIDENCE,
                            iou=0.45,
                            image_size=_CAPTURE_DETECT_IMAGE_SIZE,
                        ),
                    )
                    if not accepted:
                        return None
                    attempt_ends = min(
                        deadline, time.monotonic() + _CAPTURE_DETECT_ATTEMPT_S
                    )
                    while time.monotonic() < attempt_ends:
                        latest = state.worker.latest()
                        if latest and latest.get("source_id") == source_id:
                            return dict(latest)
                        await asyncio.sleep(0.02)
                    # An overlay frame already in flight when the claim was taken
                    # can still displace the first attempt; a second one cannot,
                    # because nothing new is being submitted while this holds.
                    _LOG.info("Captured frame was displaced in the detector; resubmitting")
                _LOG.info("Detector did not return a result for the captured frame in time")
                return None
            finally:
                detection_holds["captures"] -= 1

    def summarise_detections(result: dict[str, Any] | None) -> Any:
        """Counts and confidences for a result that belongs to this frame."""
        from collections import Counter

        from .capture_service import DetectionSummary

        state = app[detection_state_key]
        try:
            ready = state.worker.status().get("status") == "ready"
        except Exception:
            ready = False
        if not ready:
            return DetectionSummary("no_model")
        if result is None:
            return DetectionSummary("no_result")

        boxes = result.get("detections") or []
        counts = Counter(str(item.get("label", "?")) for item in boxes)
        scores: dict[str, list[float]] = {}
        for item in boxes:
            with suppress(TypeError, ValueError):
                scores.setdefault(str(item.get("label", "?")), []).append(
                    float(item.get("confidence", 0.0))
                )
        confidence = {
            label: {
                "mean": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
            for label, values in scores.items()
            if values
        }
        return DetectionSummary(
            status="ok",
            counts=dict(counts),
            confidence=confidence,
            total=sum(counts.values()),
            # Zero, and meant literally: this is the frame that was inferred.
            age_ms=0.0,
        )

    def shot_verifier(live_records: list[Any]) -> Any:
        """Build the callback that re-reads a saved frame.

        Deliberately a fresh pipeline each time: feeding the frame through the
        one that produced the live reading would put it into the same temporal
        voting, and a check sharing state with the thing it checks is not a
        check.
        """
        from .capture_service import verify_shot
        from .pipeline import TelemetryPipeline

        def verify(image_path: Path) -> Any:
            # A config of its own, with the fallback recipes switched on.
            #
            # The realtime path disables them to hold its frame budget: a second
            # preprocessing pass per struggling ROI is time it does not have. A
            # capture has all the time it needs, and this is the reading that
            # goes out, so it should be the most thorough one the pipeline can
            # produce rather than the cheapest.
            thorough = clone_config(web_config)
            thorough.runtime.enable_fallback = True
            return verify_shot(image_path, live_records, TelemetryPipeline(thorough, recognizer))

        return verify

    def capture_session_for(name: str, source: str, is_stream: bool = False) -> Any:
        from .capture_service import CaptureSession

        existing = capture_sessions.get(name)
        if existing is not None:
            return existing
        session = CaptureSession(
            stream_output_dir / "sessions",
            name,
            source,
            [field.field_id for field in web_config.fields],
            is_stream=is_stream,
        )
        capture_sessions[name] = session
        return session

    async def capture_shot(request: Any) -> Any:
        """Keep this frame and what it says.

        Two sources, because two different things own the video. A running
        stream is already being read by this process, so the reading it has just
        published is reused: recognising the frame again would put a second
        thread inside PaddleOCR, which is the collision the stream guard exists
        to prevent, and would answer slightly differently for the same picture.
        A file playing in the browser is not being read by anyone until asked,
        so the posted frame is recognised here.
        """
        name = (request.query.get("session") or "").strip()
        if not name:
            raise ValueError("session is required")
        source_time_ms = float(request.query.get("source_time_ms", "0"))
        if not 0 <= source_time_ms < 86_400_000:
            raise ValueError("source_time_ms is outside the supported range")

        service = stream_state["service"]
        if service is not None and service.running:
            # One call, one lock: see StreamService.current_reading.
            frame, snapshot = service.current_reading()
            if frame is None:
                return web.json_response(
                    {"error": "The stream has not produced a frame yet"}, status=409
                )
            records = [
                {
                    "source_time_ms": snapshot.source_time_ms,
                    "frame_index": snapshot.frame_index,
                    "field_id": field_id,
                    "value": item.get("value"),
                    "raw_text": item.get("raw_text"),
                    "confidence": item.get("confidence"),
                    "confirmed": item.get("confirmed"),
                    "preprocess_variant": item.get("preprocess_variant", ""),
                    "latency_ms": snapshot.latency_ms,
                }
                for field_id, item in snapshot.fields.items()
            ]
            session = capture_session_for(name, service.url, is_stream=True)
            detected = await detect_on_frame(frame)
            boxes, box_source = _boxes_of(detected)
            shot = await asyncio.to_thread(
                session.save,
                frame,
                snapshot.source_time_ms,
                records,
                summarise_detections(detected),
                shot_verifier(records),
                boxes,
                box_source,
            )
            session.record_verification(shot)
            return web.json_response(asdict(shot))

        browser_session = app[sessions_key].get(request.query.get("session_id", ""))
        if browser_session is None:
            raise web.HTTPNotFound(
                text='{"error":"Session not found"}', content_type="application/json"
            )
        browser_session.last_access = time.monotonic()
        payload = await request.read()
        if not payload:
            raise ValueError("Capture frame body is empty")
        frame = await asyncio.to_thread(_decode_image, payload)
        packet = FramePacket(frame, 0, source_time_ms, time.perf_counter())
        async with app[ocr_lock_key]:
            recognised, _ = await asyncio.to_thread(
                browser_session.pipeline.process_frame, packet
            )
        session = capture_session_for(name, request.query.get("source", "browser"))
        detected = await detect_on_frame(frame)
        boxes, box_source = _boxes_of(detected)
        shot = await asyncio.to_thread(
            session.save,
            frame,
            source_time_ms,
            recognised,
            summarise_detections(detected),
            shot_verifier([_as_record(item) for item in recognised]),
            boxes,
            box_source,
        )
        session.record_verification(shot)
        return web.json_response(asdict(shot))

    def open_session(name: str) -> Any:
        """The session by that name, reopened from disk if this run did not make it."""
        session = capture_sessions.get(name)
        if session is None and (stream_output_dir / "sessions" / name).is_dir():
            session = capture_session_for(name, "reopened")
        return session

    async def capture_shots(request: Any) -> Any:
        """Everything captured in this session, newest first.

        Enough per shot for the page to show it and decide about it -- the
        reading, what was detected, whether a marked-up copy exists -- without a
        second request per row.
        """
        name = (request.query.get("session") or "").strip()
        if not name:
            raise ValueError("session is required")
        session = open_session(name)
        if session is None:
            return web.json_response({"session": name, "shots": []})

        def collect() -> list[dict[str, Any]]:
            shots = []
            for shot in session.all_shots():
                data = asdict(shot)
                # The records are the reading the image itself gave, which is
                # what the page shows; the raw manifest carries more than that.
                data["records"] = [
                    {
                        "field_id": item.field_id,
                        "value": item.value,
                        "raw_text": item.raw_text,
                        "confidence": item.confidence,
                        "confirmed": item.confirmed,
                    }
                    for item in shot.verification.fields
                ] or shot.records
                # Where this shot got to, in one word the page can act on. The
                # frame is written before the model is asked, so a shot that
                # exists is not yet a shot that has been through best.pt, and
                # "detected nothing" must not look the same as "never ran".
                data["stage"] = _shot_stage(shot)
                shots.append(data)
            shots.reverse()
            return shots

        return web.json_response(
            {"session": name, "shots": await asyncio.to_thread(collect)}
        )

    async def capture_discard(request: Any) -> Any:
        """Throw away one capture.

        Separate from the tick that chooses what to send, because they are
        different intentions and only one of them cannot be undone. The page
        asks before calling this.
        """
        name = (request.query.get("session") or "").strip()
        if not name:
            raise ValueError("session is required")
        session = open_session(name)
        if session is None:
            raise web.HTTPNotFound(
                text='{"error":"No such capture session"}', content_type="application/json"
            )
        indexes = [
            _non_negative_int(value, "frame_index")
            for value in request.query.getall("frame_index", [])
            if value != ""
        ]
        if not indexes:
            raise ValueError("frame_index is required")
        def discard_all() -> tuple[list[int], int]:
            gone = [index for index in dict.fromkeys(indexes) if session.discard(index)]
            # How many are left, counted from the manifest. `frame_count` is the
            # next number to hand out, which after a deletion is larger than the
            # number of pictures -- reporting it told the operator "2 removed,
            # 4 remaining" about a session holding 2.
            return gone, len(session.all_shots())

        removed, remaining = await asyncio.to_thread(discard_all)
        return web.json_response(
            {"session": name, "removed": removed, "remaining": remaining}
        )

    async def capture_image(request: Any) -> Any:
        """One captured frame, for the page to display.

        Reached only from the local page, so the ordinary guard applies. The
        path is taken from the manifest rather than from the request, which is
        what keeps a frame index from naming a file elsewhere on the disk.
        """
        name = (request.query.get("session") or "").strip()
        if not name:
            raise ValueError("session is required")
        session = open_session(name)
        if session is None:
            raise web.HTTPNotFound(text="No such session")
        index = _non_negative_int(request.query.get("frame_index", "0"), "frame_index")
        shot = await asyncio.to_thread(session.shot_at, index)
        if shot is None:
            raise web.HTTPNotFound(text="No such frame")
        wants_boxes = request.query.get("annotated") == "1"
        relative = (shot.annotated if wants_boxes else None) or shot.image
        path = session.directory / relative
        if not path.is_file():
            raise web.HTTPNotFound(text="Frame is missing from disk")
        if request.query.get("thumb") == "1":
            # The strip shows every capture in the session at once. At full size
            # that is 460 KB and a 1920x1080 bitmap per thumbnail, so a session
            # of a hundred would have the browser decoding around 800 MB of
            # pixels to draw a row of postage stamps.
            small = await asyncio.to_thread(
                _thumbnail_of, path, session.directory / "thumbs"
            )
            if small is not None:
                path = small
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def line_status(_: Any) -> Any:
        from .line_client import load_line_config

        config = load_line_config()
        if config is None:
            return web.json_response(
                {
                    "configured": False,
                    "how": (
                        "ตั้ง TELEMETRY_OCR_LINE_TOKEN และ TELEMETRY_OCR_LINE_TO "
                        "หรือสร้าง configs/line.json"
                    ),
                }
            )
        # The destination is masked and the token is never included at all.
        return web.json_response(
            {
                "configured": True,
                "to": config.masked_to,
                # Whether images can go too, which is a separate setting and a
                # separate decision about exposing the pictures at all.
                "images": bool(config.public_base_url) or quick_tunnel,
                "shared_now": share_store.live_count(),
                "quick_tunnel": (
                    share_state["tunnel"].status() if share_state["tunnel"] else None
                ),
            }
        )

    async def line_send(request: Any) -> Any:
        """Push one captured shot to LINE as text.

        Rate limited on this side rather than only by disabling the button. The
        endpoint can be called directly, and every call spends a message from the
        account's monthly quota, so a stuck caller must not be able to drain it.
        """
        from .line_client import (
            MAX_IMAGES_PER_PUSH,
            format_capture_digest,
            load_line_config,
            push_capture,
        )

        config = load_line_config()
        if config is None:
            raise ValueError(
                "ยังไม่ได้ตั้งค่า LINE: ต้องมี TELEMETRY_OCR_LINE_TOKEN และ "
                "TELEMETRY_OCR_LINE_TO หรือไฟล์ configs/line.json"
            )

        name = (request.query.get("session") or "").strip()
        if not name:
            raise ValueError("session is required")
        # Reopened from disk when this run did not create it: sessions live in
        # memory only for the life of the process, and a restart should not make
        # an existing recording unsendable.
        session = open_session(name)
        if session is None:
            raise web.HTTPNotFound(
                text='{"error":"No such capture session"}', content_type="application/json"
            )

        # One frame or several. Several because the reason to look through a
        # session is usually to compare frames, and sending them one at a time
        # would spend one message from the monthly allowance per picture.
        wanted = [
            _non_negative_int(value, "frame_index")
            for raw in request.query.getall("frame_index", [])
            for value in [raw]
            if raw != ""
        ]
        for raw in (request.query.get("frames") or "").split(","):
            if raw.strip():
                wanted.append(_non_negative_int(raw.strip(), "frames"))
        if len(wanted) > MAX_IMAGES_PER_PUSH:
            raise ValueError(
                f"เลือกได้สูงสุด {MAX_IMAGES_PER_PUSH} รูปต่อการส่งหนึ่งครั้ง "
                "(LINE รับได้ 5 ข้อความต่อ 1 request และข้อความแรกคือ telemetry)"
            )
        if not wanted:
            # A reopened session has taken no shots this run, so fall back to the
            # last frame the manifest knows about.
            first = session.last_shot or await asyncio.to_thread(
                session.shot_at, session.frame_count - 1
            )
            shots = [first] if first is not None else []
        else:
            found = await asyncio.to_thread(
                lambda: [session.shot_at(index) for index in dict.fromkeys(wanted)]
            )
            shots = [item for item in found if item is not None]
        if not shots:
            return web.json_response(
                {"error": "ยังไม่มีเฟรมที่ capture ไว้ใน session นี้"}, status=409
            )
        shots.sort(key=lambda item: item.frame_index)
        shot = shots[0]

        now = time.monotonic()
        waited = now - float(line_state["last_sent"])
        if waited < _LINE_MIN_INTERVAL_S:
            return web.json_response(
                {
                    "error": (
                        f"ส่งถี่เกินไป รออีก {_LINE_MIN_INTERVAL_S - waited:.1f} วินาที"
                    ),
                    "retry_after_s": round(_LINE_MIN_INTERVAL_S - waited, 1),
                },
                status=429,
            )
        line_state["last_sent"] = now

        text = format_capture_digest(shots, directory=session.directory)

        # Images only when a public base URL is configured. Without one there is
        # nowhere LINE could fetch from, and the message goes as text.
        images: list[str] = []
        share_error = ""
        base_url = config.public_base_url
        if not base_url and quick_tunnel:
            # A quick tunnel is given a fresh address every time it starts, so it
            # is asked at send time rather than read from a setting written once.
            base_url = await ensure_quick_tunnel() or ""
            if not base_url:
                share_error = "quick tunnel is not available"
        if base_url:
            try:
                await ensure_share_server()
                images = await asyncio.to_thread(
                    _share_images, share_store, session, shots, base_url
                )
                # Fetch it ourselves before LINE does.
                #
                # A tunnel process that is still alive is not the same as a
                # tunnel that still resolves: a quick tunnel can be torn down
                # while cloudflared keeps running, and the server went on
                # advertising the dead hostname. LINE fetches once, gets
                # nothing, and shows a broken picture that no retry ever fixes.
                # A fresh tunnel also needs a moment to propagate before its
                # first request succeeds, which this waits out.
                if images and not await _reachable(images[0]):
                    if quick_tunnel:
                        _LOG.warning("Public URL unreachable; restarting the tunnel")
                        tunnel = share_state["tunnel"]
                        if tunnel is not None:
                            await asyncio.to_thread(tunnel.stop)
                        share_state["tunnel"] = None
                        retry_url = await ensure_quick_tunnel()
                        images = (
                            await asyncio.to_thread(
                                _share_images, share_store, session, shots, retry_url
                            )
                            if retry_url
                            else []
                        )
                        if images and not await _reachable(images[0]):
                            images = []
                    else:
                        images = []
                    if not images:
                        share_error = (
                            "ส่งข้อความอย่างเดียว: URL สาธารณะยังเข้าถึงไม่ได้"
                        )
            except Exception as exc:
                # A picture that cannot be published is not a reason to withhold
                # the telemetry, which is the part someone is waiting for.
                _LOG.exception("Could not publish a capture image")
                share_error = str(exc)

        result = await push_capture(config, text, images)
        payload = {
            **result,
            "session": name,
            # The first, kept for callers that send one frame; the list is what
            # actually went.
            "frame_index": shot.frame_index,
            "frame_indexes": [item.frame_index for item in shots],
            "text": text,
            "images": len(images),
            "share_error": share_error,
        }
        if not result["ok"]:
            # Released so a failed send is not also a lockout; the operator will
            # want to retry immediately once they have fixed the cause.
            line_state["last_sent"] = 0.0
            return web.json_response(payload, status=502)
        return web.json_response(payload)

    async def capture_status(request: Any) -> Any:
        name = (request.query.get("session") or "").strip()
        session = capture_sessions.get(name)
        if session is None:
            return web.json_response({"session": name, "frames": 0, "open": False})
        return web.json_response({**session.status(), "open": True})

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
    app.router.add_post("/api/stream/start", stream_start)
    app.router.add_post("/api/stream/stop", stream_stop)
    app.router.add_get("/api/stream/status", stream_status)
    app.router.add_get("/api/stream/telemetry", stream_telemetry)
    app.router.add_get("/api/stream/frame.jpg", stream_frame)
    app.router.add_get("/api/stream/mjpeg", stream_mjpeg)
    app.router.add_post("/api/capture/shot", capture_shot)
    app.router.add_get("/api/capture/status", capture_status)
    app.router.add_get("/api/capture/shots", capture_shots)
    app.router.add_get("/api/capture/image", capture_image)
    app.router.add_delete("/api/capture/shot", capture_discard)
    app.router.add_get("/api/line/status", line_status)
    app.router.add_post("/api/line/send", line_send)
    app.router.add_route("OPTIONS", "/api/{tail:.*}", options)
    app.router.add_static("/assets/", assets, show_index=False)
    app.on_startup.append(start_pump)
    app.on_cleanup.append(cleanup)
    return app


def run_web_app(
    config: AppConfig,
    recognizer: Recognizer,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    stream_url: str | None = None,
    share_port: int = DEFAULT_SHARE_PORT,
    quick_tunnel: bool = False,
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
        config,
        recognizer,
        stream_url=stream_url,
        share_port=share_port,
        quick_tunnel=quick_tunnel,
        allowed_origins=allowed_origins,
        allowed_hosts=(host,),
    )
    url = f"http://{host}:{port}"
    if open_browser:

        async def launch_browser(_: Any) -> None:
            asyncio.get_running_loop().call_later(0.5, webbrowser.open, url)

        app.on_startup.append(launch_browser)
    print(f"Telemetry OCR web UI: {url}")
    web.run_app(app, host=host, port=port, print=None)


async def _reachable(
    url: str, attempts: int = 8, delay_s: float = 4.0, first_delay_s: float = 5.0
) -> bool:
    """Whether that URL actually returns the image, from outside this process.

    Waits before the first attempt rather than asking immediately. A hostname
    that has just been allocated does not resolve yet, and a failed lookup is
    negatively cached by the resolver -- so asking too early does not merely
    fail, it poisons the retries that follow.
    """
    await asyncio.sleep(first_delay_s)
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp ships with the web extra
        return False

    timeout = aiohttp.ClientTimeout(total=10)
    for attempt in range(attempts):
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(url) as response,
            ):
                if response.status == 200:
                    return True
                _LOG.info("Public URL answered %s (attempt %d)", response.status, attempt + 1)
        except Exception as exc:
            _LOG.info("Public URL not reachable yet: %s (attempt %d)", exc, attempt + 1)
        if attempt + 1 < attempts:
            await asyncio.sleep(delay_s)
    return False


# Wide enough to stay sharp on a high-DPI screen at the size the strip draws
# them, small enough that a hundred of them cost less than one full frame.
_THUMBNAIL_WIDTH = 320


def _thumbnail_of(path: Path, cache_dir: Path) -> Path | None:
    """A small copy of a captured frame, made once and cached.

    Written to disk rather than resized per request because the strip asks for
    every thumbnail again on each refresh, and a capture never changes after it
    is written -- so the second request onwards is a plain file read.

    Kept in its own directory rather than beside the frame. `frames/` is the
    recording that `extract` reads, and the next frame index is worked out by
    scanning it: a `00000007.thumb.jpg` sorts after `00000007.jpg`, which would
    have made the scan fall back to counting files and start overwriting the
    session from an index it had already used.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV ships with the app
        return None
    thumb = cache_dir / path.name
    if thumb.is_file() and thumb.stat().st_mtime >= path.stat().st_mtime:
        return thumb
    cache_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    if width <= _THUMBNAIL_WIDTH:
        return path
    scale = _THUMBNAIL_WIDTH / width
    small = cv2.resize(
        image, (_THUMBNAIL_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    # Written under a temporary name and moved into place: the strip reads these
    # while captures are still being taken, and a half-written file is a broken
    # image in the page. Encoded first rather than imwrite'd to the temporary
    # name, because cv2 picks the format from the extension and ".part" is not
    # one it knows.
    staging = thumb.with_name(thumb.name + ".part")
    staging.write_bytes(buffer.tobytes())
    staging.replace(thumb)
    return thumb


def _shot_stage(shot: Any) -> str:
    """How far a capture got: "ready", "no_model" or "no_result".

    Kept as three words rather than a boolean because they are fixed in three
    different places. "no_model" means best.pt was never loaded, "no_result"
    means it was asked and did not answer in time, and "ready" means it ran --
    including when it ran and found nothing, which is a real answer and belongs
    with the successes.
    """
    status = getattr(getattr(shot, "detections", None), "status", "")
    if status in {"no_model", "no_result"}:
        return status
    return "ready"


def _share_images(store: Any, session: Any, shots: Any, base_url: str) -> list[str]:
    """Publish the pictures for these shots and return their URLs, in order.

    Only the annotated copy of each. Sending the original too put two pictures
    in the chat that are identical apart from the boxes, and the reader could
    not tell which was which without opening both. The unmarked frame stays on
    disk, and the message names its path.
    """
    from .public_share import public_url

    urls = []
    for shot in shots:
        name = shot.annotated or shot.image
        if name:
            urls.append(public_url(base_url, store.mint(session.directory / name)))
    return urls


def _encode_for_detection(frame: Any) -> bytes | None:
    """The captured frame as JPEG, at the size the detector will work on."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV ships with the app
        return None
    height, width = frame.shape[:2]
    scale = min(1.0, _CAPTURE_DETECT_IMAGE_SIZE / max(width, height))
    resized = (
        frame
        if scale >= 1.0
        else cv2.resize(frame, (round(width * scale), round(height * scale)))
    )
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return buffer.tobytes() if ok else None


def _boxes_of(result: dict[str, Any] | None) -> tuple[list[Any], tuple[int, int] | None]:
    """Boxes and the frame size they are expressed in, for drawing."""
    if not result:
        return [], None
    width = int(result.get("width") or 0)
    height = int(result.get("height") or 0)
    return list(result.get("detections") or []), ((width, height) if width else None)


def _as_record(record: Any) -> dict[str, Any]:
    """A TelemetryRecord as the plain dict the verifier compares against."""
    from dataclasses import asdict as _asdict

    return _asdict(record)


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
