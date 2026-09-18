from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import webbrowser
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from . import __version__
from .config import AppConfig, clone_config
from .errors import ConfigError, DependencyError

# line_client is imported lazily everywhere else in this file. These three are
# the exception because the sharing helpers below are module-level functions run
# on a worker thread, and because they are pure stdlib -- unlike the rest of that
# module's surface, which is only ever reached from a request handler.
from .line_client import ImageMessage, needs_smaller_preview, oversized_for_line
from .ocr import Recognizer
from .pipeline import FramePacket, TelemetryPipeline
from .public_share import DEFAULT_SHARE_PORT, ShareServer, ShareStore
from .render_export import ExportBusy, ExportManager, ExportUnavailable
from .render_job import (
    LEGACY_FORMAT,
    LEGACY_VIDEO_FILE,
    RenderBusy,
    RenderManager,
    RenderSettings,
    RenderUnavailable,
    is_current,
    made_from,
    read_meta,
    render_format,
    render_id_for,
    with_track_ids,
)
from .video_library import SourceFrames, VideoInfo, VideoLibrary


@dataclass(slots=True)
class BrowserSession:
    pipeline: TelemetryPipeline
    last_access: float


_LOG = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Where the server's own log goes. Beside the startup script's log rather than
# in it: that file is the wrapper saying which build it chose, and this one is
# the server saying what happened afterwards.
DEFAULT_LOG_DIR = Path("output/logs")
LOG_FILENAME = "ocr-web.log"
# A day of INFO from a running server is a few hundred KB, so five files reach
# back across several restarts while bounding the directory at 10 MB.
_LOG_MAX_BYTES = 2 * 1024**2
_LOG_BACKUPS = 4

# A request carrying this header can never be a CORS-safelisted "simple" request,
# so the browser must send a preflight the guard below can reject. Without it a
# form POST from any site the user has open could start a GPU render through
# /api/renders or change which checkpoints /api/models registers.
GUARD_HEADER = "X-Telemetry-OCR"

_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

SESSION_TTL_SECONDS = 3600.0

# Smallest gap between two LINE pushes. Every push spends one message from the
# account's monthly quota, so a wedged client or a held button must not be able
# to drain it; three seconds is far below anything a person does deliberately.
_LINE_MIN_INTERVAL_S = 3.0
# How long a quota reading is treated as current. The number changes only when
# this server pushes, and a send updates the cache itself, so this is really the
# window in which a send from somewhere else goes unnoticed.
_LINE_QUOTA_TTL_S = 300.0


def create_app(
    config: AppConfig,
    recognizer: Recognizer,
    *,
    static_dir: str | Path | None = None,
    max_sessions: int = 4,
    output_dir: str | Path = "output",
    share_port: int = DEFAULT_SHARE_PORT,
    quick_tunnel: bool = False,
    allowed_origins: Iterable[str] = (),
    allowed_hosts: Iterable[str] = (),
    model_registry: Any = None,
    video_library: VideoLibrary | None = None,
    render_manager: RenderManager | None = None,
    export_manager: ExportManager | None = None,
) -> Any:
    try:
        from aiohttp import web

        from .file_response import LargeFileResponse
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
        except (ConfigError, DependencyError, TypeError, ValueError) as exc:
            _LOG.warning("%s %s rejected: %s", request.method, request.path, exc)
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            # The client only receives the exception type, so without this the
            # reason for a 500 is lost entirely.
            _LOG.exception("%s %s failed", request.method, request.path)
            return web.json_response(
                {"error": f"OCR request failed: {type(exc).__name__}"}, status=500
            )

    output_dir = Path(output_dir)
    # The largest body any endpoint takes is a captured full frame: a JPEG of up
    # to 4096x2160 at q95. This was 256 MB while it also had to admit an uploaded
    # checkpoint. Nothing takes an upload now -- a capture reads its frame from
    # the video file -- and a smaller cap is less a stray request can make this
    # process buffer.
    request_body_limit = 32 * 1024**2
    sessions_key = web.AppKey("sessions", dict)
    ocr_lock_key = web.AppKey("ocr_lock", asyncio.Lock)
    max_sessions_key = web.AppKey("max_sessions", int)
    # Open capture sessions, by name. Held for the life of the server so that
    # pressing the button repeatedly appends to one recording rather than
    # reopening a directory and re-reading its numbering every time.
    capture_sessions: dict[str, Any] = {}
    # When the last LINE push went out, so the rate limit survives across
    # requests without being stored on the frozen Application.
    line_state: dict[str, float] = {"last_sent": 0.0}
    # The month's allowance, cached. Two HTTP calls to LINE, and the number
    # moves only when this server sends something, so asking on every status
    # poll would spend more requests on watching the quota than on using it.
    line_quota_state: dict[str, Any] = {"at": 0.0, "quota": None}

    async def line_quota(config: Any, *, force: bool = False) -> Any:
        from .line_client import fetch_quota

        cached = line_quota_state["quota"]
        fresh = time.monotonic() - float(line_quota_state["at"]) < _LINE_QUOTA_TTL_S
        # A cached failure is not worth keeping: whatever was wrong may have been
        # fixed, and the retry costs one request.
        if cached is not None and fresh and not force and not cached.error:
            return cached
        quota = await fetch_quota(config)
        line_quota_state["quota"] = quota
        line_quota_state["at"] = time.monotonic()
        return quota

    # Tokens for images LINE is allowed to fetch. Created only when a send needs
    # one, and expiring on their own.
    share_store = ShareStore(Path(output_dir) / "sessions")
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
    library = video_library if video_library is not None else VideoLibrary(None)
    renders = (
        render_manager if render_manager is not None else RenderManager(output_dir / "renders")
    )
    exports = (
        export_manager
        if export_manager is not None
        else ExportManager(renders, output_dir / "exports")
    )
    if renders.ffmpeg:
        _LOG.info("Rendering encodes with %s", renders.ffmpeg)
    else:
        _LOG.warning("ffmpeg not found; videos cannot be rendered through a model")
    # Captures read the original frame from the file on the server. What the
    # page shows may be the rendered video, whose boxes cover the OSD digits.
    source_frames = SourceFrames()
    app = web.Application(
        client_max_size=request_body_limit,
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

    # -- models ------------------------------------------------------------------
    #
    # What may be loaded is decided on this machine: a model is added by the id
    # of a checkpoint the registry itself found under the repository, removed
    # without its file being deleted, and never named by a path in a request.

    def require_registry() -> Any:
        if model_registry is None:
            raise web.HTTPServiceUnavailable(
                text=json.dumps(
                    {"error": "ทะเบียนโมเดลใช้งานไม่ได้ (อ่าน configs/paths.yaml ไม่สำเร็จ)"}
                ),
                content_type="application/json",
            )
        return model_registry

    def model_payload(entry: Any) -> dict[str, Any]:
        return {
            "name": entry.name,
            "label": entry.label,
            "available": entry.available,
            "size_bytes": entry.size_bytes,
            "default": entry.default,
        }

    async def list_models(_: Any) -> Any:
        if model_registry is None:
            return web.json_response(
                {"models": [], "default": None, "error": "ทะเบียนโมเดลใช้งานไม่ได้"}
            )
        entries = await asyncio.to_thread(model_registry.entries)
        return web.json_response(
            {
                "models": [model_payload(entry) for entry in entries],
                "default": next((entry.name for entry in entries if entry.default), None),
            }
        )

    async def model_candidates(_: Any) -> Any:
        registry = require_registry()
        found = await asyncio.to_thread(registry.candidates)
        return web.json_response(
            {
                "candidates": [
                    {
                        "id": item.id,
                        "label": item.label,
                        "size_bytes": item.size_bytes,
                        "modified": item.modified,
                        "suggested_name": item.suggested_name,
                    }
                    for item in found
                ]
            }
        )

    async def add_model(request: Any) -> Any:
        registry = require_registry()
        body = await _json_object(request)
        entry = await asyncio.to_thread(
            registry.add, str(body.get("candidate_id", "")), str(body.get("name", ""))
        )
        _LOG.info("Registered model %s -> %s", entry.name, entry.label)
        return web.json_response(model_payload(entry), status=201)

    async def remove_model(request: Any) -> Any:
        registry = require_registry()
        name = request.match_info["name"]
        await asyncio.to_thread(registry.remove, name)
        _LOG.info("Removed model %s from the registry; its file is kept", name)
        return await list_models(request)

    async def set_default_model(request: Any) -> Any:
        registry = require_registry()
        body = await _json_object(request)
        await asyncio.to_thread(registry.set_default, str(body.get("name", "")))
        return await list_models(request)

    # -- videos -------------------------------------------------------------------

    def video_payload(video: VideoInfo) -> dict[str, Any]:
        return {
            "id": video.id,
            "name": video.name,
            "size_bytes": video.size_bytes,
            "fps": video.fps,
            "frames": video.frames,
            "width": video.width,
            "height": video.height,
            "duration_ms": video.duration_ms,
        }

    def find_video(video_id: str) -> VideoInfo:
        wanted = video_id.strip()
        video = library.get(wanted) if wanted else None
        if video is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": "ไม่พบวิดีโอนี้ในโฟลเดอร์วิดีโอ"}),
                content_type="application/json",
            )
        return video

    async def list_videos(_: Any) -> Any:
        videos = await asyncio.to_thread(library.videos)
        return web.json_response(
            {
                "directory": str(library.directory) if library.directory else None,
                "videos": [video_payload(video) for video in videos],
            }
        )

    async def video_source(request: Any) -> Any:
        video = await asyncio.to_thread(find_video, request.query.get("video", ""))
        # FileResponse answers Range requests, which is what lets the page seek
        # a 400 MB file without downloading it first. Large, because a long
        # flight is over the 2 GiB Windows sends in one piece.
        return LargeFileResponse(video.path, headers={"Cache-Control": "no-cache"})

    # -- renders ------------------------------------------------------------------

    def render_payload(
        meta: dict[str, Any], video: VideoInfo | None, entry: Any
    ) -> dict[str, Any]:
        model_path = entry.path if entry is not None and entry.available else None
        return {
            "render_id": meta.get("render_id"),
            "video": meta.get("video"),
            "model": meta.get("model"),
            "model_label": meta.get("model_label"),
            "settings": meta.get("settings") or {},
            "fps": meta.get("fps"),
            "source_fps": meta.get("source_fps"),
            "stride": meta.get("stride", 1),
            "frames_written": meta.get("frames_written"),
            "duration_ms": meta.get("duration_ms"),
            "elapsed_s": meta.get("elapsed_s"),
            "created_at": meta.get("created_at"),
            "format": render_format(meta),
            # "complete", "partial" (stopped, can continue) or "running".
            "status": meta.get("status"),
            "processed_ms": meta.get("processed_ms"),
            "size_bytes": meta.get("size_bytes"),
            # False once the video or the checkpoint has been replaced since: a
            # finished render is then out of date, and a partial one starts over.
            "current": made_from(meta, video.path if video else None, model_path),
        }

    def collect_renders(video_id: str | None) -> list[dict[str, Any]]:
        entries = {
            entry.name: entry for entry in (model_registry.entries() if model_registry else [])
        }
        videos = {video.id: video for video in library.videos()}
        return [
            render_payload(
                meta, videos.get(str(meta.get("video"))), entries.get(str(meta.get("model")))
            )
            for meta in renders.renders(video_id)
        ]

    async def list_renders(request: Any) -> Any:
        video_id = (request.query.get("video") or "").strip() or None
        listed = await asyncio.to_thread(collect_renders, video_id)
        return web.json_response(
            {
                "renders": listed,
                "job": await asyncio.to_thread(renders.status),
                "ffmpeg": bool(renders.ffmpeg),
            }
        )

    async def start_render(request: Any) -> Any:
        registry = require_registry()
        body = await _json_object(request)
        video = await asyncio.to_thread(find_video, str(body.get("video", "")))
        name = str(body.get("model", "")).strip()
        entry = await asyncio.to_thread(registry.get, name)
        if entry is None:
            raise ValueError(f"ไม่มีโมเดลชื่อ {name!r} ในทะเบียน")
        if not entry.available:
            raise ValueError(f"ไม่พบไฟล์ของโมเดล {name}: {entry.label}")
        settings = RenderSettings(
            confidence=float(body.get("confidence", 0.45)),
            image_size=int(body.get("image_size", 640)),
            stride=int(body.get("stride", 1)),
        )
        settings.validate()
        render_id = render_id_for(video.id, entry.name, settings)
        # Without force, a partial render of the same files carries on from where
        # it stopped; with it, whatever is there is made again from the start.
        fresh = bool(body.get("force"))
        if not fresh:
            directory = renders.directory(render_id)
            meta = read_meta(directory)
            if (
                meta is not None
                and is_current(meta, video.path, entry.path)
                and (
                    render_format(meta) != LEGACY_FORMAT
                    or (directory / LEGACY_VIDEO_FILE).is_file()
                )
            ):
                # Already made from these exact files: play it rather than spend
                # half an hour of GPU making it again.
                return web.json_response(
                    {"existing": render_payload(meta, video, entry), "job": renders.status()}
                )
        try:
            job = await asyncio.to_thread(
                renders.start,
                render_id=render_id,
                video_id=video.id,
                video_path=video.path,
                model_name=entry.name,
                model_label=entry.label,
                model_path=entry.path,
                settings=settings,
                fresh=fresh,
            )
        except RenderBusy as exc:
            return web.json_response({"error": str(exc), "job": renders.status()}, status=409)
        except RenderUnavailable as exc:
            return web.json_response({"error": str(exc)}, status=503)
        return web.json_response({"job": job}, status=202)

    async def render_status(_: Any) -> Any:
        return web.json_response(await asyncio.to_thread(renders.status))

    async def cancel_render(_: Any) -> Any:
        cancelled = await asyncio.to_thread(renders.cancel)
        return web.json_response({"cancelled": cancelled, "job": renders.status()})

    def render_not_found() -> web.HTTPNotFound:
        return web.HTTPNotFound(
            text=json.dumps({"error": "ไม่พบผลการประมวลผลนี้"}),
            content_type="application/json",
        )

    def legacy_video(render_id: str) -> Path:
        directory = renders.directory(render_id.strip())
        meta = read_meta(directory)
        path = directory / LEGACY_VIDEO_FILE
        if meta is None or meta.get("status") != "complete" or not path.is_file():
            raise render_not_found()
        return path

    async def render_video(request: Any) -> Any:
        """A render made before segments: one finished MP4, played as a file."""
        path = await asyncio.to_thread(legacy_video, request.query.get("render", ""))
        return LargeFileResponse(path, headers={"Cache-Control": "no-cache"})

    async def render_stream(request: Any) -> Any:
        """A render's finished segments, which the page polls while the render grows."""
        render_id = (request.query.get("render") or "").strip()
        index = await asyncio.to_thread(renders.stream, render_id)
        if index is None:
            raise render_not_found()
        return web.json_response(index, headers={"Cache-Control": "no-store"})

    async def render_segment(request: Any) -> Any:
        """One init or media segment. Only names a render writes are served."""
        path = await asyncio.to_thread(
            renders.segment_path,
            (request.query.get("render") or "").strip(),
            request.query.get("run", ""),
            request.query.get("file", ""),
        )
        if path is None:
            raise render_not_found()
        # The same names come back when a render is made again, so never cached.
        return web.FileResponse(
            path, headers={"Cache-Control": "no-cache", "Content-Type": "video/mp4"}
        )

    async def start_export(request: Any) -> Any:
        """Join a render into one MP4 to download. The file is made, then fetched.

        Two steps rather than one long request because the 720p version of a
        25-minute render takes over a minute to encode, and a request that waits
        that long before its first byte is one a tunnel or a proxy gives up on.
        """
        body = await _json_object(request)
        render_id = str(body.get("render", "")).strip()
        quality = str(body.get("quality", "full")).strip()
        try:
            job = await asyncio.to_thread(exports.start, render_id, quality)
        except ExportBusy as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except ExportUnavailable as exc:
            return web.json_response({"error": str(exc)}, status=404)
        return web.json_response({"export": job}, status=202)

    async def export_status(request: Any) -> Any:
        status = exports.status((request.query.get("id") or "").strip())
        if status is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": "ไม่พบไฟล์ที่เตรียมไว้"}), content_type="application/json"
            )
        return web.json_response(status, headers={"Cache-Control": "no-store"})

    async def export_file(request: Any) -> Any:
        """Hand the finished MP4 over, then remove it: the render is the copy kept."""
        export_id = (request.query.get("id") or "").strip()
        ready = exports.ready_file(export_id)
        if ready is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": "ไฟล์ยังไม่พร้อม หรือดาวน์โหลดไปแล้ว"}),
                content_type="application/json",
            )
        path, filename = ready
        response = web.StreamResponse(
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(path.stat().st_size),
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)
        completed = False
        try:
            with path.open("rb") as stream:
                while chunk := await asyncio.to_thread(stream.read, 1024 * 1024):
                    await response.write(chunk)
            await response.write_eof()
            completed = True
        finally:
            # Kept for another try when the download broke off partway.
            if completed:
                await asyncio.to_thread(exports.finished, export_id)
        return response

    async def delete_render(request: Any) -> Any:
        render_id = (request.query.get("render") or "").strip()
        try:
            await asyncio.to_thread(renders.delete, render_id)
        except RenderBusy as exc:
            return web.json_response({"error": str(exc)}, status=409)
        _LOG.info("Deleted render %s", render_id)
        return web.json_response({"deleted": render_id})

    async def options(_: Any) -> Any:
        return web.Response(status=204)

    async def cleanup(_: Any) -> None:
        # First: it holds a child process, and leaving cloudflared running after
        # the server it fronts has gone would leave an address answering 502.
        tunnel = share_state["tunnel"]
        if tunnel is not None:
            await asyncio.to_thread(tunnel.stop)
        server = share_state["server"]
        if server is not None:
            await server.stop()
        # A render holds the GPU and an ffmpeg process. It is stopped before the
        # server goes rather than being left running with nothing to report to.
        await asyncio.to_thread(renders.shutdown)
        await asyncio.to_thread(exports.shutdown)
        await asyncio.to_thread(source_frames.close)
        # The extra recognizers a capture votes with. Built lazily on the first
        # capture, so a server that never captured has none to release.
        from .capture_service import close_ensemble

        await asyncio.to_thread(close_ensemble)

    def summarise_boxes(boxes: list[dict[str, Any]] | None) -> Any:
        """Counts and confidences for the boxes a render drew on this frame."""
        from collections import Counter

        from .capture_service import DetectionSummary

        if boxes is None:
            # The render has no line for this frame, which a complete render
            # should never do; said rather than reported as "found nothing".
            return DetectionSummary("no_result")
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
            # Zero, and meant literally: these boxes were drawn on this frame.
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

    def capture_session_for(name: str, source: str) -> Any:
        from .capture_service import CaptureSession

        existing = capture_sessions.get(name)
        if existing is not None:
            return existing
        session = CaptureSession(
            output_dir / "sessions",
            name,
            source,
            [field.field_id for field in web_config.fields],
        )
        capture_sessions[name] = session
        return session

    async def capture_shot(request: Any) -> Any:
        """Keep this frame and what it says.

        The frame is read here, from the original file, rather than posted by the
        page. What the page shows may be the rendered video, whose boxes sit over
        the OSD digits OCR reads; the original never has anything drawn on it.
        The boxes come from the render's own record of that frame, so the marked
        copy matches the video on screen without asking the model again -- also
        while the render is still being made, for the frames it has got to.
        """
        from .capture_service import DetectionSummary

        name = (request.query.get("session") or "").strip()
        if not name:
            raise ValueError("session is required")
        source_time_ms = float(request.query.get("source_time_ms", "0"))
        if not 0 <= source_time_ms < 86_400_000:
            raise ValueError("source_time_ms is outside the supported range")

        browser_session = app[sessions_key].get(request.query.get("session_id", ""))
        if browser_session is None:
            raise web.HTTPNotFound(
                text='{"error":"Session not found"}', content_type="application/json"
            )
        browser_session.last_access = time.monotonic()

        video = await asyncio.to_thread(find_video, request.query.get("video", ""))
        render_id = (request.query.get("render") or "").strip()
        meta: dict[str, Any] | None = None
        if render_id:
            meta = read_meta(renders.directory(render_id))
            if (
                meta is None
                or meta.get("status") not in {"complete", "partial", "running"}
                or meta.get("video") != video.id
            ):
                raise ValueError("ผลการประมวลผลนี้ไม่ใช่ของวิดีโอที่เลือก")
        fps = float((meta or {}).get("source_fps") or video.fps)
        stride = int((meta or {}).get("stride") or 1)
        # The frame on screen. A render made with a stride shows every
        # stride-th source frame, so the time is rounded to one of those.
        frame_index = round(source_time_ms * fps / stride / 1000.0) * stride
        if video.frames:
            frame_index = min(frame_index, (video.frames - 1) // stride * stride)
        frame = await asyncio.to_thread(source_frames.read, video.path, frame_index)
        frame_time_ms = round(frame_index * 1000.0 / fps, 3)

        packet = FramePacket(frame, frame_index, frame_time_ms, time.perf_counter())
        async with app[ocr_lock_key]:
            recognised, _ = await asyncio.to_thread(
                browser_session.pipeline.process_frame, packet
            )
        session = capture_session_for(name, video.name)
        boxes: list[dict[str, Any]] | None = None
        box_source: tuple[int, int] | None = None
        if meta is None:
            summary = DetectionSummary("no_render")
        else:
            detections = await asyncio.to_thread(renders.detections_at, render_id, frame_index)
            if detections is None and meta.get("status") != "complete":
                # A render still being made, or stopped, that has not got this far.
                summary = DetectionSummary("not_processed")
            else:
                summary = summarise_boxes(detections)
            if detections:
                boxes = with_track_ids(detections)
                box_source = (int(frame.shape[1]), int(frame.shape[0]))
        shot = await asyncio.to_thread(
            session.save,
            frame,
            frame_time_ms,
            recognised,
            summary,
            shot_verifier([_as_record(item) for item in recognised]),
            boxes,
            box_source,
        )
        session.record_verification(shot)
        return web.json_response(asdict(shot))

    def open_session(name: str) -> Any:
        """The session by that name, reopened from disk if this run did not make it."""
        session = capture_sessions.get(name)
        if session is None and (output_dir / "sessions" / name).is_dir():
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
                # What is left of the month. Cached, so a reload does not spend
                # two requests on LINE every time the page opens.
                "quota": (await line_quota(config)).as_dict(),
            }
        )

    # How many decimals of arc-second each coordinate carries, read once from the
    # ROI configuration. Built out here rather than inside line_send because that
    # function binds `config` to the LINE credentials, not to this one.
    dms_decimals_by_field = {
        item.field_id: item.dms_decimal_places
        for item in config.fields
        if item.data_type == "dms" and item.dms_decimal_places is not None
    }

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

        # Refused here rather than at LINE. The push would come back 429 having
        # spent nothing, but the operator would have taken a shot and pressed
        # send before finding out, and LINE's own wording does not say when the
        # allowance comes back.
        quota = await line_quota(config)
        if quota.exhausted:
            return web.json_response(
                {
                    "error": (
                        f"โควต้าเดือนนี้เต็มแล้ว ({quota.used}/{quota.limit} ข้อความ) "
                        "รอเดือนถัดไปหรืออัปเกรดแพ็กเกจของ OA"
                    ),
                    "quota": quota.as_dict(),
                },
                status=429,
            )
        line_state["last_sent"] = now

        text = format_capture_digest(shots, dms_decimals=dms_decimals_by_field)

        # Images only when a public base URL is configured. Without one there is
        # nowhere LINE could fetch from, and the message goes as text.
        images: list[ImageMessage] = []
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
                images, notes = await asyncio.to_thread(
                    _share_images, share_store, session, shots, base_url
                )
                # A frame LINE would refuse is not a reason to withhold the
                # telemetry, but it is a reason to say which frame and why.
                if notes:
                    share_error = " · ".join(notes)
                # Fetch it ourselves before LINE does.
                #
                # A tunnel process that is still alive is not the same as a
                # tunnel that still resolves: a quick tunnel can be torn down
                # while cloudflared keeps running, and the server went on
                # advertising the dead hostname. LINE fetches once, gets
                # nothing, and shows a broken picture that no retry ever fixes.
                # A fresh tunnel also needs a moment to propagate before its
                # first request succeeds, which this waits out.
                if images and not await _all_reachable(images):
                    if quick_tunnel:
                        _LOG.warning("Public URL unreachable; restarting the tunnel")
                        tunnel = share_state["tunnel"]
                        if tunnel is not None:
                            await asyncio.to_thread(tunnel.stop)
                        share_state["tunnel"] = None
                        retry_url = await ensure_quick_tunnel()
                        images = []
                        if retry_url:
                            images, retry_notes = await asyncio.to_thread(
                                _share_images, share_store, session, shots, retry_url
                            )
                            # The retry's own skip reasons, which were being
                            # assigned to `notes` and then never read.
                            notes = retry_notes
                        if images and not await _all_reachable(images):
                            images = []
                    else:
                        images = []
                    if not images:
                        # Kept alongside any per-frame reasons rather than over
                        # them: a frame skipped for size and a tunnel that will
                        # not answer are different problems with different
                        # fixes, and reporting only the second hides the first.
                        reasons = [*notes, "URL สาธารณะยังเข้าถึงไม่ได้"]
                        share_error = "ส่งข้อความอย่างเดียว: " + " · ".join(reasons)
            except Exception as exc:
                # A picture that cannot be published is not a reason to withhold
                # the telemetry, which is the part someone is waiting for.
                _LOG.exception("Could not publish a capture image")
                share_error = str(exc)

        result = await push_capture(config, text, images)
        if result["ok"]:
            # One destination, so one message off the allowance however many
            # pictures went with it. Applied locally rather than re-read: the
            # number is known, and the send is already slow enough.
            quota = quota.spend()
            line_quota_state["quota"] = quota
        payload = {
            **result,
            "quota": quota.as_dict(),
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
    app.router.add_get("/api/models", list_models)
    app.router.add_get("/api/models/candidates", model_candidates)
    app.router.add_post("/api/models", add_model)
    app.router.add_post("/api/models/default", set_default_model)
    app.router.add_delete("/api/models/{name}", remove_model)
    app.router.add_get("/api/videos", list_videos)
    app.router.add_get("/api/videos/source", video_source)
    app.router.add_get("/api/renders", list_renders)
    app.router.add_post("/api/renders", start_render)
    app.router.add_delete("/api/renders", delete_render)
    app.router.add_get("/api/renders/current", render_status)
    app.router.add_delete("/api/renders/current", cancel_render)
    app.router.add_get("/api/renders/video", render_video)
    app.router.add_get("/api/renders/stream", render_stream)
    app.router.add_get("/api/renders/segment", render_segment)
    app.router.add_post("/api/renders/export", start_export)
    app.router.add_get("/api/renders/export", export_status)
    app.router.add_get("/api/renders/export/file", export_file)
    app.router.add_post("/api/capture/shot", capture_shot)
    app.router.add_get("/api/capture/status", capture_status)
    app.router.add_get("/api/capture/shots", capture_shots)
    app.router.add_get("/api/capture/image", capture_image)
    app.router.add_delete("/api/capture/shot", capture_discard)
    app.router.add_get("/api/line/status", line_status)
    app.router.add_post("/api/line/send", line_send)
    app.router.add_route("OPTIONS", "/api/{tail:.*}", options)
    app.router.add_static("/assets/", assets, show_index=False)
    app.on_cleanup.append(cleanup)
    return app


def configure_logging(log_dir: Path | str | None = DEFAULT_LOG_DIR) -> Path | None:
    """Send the server's log to a file as well as to stderr.

    The scheduled task runs this hidden, which means the process has no console
    at all and everything written to stderr is discarded. Configuring a stream
    handler and calling it a log was the bug: the only record of a tunnel that
    would not connect, a public URL that answered 404 or a LINE push that was
    refused went nowhere, so the operator saw a message arrive without its
    picture and had no way to find out why.

    Returns the file being written, or None when only stderr is available. A
    directory that cannot be created is not a reason to refuse to serve, so it
    degrades to stderr and says so rather than raising.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    path: Path | None = None
    if log_dir is not None:
        try:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            candidate = directory / LOG_FILENAME
            handlers.append(
                RotatingFileHandler(
                    candidate,
                    maxBytes=_LOG_MAX_BYTES,
                    backupCount=_LOG_BACKUPS,
                    encoding="utf-8",
                )
            )
            path = candidate
        except OSError as exc:
            # Reported through the stream handler installed below, so it is at
            # least visible to someone running the server from a terminal.
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
                handlers=[logging.StreamHandler()],
                force=True,
            )
            _LOG.warning("Could not open a log file under %s: %s", log_dir, exc)
            return None
    # force=True because a caller may already have configured the root logger,
    # and basicConfig is otherwise a no-op that would leave the file unwritten.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return path


def run_web_app(
    config: AppConfig,
    recognizer: Recognizer,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    share_port: int = DEFAULT_SHARE_PORT,
    quick_tunnel: bool = False,
    allowed_origins: Iterable[str] = (),
    log_dir: Path | str | None = DEFAULT_LOG_DIR,
) -> None:
    try:
        from aiohttp import web
    except ImportError as exc:
        raise DependencyError(
            "The web UI requires aiohttp; install the project with `pip install -e .[web]`."
        ) from exc
    log_path = configure_logging(log_dir)
    registry, video_dir = _load_registry_and_videos()
    app = create_app(
        config,
        recognizer,
        model_registry=registry,
        video_library=VideoLibrary(video_dir),
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
    # Printed rather than only logged: when the server is started by hand this
    # is where to look after it has been left running hidden, and the operator
    # should not have to read the source to find out where that is.
    print(f"Log: {log_path}" if log_path else "Log: stderr only (no log file)")
    _LOG.info("Telemetry OCR %s serving on %s (quick_tunnel=%s)", __version__, url, quick_tunnel)
    # access_log=None, and it matters now that the log is a file.
    #
    # aiohttp logs every request at INFO by default. That was invisible while
    # this went to a stderr nobody read; against a rotating file it is a flood.
    # The page long-polls /api/detection/result about once a second, so the
    # first day of this wrote 3,604 access lines and one line about anything
    # else, and rotated every 29 minutes -- a tunnel that failed an hour ago was
    # already gone. This is a loopback server with one user; who fetched what is
    # not the question the log exists to answer.
    web.run_app(app, host=host, port=port, print=None, access_log=None)


def _load_registry_and_videos() -> tuple[Any, Path | None]:
    """The model registry and the video folder, both from configs/paths.yaml.

    Through vision_lab, which parses YAML and imports no ML framework: the rule
    this process lives by is that torch never loads in it. Either failing leaves
    the page without that half rather than refusing to start the server.
    """
    try:
        from vision_lab.model_registry import ModelRegistry
        from vision_lab.paths import load_paths

        paths = load_paths()
    except Exception as exc:
        _LOG.warning("configs/paths.yaml unavailable, so no models or videos are offered: %s", exc)
        return None, None
    video_dir = paths.video_dir
    if not video_dir.is_dir():
        _LOG.warning("The video folder %s does not exist yet, so no videos are listed", video_dir)
    return ModelRegistry(paths.root), video_dir


async def _json_object(request: Any) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise ValueError("ส่งข้อมูลเป็น JSON object") from exc
    if not isinstance(body, dict):
        raise ValueError("ส่งข้อมูลเป็น JSON object")
    return body


async def _all_reachable(images: list[ImageMessage]) -> bool:
    """Every URL LINE will be given, not just the first of them.

    A send carries up to four pictures and two URLs each. Checking only
    images[0] proved the tunnel was up and assumed the rest of the tokens were,
    which they need not be -- a token evicted by the store's cap, a preview
    written after the check, a file removed between minting and pushing. LINE
    fetches each URL once, so one that fails is a picture permanently broken in
    the chat with nothing said about it.

    The first URL pays the tunnel's warm-up wait; the rest are asked once each,
    because by then the hostname resolves and a second URL that does not answer
    is a fact rather than a timing problem.
    """
    if not images:
        return True
    if not await _reachable(images[0].original):
        return False
    rest = [images[0].preview]
    for image in images[1:]:
        rest.append(image.original)
        rest.append(image.preview)
    for url in dict.fromkeys(rest):
        if url == images[0].original:
            continue
        if not await _reachable(url, attempts=1, first_delay_s=0.0):
            _LOG.warning("A shared image URL did not answer; sending text only")
            return False
    return True


async def _reachable(
    url: str, attempts: int = 4, delay_s: float = 4.0, first_delay_s: float = 5.0
) -> bool:
    """Whether that URL actually returns the image, from outside this process.

    Waits before the first attempt rather than asking immediately. A hostname
    that has just been allocated does not resolve yet, and a failed lookup is
    negatively cached by the resolver -- so asking too early does not merely
    fail, it poisons the retries that follow.

    Four attempts, not eight. At eight this was 113 seconds in the worst case,
    the caller ran it twice around a tunnel restart, and each restart waits up
    to 45 more -- so one press of Send could hold the request for five minutes
    with the button stuck on "กำลังส่ง..." and no way out. A quick tunnel that
    has not answered in the first 45 seconds is not about to; the retry with a
    fresh tunnel is the real second chance, and it still gets its own four.
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
    """How far a capture got: "ready", "no_render", "not_processed", "no_model" or "no_result".

    Kept as three words rather than a boolean because they are fixed in three
    different places. "no_model" means best.pt was never loaded, "no_result"
    means it was asked and did not answer in time, and "ready" means it ran --
    including when it ran and found nothing, which is a real answer and belongs
    with the successes.
    """
    status = getattr(getattr(shot, "detections", None), "status", "")
    # "no_render": taken from a video that had not been run through a model.
    # "not_processed": the render had not got to that frame yet.
    # "no_model" is kept for sessions recorded before renders existed.
    if status in {"no_render", "not_processed", "no_model", "no_result"}:
        return status
    return "ready"


# What a LINE preview is downscaled to when the full frame is too heavy to be
# one. Wide enough to make out the scene in a chat bubble, far enough under the
# 1 MB ceiling that a busy frame cannot creep back over it.
_LINE_PREVIEW_WIDTH = 1024
_LINE_PREVIEW_QUALITY = 80


def _line_preview_of(path: Path, cache_dir: Path) -> Path | None:
    """A copy of that frame small enough to be its own preview, or None.

    Kept beside the thumbnails rather than in `frames/`, for the reason given on
    _thumbnail_of: the next frame index is worked out by scanning that directory
    and anything else living there corrupts the count.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV ships with the app
        return None
    preview = cache_dir / f"{path.stem}.line{path.suffix}"
    if preview.is_file() and preview.stat().st_mtime >= path.stat().st_mtime:
        return preview
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    if width > _LINE_PREVIEW_WIDTH:
        scale = _LINE_PREVIEW_WIDTH / width
        image = cv2.resize(
            image,
            (_LINE_PREVIEW_WIDTH, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), _LINE_PREVIEW_QUALITY])
    if not ok:
        return None
    data = buffer.tobytes()
    if needs_smaller_preview(len(data)):
        # The width alone was not enough. Rather than guess a second time, say
        # so: a preview still over the ceiling would have the push refused, and
        # sending the text without a picture is the better failure.
        _LOG.warning("A downscaled preview of %s is still %d bytes", path.name, len(data))
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = preview.with_name(preview.name + ".part")
    staging.write_bytes(data)
    staging.replace(preview)
    return preview


def _share_images(
    store: Any, session: Any, shots: Any, base_url: str
) -> tuple[list[ImageMessage], list[str]]:
    """Publish the pictures for these shots, and say what could not be sent.

    Only the annotated copy of each. Sending the original too put two pictures
    in the chat that are identical apart from the boxes, and the reader could
    not tell which was which without opening both. The unmarked frame stays on
    disk, and the message names its path.

    A frame usually stands in for its own preview. It is re-encoded only when it
    is over LINE's preview ceiling, which is a tenth of the ceiling on the
    picture itself and the one a full-resolution capture actually approaches.
    """
    from .public_share import public_url

    images: list[ImageMessage] = []
    notes: list[str] = []
    for shot in shots:
        name = shot.annotated or shot.image
        if not name:
            continue
        path = session.directory / name
        try:
            size = path.stat().st_size
        except OSError as exc:
            notes.append(f"{name}: {exc}")
            continue

        too_big = oversized_for_line(size)
        if too_big:
            notes.append(f"{name}: {too_big}")
            continue

        original = public_url(base_url, store.mint(path))
        if not needs_smaller_preview(size):
            images.append(ImageMessage.same(original))
            continue

        preview_path = _line_preview_of(path, session.directory / "thumbs")
        if preview_path is None:
            notes.append(f"{name}: ย่อรูปให้เล็กพอสำหรับ preview ของ LINE ไม่ได้")
            continue
        images.append(
            ImageMessage(original, public_url(base_url, store.mint(preview_path)))
        )
    return images, notes


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


def _non_negative_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


