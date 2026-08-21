"""Hand one captured image to LINE, briefly, and to nothing else.

LINE cannot be sent an image. It can only be given an HTTPS URL that its own
servers fetch, which means a picture on this machine has to become reachable
from the public internet for as long as that fetch takes.

Everything here exists to make that window as small and as narrow as it can be:

- one token per image, from the system random source, long enough not to be
  guessed;
- the token maps to one file that already exists inside the sessions directory,
  and nothing else on the disk is addressable;
- it expires, so a URL that leaks or is logged somewhere stops working on its
  own rather than staying live because nobody remembered to revoke it;
- it is read-only, and the path it lives under is the only one the tunnel is
  configured to route.

None of that changes the fact that LINE fetches the image and keeps its own
copy. Once a picture has been sent it is out, and no expiry here reaches into
LINE's cache. That is a property of sending an image to a chat service at all,
and the operator should decide with it in mind.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# Long enough that guessing is hopeless, short enough for a URL. 32 bytes of
# urlsafe base64 is 43 characters.
_TOKEN_BYTES = 32

# How long a minted URL stays valid. LINE fetches within seconds of the push, so
# this is generous already; the point is that it is finite.
DEFAULT_TTL_S = 600.0

# A ceiling on how many live tokens are kept, so a server left running for weeks
# cannot accumulate them without bound. Oldest first, since they expire anyway.
_MAX_TOKENS = 256


@dataclass(frozen=True, slots=True)
class SharedFile:
    path: Path
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at


class ShareStore:
    """Short-lived tokens for files under one root directory."""

    def __init__(self, root: Path, ttl_s: float = DEFAULT_TTL_S) -> None:
        self.root = Path(root).resolve()
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._tokens: dict[str, SharedFile] = {}

    def mint(self, path: Path) -> str:
        """Publish one file. Raises if it is not inside the root.

        The containment check is the important half. The caller passes a path
        assembled from a session name and a manifest entry, and neither is
        beyond reach of something malformed; resolving both and comparing is
        what stops a token ever addressing a file outside the recordings.
        """
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Refusing to share a file outside {self.root}: {resolved}")

        token = secrets.token_urlsafe(_TOKEN_BYTES)
        with self._lock:
            self._prune_locked()
            self._tokens[token] = SharedFile(resolved, time.monotonic() + self.ttl_s)
        return token

    def resolve(self, token: str) -> Path | None:
        """The file this token names, or None when it is unknown or expired."""
        with self._lock:
            entry = self._tokens.get(token)
            if entry is None:
                return None
            if entry.expired:
                # Dropped on the way past rather than by a timer: a token nobody
                # asks for costs nothing, and this keeps the store honest about
                # what it would actually serve.
                del self._tokens[token]
                return None
            return entry.path

    def revoke(self, token: str) -> None:
        with self._lock:
            self._tokens.pop(token, None)

    def live_count(self) -> int:
        with self._lock:
            self._prune_locked()
            return len(self._tokens)

    def _prune_locked(self) -> None:
        for token in [key for key, item in self._tokens.items() if item.expired]:
            del self._tokens[token]
        while len(self._tokens) >= _MAX_TOKENS:
            self._tokens.pop(next(iter(self._tokens)))


def public_url(base_url: str, token: str) -> str:
    """Where LINE will come looking.

    HTTPS is required rather than preferred: LINE rejects an image message whose
    URL is not, so a base configured as http:// would fail at the push with a
    message about the URL rather than about the setting that produced it.
    """
    base = base_url.strip().rstrip("/")
    if not base.lower().startswith("https://"):
        raise ValueError(
            f"The public base URL must be https:// for LINE to accept it: {base_url!r}"
        )
    return f"{base}{PUBLIC_PREFIX}/{token}"


# The one path the tunnel is meant to route. Kept here so the server, the config
# it writes and the tests all agree on it.
PUBLIC_PREFIX = "/public/img"

# The share server listens here, and the tunnel points at this rather than at the
# application. See ShareServer for why that is not the same thing as trusting the
# application's own guard.
DEFAULT_SHARE_PORT = 8766


class ShareServer:
    """A second HTTP server whose entire vocabulary is one image route.

    The tunnel points here instead of at the application, and the reason is
    quick tunnels: `cloudflared tunnel --url` takes no ingress rules, so whatever
    port it is given is forwarded in full. Aimed at the application that would
    put the whole API on the internet with one middleware standing in front of
    it -- and a middleware is a thing that can have a bug.

    Aimed here, a flaw in that middleware reaches a process that can do nothing
    except return a file for a valid token. There is no session endpoint to
    reach, no model upload, no stream control, because none of them exist in
    this app at all.
    """

    def __init__(self, store: ShareStore, port: int = DEFAULT_SHARE_PORT) -> None:
        self.store = store
        self.port = port
        self._runner: Any = None

    def build_app(self) -> Any:
        """The one-route application, separately so tests can drive it.

        Exposed rather than inlined into start() because binding a fixed port in
        a test suite is a way to fail on a machine where something else already
        has it.
        """
        from aiohttp import web

        async def serve(request: Any) -> Any:
            token = request.match_info.get("token", "")
            path = self.store.resolve(token)
            if path is None:
                # One answer for unknown and expired alike: telling them apart
                # would let someone probing learn which tokens had been real.
                _LOG.info("Share request for an unknown or expired token")
                return web.Response(status=404, text="Not found")
            return web.FileResponse(
                path,
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "Content-Disposition": "inline",
                },
            )

        app = web.Application()
        app.router.add_get(f"{PUBLIC_PREFIX}/{{token}}", serve)
        return app

    async def start(self) -> None:
        from aiohttp import web

        self._runner = web.AppRunner(self.build_app(), access_log=None)
        await self._runner.setup()
        # Loopback only. The tunnel reaches it from this machine; nothing else
        # on the network can, even before the tunnel is considered.
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        _LOG.info("Share server listening on 127.0.0.1:%d%s", self.port, PUBLIC_PREFIX)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
