"""Run a Cloudflare quick tunnel and know what address it was given.

A quick tunnel needs no account, no domain and no login, which is what makes it
usable here. The price is that its hostname is assigned fresh every time it
starts, so nothing can be written down in advance: the address has to be read
back out of cloudflared as it comes up, and re-read whenever it is restarted.

That is the whole reason this is a class in the server rather than a line in a
setup script. A script could start the tunnel, but it could not tell the running
server what address to give LINE half an hour later after the tunnel dropped and
came back with a different one.

It is pointed at the share server, never at the application. A quick tunnel
takes no ingress rules -- whatever port it is given is forwarded in full -- so
aiming it at the API would put all of it on the internet behind a single
middleware.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

_LOG = logging.getLogger(__name__)

# cloudflared prints the assigned address inside a box of plus signs, on its own
# line, to stderr. Matching the URL itself rather than the surrounding decoration
# keeps this working when the box changes.
_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

# cloudflared prints the address as soon as it has been allocated, several
# seconds before the tunnel is actually carrying traffic -- its own banner says
# "it may take some time to be reachable". Handing that address to LINE in the
# gap produced image messages pointing at a hostname that did not resolve yet,
# and LINE fetches once. This line is the real readiness signal.
_READY_PATTERN = re.compile(r"Registered tunnel connection")

# Where cloudflared usually lands on Windows, checked before PATH so a machine
# with it installed but not on PATH still works.
_KNOWN_PATHS = (
    r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    r"C:\Program Files\cloudflared\cloudflared.exe",
)

_STARTUP_TIMEOUT_S = 45.0


def find_cloudflared() -> str | None:
    for candidate in _KNOWN_PATHS:
        if Path(candidate).is_file():
            return candidate
    return shutil.which("cloudflared")


class QuickTunnel:
    """One `cloudflared tunnel --url` process, and the address it announced."""

    def __init__(self, port: int, executable: str | None = None) -> None:
        self.port = port
        self.executable = executable or find_cloudflared()
        self._lock = threading.Lock()
        self._url: str | None = None
        self._connected = False
        self._error: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def url(self) -> str | None:
        """The address, once the tunnel is actually carrying traffic."""
        with self._lock:
            return self._url if self._connected else None

    def status(self) -> dict[str, object]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "available": bool(self.executable),
                "running": running,
                # Named but not yet connected is a real and misleading state, so
                # it is reported rather than folded into "running".
                "connected": self._connected,
                "url": self._url,
                "error": self._error,
                "port": self.port,
            }

    def start(self, wait: bool = True) -> str | None:
        """Launch the tunnel. Returns the address once it has one."""
        if not self.executable:
            with self._lock:
                self._error = (
                    "cloudflared was not found. Install it, or set public_base_url "
                    "to a hostname you already have."
                )
            _LOG.warning("%s", self._error)
            return None

        self._stopping.clear()
        try:
            # stderr is where the address appears; merged so one reader sees
            # everything and cloudflared never blocks on a full pipe.
            self._process = subprocess.Popen(
                [
                    self.executable,
                    "tunnel",
                    "--no-autoupdate",
                    "--url",
                    f"http://127.0.0.1:{self.port}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            with self._lock:
                self._error = f"Could not start cloudflared: {exc}"
            _LOG.exception("Could not start cloudflared")
            return None

        self._reader = threading.Thread(target=self._read_output, name="quick-tunnel", daemon=True)
        self._reader.start()
        if not wait:
            return None

        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.url:
                return self.url
            if self._process.poll() is not None:
                with self._lock:
                    self._error = self._error or "cloudflared exited before connecting"
                return None
            time.sleep(0.2)
        with self._lock:
            self._error = f"cloudflared did not connect within {_STARTUP_TIMEOUT_S:.0f}s"
        _LOG.warning("%s", self._error)
        return None

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self._stopping.is_set():
                return
            self._consume(line)
        # The process ended. The address it had is gone with it, and continuing
        # to hand it to LINE would produce image messages that fetch nothing.
        with self._lock:
            self._url = None
            self._connected = False
            if not self._stopping.is_set():
                self._error = "cloudflared exited"

    def _consume(self, line: str) -> None:
        """One line of cloudflared output.

        Separate from the read loop so the two states it distinguishes --
        allocated, and actually connected -- can be exercised without a live
        process, which is the distinction that was got wrong.
        """
        match = _URL_PATTERN.search(line)
        if match:
            with self._lock:
                self._url = match.group(0)
                self._error = None
            _LOG.info("Quick tunnel allocated %s", match.group(0))
        if _READY_PATTERN.search(line):
            with self._lock:
                self._connected = True
            _LOG.info("Quick tunnel connected")

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        process = self._process
        if process is None:
            return
        with self._lock:
            self._url = None
            self._connected = False
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
        self._process = None
