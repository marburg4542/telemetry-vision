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
        # Every cloudflared this object has launched, so stop() can be a
        # guarantee rather than an assumption. start() no longer replaces a live
        # process, but a handle that was never terminated is a public hostname
        # left answering after the server behind it is gone, and that is not a
        # thing to leave to reasoning about which paths can reach it.
        self._processes: list[subprocess.Popen[str]] = []
        self._reader: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def url(self) -> str | None:
        """The address, once the tunnel is actually carrying traffic."""
        with self._lock:
            return self._url if self._connected else None

    def status(self) -> dict[str, object]:
        with self._lock:
            running = self._live(self._process)
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
        """Launch the tunnel. Returns the address once it has one.

        Calling this while a cloudflared is already running waits for that one
        rather than launching a second. It reads like a defensive nicety and is
        not: the caller in the server asks for a tunnel whenever `url` is None,
        and `url` is None for the whole window between a process being spawned
        and it registering a connection. Spawning there replaced `_process`, so
        stop() could no longer reach the first -- and its reader thread went on
        writing to this object, clearing the live tunnel's address the moment
        the orphan died. The address then handed to LINE was nothing at all.
        """
        if not self.executable:
            with self._lock:
                self._error = (
                    "cloudflared was not found. Install it, or set public_base_url "
                    "to a hostname you already have."
                )
            _LOG.warning("%s", self._error)
            return None

        self._stopping.clear()
        with self._lock:
            running = self._process if self._live(self._process) else None
        if running is not None:
            _LOG.info("Quick tunnel already running; waiting for it rather than starting another")
            return self._await_url(running) if wait else None

        try:
            # stderr is where the address appears; merged so one reader sees
            # everything and cloudflared never blocks on a full pipe.
            process: subprocess.Popen[str] = subprocess.Popen(
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

        with self._lock:
            self._process = process
            self._processes.append(process)
            # A new process starts from no address rather than inheriting the
            # last one's. Keeping it would hand LINE the previous tunnel's
            # hostname during the seconds before this one connects.
            self._url = None
            self._connected = False
        self._reader = threading.Thread(
            target=self._read_output, args=(process,), name="quick-tunnel", daemon=True
        )
        self._reader.start()
        if not wait:
            return None
        return self._await_url(process)

    def _await_url(self, process: subprocess.Popen[str]) -> str | None:
        """Block until that process is carrying traffic, or has given up."""
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            url = self.url
            if url:
                return url
            if process.poll() is not None:
                with self._lock:
                    self._error = self._error or "cloudflared exited before connecting"
                return None
            time.sleep(0.2)
        with self._lock:
            self._error = f"cloudflared did not connect within {_STARTUP_TIMEOUT_S:.0f}s"
        _LOG.warning("%s", self._error)
        return None

    @staticmethod
    def _live(process: subprocess.Popen[str] | None) -> bool:
        return process is not None and process.poll() is None

    def _owns(self, process: subprocess.Popen[str]) -> bool:
        """Whether this process is still the one this object speaks for.

        Must be read under the lock together with whatever it guards. A reader
        thread outlives the process it was started for, and everything it has to
        say about a process that has been replaced is stale.
        """
        return self._process is process

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            if self._stopping.is_set():
                return
            self._consume(line, process)
        # The process ended. The address it had is gone with it, and continuing
        # to hand it to LINE would produce image messages that fetch nothing --
        # but only if this is still the process the object is speaking for.
        with self._lock:
            if not self._owns(process):
                return
            self._url = None
            self._connected = False
            if not self._stopping.is_set():
                self._error = "cloudflared exited"

    def _consume(self, line: str, process: subprocess.Popen[str] | None = None) -> None:
        """One line of cloudflared output.

        Separate from the read loop so the two states it distinguishes --
        allocated, and actually connected -- can be exercised without a live
        process, which is the distinction that was got wrong.

        `process` is the one this line came from, and passing it makes the line
        ignored once that process has been replaced. It is optional so a test
        can drive the parsing on its own.
        """
        url = _URL_PATTERN.search(line)
        ready = _READY_PATTERN.search(line)
        if url is None and ready is None:
            return
        with self._lock:
            if process is not None and not self._owns(process):
                return
            if url is not None:
                self._url = url.group(0)
                self._error = None
            if ready is not None:
                self._connected = True
        # Logged outside the lock: a handler can block on a file or a console.
        if url is not None:
            _LOG.info("Quick tunnel allocated %s", url.group(0))
        if ready is not None:
            _LOG.info("Quick tunnel connected")

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate every cloudflared this object launched, not just the last."""
        self._stopping.set()
        with self._lock:
            self._url = None
            self._connected = False
            processes = list(self._processes)
            self._processes.clear()
            self._process = None
        for process in processes:
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
