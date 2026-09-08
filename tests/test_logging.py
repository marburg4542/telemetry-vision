"""The server's log has to survive being run hidden.

The scheduled task starts this with no console, so anything written to stderr is
discarded before it exists. For as long as the only handler was a stream
handler, a tunnel that would not connect, a public URL that answered 404 and a
LINE push that was refused all produced exactly the same evidence: none. The
operator saw a message arrive without its picture and had nowhere to look.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import unittest
from pathlib import Path


class _LoggingTestCase(unittest.TestCase):
    """A scratch directory, and the root logger put back the way it was found.

    configure_logging uses force=True, which removes whatever handlers are
    already installed -- including the ones pytest reports through. Leaving that
    in place would make this file quietly break the output of every test that
    runs after it.

    The order of the two cleanups matters and is why the directory is not a
    `with` block inside each test. Cleanups run last-registered-first, so the
    handler is closed before the tree is removed: on Windows a file still open
    cannot be deleted, and the removal fails rather than the test.
    """

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        root = logging.getLogger()
        handlers = list(root.handlers)
        level = root.level
        self.addCleanup(self._restore, handlers, level)

    def _restore(self, handlers: list[logging.Handler], level: int) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers = handlers
        root.setLevel(level)

    def _flush(self) -> None:
        for handler in logging.getLogger().handlers:
            handler.flush()


class LogFileTests(_LoggingTestCase):
    def test_the_log_reaches_a_file_and_not_only_stderr(self) -> None:
        from telemetry_ocr.webapp import LOG_FILENAME, configure_logging

        path = configure_logging(self.directory)
        self.assertEqual(path, self.directory / LOG_FILENAME)
        assert path is not None

        logging.getLogger("telemetry_ocr.quick_tunnel").warning(
            "Public URL unreachable; restarting the tunnel"
        )
        self._flush()

        written = path.read_text(encoding="utf-8")
        self.assertIn("Public URL unreachable", written)
        # The logger name is what says which half of the send failed, so it has
        # to be in the line and not only in the message.
        self.assertIn("telemetry_ocr.quick_tunnel", written)

    def test_info_is_recorded_because_the_tunnel_reports_at_that_level(self) -> None:
        # "Quick tunnel allocated" and "Quick tunnel connected" are INFO, and
        # they are the two lines that distinguish a tunnel that was named from
        # one that is carrying traffic. A WARNING-only log would lose both.
        from telemetry_ocr.webapp import configure_logging

        path = configure_logging(self.directory)
        assert path is not None
        logging.getLogger("telemetry_ocr.quick_tunnel").info(
            "Quick tunnel allocated https://example.trycloudflare.com"
        )
        self._flush()
        self.assertIn("Quick tunnel allocated", path.read_text(encoding="utf-8"))

    def test_the_directory_is_created_when_it_does_not_exist(self) -> None:
        from telemetry_ocr.webapp import configure_logging

        target = self.directory / "output" / "logs"
        path = configure_logging(target)
        self.assertIsNotNone(path)
        self.assertTrue(target.is_dir())

    def test_stderr_keeps_its_handler_so_a_terminal_run_still_prints(self) -> None:
        from telemetry_ocr.webapp import configure_logging

        configure_logging(self.directory)
        kinds = {type(handler).__name__ for handler in logging.getLogger().handlers}
        self.assertIn("StreamHandler", kinds)
        self.assertIn("RotatingFileHandler", kinds)


class DegradedTests(_LoggingTestCase):
    def test_a_log_directory_that_cannot_be_made_does_not_stop_the_server(self) -> None:
        """Serving without a log beats refusing to serve.

        Returns None rather than raising, and the caller prints that instead of
        a path, so the operator is told which of the two happened.
        """
        from telemetry_ocr.webapp import configure_logging

        blocker = self.directory / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        path = configure_logging(blocker / "logs")
        self.assertIsNone(path)
        # Still configured, so the stream handler is there to carry the warning
        # about why there is no file.
        self.assertTrue(logging.getLogger().handlers)

    def test_passing_none_configures_stderr_only(self) -> None:
        from telemetry_ocr.webapp import configure_logging

        self.assertIsNone(configure_logging(None))
        kinds = {type(handler).__name__ for handler in logging.getLogger().handlers}
        self.assertEqual(kinds, {"StreamHandler"})


class AccessLogTests(unittest.TestCase):
    def test_the_request_log_is_off_so_the_file_keeps_what_matters(self) -> None:
        """aiohttp logs every request at INFO, and the page polls once a second.

        Harmless while the log went to a discarded stderr; against a rotating
        file it is a flood. Measured on the first day of the file: 3,604 access
        lines to one line about anything else, rotating every 29 minutes -- so a
        tunnel that failed an hour earlier had already aged out of the log added
        to explain exactly that.
        """
        import inspect

        from telemetry_ocr import webapp

        source = inspect.getsource(webapp.run_web_app)
        self.assertIn("access_log=None", source)


class ForceTests(_LoggingTestCase):
    def test_it_installs_the_file_even_when_the_root_logger_is_already_set_up(self) -> None:
        """basicConfig without force= is a no-op once anything has configured it.

        Something else importing this package and calling basicConfig first is
        enough to leave the file unwritten, which is the silent version of the
        bug this whole file exists for.
        """
        from telemetry_ocr.webapp import configure_logging

        logging.basicConfig(level=logging.INFO, force=True)
        path = configure_logging(self.directory)
        assert path is not None
        logging.getLogger("telemetry_ocr.webapp").warning("after a prior basicConfig")
        self._flush()
        self.assertIn("after a prior basicConfig", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
