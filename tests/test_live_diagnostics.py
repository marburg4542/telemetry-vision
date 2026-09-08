"""What the operator can see while something is still running.

Both cases here are the same mistake in different places: a number or a cleanup
that only exists once the work has finished, in code whose whole point is work
that does not finish. A stream runs until someone stops it, and a detection
sidecar that never became ready never reaches the line that shuts it down.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any
from unittest import mock


class StreamHealthTests(unittest.TestCase):
    """Connection health has to be readable while the connection is up.

    PipelineSummary carries `reconnects` and `decode_failures`, and its own
    comment calls them the difference between "OCR is struggling" and "the feed
    is unreliable". The summary is only assigned after `pipeline.run` returns,
    which for a live camera is never -- so status() reported a flat zero for
    both for the entire life of a stream, and one reconnecting every twenty
    seconds looked perfectly healthy.
    """

    def _service(self) -> Any:
        from telemetry_ocr.stream_service import StreamService

        service = StreamService.__new__(StreamService)
        service.url = "rtsp://cam/live"
        service._lock = threading.Lock()
        service._summary = None
        service._error = None
        service._started_at = None
        service._stop = threading.Event()
        service._pipeline = mock.Mock()
        service._pipeline.decoder = None
        # `running` is a read-only property over the runner thread.
        service._runner = mock.Mock()
        service._runner.is_alive.return_value = False
        from telemetry_ocr.stream_service import StreamSnapshot

        service._snapshot = StreamSnapshot()
        return service

    def test_a_running_stream_reports_the_decoders_live_counters(self) -> None:
        service = self._service()
        service._runner.is_alive.return_value = True

        decoder = mock.Mock()
        decoder.reconnects = 4
        decoder.decode_failures = 17
        service._pipeline.decoder = decoder

        status = service.status()
        self.assertEqual(status["reconnects"], 4)
        self.assertEqual(status["decode_failures"], 17)

    def test_a_finished_run_still_reports_its_summary(self) -> None:
        # The summary is the authority once it exists: it covers the whole run,
        # including a decoder that has since been replaced by a later one.
        service = self._service()

        summary = mock.Mock()
        summary.reconnects = 9
        summary.decode_failures = 2
        summary.update_hz = 9.9
        summary.latency_mean_ms = 42.0
        service._summary = summary

        decoder = mock.Mock()
        decoder.reconnects = 0
        decoder.decode_failures = 0
        service._pipeline.decoder = decoder

        status = service.status()
        self.assertEqual(status["reconnects"], 9)
        self.assertEqual(status["decode_failures"], 2)

    def test_nothing_has_run_yet_and_that_is_zero_not_a_crash(self) -> None:
        service = self._service()
        status = service.status()
        self.assertEqual(status["reconnects"], 0)
        self.assertEqual(status["decode_failures"], 0)

    def test_the_pipeline_publishes_the_decoder_it_is_running(self) -> None:
        # The other half of the same fix: status() can only read a live counter
        # if run() actually hands the decoder over, and it starts as None so a
        # pipeline between runs reads as "nothing to report" rather than
        # raising.
        import inspect

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.pipeline import TelemetryPipeline

        config = AppConfig(
            runtime=RuntimeConfig(),
            fields=[
                FieldConfig(
                    field_id="alt",
                    box=(0.1, 0.1, 0.2, 0.2),
                    allowed_chars="0123456789",
                    regex=r"^\d+$",
                )
            ],
        )
        pipeline = TelemetryPipeline(config, recognizer=mock.Mock())
        self.assertIsNone(pipeline.decoder)
        self.assertIn("self.decoder = decoder", inspect.getsource(TelemetryPipeline.run))


class SidecarShutdownTests(unittest.TestCase):
    """A sidecar that never became ready still has to be shut down.

    start() spawns the worker and opens the output file before it waits up to
    two minutes for the model. Raising out of that wait from outside the
    try/finally skipped close(), leaving the YOLO child to multiprocessing's
    exit handler -- terminate() then an unbounded join(). A child stuck in a
    slow CUDA init, which is what caused the timeout in the first place, can sit
    through terminate() and hang the CLI at exit with nothing printed.
    """

    def test_start_sits_inside_the_try_that_owns_close(self) -> None:
        # Checked against the source because the alternative is standing up a
        # whole run of the CLI with a deliberately wedged CUDA child. The
        # ordering is the entire fix, and it is one thing to look at.
        import inspect

        from telemetry_ocr import cli

        source = inspect.getsource(cli.main)
        start_at = source.index("sidecar.start()")
        try_at = source.rindex("try:", 0, start_at)
        finally_at = source.index("sidecar.close()", start_at)
        between = source[try_at:start_at]
        self.assertNotIn(
            "finally:",
            between,
            "sidecar.start() must sit inside the try that owns close(), not after it",
        )
        self.assertLess(start_at, finally_at)

    def test_close_survives_a_start_that_never_finished(self) -> None:
        """close() is called on a half-built sidecar, so it must not assume one.

        The worker exists (start() spawned it before it raised) but the handle
        may not, and drain() will fail because nothing is ready.
        """
        from telemetry_ocr.detect_sidecar import DetectionSidecar

        sidecar = DetectionSidecar.__new__(DetectionSidecar)
        sidecar._worker = mock.Mock()
        sidecar._worker.status.side_effect = RuntimeError("never became ready")
        sidecar._handle = None
        sidecar.output_path = "output/detections.jsonl"
        sidecar.submitted = 0
        sidecar.written = 0
        sidecar.drain = mock.Mock(side_effect=RuntimeError("nothing to drain"))

        result = sidecar.close()

        # The escalating stop is the part that must run: it is the only thing
        # that reliably kills a child wedged in CUDA init.
        sidecar._worker.stop.assert_called_once()
        self.assertEqual(result["submitted"], 0)
        self.assertEqual(result["written"], 0)


if __name__ == "__main__":
    unittest.main()
