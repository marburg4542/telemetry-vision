"""Cleanup that has to happen even when the work never properly started.

A detection sidecar that never became ready never reaches the line that shuts it
down, unless that line is written for exactly that case.
"""

from __future__ import annotations

import unittest
from unittest import mock


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
