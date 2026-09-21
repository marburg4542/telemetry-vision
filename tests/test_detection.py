import unittest

from telemetry_ocr.detection import UltralyticsDetector


class _Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return self.value


class _Boxes:
    # The second box is one the tracker matched in its low-score stage: tracked,
    # but under the confidence asked for, so never drawn.
    def __init__(self):
        self.xyxy = _Tensor([[10.0, 20.0, 110.0, 90.0], [200.0, 20.0, 230.0, 60.0]])
        self.conf = _Tensor([0.91, 0.2])
        self.cls = _Tensor([1, 0])
        self.id = _Tensor([42, 43])

    def __len__(self):
        return 2


class _Gmc:
    def __init__(self):
        self.downscale = 2
        self.resets = 0

    def reset_params(self):
        self.resets += 1


class _Tracker:
    def __init__(self):
        self.reset_count = 0
        self.gmc = _Gmc()

    def reset(self):
        self.reset_count += 1


class _Model:
    def __init__(self):
        self.names = {0: "car", 1: "truck"}
        self.tracker = _Tracker()
        self.predictor = type("Predictor", (), {"trackers": [self.tracker]})()
        self.task = "detect"
        self.ckpt = {
            "train_args": {
                "model": "yolov8m.pt",
                "epochs": 100,
                "imgsz": [640, 640],
            }
        }
        self.options = None

    def track(self, **options):
        self.options = options
        return [type("Result", (), {"boxes": _Boxes()})()]


class UltralyticsDetectorAdapterTests(unittest.TestCase):
    def setUp(self):
        self.detector = object.__new__(UltralyticsDetector)
        self.detector.model = _Model()
        self.detector.device = "cpu"

    def test_track_matches_the_realtime_test_settings_and_returns_ids(self):
        frame = object()
        detections = self.detector.track(
            frame,
            confidence=0.45,
            iou=0.45,
            image_size=640,
            tracker="botsort.yaml",
        )

        # Tracked down to the floor, drawn only from the confidence asked for.
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].track_id, 42)
        self.assertEqual(detections[0].label, "truck")
        self.assertAlmostEqual(detections[0].confidence, 0.91)
        self.assertEqual(
            self.detector.model.options,
            {
                "source": frame,
                "conf": 0.1,
                "iou": 0.45,
                "imgsz": 640,
                "max_det": 300,
                "persist": True,
                "tracker": "botsort.yaml",
                "device": "cpu",
                "verbose": False,
            },
        )
        # Passing half at all — even False — makes Ultralytics 8.4 emit a
        # deprecation warning on every single call, which reached 17k lines of
        # log in one session. The default is already FP32, which benchmarks
        # five times faster than FP16 on this card.
        self.assertNotIn("half", self.detector.model.options)

    def test_the_camera_motion_estimate_is_shrunk_once_and_restarted(self):
        gmc = self.detector.model.tracker.gmc
        for _ in range(3):
            self.detector.track(
                object(), confidence=0.45, iou=0.45, image_size=640, tracker="botsort.yaml"
            )
        self.assertEqual(gmc.downscale, 4)
        # Once: a pyramid kept at the old size would fail to match the next frame.
        self.assertEqual(gmc.resets, 1)

    def test_a_confidence_below_the_floor_is_asked_for_as_it_is(self):
        self.detector.track(
            object(), confidence=0.05, iou=0.45, image_size=640, tracker="botsort.yaml"
        )
        self.assertEqual(self.detector.model.options["conf"], 0.05)

    def test_reset_and_checkpoint_metadata(self):
        self.detector.reset_tracking()
        self.assertEqual(self.detector.model.tracker.reset_count, 1)
        self.assertEqual(
            self.detector.metadata(),
            {
                "classes": ["car", "truck"],
                "architecture": "yolov8m",
                "epochs": 100,
                "trained_image_size": 640,
                "task": "detect",
                "device": "cpu",
            },
        )

    def test_track_follows_the_configured_device(self):
        self.detector.device = "cuda:0"
        self.detector.track(object(), confidence=0.4, iou=0.5, image_size=416)
        self.assertEqual(self.detector.model.options["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
