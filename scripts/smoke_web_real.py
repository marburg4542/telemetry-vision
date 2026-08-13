"""Run one real PaddleOCR frame through the local web API without a browser."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-time-ms", type=float, default=240_000.0)
    parser.add_argument("--requests", type=int, default=3)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    import cv2
    from aiohttp.test_utils import TestClient, TestServer

    from telemetry_ocr.config import load_config
    from telemetry_ocr.ocr import create_recognizer
    from telemetry_ocr.webapp import create_app

    config = load_config(args.config)
    recognizer = create_recognizer(config.runtime)
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.source_time_ms)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode frame from {args.video}")
    encoded_ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not encoded_ok:
        raise RuntimeError("Cannot encode the real-video frame")

    client = TestClient(TestServer(create_app(config, recognizer)))
    await client.start_server()
    try:
        session_response = await client.post("/api/session")
        session_id = (await session_response.json())["session_id"]
        payload = None
        for index in range(args.requests):
            source_time_ms = args.source_time_ms + index * 100
            response = await client.post(
                f"/api/session/{session_id}/frame"
                f"?source_time_ms={source_time_ms}&frame_index={index}",
                data=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"},
            )
            if response.status != 200:
                raise RuntimeError(await response.text())
            payload = await response.json()
        assert payload is not None
        summary = {
            "http": 200,
            "dimensions": [payload["width"], payload["height"]],
            "records": len(payload["records"]),
            "confirmed": sum(bool(row["confirmed"]) for row in payload["records"]),
            "latency_ms": round(payload["latency_ms"], 2),
            "sample": [
                {
                    "field_id": row["field_id"],
                    "value": row["value"],
                    "raw_text": row["raw_text"],
                }
                for row in payload["records"][:3]
            ],
        }
        print(summary)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
