from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import RuntimeConfig
from .errors import DependencyError, RecognitionError


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    confidence: float


class Recognizer(Protocol):
    def recognize(self, images: Sequence[Any]) -> list[OCRResult]: ...


class PaddleRecognizer:
    def __init__(self, config: RuntimeConfig):
        # PaddleX reads these values while paddleocr is imported, so configure
        # them before the lazy import. Keeping the cache inside the project also
        # makes native Windows and WSL runs share one explicit model location.
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", config.model_source)
        os.environ.setdefault(
            "PADDLE_PDX_CACHE_HOME", str(Path(config.model_cache_dir).resolve())
        )
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise DependencyError(
                "PaddleOCR is not installed. Install PaddlePaddle for your CPU, "
                "then `pip install -e .[ocr]`."
            ) from exc
        self.batch_size = config.batch_size
        try:
            self.model = TextRecognition(
                model_name=config.model_name,
                device=config.device,
                enable_hpi=config.enable_hpi,
                enable_mkldnn=config.enable_mkldnn,
                cpu_threads=config.cpu_threads,
            )
        except Exception as exc:
            raise RecognitionError(
                f"Cannot initialize PaddleOCR model {config.model_name}: {exc}"
            ) from exc

    def recognize(self, images: Sequence[Any]) -> list[OCRResult]:
        if not images:
            return []
        try:
            raw_results = list(self.model.predict(input=list(images), batch_size=self.batch_size))
        except Exception as exc:
            raise RecognitionError(f"PaddleOCR inference failed: {exc}") from exc
        results = [extract_ocr_result(item) for item in raw_results]
        if len(results) != len(images):
            raise RecognitionError(
                f"PaddleOCR returned {len(results)} results for {len(images)} input crops"
            )
        return results


class ParallelRecognizer:
    """Split one OCR batch across independent recognizer instances."""

    def __init__(self, recognizers: Sequence[Recognizer]):
        if len(recognizers) < 2:
            raise ValueError("ParallelRecognizer requires at least two recognizers")
        self.recognizers = list(recognizers)
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.recognizers), thread_name_prefix="paddle-rec"
        )

    def recognize(self, images: Sequence[Any]) -> list[OCRResult]:
        if not images:
            return []
        worker_count = min(len(images), len(self.recognizers))
        chunk_size = (len(images) + worker_count - 1) // worker_count
        chunks = [
            images[index : index + chunk_size]
            for index in range(0, len(images), chunk_size)
        ]
        futures = [
            self.executor.submit(recognizer.recognize, chunk)
            # Fewer chunks than recognizers is normal: a batch smaller than the
            # worker count leaves the surplus recognizers idle on purpose.
            for recognizer, chunk in zip(self.recognizers, chunks, strict=False)
        ]
        return [result for future in futures for result in future.result()]

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def close_recognizer(recognizer: Recognizer) -> None:
    """Release whatever a recognizer holds open, if it holds anything."""
    close = getattr(recognizer, "close", None)
    if callable(close):
        close()


def create_recognizer(config: RuntimeConfig) -> Recognizer:
    if config.recognizer_workers == 1:
        return PaddleRecognizer(config)
    return ParallelRecognizer([PaddleRecognizer(config) for _ in range(config.recognizer_workers)])


def extract_ocr_result(result: Any) -> OCRResult:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        import json

        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RecognitionError("PaddleOCR result JSON is malformed") from exc
    if not isinstance(payload, dict):
        raise RecognitionError(f"Unsupported PaddleOCR result type: {type(payload).__name__}")
    if isinstance(payload.get("res"), dict):
        payload = payload["res"]
    text = payload.get("rec_text")
    score = payload.get("rec_score")
    if text is None or score is None:
        raise RecognitionError("PaddleOCR result does not contain rec_text and rec_score")
    try:
        return OCRResult(text=str(text), confidence=max(0.0, min(1.0, float(score))))
    except (TypeError, ValueError) as exc:
        raise RecognitionError("PaddleOCR rec_score is not numeric") from exc
