from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .config import AppConfig, FieldConfig, TemporalProfileConfig
from .validation import ValidatedValue

# A median needs three samples before it can hold an outlier out of the middle.
# With two, the "median" is whichever reading arrived last.
_MIN_MEDIAN_SAMPLES = 3


@dataclass(frozen=True, slots=True)
class Observation:
    timestamp_ms: float
    text: str
    value: Any
    confidence: float


@dataclass(frozen=True, slots=True)
class StabilizedValue:
    value: Any
    text: str | None
    confirmed: bool
    support: int


class FieldStabilizer:
    def __init__(self, field: FieldConfig, profile: TemporalProfileConfig):
        self.field = field
        self.profile = profile
        self.history: deque[Observation | None] = deque(maxlen=profile.window)
        self.last_value: Any = None
        self.last_text: str | None = None
        self.last_timestamp_ms: float | None = None

    def add(
        self, validated: ValidatedValue, confidence: float, timestamp_ms: float
    ) -> StabilizedValue:
        if validated.valid:
            self.history.append(
                Observation(
                    timestamp_ms=timestamp_ms,
                    text=validated.normalized_text,
                    value=validated.value,
                    confidence=confidence,
                )
            )
        else:
            # Invalid reads still consume time in the voting window so stale
            # consensus eventually expires instead of remaining confirmed forever.
            self.history.append(None)
        counts = Counter(item.text for item in self.history if item is not None)
        weights: dict[str, float] = defaultdict(float)
        for item in self.history:
            if item is not None:
                weights[item.text] += item.confidence
        if not counts:
            return StabilizedValue(self.last_value, self.last_text, False, 0)
        winner = max(counts, key=lambda text: (counts[text], weights[text]))
        support = counts[winner]
        confirmed = support >= self.profile.required
        winning_observation = next(
            item for item in reversed(self.history) if item is not None and item.text == winner
        )
        numeric_observations = [
            item
            for item in self.history
            if item is not None and isinstance(item.value, (int, float))
        ]
        if (
            not confirmed
            and len(numeric_observations) >= max(self.profile.required, _MIN_MEDIAN_SAMPLES)
            and self._window_is_coherent(numeric_observations)
        ):
            # Continuously changing telemetry may never repeat an exact string.
            # Use the observed median so a single valid-but-wrong spike cannot
            # become the confirmed value merely because it is the latest read.
            ordered = sorted(numeric_observations, key=lambda item: item.value)
            if len(ordered) % 2:
                winning_observation = ordered[len(ordered) // 2]
            else:
                middle = {
                    id(item)
                    for item in ordered[len(ordered) // 2 - 1 : len(ordered) // 2 + 1]
                }
                winning_observation = next(
                    item for item in reversed(numeric_observations) if id(item) in middle
                )
            support = len(numeric_observations)
            confirmed = True
        if confirmed and self._passes_delta(winning_observation):
            changed = winning_observation.text != self.last_text
            self.last_value = winning_observation.value
            self.last_text = winning_observation.text
            if changed or self.last_timestamp_ms is None:
                self.last_timestamp_ms = timestamp_ms
        else:
            confirmed = False
        return StabilizedValue(self.last_value, self.last_text, confirmed, support)

    def _window_is_coherent(self, observations: list[Observation]) -> bool:
        """Reject a median drawn from readings that disagree more than the field can move.

        `_passes_delta` asks whether the winning value jumped away from the last
        confirmed one. This asks the different question of whether the window
        agrees with itself well enough for its median to mean anything.
        """
        limit = self.field.max_delta_per_second
        if limit is None:
            return True
        values = [observation.value for observation in observations]
        elapsed_seconds = max(
            (observations[-1].timestamp_ms - observations[0].timestamp_ms) / 1000.0, 0.001
        )
        return max(values) - min(values) <= limit * elapsed_seconds

    def _passes_delta(self, observation: Observation) -> bool:
        limit = self.field.max_delta_per_second
        if (
            limit is None
            or self.last_value is None
            or self.last_timestamp_ms is None
            or not isinstance(observation.value, (int, float))
            or not isinstance(self.last_value, (int, float))
        ):
            return True
        elapsed_seconds = max((observation.timestamp_ms - self.last_timestamp_ms) / 1000.0, 0.001)
        return abs(observation.value - self.last_value) <= limit * elapsed_seconds


class TemporalStabilizer:
    def __init__(self, config: AppConfig):
        profiles = {"fast": config.temporal.fast, "stable": config.temporal.stable}
        self.fields = {
            item.field_id: FieldStabilizer(item, profiles[item.profile]) for item in config.fields
        }

    def add(
        self,
        field_id: str,
        validated: ValidatedValue,
        confidence: float,
        timestamp_ms: float,
    ) -> StabilizedValue:
        return self.fields[field_id].add(validated, confidence, timestamp_ms)
