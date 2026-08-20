"""Shipped destinations for redacted trace events."""

from penampakan.trace_sinks.jsonl import JsonlTraceSink, JsonlTraceSinkStats
from penampakan.trace_sinks.memory import InMemoryTraceSink, InMemoryTraceSinkStats
from penampakan.trace_sinks.opentelemetry import (
    OpenTelemetryTraceSink,
    OpenTelemetryTraceSinkStats,
    OpenTelemetryUnavailableError,
)

__all__ = [
    "InMemoryTraceSink",
    "InMemoryTraceSinkStats",
    "JsonlTraceSink",
    "JsonlTraceSinkStats",
    "OpenTelemetryTraceSink",
    "OpenTelemetryTraceSinkStats",
    "OpenTelemetryUnavailableError",
]
