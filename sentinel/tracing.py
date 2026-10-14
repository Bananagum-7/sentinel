"""OpenTelemetry tracing.

Every request is a span tree:
    gateway.request
      guard.injection
      guard.pii
      cache.lookup
      upstream.call      (one per attempt, with provider/model/status)
      guard.output

Set OTEL_EXPORTER_OTLP_ENDPOINT (and OTEL_EXPORTER_OTLP_HEADERS for auth) to
ship spans to any OTLP backend: Langfuse, Jaeger, Grafana Tempo, Honeycomb.
OpenTelemetry is the vendor-neutral standard, so switching backends is a
config change, not a code change. Without an endpoint, spans are created
but dropped (near-zero cost).

Span attributes never include prompt text or PII.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

_tracer = None


def get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "sentinel"}))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("sentinel")
    except ImportError:
        _tracer = _NoopTracer()
    return _tracer


class _NoopSpan:
    def set_attribute(self, *_):
        pass

    def set_attributes(self, *_):
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, *_, **__):
        yield _NoopSpan()


@contextmanager
def span(name: str, **attributes):
    with get_tracer().start_as_current_span(name) as s:
        clean = {k: v for k, v in attributes.items() if v is not None}
        if clean:
            s.set_attributes(clean)
        yield s
