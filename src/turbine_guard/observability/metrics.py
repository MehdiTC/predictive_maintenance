"""Per-application Prometheus metrics without global collector collisions."""

import threading
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info, generate_latest


@dataclass(frozen=True)
class MetricsSnapshot:
    request_count: int
    prediction_count: int
    validation_failures: int
    database_failures: int
    model_load_failures: int
    prediction_failures: int
    conflict_count: int
    average_prediction_latency_ms: float | None


class OnlineMetrics:
    """Metrics and a small in-process summary for one FastAPI app instance."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "turbine_guard_http_requests_total",
            "HTTP requests by low-cardinality route, method, and status.",
            ("route", "method", "status"),
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "turbine_guard_http_request_duration_seconds",
            "HTTP request duration by route and method.",
            ("route", "method"),
            registry=self.registry,
        )
        self.sensor_ingestions = Counter(
            "turbine_guard_sensor_ingestions_total",
            "Accepted sensor-ingestion requests.",
            registry=self.registry,
        )
        self.predictions = Counter(
            "turbine_guard_predictions_total",
            "Persisted or idempotently reused predictions.",
            registry=self.registry,
        )
        self.prediction_failures_metric = Counter(
            "turbine_guard_prediction_failures_total",
            "Prediction requests that failed before commit.",
            registry=self.registry,
        )
        self.model_load_failures_metric = Counter(
            "turbine_guard_model_load_failures_total",
            "Champion load or feature-contract failures.",
            registry=self.registry,
        )
        self.database_failures_metric = Counter(
            "turbine_guard_database_failures_total",
            "Operational database failures.",
            registry=self.registry,
        )
        self.validation_failures_metric = Counter(
            "turbine_guard_validation_failures_total",
            "Request-schema validation failures.",
            registry=self.registry,
        )
        self.conflicts = Counter(
            "turbine_guard_conflicts_total",
            "Idempotency or history conflicts.",
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            "turbine_guard_prediction_duration_seconds",
            "Model prediction duration.",
            registry=self.registry,
        )
        self.risk_predictions = Counter(
            "turbine_guard_risk_predictions_total",
            "Predictions by constrained risk level.",
            ("risk_level",),
            registry=self.registry,
        )
        self.current_model = Info(
            "turbine_guard_current_model",
            "Current registered model identity.",
            registry=self.registry,
        )
        self.ready = Gauge(
            "turbine_guard_ready",
            "Whether all configured dependencies are ready.",
            registry=self.registry,
        )
        self._lock = threading.Lock()
        self._request_count = 0
        self._prediction_count = 0
        self._validation_failures = 0
        self._database_failures = 0
        self._model_load_failures = 0
        self._prediction_failures = 0
        self._conflict_count = 0
        self._prediction_latency_total_ms = 0.0

    def record_http(self, route: str, method: str, status: int, seconds: float) -> None:
        self.http_requests.labels(route=route, method=method, status=str(status)).inc()
        self.http_latency.labels(route=route, method=method).observe(seconds)
        with self._lock:
            self._request_count += 1
            if status == 422:
                self._validation_failures += 1
                self.validation_failures_metric.inc()

    def record_prediction(self, risk_level: str, latency_ms: float) -> None:
        self.sensor_ingestions.inc()
        self.predictions.inc()
        self.prediction_latency.observe(latency_ms / 1000.0)
        self.risk_predictions.labels(risk_level=risk_level).inc()
        with self._lock:
            self._prediction_count += 1
            self._prediction_latency_total_ms += latency_ms

    def record_failure(self, kind: str) -> None:
        with self._lock:
            if kind == "database":
                self._database_failures += 1
                self.database_failures_metric.inc()
            elif kind == "model":
                self._model_load_failures += 1
                self.model_load_failures_metric.inc()
            elif kind == "conflict":
                self._conflict_count += 1
                self.conflicts.inc()
            else:
                self._prediction_failures += 1
                self.prediction_failures_metric.inc()

    def set_model(self, name: str, version: str, alias: str) -> None:
        self.current_model.info({"name": name, "version": version, "alias": alias})

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            average = (
                self._prediction_latency_total_ms / self._prediction_count
                if self._prediction_count
                else None
            )
            return MetricsSnapshot(
                request_count=self._request_count,
                prediction_count=self._prediction_count,
                validation_failures=self._validation_failures,
                database_failures=self._database_failures,
                model_load_failures=self._model_load_failures,
                prediction_failures=self._prediction_failures,
                conflict_count=self._conflict_count,
                average_prediction_latency_ms=average,
            )

    def render(self) -> bytes:
        return generate_latest(self.registry)
