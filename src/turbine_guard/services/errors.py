"""Stable service-layer failures mapped to safe HTTP errors."""


class ServiceError(RuntimeError):
    """Base error with a stable external code and safe message."""

    code = "service_error"
    status_code = 500


class AssetNotFoundError(ServiceError):
    code = "asset_not_found"
    status_code = 404


class HistoryConflictError(ServiceError):
    code = "history_conflict"
    status_code = 409


class ModelUnavailableError(ServiceError):
    code = "model_unavailable"
    status_code = 503


class FeatureContractError(ServiceError):
    code = "feature_contract_unavailable"
    status_code = 503


class DatabaseUnavailableError(ServiceError):
    code = "database_unavailable"
    status_code = 503


class RequestParameterError(ServiceError):
    code = "request_validation_failed"
    status_code = 422
