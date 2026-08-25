"""Domain level exceptions mapped to HTTP responses in ``app.main``."""

from __future__ import annotations


class KorisQuantError(Exception):
    """Base class for every business error raised by the platform."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}


class DataUnavailableError(KorisQuantError):
    status_code = 503
    code = "data_unavailable"


class SymbolNotFoundError(KorisQuantError):
    status_code = 404
    code = "symbol_not_found"


class ModelNotTrainedError(KorisQuantError):
    status_code = 409
    code = "model_not_trained"


class InvalidRequestError(KorisQuantError):
    status_code = 422
    code = "invalid_request"


class PortfolioError(KorisQuantError):
    status_code = 400
    code = "portfolio_error"
