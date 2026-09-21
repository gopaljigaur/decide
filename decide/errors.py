from __future__ import annotations


class DecideError(Exception):
    """Base class for all errors raised by decide."""


class ConfigError(DecideError):
    """Raised for invalid configuration, such as an unknown or uninstalled backend."""


class BackendError(DecideError):
    """Raised when a backend fails to produce a usable response."""

    def __init__(self, backend: str, message: str, cause: BaseException | None = None) -> None:
        self.backend = backend
        self.message = message
        self.cause = cause
        super().__init__(f"{backend}: {message}")


class AuthError(BackendError):
    """The backend rejected our credentials."""


class RateLimitError(BackendError):
    """The backend is throttling us."""


class BackendConnectionError(BackendError):
    """We could not reach the backend."""


class BadResponseError(BackendError):
    """The backend's response was unparsable or did not match the expected schema."""


class AllBackendsFailed(DecideError):
    """Every backend in the route failed."""

    def __init__(self, route: list[str], errors: list[BackendError]) -> None:
        self.route = route
        self.errors = errors
        super().__init__(f"all backends failed: {route}")
