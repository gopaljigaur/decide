from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decide.types import Response


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
    """Every backend in the route failed.

    For a single `decide()` call, `route` and `errors` describe the one
    request that failed.

    For `decide_batch()`/`AsyncClient.decide_batch()`, a batch is only
    unroutable as a whole if at least one state in it could not be resolved
    by any backend (no response at all, not even a low-confidence one). In
    that case `route` and `errors` are the concatenation, in input order, of
    the route and errors of every unroutable state; `partial` holds the
    `Response` for every state in the batch that *did* resolve, keyed by its
    index in the input sequence; `failed` holds the route accumulated so far
    for every state that did not resolve, also keyed by its input index.
    """

    def __init__(
        self,
        route: list[str],
        errors: list[BackendError],
        *,
        partial: dict[int, Response] | None = None,
        failed: dict[int, list[str]] | None = None,
    ) -> None:
        self.route = route
        self.errors = errors
        self.partial = partial if partial is not None else {}
        self.failed = failed if failed is not None else {}
        super().__init__(f"all backends failed: {route}")
