class RequestError(ValueError):
    """A request refusal: HTTP 400, or a named error after a stream starts."""


class CapacityError(RequestError):
    """A transient capacity refusal: HTTP 503, retry shortly."""


class RoundError(RuntimeError):
    """A failed round's type and message, without its engine frames or exception payloads."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
