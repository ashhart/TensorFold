class RequestError(ValueError):
    """A request refusal: HTTP 400, or a named error after a stream starts."""


class CapacityError(RequestError):
    """A transient capacity refusal: HTTP 503, retry shortly."""


class RoundError(RuntimeError):
    """A failed round's type and message, without its engine frames or exception payloads."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class ContextLengthError(RequestError):
    """A prompt and its reply past the context window: OpenAI's context_length_exceeded, which clients compact on."""

    code = "context_length_exceeded"


CONTEXT_LIMIT = "This server's maximum context length is"      # OpenAI's wording, which clients match to compact


def refusal(problem: str) -> RequestError:
    """A refusal string as its error: a context-window one carries OpenAI's code."""

    return (ContextLengthError if problem.startswith(CONTEXT_LIMIT) else RequestError)(problem)


def error_body(exc: Exception) -> dict:
    """OpenAI's error object: the message, its type, and the code clients key on where there is one."""

    body = {"message": str(exc), "type": "invalid_request_error"}
    if getattr(exc, "code", None):
        body["code"] = exc.code
    return body
