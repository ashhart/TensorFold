"""Opt-in bearer authentication and report redaction for standard-library benchmark clients."""
import os


def validate_api_key(key: str) -> None:
    """Reject empty, non-ASCII and whitespace-containing credentials without including their value in errors."""
    if not key or any(not 33 <= ord(char) <= 126 for char in key):
        raise ValueError("API key must contain printable ASCII characters without whitespace")


def read_api_key(parser, name: str | None) -> str | None:
    """Read a credential only when an environment variable was explicitly selected."""
    if name is None:
        return None
    key = os.environ.get(name)
    if key is None:
        parser.error("API key environment variable is not set")
    try:
        validate_api_key(key)
    except ValueError:
        parser.error("API key must contain printable ASCII characters without whitespace")
    return key


def authorize(request, key: str | None) -> None:
    """Attach credentials to the initial request only; urllib must not forward them on redirects."""
    if key is not None:
        validate_api_key(key)
        request.add_unredirected_header("Authorization", "Bearer " + key)


def redact(value, key: str | None):
    """Remove literal credentials from report strings, including server output and user-supplied labels."""
    if not key:
        return value
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]")
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {redact(name, key): redact(item, key) for name, item in value.items()}
    return value
