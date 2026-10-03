"""Service profiles and monitoring pass credentials without logging them."""

import sys

from tensorfold.control.cli import parser as control_parser, _profile as profile_from
from tensorfold.control.config import Profile
from tensorfold.control.safety import redact
from tensorfold.control.telemetry import Client
from tensorfold.cli import build_parser


def test_serve_accepts_repeated_keys_and_metrics_exception():
    args = build_parser().parse_args(["serve", "fixture", "--api-key", "fixture-first", "--api-key", "fixture-second",
                                     "--api-key-file", "keys.txt", "--metrics-open"])
    assert args.api_key == ["fixture-first", "fixture-second"]
    assert args.api_key_file == "keys.txt" and args.metrics_open is True


def test_launch_agent_passes_the_file_option(tmp_path):
    path = str(tmp_path / "keys")
    profile = Profile("fixture", "fixture-model", python=sys.executable, api_key_file=path)
    command = profile.command()
    assert command[-2:] == ["--api-key-file", path]
    args = control_parser().parse_args(["service", "install", "fixture-model", "--api-key-file", path])
    assert profile_from(args).api_key_file == path


def test_redaction_hides_both_header_forms():
    assert redact("Authorization: Bearer fixture-secret") == "Authorization: Bearer [REDACTED]"
    assert redact("x-api-key: fixture-secret") == "x-api-key: [REDACTED]"
    assert redact("X-API-Key=fixture-secret") == "X-API-Key=[REDACTED]"


def test_monitoring_sends_bearer_to_metrics_without_credentials_in_the_url():
    from tests.control.test_telemetry import endpoint

    received = []
    def route(path, headers):
        received.append((path, headers.get("Authorization")))
        return 200, b"{}", {}
    with endpoint(route) as url:
        assert Client(url, token="fixture-secret").get("/metrics").status == 200
    assert received == [("/metrics", "Bearer fixture-secret")]


def test_monitoring_reports_gated_metrics_even_when_health_is_open():
    from tests.control.test_telemetry import endpoint

    def route(path, headers):
        return (200, b'{"status":"ok"}', {}) if path == "/health" else (401, b'{}', {})
    with endpoint(route) as url:
        sample = Client(url).sample()
    assert sample.phase == "unauthorized" and "token-env" in sample.error
