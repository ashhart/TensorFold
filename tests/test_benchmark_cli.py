"""CLI runs against local fake SSE and upload servers, with no GPU or model load."""

from __future__ import annotations

import contextlib
import io
import json
import stat
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tensorfold.benchmark import hardware, metadata, publish, receipt
from tensorfold.cli import main
from test_benchmark_receipt import valid_receipt

_TOKEN = "synthetic-contributor-token-0001"
_MODEL = "example-org/test-model"


@pytest.fixture(autouse=True)
def _fictional_public_model(monkeypatch, request):
    if not request.node.name.startswith("test_public_model_verifier_"):
        monkeypatch.setattr(publish, "verify_public_models", lambda *args, **kwargs: None, raising=False)


def _event(value):
    data = value if isinstance(value, str) else json.dumps(value)
    return f"data: {data}\n\n".encode()


@contextlib.contextmanager
def _endpoints(*, upload_status=202, redirect=False, error_stream=False):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": None})
            if self.path.endswith("/me"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"display_name":"example-contributor"}')
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            calls.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": body})
            if self.path.endswith("/submissions"):
                if redirect:
                    self.send_response(307)
                    self.send_header("Location", "/untrusted-destination")
                    self.end_headers()
                    return
                self.send_response(upload_status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                result = {"id": "synthetic-submission", "status": "pending"} if upload_status == 202 else {
                    "error": {"message": "private upstream details must not be printed"},
                }
                self.wfile.write(json.dumps(result).encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if error_stream:
                self.wfile.write(_event({"error": {"message": "private upstream details must not be printed"}}))
                return
            is_chat = "/chat/" in self.path
            first = {"delta": {"reasoning_content": "synthetic reasoning"}} if is_chat else {"text": "synthetic"}
            second = {"delta": {"content": " output"}} if is_chat else {"text": " output"}
            self.wfile.write(_event({"choices": [{"index": 0, **first}]}))
            self.wfile.flush()
            time.sleep(0.015)
            self.wfile.write(_event({"choices": [{"index": 0, **second}]}))
            final = {"choices": [{"index": 0, "finish_reason": "length", "delta": {} }],
                     "usage": {"prompt_tokens": 24, "completion_tokens": body["max_tokens"],
                               "prompt_tokens_details": {"cached_tokens": 0}},
                     "tensorfold": {"decode_tps": 42.0, "decode_s": 0.25, "prefill_s": 0.03,
                                    "token_sha": "0123456789ab"}}
            self.wfile.write(_event(final) + _event("[DONE]"))
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _run_args(base, output):
    return ["benchmark", _MODEL, "--server", base, "--backend", "cuda", "--tokens", "4",
            "--reps", "1", "--temperatures", "0", "--output", str(output)]


def test_real_cli_saves_receipt_and_never_misattributes_attached_hardware(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Client hardware/checkpoint metadata cannot describe an attached server")
    monkeypatch.setattr(hardware, "detect", forbidden)
    monkeypatch.setattr(metadata, "collect", forbidden)
    destination = tmp_path / "run.json"
    with _endpoints() as (base, calls):
        assert main(_run_args(base, destination)) == 0
    value = receipt.load(destination)
    assert len(value["samples"]) == 2
    assert value["settings"]["managed"] is False and value["settings"]["rank_count"] is None
    assert value["hardware"] == {"cpu_model": None, "system_memory_bytes": None, "memory_type": "unknown",
                                "gpus": [], "gpu_count": None, "source": "unavailable"}
    assert value["runtime"]["tensorfold_version"] == "unknown"
    assert value["model"]["revision"] is None
    assert all(sample["status"] == "ok" for sample in value["samples"])
    text = destination.read_text()
    assert "synthetic reasoning" not in text and base not in text and _TOKEN not in text
    assert len(calls) == 4 and all(call["path"].startswith("/v1/") for call in calls)
    output = capsys.readouterr().out
    assert "unranked" in output and "Receipt saved" in output


def test_saved_generated_receipt_publishes_with_explicit_consent_and_auth(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TENSORFOLD_BENCHMARK_TOKEN", _TOKEN)
    destination = tmp_path / "run.json"
    with _endpoints() as (base, calls):
        assert main(_run_args(base, destination)) == 0
        assert main(["benchmark", "publish", str(destination), "--yes", "--upload-url", base + "/api"]) == 0
    uploads = [call for call in calls if call["path"].endswith("/submissions")]
    assert len(uploads) == 1
    assert uploads[0]["authorization"] == "Bearer " + _TOKEN
    assert uploads[0]["body"] == receipt.load(destination)
    output = capsys.readouterr()
    assert "Public upload preview" in output.out
    assert "synthetic-submission: pending" in output.out
    assert _TOKEN not in output.out + output.err
    assert "authorization" not in destination.read_text().lower()


def test_noninteractive_publish_requires_yes_without_making_network_request(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    receipt.save(valid_receipt(), path)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setenv("TENSORFOLD_BENCHMARK_TOKEN", _TOKEN)
    with _endpoints() as (base, calls):
        assert main(["benchmark", "publish", str(path), "--upload-url", base]) == 1
    assert calls == []
    output = capsys.readouterr()
    assert "Public upload preview" in output.out
    assert "explicit consent" in output.err
    assert _TOKEN not in output.out + output.err


def test_interactive_decline_leaves_receipt_local(tmp_path, monkeypatch, capsys):
    class InteractiveInput(io.StringIO):
        def isatty(self):
            return True
    path = tmp_path / "run.json"
    receipt.save(valid_receipt(), path)
    monkeypatch.setattr("sys.stdin", InteractiveInput("n\n"))
    with _endpoints() as (base, calls):
        assert main(["benchmark", "publish", str(path), "--upload-url", base]) == 0
    assert calls == [] and path.exists()
    assert "Upload cancelled" in capsys.readouterr().out


def test_login_stores_600_credentials_without_token_in_receipt_or_output(tmp_path, monkeypatch, capsys):
    path = tmp_path / ".tensorfold" / "benchmarks" / "auth.json"
    monkeypatch.setattr(publish, "auth_path", lambda: path)
    monkeypatch.setattr("sys.stdin", io.StringIO(_TOKEN + "\n"))
    monkeypatch.delenv("TENSORFOLD_BENCHMARK_TOKEN", raising=False)
    with _endpoints() as (base, calls):
        assert main(["benchmark", "login", "--token-stdin", "--upload-url", base]) == 0
        assert publish.credential(base) == _TOKEN
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"api": base, "token": _TOKEN}
    assert calls[0]["authorization"] == "Bearer " + _TOKEN
    output = capsys.readouterr()
    assert _TOKEN not in output.out + output.err
    assert "example-contributor" in output.out
    assert main(["benchmark", "logout"]) == 0
    assert not path.exists()


def test_login_credentials_refuse_broad_permissions_and_other_receiver(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    monkeypatch.setattr(publish, "auth_path", lambda: path)
    monkeypatch.delenv("TENSORFOLD_BENCHMARK_TOKEN", raising=False)
    path.write_text(json.dumps({"api": "https://example.test/api", "token": _TOKEN}))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="broad file permissions"):
        publish.credential("https://example.test/api")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="different API"):
        publish.credential("https://another.example.test/api")


def test_upload_rejection_does_not_print_server_error_or_token(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    receipt.save(valid_receipt(), path)
    monkeypatch.setenv("TENSORFOLD_BENCHMARK_TOKEN", _TOKEN)
    with _endpoints(upload_status=401) as (base, _):
        assert main(["benchmark", "publish", str(path), "--yes", "--upload-url", base]) == 1
    output = capsys.readouterr()
    assert "Upload token was refused" in output.err
    assert "private upstream" not in output.out + output.err and _TOKEN not in output.out + output.err


def test_upload_does_not_forward_authorization_to_redirect(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    receipt.save(valid_receipt(), path)
    monkeypatch.setenv("TENSORFOLD_BENCHMARK_TOKEN", _TOKEN)
    with _endpoints(redirect=True) as (base, calls):
        assert main(["benchmark", "publish", str(path), "--yes", "--upload-url", base]) == 1
    assert [call["path"] for call in calls] == ["/submissions"]
    assert "HTTP 307" in capsys.readouterr().err


def test_real_cli_preserves_failed_warmups_and_samples_in_receipt(tmp_path, capsys):
    path = tmp_path / "run.json"
    with _endpoints(error_stream=True) as (base, _):
        assert main(_run_args(base, path)) == 2
    value = receipt.load(path)
    assert len(value["samples"]) == 4
    assert [sample["repeat"] for sample in value["samples"]] == [-1, 0, -1, 0]
    assert all(sample["error_code"] == "server_error" for sample in value["samples"])
    output = capsys.readouterr()
    assert "private upstream" not in output.out + output.err + path.read_text()


def test_malformed_auth_json_fails_cleanly(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.write_text("[]")
    path.chmod(0o600)
    monkeypatch.setattr(publish, "auth_path", lambda: path)
    monkeypatch.delenv("TENSORFOLD_BENCHMARK_TOKEN", raising=False)
    with pytest.raises(ValueError):
        publish.credential("https://example.test/api")


def test_cancelled_benchmark_saves_receipt_and_cannot_publish(tmp_path, monkeypatch, capsys):
    from tensorfold.benchmark import runner
    cancelled = valid_receipt()["samples"][0]
    cancelled.update(repeat=-1, status="error", error_code="cancelled")
    monkeypatch.setattr(runner, "run_suite", lambda *args, **kwargs: [cancelled])
    monkeypatch.setenv("TENSORFOLD_BENCHMARK_TOKEN", _TOKEN)
    path = tmp_path / "run.json"
    with _endpoints() as (base, calls):
        args = _run_args(base, path) + ["--publish", "--yes", "--upload-url", base]
        assert main(args) == 130
    assert calls == []
    assert receipt.load(path)["samples"][0]["error_code"] == "cancelled"
    assert "Public upload preview" not in capsys.readouterr().out


def test_attached_server_requires_explicit_backend(tmp_path, capsys):
    path = tmp_path / "run.json"
    with _endpoints() as (base, calls):
        assert main(["benchmark", _MODEL, "--server", base, "--output", str(path)]) == 1
    assert calls == [] and not path.exists()
    assert "backend" in capsys.readouterr().err.lower()


class _MetadataResponse:
    headers = {"Content-Type": "application/json"}

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return json.dumps(self.value).encode()[:limit]


@pytest.mark.parametrize("reply", [404, {"private": True}, {}], ids=("404", "private", "missing-visibility"))
def test_public_model_verifier_refuses_private_model_before_preview_and_upload(tmp_path, monkeypatch, capsys, reply):
    calls = []

    class Opener:
        def open(self, request, **kwargs):
            calls.append(request)
            if reply == 404:
                raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, io.BytesIO(b"private details"))
            return _MetadataResponse(reply)

    monkeypatch.setattr(publish.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setenv("TENSORFOLD_BENCHMARK_TOKEN", _TOKEN)
    path = tmp_path / "run.json"
    receipt.save(valid_receipt(), path)
    assert main(["benchmark", "publish", str(path), "--yes"]) == 1
    assert len(calls) == 1 and calls[0].full_url == "https://huggingface.co/api/models/" + _MODEL
    assert calls[0].get_method() == "GET" and calls[0].get_header("Authorization") is None
    output = capsys.readouterr()
    assert "Public upload preview" not in output.out
    assert "publicly accessible" in output.err
    assert _TOKEN not in output.out + output.err and "private details" not in output.out + output.err


def test_public_model_verifier_checks_model_and_drafter_without_hf_credentials(monkeypatch):
    calls = []

    class Opener:
        def open(self, request, **kwargs):
            calls.append(request)
            return _MetadataResponse({"private": False})

    monkeypatch.setattr(publish.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf-token-not-forwarded")
    value = valid_receipt()
    value["model"]["drafter_repo_id"] = "example-org/test-draft"
    assert publish.verify_public_models(value) is None
    assert {request.full_url for request in calls} == {
        "https://huggingface.co/api/models/example-org/test-model",
        "https://huggingface.co/api/models/example-org/test-draft",
    }
    assert all(request.get_header("Authorization") is None for request in calls)
    assert all(request.get_header("Cookie") is None for request in calls)


def test_publisher_tls_keeps_certificate_verification_and_no_redirects():
    import ssl
    import urllib.request
    opener = publish._opener()
    secure = next(handler for handler in opener.handlers if isinstance(handler, urllib.request.HTTPSHandler))
    assert secure._context.verify_mode == ssl.CERT_REQUIRED
    assert secure._context.check_hostname is True
    assert any(isinstance(handler, publish._NoRedirect) for handler in opener.handlers)
