"""Shared authentication before either HTTP server dispatches a request."""

import json

from tensorfold.server import metrics


def gated(path, metrics_open=False):
    path = path.split("?", 1)[0].rstrip("/")
    if metrics_open and path in {"/metrics", "/v1/metrics"}:
        return False
    return (path == "/v1" or path.startswith("/v1/") or
            path in {"/metrics", "/tokenize", "/detokenize", "/models", "/messages",
                     "/messages/count_tokens", "/chat/completions", "/completions", "/decisions"} or
            path == "/responses" or path.startswith("/responses/") or
            path.endswith(("/chat/completions", "/completions", "/decisions", "/models")))


class Authenticated:
    """Connection reuse resets identity before every request."""

    def _authenticate(self):
        store = self.auth
        if self.command == "OPTIONS" or store is None or not store.enabled:
            return True
        self.key_label = store.match(self.headers)
        if not gated(self.path, store.metrics_open) or self.key_label is not None:
            return True
        path = self.path.split("?", 1)[0].rstrip("/")
        message = "A valid API key is required"
        if path in {"/v1/messages", "/v1/messages/count_tokens", "/messages", "/messages/count_tokens"}:
            payload = {"type": "error", "error": {"type": "authentication_error", "message": message}}
        else:
            payload = {"error": {"message": message, "type": "invalid_request_error",
                                 "param": None, "code": "invalid_api_key"}}
        self.close_connection = True
        blob = json.dumps(payload).encode()
        self.send_response(401)
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(blob)
        return False

    def parse_request(self):
        self.key_label = None
        self._auth_counted = [False]
        return super().parse_request() and self._authenticate()

    def handle_expect_100(self):
        return self._authenticate() and super().handle_expect_100()

    def send_response(self, code, message=None):
        counted = getattr(self, "_auth_counted", [False])
        if self.auth is not None and self.auth.enabled and not counted[0]:
            if code >= 200:
                counted[0] = True
                metrics.http_request(self.auth_app, self.key_label or "unauthenticated", code)
        return super().send_response(code, message)

    def log_message(self, fmt, *args):
        if self.auth is not None and self.auth.enabled:
            fmt += " key=" + (getattr(self, "key_label", None) or "unauthenticated")
        super().log_message(fmt, *args)


def handler(base, app):
    store = getattr(app, "auth", None)

    class Handler(Authenticated, base):
        auth = store
        auth_app = app

    return Handler
