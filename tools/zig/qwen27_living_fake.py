#!/usr/bin/env python3
"""Protocol-only owned-process fixture for the lesson runner; it implements no model or GPU arithmetic."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys


def main():
    args = sys.argv[1:]
    if args and args[0] != "serve" and args[0] != "slide":
        print("Q27_POCKET compared=1 untaught_equal=true logits_equal=true")
        return
    if args[:2] == ["slide", "rollback"]:
        model = Path(args[2])
        (model / "state.json").write_text('{"learned":false}')
        return
    model = Path(args[1])
    port = int(args[args.index("--port") + 1])
    graph = Path(args[args.index("--slide-graph") + 1])
    fault = args[args.index("--fake-fault") + 1] if "--fake-fault" in args else "none"

    def state():
        return json.loads((model / "state.json").read_text()).get("learned", False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, kind="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def do_GET(self):
            if self.path == "/health":
                self.send({"ok": True})
            elif self.path == "/v1/slide/graph":
                self.send({"facts": [{"id": 1, "state": "learned" if state() else "deleted"}] if state() or graph.exists() else []})
            else:
                self.send_error(404)

        def do_POST(self):
            body = self.body()
            if self.path == "/tokenize":
                self.send({"tokens": [3] * 4096 if "calibration words" in body["messages"][0]["content"] else [3, 5, 7]})
            elif self.path == "/v1/slide/learn":
                if fault == "refused":
                    self.send('event: learned\ndata: {"recalled":false}\n\ndata: [DONE]\n\n', "text/event-stream")
                    return
                (model / "state.json").write_text('{"learned":true}')
                graph.write_text('{"facts":[{"id":1,"state":"learned"}]}')
                self.send('event: learned\ndata: {"recalled":true}\n\nevent: saved\ndata: {"saved":true}\n\ndata: [DONE]\n\n', "text/event-stream")
            elif self.path == "/v1/chat/completions":
                prompt = body["messages"][0]["content"]
                subject = "Veyra" in prompt
                text = "taluno" if subject and state() else "unknown" if subject else "stable"
                ids = list(text.encode()) + [int(body["temperature"] * 10)]
                if fault == "draft" and body.get("draft"):
                    ids.append(91)
                if fault == "disturb" and state() and not subject:
                    ids.append(92)
                if fault == "import-disturb" and not subject and json.loads((model / "state.json").read_text()).get("imported"):
                    ids.append(93)
                sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
                self.send({"choices": [{"message": {"content": text}}], "tensorfold": {"token_ids": ids, "token_sha256": sha},
                           "speculative": {"rounds": 3, "drafted": 0 if fault == "no-drafts" else 4,
                                           "accepted": 0 if fault == "no-drafts" else 2}})
            else:
                self.send_error(404)

        def do_DELETE(self):
            self.body()
            if fault != "delete":
                (model / "state.json").write_text('{"learned":false}')
            self.send('event: saved\ndata: {"saved":true}\n\ndata: [DONE]\n\n', "text/event-stream")

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
