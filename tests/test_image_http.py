import json
from io import BytesIO
from types import SimpleNamespace

from tensorfold.server.http import make_handler
from tests.http_fakes import post


class VisionApp:
    served_name = "vision"
    model_ids = ["vision"]
    max_batch_size = 1
    exact_mode = {"mode": "exact"}
    accepts_images = True
    accepts_raw_prompt = True
    accepts_sampling = False
    accepts_cancellation = False
    streams_prose_with_tools = False

    def __init__(self):
        self.prepared = None
        self.vision_prompt = None

    def prepare_image(self, messages, image, **kwargs):
        self.prepared = (messages, image, kwargs)
        return SimpleNamespace(input_ids=[1, 2], image_digest=image.sha256)

    def chat(self, messages, *, vision_prompt=None, **kwargs):
        self.vision_prompt = vision_prompt
        return {
            "content": "seen",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "cached_tokens": 0,
            "completion_tokens": 1,
        }


def _data_url():
    from tests.test_image_inputs import png_data_url

    return png_data_url()


def test_image_is_prepared_before_chat_and_forwarded_as_request_metadata():
    app = VisionApp()
    status, response = post(
        app,
        {
            "model": "vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url()}},
                        {"type": "text", "text": "Describe it."},
                    ],
                }
            ],
        },
    )
    assert status == 200
    assert json.loads(response)["choices"][0]["message"]["content"] == "seen"
    assert app.prepared is not None
    assert app.vision_prompt is not None


def test_invalid_image_fails_as_json_even_when_streaming_was_requested():
    app = VisionApp()
    status, response = post(
        app,
        {
            "model": "vision",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,invalid"},
                        }
                    ],
                }
            ],
        },
    )
    assert status == 400
    assert response.startswith('{"error"')
    assert "data:" not in response
    assert app.prepared is None


def test_request_log_redacts_image_payload(tmp_path, monkeypatch):
    from tensorfold.server import http

    request_log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(http, "_REQUEST_LOG", str(request_log))
    app = VisionApp()
    status, _ = post(
        app,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url()}},
                    ],
                }
            ]
        },
    )
    assert status == 200
    logged = request_log.read_text()
    assert "<redacted>" in logged
    assert "base64," not in logged


def test_oversized_request_body_is_rejected_before_reading():
    incoming = (
        b"POST /v1/chat/completions HTTP/1.0\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 999999999\r\n\r\n"
    )

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *args):
            return BytesIO(incoming)

        def sendall(self, data):
            self.output.extend(data)

    connection = Connection()
    make_handler(VisionApp())(connection, ("127.0.0.1", 0), None)
    headers, response = bytes(connection.output).split(b"\r\n\r\n", 1)
    assert headers.split()[1] == b"400"
    assert "must not exceed" in response.decode()
