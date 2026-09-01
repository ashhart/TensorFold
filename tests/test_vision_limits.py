"""Image-count policy applies to the full history, with recovery after a refusal."""

import copy
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, serve_options
from tensorfold.cuda.http import make_handler as cuda_handler
from tensorfold.server.errors import RequestError
from tensorfold.server.http import make_handler as mlx_handler
from tensorfold.server.prompts import prepare_images, prepare_prompt
from tensorfold.vision.images import ImageLimits
from tests.test_server_openai_compat import FakeApp, post_json
from tests.test_vision_server import Frontend, cuda_app, image_messages, prompt_app


@pytest.mark.parametrize("limit", [None, 8])
def test_mlx_app_count_reaches_request_preparation(limit, monkeypatch):
    from tensorfold.server.app import ChatApp
    from tensorfold.server.scheduler import Scheduler
    from tests.test_lane_server import FakeTokenizer

    # Preparation needs no engine rounds; don't start the model worker or watchdog.
    monkeypatch.setattr(Scheduler, "start", lambda self: None)
    app = ChatApp(NS(vision=Frontend()), FakeTokenizer(), served_name="fixture", checkpoint_slots=0,
                  engine_factory=lambda *args, **kwargs: NS(), vision_max_images=limit)
    accepted = 4 if limit is None else 8
    prepared = prepare_prompt(app, image_messages() * accepted, [], False, None, {})
    assert len(prepared.vision.image_hashes) == accepted
    with pytest.raises(RequestError, match=f"at most {accepted}"):
        prepare_prompt(app, image_messages() * (accepted + 1), [], False, None, {})


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_configured_count_applies_to_splitting_and_decoding(backend):
    app = prompt_app(Frontend()) if backend == "mlx" else cuda_app(Frontend())
    app.image_limits = ImageLimits(max_images=8)
    messages = image_messages() * 5
    if backend == "mlx":
        prepared = prepare_prompt(app, messages, [], False, None, {})
    else:
        prepared = app.prepare({"messages": messages, "max_tokens": 2}, True)
    assert len(prepared.vision.image_hashes) == 5
    # A second server with the default policy must not inherit the first server's limit.
    with pytest.raises(RequestError, match="at most 4"):
        prepare_images(Frontend(), messages, str)


@pytest.mark.parametrize("bounds, message", [
    ({"max_total_encoded_bytes": 100}, "byte"),
    ({"max_total_pixels": 16}, "pixel"),
])
def test_higher_count_preserves_other_image_budgets(bounds, message):
    with pytest.raises(RequestError, match=message):
        prepare_images(Frontend(), image_messages() * 5, str, limits=ImageLimits(max_images=8, **bounds))


class ImageApp(FakeApp):
    """Keep MLX's real prompt preparation; replace only model/tokenizer work."""

    def __init__(self, limit):
        super().__init__()
        self.__dict__.update(vars(prompt_app(Frontend())))
        self.image_limits = ImageLimits(max_images=limit)

    def chat(self, messages, **kwargs):
        prepare_prompt(self, messages, kwargs.get("tools"), False, None, {})
        return super().chat(messages, **kwargs)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("limit", [4, 8])
def test_image_history_refusal_and_recovery_on_the_same_server(backend, stream, limit):
    app = ImageApp(limit) if backend == "mlx" else cuda_app(Frontend())
    app.image_limits = ImageLimits(max_images=limit)
    handler = (mlx_handler if backend == "mlx" else cuda_handler)(app)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    history = []
    for index in range(limit + 1):
        history.extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"read_{index}", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"read_{index}", "content": "(see attached image)"},
            *image_messages(),
        ])
    before = copy.deepcopy(history)
    hello = [{"role": "user", "content": "hello"}]
    try:
        for messages, refused in [(history[:-3], False), (history, True), (history + hello, True),
                                  (hello, False), (history[3:] + hello, False)]:
            status, payload = post_json(httpd, "/v1/chat/completions", {
                "messages": messages, "tools": tools, "stream": stream, "max_tokens": 2})
            if stream and status == 200:
                events = [json.loads(line[6:]) for line in payload.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                error = next((event["error"] for event in events if "error" in event), None)
                assert payload.endswith("data: [DONE]\n\n")
            else:
                error = json.loads(payload).get("error")
            if refused:
                assert status == (200 if backend == "mlx" and stream else 400)
                assert error["type"] == "invalid_request_error"
                assert f"at most {limit}" in error["message"]
                assert "history" in error["message"]
                assert "--vision-max-images" in error["message"]
                assert "remove" in error["message"]
            else:
                assert status == 200 and error is None
        assert history == before
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("value, vision, message", [("0", True, "positive integer"),
                                                   ("-1", True, "positive integer"),
                                                   ("8", False, "needs --vision")])
def test_bad_image_count_options_are_refused_before_checkpoint_reads(backend, value, vision, message, monkeypatch):
    from tensorfold import families

    monkeypatch.setattr(families, "read_config", lambda *args: pytest.fail("read checkpoint before refusal"))
    args = cli.build_parser().parse_args(["serve", "owner/model", "--vision-max-images", value,
                                         *(["--vision"] if vision else [])])
    with pytest.raises(ValueError, match=message):
        serve_options.check(args, NS(model_type="qwen3_5"), backend, "unused")


@pytest.mark.parametrize("flag, accepted", [([], 4), (["--vision-max-images", "8"], 8)])
def test_cuda_cli_count_reaches_request_preparation(tmp_path, monkeypatch, flag, accepted):
    from tensorfold.cuda import server
    from tests.test_cuda_admission import model_dir as _model_dir

    # The real app reads a tiny local tokenizer/config, never model weights.
    folder = _model_dir.__wrapped__(tmp_path)
    engine = NS(vision=Frontend(), eos=(0,), context_window=32)
    family = NS(title="fixture", model_type="qwen3_5", package=NS(cuda_engine=lambda *a, **k: engine))
    seen = []

    def serve(app, *_):
        # A tokenizer template need only preserve markers for our model-free processor.
        app.template = NS(render=lambda *args, **kwargs: "rendered prompt")
        prepared = app.prepare({"messages": image_messages() * accepted, "max_tokens": 2}, True)
        seen.append(len(prepared.vision.image_hashes))
        with pytest.raises(RequestError, match=f"at most {accepted}"):
            app.prepare({"messages": image_messages() * (accepted + 1)}, True)

    monkeypatch.setattr(server, "serve", serve)
    args = cli.build_parser().parse_args(["serve", str(folder), "--vision", "--no-drafts", *flag])
    assert cli._serve_cuda(args, family, folder, 32) == 0
    assert seen == [accepted]
