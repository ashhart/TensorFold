"""Native Qwen admission and real-model checks skip cleanly without their compiled tools, weights or MLX."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = Path(__file__).resolve().parents[1]


def program(name):
    path = ROOT / "zig-out" / ("native/bin" if name == "tensorfold-native" else "bin") / name
    if not path.is_file():
        pytest.skip(f"build {name} first")
    return path


def checkpoint():
    mlx = pytest.importorskip("mlx.core")
    if not mlx.metal.is_available():
        pytest.skip("Apple Metal is unavailable")
    path = os.environ.get("TENSORFOLD_QWEN35_MODEL")
    if not path or not (Path(path) / "model.safetensors").is_file():
        pytest.skip("TENSORFOLD_QWEN35_MODEL must name the pinned MLX affine 4-bit checkpoint")
    return Path(path)


def test_native_qwen_family_is_advertised():
    result = subprocess.run([program("tensorfold-native"), "capabilities", "--json"], check=True, capture_output=True, text=True)
    caps = json.loads(result.stdout)
    assert caps["families"]["qwen3_5"] == ["mlx-q4g64"]
    assert caps["families"]["nemotron_h"] == ["mlx-q4g64"]


def test_qwen_generated_kernels_are_current():
    subprocess.run([sys.executable, ROOT / "tools/zig/gen_qwen35_kernels.py", "--check"], check=True)


def test_native_qwen_operations(tmp_path):
    model, binary = checkpoint(), program("tf-qwen35-check")
    subprocess.run([sys.executable, ROOT / "tools/zig/capture_qwen35.py", model, tmp_path], check=True)
    subprocess.run([binary, model, tmp_path], check=True)


def test_native_qwen_windows_and_committed_states():
    subprocess.run([program("tf-qwen35-exact"), checkpoint()], check=True)


def test_native_qwen_server_turns_and_concurrent_sampling(tmp_path):
    model = checkpoint()
    from tokenizers import Tokenizer
    from tensorfold.cuda.chat_template import ChatTemplate

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log = (tmp_path / "server.log").open("w")
    process = subprocess.Popen([program("tensorfold-native"), "serve", model, "--name", "bench",
        "--port", str(port), "--parallel", "4", "--context", "2048", "--no-thinking"], stdout=log, stderr=log)

    def request(body):
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    try:
        for _ in range(300):
            assert process.poll() is None, (tmp_path / "server.log").read_text()
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            pytest.fail("native server did not become ready")
        tools = [{"type": "function", "function": {"name": "add", "description": "Add two integers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                           "required": ["a", "b"]}}}]
        conversations = [
            ([{"role": "user", "content": "Name two primary colors."}], False, None),
            ([{"role": "user", "content": "Explain why 17 plus 25 is 42."}], True, None),
            ([{"role": "user", "content": "Add 2 and 3."}, {"role": "assistant", "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"}}]},
              {"role": "tool", "tool_call_id": "call_1", "content": "5"},
              {"role": "user", "content": "Explain that result."}], False, tools),
            ([{"role": "system", "content": "Use short answers."}, {"role": "developer", "content": "Use plain English."},
              {"role": "user", "content": "What is addition?"}, {"role": "assistant", "content": "It combines numbers."},
              {"role": "system", "content": "Now include an example."}, {"role": "user", "content": "Show one."}], False, None),
            ([{"role": "user", "content": "A small public example. " * 160},
              {"role": "assistant", "content": "This earlier answer was rewritten."},
              {"role": "user", "content": "Count the words in a small public example."}], False, None),
        ]
        tok = Tokenizer.from_file(str(model / "tokenizer.json"))
        template = ChatTemplate(model)
        bodies = []
        for i, (messages, thinking, offered) in enumerate(conversations):
            body = dict(model="bench", messages=messages, max_tokens=32 + i * 4, ignore_eos=True,
                temperature=0.7, top_k=20 if i % 2 else 0, top_p=0.95, min_p=0.05, seed=1234 + i,
                chat_template_kwargs={"enable_thinking": thinking})
            if offered:
                body.update(tools=offered, tool_choice="auto")
            bodies.append(body)
        solo = [request(body) for body in bodies]
        plain = [request(dict(body, draft=False)) for body in bodies]
        with ThreadPoolExecutor(max_workers=5) as pool:
            concurrent = list(pool.map(request, bodies))
        for body, a, b, concurrent_reply, (_, thinking, offered) in zip(bodies, solo, plain, concurrent, conversations):
            assert a["tensorfold"]["token_sha"] == b["tensorfold"]["token_sha"] == concurrent_reply["tensorfold"]["token_sha"]
            assert a["tensorfold"]["engine"] == "lanes" and a["tensorfold"]["drafts"]
            assert a["tensorfold"]["enable_thinking"] == thinking
            assert a["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            rendered = template.render(body["messages"], tools=offered, enable_thinking=thinking)
            assert a["usage"]["prompt_tokens"] == len(tok.encode(rendered, add_special_tokens=False).ids)
            assert a["usage"]["completion_tokens"] == body["max_tokens"]
        invalid = [dict(bodies[0], max_tokens=2048), dict(bodies[2], tool_choice="required"),
            dict(bodies[0], response_format={"type": "json_object"}),
            dict(bodies[0], messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/fixture.png"}}]}])]
        for body in invalid:
            with pytest.raises(urllib.error.HTTPError) as caught:
                request(body)
            assert caught.value.code == 400
        assert request(bodies[0])["tensorfold"]["token_sha"] == solo[0]["tensorfold"]["token_sha"]
        normal = dict(model="bench", messages=[{"role": "user", "content": "Reply with exactly one word: amber."}],
                      max_tokens=64, temperature=0, chat_template_kwargs={"enable_thinking": False})
        reply = request(normal)
        assert reply["choices"][0]["finish_reason"] == "stop"
        assert "<|im_end|>" not in reply["choices"][0]["message"]["content"]
        repeated = request(dict(normal, draft=False))
        assert repeated["tensorfold"]["token_sha"] == reply["tensorfold"]["token_sha"]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
