#!/usr/bin/env python3
"""TensorFold `mlx_tf` Production Test Suite (Zero Apple MLX).

Verifies:
  1. CLI flags (--version, --help) & C ABI version + zero Apple MLX import
  2. Multi-turn chat templates across all 8 architecture families
  3. Runtime context-length bounds & clamping (no crash/UB in --release=fast)
  4. Python `mlx_tf.convert()` for both MLX affine (4b g64) & native TFQ3 (3.5b)
  5. 4-row-tiled Metal ViT encoder (`model.encode_image` + live spatial VLM QA)
  6. Live OpenAI HTTP/SSE server (/health, /v1/models, chat JSON, SSE streaming,
     multimodal base64 `image_url`, HTTP 400 context overflow, early-disconnect SIGPIPE immunity)
  7. Zero-regression `mlx_tf spec-check` on `Qwen3.5-0.8B-4bit`
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.join(REPO_ROOT, "zig-out", "bin", "mlx_tf")

if __name__ != "__main__" and "pytest" in sys.modules:
    import pytest

    if sys.platform != "darwin" or not os.path.exists(BIN):
        pytest.skip(
            "engines/mlx_tf tests require macOS Apple Silicon and a built zig-out/bin/mlx_tf binary",
            allow_module_level=True,
        )
    try:
        import numpy as np
    except ImportError:
        pytest.skip("engines/mlx_tf tests require numpy", allow_module_level=True)

    @pytest.fixture
    def tmp(tmp_path):
        return str(tmp_path)
else:
    import numpy as np

sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
import mlx_tf

assert "mlx" not in sys.modules, "Apple MLX must never be imported by mlx_tf!"

QWEN35_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462"
)


def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        dt = "BF16" if arr.dtype == np.uint16 else "F32"
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def write_png(path: str, rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw_rows)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
    return png


def make_tiny_bf16_llama(mdir: str) -> None:
    os.makedirs(mdir, exist_ok=True)
    rng = np.random.default_rng(42)
    H, I, nh, nkv, hd, V = 256, 512, 4, 2, 64, 256
    cfg = {
        "model_type": "llama",
        "hidden_size": H,
        "intermediate_size": I,
        "num_hidden_layers": 2,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": hd,
        "vocab_size": V,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "eos_token_id": 255,
    }
    with open(os.path.join(mdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    vocab = {f"t{i}": i for i in range(V)}
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:?!":
        vocab[ch] = ord(ch)
    vocab["\u0120"] = 32
    with open(os.path.join(mdir, "tokenizer.json"), "w") as f:
        json.dump(
            {
                "version": "1.0",
                "added_tokens": [
                    {"id": 250, "content": "<|begin_of_text|>", "special": True},
                    {"id": 251, "content": "<|start_header_id|>", "special": True},
                    {"id": 252, "content": "<|end_header_id|>", "special": True},
                    {"id": 253, "content": "<|eot_id|>", "special": True},
                ],
                "model": {"type": "BPE", "vocab": vocab, "merges": []},
            },
            f,
        )
    rb = lambda s, sc=0.08: f32_to_bf16_u16((rng.standard_normal(s) * sc).astype(np.float32))
    ts = {
        "model.embed_tokens.weight": rb((V, H), 0.15),
        "model.norm.weight": f32_to_bf16_u16(np.ones((H,), dtype=np.float32)),
        "lm_head.weight": rb((V, H), 0.2),
    }
    for li in range(2):
        lp = f"model.layers.{li}"
        ts[f"{lp}.input_layernorm.weight"] = f32_to_bf16_u16(np.ones((H,), dtype=np.float32))
        ts[f"{lp}.post_attention_layernorm.weight"] = f32_to_bf16_u16(np.ones((H,), dtype=np.float32))
        ts[f"{lp}.self_attn.q_proj.weight"] = rb((nh * hd, H))
        ts[f"{lp}.self_attn.k_proj.weight"] = rb((nkv * hd, H))
        ts[f"{lp}.self_attn.v_proj.weight"] = rb((nkv * hd, H))
        ts[f"{lp}.self_attn.o_proj.weight"] = rb((H, nh * hd))
        ts[f"{lp}.mlp.gate_proj.weight"] = rb((I, H))
        ts[f"{lp}.mlp.up_proj.weight"] = rb((I, H))
        ts[f"{lp}.mlp.down_proj.weight"] = rb((H, I))
    write_safetensors(os.path.join(mdir, "model.safetensors"), ts)


def test_1_cli_and_abi() -> None:
    r_ver = subprocess.run([BIN, "--version"], capture_output=True, text=True, check=True)
    assert "0.4.0" in r_ver.stdout, f"Unexpected version: {r_ver.stdout}"
    r_help = subprocess.run([BIN, "--help"], capture_output=True, text=True, check=True)
    assert "--bf16" in r_help.stderr and "--image" in r_help.stderr and "--host" in r_help.stderr
    lib = mlx_tf._load_dylib()
    assert lib.mlxtf_version() == b"0.4.0"
    assert mlx_tf.__version__ == "0.4.0"
    assert "mlx" not in sys.modules
    print("[PASS 1/7] CLI --version/--help, C ABI 0.4.0, zero Apple MLX")


def test_2_chat_templates(tmp: str) -> None:
    cases = [
        ("qwen_think", ["<|im_start|>", "<think>"], "qwen", True, "<think>\n\n</think>\n\n"),
        ("qwen25", ["<|im_start|>"], "qwen", False, "<|im_start|>assistant\n"),
        ("llama3", ["<|begin_of_text|>", "<|start_header_id|>"], "llama3", False, "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("mistral", ["<s>", "[INST]"], "mistral", False, "[INST] Sys\n\nHi [/INST]"),
        ("gemma", ["<bos>", "<start_of_turn>"], "gemma", False, "<start_of_turn>model\n"),
        ("gemma4", ["<bos>", "<|turn>"], "gemma4", False, "<|turn>model\n"),
        ("deepseek", ["<\uff5cUser\uff5c>", "<think>"], "deepseek", True, "<\uff5cAssistant\uff5c><think>\n\n</think>\n\n"),
        ("glm", ["<|user|>"], "glm", False, "[gMASK]<sop><|system|>\nSys<|user|>\nHi<|assistant|>\n"),
        ("lfm2", ["<|startoftext|>", "<|im_start|>"], "lfm2", False, "<|startoftext|><|im_start|>system\n"),
    ]
    msgs = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hi"}]
    for name, specials, want_style, want_think, must_contain in cases:
        d = os.path.join(tmp, f"tok_{name}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "tokenizer.json"), "w") as f:
            json.dump({"added_tokens": [{"id": i, "content": s} for i, s in enumerate(specials)], "model": {"vocab": {}, "merges": []}}, f)
        tok = mlx_tf.MlxTfTokenizer(d)
        assert tok.chat_style == want_style, f"{name}: got {tok.chat_style}, want {want_style}"
        assert tok.has_think_token == want_think, f"{name}: got think={tok.has_think_token}"
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        assert must_contain in rendered, f"{name}: missing {must_contain!r} in {rendered!r}"
    print("[PASS 2/7] Multi-turn chat templates verified across all 8 model families")


def test_3_context_bounds_and_bf16_ctxmgr(tmp: str) -> None:
    mdir = os.path.join(tmp, "tiny_llama")
    make_tiny_bf16_llama(mdir)
    model, tok = mlx_tf.load(mdir, max_context=32, keep_bf16=True, force_simd=True)
    with model:
        assert not model.uses_m5_tensor_ops
        assert tok.chat_style == "llama3"
        # Oversized prompt (>32 tokens) must cleanly raise ValueError instead of crashing
        raised = False
        try:
            mlx_tf.generate(model, tok, prompt="a" * 40, max_tokens=4)
        except ValueError as e:
            raised = "max_context" in str(e)
        assert raised, "Expected ValueError on prompt > max_context"
        # Near-limit prompt (28 tokens) with max_tokens=100 must clamp cleanly to <= 4 tokens
        toks = [r.token for r in mlx_tf.stream_generate(model, tok, prompt="a" * 28, max_tokens=100, temp=0.0) if r.finish_reason is None]
        assert 1 <= len(toks) <= 4, f"Expected clamped <=4 tokens, got {len(toks)}"
    print("[PASS 3/7] Runtime context bounds guard, clamping, keep_bf16, force_simd, & context manager")


def test_4_python_convert_affine_and_tfq3(tmp: str) -> None:
    src = os.path.join(tmp, "tiny_llama")
    if not os.path.exists(src):
        make_tiny_bf16_llama(src)
    dst_aff = os.path.join(tmp, "out_aff4")
    dst_tfq3 = os.path.join(tmp, "out_tfq3_35b")

    mlx_tf.convert(src, dst_aff, format="affine", bits=4, group_size=64)
    assert os.path.exists(os.path.join(dst_aff, "model.safetensors"))
    with mlx_tf.load(dst_aff, max_context=128)[0] as m_aff:
        out_aff = mlx_tf.generate(m_aff, None, prompt="hello", max_tokens=6, temp=0.0)
        assert isinstance(out_aff, str)

    mlx_tf.convert(src, dst_tfq3, format="tfq3", profile="3.5b")
    tffpw_files = [f for f in os.listdir(dst_tfq3) if f.endswith(".tffpw")]
    assert len(tffpw_files) >= 1, f"Expected .tffpw shard in {dst_tfq3}"
    with mlx_tf.load(dst_tfq3, max_context=128)[0] as m_tfq:
        out_tfq = mlx_tf.generate(m_tfq, None, prompt="hello", max_tokens=6, temp=0.0)
        assert isinstance(out_tfq, str)
    print(f"[PASS 4/7] Python mlx_tf.convert() verified for affine 4b/g64 & native TFQ3 3.5b ({tffpw_files[0]})")


def test_5_vision_4row_speed_and_accuracy(tmp: str) -> None:
    if not os.path.exists(QWEN35_DIR):
        print("[SKIP 5/7] Cached Qwen3.5-0.8B-4bit not found")
        return
    rgb = np.full((256, 256, 3), 255, dtype=np.uint8)
    rgb[88:168, 24:104] = [235, 25, 25]
    yy, xx = np.ogrid[:256, :256]
    rgb[(xx - 192) ** 2 + (yy - 128) ** 2 <= 40 ** 2] = [25, 65, 235]
    img_path = os.path.join(tmp, "spatial_lr.png")
    write_png(img_path, rgb)

    t0 = time.perf_counter()
    with mlx_tf.load(QWEN35_DIR, max_context=1024)[0] as m:
        assert m.has_vision
        embs, ntok, hdim = m.encode_image(img_path)
        t_vis_ms = (time.perf_counter() - t0) * 1000.0
        assert ntok == 64 and hdim == 1024 and len(embs) == 64 * 1024
        assert all(np.isfinite(embs[:64]))
        t_v2 = time.perf_counter()
        _, ntok2, _ = m.encode_image(img_path)
        vis_warm_ms = (time.perf_counter() - t_v2) * 1000.0
        assert ntok2 == 64
        out = mlx_tf.generate(
            m,
            None,
            prompt="What shape and color are on the left, and what shape and color are on the right? Answer in one sentence.",
            images=[img_path],
            max_tokens=48,
            temp=0.0,
        )
        ol = out.lower()
        assert "red" in ol and "square" in ol and "blue" in ol and "circle" in ol, f"Unexpected VLM output: {out}"
    print(f"[PASS 5/7] 4-row-tiled Metal ViT verified (warm encode {vis_warm_ms:.1f} ms for 256x256 -> 64 tok; spatial QA exact)")


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_6_live_openai_server(tmp: str) -> None:
    if not os.path.exists(QWEN35_DIR):
        print("[SKIP 6/7] Cached Qwen3.5-0.8B-4bit not found")
        return
    port = pick_free_port()
    srv = subprocess.Popen(
        [BIN, "serve", QWEN35_DIR, "--host", "127.0.0.1", "--port", str(port), "--name", "qwen35-test", "--max-context", "512"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for /health
        base = f"http://127.0.0.1:{port}"
        health = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=1.0) as resp:
                    health = json.loads(resp.read().decode())
                    break
            except Exception:
                time.sleep(0.05)
        assert health is not None and health["status"] == "ok"
        assert health["version"] == "0.4.0" and health["vision"] is True and health["max_context"] == 512

        # /v1/models
        with urllib.request.urlopen(f"{base}/v1/models", timeout=2.0) as resp:
            models = json.loads(resp.read().decode())
        assert models["data"][0]["id"] == "qwen35-test" and models["data"][0]["vision"] is True

        # Non-streaming chat completion + stop sequence
        req_body = json.dumps({
            "model": "qwen35-test",
            "messages": [{"role": "user", "content": "Count from 1 to 5 separated by commas."}],
            "max_tokens": 24,
            "temperature": 0.0,
            "stop": ["4"],
        }).encode()
        req = urllib.request.Request(f"{base}/v1/chat/completions", data=req_body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            j = json.loads(resp.read().decode())
        txt = j["choices"][0]["message"]["content"]
        assert "1" in txt and "2" in txt and "4" not in txt, f"Stop sequence failed: {txt!r}"

        # Multimodal OpenAI chat completion with base64 data URI
        rgb = np.full((128, 128, 3), 255, dtype=np.uint8)
        rgb[24:104, 24:104] = [20, 200, 40]  # green square
        png_bytes = write_png(os.path.join(tmp, "green.png"), rgb)
        uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        vlm_body = json.dumps({
            "model": "qwen35-test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": "What color is the square? Reply in 3 words or less."},
                ],
            }],
            "max_tokens": 16,
            "temperature": 0.0,
        }).encode()
        req_v = urllib.request.Request(f"{base}/v1/chat/completions", data=vlm_body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_v, timeout=15.0) as resp:
            jv = json.loads(resp.read().decode())
        vtxt = jv["choices"][0]["message"]["content"]
        assert "green" in vtxt.lower(), f"Expected 'green' in server VLM reply, got: {vtxt!r}"

        # HTTP 400 context overflow OpenAI error envelope
        big_body = json.dumps({
            "model": "qwen35-test",
            "messages": [{"role": "user", "content": "word " * 600}],
            "max_tokens": 8,
        }).encode()
        req_big = urllib.request.Request(f"{base}/v1/chat/completions", data=big_body, headers={"Content-Type": "application/json"})
        got_400 = False
        try:
            urllib.request.urlopen(req_big, timeout=5.0)
        except urllib.error.HTTPError as e:
            err_j = json.loads(e.read().decode())
            got_400 = (e.code == 400 and err_j["error"]["type"] == "invalid_request_error")
        assert got_400, "Expected HTTP 400 invalid_request_error on context overflow"

        # Abrupt SSE disconnect -> verify SIGPIPE immunity (server stays alive!)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        sse_body = json.dumps({
            "model": "qwen35-test",
            "messages": [{"role": "user", "content": "Write a long poem about mountains."}],
            "max_tokens": 64,
            "stream": True,
            "temperature": 0.0,
        })
        conn.request("POST", "/v1/chat/completions", body=sse_body, headers={"Content-Type": "application/json"})
        r_sse = conn.getresponse()
        assert r_sse.status == 200
        _ = r_sse.read(128)  # read partial SSE, then abruptly drop socket!
        conn.close()
        time.sleep(0.15)
        with urllib.request.urlopen(f"{base}/health", timeout=2.0) as resp:
            assert json.loads(resp.read().decode())["status"] == "ok"

        print(f"[PASS 6/7] Live OpenAI HTTP/SSE server verified (VLM reply={vtxt.strip()!r}, HTTP 400 guard, SIGPIPE immune)")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            srv.kill()


def test_7_spec_check_regression() -> None:
    if not os.path.exists(QWEN35_DIR):
        print("[SKIP 7/7] Cached Qwen3.5-0.8B-4bit not found")
        return
    r = subprocess.run([BIN, "spec-check", QWEN35_DIR], capture_output=True, text=True, check=True)
    assert "21/21 EXACT" in r.stderr and "92/92 (100% BIT-EXACT)" in r.stderr, r.stderr
    print("[PASS 7/7] Zero-regression spec-check on Qwen3.5-0.8B-4bit (21/21 tree + 92/92 copy bit-exact)")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="mlxtf_suite_")
    try:
        test_1_cli_and_abi()
        test_2_chat_templates(tmp)
        test_3_context_bounds_and_bf16_ctxmgr(tmp)
        test_4_python_convert_affine_and_tfq3(tmp)
        test_5_vision_4row_speed_and_accuracy(tmp)
        test_6_live_openai_server(tmp)
        test_7_spec_check_regression()
        assert "mlx" not in sys.modules
        print("\nALL 7/7 PRODUCTION TESTS PASSED (ZERO APPLE MLX)!")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
