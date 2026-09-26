"""The server's incremental decoder streams exactly the text that decoding everything at once gives."""

import random
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")

from tensorfold.cuda.server import StreamDecoder

def _tokenizer() -> Path:
    from tensorfold import hub

    found = hub.cached("Vontra/Qwen3.8-27B-MLX-4bit")
    return found / "tokenizer.json" if found is not None else Path("/nonexistent")


TOKENIZER = _tokenizer()


@pytest.mark.skipif(not TOKENIZER.exists(), reason="needs the Qwen3.8-27B tokenizer (tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit)")
def test_stream_text_matches_full_decode():
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    text = ("def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
            "Matrix multiplication 矩阵乘法 uses GPUs 🚀🚀 — naïve café, 数学 and emoji 👩‍💻.\n") * 3
    ids = tok.encode(text, add_special_tokens=False).ids + [248044, 11, 12]
    eos = (248044,)
    rng = random.Random(0)
    for _ in range(20):
        stream = StreamDecoder(tok, eos)
        i = 0
        while i < len(ids):
            step = rng.randint(1, 9)
            shown = stream.add(ids[i:i + step])
            i += step
            kept = [t for t in ids[:i] if t not in eos]
            full = tok.decode(kept, skip_special_tokens=False)
            assert full.startswith(shown)                  # never shows text the full decode lacks
            assert "�" not in shown
        assert stream.final() == tok.decode([t for t in ids if t not in eos], skip_special_tokens=False)
