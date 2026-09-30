"""Flash Next image features, rotary offsets and image-cache isolation on tiny CUDA weights."""
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.vision.qwen_cuda import EncodedVision


def _image(w, prompt, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    features = torch.randn((4, w.cfg.hidden), device="cuda", generator=generator, dtype=torch.bfloat16)
    positions = torch.arange(len(prompt), device="cuda", dtype=torch.int32).repeat(3, 1)
    positions[:, 4:] -= 2
    positions[:, 1:5] = torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]], device="cuda")
    return EncodedVision((1, 2, 3, 4), features, positions, -2)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_image_prefill_is_chunk_invariant_and_clears_positions(kv_dtype):
    w = _model()
    prompt = [5, 17, 17, 17, 17] + list(range(20, 55))
    image = _image(w, prompt, 9)
    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=rows, kv_dtype=kv_dtype) for rows in (16, 64)]
    tokens = []
    for engine in engines:
        first = prefill(engine, prompt, None, mtp=False, vision=image)
        assert engine.pbuf.rope_rows is None and engine.st.rope_delta == -2
        tokens.append(serial_decode(engine, first, 8, None).tokens)
    assert tokens[0] == tokens[1]
    with pytest.raises(ValueError, match="from its start"):
        prefill(engines[0], prompt, None, resume={}, vision=image)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_image_streams_match_serial_and_never_reuse_placeholder_states(kv_dtype):
    w = _model()
    prompt = [5, 17, 17, 17, 17] + list(range(20, 55))
    images = [_image(w, prompt, seed) for seed in (9, 12)]
    tower = SimpleNamespace(encode=lambda prepared, ids: prepared)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, vision=tower)
    for image in images:
        engine = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        reference = serial_decode(engine, prefill(engine, prompt, None, mtp=False, vision=image), 8, None).tokens
        stream = Stream(prompt, 8, vision=image)
        dec.admit(stream)
        while dec.live():
            dec.finish(dec.round())
        assert stream.out == reference and stream.cached == 0 and not dec.kept
    assert len(dec.free) == 2
