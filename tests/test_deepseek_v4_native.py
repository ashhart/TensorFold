"""Native ownership, engine callbacks, early admission, and backend selection without model allocation."""

from __future__ import annotations

import ctypes as C
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tensorfold import families
from tensorfold.engine.exact_sampling import Sampling, choose
from tensorfold.families.deepseek_v4.cuda import capacity
from tensorfold.families.deepseek_v4.cuda.build import PIN
from tensorfold.families.deepseek_v4.cuda.engine import DeepSeekEngine
from tensorfold.families.deepseek_v4.cuda.native import NativeError, NativeLibrary, NativeSession


def test_pinned_cpu_library_iq2_donor_primitive():
    path = os.environ.get("TENSORFOLD_TEST_CPU_LIBRARY")
    if not path:
        pytest.skip("set TENSORFOLD_TEST_CPU_LIBRARY to the separately built CPU ABI library")
    api = NativeLibrary(path)
    assert api.backend() == 2
    activation = (C.c_int8 * 256)(*([1] * 256))
    # Grid zero stores eight magnitudes of 8, scale .125 gives unit weights.
    # Sign code 127 flips all eight signs; top nibble 1 multiplies scale by 3.
    for scale, word, expected in ((1.0, 0, 256.0), (0.0, 0, 0.0), (1.0, 0x0FFFFFFF, -256.0), (1.0, 0x10000000, 768.0)):
        payload = struct.pack("<e", scale) + (bytes(4) + struct.pack("<I", word)) * 8
        packed, result = C.create_string_buffer(payload), C.c_float()
        assert api.iq2_dot(packed, 66, activation, 256, C.byref(result)) == 0
        assert result.value == expected


def test_native_rpc_ownership_and_fatal_exit(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required for process boundary fixture")
    source = tmp_path / "stub.c"
    source.write_text(
        r"""
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
int tf_ds4_abi(void) { return 2; }
const char *tf_ds4_revision(void) { return "REVISION"; }
int tf_ds4_backend(void) { return 1; }
int tf_ds4_open(const char *p,int n,int t,void **out,char *err,size_t cap) {
    if (strstr(p,"crash")) _exit(7);
    if (strstr(p,"error")) { snprintf(err,cap,"fixture failure"); return 1; }
    int *v=malloc(sizeof(int)); *v=n; *out=v; return 0;
}
void tf_ds4_close(void *p) { free(p); }
int tf_ds4_vocab(void *p) { return 4; }
int tf_ds4_eos(void *p) { return 3; }
int tf_ds4_context(void *p) { return *(int *)p; }
void tf_ds4_reset(void *p) {}
int tf_ds4_cached(void *p) { return 1; }
int tf_ds4_sync(void *p,const int *ids,int n,char *err,size_t cap) { return 0; }
int tf_ds4_eval(void *p,int id,char *err,size_t cap) { return 0; }
int tf_ds4_logits(void *p,float *out,int n) {
    for(int i=0;i<n;i++) out[i]=(float)i; return n;
}
int tf_ds4_encode(void *p,const char *s,int rendered,int **out) {
    *out=malloc(sizeof(int)); **out=rendered?2:1; return 1;
}
void tf_ds4_free(void *p) { free(p); }
char *tf_ds4_token_text(void *p,int id,size_t *len) {
    *len=2; return strdup("ok");
}
int tf_ds4_encode_bytes(void *p,const char *s,size_t bytes,int rendered,int **out) {
    *out=malloc(bytes*sizeof(int));
    for(size_t i=0;i<bytes;i++) (*out)[i]=((unsigned char)s[i])%4;
    return (int)bytes;
}
int tf_ds4_iq2_dot(void *p,int b,void *a,int n,float *out) { return 1; }
""".replace("REVISION", PIN).replace("#include <stddef.h>", "#include <stddef.h>\n#include <stdio.h>")
    )
    library = tmp_path / "stub.so"
    subprocess.run([cc, "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    with NativeSession(library=library, model_path="normal", context=8, timeout=10) as session:
        session.reset()
        assert session.sync([0, 1]) == 1
        np.testing.assert_array_equal(session.logits(), np.arange(4, dtype=np.float32))
        session.eval(2)
        np.testing.assert_array_equal(session.eval_logits(1), np.arange(4, dtype=np.float32))
        assert session.encode("text") == [0, 1, 0, 0]
        assert session.encode("abc\0xyz", rendered=True) == [1, 2, 3, 0, 0, 1, 2]
        assert session.token_text(2) == b"ok"
    assert session._closed
    with pytest.raises(NativeError, match="fixture failure"):
        NativeSession(library=library, model_path="error", context=8, timeout=10)
    with pytest.raises(NativeError, match="exited"):
        NativeSession(library=library, model_path="crash", context=8, timeout=10)
    # The parent remains usable after a C _exit() in a preceding session.
    with NativeSession(library=library, model_path="normal", context=8, timeout=10) as session:
        assert session.vocab_size == 4
    wrong = tmp_path / "wrong-abi.so"
    source.write_text(source.read_text().replace("tf_ds4_abi(void) { return 2;", "tf_ds4_abi(void) { return 0;"))
    subprocess.run([cc, "-shared", "-fPIC", str(source), "-o", str(wrong)], check=True)
    with pytest.raises(NativeError, match="ABI/revision mismatch"):
        NativeLibrary(wrong)


class Session:
    vocab_size = 4
    eos = 3

    def __init__(self):
        self.pos = 0
        self.closed = False
        self.calls = []
        self.cached = 0

    def reset(self):
        self.calls.append("reset")

    def sync(self, tokens):
        self.pos = len(tokens)
        self.calls.append(("sync", list(tokens)))
        return self.cached

    def logits(self):
        return np.array([0, 1, 4, 7 if self.pos >= 3 else 2], dtype=np.float32)

    def eval(self, token):
        self.calls.append(("eval", token))
        self.pos += 1

    def close(self):
        self.closed = True


def engine(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "gguf"},
                "gguf_file": str(tmp_path / "target.gguf"),
                "native_library": str(tmp_path / "native.so"),
            }
        )
    )
    session = Session()
    instance = DeepSeekEngine(
        tmp_path,
        no_drafts=True,
        context=8,
        _admission=lambda *_: {"context_window": 8, "cache_slots": 8},
        _session_factory=lambda **_: session,
    )
    return instance, session


def test_serial_eos_callbacks_and_repeat_request(tmp_path):
    instance, session = engine(tmp_path)
    out = []
    stats = instance.generate([0, 1], 4, None, lambda ids: out.extend(ids))
    assert out == [2, 3]
    assert session.calls == ["reset", ("sync", [0, 1]), ("eval", 2), ("eval", 3)]
    assert stats["generated"] == 2 and stats["drafts"] is False
    out.clear()
    instance.generate([0, 1], 1, None, lambda ids: out.extend(ids))
    assert out == [2]
    assert session.calls.count("reset") == 1
    instance.close()
    assert session.closed


def test_zero_tokens_callback_cancel_ignore_eos_and_capacity(tmp_path):
    instance, session = engine(tmp_path)
    assert instance.generate([0], 0, None, lambda _: pytest.fail("callback"))["generated"] == 0
    assert session.calls == []
    out = []
    instance.generate([0, 1], 4, None, lambda ids: out.extend(ids) or True)
    assert out == [2]
    out.clear()
    instance.generate([0, 1], 3, None, lambda ids: out.extend(ids), stop_eos=False)
    assert out == [2, 3, 3]
    with pytest.raises(ValueError, match="capacity"):
        instance.generate([0] * 7, 2, None, lambda _: None)
    instance.close()


def test_shared_keyed_sampling_and_failure_cleanup(tmp_path):
    instance, session = engine(tmp_path)
    sampling = Sampling(123, temperature=0.8, top_k=3, top_p=0.9)
    expected = choose(session.logits(), np.arange(4), 1, sampling)
    out = []
    instance.generate([0], 1, sampling, lambda ids: out.extend(ids))
    assert out == [expected]

    def fail(_):
        raise RuntimeError("native failure")

    session.eval = fail
    with pytest.raises(RuntimeError, match="native failure"):
        instance.generate([0], 1, None, lambda _: pytest.fail("uncommitted callback"))
    assert session.closed


def test_bad_options_refuse_before_admission_or_native_loading(tmp_path):
    def forbidden(*_, **__):
        pytest.fail("allocation/admission reached")

    for options in ({"tp": 2}, {"parallel": 2}, {"drafter": "legacy-mtp"}, {"rank": 1}):
        with pytest.raises(ValueError):
            DeepSeekEngine(tmp_path, _admission=forbidden, _session_factory=forbidden, **options)


def test_dspark_uses_keyed_target_sampler_and_committed_callbacks(tmp_path):
    instance, session = engine(tmp_path)
    instance.drafter = tmp_path / "draft.gguf"
    instance.drafts_enabled = True
    sampling = Sampling(123, temperature=0.8, top_k=3, top_p=0.9)
    positions = []

    def generate(prompt, budget, sample, emit, draft, stop_eos):
        assert prompt == [0] and budget == 3 and draft and stop_eos
        for position in (1, 2):
            logits = session.logits()
            token = sample(logits)
            assert token == choose(logits, np.arange(4), position, sampling)
            positions.append(position)
            if emit([token]):
                break
        return {"generated": len(positions), "cached": 0, "drafts": draft}

    session.generate_draft = generate
    out = []
    stats = instance.generate([0], 3, sampling, lambda ids: out.extend(ids) or len(out) == 2)
    assert stats["generated"] == 2 and positions == [1, 2] and len(out) == 2
    assert session.calls == []  # No serial prefill/eval path mixed into the bank.


def test_native_sync_owns_prefix_reuse_and_reports_cached_tokens(tmp_path):
    instance, session = engine(tmp_path)
    instance.generate([0, 1], 1, None, lambda _: None)
    session.calls.clear()
    session.cached = 2
    stats = instance.generate([0, 1, 2, 0], 1, None, lambda _: None)
    assert session.calls[0] == ("sync", [0, 1, 2, 0])
    assert "reset" not in session.calls and stats["cached"] == 2
    session.calls.clear()
    session.cached = 0
    stats = instance.generate([1, 0], 1, None, lambda _: None)
    assert session.calls[0] == ("sync", [1, 0]) and stats["cached"] == 0
    session.cached = 3
    with pytest.raises(RuntimeError, match="cached prefix"):
        instance.generate([0, 1], 1, None, lambda _: None)
    instance.close()


GIB = 1 << 30


def model(tmp_path):
    config = {
        "model_type": "deepseek_v4",
        "quantization_config": {"quant_method": "gguf"},
        "gguf_file": "/mock.gguf",
        "native_library": "/mock.so",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "descriptor.json").write_text(json.dumps({"reserve_gib": 2}))
    return {
        "aligned_artifact_extra_bytes": GIB,
        "source_size": 10 * GIB,
        "arch": {"deepseek4.context_length": 262144},
        "header_sha256": "header",
        "source_identity": {},
    }


def test_admitted_context_counts_growth_once(tmp_path):
    report = model(tmp_path)
    with (
        patch.object(capacity, "inspect_inputs", return_value=report),
        patch.object(capacity, "estimate", return_value={"graph_bytes": 6 * GIB, "snapshot_bytes": GIB}),
        patch.object(capacity, "available", return_value=32 * GIB),
    ):
        plan = capacity.admit(tmp_path, 262144, True)
    assert plan["context_window"] == plan["cache_slots"] == 262144
    assert plan["required_bytes"] == 29 * GIB
    assert plan["snapshot_reserve_bytes"] == GIB
    assert plan["companion_growth_bytes"] == 2 * GIB


def test_refusal_never_starts_session_and_preserves_explicit_context(tmp_path):
    report = model(tmp_path)
    with (
        patch.object(capacity, "inspect_inputs", return_value=report),
        patch.object(capacity, "estimate", return_value={"graph_bytes": 6 * GIB, "snapshot_bytes": GIB}),
        patch.object(capacity, "available", return_value=20 * GIB),
        pytest.raises(ValueError, match="cannot fit.*262144"),
    ):
        DeepSeekEngine(tmp_path, context=262144, _session_factory=lambda **_: pytest.fail("model load reached"))
    with (
        patch.object(capacity, "available", side_effect=ValueError("memory unavailable")),
        patch.object(capacity, "estimate", side_effect=AssertionError("native reached")),
        pytest.raises(ValueError, match="memory unavailable"),
    ):
        capacity.admit(tmp_path, 262144, True)


def _write_config(tmp_path: Path, config: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


GGUF_IQ2XXS = {
    "model_type": "deepseek_v4",
    "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "IQ2_XXS"},
}


GGUF_Q2K = {
    "model_type": "deepseek_v4",
    "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "Q2_K"},
}


GGUF_Q8_0 = {"model_type": "deepseek_v4", "quantization_config": {"quant_method": "gguf", "bits": 8, "format": "Q8_0"}}


GGUF_MIXED = {
    "model_type": "deepseek_v4",
    "quantization_config": {"quant_method": "gguf", "bits": 2.5625, "format": "mixed"},
}


DS_FAMILY = families.families()["deepseek_v4"]


def test_deepseek_v4_registers_cuda_backend():
    assert set(families.backends_of(DS_FAMILY)) == {"mlx", "cuda"}
    assert callable(DS_FAMILY.package.cuda_engine)


def test_cuda_drafter_selection_preserves_local_gguf_and_excludes_mlx(tmp_path, monkeypatch):
    from tensorfold import hub
    from tensorfold.cli import _drafter

    mlx = tmp_path / "mlx-drafter"
    mlx.mkdir()
    monkeypatch.setattr(hub, "cached", lambda _: mlx)
    monkeypatch.setattr(hub, "_cached_weights_complete", lambda _: True)
    assert _drafter(DS_FAMILY, "auto", "cuda") == ""
    assert _drafter(DS_FAMILY, "auto", "mlx") == str(mlx)
    gguf = tmp_path / "DSpark.gguf"
    gguf.touch()
    assert _drafter(DS_FAMILY, str(gguf), "cuda") == str(gguf)


def test_deepseek_v4_declares_backend_specific_quant_methods():
    assert DS_FAMILY.package.QUANT_METHODS == {"mlx": ("mlx",), "cuda": ("gguf",)}


@pytest.mark.parametrize(
    "name,config",
    [
        ("IQ2_XXS", GGUF_IQ2XXS),
        ("Q2_K", GGUF_Q2K),
        ("Q8_0", GGUF_Q8_0),
        ("mixed", GGUF_MIXED),
    ],
)
def test_gguf_quant_declared_on_cuda(name, config):
    families.require_readable(DS_FAMILY, config, "cuda")


@pytest.mark.parametrize(
    "name,config",
    [
        ("IQ2_XXS", GGUF_IQ2XXS),
        ("Q2_K", GGUF_Q2K),
        ("Q8_0", GGUF_Q8_0),
        ("mixed", GGUF_MIXED),
    ],
)
def test_gguf_quant_refused_on_mlx(name, config):
    with pytest.raises(ValueError, match="does not read"):
        families.require_readable(DS_FAMILY, config, "mlx")


def test_no_vision_or_pro_family_registered():
    # issue #14 asked for DeepSeek-V4-flash-vision-exp and DeepSeek-V4-Pro; neither is registered.
    available = families.families()
    assert "deepseek_v4_flash_vision_exp" not in available
    assert "deepseek_v4_pro" not in available
    assert "deepseek_v4_vision" not in available


def test_detect_refuses_unknown_deepseek_model_types(tmp_path):
    for model_type in ("deepseek_v4_flash_vision_exp", "deepseek_v4_pro", "deepseek_v4_vision"):
        folder = _write_config(tmp_path / model_type.replace("/", "_"), {"model_type": model_type})
        with pytest.raises(ValueError, match="no recipe"):
            families.detect(folder)


def test_gguf_check_requires_provenance_without_mlx(tmp_path):
    from tensorfold.families import deepseek_v4

    folder = _write_config(tmp_path / "gguf", GGUF_IQ2XXS)
    with pytest.raises(ValueError, match="descriptor"):
        deepseek_v4.check(folder)
    (folder / "descriptor.json").write_text(json.dumps({"source": "candidate.gguf"}))
    deepseek_v4.check(folder)
