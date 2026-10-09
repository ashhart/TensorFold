"""Fixtures for the ggml block-quant kernels: rows of real GGUF tensors dequantized by libggml (ctypes to_float)."""
# usage: xpu_fixtures_ggml.py OUT_DIR; env TF_GGUF_MAIN, TF_GGUF_Q5K_XL, TF_GGUF_RECIPE_B2 (GGUF files), TF_LIBGGML_BASE
import ctypes
import os
import sys
import numpy as np
import gguf

out_dir = sys.argv[1]
env = os.environ.get
paths = {
    "main": env("TF_GGUF_MAIN", ""),
    "xl": env("TF_GGUF_Q5K_XL", ""),
    "b2": env("TF_GGUF_RECIPE_B2", ""),
}
lib = ctypes.CDLL(env("TF_LIBGGML_BASE", "libggml-base.so"))


class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p), ("from_float_ref", ctypes.c_void_p)]


lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
lib.ggml_get_type_traits.argtypes = [ctypes.c_int]

# key -> (checkpoint, tensor, rows in the slice, first row); the raw blocks are weight slices: never commit the outputs
PICK = {
    "q2_k": ("main", "blk.7.attn_q.weight", 64, 4000),
    "q4_k": ("main", "output.weight", 64, 123456),
    "iq4_xs": ("main", "blk.0.attn_gate.weight", 64, 3000),
    "iq2_xxs": ("main", "blk.0.ffn_up.weight", 64, 9000),
    "iq2_xs": ("main", "blk.0.ffn_gate.weight", 64, 9000),
    "iq2_s": ("main", "blk.0.ffn_down.weight", 32, 2000),
    "iq2_s_embd": ("main", "token_embd.weight", 64, 200000),
    "iq3_xxs": ("main", "blk.1.attn_gate.weight", 64, 3000),
    "iq3_s": ("main", "blk.1.attn_qkv.weight", 64, 5000),
    "iq1_m": ("main", "blk.13.ffn_gate.weight", 64, 9000),
    "q6_k": ("main", "blk.64.attn_q.weight", 64, 6000),
    "q5_k": ("b2", "blk.0.ssm_alpha.weight", 48, 0),
    "q8_0": ("xl", "output.weight", 64, 123456),
    "iq4_nl": ("xl", "blk.17.ffn_gate.weight", 64, 3000),
}

readers = {}


def tensors(path):
    if path not in readers:
        readers[path] = {t.name: t for t in gguf.GGUFReader(path).tensors}
    return readers[path]


rng = np.random.default_rng(7)
FRESH = ("q8_0", "iq4_nl")  # drawn from a generator of their own, as when each was made alone
os.makedirs(out_dir, exist_ok=True)
for key, (ck, name, rows, first) in PICK.items():
    if env("ONLY") and key != env("ONLY"):
        continue
    t = tensors(paths[ck])[name]
    tt = int(t.tensor_type)
    tr = lib.ggml_get_type_traits(tt).contents
    K = int(t.shape[0])
    nb = K // 256
    bytes_row = K // tr.blck_size * tr.type_size
    raw = np.asarray(t.data).reshape(-1)[first * bytes_row:(first + rows) * bytes_row].copy()
    assert raw.size == rows * bytes_row and 256 % tr.blck_size == 0
    deq = np.empty((rows, K), np.float32)
    f = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(tr.to_float)
    for i in range(rows):
        f(raw[i * bytes_row:].ctypes.data, deq[i].ctypes.data, K)
    x = (np.random.default_rng(7) if key in FRESH else rng).standard_normal(K).astype(np.float32)
    x[::97] *= 8.0
    xb = (x.view(np.uint32) >> 16).astype(np.uint16)
    xf = (xb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    y = deq.astype(np.float64) @ xf
    # file: u32 rows, K, nb, block bytes | raw blocks | f32 dequant [rows][K] | u16 bf16 x [K] | f64 y [rows]
    with open(f"{out_dir}/ggml_{key}.bin", "wb") as o:
        o.write(np.array([rows, K, nb, bytes_row // nb], np.uint32).tobytes())
        o.write(raw.tobytes() + deq.tobytes() + xb.tobytes() + y.tobytes())
    print(key, name, gguf.GGMLQuantizationType(tt).name, rows, K, raw.size)
