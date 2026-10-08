"""Generate the oracle fixture for `zig/src/core/exl3_format.zig` (the EXL3 trellis
reference decoder's test data).

The fixture's values come from TensorFold's numpy EXL3 decoder
(`tensorfold.cuda.exl3.format`, the `python-0.6` line) so the Zig decoder is validated
against the implementation the CUDA reader already trusts. The committed fixture at
`zig/tests/fixtures/exl3_oracle.bin` is byte-identical to this tool's output for the
default arguments; rerun it after any change to the numpy decoder's arithmetic.

Fixture layout (little-endian throughout):

    magic "EXL3ORAC"                        8 bytes
    u64 record_count
    per record:                             # one codebook x width combination
        u16 halves                          # width in half bits: 2*bits, odd for mul1 half steps
        u8  codebook                        # 0=3inst 1=mcg 2=mul1
        u8  reserved
        u32 tiles_k, tiles_n
        i16 trellis[tiles_k * tiles_n][16 * bits]   # every tile's words, k-major
        u32 states[tiles_k * tiles_n][256]          # the numpy decoder's exact states, k-major
    u32 spot_count
    per spot: u16 codebook, u16 state, u16 value    # the fp16 bit pattern of codebook[state]
    real block:                                     # one 128x128 Hadamard block of a live group
        u16 halves; u32 tiles_k, tiles_n, trellis_bytes
        i16 trellis[...]; u32 suh_len; f16 suh[...]; u32 svh_len; f16 svh[...]
        u32 wq_len; f16 wq[...]; u32 dq_len; f64 dq[...]

Dependencies: a `python-0.6` checkout of this repository (for the numpy oracle) and
`huggingface_hub` (for the checkpoint). Point `TENSORFOLD_SRC` at the checkout's `src`
directory when the package is not installed:

    TENSORFOLD_SRC=../TensorFold-python/src python tools/exl3_fixture.py \
        --out zig/tests/fixtures/exl3_oracle.bin

The default checkpoint is the 8.00 bpw Qwen3.5-2B pack (the geometry the native Metal
engine pins); its first run downloads ~3.2 GiB through huggingface_hub - plain HTTP
range requests against xet-backed repositories return wrong bytes outside the first
chunk, so do not substitute a manual ranged fetch.
"""

from __future__ import annotations

import argparse
import os
import json
import struct
import sys
from pathlib import Path

import numpy as np

CB = {"3inst": 0, "mcg": 1, "mul1": 2}
WIDTHS = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)   # half steps exist only with mul1
CODEBOOKS = ("3inst", "mcg", "mul1")
DEFAULT_CHECKPOINT = "UnstableLlama/Qwen3.5-2B-exl3-8.00bpw"
DEFAULT_GROUP = "model.language_model.layers.3.self_attn.q_proj"
BLOCK_TILES = 8                                     # 8 x 8 tiles = one 128x128 Hadamard block


def synthetic_records(rng: np.random.Generator) -> list[tuple[float, str, np.ndarray, np.ndarray]]:
    records = []
    for bits in WIDTHS:
        for codebook in CODEBOOKS:
            if bits in (1.5, 2.5, 3.5) and codebook != "mul1":
                continue
            trellis = rng.integers(-32768, 32767, size=(3, 2, int(16 * bits)), dtype=np.int16)
            records.append((bits, codebook, trellis, fmt.states(trellis, bits).astype(np.uint32)))
    return records


def write_synthetic(buf: bytearray, records) -> None:
    buf += struct.pack("<Q", len(records))
    for bits, codebook, trellis, states in records:
        buf += struct.pack("<HBB", int(bits * 2), CB[codebook], 0)
        buf += struct.pack("<II", trellis.shape[0], trellis.shape[1])
        buf += trellis.astype(np.int16).tobytes()
        buf += states.tobytes()


def write_codebook_spots(buf: bytearray) -> None:
    spots = [(CB[cb], state, int(fmt.codebook(cb)[state].view(np.uint16)))
             for cb in CODEBOOKS for state in range(0, 65536, 512)]
    buf += struct.pack("<I", len(spots))
    for codebook, state, value in spots:
        buf += struct.pack("<HHH", codebook, state, value)


def real_block(checkpoint_dir: str, group: str) -> None:
    with open(f"{checkpoint_dir}/model.safetensors", "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data0 = 8 + header_len

        def tensor(name: str) -> bytes:
            begin, end = header[f"{group}.{name}"]["data_offsets"]
            f.seek(data0 + begin)
            return f.read(end - begin)

        trellis_spec = header[f"{group}.trellis"]
        trellis = np.frombuffer(tensor("trellis"), dtype=np.int16).reshape(trellis_spec["shape"]).copy()
        suh = np.frombuffer(tensor("suh"), dtype=np.float16).copy()
        svh = np.frombuffer(tensor("svh"), dtype=np.float16).copy()
    bits = fmt.bits_of(trellis_spec["shape"])
    block = trellis[:BLOCK_TILES, :BLOCK_TILES]
    k = n = BLOCK_TILES * 16
    wq = fmt.unpack(block, bits, "mcg")
    dq = fmt.dequantize(block, suh[:k], svh[:n], bits, "mcg")
    if not (np.isfinite(suh[:k]).all() and np.isfinite(svh[:n]).all()):
        raise SystemExit(f" !! {group}: scales are not finite; the pack or the fetch is wrong")
    out = bytearray(struct.pack("<HIII", int(bits * 2), BLOCK_TILES, BLOCK_TILES, block.nbytes))
    out += block.astype(np.int16).tobytes()
    for name, values in (("suh", suh[:k]), ("svh", svh[:n])):
        out += struct.pack("<I", k * 2) + np.ascontiguousarray(values, dtype=np.float16).tobytes()
    out += struct.pack("<I", wq.size * 2) + wq.astype(np.float16).tobytes()
    out += struct.pack("<I", dq.size * 8) + dq.astype(np.float64).tobytes()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF repo id of an EXL3 pack")
    parser.add_argument("--revision", default=None, help="checkpoint revision (default: the pack's default)")
    parser.add_argument("--group", default=DEFAULT_GROUP, help="the projection group for the real block")
    parser.add_argument("--seed", type=int, default=7, help="the synthetic records' rng seed")
    parser.add_argument("--out", default="zig/tests/fixtures/exl3_oracle.bin")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot = snapshot_download(args.checkpoint, revision=args.revision,
                                 allow_patterns=["model.safetensors", "model.safetensors.index.json",
                                                 "config.json", "*.jinja", "tokenizer*"])
    path = hf_hub_download(args.checkpoint, "model.safetensors", revision=args.revision)

    rng = np.random.default_rng(args.seed)
    buf = bytearray(b"EXL3ORAC")
    write_synthetic(buf, synthetic_records(rng))
    write_codebook_spots(buf)
    buf += real_block(str(snapshot), args.group)
    with open(args.out, "wb") as f:
        f.write(bytes(buf))
    print(f"wrote {args.out}: {len(buf)} bytes ({len(synthetic_records(rng))} synthetic records "
          f"from seed {args.seed}, real block from {args.checkpoint} {args.group})")
    return 0


if __name__ == "__main__":
    _src = os.environ.get("TENSORFOLD_SRC")     # the numpy oracle lives on the python-0.6 line
    if _src:
        sys.path.insert(0, _src)
    try:
        from tensorfold.cuda.exl3 import format as fmt
    except ImportError:
        raise SystemExit(" !! tensorfold's numpy EXL3 decoder is required: install the python-0.6"
                         " line or set TENSORFOLD_SRC to its checkout's src directory")
    raise SystemExit(main())
