"""The GGUF fast prefill's Q8_1 activation buffer covers every byte Gufo's kernels touch, on every route (no GPU).

``tf_gguf_q8_1_bytes`` (capi.hip) sizes the buffer that ``tf_gguf_prefill_linear`` quantizes into and that
``LaunchBatchedQuantGEMMPreQuantized`` reads. This test models, from the vendored sources, the end (exclusive byte
offset) of every access to that buffer and checks it against the allocation for rows 1..4096 at the 27B's K values:

* quantizer writes (``StoreQ8ActLane``, ``ZeroQ8ActTailKernel``): ceil(rows/16) tiles of 576 bytes per 32-wide K block,
  then the 16-float-per-tile sum sidecar. Exactly ``QuantizedActivationBytes``.
* K-quant/IQ WMMA kernel and the wave64 route (``WKQuantA8BlockedWmmaGEMMKernel``): token tiles bounded by
  ``wanted < num_act_tiles``, sidecar read at the same tile index. Exactly ``QuantizedActivationBytes``.
* Q8_0 W8A8 kernel (``W8A8BlockedWmmaGEMMKernel``, taken when the wave64 route declines: rows < 96, m < 1024,
  k < 1024 or k % 256): every block stages all BN/16 token tiles with no bound on the tile count, so it reads
  ceil(rows/BN) * BN/16 tiles. BN is 16 or 32 (rows <= 8, m < 3k or m >= 3k), 64 (rows < 96) or 128 (rows >= 96).

The constants and branches are read from the sources, so a vendor update that changes a tile width or drops a
bound fails here rather than on the GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

GGUF = Path(__file__).resolve().parents[1] / "src" / "tensorfold" / "cuda" / "gguf"
KERNELS = GGUF / "vendor/src/models/qwen/hip/kernels"
KS = (5120, 6144, 17408, 2048, 4096, 8192)                    # the 27B's K values, then general ones
ROWS = range(1, 4097)


def _int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    assert match, pattern
    return int(match.group(1))


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i)]


CAPI = (GGUF / "capi.hip").read_text()
HPP = (KERNELS / "prefill_quant_gemm.hpp").read_text()
HIP = (KERNELS / "prefill_quant_gemm.hip").read_text()
WAVE64 = (KERNELS / "prefill_quant_wave64.hip").read_text()

TILE_TOKENS = _int(r"kQ8ActTileTokens = (\d+);", HPP)
TILE_BYTES = _int(r"kQ8ActTileBytes = (\d+);", HPP)
SUM_TILE_BYTES = TILE_TOKENS * 4                              # kQ8ActSumTileBytes = kQ8ActTileTokens * sizeof(float)
SLACK_TILES = _int(r"kSlackTiles = (\d+)", CAPI)
LAUNCHER = _between(HIP, "void LaunchBatchedQuantGEMMPreQuantized(", "void LaunchBatchedDualQuantGEMMPreQuantized(")
W8A8 = _between(HIP, "__global__ void W8A8BlockedWmmaGEMMKernel(", "TiledBatchedQ8_0GEMMKernel")


def _tiles(rows: int) -> int:
    return -(-rows // TILE_TOKENS)


def quantized_bytes(rows: int, k: int) -> int:
    """Gufo's QuantizedActivationBytes: Q8ActTiledBytes + Q8ActSumBytes (prefill_quant_gemm.hpp:44-57)."""

    return _tiles(rows) * (k // 32) * (TILE_BYTES + SUM_TILE_BYTES)


def allocated(rows: int, k: int, slack_tiles: int = SLACK_TILES) -> int:
    """tf_gguf_q8_1_bytes (capi.hip): the payload, the sidecar, then ``slack_tiles`` tiles per K block."""

    return quantized_bytes(rows, k) + slack_tiles * (k // 32) * TILE_BYTES


def q8_0_tile_widths(rows: int) -> tuple[int, ...]:
    """BN of the W8A8 kernels LaunchBatchedQuantGEMMPreQuantized can launch for Q8_0 at ``rows`` (any m)."""

    if rows <= 8:
        return (16, 32)                                        # m < 3k / m >= 3k
    return (64,) if rows < 96 else (128,)                      # rows >= 96 only when the wave64 route declines


def accesses(rows: int, k: int) -> dict[str, int]:
    """End (exclusive byte offset) of each route's accesses to the Q8_1 buffer."""

    nb = k // 32
    ends = {"quantize writes": quantized_bytes(rows, k), "K-quant/IQ + wave64 reads": quantized_bytes(rows, k)}
    for bn in q8_0_tile_widths(rows):
        ends[f"Q8_0 W8A8 BN={bn} reads"] = -(-rows // bn) * (bn // TILE_TOKENS) * nb * TILE_BYTES
    return ends


def gaps(slack_tiles: int) -> list[tuple[str, int, int, int, int]]:
    return [(route, rows, k, end, allocated(rows, k, slack_tiles))
            for k in KS for rows in ROWS for route, end in accesses(rows, k).items()
            if end > allocated(rows, k, slack_tiles)]


def test_model_matches_the_sources():
    assert (TILE_TOKENS, TILE_BYTES) == (16, 576)
    assert re.search(r"kTileBytes = 576", CAPI)
    assert "QuantizedActivationBytes(rows, k) + kSlackTiles * (k / 32) * kTileBytes" in CAPI
    # the launcher's Q8_0 branches and tile widths
    assert "batch <= 8" in LAUNCHER and "batch >= 96" in LAUNCHER and "m >= (3 * k)" in LAUNCHER
    widths = {int(w) for w in re.findall(r"W8A8BlockedWmmaGEMMKernel<kBM, (\d+),", LAUNCHER)}
    widths |= {int(w) for w in re.findall(r"constexpr int kBN = (\d+);", LAUNCHER)}
    assert widths == {16, 32, 64, 128}
    assert "type != core::GgmlType::kQ8_0" in LAUNCHER        # every other type: the bounded K-quant kernel
    assert "TryLaunchQuantPrefillWave64" in LAUNCHER and "batch < 96" in WAVE64
    # the Q8_0 kernel stages one tile per wave with no tile-count bound; the K-quant kernel bounds it
    assert "const bool b_live = b_tile < kTokTiles;" in W8A8
    assert "(wanted < num_act_tiles)" in HPP
    # output stores are guarded by tok < batch and r < m in both kernels, so y (rows, m) is never overrun
    assert W8A8.count("if (tok < batch && r < m)") == 1 and HPP.count("if (tok < batch && r < m)") == 1
    assert W8A8.count("t0 + 16 <= batch && r0 + 16 <= m") == 1 and HPP.count("t0 + 16 <= batch && r0 + 16 <= m") == 1


def test_every_route_fits_the_allocation():
    assert gaps(SLACK_TILES) == []
    margin = min((allocated(r, k) - end) // (k // 32) for k in KS for r in ROWS for end in accesses(r, k).values())
    assert margin >= TILE_BYTES                               # at least one tile per K block to spare (now: 2)


@pytest.mark.parametrize("slack", [0, 5])
def test_negative_control_smaller_slack_fails(slack):
    found = gaps(slack)
    assert found and all(route.startswith("Q8_0 W8A8") for route, *_ in found)
    if slack == 0:                                            # the old exact-size buffer: Strix's illegal access
        assert ("Q8_0 W8A8 BN=128 reads", 96, 5120) in {g[:3] for g in found}


def test_the_least_sufficient_slack_is_six_tiles():
    assert gaps(6) == [] and gaps(5) != []
    assert SLACK_TILES >= 6
