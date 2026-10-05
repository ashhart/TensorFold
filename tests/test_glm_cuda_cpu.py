"""GLM-5.3-Flash's CUDA forward on the CPU, two ranks over gloo (tests/glm_cuda_cpu.py): against an independent
float64 reference (tests/glm_flash_reference.py) through prompt chunks, decode windows, rows past the dense limit
and the MTP head; and drafted windows with partial commits against serial steps, bit for bit."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton", reason="Triton's interpreter runs the CUDA path's kernels")
pytestmark = pytest.mark.torch

from glm_cuda_cpu import TOPK, V, run_ranks, write_checkpoint
from glm_flash_reference import Reference

PROMPT = [5, 917, 33, 402, 88, 731, 6, 250, 19, 664, 300, 47]
WINDOWS = [([12, 980, 7, 333], 4), ([71, 5, 820, 46], 2), ([903, 14, 255, 61], 4), ([8, 777, 120, 509], 4)]
LONG = [(83 * i + 11) % V for i in range(26)]         # a prompt chunk crossing the dense limit (19 tokens)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm53flash")
    write_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def ref(folder):
    return Reference(folder)


def drafted(rank):
    """The prompt, then windows committing ``keep`` rows each: logits of every row and the state left."""
    out = {"prompt": rank.prompt(PROMPT), "windows": [rank.window(t, keep) for t, keep in WINDOWS],
           "state": rank.state()}
    rank.fresh()
    out["serial"] = [rank.prompt(PROMPT)]
    for tokens, keep in WINDOWS:
        out["serial"] += [rank.window([t]) for t in tokens[:keep]]
    out["serial_state"] = rank.state()
    rank.fresh()
    out["long"] = rank.prompt(LONG)
    out["long_state"] = rank.state()
    rank.fresh()
    out["long_split"] = [rank.prompt(LONG[:16]), rank.prompt(LONG[16:])]
    out["long_split_state"] = rank.state()
    return out


def mtp(rank):
    """The MTP head as the engine feeds it (row j: final-normed row j, token j + 1): a prompt's rows on the prompt
    buffers (the last row's logits), then a window's rows past the dense limit (every row's logits)."""
    from tensorfold.families.glm5_next.cuda.mtp import mtp_compute

    w, st, pb, b = rank.w, rank.st, rank.pbuf, rank.buf
    rank.prompt(LONG[:20])
    pb.hin[:19], pb.ids[:19], pb.zero_first = pb.fnormed[:19], torch.tensor(LONG[1:20], dtype=torch.int32), True
    first = mtp_compute(w, st, pb, 19, nch=1, host_pos=0).float().clone()
    st.set_mtp_len(19)
    last = pb.fnormed[19:20].clone()
    rank.window(LONG[20:23])
    b.hin[:4] = torch.cat([last, b.fnormed[:3]])
    b.ids[:4], b.zero_first = torch.tensor(LONG[20:24], dtype=torch.int32), False
    return first, mtp_compute(w, st, b, 4, last_only=False, nch=1, host_pos=19).float().clone()


@pytest.fixture(scope="module")
def runs(folder):
    return run_ranks(folder, drafted)


def both(parts):
    """The two ranks' vocabulary halves side by side."""
    return torch.cat([parts[0], parts[1]], -1)


def close(got, want, what, tied=()):
    """Within bf16's drift from float64 (up to 5% here): every logit within 8% of their range of the reference's, the
    same best token where the reference's lead is over twice the row's error. Rows near a tie of pools or experts
    (``Reference.tied``) are left out: bf16 rounding may choose otherwise there."""
    got, want = got.double().reshape(-1, want.shape[-1]), want.double().reshape(-1, want.shape[-1])
    keep = [r for r in range(want.shape[0]) if r not in tied]
    assert keep, f"{what}: every row is near a tie"
    got, want = got[keep], want[keep]
    err = (got - want).abs().amax(-1)
    scale = want.abs().max().item()
    assert err.max().item() <= 0.08 * scale, f"{what}: max error {err.max().item():.4f} of {scale:.4f}"
    top = want.topk(2, -1).values
    clear = (top[:, 0] - top[:, 1]) > 2 * err
    assert torch.equal(got.argmax(-1)[clear], want.argmax(-1)[clear]), f"{what}: best tokens differ"


def test_prompt_and_windows_match_the_reference(runs, ref):
    seq = list(PROMPT)
    want, _ = ref.forward(seq)
    close(both([r["prompt"] for r in runs]), want[-1], "prompt", {0} if len(seq) - 1 in ref.tied else ())
    for w, (tokens, keep) in enumerate(WINDOWS):
        want, _ = ref.forward(seq + tokens)
        got = both([r["windows"][w] for r in runs])
        close(got, want[len(seq):], f"window {w} at {len(seq)}", {t - len(seq) for t in ref.tied})
        seq += tokens[:keep]
    assert len(seq) > TOPK + 3                      # the last windows' rows attend to their selected pools


def test_a_prompt_chunk_past_the_dense_limit_matches_the_reference(runs, ref):
    want, _ = ref.forward(LONG)
    close(both([r["long"] for r in runs]), want[-1], "26-token prompt", {0} if len(LONG) - 1 in ref.tied else ())


def test_drafted_windows_equal_serial_steps(runs):
    """Every committed row of a window has its one-row step's logits; the state left is the same, bit for bit."""
    for r in runs:
        rows = [r["prompt"]] + [row for (_, keep), w in zip(WINDOWS, r["windows"]) for row in w[:keep]]
        serial = [r["serial"][0]] + [s[0] for s in r["serial"][1:]]
        assert len(rows) == len(serial) and all(torch.equal(a, b) for a, b in zip(rows, serial))
        assert r["state"].keys() == r["serial_state"].keys()
        assert all(torch.equal(torch.as_tensor(r["state"][k]), torch.as_tensor(r["serial_state"][k]))
                   for k in r["state"])


def test_prompt_chunking_keeps_the_bits(runs):
    for r in runs:
        assert torch.equal(r["long"], r["long_split"][1])
        assert all(torch.equal(torch.as_tensor(r["long_state"][k]), torch.as_tensor(r["long_split_state"][k]))
                   for k in r["long_state"])


def test_mtp_head_matches_the_reference(folder, ref):
    ranks = run_ranks(folder, mtp)
    _, normed = ref.forward(LONG[:23])
    tied = set(ref.tied)
    want = ref.mtp(normed, LONG[1:24])
    tied |= ref.tied
    close(both([r[0] for r in ranks]), want[18], "MTP head after the prompt", {0} if 18 in tied else ())
    close(both([r[1] for r in ranks]), want[19:], "MTP head past the dense limit", {t - 19 for t in tied})
