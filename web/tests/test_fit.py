from itertools import pairwise

from tensorfold_web import fit

GIB = 1024**3
W27 = int(4.5 * GIB)       # the 27B's prompt chunk and draft rows, from issue #50's refusal line
MAC32 = fit.mlx_capacity(32 * GIB, None)
MAC32_CEIL = fit.mlx_capacity(32 * GIB, 25 * GIB)     # macOS's default GPU working set on a 32 GB Mac mini


def test_27b_on_32gb_is_tight():
    v = fit.verdict(MAC32, int(17.2 * GIB), bytes_per_token=64_000, max_context=262_144, workspace_bytes=W27)
    assert v.level == "tight"
    assert v.budget_fraction == fit.RAISED_FRACTION
    assert 8_192 <= v.context_tokens <= 32_768
    assert v.basis == "mlx-calibrated" and v.weights_bytes == int(17.2 * GIB)


def test_gemma_on_32gb_fits():
    v = fit.verdict(MAC32, int(14.3 * GIB), bytes_per_token=40_000, max_context=262_144, workspace_bytes=int(2.5 * GIB))
    assert v.level == "fits"
    assert v.left_for_apps_bytes > 0
    assert 24_576 <= v.context_tokens <= 43_008        # the Mac mini admitted 43,008; the estimate stays under it


def test_flash_next_on_32gb_does_not_fit():
    v = fit.verdict(MAC32, int(105 * GIB), bytes_per_token=30_784, max_context=262_144)
    assert v.level == "no"
    assert v.needs_ram_gb in (192, 256)


def test_requested_fraction_moves_the_estimate():
    low = fit.verdict(MAC32, int(17.2 * GIB), bytes_per_token=64_000, max_context=262_144, workspace_bytes=W27,
                      requested_fraction=0.80)
    high = fit.verdict(MAC32, int(17.2 * GIB), bytes_per_token=64_000, max_context=262_144, workspace_bytes=W27,
                       requested_fraction=0.85)
    assert high.context_tokens > low.context_tokens


def test_27b_refused_at_default_budget_like_issue_50():
    v = fit.verdict(MAC32, int(17.2 * GIB), bytes_per_token=64_000, max_context=262_144, workspace_bytes=W27,
                    requested_fraction=0.70)
    assert v.level == "tight" and v.budget_fraction == fit.RAISED_FRACTION


def test_gpu_ceiling_caps_the_raised_budget():
    v = fit.verdict(MAC32_CEIL, int(15.0 * GIB) + int(1.1 * GIB), bytes_per_token=64_000, max_context=262_144,
                    workspace_bytes=W27)
    assert v.level == "tight" and v.budget_bytes == 25 * GIB and v.context_tokens < 16_384


def test_strict_rows_never_answer_for_another_budget():
    """The settings slider: the row labelled 70% describes 70%, even when 70% cannot run the model."""

    rows = [fit.verdict(MAC32, int(17.2 * GIB), bytes_per_token=64_000, max_context=262_144, workspace_bytes=W27,
                        requested_fraction=f, strict=True) for f in (0.70, 0.75, 0.80, 0.85)]
    assert [r.budget_fraction for r in rows] == [0.70, 0.75, 0.80, 0.85]
    assert rows[0].level == "no"
    assert all(a.context_tokens <= b.context_tokens for a, b in pairwise(rows))


def test_cuda_discrete_card_uses_its_own_memory():
    """A 64 GB PC with an 8 GB card cannot hold a 15 GB model, whatever the RAM says."""

    cap = fit.cuda_capacity(8 * GIB, unified=False)
    v = fit.verdict(cap, int(15 * GIB), bytes_per_token=64_000, max_context=262_144, workspace_bytes=W27)
    assert v.level == "no" and v.basis == "cuda-unverified"
    big = fit.cuda_capacity(96 * GIB, unified=False)
    v = fit.verdict(big, int(15 * GIB), bytes_per_token=64_000, max_context=262_144, workspace_bytes=W27)
    assert v.level == "fits"


def test_a_downloaded_native_model_is_judged_on_the_engines_own_floor():
    from tensorfold_web.catalog import Catalog

    cat = Catalog()
    nemo = next(e for e in cat.entries if e.native)
    estimate = cat.fit_inputs(nemo)
    assert estimate[2] is False
    # `tensorfold info` on the M6, 10 Oct 2026: 17.27 GiB of weights, a floor 2.00 GiB above them
    cat._facts[nemo.id] = {"weights_bytes": 18544234457, "floor_bytes": int(19.27 * GIB)}
    weights, workspace, from_engine = cat.fit_inputs(nemo)
    assert from_engine and weights == 18544234457 and abs(workspace - 2 * GIB) < 0.01 * GIB



def test_a_native_model_on_a_mac_is_judged_by_the_native_engines_own_sizing():
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.machine import Machine

    cat = Catalog()
    nemo = next(e for e in cat.entries if e.native)
    cat._facts[nemo.id] = {"weights_bytes": 18544234457, "floor_bytes": int(19.27 * GIB)}
    m = Machine(os="macOS", os_version="27.0", arch="arm64", chip="Apple M6", ram_bytes=32 * GIB,
                ram_available_bytes=20 * GIB, unified_memory=True, tensorfold_version="1.0.2", hostname="x")
    m.backend, m.gpu_ceiling_bytes = "mlx", 28672 * 1024**2
    weights, workspace, _ = cat.fit_inputs(nemo)
    v = cat.verdict_of(nemo, m, weights, workspace, fraction=0.70)        # the budget setting does not apply to it
    # measured on the M6, 10 Oct 2026: the engine serves the full 262,144-token window, 27 GB at a 128k prompt
    assert v.level == "fits" and v.context_tokens == 262144 and v.basis == "native-measured"
    assert v.budget_bytes == 28672 * 1024**2
    m.ram_bytes, m.gpu_ceiling_bytes = 16 * GIB, int(10.7 * GIB)
    assert cat.verdict_of(nemo, m, weights, workspace, None).level == "no"


def test_a_native_verdict_on_a_guessed_gpu_limit_says_so():
    from tensorfold_web import fit

    cap = fit.native_capacity(32 * fit.GIB, None)
    assert cap.basis == "native-unverified"
    v = fit.verdict(cap, 5 * fit.GIB, bytes_per_token=13_000, max_context=262_144, copies=fit.NATIVE_COPIES)
    assert v.level == "fits" and v.label == "Fits, probably" and "could not be read" in v.reason
    assert fit.native_capacity(32 * fit.GIB, 25 * fit.GIB).basis == "native-measured"
