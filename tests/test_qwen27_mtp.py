"""The Qwen3.8-27B MTP head as a draft proposer (``families.qwen3_5.mtp``).

A tiny random head checks the loader (quantization, the checkpoint's 1 + w norms) and the proposer's bookkeeping:
which rows the head has absorbed at which positions, the chained drafts it forgets, the history a resumed prompt
gets back, and the depth policy. With ``TF_TEST_Q27_MODEL`` (a 4-bit Qwen3.8-27B directory) and
``TF_TEST_Q27_MTP_HEAD`` (``tools/qwen27_mtp_head.py``'s file) on a Mac without tensor units, drafted decoding is
checked against serial decoding on the real model, greedy and sampled.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.qwen3_5 import mtp

HIDDEN, VOCAB = 256, 512


def _args() -> Any:
    from mlx_lm.models.qwen3_5 import TextModelArgs

    return TextModelArgs(model_type="qwen3_5", hidden_size=HIDDEN, intermediate_size=512, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=64, vocab_size=VOCAB,
                         rms_norm_eps=1e-6, full_attention_interval=4,
                         rope_parameters={"type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25})


class _Target:
    """What MTPDrafter reads of the model: args, the embedding and a 4-bit vocabulary head."""

    def __init__(self) -> None:
        mx.random.seed(3)
        self.args = _args()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.lm_head = nn.QuantizedLinear.from_linear(head, group_size=64, bits=4)
        mx.eval(self.model.embed_tokens.parameters(), self.lm_head.parameters())


def _head_file(tmp_path: Path, *, bits: int = 4, metadata: bool = True) -> Path:
    """A random ``mtp.*`` file in tools/qwen27_mtp_head.py's layout (norms stored as w for 1 + w)."""

    from tensorfold.families.qwen3_5.mtp import MTPHead

    mx.random.seed(5)
    head = MTPHead(_args())
    raw = dict(_flatten(head.parameters()))
    out: dict[str, Any] = {}
    for name, value in raw.items():
        if name.endswith("weight") and value.ndim == 2:
            w, s, b = mx.quantize(value.astype(mx.bfloat16), group_size=64, bits=bits)
            stem = name[: -len(".weight")]
            out[f"mtp.{stem}.weight"], out[f"mtp.{stem}.scales"], out[f"mtp.{stem}.biases"] = w, s, b
        else:
            out[f"mtp.{name}"] = (mx.random.normal(value.shape) * 0.01).astype(mx.bfloat16)
    path = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(path), out, metadata={"bits": str(bits), "group_size": "64"} if metadata else {})
    return path


def _flatten(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    return tree_flatten(tree)


@pytest.mark.parametrize("bits,metadata", [(4, True), (8, True), (4, False)])
def test_head_loads_quantized_with_shifted_norms(tmp_path: Path, bits: int, metadata: bool) -> None:
    from tensorfold.families.qwen3_5.mtp import load_head

    path = _head_file(tmp_path, bits=bits, metadata=metadata)
    stored = mx.load(str(path))
    head = load_head(path, _args())
    assert isinstance(head.fc, nn.QuantizedLinear) and head.fc.bits == bits
    assert isinstance(head.layers[0].mlp.down_proj, nn.QuantizedLinear)
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        got = getattr(head, name).weight.astype(mx.float32)
        want = stored[f"mtp.{name}.weight"].astype(mx.float32) + 1.0
        assert float(mx.max(mx.abs(got - want)).item()) < 1e-2
    q_norm = head.layers[0].self_attn.q_norm.weight.astype(mx.float32)
    assert float(mx.min(q_norm).item()) > 0.9           # 1 + a small stored w


@pytest.fixture()
def drafter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from tensorfold.families.qwen3_5.mtp import MTPDrafter

    monkeypatch.setenv("TF_MTP_DEPTH", "3")
    monkeypatch.delenv("TF_MTP_WINDOW", raising=False)
    monkeypatch.delenv("TF_MTP_CUT", raising=False)
    return MTPDrafter(_Target(), _head_file(tmp_path), most=7)


def _prefill(proposer: Any, prompt: list[int], start: int = 0) -> None:
    proposer.begin_prefill(prompt, start)
    rows = len(prompt) - start
    proposer.sink.append((mx.random.normal((1, rows, HIDDEN)) * 0.5).astype(mx.bfloat16))
    proposer.on_prefill(len(prompt))


def test_proposer_follows_the_kept_rows(drafter: Any) -> None:
    proposer = mtp.MTPProposer(drafter)
    prompt = list(range(10, 50))
    _prefill(proposer, prompt)
    # every prompt row but the last is in the head's cache; the last waits for the first sampled token
    assert proposer.cache.offset == len(prompt) - 1
    assert [(first, int(h.shape[0])) for first, h in proposer.rows] == [(len(prompt) - 1, 1)]
    context = prompt + [7]
    drafts = proposer.propose(context, 15)
    assert len(drafts) == 3 and all(0 <= d < VOCAB for d in drafts)
    # the last prompt row absorbed, then two chained drafts that the next absorb forgets
    assert proposer.cache.offset == len(prompt) + 2 and proposer.drafted == 2
    # the round verifies [pending, d0, d1, d2]: d0 kept, then the target's own token
    proposer.sink.append((mx.random.normal((1, 4, HIDDEN)) * 0.5).astype(mx.bfloat16))
    proposer.observe(3, 1)
    proposer.on_rows([0, 1])
    assert proposer.next_row == len(context) + 1
    context = context + [drafts[0], 99]
    again = proposer.propose(context, 2)
    assert len(again) == 2
    # the chained entries were dropped, the two kept rows absorbed, one new chained draft
    assert proposer.cache.offset == len(context) - 1 + 1 and proposer.drafted == 1


def test_proposer_stops_when_positions_are_lost(drafter: Any) -> None:
    proposer = mtp.MTPProposer(drafter)
    prompt = list(range(10, 30))
    _prefill(proposer, prompt)
    assert proposer.propose(prompt + [5, 6], 4) == []        # a token the head never saw: no drafts from now on
    assert not proposer.ready and proposer.propose(prompt + [5, 6, 7], 4) == []


def test_resumed_prompt_gets_the_head_history_back(drafter: Any) -> None:
    first = mtp.MTPProposer(drafter)
    prompt = list(range(100, 164))
    _prefill(first, prompt)
    kept = first.cache.keys[..., : len(prompt) - 1, :]
    second = mtp.MTPProposer(drafter)
    longer = prompt + [3, 4, 5, 6]
    second.begin_prefill(longer, 48)                           # resumed from a 48-token snapshot
    assert second.restored == 48 and second.cache.offset == 48 and second.contiguous
    assert bool(mx.array_equal(second.cache.keys[..., :48, :], kept[..., :48, :]).item())
    third = mtp.MTPProposer(drafter)
    other = prompt[:20] + [9] * 44
    third.begin_prefill(other, 48)                             # shares 20 tokens: 19 rows read only those
    assert third.restored == 19 and not third.contiguous


def test_depth_policy(drafter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    proposer = mtp.MTPProposer(drafter)
    assert proposer._depth(15) == 3 and proposer._depth(2) == 2     # fixed depth, capped by the round's budget
    drafter.fixed_depth = 0
    monkeypatch.setattr(drafter, "window_costs", None)            # no costs: by first-draft acceptance
    proposer._rates[0] = 0.85
    assert proposer._depth(7) == 2
    monkeypatch.setattr(drafter, "window_costs", {w: 30.0 + 0.1 * w for w in range(1, 17)})
    proposer._rates = [0.95] * len(proposer._rates)
    assert proposer._depth(7) == 7                                   # flat costs, confident head: all the way
    monkeypatch.setattr(drafter, "window_costs", {w: 25.0 + 10.0 * w for w in range(1, 17)})
    proposer._rates = [0.5] * len(proposer._rates)
    proposer._rounds = 1
    assert proposer._depth(7) == 1                                   # steep costs, doubtful head: one draft
    for _ in range(40):                                              # acceptance tracks what the rounds keep
        proposer._last_was_copy = False
        proposer.observe(3, 3)
    assert proposer._rates[0] > 0.95 and proposer._rates[2] > 0.95


def test_history_window_skips_old_prompt_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorfold.families.qwen3_5.mtp import MTPDrafter

    monkeypatch.setenv("TF_MTP_WINDOW", "16")
    monkeypatch.setenv("TF_MTP_DEPTH", "3")
    drafter = MTPDrafter(_Target(), _head_file(tmp_path), most=7)
    proposer = mtp.MTPProposer(drafter)
    prompt = list(range(10, 60))
    _prefill(proposer, prompt)
    assert proposer.cache.offset == 16 and not proposer.contiguous and drafter.history == []
    assert len(proposer.propose(prompt + [1], 3)) == 3


@pytest.mark.skipif(not (os.environ.get("TF_TEST_Q27_MODEL") and os.environ.get("TF_TEST_Q27_MTP_HEAD")),
                    reason="needs TF_TEST_Q27_MODEL and TF_TEST_Q27_MTP_HEAD (the real 27B and its MTP head)")
@pytest.mark.parametrize("seed", [None, 1234])
def test_mtp_drafted_decoding_equals_serial_on_the_real_model(seed: int | None,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorfold.drafters.dflash_drafter import DFlashProposer
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream, SuffixLookupProposer
    from tensorfold.families import qwen3_5

    # loading sets the engine's class-wide switches (the lane decoder, window costs): put them back afterwards so
    # the fake-model tests that run later see the defaults
    for name in ("precise_single", "lane_forward", "lane_commit", "tree_drafts", "tree_nodes", "chain_nodes",
                 "exact_window", "cheap_window", "window_costs", "lane_prefill", "prefill_align"):
        monkeypatch.setattr(LaneEngine, name, LaneEngine.__dict__[name])
    monkeypatch.setattr(DFlashProposer, "tree_nodes", DFlashProposer.tree_nodes)
    model, tokenizer = qwen3_5.load(Path(os.environ["TF_TEST_Q27_MODEL"]))
    if getattr(model, "_tensorfold_lanes", False):
        pytest.skip("the MTP proposer drafts on the lane decoder without tensor units")

    class App:
        dflash = None

    app = App()
    qwen3_5.setup(app, model, mtp_head=os.environ["TF_TEST_Q27_MTP_HEAD"])
    assert app.dflash is not None
    prompt = tokenizer.encode("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n")
    sampling = None if seed is None else Sampling(seed=seed, temperature=1.0, top_k=20, top_p=0.95)
    out = {}
    for kind in ("serial", "mtp"):
        engine = LaneEngine(model, **qwen3_5.engine_settings(model))
        proposer = app.dflash.proposer(copy=SuffixLookupProposer(), sampling=sampling) if kind == "mtp" else None
        stream = LaneStream(kind, list(prompt), 96, proposer=proposer, sampling=sampling, drafts=kind == "mtp")
        engine.add_stream(stream)
        engine.run()
        out[kind] = list(stream.emitted)
        if proposer is not None:
            assert proposer.proposals > 0 and proposer.accepted_tokens > 0
    assert out["mtp"] == out["serial"]
