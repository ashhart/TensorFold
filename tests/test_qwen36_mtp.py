"""Qwen3.6 MoE on Macs drafts with its MTP layer: drafted == serial, together == alone, resumed == fresh."""

from functools import partial

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.engine.prefill_plan import PrefillPlan  # noqa: E402
from tensorfold.families.qwen3_5_moe import mtp as qmtp  # noqa: E402
from tensorfold.families.qwen3_5_moe.family import Qwen36Family  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_matmul  # noqa: E402

VOCAB = 512
EIGHT = ("gate", "shared_expert_gate")           # the checkpoint's 8-bit router and gate


def _quantize(module):
    nn.quantize(module, class_predicate=lambda path, m: hasattr(m, "to_quantized") and (
        {"group_size": 64, "bits": 8} if path.split(".")[-1] in EIGHT else {"group_size": 64, "bits": 4}))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    args = TextModelArgs(model_type="qwen3_5_moe_text", hidden_size=1024, intermediate_size=512, num_hidden_layers=4,
                         num_attention_heads=16, num_key_value_heads=2, head_dim=256, rms_norm_eps=1e-6,
                         vocab_size=VOCAB, linear_num_value_heads=32, linear_num_key_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                         full_attention_interval=4, tie_word_embeddings=False, max_position_embeddings=4096,
                         num_experts=16, num_experts_per_tok=8, shared_expert_intermediate_size=512,
                         moe_intermediate_size=512, norm_topk_prob=True)
    mx.random.seed(21)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    _quantize(model)
    mx.eval(model.parameters())
    exact_attention.install()
    row_matmul.install(model, row_matmul.simd_qmm_backend())
    head = qmtp.Qwen36MTP(args)
    head.set_dtype(mx.bfloat16)
    _quantize(head)
    mx.eval(head.parameters())
    path = tmp_path_factory.mktemp("mtp") / qmtp.MTP_FILE
    mx.save_safetensors(str(path), {f"language_model.mtp.{k}": v for k, v in tree_flatten(head.parameters())})
    return model, path, head


def _family(tiny, drafts=3):
    model, path, _ = tiny
    return Qwen36Family(model, mtp_path=path, drafts=drafts, widest=row_matmul.WINDOW_ROWS, rows=True,
                        first_copy_rows=row_matmul.WINDOW_ROWS)


def _prompt(seed, n=21):
    return [(37 * seed + 11 * i + 5) % VOCAB for i in range(n)]


def _run(family, specs, *, plan=None, caches=None):
    """Streams (prompt, sampling, drafts) decoded together; their tokens and drafted/accepted counts."""

    engine = LaneEngine(family, max_rows=family.exact_width, max_draft=family.exact_width - 1,
                        prefill_plan=plan or PrefillPlan(2048))
    streams = []
    for i, (prompt, sampling, drafts) in enumerate(specs):
        stream = LaneStream(f"s{i}", list(prompt), 40, sampling=sampling, drafts=drafts)
        cache, at = (caches or {}).get(i, (None, 0))
        engine.add_stream(stream, cache=cache, cached_tokens=at)
        streams.append(stream)
    engine.run()
    return [s.emitted for s in streams], engine


def test_the_side_file_loads_with_its_stored_formats(tiny):
    _, path, head = tiny
    args = tiny[0].args
    loaded = qmtp.load(path, args)
    assert loaded.layers[0].mlp.gate.bits == 8 and loaded.layers[0].mlp.switch_mlp.up_proj.bits == 4
    mine, theirs = dict(tree_flatten(loaded.parameters())), dict(tree_flatten(head.parameters()))
    assert mine.keys() == theirs.keys()
    assert all(bool(mx.array_equal(mine[k], theirs[k]).item()) for k in mine)


def test_a_chain_row_attends_the_buffers_and_its_chain_as_one_attention(tiny):
    attn = tiny[2].layers[0].self_attn
    mx.random.seed(3)
    cache = qmtp.MTPCache()
    keys = mx.random.normal((1, 2, 37, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, 37, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(keys[:, :, :34], values[:, :, :34])
    for row in range(34, 37):
        queries = mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16)
        got = qmtp.chain_attention(attn, cache, queries, keys[:, :, row:row + 1], values[:, :, row:row + 1])
        want = mx.fast.scaled_dot_product_attention(queries, keys[:, :, :row + 1], values[:, :, :row + 1],
                                                    scale=attn.scale).transpose(0, 2, 1, 3).reshape(1, 1, -1)
        assert mx.allclose(got.astype(mx.float32), want.astype(mx.float32), atol=3e-2, rtol=3e-2).item()
    assert cache.offset == 34 and cache.drafted == 3       # the chain's rows stayed out of the buffers


@pytest.mark.parametrize("sampling", [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95)])
def test_drafted_replies_equal_serial_ones(tiny, sampling):
    family = _family(tiny)
    assert family.mtp is not None and family.drafts == 3 and family.mtp_step_ms > 0
    specs = [(_prompt(s), sampling, True) for s in range(3)]
    drafted, engine = _run(family, specs)
    serial, _ = _run(family, [(p, s, False) for p, s, _ in specs])
    assert drafted == serial
    assert engine.drafted > 0


def test_the_head_holds_one_row_a_target_position(tiny):
    family = _family(tiny)
    engine = LaneEngine(family, max_rows=family.exact_width, max_draft=family.exact_width - 1)
    stream = LaneStream("s", _prompt(4), 12)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
        cache = engine._live[0][1] if engine._live else None
        if cache is not None:
            assert cache[-1].offset == stream.cache_len          # chained rows wait in the side, apart


def _oracle(family, reference, prompt_len, wrong_every):
    """Drafts that are the reference reply's tokens, wrong every ``wrong_every``-th: accepted paths of every length."""

    calls = {"n": 0}
    draw = family._draw

    def drafted(state, samplings, positions):
        real = draw(state, samplings, positions)
        out = []
        for p in positions:
            calls["n"] += 1
            i = int(p) - prompt_len
            token = reference[i] if 0 <= i < len(reference) else 0
            out.append((token + 1) % VOCAB if calls["n"] % wrong_every == 0 else token)
        return mx.array(out, dtype=mx.uint32) + 0 * real

    family._draw = drafted


@pytest.mark.parametrize("wrong_every", [3, 7])
def test_landing_drafts_keep_rows_exactly(tiny, wrong_every):
    prompt = _prompt(9)
    serial, _ = _run(_family(tiny), [(prompt, None, False)])
    family = _family(tiny, drafts=4)
    _oracle(family, serial[0], len(prompt), wrong_every)
    drafted, engine = _run(family, [(prompt, None, True)])
    assert drafted == serial
    assert 0 < engine.accepted < engine.drafted                   # rounds kept their drafts, and rolled some back
    assert max(r.committed for r in engine.round_stats) >= 3      # a round landed two drafts and its bonus token


def test_streams_together_equal_each_alone(tiny):
    family = _family(tiny)
    sampled = Sampling(seed=11, temperature=0.8, top_k=20, top_p=0.95)
    specs = [(_prompt(1), None, True), (_prompt(2, 13), sampled, True), (_prompt(3, 30), None, True)]
    together, engine = _run(family, specs)
    alone = [_run(family, [spec])[0][0] for spec in specs]
    assert together == alone
    assert engine._shared_rounds > 0


def test_a_resumed_prompt_equals_a_fresh_one(tiny):
    family = _family(tiny)
    prompt = _prompt(5, 29)
    plan = PrefillPlan(8)
    fresh, _ = _run(family, [(prompt, None, True)], plan=plan)
    engine = LaneEngine(family, max_rows=family.exact_width, max_draft=family.exact_width - 1, prefill_plan=plan)
    stored = engine.prefill_prefix(prompt[:16])
    assert isinstance(stored[-1], qmtp.MTPCache) and stored[-1].offset == 15     # row 15 waits for token 16
    resumed, _ = _run(family, [(prompt, None, True)], plan=plan,
                      caches={0: (LaneEngine.copy_single_cache(stored), 16)})
    assert resumed == fresh
    bare = LaneEngine.copy_single_cache(stored)[:-1]           # a prefix stored without the head's cache
    again, _ = _run(family, [(prompt, None, True)], plan=plan, caches={0: (bare, 16)})
    assert again == fresh


def test_without_the_side_file_the_family_runs_plain(tiny):
    model = tiny[0]
    family = Qwen36Family(model, mtp_path=None, widest=row_matmul.WINDOW_ROWS, rows=True,
                          first_copy_rows=row_matmul.WINDOW_ROWS)
    assert family.mtp is None and family.drafts == 0
    cache = family.make_cache()
    assert not isinstance(cache[-1], qmtp.MTPCache)
    stored = _family(tiny).make_cache()
    assert len(family.adopt_cache(stored)) == len(cache)       # a drafting server's prefix, adopted without its head


def test_the_family_package_finds_the_side_file(tmp_path, monkeypatch):
    from tensorfold import hub
    from tensorfold.families import qwen3_5_moe

    monkeypatch.delenv("TF_QWEN36_MTP", raising=False)
    monkeypatch.setattr(hub, "cached", lambda repo, **kw: None)
    assert qwen3_5_moe.mtp_file(tmp_path) is None
    (tmp_path / qmtp.MTP_FILE).write_bytes(b"")
    assert qwen3_5_moe.mtp_file(tmp_path) == tmp_path / qmtp.MTP_FILE
    monkeypatch.setenv("TF_QWEN36_MTP", "0")
    assert qwen3_5_moe.mtp_file(tmp_path) is None
    tested = tmp_path / "tested"
    tested.mkdir()
    (tested / qmtp.MTP_FILE).write_bytes(b"")
    monkeypatch.delenv("TF_QWEN36_MTP")
    monkeypatch.setattr(hub, "cached", lambda repo, **kw: tested if repo == qwen3_5_moe.MODELS[0] else None)
    assert qwen3_5_moe.mtp_file(tmp_path / "elsewhere") == tested / qmtp.MTP_FILE
    monkeypatch.setenv("TF_QWEN36_MTP", str(tmp_path / "missing.safetensors"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        qwen3_5_moe.mtp_file(tmp_path)


def test_another_models_layer_is_refused_with_its_reason(tiny, tmp_path):
    model, path, _ = tiny
    other = tmp_path / qmtp.MTP_FILE
    weights = mx.load(str(path))
    weights.pop(next(k for k in weights if k.endswith("fc.scales")))
    mx.save_safetensors(str(other), weights)
    with pytest.raises(SystemExit, match="not this checkpoint's MTP layer"):
        Qwen36Family(model, mtp_path=other, widest=row_matmul.WINDOW_ROWS, rows=True,
                     first_copy_rows=row_matmul.WINDOW_ROWS)


def test_the_package_loads_the_mtp_family_unless_a_draft_model_is_named(tmp_path, monkeypatch):
    from tensorfold.families import qwen3_5, qwen3_5_moe

    made = {}
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path: ("model", "tokenizer"))
    monkeypatch.setattr(qwen3_5, "lane_family", lambda model, **kw: made.update(kw) or "family")
    monkeypatch.setattr(qwen3_5_moe, "mtp_file", lambda path: tmp_path / "mtp.safetensors")
    assert qwen3_5_moe.load(tmp_path) == ("family", "tokenizer")
    assert isinstance(made["make"], partial) and made["make"].keywords["mtp_path"] == tmp_path / "mtp.safetensors"
    made.clear()
    qwen3_5_moe.load(tmp_path, mtp_drafts=0)
    assert made["make"].keywords["mtp_path"] is None
    made.clear()
    qwen3_5_moe.load(tmp_path, drafter="/some/dflash")
    assert made["drafter"] == "/some/dflash" and "make" not in made


def test_a_snapshot_keeps_the_heads_rows_and_leaves_its_round_state(tiny, tmp_path):
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    family = _family(tiny)
    engine = LaneEngine(family, max_rows=family.exact_width, max_draft=family.exact_width - 1)
    cache = engine.prefill_prefix(_prompt(6, 24))
    head = cache[-1]
    head.side = (mx.zeros((1, 2, 1, 256)), mx.zeros((1, 2, 1, 256)))
    head.speculation = (mx.zeros((1, 1, 8)), 1, True)
    path = save_snapshot(tmp_path, "model", list(range(24)), cache)
    tokens, loaded = load_snapshot(path, "model")
    restored = loaded[-1]
    assert type(restored) is qmtp.MTPCache and restored.offset == head.offset == 23
    assert restored.side is None and restored.speculation is None
    assert bool(mx.array_equal(restored.keys[..., :23, :], head.keys[..., :23, :]).item())


def test_plain_rounds_read_their_rows_into_the_head_and_draft_nothing(tiny):
    family = _family(tiny)
    prompt = _prompt(7)
    serial, _ = _run(family, [(prompt, None, False)])
    engine = LaneEngine(family, max_rows=family.exact_width, max_draft=family.exact_width - 1)
    engine._depth = lambda stream: 0                                   # every round plain, the first one too
    engine._head_depth = lambda stream, budget=None: 0
    stream = LaneStream("s", list(prompt), 40)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
        cache = engine._live[0][1] if engine._live else None
        if cache is not None:
            assert cache[-1].offset == stream.cache_len and cache[-1].speculation is None
    assert stream.emitted == serial[0] and engine.drafted == 0
