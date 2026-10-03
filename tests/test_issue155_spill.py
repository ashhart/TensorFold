"""Issue #155: a refused boundary checkpoint is logged, never silently dropped.

A capture that memory or the prompt-cache budget refused used to die without a word: neither in memory
(refused) nor on disk (spillover fired on eviction only), so every later turn reusing that prefix
re-prefilled it from token 0 and nothing said why. These tests pin: (1) the store-level round trip (a
refused capture now reaches the session directory where disk can take it, and reads back byte-identical),
(2) the full-chain pair where a refused prompt stays correct (it re-prefills: slower, never wrong) and
every loss leaves one log line and one counter.

``FakeBatchItem`` keeps its token history as plain lists, which ``save_snapshot`` cannot serialize; that
is why the scheduler-level pairs prove the counted-and-printed loss while the first test proves the disk
round trip with a serializable stand-in.
"""

import pytest

pytest.importorskip("mlx.core")


class DiskItem:
    """A serializable checkpoint layer: one uint32 row holds the token history (read back byte-identical)."""

    def __init__(self, rows=None) -> None:
        import mlx.core as mx

        self.rows = mx.array([list(rows or [])], dtype=mx.uint32)
        self.offset = len(rows or [])


def test_a_refused_capture_reaches_the_disk_and_reads_back_byte_identical(tmp_path):
    import mlx.core as mx

    from tensorfold.engine.prefix_snapshots import DiskBlocks, load_snapshot
    from tensorfold.server.checkpoints import CheckpointStore, spill_conversation

    model = "fake-model|f32"
    prompt, capture = list(range(22)), list(range(12))

    def sizer(cache):
        return sum(len(x.rows[0]) * 4 for x in cache)

    store = CheckpointStore(1, copier=lambda c: c, budget_bytes=8, sizer=sizer,
                            on_evict=lambda e: spill_conversation(e, tmp_path, model, limit_bytes=1 << 20))
    store.insert(capture, [DiskItem(capture)], last_prompt=prompt)         # 48 B > the 8 B budget: refused
    assert store.refused == 0 and store.spilled == 1                        # disk kept it: spillover fired
    assert len(list(tmp_path.glob("*.safetensors"))) == 1

    found = DiskBlocks(tmp_path, model).best(prompt, 0)
    loaded = load_snapshot(found[0], model)
    assert loaded is not None
    tokens, cache = loaded
    assert tokens == capture and cache[0].offset == 12
    assert bool(mx.array_equal(cache[0].rows, DiskItem(capture).rows).item())


def test_a_refused_capture_that_disk_cannot_keep_still_leaves_one_log_line(capsys):
    from tensorfold.server.checkpoints import CheckpointStore

    store = CheckpointStore(1, copier=lambda c: list(c), budget_bytes=8, sizer=lambda c: len(c[0]))
    store.insert(list(range(22)), [[7] * 12], last_prompt=list(range(22)))  # no disk wired: kept nowhere
    out = capsys.readouterr().out
    assert store.refused == 1 and store.spilled == 0
    assert out.count("kept nothing at 22 tokens") == 1 and "re-prefills" in out


def test_a_refused_prompt_stays_correct_and_says_so(tmp_path, capsys):
    from tests.test_prompt_fill import GridEngine, _prompt, _run, _solo
    from tensorfold.server.checkpoints import CheckpointStore, spill_conversation
    from tensorfold.server.scheduler import ChatJob, Scheduler

    model, prompt = "fake-model|f32", _prompt(3)               # 22 tokens: chunks every 4, boundary at 12
    fired = []

    def spill(e):
        fired.append(e)
        return spill_conversation(e, tmp_path, model, limit_bytes=1 << 20)

    store = CheckpointStore(8, copier=GridEngine.copy_single_cache, budget_bytes=8,
                            sizer=lambda c: 100 * len(c), on_evict=spill)
    first = Scheduler(GridEngine(), lanes=3, eos_ids=frozenset({-1}), checkpoints=store,
                      session_dir=tmp_path, model_id=model)
    job1 = ChatJob("first", prompt, 4, 0.0, shared_prefix_lens=(12,))
    _run(first, [job1])
    assert fired and store.refused == 1                          # spillover fired at refusal (fake unserializable)
    out = capsys.readouterr().out
    assert "conversation spill failed" in out and "kept nothing at 12 tokens" in out

    retry = Scheduler(GridEngine(), lanes=3, eos_ids=frozenset({-1}), session_dir=tmp_path, model_id=model)
    job2 = ChatJob("retry", prompt, 4, 0.0)
    _run(retry, [job2])
    assert job2.error is None and _solo(job2)                    # the retry re-prefills: slower, never wrong
