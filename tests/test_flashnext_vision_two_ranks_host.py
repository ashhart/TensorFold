"""Image prompts on two ranks: rank 0 encodes, the admission carries the rows, rank 1 attaches the same features."""

import ast
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Callable

import pytest


def _methods():
    root = Path(__file__).resolve().parents[1] / 'src/tensorfold'
    module = ast.parse((root / 'families/qwen4_exp/cuda/multi_tp.py').read_text())
    owner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'TwoRanks')
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_agree', '_prepare_admission')]
    for method in methods:
        method.body = [node for node in method.body if not isinstance(node, ast.ImportFrom)]
    body = ast.ClassDef(name='Ranks', bases=[], keywords=[], body=methods, decorator_list=[], type_params=[])
    capacity = ast.parse((root / 'cuda/capacity.py').read_text())
    gather = next(node for node in capacity.body if isinstance(node, ast.FunctionDef) and node.name == 'gather_ints')
    return ast.fix_missing_locations(ast.Module(body=[gather, body], type_ignores=[]))


def _ranks(visions, expected):
    """Run rank 0 (with ``visions[0]``) and rank 1 (with ``visions[1]``) through one admission; the outcomes."""

    barrier = threading.Barrier(2)
    values, results, sent, shared = {}, [None, None], [], {}
    plan = {'slot': 0, 'cached': 0, 'resume_slot': None, 'error': None, 'points': [], 'need': 7}

    class OutOfStep(RuntimeError):
        pass

    class NoRoom(RuntimeError):
        pass

    class Tensor:
        def __init__(self, data):
            self.data, self.rows = list(data), None

        def view(self, rows, unused):
            self.rows = rows
            return self

        def tolist(self):
            width = len(self.data) // self.rows
            return [self.data[start:start + width] for start in range(0, len(self.data), width)]

    torch = SimpleNamespace(int64=None, tensor=lambda data, **kwargs: Tensor(data),
                            empty=lambda shape, **kwargs: Tensor([0] * shape[0]))
    namespace = {'hashlib': hashlib, 'json': json, 'torch': torch, 'Callable': Callable,
                 'OutOfStep': OutOfStep, 'NoRoom': NoRoom,
                 'shape': lambda dec: ['shared-state'], '_pack': lambda sampling: None,
                 'admission': lambda dec, stream: dict(plan),
                 'ready': lambda dec, proposed: True,
                 'apply': lambda dec, proposed: setattr(dec, 'applied', True)}

    def gather(rank, send, receive):
        values[rank] = list(send.data)
        barrier.wait(timeout=2)
        receive.data = [*values[0], *values[1]]
        barrier.wait(timeout=2)
    exec(compile(_methods(), 'actual-tp-planning-methods', 'exec'), namespace)

    def run(rank):
        decoder = namespace['Ranks']()
        decoder.w = SimpleNamespace(comm=SimpleNamespace(all_gather=lambda send, receive: gather(rank, send, receive)))
        decoder.next_id, decoder.link = 0, None
        if rank == 0:
            decoder.link = SimpleNamespace(send=lambda op: sent.append(op))
        decoder._share_vision = lambda s, images: shared.setdefault(rank, ('features', images))
        decoder.slots, decoder.kept, decoder.streams, decoder.filling = [object()], [], {}, []
        stream = SimpleNamespace(prompt=[1, 2, 3, 4], count=4, sampling=None, draft=True,
                                 stop_eos=False, background=False, vision=visions[rank])
        try:
            decoder._prepare_admission(stream, None if rank == 0 else dict(plan))
            results[rank] = ('admit', stream.vision)
        except Exception as error:
            results[rank] = (type(error).__name__, None)

    threads = [threading.Thread(target=run, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert [r[0] for r in results] == [expected, expected]
    return results, sent, shared


def test_text_admissions_carry_no_image_metadata_and_share_nothing():
    results, sent, shared = _ranks([None, None], 'admit')
    assert sent[0][0] == 'admit' and sent[0][-1] is None and len(sent[0]) == 10
    assert shared == {} and [r[1] for r in results] == [None, None]


def test_rank_0_sends_the_rows_it_encoded_and_both_ranks_share_the_same_tensors():
    encoded = SimpleNamespace(rows=(1, 2), rope_delta=-1, features=SimpleNamespace(shape=(2, 2560)))
    meta = {'rows': [1, 2], 'delta': -1, 'shape': [2, 2560]}
    results, sent, shared = _ranks([encoded, json.loads(json.dumps(meta))], 'admit')
    assert sent[0][-1] == meta                                   # what the follower is told, JSON-exact
    assert shared == {0: ('features', meta), 1: ('features', meta)}
    assert [r[1] for r in results] == [('features', meta)] * 2  # the stream now carries the shared features


def test_an_image_admission_one_rank_does_not_know_about_falls_out_of_step():
    encoded = SimpleNamespace(rows=(1, 2), rope_delta=-1, features=SimpleNamespace(shape=(2, 2560)))
    _, _, shared = _ranks([encoded, None], 'OutOfStep')
    assert shared == {}                                          # the digest differs before any tensor crosses


def test_begin_attaches_features_another_rank_encoded_without_a_tower():
    torch = pytest.importorskip('torch')
    from tensorfold.families.qwen4_exp.cuda import image_rows

    class State:
        def __init__(self):
            self.image_positions = self.image_rows = self.image_features = None
            self.rope_delta = 0

        def set_rope_delta(self, delta):
            self.rope_delta = delta

    st = State()
    encoded = SimpleNamespace(rows=(1, 2), features=torch.zeros((2, 8)), rope_delta=-1,
                              positions=torch.zeros((3, 4), dtype=torch.int32))
    stream = SimpleNamespace(prompt=[5, 6, 7, 8], vision=encoded)
    image_rows.begin(SimpleNamespace(st=st), stream, None)
    assert stream.vision is None and st.image_rows == (1, 2) and st.rope_delta == -1
    assert st.image_features is encoded.features and tuple(st.image_positions.shape) == (4, 3)
    with pytest.raises(ValueError, match='--vision'):            # a prepared prompt still needs the tower
        image_rows.begin(SimpleNamespace(st=State()), SimpleNamespace(prompt=[5], vision=object()), None)


@pytest.fixture
def planning():
    pytest.importorskip('torch')
    from tests.test_cuda_geometry import allocations as geometry   # noqa: F401  (patched allocation sizes)
    import tests.test_flashnext_round_plan as plans
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.multi_plan import admission, round_plan

    plans.MultiDecoder = MultiDecoder                                # what its autouse fixture would set
    return plans.decoder, admission, round_plan


def test_image_prompts_reuse_no_prefix_keep_no_points_and_stay_off_the_graph_slot(planning):
    from tensorfold.cuda.streams import Stream

    decoder, admission, round_plan = planning
    d = decoder()
    d.points = lambda prompt: [8, 16]
    kept_prompt = [7] * 30
    d.slots[0].pos, d.slots[0].mtp_len = 30, 29
    d.free = d.slots[1:]
    d.kept = [(kept_prompt, d.slots[0], {'mtp_len': 29}, None)]
    text = admission(d, Stream(kept_prompt + [9] * 10, 8))
    assert text['cached'] == 30 and text['points'] == []           # the text prompt resumes the kept prefix
    image = admission(d, Stream(kept_prompt + [9] * 10, 8, vision={'rows': [3], 'delta': 0, 'shape': [1, 8]}))
    assert image['cached'] == 0 and image['slot'] != 0 and image['points'] == []
    d = decoder()
    live = Stream([7], 12, sid=0, st=d.slots[0], out=[3], drafts=[4])
    live.st.pos, live.st.image_positions = 255, object()
    d.streams, d.free = {0: live}, d.slots[1:]
    d.solo, d.solo_on = SimpleNamespace(st=d.slots[1]), True
    assert round_plan(d)['solo'] is None                           # the graph slot's captures know no image rows
    live.st.image_positions = None
    assert round_plan(d)['solo'] == 0


# --- the serial engine (no --parallel): image prompts on one GPU and on two ranks ---

def test_serial_request_message_carries_the_image_description_and_keeps_no_points():
    pytest.importorskip('torch')
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    class Store:
        def __init__(self):
            self.kv = {}

        def set(self, key, value):
            self.kv[key] = value

    e = object.__new__(FlashNextEngine)
    e.comm, e.served, e.points = SimpleNamespace(store=Store()), 3, lambda prompt: [2, 4]
    text = e._share([1, 2, 3, 4, 5], 8, None, True, 0)
    assert text[7] is None and text[-1] == [2, 4] and len(text) == 9    # the points stay the last field
    images = {'rows': [1, 2], 'delta': -1, 'shape': [2, 2560]}
    shared = e._share([1, 2, 3, 4, 5], 8, None, True, 0, images=images)
    assert shared[7] == images and shared[-1] == []           # an image prompt keeps no prompt states
    assert json.loads(e.comm.store.kv['tensorfold/flashnext/request/3'])['images'] == images
    assert FlashNextEngine._unpack(json.dumps({'prompt': [1], 'max_tokens': 1, 'draft': True, 'cached': 0,
                                               'sampling': None}))[7] is None    # an older leader's message
    follower = object.__new__(FlashNextEngine)
    follower.served, follower.cache, follower.multi, follower.points = 0, [], None, None
    requests = iter([([1, 2, 3], 4, None, True, 0, [], True, [2]),                # eight fields: no images
                     ([1, 2, 3], 4, None, True, 0, [], True, None, [2]), None])    # nine: images before points
    follower._receive = lambda: next(requests)
    seen = []
    follower._decode = lambda *args, **kwargs: seen.append((kwargs['points'], kwargs['vision']))
    follower.follow()
    assert seen == [([2], None), ([2], None)] and follower.served == 2


def test_describe_and_exchange_validate_the_same_description_on_both_ranks():
    torch = pytest.importorskip('torch')
    from tensorfold.families.qwen4_exp.cuda import vision_ranks

    encoded = SimpleNamespace(rows=(1, 2), rope_delta=-1, features=torch.zeros((2, 16)),
                              positions=torch.zeros((3, 4), dtype=torch.int32))
    images = vision_ranks.describe(encoded)
    assert images == {'rows': [1, 2], 'delta': -1, 'shape': [2, 16]} and vision_ranks.describe(None) is None
    for bad in ({**images, 'rows': [2, 1]}, {**images, 'rows': [1, 4]}, {**images, 'shape': [3, 16]},
                {**images, 'shape': [2, 8]}):
        with pytest.raises(ValueError, match='do not fit'):
            vision_ranks.exchange(None, 1, 4, bad, hidden=16)
    with pytest.raises(ValueError, match='has not encoded'):
        vision_ranks.exchange(None, 0, 4, images, None, hidden=16)
    moved = []
    comm = SimpleNamespace(rank=0, world=2, exchange=lambda sends, recvs, peer: moved.append((len(sends), peer)))
    out = vision_ranks.exchange(comm, 0, 4, images, encoded, hidden=16)
    assert moved == [(2, 1)] and out.rows == (1, 2) and out.rope_delta == -1
    assert out.features.dtype == torch.bfloat16 and tuple(out.positions.shape) == (3, 4)


def test_serial_engine_decodes_an_image_state_eagerly_and_keeps_no_snapshot(monkeypatch):
    pytest.importorskip('torch')
    from tensorfold.families.qwen4_exp.cuda import decode, engine as engine_module
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    calls = []
    eng = object.__new__(decode.Engine)
    eng.w, eng.st, eng.buf, eng.mbuf = object(), SimpleNamespace(image_positions=None), object(), object()
    eng.graphs = SimpleNamespace(forward=lambda tokens: calls.append('graph'),
                                 mtp_forward=lambda nxt, streams: calls.append('graph-mtp'))
    monkeypatch.setattr(decode, 'forward', lambda w, st, b, tokens: calls.append('eager'))
    monkeypatch.setattr(decode, 'mtp_forward', lambda w, st, b, nxt, streams: calls.append('eager-mtp'))
    eng.forward([1])
    eng.mtp_forward([1], None)
    eng.st.image_positions = object()
    eng.forward([1])
    eng.mtp_forward([1], None)
    assert calls == ['graph', 'graph-mtp', 'eager', 'eager-mtp']

    seen = {}

    def prefill(e, prompt, sampling, **kwargs):
        seen.update(kwargs)
        return 7

    monkeypatch.setattr(decode, 'prefill', prefill)
    monkeypatch.setattr(engine_module.time, 'perf_counter', lambda: 0.0)
    import torch
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    fe = object.__new__(FlashNextEngine)
    fe.e, fe.cache, fe.eos, fe.depth, fe.points = SimpleNamespace(kept={'state': 's', 'tail': None}), [([1], {})], (), 3, None
    kept = ([1], {'state': 'old'})
    fe.cache = [kept]
    stats = fe._decode([1, 2, 3], 1, None, lambda new: True, kept, vision=SimpleNamespace(rows=(1,)))
    assert seen == {'constraint': None, 'probabilities': None, 'vision': seen['vision']}
    assert fe.cache == [] and stats['cached'] == 0            # from the start, nothing kept, nothing resumed
    seen.clear()
    fe.cache = [kept]
    fe._decode([1, 2, 3], 1, None, lambda new: True, kept)
    assert 'vision' not in seen and seen['resume'] == {'state': 'old'} and fe.cache[-1][0] == [1, 2]
