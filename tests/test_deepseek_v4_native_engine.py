"""Focused serial adapter contracts; no model weights or GPU allocations."""
import json
import numpy as np
import pytest

from tensorfold.engine.exact_sampling import Sampling, choose
from tensorfold.families.deepseek_v4.cuda.engine import DeepSeekEngine


class Session:
    vocab_size = 4
    eos = 3

    def __init__(self):
        self.pos = 0
        self.closed = False
        self.calls = []

    def reset(self):
        self.calls.append('reset')

    def sync(self, tokens):
        self.pos = len(tokens)
        self.calls.append(('sync', list(tokens)))

    def logits(self):
        return np.array([0, 1, 4, 7 if self.pos >= 3 else 2], dtype=np.float32)

    def eval(self, token):
        self.calls.append(('eval', token))
        self.pos += 1

    def close(self):
        self.closed = True


def engine(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({
        'model_type': 'deepseek_v4', 'quantization_config': {'quant_method': 'gguf'},
        'gguf_file': str(tmp_path / 'target.gguf'), 'native_library': str(tmp_path / 'native.so')}))
    session = Session()
    instance = DeepSeekEngine(tmp_path, no_drafts=True, context=8,
                             _admission=lambda *_: {'context_window': 8, 'cache_slots': 8},
                             _session_factory=lambda **_: session)
    return instance, session


def test_serial_eos_callbacks_and_repeat_request(tmp_path):
    instance, session = engine(tmp_path)
    out = []
    stats = instance.generate([0, 1], 4, None, lambda ids: out.extend(ids))
    assert out == [2, 3]
    assert session.calls == ['reset', ('sync', [0, 1]), ('eval', 2), ('eval', 3)]
    assert stats['generated'] == 2 and stats['drafts'] is False
    out.clear()
    instance.generate([0, 1], 1, None, lambda ids: out.extend(ids))
    assert out == [2]
    instance.close()
    assert session.closed


def test_zero_tokens_callback_cancel_ignore_eos_and_capacity(tmp_path):
    instance, session = engine(tmp_path)
    assert instance.generate([0], 0, None, lambda _: pytest.fail('callback'))['generated'] == 0
    assert session.calls == []
    out = []
    instance.generate([0, 1], 4, None, lambda ids: out.extend(ids) or True)
    assert out == [2]
    out.clear()
    instance.generate([0, 1], 3, None, lambda ids: out.extend(ids), stop_eos=False)
    assert out == [2, 3, 3]
    with pytest.raises(ValueError, match='capacity'):
        instance.generate([0] * 7, 2, None, lambda _: None)
    instance.close()


def test_shared_keyed_sampling_and_failure_cleanup(tmp_path):
    instance, session = engine(tmp_path)
    sampling = Sampling(123, temperature=.8, top_k=3, top_p=.9)
    expected = choose(session.logits(), np.arange(4), 1, sampling)
    out = []
    instance.generate([0], 1, sampling, lambda ids: out.extend(ids))
    assert out == [expected]
    def fail(_):
        raise RuntimeError('native failure')
    session.eval = fail
    with pytest.raises(RuntimeError, match='native failure'):
        instance.generate([0], 1, None, lambda _: pytest.fail('uncommitted callback'))
    assert session.closed


def test_bad_options_refuse_before_admission_or_native_loading(tmp_path):
    def forbidden(*_, **__):
        pytest.fail('allocation/admission reached')
    for options in ({'tp': 2}, {'parallel': 2}, {'drafter': 'legacy-mtp'}, {'rank': 1}):
        with pytest.raises(ValueError):
            DeepSeekEngine(tmp_path, _admission=forbidden, _session_factory=forbidden, **options)
