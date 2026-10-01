"""TensorFold's serial engine over the proven mapped-GGUF native backend."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time


class DeepSeekEngine:
    def __init__(self, model_dir, *, tp=1, rank=0, drafter='', no_drafts=False,
                 parallel=1, streams=1, context=None, context_explicit=None,
                 mtp_drafts=None, threads=4, _admission=None, _session_factory=None):
        if tp != 1 or rank != 0:
            raise ValueError('DeepSeek GGUF runs on one GPU: tp=1 and rank=0 are required')
        if parallel != 1 or streams != 1:
            raise ValueError('DeepSeek GGUF supports one request: parallel=1 is required')
        if drafter or mtp_drafts not in (None, 0):
            raise ValueError('DeepSeek serial GGUF backend supports no drafter or MTP head')
        if context is not None and (isinstance(context, bool) or not isinstance(context, int) or context < 1):
            raise ValueError('context must be a positive integer')
        self.model_dir = Path(model_dir)
        raw = json.loads((self.model_dir / 'config.json').read_text())
        quant = raw.get('quantization_config') or raw.get('quantization') or {}
        if raw.get('model_type') != 'deepseek_v4' or quant.get('quant_method') != 'gguf':
            raise ValueError('DeepSeek CUDA engine requires validated deepseek_v4 GGUF storage metadata')
        descriptor = self.model_dir / 'descriptor.json'
        stored = json.loads(descriptor.read_text()) if descriptor.is_file() else {}
        gguf = raw.get('gguf_file') or stored.get('source')
        library = raw.get('native_library')
        if not gguf or not library:
            raise ValueError('candidate metadata must name its GGUF source and built native_library')
        gguf, library = Path(gguf), Path(library)
        gguf = gguf if gguf.is_absolute() else self.model_dir / gguf
        library = library if library.is_absolute() else self.model_dir / library
        if _admission is None:
            try:
                from .capacity import admit
            except ImportError as exc:
                raise RuntimeError('DeepSeek capacity integration is required before native model loading') from exc
            _admission = admit
        # S02 supplies this CPU/no-allocation admission before even starting the native process.
        self.capacity_plan = _admission(self.model_dir, context, context_explicit)
        self.context_window = int(self.capacity_plan['context_window'])
        if self.context_window < 1 or (context is not None and self.context_window != context):
            raise ValueError('capacity admission did not preserve the explicit requested context')
        if _session_factory is None:
            from .native import NativeSession
            _session_factory = NativeSession
        self.session = _session_factory(library=library, model_path=gguf, context=self.context_window, threads=threads)
        self.eos = (self.session.eos,)
        self.vocab_size = self.session.vocab_size
        self.tp, self.rank, self.concurrent, self.max_rows = 1, 0, False, 1
        self._lock, self._closed = threading.Lock(), False
        self._timeline = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True):
        """Reuse an exact committed prefix; callbacks receive committed tokens once."""
        import numpy as np
        from tensorfold.engine.exact_sampling import choose

        if self._closed:
            raise RuntimeError('DeepSeek engine is closed')
        ids = list(prompt)
        if not ids or any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) or
                          not 0 <= int(t) < self.vocab_size for t in ids):
            raise ValueError('prompt must contain valid integer token IDs')
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError('max_tokens must be a nonnegative integer')
        if len(ids) + max_tokens > self.context_window:
            raise ValueError('prompt plus reply exceeds the admitted context capacity')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('DeepSeek serial engine is already handling a request')
        try:
            if max_tokens == 0:
                return {'generated': 0, 'cached': 0, 'drafts': False, 'prefill_s': 0., 'decode_s': 0.}
            start = time.perf_counter()
            ids = [int(t) for t in ids]
            cached = len(self._timeline) if (self._timeline and
                len(ids) >= len(self._timeline) and
                ids[:len(self._timeline)] == self._timeline) else 0
            if not cached:
                self.session.reset()
            self.session.sync(ids)
            prefill_s = time.perf_counter() - start
            decode_start, count = time.perf_counter(), 0
            logits = self.session.logits()
            for step in range(max_tokens):
                if (logits.shape != (self.vocab_size,) or not np.isfinite(logits).any() or
                        np.isnan(logits).any() or np.isposinf(logits).any()):
                    raise RuntimeError('native engine returned invalid logits')
                token = (int(np.argmax(logits)) if sampling is None or sampling.temperature <= 0 else
                         choose(logits, np.arange(self.vocab_size), len(ids), sampling))
                ending = step == max_tokens-1 or (stop_eos and token in self.eos)
                if not ending and hasattr(self.session, 'eval_logits'):
                    next_logits = self.session.eval_logits(token)
                else:
                    self.session.eval(token)
                    next_logits = None
                ids.append(token)
                count += 1
                if on_tokens([token]) or (stop_eos and token in self.eos):
                    break
                if step < max_tokens-1:
                    logits = next_logits if next_logits is not None else self.session.logits()
            self._timeline = ids
            return {'generated': count, 'cached': cached, 'drafts': False, 'prefill_s': prefill_s,
                    'decode_s': time.perf_counter() - decode_start, 'rounds': count,
                    'min_rows': 1, 'drafted': 0, 'accepted': 0}
        except BaseException:
            self.close()
            raise
        finally:
            self._lock.release()

    def close(self):
        if not self._closed:
            self._closed = True
            self.session.close()

    def follow(self):
        raise ValueError('DeepSeek GGUF has no follower rank; use tp=1')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
