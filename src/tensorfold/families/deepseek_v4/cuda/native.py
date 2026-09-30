"""Local binary RPC to the pinned engine. Native fatal errors stay in its child process.

TensorFold owns serving and sampling; this is not a proxy to ds4-server.
"""
from __future__ import annotations

import ctypes as C
import multiprocessing as mp
import math
from multiprocessing.connection import wait
import os
from pathlib import Path
import threading

from .build import PIN


class NativeError(RuntimeError):
    pass


class NativeLibrary:
    def __init__(self, path):
        self.lib = C.CDLL(str(Path(path).resolve()))
        signatures = {
            'abi': (C.c_int, []), 'revision': (C.c_char_p, []), 'backend': (C.c_int, []),
            'open': (C.c_int, [C.c_char_p, C.c_int, C.c_int, C.POINTER(C.c_void_p), C.c_char_p, C.c_size_t]),
            'close': (None, [C.c_void_p]), 'vocab': (C.c_int, [C.c_void_p]),
            'eos': (C.c_int, [C.c_void_p]), 'context': (C.c_int, [C.c_void_p]),
            'reset': (None, [C.c_void_p]),
            'sync': (C.c_int, [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_char_p, C.c_size_t]),
            'eval': (C.c_int, [C.c_void_p, C.c_int, C.c_char_p, C.c_size_t]),
            'logits': (C.c_int, [C.c_void_p, C.POINTER(C.c_float), C.c_int]),
            'encode': (C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.POINTER(C.c_int))]),
            'free': (None, [C.c_void_p]),
            'token_text': (C.c_void_p, [C.c_void_p, C.c_int, C.POINTER(C.c_size_t)]),
            'iq2_dot': (C.c_int, [C.c_void_p, C.c_int, C.POINTER(C.c_int8), C.c_int, C.POINTER(C.c_float)]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.lib, 'tf_ds4_' + name)
            function.restype, function.argtypes = result, args
            setattr(self, name, function)
        if self.abi() != 1 or self.revision().decode() != PIN:
            raise NativeError('native ABI/revision mismatch; rebuild the pinned TensorFold library')


def _native_worker(pipe, library):
    ctx = C.c_void_p()
    api = None
    try:
        # Child-local policy: never mutate the HTTP process's or live ds4's environment.
        os.environ['DS4_WEIGHT_RESIDENCY_BASE'] = 'mapped'
        os.environ['DS4_NO_BOOT_PREWARM'] = '1'
        os.environ['DS4_MEM_FLOOR_GB'] = '8'
        api = NativeLibrary(library)
        pipe.send({'ok': True, 'backend': api.backend()})
        vocab = 0
        while True:
            message = pipe.recv()
            op, args = message['op'], message.get('args', [])
            error = C.create_string_buffer(1024)
            if op == 'close':
                break
            if op == 'open':
                path, context, threads = args
                if ctx.value:
                    raise NativeError('native engine already open')
                if api.open(os.fsencode(path), context, threads, C.byref(ctx), error, len(error)):
                    raise NativeError(error.value.decode(errors='replace'))
                vocab, eos = api.vocab(ctx), api.eos(ctx)
                if not 0 < vocab <= 1_000_000 or not 0 <= eos < vocab or api.context(ctx) != context:
                    raise NativeError('native model vocabulary/EOS/context does not match the requested contract')
                pipe.send({'ok': True, 'value': {'vocab_size': vocab, 'eos': eos}})
                continue
            if not ctx.value:
                raise NativeError('native engine is not open')
            value = None
            if op == 'reset':
                api.reset(ctx)
            elif op == 'sync':
                tokens = args[0]
                if not tokens or len(tokens) > api.context(ctx) or any(t < 0 or t >= vocab for t in tokens):
                    raise NativeError('invalid native token prefix')
                ids = (C.c_int * len(tokens))(*tokens)
                if api.sync(ctx, ids, len(tokens), error, len(error)):
                    raise NativeError(error.value.decode(errors='replace'))
            elif op == 'eval':
                if api.eval(ctx, args[0], error, len(error)):
                    raise NativeError(error.value.decode(errors='replace'))
            elif op == 'logits':
                logits = (C.c_float * vocab)()
                if api.logits(ctx, logits, vocab) != vocab:
                    raise NativeError('native logits have the wrong vocabulary width')
                pipe.send({'ok': True, 'bytes': vocab * 4})
                pipe.send_bytes(bytes(logits))
                continue
            elif op == 'encode':
                text, rendered = args
                ids = C.POINTER(C.c_int)()
                count = api.encode(ctx, text.encode('utf-8'), int(rendered), C.byref(ids))
                if count < 0:
                    raise NativeError('native tokenization failed')
                try:
                    value = [ids[i] for i in range(count)]
                finally:
                    api.free(ids)
            elif op == 'token_text':
                token = args[0]
                if not 0 <= token < vocab:
                    raise NativeError('invalid token ID')
                size = C.c_size_t()
                text = api.token_text(ctx, token, C.byref(size))
                if not text and size.value:
                    raise NativeError('native token text is null')
                try:
                    value = C.string_at(text, size.value) if text else b''
                finally:
                    api.free(text)
            else:
                raise NativeError(f'unknown native operation: {op}')
            pipe.send({'ok': True, 'value': value})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            pipe.send({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if ctx.value and api is not None:
            api.close(ctx)
        pipe.close()


class NativeSession:
    def __init__(self, *, library, model_path, context, threads=4, timeout=600):
        if (isinstance(context, bool) or not isinstance(context, int) or
                isinstance(threads, bool) or not isinstance(threads, int) or
                not 1 <= context < 2**31 or not 1 <= threads <= 64 or
                not math.isfinite(float(timeout)) or float(timeout) <= 0):
            raise ValueError('invalid native context/threads/timeout')
        self.timeout, self._lock, self._closed = float(timeout), threading.Lock(), False
        spawn = mp.get_context('spawn')
        self._pipe, child = spawn.Pipe()
        self._process = spawn.Process(target=_native_worker, args=(child, str(library)), daemon=True)
        try:
            self._process.start()
            child.close()
            info = self._receive()
            if info['backend'] != 1:
                raise NativeError('production DeepSeek engine requires the CUDA native library, not CPU')
            info = self._rpc('open', str(model_path), int(context), int(threads))
            self.vocab_size, self.eos = info['vocab_size'], info['eos']
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self):
        ready = wait([self._pipe, self._process.sentinel], self.timeout)
        if self._pipe not in ready:
            self.close()
            raise NativeError('native engine exited or timed out; TensorFold HTTP process remains alive')
        try:
            value = self._pipe.recv()
        except (EOFError, OSError) as exc:
            self.close()
            raise NativeError('native engine exited; TensorFold HTTP process remains alive') from exc
        if not value.get('ok'):
            raise NativeError(value.get('error', 'native operation failed'))
        return value

    def _rpc(self, op, *args):
        with self._lock:
            if self._closed:
                raise NativeError('native engine is closed')
            try:
                self._pipe.send({'op': op, 'args': args})
                result = self._receive()
                if 'bytes' in result:
                    import numpy as np
                    if result['bytes'] != self.vocab_size * 4 or not self._pipe.poll(self.timeout):
                        self.close()
                        raise NativeError('native logits transfer timed out or has invalid width')
                    data = self._pipe.recv_bytes(maxlength=self.vocab_size * 4)
                    if len(data) != self.vocab_size * 4:
                        raise NativeError('truncated native logits')
                    return np.frombuffer(data, dtype=np.float32).copy()
                return result.get('value')
            except (EOFError, BrokenPipeError, OSError) as exc:
                self.close()
                raise NativeError('native engine connection failed') from exc

    def reset(self):
        self._rpc('reset')

    def sync(self, ids):
        self._rpc('sync', list(ids))

    def eval(self, token):
        self._rpc('eval', int(token))

    def logits(self):
        return self._rpc('logits')

    def encode(self, text, *, rendered=False):
        if '\0' in text:
            raise ValueError('native tokenizer does not accept NUL text')
        return self._rpc('encode', str(text), bool(rendered))

    def token_text(self, token):
        return self._rpc('token_text', int(token))

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.pid is not None:
            if process.is_alive():
                try:
                    self._pipe.send({'op': 'close'})
                except (BrokenPipeError, OSError):
                    pass
                process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            process.close()
        self._pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
