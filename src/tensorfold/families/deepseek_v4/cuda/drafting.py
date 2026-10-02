"""DSpark target sampling and committed-token callbacks over the native pipe."""

import ctypes as C
import os
import time

import numpy as np

SAMPLE = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_float), C.c_int, C.c_void_p)
EMIT = C.CFUNCTYPE(C.c_int, C.c_int, C.c_void_p)


def bind(api):
    lib = api.lib
    lib.tf_ds4_draft_open.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_int,
        C.c_int,
        C.POINTER(C.c_void_p),
        C.c_char_p,
        C.c_size_t,
    ]
    lib.tf_ds4_draft_open.restype = C.c_int
    lib.tf_ds4_draft_generate.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_int),
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_int,
        SAMPLE,
        EMIT,
        C.c_void_p,
        C.POINTER(C.c_int),
        C.c_char_p,
        C.c_size_t,
    ]
    lib.tf_ds4_draft_generate.restype = C.c_int


def policy(context):
    # Fix the allocator/execution policy rather than inheriting experimental flags.
    for key in list(os.environ):
        if key.startswith(("DS4_CONT_", "DS4_DSPARK_", "DS4_MTP_")):
            os.environ.pop(key)
    os.environ["DS4_CONT_DSPARK"] = "1"
    os.environ["DS4_CONT_MTP_MODE"] = "2"
    # Width-one HC fusion and the <=2-row Q8 pair use different arithmetic
    # from wider verification. Use the same row kernels for both target paths;
    # matching the sampler alone cannot make divergent logits exact.
    os.environ["DS4_CUDA_NO_HC_STAGE_FUSED"] = "1"
    os.environ["DS4_CUDA_NO_BATCH_Q8_PAIR"] = "1"
    os.environ["DS4_BATCH_VMM_COMP"] = "0" if context < 8192 else "1"
    # GB10 shares host/device memory. Keep the support file mapped instead of
    # paying for a second promoted dense-tensor copy alongside Hunyuan.
    os.environ["DS4_WEIGHT_RESIDENCY_DRAFTER"] = "mapped"


def worker_generate(api, ctx, args, pipe, error, vocab):
    tokens, budget, draft, stop_eos = args
    ids = (C.c_int * len(tokens))(*tokens)
    cached = C.c_int()
    failure = []

    @SAMPLE
    def sample(values, width, _):
        try:
            if width != vocab:
                raise ValueError("invalid native sample width")
            pipe.send({"ok": True, "event": "sample", "bytes": width * 4})
            pipe.send_bytes(C.string_at(values, width * 4))
            token = pipe.recv()
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab:
                raise ValueError("invalid target sample")
            return token
        except BaseException as exc:  # noqa: BLE001 -- exceptions must not cross a C callback.
            failure.append(str(exc))
            return 0

    @EMIT
    def emit(token, _):
        try:
            if failure:
                return 0
            pipe.send({"ok": True, "event": "token", "token": token})
            return int(not pipe.recv())
        except BaseException as exc:  # noqa: BLE001 -- exceptions must not cross a C callback.
            failure.append(str(exc))
            return 0

    count = api.lib.tf_ds4_draft_generate(
        ctx, ids, len(tokens), budget, int(draft), int(stop_eos), sample, emit, None, C.byref(cached), error, len(error)
    )
    if count < 0 or failure:
        raise RuntimeError(failure[0] if failure else error.value.decode(errors="replace"))
    return {"generated": count, "cached": cached.value}


def generate(session, prompt, budget, sample, on_tokens, draft, stop_eos):
    with session._lock:
        if session._closed:
            raise RuntimeError("native engine is closed")
        start = time.perf_counter()
        first = None
        generated = 0
        try:
            session._pipe.send({"op": "generate", "args": [prompt, budget, draft, stop_eos]})
            while True:
                event = session._receive()
                if event.get("event") == "sample":
                    if event["bytes"] != session.vocab_size * 4 or not session._pipe.poll(session.timeout):
                        raise RuntimeError("invalid or stalled draft logits")
                    values = np.frombuffer(session._pipe.recv_bytes(maxlength=event["bytes"]), dtype=np.float32)
                    if values.size != session.vocab_size:
                        raise RuntimeError("truncated draft logits")
                    session._pipe.send(int(sample(values)))
                elif event.get("event") == "token":
                    first = time.perf_counter() if first is None else first
                    generated += 1
                    session._pipe.send(bool(on_tokens([int(event["token"])])))
                else:
                    result = event["value"]
                    if generated != result["generated"]:
                        raise RuntimeError("native committed-token count mismatch")
                    end = time.perf_counter()
                    result.update(drafts=bool(draft), prefill_s=(first or end) - start, decode_s=end - (first or end))
                    return result
        except BaseException:
            session.close()
            raise
