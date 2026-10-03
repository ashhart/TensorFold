"""Enable the experimental HC layout only after one byte-exact check per CUDA device in this process."""

import threading

import torch

from . import glue

_checked = {}
_lock = threading.Lock()


@torch.no_grad()
def _check(candidate, device) -> bool:
    """Compare every output with the released kernels without touching the model or its random generator."""

    rows, dims, streams, slots = 7, 2560, 4, 9
    with torch.cuda.device(device):
        generator = torch.Generator(device=device).manual_seed(1927)
        def random(shape, dtype=torch.bfloat16):
            return torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
        source = random((rows, streams * dims))
        scale = random((streams * dims,), torch.float32)
        inject = random((rows, streams))
        branch = random((rows, dims))
        partial = random((3, rows, dims), torch.float32)
        y, wts = random((rows, slots, dims)), random((rows, slots), torch.float32)
        for mode in range(5):
            old = (source.clone(), torch.empty((rows, dims // 256, streams), device=device, dtype=torch.float32),
                   torch.empty_like(source),
                   torch.empty((rows, streams * dims // 32), device=device, dtype=torch.float32))
            new = (source.clone(), *(torch.empty_like(t) for t in old[1:]))
            part = partial if mode in (3, 4) else branch
            glue.hc_writeback(old[0], old[0], old[1], streams, mode, branch=part, inject=inject, y=y, wts=wts)
            glue.hc_normed(old[0], old[1], scale, old[2], old[3], streams, 1e-6)
            candidate(new[0], new[1], scale, new[2], new[3], streams, 1e-6, mode, part, inject, y, wts)
            if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(old, new)):
                return False
    return True


@torch.no_grad()
def _check_upmix(candidate, device) -> bool:
    """Check projection bytes and both mix outputs before enabling the prompt epilogue."""

    from . import qmm

    n, low, streams = 10240, 320, 4
    with torch.cuda.device(device):
        generator = torch.Generator(device=device).manual_seed(2639)
        def random(shape):
            return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        words = torch.randint(-(2**31), 2**31 - 1, (n, low // 8), generator=generator,
                              device=device, dtype=torch.int64).to(torch.int32)
        scale = (random((n, low // 32)).abs() * 0.02 + 0.001).to(torch.bfloat16)
        bias = (random((n, low // 32)) * 0.02).to(torch.bfloat16)
        tiled, frag = qmm.make_q4(words, scale, bias, "tiled"), qmm.make_q4(words, scale, bias, "frag")
        for rows in (17, 32):
            act, normed = random((rows, low)).to(torch.bfloat16), random((rows, n)).to(torch.bfloat16)
            old_up = qmm.prefill_matmul(act, frag)
            old_mix = torch.empty((rows, n // streams), dtype=torch.bfloat16, device=device)
            old_sum = torch.empty((rows, n // streams // 32), dtype=torch.float32, device=device)
            glue.hc_mix(old_up, normed, old_mix, old_sum, streams)
            up, mixed, sums = torch.empty_like(old_up), torch.empty_like(old_mix), torch.empty_like(old_sum)
            candidate(act, tiled, normed, mixed, sums, streams, up=up)
            if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                   for a, b in zip((old_up, old_mix, old_sum), (up, mixed, sums))):
                return False
            candidate(act, tiled, normed, mixed, sums, streams)
            if not torch.equal(old_mix.view(torch.uint8), mixed.view(torch.uint8)) or not torch.equal(
                    old_sum.view(torch.uint8), sums.view(torch.uint8)):
                return False
    return True


def fuser(device, *, upmix: bool = False):
    """A cached callable or process-lifetime fallback; concurrent first callers share the same check."""

    device = torch.device(device)
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, upmix)
    if key in _checked:
        return _checked[key]
    with _lock:
        if key not in _checked:
            candidate = None
            device = torch.device("cuda", index)
            if torch.cuda.get_device_capability(device) == (12, 1):
                try:
                    if upmix:
                        from .hc_upmix import prefill_upmix as fused
                    else:
                        from .hc_fused import write_norm as fused
                    equal = (_check_upmix if upmix else _check)(fused, device)
                    if equal:
                        candidate = fused
                    status = "passed; enabled" if equal else "bytes differ; using released kernels"
                except Exception as exc:
                    status = f"failed ({type(exc).__name__}); using released kernels"
                label = "HC upmix" if upmix else "HC fusion"
                print(f"[tensorfold] {label} self-check on {device}: {status}", flush=True)
            _checked[key] = candidate
        return _checked[key]
