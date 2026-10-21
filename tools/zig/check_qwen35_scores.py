"""Compare finite CUDA score reductions to independent FP64 PyTorch references; not the kernel contract."""
import argparse
import ctypes as C
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fatbin", type=Path)
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch is unavailable")
        return
    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable")
        return
    torch.cuda.init()
    context_owner = torch.empty(1, device="cuda")
    driver = C.CDLL("libcuda.so.1")
    module, function = C.c_void_p(), C.c_void_p()

    def checked(status):
        if status:
            raise RuntimeError(f"CUDA driver status {status}")

    checked(driver.cuModuleLoad(C.byref(module), str(args.fatbin.resolve()).encode()))
    try:
        checked(driver.cuModuleGetFunction(C.byref(function), module, b"tf_q35_score_stats"))
        results = []
        for rows in (1, 2, 4):
            for vocab in (1, 31, 255, 256, 257, 248320):
                for scale in (0., 1., 100.):
                    torch.manual_seed(176)
                    logits = (torch.randn(rows, vocab, device="cuda") * scale).contiguous()
                    stats = torch.empty(72, device="cuda", dtype=torch.float32)
                    selected = [0, vocab // 2, vocab - 1]
                    labels = torch.tensor([[3, *selected, *([0] * 13)]] * rows,
                                          device="cuda", dtype=torch.int32)
                    x, n = C.c_void_p(logits.data_ptr()), C.c_int(vocab)
                    out, label = C.c_void_p(stats.data_ptr()), C.c_void_p(labels.data_ptr())
                    params = (C.c_void_p * 4)(*[C.addressof(v) for v in (x, n, out, label)])
                    checked(driver.cuLaunchKernel(function, rows, 1, 1, 256, 1, 1, 0,
                                                  C.c_void_p(0), params, C.c_void_p(0)))
                    torch.cuda.synchronize()
                    expected = torch.logsumexp(logits.double(), dim=1)
                    delta = (stats[:rows * 2:2].double() - expected).abs()
                    tokens = stats[:rows * 2].view(torch.int32)[1::2]
                    token_ok = torch.equal(tokens.long(), logits.argmax(dim=1))
                    selected_ok = torch.equal(stats[8:8 + rows * 16].view(rows, 16)[:, :3], logits[:, selected])
                    results.append(dict(rows=rows, vocab=vocab, scale=scale,
                                        per_row_lse_error=delta.tolist(),
                                        same_argmax=token_ok, selected_logits_equal=selected_ok,
                                        passed=bool(delta.max() <= 1e-4) and token_ok and selected_ok))
        print(json.dumps(dict(fixtures=results, passed=all(r["passed"] for r in results))))
        if not all(r["passed"] for r in results):
            raise SystemExit(1)
    finally:
        checked(driver.cuModuleUnload(module))
        del context_owner


if __name__ == "__main__":
    main()
