"""For `tf-cuda-test ling-layer-real`: one decoder layer's captured HF input, output and routing as a flat file."""
import argparse
import struct

import torch

import ling_hf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("chain", help="the captured chain: h<L>.pt (layer L's input), h<L+1>.pt (its output), r<L>.pt")
    p.add_argument("layer", type=int)
    p.add_argument("out", help="the fixture: x bf16 [T][D], HF's output fp32 [T][D], HF's top-k i32 [T][K] (K 0: dense)")
    p.add_argument("modeling", help="the folder of the HF modeling and configuration files the chain was captured with")
    a = p.parse_args()
    ling_hf.pinned(a.modeling)
    x = torch.load(f"{a.chain}/h{a.layer}.pt")[0].to(torch.bfloat16)
    y = torch.load(f"{a.chain}/h{a.layer + 1}.pt")[0].float()
    r = torch.load(f"{a.chain}/r{a.layer}.pt")
    topk = r["topk"][0].int() if "topk" in r else torch.zeros(x.shape[0], 0, dtype=torch.int32)
    T, D = x.shape
    with open(a.out, "wb") as f:
        f.write(b"TFLINGL1" + struct.pack("<4I", a.layer, T, D, topk.shape[1]))
        f.write(x.view(torch.int16).numpy().tobytes() + y.numpy().tobytes() + topk.numpy().tobytes())
    print(a.layer, r["type"], "tokens", T, "top", topk.shape[1], "clamps", r["LE"], r["LS"])


if __name__ == "__main__":
    main()
