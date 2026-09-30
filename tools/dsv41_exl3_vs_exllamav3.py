"""TensorFold EXL3 linear vs ExLlamaV3 LinearEXL3 on V4.1 tensors: `python X ref` in the vLLM image, then `python X tf` in tf-dev."""
import json, sys, torch
from safetensors import safe_open
root = "/models/dsv41/DeepSeek-V4.1-Flash-EXL3-2.9bpw/"
idx = json.load(open(root + "model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(root + idx[n], framework="pt", device="cuda:0") as f:
        return f.get_tensor(n)
NAMES = ["layers.0.attn.wq_a", "layers.0.attn.wq_b", "layers.3.attn.wo_b", "layers.3.ffn.experts.5.w1",
         "layers.20.ffn.experts.7.w2", "layers.0.ffn.shared_experts.w2", "head"]
side = sys.argv[1]
out = {}
for p in NAMES:
    t, suh, svh, m = get(p + ".trellis"), get(p + ".suh"), get(p + ".svh"), get(p + ".mul1")
    K, N = t.shape[0] * 16, t.shape[1] * 16
    g = torch.Generator(device="cuda").manual_seed(K + N)
    xs = {r: (torch.randn(r, K, device="cuda", generator=g) * 0.5).half() for r in (1, 7, 64)}
    if side == "ref":
        from exllamav3.modules.quant.exl3 import LinearEXL3
        lin = LinearEXL3(None, K, N, suh=suh, svh=svh, trellis=t, mul1=m)
        out[p] = {r: lin.forward(x, {}, out_dtype=torch.float).float().cpu() for r, x in xs.items()}
    else:
        from tensorfold.cuda.exl3.linear import Exl3Linear
        lin = Exl3Linear.from_tensors(t, suh, svh, "mul1")
        out[p] = {r: lin(x, out_dtype=torch.float32).float().cpu() for r, x in xs.items()}
torch.save(out, f"/tf/out/exl3cmp-{side}.pt")
if side == "tf":
    ref = torch.load("/tf/out/exl3cmp-ref.pt")
    for p in NAMES:
        for r in (1, 7, 64):
            a, b = ref[p][r], out[p][r]
            print(f"{p:32s} rows={r:3d} rel {((a - b).norm() / a.norm()).item():.2e} |ref| {a.norm():.1f} |tf| {b.norm():.1f}")
