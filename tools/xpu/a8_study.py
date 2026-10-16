"""W4A8 study: final hidden states of prompt rows run with int8 activations (simulated, XPU_A8 of the Zig XPU engine)
against the same rows in fp16, both through the GPTQ checkpoint's fp16 head: top-1 agreement and KL(fp16 || int8).

  python3 a8_study.py GPTQ_DIR PREFIX NAME[,NAME...] VARIANT[,VARIANT...]   (files PREFIX_NAME_VARIANT.f16, base 'fp16';
  NAME_ids.txt: the prompt ids, for the perplexity of each run on its own text)

Use raw text, not a chat turn: the chat model is trained on assistant tokens only, and inside a user turn its next-token
predictions are meaningless (the fp32 reference itself puts a quarter of a user turn's rows above 8 nats, ref_rows.py).

2026-10-12, B70, GPTQ 4/128, three raw texts (English doc 6 681 tokens, Zig code 9 685, French plan 7 495), prompt rows:
  int8 by groups of 128   perplexity −0.01 / −0.01 / −0.08 %   top-1 97.6 / 98.7 / 97.7 %   KL 2.2e-3 / 3.2e-3 / 3.1e-3
  int8 one scale a row    perplexity +2.09 / +1.61 / +3.16 %   top-1 91.4 / 94.7 / 89.2 %   KL 4.1e-2 / 3.6e-2 / 5.8e-2
  MLX 4/64 (yardstick)    perplexity −1.80 / −0.60 / −0.76 %   top-1 88.8 / 94.3 / 85.2 %   KL 7.6e-2 / 6.1e-2 / 1.1e-1
Int8 activations by groups of 128 move the model 20-40 times less than the choice between the two 4-bit checkpoints.
"""
import json, struct, sys
import numpy as np
import torch

def tensor(path_dir, name):
    idx = json.load(open(f"{path_dir}/model.safetensors.index.json"))["weight_map"]
    f = f"{path_dir}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    a, b = hdr[name]["data_offsets"]
    m = np.memmap(f, dtype=np.uint8, mode="r", offset=8 + n + a, shape=(b - a,))
    return np.frombuffer(m, dtype=np.float16).reshape(hdr[name]["shape"])

torch.set_float32_matmul_precision("highest")
dev = "xpu"
gptq, prefix = sys.argv[1], sys.argv[2]
names, variants = sys.argv[3].split(","), sys.argv[4].split(",")
W = tensor(gptq, "lm_head.weight")
V, B = W.shape[0], 16384
blocks = [torch.from_numpy(np.ascontiguousarray(W[v0:v0 + B])).to(dev).float() for v0 in range(0, V, B)]

def logits_stats(H):
    """Per row: log-sum-exp, argmax, and the list of logits blocks (kept on the device)."""
    out = [H @ w.T for w in blocks]
    full = torch.cat(out, 1)
    return full

def nll(H, ids):
    """Mean negative log-likelihood of each row's next token (rows 0 .. n − 2), on the text's own tokens."""
    tot, n = 0.0, H.shape[0] - 1
    for r0 in range(0, n, 512):
        r1 = min(n, r0 + 512)
        lp = torch.log_softmax(logits_stats(H[r0:r1].to(dev)), 1)
        tgt = torch.tensor(ids[r0 + 1:r1 + 1], device=dev)
        tot += float(-lp.gather(1, tgt[:, None]).sum())
    return tot / n

for name in names:
    base = torch.from_numpy(np.fromfile(f"{prefix}_{name}_fp16.f16", dtype=np.float16).reshape(-1, 5120).astype(np.float32))
    ids = [int(x) for x in open(f"{name}_ids.txt").read().split()]
    nb = nll(base, ids)
    print(f"{name:5s} fp16   perplexity {np.exp(nb):.4f}  (NLL {nb:.4f})", flush=True)
    for var in variants:
        h = torch.from_numpy(np.fromfile(f"{prefix}_{name}_{var}.f16", dtype=np.float16).reshape(-1, 5120).astype(np.float32))
        agree, kls = [], []
        for r0 in range(0, base.shape[0], 512):
            lb = logits_stats(base[r0:r0 + 512].to(dev))
            lv = logits_stats(h[r0:r0 + 512].to(dev))
            pb = torch.log_softmax(lb, 1)
            pv = torch.log_softmax(lv, 1)
            kls.append((pb.exp() * (pb - pv)).sum(1).cpu())
            agree.append((lb.argmax(1) == lv.argmax(1)).cpu())
        nv = nll(h, ids)
        kl = torch.cat(kls).numpy()
        ag = torch.cat(agree).numpy()
        print(f"{name:5s} {var:6s} perplexity {np.exp(nv):.4f} ({100 * (np.exp(nv - nb) - 1):+.2f} %)  top-1 agree {100 * ag.mean():6.2f} %  KL mean {kl.mean():.2e}  p99 {np.percentile(kl, 99):.2e}", flush=True)
