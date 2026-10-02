import torch, time
from tensorfold.families.glm_moe_dsa.cuda.topk import top_columns, top_keys
dev = "cuda"; K = 2048
def pack(sc, pos_last):
    n, T = sc.shape
    bits = sc.contiguous().view(torch.int32)
    key = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(torch.int64)
    g = torch.arange(T, device=dev)
    packed = (key << 32) | (0x7FFFFFFF - g)
    ok = g[None, :] <= pos_last[:, None]
    return torch.where(ok, packed, torch.full_like(packed, -9223372036854775807))
def pack32(sc, pos_last):
    n, T = sc.shape
    bits = sc.contiguous().view(torch.int32)
    key = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    g = torch.arange(T, device=dev)
    ok = g[None, :] <= pos_last[:, None]
    return torch.where(ok, key ^ -2147483648, torch.zeros_like(key))
def ref(keys):
    top = torch.topk(keys, K, dim=-1, sorted=False).values
    return torch.sort((0x7FFFFFFF - (top & 0xFFFFFFFF)).to(torch.int32), dim=-1).values
bad = 0
for T in (2049, 4096, 32768, 131072):
    for kind in ("rand", "ties", "relu"):
        n = 128
        s = torch.randn(n, T, device=dev)
        if kind == "ties": s = (s * 4).round() / 4
        if kind == "relu": s = torch.relu(s - 1.0) * (torch.rand(n, T, device=dev) > 0.5)
        pos_last = torch.randint(K, T, (n,), device=dev); pos_last[0] = T - 1
        keys = pack(s, pos_last)
        r = ref(keys)
        o = torch.empty(n, K, dtype=torch.int32, device=dev); top_columns(keys, K, o)
        kk = torch.sort(top_keys(keys, K), -1).values; rk = torch.sort(torch.topk(keys, K, -1, sorted=False).values, -1).values
        o32 = torch.empty(n, K, dtype=torch.int32, device=dev); u32 = pack32(s, pos_last); top_columns(u32, K, o32)
        ok3 = torch.equal(o32, r)
        ok1, ok2 = torch.equal(o, r), torch.equal(kk, rk); bad += (not ok1) + (not ok2) + (not ok3)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): ref(keys)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        for _ in range(5): top_columns(keys, K, o)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        for _ in range(5): top_columns(u32, K, o32)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        print(f"T={T:6d} {kind:5s} exact64 {ok1} keys {ok2} exact32 {ok3}  torch {1e3*(t1-t0)/5:6.2f} ms  radix64 {1e3*(t2-t1)/5:6.2f}  radix32 {1e3*(t3-t2)/5:6.2f} ms ({(t1-t0)/(t3-t2):.1f}x)", flush=True)
print("ALL EXACT" if bad == 0 else f"MISMATCHES {bad}")
