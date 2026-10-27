"""Group an EXL3 pack's safetensors tensors by role and print their sizes (reads headers only). Edit P for the pack directory."""
import json, struct, glob, collections
P = "/data/models/Qwen3.8-Flash-Next-exl3-3.05bpw"
g = collections.Counter()
def cat(n):
    for k in ("ngram_embedding", "ple_", ".ple."):
        if k in n and "ngram" in n: return "ple n-gram embedding shards"
    if ".ple." in n: return "ple other (projections)"
    if "mtp." in n: return "mtp (head)"
    if "experts." in n and "shared" not in n: return "routed experts"
    if "shared_expert" in n: return "shared expert"
    if "hyper_connection" in n: return "hyper-connection mixes"
    if "linear_attn" in n: return "linear (DeltaNet) attention"
    if "self_attn" in n: return "full attention"
    if "embed_tokens" in n: return "embed_tokens"
    if "lm_head" in n: return "lm_head"
    if "router" in n or "gate." in n: return "router"
    return "other: " + n.split(".")[-2] if "." in n else "other"
for f in sorted(glob.glob(P + "/*.safetensors")):
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]; hdr = json.loads(h.read(n))
    for name, m in hdr.items():
        if name == "__metadata__": continue
        a, b = m["data_offsets"]; g[cat(name)] += b - a
tot = sum(g.values())
for k, v in g.most_common(): print("%8.2f GB %5.1f%%  %s" % (v / 1e9, 100 * v / tot, k))
