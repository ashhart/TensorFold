# Per-class time of the window graph instances in a hardware-traced nsys SQLite export.
import sqlite3, sys, collections, statistics as st
# usage: python3 nsys_classes.py TRACE.sqlite [GRAPH_ID]
db = sys.argv[1]
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
rows = c.execute("select graphId, start, end, shortName, demangledName, gridX*gridY*gridZ, blockX*blockY*blockZ, dynamicSharedMemory, streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null order by start").fetchall()
by = collections.defaultdict(list)
for r in rows: by[r[0]].append(r)
def cls(n, d):
    s = n.lower(); dd = d.lower()
    for key, lab in [("gather", "exchange"), ("roce", "exchange"), ("x3ld", "experts x3ld"), ("grouped", "experts x3ld"),
                     ("seg_linear", "dense exl3"), ("linear_kernel", "dense exl3"), ("x3dn", "dense exl3"), ("dense3", "dense exl3"), ("x3seg", "dense exl3"),
                     ("rot_in", "rot_in"), ("mhc", "mhc"), ("_site", "mhc"), ("_finish", "mhc"), ("gemv", "router"), ("router", "router"),
                     ("attn", "attention core"), ("_chunks", "attention core"), ("_merge", "attention core"), ("topk", "indexer/topk"), ("radix", "indexer/topk"),
                     ("_scores", "indexer/topk"), ("dtopk", "indexer/topk"), ("_kv_store", "attn small"), ("_rope", "attn small"), ("_rms", "norms"),
                     ("epilogue", "expert small"), ("combine", "expert small"), ("group", "expert small"), ("prune", "expert small"),
                     ("engram", "engram"), ("_fuse", "engram"), ("l2", "l2pf")]:
        if key in s or key in dd: return lab
    return "other"
out = {}
for g, ks in by.items():
    # split into instances by node count
    nodes = len(set(k[0] for k in ks))
    # instances: break when gap > 2 ms
    inst = []; cur = [ks[0]]
    for k in ks[1:]:
        if k[1] - cur[-1][2] > 2_000_000: inst.append(cur); cur = [k]
        else: cur.append(k)
    inst.append(cur)
    spans = [i[-1][2] - i[0][1] for i in inst]
    print(f"graph {g}: {len(inst)} instances, kernels/inst {st.median([len(i) for i in inst])}, span median {st.median(spans)/1e3:.0f} us")
    out[g] = inst
target = int(sys.argv[2]) if len(sys.argv) > 2 else None
if target is not None:
    inst = out[target]
    agg = collections.defaultdict(list); cnt = collections.Counter(); names_in = collections.defaultdict(collections.Counter)
    busy = []
    for i in inst:
        per = collections.defaultdict(int)
        for k in i:
            lab = cls(names.get(k[3], str(k[3])), names.get(k[4], ""))
            per[lab] += k[2] - k[1]
            names_in[lab][names.get(k[3], "?")] += 1
        for lab, v in per.items(): agg[lab].append(v)
        # union busy time
        iv = sorted((k[1], k[2]) for k in i); b = 0; s0, e0 = iv[0]
        for s, e in iv[1:]:
            if s > e0: b += e0 - s0; s0, e0 = s, e
            else: e0 = max(e0, e)
        b += e0 - s0; busy.append(b)
    n = len(inst)
    print(f"instances {n}, span med {st.median([i[-1][2]-i[0][1] for i in inst])/1e3:.1f} us, busy med {st.median(busy)/1e3:.1f} us, kernels {len(inst[0])}")
    for lab, v in sorted(agg.items(), key=lambda x: -st.median(x[1])):
        print(f"  {lab:16s} {st.median(v)/1e3:8.1f} us  launches/inst {sum(names_in[lab].values())/n:6.1f}  {dict(names_in[lab].most_common(4))}")
