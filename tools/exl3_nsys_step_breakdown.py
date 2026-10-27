"""Per-decode-step kernel time by category from an nsys SQLite export of a rank (counts fn_exl3_dplan launches / 48 as steps). Edit f."""
import sqlite3, collections, re, subprocess, os
f = "~/nsys80-r0"
if not os.path.exists(f + ".sqlite"):
    subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "-o", "/dev/null", f + ".nsys-rep"], capture_output=True)
c = sqlite3.connect(f + ".sqlite")
names = dict(c.execute("select id,value from StringIds"))
rows = c.execute("select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
dp = [r for r in rows if "fn_exl3_dplan" in names[r[2]]]
steps = len(dp) // 48
t0, t1 = dp[0][0], dp[-1][1]
print("dplan launches", len(dp), "decode steps", steps, "window s", round((t1 - t0) / 1e9, 2), "ms/step wall", round((t1 - t0) / 1e6 / steps, 2))
def cat(n):
    if "AllReduce" in n or "AllGather" in n: return "nccl all-reduce/gather"
    if "fn_exl3_dcopy" in n: return "expert miss copy (side stream)"
    if "tf_exl3_linear" in n or "tf_exl3_grouped" in n or "tf_exl3_group" in n or "tf_exl3_gateup" in n or "tf_exl3_down" in n or "tf_exl3_rot" in n: return "expert/dense exl3 matmuls"
    if "qmmf" in n: return "qmmf matmuls"
    if "fn_gdn" in n: return "DeltaNet (gdn)"
    if "fn_hc_" in n: return "hyper-connection"
    if "attn" in n or "kvp_" in n or "fn_qsa" in n: return "attention / kv paging"
    if "tf_route" in n or "router" in n: return "router"
    if "dplan" in n: return "device planner"
    return "other"
tot = collections.Counter(); cnt = collections.Counter(); other = collections.Counter()
for s, e, n in rows:
    if s < t0 or e > t1: continue
    nm = names[n]; k = cat(nm); tot[k] += e - s; cnt[k] += 1
    if k == "other": other[nm[:70]] += e - s
for k, v in tot.most_common():
    print("%-34s %7.2f ms/step  %5d launches/step" % (k, v / 1e6 / steps, cnt[k] // steps))
print("sum of kernel time (all streams) ms/step", round(sum(tot.values()) / 1e6 / steps, 2))
print("top 'other':")
for k, v in other.most_common(8): print("  %6.2f ms/step %s" % (v / 1e6 / steps, k))
# main stream busy fraction: union over all non-dcopy kernels
iv = sorted((s, e) for s, e, n in rows if t0 <= s and e <= t1 and "dcopy" not in names[n])
busy = 0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce: busy += ce - cs; cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs
print("busy fraction (non-copy kernels)", round(busy / (t1 - t0), 3))
