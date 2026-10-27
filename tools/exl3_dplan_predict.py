"""Next-layer expert predictability from DPLAN v4 trace dumps (TF_FLASHNEXT_EXL3_DPLAN_TRACE_DIR). Usage: dir."""
import struct, glob, os, re, collections, sys
D = sys.argv[1]
pools = {}
for f in glob.glob(D + "/*.bin"):
    idx = int(re.search(r"pool(\d+)", f).group(1))
    w = struct.unpack("<%dI" % (os.path.getsize(f) // 4), open(f, "rb").read())
    layer, lines, maxr, sw, stride, cap, rec = w[3], w[4], w[5], w[6], w[7], w[8], w[9]
    off = 16 + 2 * sw
    rounds = []
    for k in range(rec):
        r = w[off + k * stride: off + (k + 1) * stride]
        rows = r[0]
        picks = [struct.unpack("<i", struct.pack("<I", x))[0] for x in r[stride - maxr * 11: stride - maxr * 11 + rows * 11]]
        rounds.append((rows, frozenset(e for e in picks if 0 <= e < 512)))
    pools[idx] = (layer, lines, rounds)
n = sorted(pools)
print("pools", len(n), "layers", [pools[i][0] for i in n][:5], "recorded", set(len(pools[i][2]) for i in n), "lines", set(pools[i][1] for i in n))

L = [i for i in n if len(pools[i][2]) > 0]
K = min(len(pools[i][2]) for i in L)
rows_ok = sum(1 for k in range(K) if len({pools[i][2][k][0] for i in L}) == 1)
print("layers with data", len(L), "rounds", K, "rounds with equal rows across layers", rows_ok)
sets = {i: [pools[i][2][k][1] for k in range(K)] for i in L}
LINES = pools[L[0]][1]
# per-layer LRU simulation -> miss sets
miss = {}
for i in L:
    cache = collections.OrderedDict()
    ms = []
    for k in range(K):
        need = sets[i][k]
        m = set(e for e in need if e not in cache)
        for e in need:
            if e in cache: cache.move_to_end(e)
        for e in m:
            while len(cache) >= LINES:
                for old in cache:
                    if old not in need:
                        del cache[old]; break
                else:
                    break
            cache[e] = 1
        ms.append(m)
    miss[i] = ms
tot_miss = sum(len(m) for i in L for m in miss[i]); tot_need = sum(len(s) for i in L for s in sets[i])
print("LRU(%d lines) misses per layer-round %.2f of %.2f needed (%.1f%%)" % (LINES, tot_miss / (K * len(L)), tot_need / (K * len(L)), 100 * tot_miss / tot_need))
T = K // 2
def evaluate(Ns, use_prev_round=False):
    # co-occurrence: layer i -> layer i+1 within the same round, trained on rounds < T
    out = {N: [0, 0, 0, 0] for N in Ns}  # covered misses, total misses, predicted-not-cached, useful(pred & need & notcached)
    for a, b in zip(L, L[1:]):
        co = collections.defaultdict(collections.Counter)
        freq = collections.Counter()
        for k in range(T):
            freq.update(sets[b][k])
            for e in sets[a][k]:
                co[e].update(sets[b][k])
        # replay cache state for layer b through test rounds again to know what's cached
        cache = collections.OrderedDict()
        for k in range(K):
            need = sets[b][k]
            if k >= T:
                score = collections.Counter()
                for e in sets[a][k]:
                    tot = sum(co[e].values()) or 1
                    for x, c in co[e].items(): score[x] += c / tot
                for x, c in freq.items(): score[x] += 0.01 * c / T
                ranked = [x for x, _ in score.most_common() if x not in cache]
                m = set(x for x in need if x not in cache)
                for N in Ns:
                    pred = set(ranked[:N])
                    out[N][0] += len(m & pred); out[N][1] += len(m); out[N][2] += len(pred); out[N][3] += len(m & pred)
            m = set(e for e in need if e not in cache)
            for e in need:
                if e in cache: cache.move_to_end(e)
            for e in m:
                while len(cache) >= LINES:
                    for old in cache:
                        if old not in need:
                            del cache[old]; break
                    else: break
                cache[e] = 1
    return out
res = evaluate([2, 4, 8, 16, 32])
for N, (cov, tm, pred, useful) in res.items():
    print("prefetch top-%d from previous layer's picks: covers %.1f%% of misses; %.2f copies issued per layer-round, %.1f%% of issued copies useful" % (N, 100 * cov / tm, pred / ((K - T) * (len(L) - 1)), 100 * useful / pred))
