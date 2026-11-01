//! The forward's own stages against float64: the embedding, Engram (hashes, rows, wkv, gate, add), taps, head.
const std = @import("std");
const pkg = @import("dsv41_metal");
const ref = @import("ref.zig");
const probe = @import("probe.zig");
const check_attn = @import("check_attn.zig");
const Fwd = @import("fixture_fwd.zig").Fwd;
const er = @import("dsv41").engram_rows;

const Probe = probe.Probe;
const Ref = probe.Ref;
const u = ref.u;

/// Every row's four streams are its token's table row, bit for bit, and the pre mix is (1, 0, 0, 0).
pub fn embed(p: *Probe, f: *const Fwd, ids: []const u32) !void {
    const st = try p.stage("embed + expand");
    const c = &p.r.c;
    const H: usize = c.hidden;
    const table = p.bfs(f.r.top.?.embed, @as(usize, c.vocab) * H);
    const x = p.bfs(f.b.streams[0], p.rows * 4 * H);
    for (0..p.rows) |r| for (0..4) |k| {
        st.same(std.mem.eql(u16, x[(r * 4 + k) * H ..][0..H], table[ids[r] * H ..][0..H]), r, k);
    };
    const pre = p.f32s(f.b.pres[0], p.rows * 4);
    for (0..p.rows * 4) |i| st.same(pre[i] == @as(f32, if (i % 4 == 0) 1 else 0), i / 4, i % 4);
}

/// engram.py's hash of the n-grams ending at index `at` of the history: one table row a column.
fn hash(f: *const Fwd, li: usize, hist: []const u32, at: usize, out: []u64) void {
    const h = &f.host;
    var rolling: u64 = 0;
    var blocked = false;
    for (0..h.max_ngram) |shift| {
        blocked = blocked or at < shift;
        const tok: u64 = if (blocked) @intCast(h.pad_id) else f.tokens[hist[at - shift]];
        rolling ^= tok *% @as(u64, @intCast(h.mult[li][shift]));
        if (shift == 0) continue;
        for (0..h.heads) |hd| {
            const col = (shift - 1) * h.heads + hd;
            out[col] = rolling % @as(u64, @intCast(h.primes[li][col])) + @as(u64, @intCast(h.offsets[li][col]));
        }
    }
}

/// Engram layer `li`: rows from `hist` (lookback, then the ids) and the shard, then each step on `x` (streams before).
pub fn engram(p: *Probe, f: *const Fwd, li: usize, hist: []const u32, x: []const u16) !void {
    const c = &p.r.c;
    const b = &f.b;
    const w = p.w.engram.?;
    const H: usize = c.hidden;
    const cols: usize = c.hashCols();
    const K = cols * 256;
    const N: usize = (c.hc_mult + 1) * H;
    const look = hist.len - p.rows;
    const rows_st = try p.stage("engram rows (hash + table)");
    const staged = p.bfs(b.rows[li], p.rows * K);
    var ids: [64]u64 = undefined;
    for (0..p.rows) |r| {
        hash(f, li, hist, look + r, ids[0..cols]);
        for (0..cols) |col| {
            const rec = f.shards[li][@intCast(er.header_bytes + ids[col] * er.row_bytes)..][0..er.row_bytes];
            for (0..256) |j| {
                const v = ref.e4m3(rec[j]) * ref.e8m0(rec[256 + j / 32]);
                rows_st.same(staged[r * K + col * 256 + j] == ref.toBf(v), r, col * 256 + j);
            }
        }
    }
    try p.actqRows("engram act_quant", b.rows[li], b.emb, K);
    try p.proj("engram wkv", b.emb, w.wkv, b.ekv, false, probe.all(p.rows));
    const kv = p.bfs(b.ekv, p.rows * N);
    const q = p.bfs(w.q, 4 * H);
    const k = p.bfs(w.k, 4 * H);
    const gates = p.f32s(b.gate, p.rows * 4);
    const gst = try p.stage("engram gate");
    const dscale: f64 = pkg.engram.dscale(c.hidden);
    const depth: f64 = @as(f64, @floatFromInt(H / 32)) + 5;
    for (0..p.rows) |r| for (0..4) |s| {
        var sh: f64 = 0;
        var sk: f64 = 0;
        var sd: f64 = 0;
        var msd: f64 = 0;
        for (0..H) |i| {
            const xv: f64 = ref.bf(x[(r * 4 + s) * H + i]);
            const kv_: f64 = ref.bf(kv[r * N + s * H + i]);
            const t = xv * (@as(f64, ref.bf(q[s * H + i])) * ref.bf(k[s * H + i])) * kv_;
            sh += xv * xv;
            sk += kv_ * kv_;
            sd += t;
            msd += @abs(t);
        }
        const fh: f64 = @floatFromInt(H);
        const rstd = 1 / @sqrt(sh / fh + c.eps) / @sqrt(sk / fh + c.eps);
        const dot = sd * rstd * dscale;
        const ddot = rstd * dscale * (depth * u * msd + @abs(sd) * (depth + 12) * u);
        const clamp: f64 = c.engram_gate_clamp;
        const z = std.math.copysign(@sqrt(@max(@abs(dot), clamp)), dot);
        var dz = ddot / (2 * @sqrt(@max(@abs(dot) - ddot, clamp))) + 4 * u * @abs(z);
        // a dot within its error of zero may round to either sign of the clamped root
        if (@abs(dot) <= ddot + clamp) dz = @max(dz, 2 * @sqrt(@max(@abs(dot) + ddot, clamp)));
        const want = 1 / (1 + @exp(-z));
        gst.add(gates[r * 4 + s], want, 0.25 * dz + 8 * u, r, s);
    };
    const ast = try p.stage("engram add");
    const y = p.bfs(b.streams[(p.w.layer) % 2], p.rows * 4 * H);
    for (0..p.rows) |r| for (0..4) |s| for (0..H) |i| {
        const xv: f64 = ref.bf(x[(r * 4 + s) * H + i]);
        const gv = @as(f64, gates[r * 4 + s]) * ref.bf(kv[r * N + 4 * H + i]);
        const want = xv + gv;
        ast.add(ref.bf(y[(r * 4 + s) * H + i]), want, ref.bfBound(want, @abs(xv) + @abs(gv), 2), r, s * H + i);
    };
}

/// The DSpark tap of target `t`: each row's four streams `x` averaged, at the slot's taps.
pub fn tap(p: *Probe, x: []const u16, t: usize) !void {
    const st = try p.stage("dspark tap (mean of the streams)");
    const c = &p.r.c;
    const H: usize = c.hidden;
    const T = c.dspark_targets.len;
    const taps = p.r.slot.taps.?.slice(u16, pkg.pipelines.max_rows * T * H);
    for (0..p.rows) |r| for (0..H) |i| {
        var want: f64 = 0;
        var mag: f64 = 0;
        for (0..4) |s| {
            const v: f64 = ref.bf(x[(r * 4 + s) * H + i]);
            want += v / 4;
            mag += @abs(v) / 4;
        }
        st.add(ref.bf(taps[(r * T + t) * H + i]), want, ref.bfBound(want, mag, 3), r, i);
    };
}

/// The head: the last streams `x` collapsed by the last pre mix, the final norm, the bf16 head in fp32.
pub fn head(p: *Probe, f: *const Fwd, x: []const u16, pre: []const f32) !void {
    const c = &p.r.c;
    const b = &f.b;
    try check_attn.hcPre(p, "hc pre (head)", x, pre, b.hf);
    try p.norm("final norm", b.hf, f.r.top.?.norm, b.hfn, c.hidden, probe.all(p.rows));
    try p.mv("head logits", b.hfn, f.r.top.?.head, b.logits, c.hidden, c.vocab, true, probe.all(p.rows));
}
