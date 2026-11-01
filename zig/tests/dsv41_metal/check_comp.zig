//! Float64 checks of the compressed path: the compressor, the indexer's scores and selections, and CSA2 attention.
const std = @import("std");
const pkg = @import("dsv41_metal");
const ref = @import("ref.zig");
const probe = @import("probe.zig");

const Probe = probe.Probe;
const Meta = pkg.scratch.Meta;
const u = ref.u;
const R = pkg.pipelines.max_rows;

fn ratioAt(base: usize, ratio: u32) usize {
    return base + (ratio - 1) * R;
}

/// The KV source's compressor: projections, the ratio-2 pool, the norm, index keys and the FP4 pool rows.
pub fn compressor(p: *Probe, ratio: u32, carry: ?[]const f32) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const w = p.w;
    const D: usize = c.head_dim;
    const reg = p.r.slot.regions[w.layer];
    const all = probe.all(p.rows);
    const cpos = p.meta(ratioAt(Meta.cpos, ratio));
    const dest = p.meta(ratioAt(Meta.dest, ratio));
    var done_buf: [16]bool = undefined;
    const done = probe.setRows(cpos, p.rows, &done_buf);
    if (ratio == 1) {
        try p.mv("compressor wkv (bf16)", s.hn, w.comp_wkv.?, s.lat, c.hidden, D, false, all);
    } else {
        try p.mv("compressor wkv (fp32)", s.hn, w.comp_wkv.?, s.cf, c.hidden, D, true, all);
        try p.mv("compressor wgate (fp32)", s.hn, w.comp_wgate.?, s.cg, c.hidden, D, true, all);
        try pool(p, carry.?);
    }
    try p.norm("compressor norm", s.lat, w.comp_norm.?, s.latn, D, done);
    try p.mv("indexer wk", s.latn, w.ix_wk.?, s.ik, D, c.index_dim, false, done);
    try p.norm("indexer k_norm", s.ik, w.ix_k_norm.?, s.ikn, c.index_dim, done);
    try p.rope("rope(index k)", s.ikn, s.ikr, cpos, w.layer, 1, c.index_dim);
    try p.rope("rope(latent)", s.latn, s.latr, cpos, w.layer, 1, D);
    const keys = try p.stage("index keys (fp4 pack)");
    const pool_rows = try p.stage("pool rows (fp4 round trip)");
    const ikr = p.bfs(s.ikr, p.rows * c.index_dim);
    const latr = p.bfs(s.latr, p.rows * D);
    const kb = reg.keys.?.contents();
    const pb = reg.pool.?.slice(u16, pkg.kv.poolRows(c, w.layer, p.r.slot.limit) * D);
    for (0..p.rows) |r| {
        if (dest[r] < 0) continue;
        const at: usize = @intCast(dest[r]);
        var want: [68]u8 = undefined;
        for (0..4) |b| want[64 + b] = ref.fp4pack32(ikr[r * c.index_dim + 32 * b ..][0..32], want[16 * b ..][0..16]);
        keys.same(std.mem.eql(u8, &want, kb[at * 68 ..][0..68]), r, at);
        var rt: [16]u16 = undefined;
        for (0..D / 16) |b| {
            ref.fp4rt16(latr[r * D + 16 * b ..][0..16], &rt);
            pool_rows.same(std.mem.eql(u16, &rt, pb[at * D + 16 * b ..][0..16]), r, 16 * b);
        }
    }
}

/// The ratio-2 pool: slot pos % 2 filled by each row in order, odd positions pooled by a two-way softmax; the carry.
fn pool(p: *Probe, before: []const f32) !void {
    const s = try p.stage("compressor pool (ratio 2)");
    const D: usize = p.r.c.head_dim;
    const kv = p.f32s(p.r.s.cf, p.rows * D);
    const sc = p.f32s(p.r.s.cg, p.rows * D);
    const lat = p.bfs(p.r.s.lat, p.rows * D);
    const carry = p.r.slot.regions[p.w.layer].carry.?.slice(f32, 4 * D);
    for (0..D) |d| {
        var k2 = [2]f64{ before[d], before[D + d] };
        var s2 = [2]f64{ before[2 * D + d], before[3 * D + d] };
        for (0..p.rows) |r| {
            const slot = p.pos(r) & 1;
            k2[slot] = kv[r * D + d];
            s2[slot] = sc[r * D + d];
            if (slot == 0) continue;
            const m = @max(s2[0], s2[1]);
            const e0 = @exp(s2[0] - m);
            const e1 = @exp(s2[1] - m);
            const want = (k2[0] * e0 + k2[1] * e1) / (e0 + e1);
            const mag = (@abs(k2[0] * e0) + @abs(k2[1] * e1)) / (e0 + e1);
            s.add(ref.bf(lat[r * D + d]), want, ref.bfBound(want, mag, 10), r, d);
        }
        const kept = [4]f64{ k2[0], k2[1], s2[0], s2[1] };
        for (kept, 0..) |v, i| s.same(carry[i * D + d] == @as(f32, @floatCast(v)), p.rows, i * D + d);
    }
}

/// The index source: q projection, RoPE, FP4 queries, weights, scores, candidates and the top-512.
pub fn indexer(p: *Probe, ratio: u32) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const w = p.w;
    const IX: usize = c.index_heads * c.index_dim;
    try p.proj("indexer wq_b", s.qrq, w.ix_wq_b.?, s.ixq, false, probe.all(p.rows));
    try p.rope("rope(index q)", s.ixq, s.ixqr, p.meta(Meta.pos), w.layer, c.index_heads, c.index_dim);
    const qp = try p.stage("index q (fp4 pack)");
    const qr = p.bfs(s.ixqr, p.rows * IX);
    const qc = p.bytes(s.ixqc, p.rows * IX / 2);
    const qs = p.bytes(s.ixqs, p.rows * IX / 32);
    for (0..p.rows) |r| for (0..IX / 32) |b| {
        var codes: [16]u8 = undefined;
        const sb = ref.fp4pack32(qr[r * IX + 32 * b ..][0..32], &codes);
        qp.same(sb == qs[r * IX / 32 + b] and std.mem.eql(u8, &codes, qc[r * IX / 2 + 16 * b ..][0..16]), r, 32 * b);
    };
    try p.mv("indexer weights_proj", s.hn, w.ix_wp.?, s.ixw, c.hidden, c.index_heads, false, probe.all(p.rows));
    const keys = (p.p0 + @as(u32, @intCast(p.rows))) / ratio;
    const vis = p.meta(ratioAt(Meta.vis, ratio));
    try scores(p, keys, vis, c.usesCandidates(w.layer));
    if (c.isCandidateSource(w.layer)) try candidates(p, keys, vis);
    try top512(p, keys, vis);
}

fn qAt(p: *const Probe, r: usize, h: usize, d: usize) f64 {
    const IX: usize = p.r.c.index_heads * p.r.c.index_dim;
    const code = p.bytes(p.r.s.ixqc, p.rows * IX / 2)[r * IX / 2 + (h * 128 + d) / 2];
    const sc = p.bytes(p.r.s.ixqs, p.rows * IX / 32)[r * IX / 32 + (h * 128 + d) / 32];
    return @as(f64, ref.e2m1(if (d % 2 == 0) code & 15 else code >> 4)) * ref.e8m0(sc);
}

/// Index scores: sum over heads of relu(q . k) times the head's weight times 2^-6; unreachable keys are -inf.
fn scores(p: *Probe, keys: u32, vis: []const i32, masked: bool) !void {
    const st = try p.stage("index scores");
    const c = &p.r.c;
    const src = p.r.slot.source(c, p.w.layer).?;
    const kb = src.keys.?.contents();
    const sc = p.f32s(p.r.s.scores, p.rows * keys);
    const ws = p.bfs(p.r.s.ixw, p.rows * 32);
    const mask = p.bytes(p.r.s.cmask, p.rows * p.r.s.blocks);
    for (0..p.rows) |r| for (0..keys) |k| {
        const live = k < vis[r] and (!masked or mask[r * p.r.s.blocks + k / 8] != 0);
        if (!live) {
            st.same(sc[r * keys + k] == -std.math.inf(f32), r, k);
            continue;
        }
        var want: f64 = 0;
        var mag: f64 = 0;
        for (0..32) |h| {
            var dot: f64 = 0;
            var dmag: f64 = 0;
            for (0..128) |d| {
                const kc = kb[k * 68 + d / 2];
                const kv = @as(f64, ref.e2m1(if (d % 2 == 0) kc & 15 else kc >> 4)) * ref.e8m0(kb[k * 68 + 64 + d / 32]);
                dot += qAt(p, r, h, d) * kv;
                dmag += @abs(qAt(p, r, h, d) * kv);
            }
            const wh = @as(f64, ref.bf(ws[r * 32 + h])) * 0x1p-6;
            want += @max(dot, 0) * wh;
            mag += (@max(dot, 0) + dmag) * @abs(wh);
        }
        st.add(sc[r * keys + k], want, ref.fBound(want, mag, 72), r, k);
    };
}

/// The candidate blocks on the GPU's scores: block maxima, the newest pinned, the top `candidate_blocks`, lower first.
fn candidates(p: *Probe, keys: u32, vis: []const i32) !void {
    const st = try p.stage("candidate blocks");
    const nb = p.r.s.blocks;
    const sc = p.f32s(p.r.s.scores, p.rows * keys);
    const mask = p.bytes(p.r.s.cmask, p.rows * nb);
    const bs = try p.a.alloc(f32, nb);
    defer p.a.free(bs);
    for (0..p.rows) |r| {
        const last: i64 = if (vis[r] >= 1) @divFloor(vis[r] - 1, 8) else -1;
        for (bs, 0..) |*b, j| {
            b.* = -std.math.inf(f32);
            for (@min(8 * j, keys)..@min(8 * j + 8, keys)) |i| b.* = @max(b.*, sc[r * keys + i]);
            if (j == last) b.* = std.math.inf(f32);
        }
        for (bs, 0..) |b, j| {
            var rank: usize = 0;
            for (bs, 0..) |o, i| rank += @intFromBool(o > b or (o == b and i < j));
            const want = b > -std.math.inf(f32) and rank < p.r.c.candidate_blocks;
            st.same((mask[r * nb + j] != 0) == want, r, j);
        }
    }
}

fn topKey(v: f32) f32 {
    if (std.math.isNan(v)) return -std.math.inf(f32);
    return if (v == 0) 0 else v;
}

/// The top-512 on the GPU's scores: the largest min(visible, 512), ties to the lower index, ascending, then -1.
fn top512(p: *Probe, keys: u32, vis: []const i32) !void {
    const st = try p.stage("top-512");
    const width = @max(keys, 1);
    const sc = p.f32s(p.r.s.scores, p.rows * width);
    const sel = p.i32s(p.r.s.sel, p.rows * 512);
    const idx = try p.a.alloc(u32, width);
    defer p.a.free(idx);
    for (0..p.rows) |r| {
        const n = @min(@as(u32, @intCast(@max(vis[r], 0))), width);
        for (idx[0..n], 0..) |*v, i| v.* = @intCast(i);
        const Cmp = struct {
            row: []const f32,
            fn less(ctx: @This(), a: u32, b: u32) bool {
                const x = topKey(ctx.row[a]);
                const y = topKey(ctx.row[b]);
                return x > y or (x == y and a < b);
            }
        };
        std.mem.sort(u32, idx[0..n], Cmp{ .row = sc[r * width ..][0..n] }, Cmp.less);
        const keep = @min(n, 512);
        std.mem.sort(u32, idx[0..keep], {}, std.sort.asc(u32));
        var ok = true;
        for (0..512) |i| ok = ok and sel[r * 512 + i] == if (i < keep) @as(i32, @intCast(idx[i])) else -1;
        st.same(ok, r, keep);
    }
}

/// CSA2 attention in float64 over the window and the selected pool rows, with the sink and the inverse RoPE.
pub fn attention(p: *Probe) !void {
    const st = try p.stage("csa2 attention");
    const c = &p.r.c;
    const D: usize = c.head_dim;
    const H: usize = c.heads;
    const comp = c.compressRatio(p.w.layer) > 0;
    const reg = p.r.slot.regions[p.w.layer];
    const win = reg.window.slice(u16, 2 * @as(usize, pkg.kv.windowSlots(c)) * D);
    const rows_of = if (p.r.slot.source(c, p.w.layer)) |sr| sr.pool.?.contents() else null;
    const q = p.bfs(p.r.s.qrope, p.rows * H * D);
    const o = p.bfs(p.r.s.o, p.rows * H * D);
    const sel = p.i32s(p.r.s.sel, p.rows * 512);
    const sink = p.f32s(p.w.sink, H);
    const kind: pkg.rope.Kind = if (comp) .compressed else .window;
    const tab = p.r.tables.of(kind).slice(f32, p.r.tables.positions * 64);
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(D)));
    var keys: std.ArrayList([]const u16) = .empty;
    defer keys.deinit(p.a);
    const sv = try p.a.alloc(f64, 1024);
    defer p.a.free(sv);
    for (0..p.rows) |r| {
        keys.clearRetainingCapacity();
        const pos = p.pos(r);
        var t: u32 = pos -| (c.window - 1);
        while (t <= pos) : (t += 1) try keys.append(p.a, win[pkg.kv.windowRows(c, t)[0] * D ..][0..D]);
        if (comp) for (sel[r * 512 ..][0..512]) |i| if (i >= 0) {
            const at: usize = @intCast(i);
            try keys.append(p.a, @alignCast(std.mem.bytesAsSlice(u16, rows_of.?[at * D * 2 ..][0 .. D * 2])));
        };
        for (0..H) |h| {
            const qh = q[(r * H + h) * D ..][0..D];
            var m: f64 = sink[h];
            var es: f64 = 0;
            for (keys.items, 0..) |k, i| {
                var dot: f64 = 0;
                var mag: f64 = 0;
                for (0..D) |d| {
                    dot += @as(f64, ref.bf(qh[d])) * ref.bf(k[d]);
                    mag += @abs(@as(f64, ref.bf(qh[d])) * ref.bf(k[d]));
                }
                sv[i] = dot * scale;
                m = @max(m, sv[i]);
                es = @max(es, 22 * u * mag * scale + 2 * u * @abs(sv[i]) + 6 * u);
            }
            var den: f64 = @exp(sink[h] - m);
            for (keys.items, 0..) |_, i| den += @exp(sv[i] - m);
            var y: [512]f64 = undefined;
            var b: [512]f64 = undefined;
            for (0..D) |d| {
                var acc: f64 = 0;
                var mag: f64 = 0;
                for (keys.items, 0..) |k, i| {
                    const pr = @exp(sv[i] - m);
                    acc += pr * ref.bf(k[d]);
                    mag += pr * @abs(ref.bf(k[d]));
                }
                y[d] = acc / den;
                const n: f64 = @floatFromInt(keys.items.len + 8);
                b[d] = (2 * es + n * u) * mag / den;
            }
            for (0..D) |d| {
                var want = y[d];
                var bound = b[d];
                if (d >= D - 64) {
                    const pr = (d - (D - 64)) / 2;
                    const cs: f64 = tab[@as(usize, pos) * 64 + pr];
                    const sn: f64 = tab[@as(usize, pos) * 64 + 32 + pr];
                    const e = d - d % 2;
                    want = if (d % 2 == 0) y[e] * cs + y[e + 1] * sn else y[e + 1] * cs - y[e] * sn;
                    bound = @abs(cs) * b[e] + @abs(sn) * b[e + 1] + 3 * u * (@abs(y[e] * cs) + @abs(y[e + 1] * sn));
                }
                st.add(ref.bf(o[(r * H + h) * D + d]), want, ref.halfUlp(want + std.math.sign(want) * bound) + bound, r, h * D + d);
            }
        }
    }
}
