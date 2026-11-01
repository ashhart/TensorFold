//! The drafter's float64 checks: the main rows into the rings, a draft block's stages, its head and the Markov chain.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const ref = @import("ref.zig");
const probe = @import("probe.zig");
const check_attn = @import("check_attn.zig");
const check_moe = @import("check_moe.zig");
const fixture_ds = @import("fixture_ds.zig");

const Probe = probe.Probe;
const Ref = probe.Ref;
const Ds = fixture_ds.Ds;
const Launch = pkg.dspark.Launch;
const u = ref.u;

fn ringU16(ds: *const Ds, which: usize) []const u16 {
    const c = &ds.r.c;
    const n = ds.pass.opt.slots * pkg.dspark_plan.poolRows(c) * c.head_dim;
    return ds.pass.rings[which].slice(u16, n);
}

/// main_proj, main_norm: the main rows' shared part, checked after `mainRows` ran (s.hq is its result).
pub fn mainRows(p: *Probe, ds: *const Ds, taps: Ref) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const K: u32 = ds.w.main_proj.k;
    const quant: Ref = .{ .buf = ds.pass.quant };
    const n: u32 = @intCast(p.rows);
    try p.actq("main act_quant(taps)", taps, quant, .{ .k = K, .nb = n, .sb = K, .tb = K });
    try p.proj("main_proj", quant, ds.w.main_proj, s.h, false, probe.all(p.rows));
    try p.norm("main_norm", s.h, ds.w.main_norm, s.hn, c.hidden, probe.all(p.rows));
    try p.actqRows("act_quant(main_x)", s.hn, s.hq, c.hidden);
}

/// A stage's key of rows (main rows or the block's) after `keys` ran: wkv .. act_quant, then the pool rows written.
pub fn keysOf(p: *Probe, ds: *const Ds, which: usize, at: []const i32, dest: []const i32) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const D: usize = c.head_dim;
    const rows = probe.all(p.rows);
    try p.proj("wkv", s.hq, p.w.wkv, s.kv, false, rows);
    try p.norm("kv_norm", s.kv, p.w.kv_norm, s.kvn, D, rows);
    try p.rope("rope(kv)", s.kvn, s.kvr, at, p.w.layer, 1, D);
    try p.actqRows("act_quant(kv)", s.kvr, s.kvq, D);
    const st = try p.stage("key write (pool rows)");
    const ring = ringU16(ds, which);
    const kq = p.bfs(s.kvq, p.rows * D);
    for (0..p.rows) |r| st.same(std.mem.eql(u16, ring[@as(usize, @intCast(dest[r])) * D ..][0..D], kq[r * D ..][0..D]), r, @intCast(dest[r]));
}

/// The block's ids in all four streams of every row, the pre mix one-hot.
pub fn embed(p: *Probe, ds: *const Ds, l: *const Launch) !void {
    const st = try p.stage("embed + expand");
    const c = &p.r.c;
    const H: usize = c.hidden;
    const table = p.bfs(ds.r.top.?.embed, @as(usize, c.vocab) * H);
    const x = p.bfs(ds.pass.b.streams[0], l.rows * 4 * H);
    const n = c.dspark_block;
    for (0..l.rows) |r| {
        const id: u32 = if (r % n == 0) l.asks[r / n].anchor else c.dspark_noise_token;
        for (0..4) |k| st.same(std.mem.eql(u16, x[(r * 4 + k) * H ..][0..H], table[id * H ..][0..H]), r, k);
    }
    const pre = p.f32s(ds.pass.b.pres[0], l.rows * 4);
    for (0..l.rows * 4) |i| st.same(pre[i] == @as(f32, if (i % 4 == 0) 1 else 0), i / 4, i % 4);
}

/// Attention of a draft block in float64: each row over its slot's context ring and all of the block's keys.
fn attention(p: *Probe, ds: *const Ds, l: *const Launch, which: usize, ctx_first: []const u64, ctx_count: []const u32) !void {
    const st = try p.stage("drafter attention");
    const c = &p.r.c;
    const D: usize = c.head_dim;
    const H: usize = c.heads;
    const n: usize = c.dspark_block;
    const ring = ringU16(ds, which);
    const q = p.bfs(p.r.s.qrope, p.rows * H * D);
    const o = p.bfs(p.r.s.o, p.rows * H * D);
    const sink = p.f32s(p.w.sink, H);
    const tab = p.r.tables.window.slice(f32, p.r.tables.positions * 64);
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(D)));
    var keys: std.ArrayList([]const u16) = .empty;
    defer keys.deinit(p.a);
    const sv = try p.a.alloc(f64, 1024);
    defer p.a.free(sv);
    for (0..p.rows) |r| {
        const slot = l.asks[r / n].slot;
        keys.clearRetainingCapacity();
        for (0..ctx_count[r / n]) |t| try keys.append(p.a, ring[pkg.dspark_plan.ringRow(c, slot, ctx_first[r / n] + t) * D ..][0..D]);
        for (0..n) |b| try keys.append(p.a, ring[pkg.dspark_plan.blockRow(c, slot, @intCast(b)) * D ..][0..D]);
        const pos: usize = l.at[r];
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
                const cnt: f64 = @floatFromInt(keys.items.len + 8);
                b[d] = (2 * es + cnt * u) * mag / den;
            }
            for (0..D) |d| {
                var want = y[d];
                var bound = b[d];
                if (d >= D - 64) {
                    const pr = (d - (D - 64)) / 2;
                    const cs: f64 = tab[pos * 64 + pr];
                    const sn: f64 = tab[pos * 64 + 32 + pr];
                    const e = d - d % 2;
                    want = if (d % 2 == 0) y[e] * cs + y[e + 1] * sn else y[e + 1] * cs - y[e] * sn;
                    bound = @abs(cs) * b[e] + @abs(sn) * b[e + 1] + 3 * u * (@abs(y[e] * cs) + @abs(y[e + 1] * sn));
                }
                st.add(ref.bf(o[(r * H + h) * D + d]), want, ref.halfUlp(want + std.math.sign(want) * bound) + bound, r, h * D + d);
            }
        }
    }
}

/// What the test knows of each ask's context independently of the pass: first position and count.
pub const Ctx = struct { first: []const u64, count: []const u32 };

/// One stage after it ran: every step of the draft block's layer against float64 on the GPU's own inputs.
pub fn stage(p: *Probe, ds: *const Ds, l: *const Launch, i: usize, x: []const u16, pre: []const f32, cx: Ctx) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const w = p.w;
    const D: usize = c.head_dim;
    const R = p.rows;
    const rows = probe.all(R);
    const io = ds.pass.io(i);
    var at: [16]i32 = undefined;
    var dest: [16]i32 = undefined;
    for (0..R) |r| {
        at[r] = @intCast(l.at[r]);
        dest[r] = @intCast(pkg.dspark_plan.blockRow(c, l.asks[r / c.dspark_block].slot, @intCast(r % c.dspark_block)));
    }
    try check_attn.hcMix(p, "hc mix (attn)", x, w.hc_attn, s.mix_a);
    try check_attn.hcPre(p, "hc pre (attn)", x, pre, s.h);
    try p.norm("attn_norm", s.h, w.attn_norm, s.hn, c.hidden, rows);
    try p.actqRows("act_quant(x)", s.hn, s.hq, c.hidden);
    try p.proj("wq_a", s.hq, w.wq_a, s.qa, false, rows);
    try p.norm("q_norm", s.qa, w.q_norm, s.qr, c.q_lora, rows);
    try p.actqRows("act_quant(qr)", s.qr, s.qrq, c.q_lora);
    try p.proj("wq_b", s.qrq, w.wq_b, s.q, false, rows);
    try p.rope("rope(q)", s.q, s.qrope, at[0..R], w.layer, c.heads, D);
    try keysOf(p, ds, i, at[0..R], dest[0..R]);
    try attention(p, ds, l, i, cx.first, cx.count);
    try check_attn.output(p);
    try check_attn.hcPost(p, "hc post (attn)", s.aout, x, s.mix_a, s.x1);
    try check_moe.check(p, io.pre_out, io.x_out);
}

/// The head: the last streams collapsed (hf), the drafter's norm, the shared head's logits.
pub fn head(p: *Probe, ds: *const Ds, x: []const u16, pre: []const f32) !void {
    const c = &p.r.c;
    const b = &ds.pass.b;
    try check_attn.hcPre(p, "hc pre (head)", x, pre, b.hf);
    try p.norm("drafter norm", b.hf, ds.w.norm, b.hfn, c.hidden, probe.all(p.rows));
    try p.mv("head logits", b.hfn, ds.r.top.?.head, b.logits, c.hidden, c.vocab, true, probe.all(p.rows));
}

/// The Markov chain of every ask in float64 on the GPU's own logits: the drafts (greedy) and the confidences.
pub fn chain(p: *Probe, ds: *const Ds, l: *const Launch, outs: []const fixture_ds.Out) !void {
    const dr = try p.stage("chain drafts");
    const cf = try p.stage("chain confidence");
    const c = &p.r.c;
    const V: usize = c.vocab;
    const D: usize = c.hidden;
    const rank: usize = c.dspark_markov_rank;
    const n: usize = c.dspark_block;
    const logits = p.f32s(ds.pass.b.logits, l.rows * V);
    const hf = p.bfs(ds.pass.b.hf, l.rows * D);
    const w1 = ds.w.markov_embed;
    const w2 = ds.w.markov_head;
    const cw = ds.w.conf.?;
    const z = try p.a.alloc(f64, V);
    defer p.a.free(z);
    const dz = try p.a.alloc(f64, V);
    defer p.a.free(dz);
    for (l.asks, outs, 0..) |a, o, j| {
        var prev: u32 = a.anchor;
        for (0..n) |i| {
            const r = j * n + i;
            var best: usize = 0;
            for (0..V) |id| {
                var s: f64 = 0;
                var mag: f64 = 0;
                for (0..rank) |k| {
                    const t = @as(f64, ref.bf(w1[prev * rank + k])) * ref.bf(w2[id * rank + k]);
                    s += t;
                    mag += @abs(t);
                }
                z[id] = logits[r * V + id] + s;
                dz[id] = @as(f64, @floatFromInt(rank + 2)) * u * mag + 2 * u * @abs(z[id]);
                if (z[id] > z[best]) best = id;
            }
            const got: usize = o.drafts[i];
            dr.add(z[got], z[best], dz[got] + dz[best], r, got);
            var s: f64 = 0;
            var mag: f64 = 0;
            for (0..D) |t| {
                const v = @as(f64, cw[t]) * ref.bf(hf[r * D + t]);
                s += v;
                mag += @abs(v);
            }
            for (0..rank) |k| {
                const v = @as(f64, cw[D + k]) * ref.bf(w1[prev * rank + k]);
                s += v;
                mag += @abs(v);
            }
            const want = 1 / (1 + @exp(-s));
            cf.add(o.conf[i], want, 0.25 * (@as(f64, @floatFromInt(D + rank + 8)) * u * mag) + 4 * u, r, got);
            prev = o.drafts[i];
        }
    }
}
