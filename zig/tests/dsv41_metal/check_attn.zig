//! The attention half of a layer's float64 checks: hc mixes, q and window KV, compressor, indexer, attention, output.
const std = @import("std");
const pkg = @import("dsv41_metal");
const ref = @import("ref.zig");
const probe = @import("probe.zig");

const Probe = probe.Probe;
const Ref = probe.Ref;
const Meta = pkg.scratch.Meta;
const u = ref.u;

/// What the checks need from before the launch: the input streams and pre mix, and the ratio-2 carry.
pub const Before = struct { x: []const u16, pre: []const f32, carry: ?[]const f32 };

/// The hc mixes of rows x [rows][4H] in float64, against the GPU's pre, post and comb.
pub fn hcMix(p: *Probe, name: []const u8, xs: []const u16, w: [3]Ref, out: [3]Ref) !void {
    const s = try p.stage(name);
    const c = &p.r.c;
    const dim: usize = 4 * c.hidden;
    const fnw = p.f32s(w[0], 24 * dim);
    const sc = p.f32s(w[1], 3);
    const base = p.f32s(w[2], 24);
    const depth: f64 = @floatFromInt(dim / 256 + 13);
    for (0..p.rows) |r| {
        var ss: f64 = 0;
        for (0..dim) |i| ss += @as(f64, ref.bf(xs[r * dim + i])) * ref.bf(xs[r * dim + i]);
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(dim)) + c.eps);
        var z: [24]f64 = undefined;
        var dz: [24]f64 = undefined;
        for (0..24) |m| {
            var dot: f64 = 0;
            var mag: f64 = 0;
            for (0..dim) |i| {
                const t = @as(f64, fnw[m * dim + i]) * ref.bf(xs[r * dim + i]);
                dot += t;
                mag += @abs(t);
            }
            const mix = dot * inv;
            const dmix = depth * u * mag * inv + @abs(mix) * (depth / 2 + 4) * u;
            const k: usize = if (m < 4) 0 else if (m < 8) 1 else 2;
            z[m] = mix * sc[k] + base[m];
            dz[m] = @abs(sc[k]) * dmix + 3 * u * (@abs(mix * sc[k]) + @abs(base[m]));
        }
        const pre = p.f32s(out[0], p.rows * 4);
        const post = p.f32s(out[1], p.rows * 4);
        const comb = p.f32s(out[2], p.rows * 16);
        for (0..4) |j| {
            const sg = 1 / (1 + @exp(-z[j]));
            s.add(pre[r * 4 + j], sg + c.hc_eps, dz[j] / 4 + 6 * u, r, j);
            const sp = c.hc_post_alpha / (1 + @exp(-z[4 + j]));
            s.add(post[r * 4 + j], sp, c.hc_post_alpha * (dz[4 + j] / 4 + 6 * u), r, 4 + j);
        }
        var cm: [4][4]f64 = undefined;
        var dzc: f64 = 0;
        for (0..16) |q| dzc = @max(dzc, dz[8 + q]);
        for (0..4) |j| {
            var mx: f64 = -std.math.inf(f64);
            for (0..4) |k| mx = @max(mx, z[8 + 4 * j + k]);
            var sum: f64 = 0;
            for (0..4) |k| {
                cm[j][k] = @exp(z[8 + 4 * j + k] - mx);
                sum += cm[j][k];
            }
            for (0..4) |k| cm[j][k] = cm[j][k] / sum + c.hc_eps;
        }
        for (0..c.hc_sinkhorn_iters) |it| {
            if (it > 0) for (0..4) |j| {
                const t = cm[j][0] + cm[j][1] + cm[j][2] + cm[j][3];
                for (0..4) |k| cm[j][k] /= t + c.hc_eps;
            };
            for (0..4) |k| {
                const t = cm[0][k] + cm[1][k] + cm[2][k] + cm[3][k];
                for (0..4) |j| cm[j][k] /= t + c.hc_eps;
            }
        }
        const iters: f64 = @floatFromInt(c.hc_sinkhorn_iters);
        for (0..4) |j| for (0..4) |k| s.add(comb[r * 16 + 4 * j + k], cm[j][k], cm[j][k] * (6 * iters + 16) * u + 2 * dzc, r, 8 + 4 * j + k);
    }
}

/// hc_pre: the four streams collapsed by `pre` (fp32), bf16 out.
pub fn hcPre(p: *Probe, name: []const u8, x: []const u16, pre: []const f32, y: Ref) !void {
    const s = try p.stage(name);
    const H = p.r.c.hidden;
    const ys = p.bfs(y, p.rows * H);
    for (0..p.rows) |r| for (0..H) |d| {
        var want: f64 = 0;
        var mag: f64 = 0;
        for (0..4) |k| {
            const t = @as(f64, pre[r * 4 + k]) * ref.bf(x[(r * 4 + k) * H + d]);
            want += t;
            mag += @abs(t);
        }
        s.add(ref.bf(ys[r * H + d]), want, ref.bfBound(want, mag, 5), r, d);
    };
}

/// hc_post: stream k = post[k] * branch + sum_j comb[j][k] * residual j, bf16 out.
pub fn hcPost(p: *Probe, name: []const u8, branch: Ref, res: []const u16, mix: [3]Ref, y: Ref) !void {
    const s = try p.stage(name);
    const H = p.r.c.hidden;
    const b = p.bfs(branch, p.rows * H);
    const post = p.f32s(mix[1], p.rows * 4);
    const comb = p.f32s(mix[2], p.rows * 16);
    const ys = p.bfs(y, p.rows * 4 * H);
    for (0..p.rows) |r| for (0..4) |k| for (0..H) |d| {
        var want = @as(f64, post[r * 4 + k]) * ref.bf(b[r * H + d]);
        var mag = @abs(want);
        for (0..4) |j| {
            const t = @as(f64, comb[r * 16 + 4 * j + k]) * ref.bf(res[(r * 4 + j) * H + d]);
            want += t;
            mag += @abs(t);
        }
        s.add(ref.bf(ys[(r * 4 + k) * H + d]), want, ref.bfBound(want, mag, 6), r, k * H + d);
    };
}

/// The attention half of layer w for the launch: every stage against float64 on the GPU's own inputs.
pub fn check(p: *Probe, before: Before) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const w = p.w;
    const l = w.layer;
    const D: usize = c.head_dim;
    const R = p.rows;
    const rows = probe.all(R);
    const pos = p.meta(Meta.pos);
    try hcMix(p, "hc mix (attn)", before.x, w.hc_attn, s.mix_a);
    try hcPre(p, "hc pre (attn)", before.x, before.pre, s.h);
    try p.norm("attn_norm", s.h, w.attn_norm, s.hn, c.hidden, rows);
    try p.actqRows("act_quant(x)", s.hn, s.hq, c.hidden);
    try p.proj("wq_a", s.hq, w.wq_a, s.qa, false, rows);
    try p.norm("q_norm", s.qa, w.q_norm, s.qr, c.q_lora, rows);
    try p.actqRows("act_quant(qr)", s.qr, s.qrq, c.q_lora);
    try p.proj("wq_b", s.qrq, w.wq_b, s.q, false, rows);
    try p.rope("rope(q)", s.q, s.qrope, pos, l, c.heads, D);
    try p.proj("wkv", s.hq, w.wkv, s.kv, false, rows);
    try p.norm("kv_norm", s.kv, w.kv_norm, s.kvn, D, rows);
    try p.rope("rope(kv)", s.kvn, s.kvr, pos, l, 1, D);
    try p.actqRows("act_quant(kv)", s.kvr, s.kvq, D);
    try window(p);
    const ratio = c.compressRatio(l);
    if (ratio > 0 and c.isKvSource(l)) try @import("check_comp.zig").compressor(p, ratio, before.carry);
    if (ratio > 0 and c.isIndexSource(l)) try @import("check_comp.zig").indexer(p, ratio);
    try @import("check_comp.zig").attention(p);
    try output(p);
    try hcPost(p, "hc post (attn)", s.aout, before.x, s.mix_a, s.x1);
}

/// The window ring: each row's quantised key in both mirror slots of its position.
fn window(p: *Probe) !void {
    const st = try p.stage("window write");
    const c = &p.r.c;
    const D: usize = c.head_dim;
    const reg = p.r.slot.regions[p.w.layer];
    const win = reg.window.slice(u16, 2 * @as(usize, pkg.kv.windowSlots(c)) * D);
    const kq = p.bfs(p.r.s.kvq, p.rows * D);
    for (0..p.rows) |r| for (pkg.kv.windowRows(c, p.pos(r))) |slot| {
        st.same(std.mem.eql(u16, win[slot * D ..][0..D], kq[r * D ..][0..D]), r, slot);
    };
}

/// act_quant per group, wo_a group by group, act_quant again, wo_b.
pub fn output(p: *Probe) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const w = p.w;
    const R: u32 = @intCast(p.rows);
    const G = c.o_groups;
    const kg = c.heads / G * c.head_dim;
    const ol = c.o_lora;
    try p.actq("act_quant(o) by group", s.o, s.oq, .{ .k = kg, .nb = R, .na = G, .sa = kg, .sb = c.heads * c.head_dim, .ta = R * kg, .tb = kg });
    for (0..G) |gi| {
        const wg: pkg.weights.Fp8 = .{ .w = pkg.ctx.plus(w.wo_a.w, gi * ol * kg), .s = pkg.ctx.plus(w.wo_a.s, gi * ol / 32 * (kg / 32)), .k = kg, .n = ol };
        try p.proj("wo_a (a group)", pkg.ctx.plus(s.oq, gi * R * kg * 2), wg, pkg.ctx.plus(s.oa, gi * R * ol * 2), false, probe.all(p.rows));
    }
    try p.actq("act_quant(o_a)", s.oa, s.ob, .{ .k = ol, .nb = R, .na = G, .sa = R * ol, .sb = ol, .ta = ol, .tb = G * ol });
    try p.proj("wo_b", s.ob, w.wo_b, s.aout, false, probe.all(p.rows));
}
