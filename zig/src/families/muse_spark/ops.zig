//! The decode forward's ops on a Model's buffers: norms, residual gates, EXL3 projections, attention and the MoE.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const load_ = @import("load.zig");
const exl3 = @import("exl3.zig");
const Model = fwd.Model;
const Stream = cuda.Stream;
const CK = fwd.CK;
const NORM_THREADS = fwd.NORM_THREADS;
const multi_k2s = fwd.multi_k2s;

fn u(x: usize) u32 {
    return @intCast(x);
}
fn ci(x: usize) c_int {
    return @intCast(x);
}

pub fn norm(m: *Model, s: Stream, x: u64, x_half: bool, w: u64, out: u64, n: usize, dim: usize, eps: f64) !void {
    return m.normParts(s, x, x_half, w, out, n, dim, eps, 0);
}

/// RMS norm of n rows; `parts` > 0: x is the ranks' gathered column slices of that width.
pub fn normParts(m: *Model, s: Stream, x: u64, x_half: bool, w: u64, out: u64, n: usize, dim: usize, eps: f64, parts: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, @intFromBool(x_half)));
    a.add(w);
    a.add(out);
    a.add(ci(dim));
    a.add(@as(f32, @floatCast(eps)));
    a.add(ci(parts));
    try cuda.launch.launch(m.f_rms, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = NORM_THREADS, .y = 1, .z = 1 } }, s, &a);
}

/// The ranks' fp32 partials [n, count]: all-gathered into recv (rank-major); returns (rank 0's, rank 1's or 0).
fn gather(m: *Model, s: Stream, part: u64, count: usize) ![2]u64 {
    const c = m.comm orelse return .{ part, 0 };
    try c.allGather(part, m.b.recv, count, .f32, s.handle);
    return .{ m.b.recv, m.b.recv + count * 4 };
}

/// res = res * alpha + rms(y') * beta (y the ranks' partials in rank order), then x = half(rms(res) * w_next).
pub fn gateNorm(m: *Model, s: Stream, part: u64, scale: u64, ab: u64, w_next: u64, xf: u64, n: usize) !void {
    const y = try gather(m, s, part, n * m.w.cfg.hidden);
    try gateNormAt(m, s, m.b.res, y, scale, ab, w_next, m.b.x, xf, n);
}

/// muse_gate_norm on explicit buffers: res [n, D] fp32 in place, y the ranks' partials, x [n, D] fp16 out.
pub fn gateNormAt(m: *Model, s: Stream, res: u64, y: [2]u64, scale: u64, ab: u64, w_next: u64, x: u64, xf: u64, n: usize) !void {
    const cfg = m.w.cfg;
    var a: cuda.Args = .{};
    a.add(res);
    a.add(y[0]);
    a.add(y[1]);
    a.add(scale);
    a.add(@as(c_int, @intFromBool(scale != 0)));
    a.add(@as(f32, @floatCast(cfg.post_eps)));
    a.add(ab);
    a.add(w_next);
    a.add(@as(f32, @floatCast(cfg.eps)));
    a.add(x);
    a.add(xf);
    a.add(ci(cfg.hidden));
    try m.rowLaunch(m.f_gate, m.f_gate_c, cfg.hidden, n, s, &a);
}

const Job = extern struct { xh: u64, t: u64, stride_k: i64, stride_nb: i64, svh: u64, bias: u64, y: u64, z: u64, counters: u64, y_dtype: c_int, k: c_int, n: c_int, sk: c_int, b0: c_int, pad: c_int = 0 };
const Batch = extern struct { j: [4]Job, n: c_int, m: c_int, pad: [2]c_int = .{ 0, 0 } };
const Rot = extern struct { suh: [4]u64, xh: [4]u64 };

/// q, k, v and gate (fp16): one rot_in and one linear launch when they share a width and plan, else four.
pub fn qkvg(m: *Model, s: Stream, L: load_.Layer, x: u64, q: u64, k: u64, v: u64, g: u64, n: usize) !void {
    const ls = [4]exl3.Linear{ L.q, L.k, L.v, L.g };
    const outs = [4]u64{ q, k, v, g };
    var ok = n <= 16; // wider windows: proj's 32-row passes
    const fi = std.mem.indexOfScalar(u32, &multi_k2s, L.q.k2);
    if (fi == null) ok = false;
    for (ls) |l| if (l.k2 != L.q.k2 or l.split[1] != 4 or l.k != L.q.k or l.bias != 0) {
        ok = false;
    };
    if (!ok) {
        for (ls, outs) |l, o| try m.proj(s, l, x, o, false, n);
        return;
    }
    const K = L.q.k;
    const D = m.w.cfg.hidden;
    var rot: Rot = undefined;
    var bt: Batch = .{ .j = undefined, .n = 4, .m = ci(n) };
    var b0: usize = 0;
    var zat = m.b.z4;
    for (ls, outs, 0..) |l, o, j| {
        rot.suh[j] = l.suh;
        rot.xh[j] = m.b.xh4 + j * m.rows * D * 2;
        const st = l.strides();
        const sk = l.split[0];
        bt.j[j] = .{ .xh = rot.xh[j], .t = l.words, .stride_k = st[0], .stride_nb = st[1], .svh = l.svh, .bias = 0, .y = o, .z = if (sk > 1) zat else 0, .counters = l.counters, .y_dtype = 0, .k = ci(l.k), .n = ci(l.n), .sk = ci(sk), .b0 = ci(b0) };
        if (sk > 1) zat += sk * n * l.n * 4;
        b0 += (l.n / 128) * sk;
    }
    {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, 0));
        a.add(rot);
        a.add(ci(K));
        try cuda.launch.launch(m.f_rotm, .{ .grid = .{ .x = u(((K / 128) + 3) / 4), .y = u(n), .z = 4 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }
    const f = m.f_multi[fi.?];
    const smem: u32 = u(4 * @as(usize, @min(n, 8)) * 128 * 4);
    var a: cuda.Args = .{};
    a.add(bt);
    try cuda.launch.launch(f, .{ .grid = .{ .x = u(b0), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 }, .shared = smem }, s, &a);
}

/// A glue kernel's cluster form (8 x 128 threads a row), or one 1024-thread block a row past its width limit.
pub fn rowLaunch(_: *const Model, f: cuda.Function, fc: cuda.Function, width: usize, n: usize, s: Stream, a: *cuda.Args) !void {
    if (width <= 16 * NORM_THREADS) {
        try cuda.launch.launch(fc, .{ .grid = .{ .x = u(n * 8), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, a);
    } else try cuda.launch.launch(f, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = NORM_THREADS, .y = 1, .z = 1 } }, s, a);
}

/// An EXL3 projection of n rows: x [n, K] fp16 -> out [n, N] fp16 or fp32.
pub fn proj(m: *Model, s: Stream, l: exl3.Linear, x: u64, out: u64, out_f32: bool, n: usize) !void {
    const z = if (l.split[0] > 1) m.b.z else 0;
    const ki = std.mem.indexOfScalar(u32, &multi_k2s, l.k2);
    const wi: ?usize = if (l.split[1] == 4) 0 else if (l.split[1] == 8) 1 else null;
    if (n <= 16 or ki == null or wi == null or n > 128) {
        return m.k3.linear(s, l, x, .f16, m.b.xh, out, if (out_f32) .f32 else .f16, z, n);
    }
    // the shared linear's arguments on linear_body2 (32-row passes), a (N / 128, SK) grid
    try m.k3.rotIn(s, x, .f16, l.suh, m.b.xh, n, l.k);
    const sk, const wk = l.split;
    if ((l.k / 16) % (sk * wk) != 0) return error.KSplit;
    const f = m.f_lin2[ki.?][wi.?];
    const smem: u32 = u(wk * @as(usize, @min(n, 8)) * 128 * 4);
    if (smem > 48 * 1024) try f.allowDynamicShared(smem);
    const st = l.strides();
    var a: cuda.Args = .{};
    inline for (.{ m.b.xh, l.words }) |p| a.add(p);
    a.add(st[0]);
    a.add(st[1]);
    inline for (.{ l.svh, l.bias, out }) |p| a.add(p);
    a.add(@as(c_int, if (out_f32) 2 else 0));
    a.add(z);
    a.add(l.counters);
    a.add(ci(n));
    a.add(ci(l.k));
    a.add(ci(l.n));
    a.add(ci(sk));
    try cuda.launch.launch(f, .{ .grid = .{ .x = u(l.n / 128), .y = u(sk), .z = 1 }, .block = .{ .x = u(wk * 32), .y = 1, .z = 1 }, .shared = smem }, s, &a);
}

/// A window's attention buffers: q, k, v, g in; qo scratch; att out; the rows' positions and slots.
pub const AttnIo = struct { q: u64, k: u64, v: u64, g: u64, qo: u64, att: u64, pos: u64, slot: u64 };

pub fn attention(m: *Model, s: Stream, i: usize, n: usize, chunks: usize) !void {
    return attentionIo(m, s, i, n, chunks, .{ .q = m.b.q, .k = m.b.k, .v = m.b.v, .g = m.b.g, .qo = m.b.qo, .att = m.b.att, .pos = m.b.pos, .slot = m.b.slot });
}

/// Attention of n <= rows rows (partials in the decode buffers): QK prep and cache write, chunks, merge + gate.
pub fn attentionIo(m: *Model, s: Stream, i: usize, n: usize, chunks: usize, io: AttnIo) !void {
    if (n > m.rows) return error.WindowTooWide;
    const cfg = m.w.cfg;
    const H = cfg.heads;
    const HK = cfg.kv_heads;
    const full = cfg.full[i];
    const len = if (full) m.context else m.ring;
    {
        var a: cuda.Args = .{};
        inline for (.{ io.q, io.k, io.v, m.cos, m.sin }) |p| a.add(p);
        a.add(@as(c_int, @intFromBool(cfg.rope(i))));
        inline for (.{ io.pos, io.slot, io.qo, m.kc[i], m.vc[i] }) |p| a.add(p);
        a.add(ci(len));
        a.add(ci(H));
        a.add(ci(HK));
        a.add(@as(f32, @floatCast(cfg.eps)));
        try cuda.launch.launch(m.f_prep, .{ .grid = .{ .x = u(n), .y = u(H + HK), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }
    const win: usize = if (full) 0 else cfg.window;
    const nch: usize = if (full) chunks else (cfg.window + 1 + CK - 1) / CK;
    {
        var a: cuda.Args = .{};
        inline for (.{ io.qo, m.kc[i], m.vc[i], io.pos, io.slot, m.b.po, m.b.pm, m.b.pl }) |p| a.add(p);
        a.add(ci(len));
        a.add(ci(nch));
        a.add(if (full) -ci((m.context + CK - 1) / CK * CK) else ci(CK)); // full: each row's own chunk width
        a.add(ci(win));
        a.add(ci(H));
        a.add(ci(HK));
        a.add(@as(f32, @floatCast(cfg.smScale())));
        try cuda.launch.launch(m.f_rows, .{ .grid = .{ .x = u(n), .y = u(HK), .z = u(nch) }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }
    {
        var a: cuda.Args = .{};
        inline for (.{ m.b.po, m.b.pm, m.b.pl }) |p| a.add(p);
        a.add(ci(nch));
        a.add(io.g);
        a.add(io.att);
        a.add(ci(H));
        a.add(@as(f32, @floatCast(cfg.eps)));
        a.add(@as(c_int, @intFromBool(full)));
        try cuda.launch.launch(m.f_merge, .{ .grid = .{ .x = u(n), .y = u(H / 8), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }
}

/// MuseSparkMoE: fp32 router, latent projection and norm, routed experts with the per-expert norm, into part.
pub fn moe(m: *Model, s: Stream, L: load_.Layer, n: usize) !void {
    const cfg = m.w.cfg;
    const E = cfg.experts;
    const K = cfg.top_k;
    const rs = s;
    {
        var a: cuda.Args = .{};
        a.add(m.b.xf);
        a.add(L.router);
        a.add(@as(c_int, @intFromBool(L.router_bf16)));
        a.add(m.b.logits);
        a.add(ci(cfg.hidden));
        a.add(ci(E));
        const rw: usize = 2; // 2 warps a block spread one row's 256 experts over every SM
        try cuda.launch.launch(m.f_router, .{ .grid = .{ .x = u((E + rw - 1) / rw), .y = u(n), .z = 1 }, .block = .{ .x = u(32 * rw), .y = 1, .z = 1 } }, rs, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(m.b.logits);
        a.add(L.bias);
        a.add(ci(E));
        a.add(ci(K));
        a.add(@as(f32, @floatCast(cfg.out_mult)));
        a.add(@as(f32, @floatCast(cfg.route_scale)));
        a.add(m.b.pick);
        a.add(m.b.wts);
        const t = std.math.ceilPowerOfTwo(usize, @max(E, 32)) catch unreachable;
        try cuda.launch.launch(m.f_topk, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = u(t), .y = 1, .z = 1 } }, rs, &a);
    }
    try m.proj(s, L.pre, m.b.x, m.b.lat, false, n);
    if (m.comm) |c| {
        const half = L.pre.n; // this rank's latent outputs
        try c.allGather(m.b.lat, m.b.latg, n * half, .f16, s.handle);
        try m.normParts(s, m.b.latg, true, L.pre_norm, m.b.latn, n, cfg.latent, cfg.eps, half);
    } else try m.norm(s, m.b.lat, true, L.pre_norm, m.b.latn, n, cfg.latent, cfg.eps);
    // routed experts (shared operator): group, rotate, gate / up, SwiGLU (fp32), down; per-slot outputs in sc.y
    const e = L.experts;
    const p = n * K;
    try m.k3.routed(s, e, m.b.latn, cfg.latent, m.b.pick, n, m.b.sc);
    {
        const y = try gather(m, s, m.b.sc.y, p * e.dims); // each rank's partial over its half of every expert's width
        var a: cuda.Args = .{};
        a.add(y[0]);
        a.add(y[1]);
        a.add(m.b.wts);
        a.add(L.expert_norm);
        a.add(@as(f32, @floatCast(cfg.post_eps)));
        a.add(m.b.acc);
        a.add(ci(cfg.latent));
        a.add(ci(K));
        a.add(ci(m.w.rank * L.post.k)); // this rank's post_expert_proj inputs
        a.add(ci(L.post.k));
        try m.rowLaunch(m.f_esum, m.f_esum_c, cfg.latent, n, s, &a);
    }
    try m.proj(s, L.post, m.b.acc, m.b.part, true, n);
}
