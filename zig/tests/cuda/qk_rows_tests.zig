//! qk_rows.cu against core/qk_ref.zig: q, k within float64 tolerance, v and the caches exact, bits at any row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const qr = @import("qk_round_check.zig");

const counts = [_]usize{ 1, 3, 17, 300 };
const most = counts[counts.len - 1];
const positions = 4096;

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Case = struct { h: usize, hk: usize, dim: usize, rot: usize, norm: bool };

fn run1(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.qk_rows.QkRows, rng: std.Random, c: Case, worst: *f64) !void {
    const heads = c.h + 2 * c.hk;
    const width = heads * c.dim;
    const xs = try a.alloc(u16, most * width);
    for (xs, 0..) |*h, i| h.* = bf16(rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 8) else 1));
    const nw = try a.alloc(f32, 2 * c.dim);
    for (nw) |*w| w.* = 0.5 + rng.float(f32);
    const tc = try a.alloc(f32, positions * @max(c.rot, 1));
    const ts = try a.alloc(f32, positions * @max(c.rot, 1));
    for (0..positions) |p| for (0..c.rot) |i| {
        const j = i % (c.rot / 2);
        const ang = @as(f64, @floatFromInt(p)) / std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(c.rot)));
        tc[p * c.rot + i] = @floatCast(@cos(ang));
        ts[p * c.rot + i] = @floatCast(@sin(ang));
    };
    const len = 600;
    const pos = try a.alloc(i32, most);
    const slot = try a.alloc(i32, most);
    for (pos, slot, 0..) |*p, *sl, i| {
        p.* = @intCast((i * 7919 + 13) % positions);
        sl.* = @intCast(i % 2);
    }
    // distinct (slot, pos % len) a row, so no row's cache entry overwrites another's
    for (0..most) |i| pos[i] = @intCast((i / 2) % len + len * (i % 5));
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const up = struct {
        fn f(gp: check.Gpu, l: *std.ArrayList(cuda.DeviceBuffer), bytes: []const u8) !u64 {
            const b = try cuda.DeviceBuffer.fromHost(gp.d, bytes);
            try l.append(gp.gpa, b);
            return b.ptr;
        }
        fn z(gp: check.Gpu, l: *std.ArrayList(cuda.DeviceBuffer), st: cuda.Stream, n: usize) !u64 {
            const b = try cuda.DeviceBuffer.alloc(gp.d, n);
            try l.append(gp.gpa, b);
            try b.fill8(0xff, st.handle);
            return b.ptr;
        }
    };
    const dx = try up.f(gpu, &bufs, std.mem.sliceAsBytes(xs));
    const dnw = try up.f(gpu, &bufs, std.mem.sliceAsBytes(nw));
    const dc = try up.f(gpu, &bufs, std.mem.sliceAsBytes(tc));
    const ds = try up.f(gpu, &bufs, std.mem.sliceAsBytes(ts));
    const dp = try up.f(gpu, &bufs, std.mem.sliceAsBytes(pos));
    const dsl = try up.f(gpu, &bufs, std.mem.sliceAsBytes(slot));
    const shape: cuda.qk_rows.Shape = .{ .norms = if (c.norm) dnw else null, .cos = if (c.rot > 0) dc else null, .sin = if (c.rot > 0) ds else null, .rot = c.rot };
    const cache_bytes = 2 * len * c.hk * c.dim * 2;
    const outs = try a.alloc(cuda.qk_rows.Out, counts.len);
    for (counts, outs) |m, *o| {
        o.* = .{ .q = try up.z(gpu, &bufs, s, m * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, m * c.hk * c.dim * 2), .v = try up.z(gpu, &bufs, s, m * c.hk * c.dim * 2), .kc = try up.z(gpu, &bufs, s, cache_bytes), .vc = try up.z(gpu, &bufs, s, cache_bytes), .len = len };
        try g.rows(s, dx, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, shape, dp, dsl, o.*, m);
    }
    try s.synchronize();
    const o = outs[counts.len - 1];
    const back = struct {
        fn f(gp: check.Gpu, al: std.mem.Allocator, ptr: u64, n: usize) ![]u16 {
            const raw = try al.alloc(u16, n);
            const b: cuda.DeviceBuffer = .{ .d = gp.d, .ptr = ptr, .len = n * 2 };
            try b.download(0, std.mem.sliceAsBytes(raw));
            return raw;
        }
    };
    const gq = try back.f(gpu, a, o.q, most * c.h * c.dim);
    const gk = try back.f(gpu, a, o.k, most * c.hk * c.dim);
    const gv = try back.f(gpu, a, o.v, most * c.hk * c.dim);
    const gkc = try back.f(gpu, a, o.kc.?, cache_bytes / 2);
    const gvc = try back.f(gpu, a, o.vc.?, cache_bytes / 2);
    const xv = try a.alloc(f64, c.dim);
    const wq = try a.alloc(f64, c.dim);
    const wk = try a.alloc(f64, c.dim);
    for (wq, wk, 0..) |*p, *q, i| {
        p.* = nw[i];
        q.* = nw[c.dim + i];
    }
    const cv = try a.alloc(f64, @max(c.rot, 1));
    const sv = try a.alloc(f64, @max(c.rot, 1));
    const y = try a.alloc(f64, c.dim);
    var q_most: f64 = 0;
    for (0..most) |r| {
        const p: usize = @intCast(pos[r]);
        for (0..c.rot) |i| {
            cv[i] = tc[p * c.rot + i];
            sv[i] = ts[p * c.rot + i];
        }
        const at = @as(usize, @intCast(slot[r])) * len + p % len;
        for (0..heads) |hd| {
            for (xv, 0..) |*v, i| v.* = bf(xs[r * width + hd * c.dim + i]);
            if (hd >= c.h + c.hk) { // v: exact copies, in the output and the cache
                const k = hd - c.h - c.hk;
                for (0..c.dim) |i| {
                    try check.expect(gv[(r * c.hk + k) * c.dim + i] == xs[r * width + hd * c.dim + i], "qk rows: v copied", .{});
                    try check.expect(gvc[(at * c.hk + k) * c.dim + i] == xs[r * width + hd * c.dim + i], "qk rows: v cached", .{});
                }
                continue;
            }
            const is_q = hd < c.h;
            core.qk_ref.head(xv, if (c.norm) (if (is_q) wq else wk) else null, 1e-6, if (c.rot > 0) cv else null, if (c.rot > 0) sv else null, c.rot, y);
            for (y, 0..) |want, i| {
                const got = if (is_q) bf(gq[(r * c.h + hd) * c.dim + i]) else bf(gk[(r * c.hk + hd - c.h) * c.dim + i]);
                const tol = (@abs(want) + 2 * @abs(xv[i]) + 1e-3) * std.math.pow(f64, 2, -8);
                q_most = @max(q_most, @abs(got - want) / tol);
                if (!is_q) try check.expect(gkc[(at * c.hk + hd - c.h) * c.dim + i] == gk[(r * c.hk + hd - c.h) * c.dim + i], "qk rows: k cached as output", .{});
            }
        }
    }
    worst.* = @max(worst.*, q_most);
    // the same heads from [q | gate] rows and a padded k buffer, no v: the same q, k and k cache bits
    const qg = try a.alloc(u16, most * c.h * 2 * c.dim);
    const kb = try a.alloc(u16, most * (c.hk * c.dim + 64));
    @memset(qg, 0x7fc0); // the gate halves: NaN, never read
    @memset(kb, 0x7fc0);
    for (0..most) |r| {
        for (0..c.h) |hd| @memcpy(qg[(r * c.h + hd) * 2 * c.dim ..][0..c.dim], xs[r * width + hd * c.dim ..][0..c.dim]);
        @memcpy(kb[r * (c.hk * c.dim + 64) ..][0 .. c.hk * c.dim], xs[r * width + c.h * c.dim ..][0 .. c.hk * c.dim]);
    }
    const src: cuda.qk_rows.Src = .{ .q = try up.f(gpu, &bufs, std.mem.sliceAsBytes(qg)), .q_ld = @intCast(c.h * 2 * c.dim), .q_hs = @intCast(2 * c.dim), .k = try up.f(gpu, &bufs, std.mem.sliceAsBytes(kb)), .k_ld = @intCast(c.hk * c.dim + 64), .k_hs = @intCast(c.dim) };
    const o2: cuda.qk_rows.Out = .{ .q = try up.z(gpu, &bufs, s, most * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .v = 0, .kc = try up.z(gpu, &bufs, s, cache_bytes), .len = len };
    try g.rowsFrom(s, src, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, shape, dp, dsl, o2, most);
    try s.synchronize();
    try check.sameBytes("qk rows from strided heads: q", std.mem.sliceAsBytes(try back.f(gpu, a, o2.q, most * c.h * c.dim)), std.mem.sliceAsBytes(gq));
    try check.sameBytes("qk rows from strided heads: k", std.mem.sliceAsBytes(try back.f(gpu, a, o2.k, most * c.hk * c.dim)), std.mem.sliceAsBytes(gk));
    try check.sameBytes("qk rows from strided heads: k cache", std.mem.sliceAsBytes(try back.f(gpu, a, o2.kc.?, cache_bytes / 2)), std.mem.sliceAsBytes(gkc));
    // head-major outputs [heads, rows, dim] (strides): the packed call's bits, rearranged
    {
        const o4: cuda.qk_rows.Out = .{ .q = try up.z(gpu, &bufs, s, most * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .v = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .len = len, .q_ld = c.dim, .q_hs = most * c.dim, .kv_ld = c.dim, .kv_hs = most * c.dim };
        try g.rows(s, dx, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, shape, dp, dsl, o4, most);
        try s.synchronize();
        const hq = try back.f(gpu, a, o4.q, most * c.h * c.dim);
        const hk_ = try back.f(gpu, a, o4.k, most * c.hk * c.dim);
        const hv = try back.f(gpu, a, o4.v, most * c.hk * c.dim);
        for (0..most) |r| {
            for (0..c.h) |hd| try check.sameBytes("qk rows head-major: q", std.mem.sliceAsBytes(hq[(hd * most + r) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(gq[(r * c.h + hd) * c.dim ..][0..c.dim]));
            for (0..c.hk) |hd| {
                try check.sameBytes("qk rows head-major: k", std.mem.sliceAsBytes(hk_[(hd * most + r) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(gk[(r * c.hk + hd) * c.dim ..][0..c.dim]));
                try check.sameBytes("qk rows head-major: v", std.mem.sliceAsBytes(hv[(hd * most + r) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(gv[(r * c.hk + hd) * c.dim ..][0..c.dim]));
            }
        }
    }
    // bf16 norm weights at their own addresses and RoPE from inv (angle = pos * inv in fp32): float64 tolerance
    if (c.norm) {
        const wqb = try a.alloc(u16, c.dim);
        const wkb = try a.alloc(u16, c.dim);
        for (wqb, wkb, 0..) |*p, *q, i| {
            p.* = bf16(nw[i]);
            q.* = bf16(nw[c.dim + i]);
        }
        const half = @max(c.rot / 2, 1);
        const inv = try a.alloc(f32, half);
        for (inv, 0..) |*v, j| v.* = @floatCast(1 / std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(@max(c.rot, 1)))));
        const sh3: cuda.qk_rows.Shape = .{ .norms = try up.f(gpu, &bufs, std.mem.sliceAsBytes(wqb)), .norm_k = try up.f(gpu, &bufs, std.mem.sliceAsBytes(wkb)), .norms_bf16 = true, .inv = if (c.rot > 0) try up.f(gpu, &bufs, std.mem.sliceAsBytes(inv)) else null, .rot = c.rot };
        const o3: cuda.qk_rows.Out = .{ .q = try up.z(gpu, &bufs, s, most * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .v = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .len = len };
        try g.rows(s, dx, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, sh3, dp, dsl, o3, most);
        try s.synchronize();
        const q3 = try back.f(gpu, a, o3.q, most * c.h * c.dim);
        const k3 = try back.f(gpu, a, o3.k, most * c.hk * c.dim);
        for (wq, wk, 0..) |*p, *q, i| {
            p.* = bf(wqb[i]);
            q.* = bf(wkb[i]);
        }
        for (0..most) |r| {
            const p: usize = @intCast(pos[r]);
            for (0..c.rot) |i| {
                const ang: f64 = @as(f32, @floatFromInt(p)) * inv[i % half];
                cv[i] = @cos(ang);
                sv[i] = @sin(ang);
            }
            for (0..c.h + c.hk) |hd| {
                for (xv, 0..) |*v, i| v.* = bf(xs[r * width + hd * c.dim + i]);
                const is_q = hd < c.h;
                core.qk_ref.head(xv, if (is_q) wq else wk, 1e-6, if (c.rot > 0) cv else null, if (c.rot > 0) sv else null, c.rot, y);
                for (y, 0..) |want, i| {
                    const got = if (is_q) bf(q3[(r * c.h + hd) * c.dim + i]) else bf(k3[(r * c.hk + hd - c.h) * c.dim + i]);
                    q_most = @max(q_most, @abs(got - want) / ((@abs(want) + 2 * @abs(xv[i]) + 1e-3) * std.math.pow(f64, 2, -8)));
                }
            }
        }
        worst.* = @max(worst.*, q_most);
    }
    try check.expect(q_most <= 1, "qk rows {d}/{d}x{d}, rot {d}{s}: within float64 tolerance (worst {d:.3})", .{ c.h, c.hk, c.dim, c.rot, if (c.norm) ", normed" else "", q_most });
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, oo| {
        try check.sameBytes("qk rows: q at a smaller row count", std.mem.sliceAsBytes(try back.f(gpu, a, oo.q, m * c.h * c.dim)), std.mem.sliceAsBytes(gq[0 .. m * c.h * c.dim]));
        try check.sameBytes("qk rows: k at a smaller row count", std.mem.sliceAsBytes(try back.f(gpu, a, oo.k, m * c.hk * c.dim)), std.mem.sliceAsBytes(gk[0 .. m * c.hk * c.dim]));
    }
}

const Rope = enum { none, sliding, proportional, table };

/// A Gemma-shaped call: dim 256 or 512, the rotation, MLX's rows or one rounding, v's unit norm, v from k's columns.
const Wide = struct { h: usize, hk: usize, dim: usize, rope: Rope, round: bool, v_norm: bool, v_from_k: bool = false };

const Dev = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn deinit(d: *Dev) void {
        for (d.bufs.items) |*b| b.free();
        d.bufs.deinit(d.gpu.gpa);
    }

    fn up(d: *Dev, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(d.gpu.d, bytes);
        try d.bufs.append(d.gpu.gpa, b);
        return b.ptr;
    }

    fn poisoned(d: *Dev, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(d.gpu.d, @max(n, 2));
        try d.bufs.append(d.gpu.gpa, b);
        try b.fill8(0xff, d.s.handle);
        return b.ptr;
    }

    fn back(d: *Dev, a: std.mem.Allocator, ptr: u64, n: usize) ![]u16 {
        try d.s.synchronize();
        const raw = try a.alloc(u16, n);
        const b: cuda.DeviceBuffer = .{ .d = d.gpu.d, .ptr = ptr, .len = n * 2 };
        try b.download(0, std.mem.sliceAsBytes(raw));
        return raw;
    }
};

/// One Wide call at every row count: outputs in float64's ranges, caches as the outputs, bits equal at every count.
fn wide(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.qk_rows.QkRows, rng: std.Random, c: Wide, stats: *qr.Stats) !void {
    var d: Dev = .{ .gpu = gpu, .s = s };
    defer d.deinit();
    const ld = (c.h + (if (c.v_from_k) c.hk else 2 * c.hk)) * c.dim;
    const xs = try a.alloc(u16, most * ld);
    for (xs, 0..) |*h, i| h.* = bf16(rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 8) else 1));
    const wb = try a.alloc(u16, 2 * c.dim); // Gemma 4 12B's: q 0.95 to 1.05, k 0.06 to 0.13
    for (wb, 0..) |*v, i| v.* = bf16(if (i < c.dim) 0.9 + 0.2 * rng.float(f32) else 0.05 + 0.1 * rng.float(f32));
    const rot: usize = if (c.rope == .none) 0 else c.dim;
    const half = @max(rot / 2, 1);
    const inv = try a.alloc(f32, half); // families/gemma4/config.zig's inverseFrequencies
    for (inv, 0..) |*v, j| v.* = if (c.rope == .proportional and j >= 64) 0 else @floatCast(1 / std.math.pow(f64, if (c.rope == .proportional) 1e6 else 1e4, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(c.dim))));
    const tc = try a.alloc(f32, positions * @max(rot, 1));
    const ts = try a.alloc(f32, positions * @max(rot, 1));
    if (c.rope == .table) for (0..positions) |p| for (0..rot) |i| {
        const ang = @as(f64, @floatFromInt(p)) * inv[i % half];
        tc[p * rot + i] = @floatCast(@cos(ang));
        ts[p * rot + i] = @floatCast(@sin(ang));
    };
    const len = 600;
    const pos = try a.alloc(i32, most);
    const slot = try a.alloc(i32, most);
    for (pos, slot, 0..) |*p, *sl, i| { // distinct (slot, pos % len) a row
        p.* = @intCast((i / 2) % len + len * (i % 5));
        sl.* = @intCast(i % 2);
    }
    const cs = try a.alloc(f64, most * @max(rot, 1));
    const sn = try a.alloc(f64, most * @max(rot, 1));
    for (0..most) |r| for (0..rot) |i| {
        const p: usize = @intCast(pos[r]);
        const ang: f64 = @as(f32, @floatFromInt(p)) * inv[i % half];
        cs[r * rot + i] = if (c.rope == .table) tc[p * rot + i] else @cos(ang);
        sn[r * rot + i] = if (c.rope == .table) ts[p * rot + i] else @sin(ang);
    };
    const dx = try d.up(std.mem.sliceAsBytes(xs));
    const nw = try d.up(std.mem.sliceAsBytes(wb));
    var shape: cuda.qk_rows.Shape = .{ .norms = nw, .norms_bf16 = true, .rot = rot, .round = c.round, .v_norm = c.v_norm };
    switch (c.rope) {
        .none => {},
        .table => {
            shape.cos = try d.up(std.mem.sliceAsBytes(tc));
            shape.sin = try d.up(std.mem.sliceAsBytes(ts));
        },
        .sliding, .proportional => shape.inv = try d.up(std.mem.sliceAsBytes(inv)),
    }
    const dp = try d.up(std.mem.sliceAsBytes(pos));
    const dsl = try d.up(std.mem.sliceAsBytes(slot));
    const kp = dx + c.h * c.dim * 2;
    const src: cuda.qk_rows.Src = .{ .q = dx, .q_ld = @intCast(ld), .q_hs = @intCast(c.dim), .k = kp, .k_ld = @intCast(ld), .k_hs = @intCast(c.dim), .v = if (c.hk == 0) 0 else if (c.v_from_k) kp else kp + c.hk * c.dim * 2, .v_ld = @intCast(ld), .v_hs = @intCast(c.dim) };
    const cache = 2 * len * c.hk * c.dim;
    var outs: [counts.len]cuda.qk_rows.Out = undefined;
    for (counts, &outs) |m, *o| {
        o.* = .{ .q = try d.poisoned(m * c.h * c.dim * 2), .k = try d.poisoned(m * c.hk * c.dim * 2), .v = try d.poisoned(m * c.hk * c.dim * 2), .kc = if (c.hk > 0) try d.poisoned(cache * 2) else null, .vc = if (c.hk > 0) try d.poisoned(cache * 2) else null, .len = len };
        try g.rowsFrom(s, src, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, shape, dp, dsl, o.*, m);
    }
    const o = outs[counts.len - 1];
    const nq = most * c.h * c.dim;
    const nk = most * c.hk * c.dim;
    const out = try a.alloc(u16, nq + 2 * nk);
    @memcpy(out[0..nq], try d.back(a, o.q, nq));
    @memcpy(out[nq..][0..nk], try d.back(a, o.k, nk));
    @memcpy(out[nq + nk ..], try d.back(a, o.v, nk));
    const wv = try a.alloc(f64, 2 * c.dim);
    for (wv, wb) |*v, h| v.* = bf(h);
    const k: qr.Case = .{ .rows = most, .h = c.h, .hk = c.hk, .dim = c.dim, .rot = rot, .x = xs, .ld = ld, .v_from_k = c.v_from_k, .wq = wv[0..c.dim], .wk = wv[c.dim..], .cos = if (rot > 0) cs else null, .sin = if (rot > 0) sn else null, .computed = c.rope == .sliding or c.rope == .proportional, .points = if (c.round) .{ .row = true, .out = true } else .{}, .v_norm = c.v_norm };
    var msg: [200]u8 = undefined;
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "qk rows {d}/{d}x{d} {t}{s}{s}{s}", .{ c.h, c.hk, c.dim, c.rope, if (c.round) ", MLX's rows" else "", if (c.v_norm) ", unit v norm" else "", if (c.v_from_k) ", v from k" else "" });
    if (try qr.verify(a, k, out, stats, &msg)) |why| try check.expect(false, "{s}: {s}", .{ label, why });
    if (c.hk > 0) { // the caches hold what the outputs hold, at (slot * len + pos % len)
        const kc = try d.back(a, o.kc.?, cache);
        const vc = try d.back(a, o.vc.?, cache);
        for (0..most) |r| {
            const at = @as(usize, @intCast(slot[r])) * len + @as(usize, @intCast(pos[r])) % len;
            for (0..c.hk) |j| {
                try check.sameBytes(label, std.mem.sliceAsBytes(kc[(at * c.hk + j) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(out[nq + (r * c.hk + j) * c.dim ..][0..c.dim]));
                try check.sameBytes(label, std.mem.sliceAsBytes(vc[(at * c.hk + j) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(out[nq + nk + (r * c.hk + j) * c.dim ..][0..c.dim]));
            }
        }
    }
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, oo| {
        try check.sameBytes(label, std.mem.sliceAsBytes(try d.back(a, oo.q, m * c.h * c.dim)), std.mem.sliceAsBytes(out[0 .. m * c.h * c.dim]));
        try check.sameBytes(label, std.mem.sliceAsBytes(try d.back(a, oo.k, m * c.hk * c.dim)), std.mem.sliceAsBytes(out[nq..][0 .. m * c.hk * c.dim]));
        try check.sameBytes(label, std.mem.sliceAsBytes(try d.back(a, oo.v, m * c.hk * c.dim)), std.mem.sliceAsBytes(out[nq + nk ..][0 .. m * c.hk * c.dim]));
    }
    if (c.dim <= 256 and !c.round and c.v_norm) { // q and k as tf_qk_rows writes them: v's norm alone moves the call
        const o2: cuda.qk_rows.Out = .{ .q = try d.poisoned(nq * 2), .k = try d.poisoned(nk * 2), .v = try d.poisoned(nk * 2), .len = len };
        var plain = shape;
        plain.v_norm = false;
        try g.rowsFrom(s, src, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, plain, dp, dsl, o2, most);
        try check.sameBytes("qk rows once: q as tf_qk_rows'", std.mem.sliceAsBytes(try d.back(a, o2.q, nq)), std.mem.sliceAsBytes(out[0..nq]));
        try check.sameBytes("qk rows once: k as tf_qk_rows'", std.mem.sliceAsBytes(try d.back(a, o2.k, nk)), std.mem.sliceAsBytes(out[nq..][0..nk]));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.qk_rows.QkRows.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(31);
    var worst: f64 = 0;
    const a = arena.allocator();
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .dim = 128, .rot = 128, .norm = true }, &worst); // Kolibri, sliding
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .dim = 128, .rot = 0, .norm = true }, &worst); // Kolibri, full
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 8, .hk = 2, .dim = 64, .rot = 32, .norm = false }, &worst); // partial rotary
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 16, .hk = 2, .dim = 128, .rot = 128, .norm = false }, &worst); // a warp a head, no norm
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 24, .hk = 4, .dim = 256, .rot = 64, .norm = true }, &worst); // Qwen3.8-27B
    check.pass("qk rows: q and k within float64 tolerance (worst {d:.3}), v and the caches exact, bits at any row count", .{worst});
    var stats: qr.Stats = .{};
    for ([_]usize{ 16, 32 }) |h| { // Gemma 4 12B and 31B: sliding 2:1 x 256, global (k's columns as v) 16:1 and 8:1 x 512
        try wide(gpu, a, s, &g, prng.random(), .{ .h = h, .hk = h / 2, .dim = 256, .rope = .sliding, .round = true, .v_norm = true }, &stats);
        try wide(gpu, a, s, &g, prng.random(), .{ .h = h, .hk = if (h == 16) 1 else 4, .dim = 512, .rope = .proportional, .round = true, .v_norm = true, .v_from_k = true }, &stats);
    }
    for ([_]usize{ 256, 512 }) |dim| try wide(gpu, a, s, &g, prng.random(), .{ .h = 16, .hk = 0, .dim = dim, .rope = if (dim == 256) .sliding else .proportional, .round = true, .v_norm = false }, &stats); // the 12B's assistant: q only
    try wide(gpu, a, s, &g, prng.random(), .{ .h = 32, .hk = 4, .dim = 512, .rope = .table, .round = false, .v_norm = false }, &stats); // one rounding at 512
    try wide(gpu, a, s, &g, prng.random(), .{ .h = 32, .hk = 4, .dim = 512, .rope = .none, .round = true, .v_norm = true }, &stats);
    try wide(gpu, a, s, &g, prng.random(), .{ .h = 24, .hk = 4, .dim = 256, .rope = .table, .round = false, .v_norm = true }, &stats); // the 27B's shape, v normed
    try wide(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .dim = 128, .rope = .sliding, .round = false, .v_norm = true }, &stats);
    try std.testing.expectError(error.HeadShape, g.rows(s, 0, .{ .h = 1, .hk = 1, .dim = 544 }, .{}, 0, null, .{ .q = 0, .k = 0, .v = 0 }, 1));
    check.pass("qk rows dims to 512, MLX's rows, unit v norm: {d} outputs bit-equal to float64's at the rounding points, {d} within the fp32 inverse's or cosf's range, {d} non-finite as float64; caches as the outputs; every row count's bits equal", .{ stats.decided, stats.ranged, stats.nonfinite });
}
