//! The image encoder's map kernels under the contract: convolution rows, SAM's windows and its relative positions.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const hub = @import("metal_uocr.zig");
const v = @import("metal_uocr_vision.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Mutation = v.Mutation;

const S = hub.sam_width;
const heads = 12;
const dh = 64;
const window = 14;

/// Patches (16 x 16, planar pixels), the neck's 3 x 3 and stride-2 rows, the windows, the relative positions.
pub const Kind = enum { cols_patch, cols_neck, cols_down, window, unwindow, rel };

fn function(k: Kind) []const u8 {
    return switch (k) {
        .cols_patch, .cols_neck, .cols_down => "uv_im2col",
        .window => "uv_window",
        .unwindow => "uv_unwindow_add",
        .rel => "uv_rel_pos",
    };
}

/// A dot of 64 in one fma chain.
fn relDepth(_: c.Shape) usize {
    return dh;
}

/// Rows: output positions, map positions, or query rows (whole side^2 groups); n: a row's width or the side; k: head.
pub fn spec(k: Kind) c.Spec {
    return switch (k) {
        .cols_patch => .{ .name = "uocr vision uv_im2col 16 x 16 stride 16 (patches)", .accum = .none, .depth = hub.noSums, .out = .bf16, .served = &.{ .{ .rows = 4096, .n = 768, .k = 1 }, .{ .rows = 1600, .n = 768, .k = 1 } } },
        .cols_neck => .{ .name = "uocr vision uv_im2col 3 x 3 pad 1 (neck)", .accum = .none, .depth = hub.noSums, .out = .bf16, .served = &.{ .{ .rows = 4096, .n = 256 * 9, .k = 1 }, .{ .rows = 1600, .n = 256 * 9, .k = 1 } } },
        .cols_down => .{ .name = "uocr vision uv_im2col 3 x 3 stride 2 pad 1 (net_2, net_3)", .accum = .none, .depth = hub.noSums, .out = .bf16, .served = &.{ .{ .rows = 1024, .n = 256 * 9, .k = 1 }, .{ .rows = 256, .n = 512 * 9, .k = 1 }, .{ .rows = 400, .n = 256 * 9, .k = 1 }, .{ .rows = 100, .n = 512 * 9, .k = 1 } } },
        .window => .{ .name = "uocr vision uv_window 14", .accum = .none, .depth = hub.noSums, .out = .bf16, .served = &.{ .{ .rows = 4096, .n = S, .k = 1 }, .{ .rows = 1600, .n = S, .k = 1 } } },
        .unwindow => .{ .name = "uocr vision uv_unwindow_add 14", .accum = .none, .depth = hub.noSums, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 4096, .n = S, .k = 1 }, .{ .rows = 1600, .n = S, .k = 1 } } },
        // windows of the 1024 view and of a tile, the global blocks of each
        .rel => .{ .name = "uocr vision uv_rel_pos (12 heads of 64)", .accum = .f32, .depth = relDepth, .out = .bf16, .served = &.{ .{ .rows = 25 * 196, .n = window, .k = dh }, .{ .rows = 9 * 196, .n = window, .k = dh }, .{ .rows = 4096, .n = 64, .k = dh }, .{ .rows = 1600, .n = 40, .k = dh } } },
    };
}

/// The least side whose square holds `n`.
fn sideOf(n: usize) usize {
    var s: usize = 1;
    while (s * s < n) s += 1;
    return s;
}

const Cols = extern struct { channels: u32, in_side: u32, out_side: u32, span: u32, stride: u32, pad: u32, cstride: u32, pstride: u32 };
const Win = extern struct { side: u32, padded: u32, window: u32, channels: u32 };
const Rel = extern struct { groups: u32, side: u32, heads: u32, width: u32 };

pub const Map = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    m: Mutation,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    cols: Cols = undefined,
    win: Win = undefined,
    rel: Rel = undefined,
    host: [3][]u16 = .{ &.{}, &.{}, &.{} },
    ins: [3]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, m: Mutation) !Map {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .m = m, .pipe = try v.pipeline(a, device, function(kind), m) };
    }
    pub fn deinit(t: *Map) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Map) void {
        for (&t.host) |*h| {
            t.a.free(h.*);
            h.* = &.{};
        }
    }
    pub fn kernel(t: *Map) g.Kernel {
        return v.mark(.{ .spec = spec(t.kind), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Map);

    fn input(t: *Map, i: usize, n: usize, rng: std.Random) ![]u16 {
        t.host[i] = try t.a.alloc(u16, n);
        v.fill(rng, t.host[i], -4, 4);
        return t.host[i];
    }

    pub fn prepare(t: *Map, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const P = hub.planted;
        const nf = fill == .nonfinite;
        switch (t.kind) {
            .cols_patch, .cols_neck, .cols_down => {
                const out_side = sideOf(shape.rows);
                const span: usize = if (t.kind == .cols_patch) 16 else 3;
                const ch = shape.n / (span * span);
                if (ch * span * span != shape.n) return false;
                const stride: usize = switch (t.kind) {
                    .cols_patch => 16,
                    .cols_neck => 1,
                    else => 2,
                };
                const in_side = out_side * stride;
                const planar = t.kind == .cols_patch;
                t.cols = .{ .channels = @intCast(ch), .in_side = @intCast(in_side), .out_side = @intCast(out_side), .span = @intCast(span), .stride = @intCast(stride), .pad = if (planar) 0 else 1, .cstride = @intCast(if (planar) in_side * in_side else 1), .pstride = @intCast(if (planar) 1 else ch) };
                const x = try t.input(0, ch * in_side * in_side, rng);
                if (nf) for (P, 0..) |val, i| {
                    x[(i * 977) % x.len] = c.toBf16(val);
                };
            },
            .window, .unwindow => {
                const side = sideOf(shape.rows);
                const padded = (side + window - 1) / window * window;
                t.win = .{ .side = @intCast(side), .padded = @intCast(padded), .window = window, .channels = @intCast(shape.n) };
                const x = try t.input(0, side * side * shape.n, rng);
                if (t.kind == .unwindow) _ = try t.input(1, padded * padded * shape.n, rng);
                if (nf) for (P, 0..) |val, i| {
                    x[(i * 977) % x.len] = c.toBf16(val);
                };
                if (nf and t.kind == .unwindow) t.host[1][0] = c.toBf16(P[1]); // with x's NaN at 0: NaN
            },
            .rel => {
                const side = shape.n;
                const groups = @max(1, shape.rows / (side * side));
                t.rel = .{ .groups = @intCast(groups), .side = @intCast(side), .heads = heads, .width = S };
                const n = groups * side * side;
                const qkv = try t.input(0, n * 3 * S, rng);
                const rh = try t.input(1, (2 * side - 1) * dh, rng);
                const rw = try t.input(2, (2 * side - 1) * dh, rng);
                if (fill == .stress) stress(rng, qkv, rh, rw, n);
                if (nf) {
                    qkv[5] = c.toBf16(P[0]); // row 0, head 0: every term of it NaN
                    rh[3] = c.toBf16(P[1]); // offset row 0: +inf where q's dim 3 is positive, -inf where negative
                    rw[dh + 1] = c.toBf16(P[2]);
                }
            },
        }
        for (t.host, 0..) |h, i| if (h.len > 0) {
            t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(h));
        };
        if (t.kind == .rel) {
            const each = t.count() / 2 * 2;
            t.outs[0] = try t.rig.output(each);
            t.outs[1] = try t.rig.output(each);
        } else t.outs[0] = try t.rig.output(t.count() * 2);
        return true;
    }

    /// Gate 4's q, tables: integers 128-255 (partial sums to 2^21), rows' halves negated mirrors (2 dims nudged).
    fn stress(rng: std.Random, qkv: []u16, rh: []u16, rw: []u16, n: usize) void {
        for (0..n) |r| for (0..heads) |h| {
            const q = qkv[r * 3 * S + h * dh ..][0..dh];
            for (0..dh / 2) |e| {
                q[e] = c.toBf16(@floatFromInt(128 + rng.uintLessThan(u32, 128)));
                q[dh - 1 - e] = q[e];
            }
        };
        for ([2][]u16{ rh, rw }) |tab| for (0..tab.len / dh) |r| {
            const row = tab[r * dh ..][0..dh];
            for (0..dh / 2) |e| {
                const x: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
                const nudge: f32 = if (e < 2) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 3))) - 1) else 0;
                row[e] = c.toBf16(x);
                row[dh - 1 - e] = c.toBf16(nudge - x);
            }
        };
    }

    pub fn run(t: *Map) !void {
        // the return of the windows adds in place: each launch starts from the prepared map copied into the output
        if (t.kind == .unwindow) @memcpy(t.rig.bytes(t.outs[0], t.host[0].len * 2), std.mem.sliceAsBytes(t.host[0]));
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        const tg = mtl.Size.of(256, 1, 1);
        switch (t.kind) {
            .cols_patch, .cols_neck, .cols_down => {
                e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
                e.setBuffer(t.outs[0].buf, t.outs[0].off, 1);
                e.setValue(t.cols, 2);
                const o = t.cols.out_side;
                e.dispatchThreads(mtl.Size.of(t.shape.n, o * o, 1), tg);
            },
            .window => {
                e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
                e.setBuffer(t.outs[0].buf, t.outs[0].off, 1);
                e.setValue(t.win, 2);
                e.dispatchThreads(mtl.Size.of(t.win.channels, t.win.padded * t.win.padded, 1), tg);
            },
            .unwindow => {
                e.setBuffer(t.outs[0].buf, t.outs[0].off, 0);
                e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
                e.setValue(t.win, 2);
                e.dispatchThreads(mtl.Size.of(t.win.channels, t.win.side * t.win.side, 1), tg);
            },
            .rel => {
                for (t.ins, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBuffer(t.outs[0].buf, t.outs[0].off, 3);
                e.setBuffer(t.outs[1].buf, t.outs[1].off, 4);
                e.setValue(t.rel, 5);
                const n = t.rel.groups * t.rel.side * t.rel.side;
                e.dispatchThreads(mtl.Size.of(2 * t.rel.side, n, heads), mtl.Size.of(32, 8, 1));
            },
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Map) usize {
        return switch (t.kind) {
            .cols_patch, .cols_neck, .cols_down => t.cols.out_side * t.cols.out_side * t.shape.n,
            .window => t.win.padded * t.win.padded * t.win.channels,
            .unwindow => t.win.side * t.win.side * t.win.channels,
            .rel => 2 * heads * t.rel.groups * t.rel.side * t.rel.side * t.rel.side,
        };
    }

    fn at(t: *Map, i: usize, j: usize) f64 {
        return c.fromBf16(t.host[i][j]);
    }

    /// Output i of a relative-position launch: its terms q_e R_e (into `terms`); h's outputs, then w's.
    fn relTerms(t: *Map, i: usize, terms: []f64) void {
        const side = t.rel.side;
        const n = t.rel.groups * side * side;
        const half = heads * n * side;
        const across = i >= half;
        const o = i % half;
        const h = o / (n * side);
        const row = (o / side) % n;
        const j = o % side;
        const in = row % (side * side);
        const from = (if (across) in % side else in / side) + side - 1 - j;
        for (terms, 0..) |*x, e| x.* = t.at(0, row * 3 * S + h * dh + e) * t.at(if (across) 2 else 1, from * dh + e);
    }

    pub fn reference(t: *Map, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        switch (t.kind) {
            .cols_patch, .cols_neck, .cols_down => {
                const p = t.cols;
                const width = t.shape.n;
                const taps = p.span * p.span;
                for (want, 0..) |*o, i| {
                    const r = i / width;
                    const col = i % width;
                    const ch = col / taps;
                    const tap = col % taps;
                    const iy = @as(isize, @intCast((r / p.out_side) * p.stride + tap / p.span)) - p.pad;
                    const ix = @as(isize, @intCast((r % p.out_side) * p.stride + tap % p.span)) - p.pad;
                    const inside = iy >= 0 and ix >= 0 and iy < p.in_side and ix < p.in_side;
                    o.* = if (inside) t.at(0, ch * p.cstride + (@as(usize, @intCast(iy)) * p.in_side + @as(usize, @intCast(ix))) * p.pstride) else 0;
                }
            },
            .window => {
                const w = t.win;
                for (want, 0..) |*o, i| {
                    const r = i / w.channels;
                    const win = r / (window * window);
                    const in = r % (window * window);
                    const across = w.padded / window;
                    const y = (win / across) * window + in / window;
                    const x = (win % across) * window + in % window;
                    o.* = if (y < w.side and x < w.side) t.at(0, (y * w.side + x) * w.channels + i % w.channels) else 0;
                }
            },
            .unwindow => {
                const w = t.win;
                for (want, 0..) |*o, i| {
                    const r = i / w.channels;
                    const y = r / w.side;
                    const x = r % w.side;
                    const across = w.padded / window;
                    const from = ((y / window) * across + x / window) * window * window + (y % window) * window + x % window;
                    o.* = t.at(0, i) + t.at(1, from * w.channels + i % w.channels);
                }
            },
            .rel => {
                var terms: [dh]f64 = undefined;
                for (want, mag, 0..) |*o, *m, i| {
                    t.relTerms(i, &terms);
                    o.* = 0;
                    for (terms) |x| {
                        o.* += x;
                        m.* += @abs(x);
                    }
                }
            },
        }
    }

    pub fn simulate(t: *Map, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (t.kind != .rel) return error.NoSums;
        var terms: [dh]f64 = undefined;
        for (out, 0..) |*o, i| {
            t.relTerms(i, &terms);
            o.* = rows.sumIn(acc, order, &terms);
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    made: std.ArrayList(*Map) = .empty,

    const controls = [_]struct { Kind, Mutation }{
        .{ .cols_neck, .cols_pad_down }, .{ .window, .window_transposed }, .{ .unwindow, .unwindow_no_add },
        .{ .rel, .rel_mirrored },        .{ .rel, .rel_half_sums },
    };

    fn one(s: *Suite, kind: Kind, m: Mutation, only: ?[]const u8) !void {
        if (!v.wanted(only, if (m == .none) spec(kind).name else m.name())) return;
        const x = try s.a.create(Map);
        x.* = try Map.init(s.a, s.device, s.queue, kind, m);
        try s.made.append(s.a, x);
        try s.kernels.append(s.a, x.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Kind)) |kind| try s.one(kind, .none, only);
        for (controls) |ctl| try s.one(ctl[0], ctl[1], only);
    }
    pub fn deinit(s: *Suite) void {
        for (s.made.items) |x| x.deinit();
        s.made.deinit(s.a);
    }
};
