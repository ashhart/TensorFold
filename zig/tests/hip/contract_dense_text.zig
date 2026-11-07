//! Dense BF16 gather and Qwen2 RoPE under the contract, including cache holes and the full 131072-position range.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const iv = core.l2_ref;
const bf = core.bf16_round.bf16;
const vocab = 151936;
const context = 131072;
const dim = 128;

pub const Which = enum { embed, query, cache };
fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec(w: Which) c.Spec {
    return .{
        .name = switch (w) {
            .embed => "hip dense BF16 embedding",
            .query => "hip Torch BF16 query RoPE",
            .cache => "hip Torch BF16 cache RoPE",
        },
        .accum = .none,
        .depth = depth,
        .out = if (w == .query) .f32 else .bf16,
        .tiles = .{ .{ .size = 256 }, .{}, .{} },
        .served = switch (w) {
            .embed => &.{ .{ .rows = context, .n = 1536, .k = vocab }, .{ .rows = 17, .n = 1536, .k = vocab } },
            .query => &.{ .{ .rows = context, .n = 12, .k = dim }, .{ .rows = 17, .n = 12, .k = dim } },
            .cache => &.{ .{ .rows = context, .n = 2, .k = dim }, .{ .rows = 17, .n = 2, .k = dim } },
        },
    };
}

pub const Text = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    ids: []i32 = &.{},
    positions: []i32 = &.{},
    low: []f64 = &.{},
    high: []f64 = &.{},
    dx: u64 = 0,
    di: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Text {
        return .{ .rig = .{ .a = a, .r = r, .s = s }, .l = l, .which = w, .arena = .init(a) };
    }
    pub fn deinit(t: *Text) void {
        t.arena.deinit();
        t.rig.deinit();
    }
    const vt = g.hooks(Text);
    pub fn kernel(t: *Text) g.Kernel {
        return .{ .spec = spec(t.which), .ctx = t, .vt = &vt };
    }

    pub fn prepare(t: *Text, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > context or s.n == 0 or s.n > 1536 or
            (t.which == .embed and s.k != vocab) or (t.which != .embed and s.k != dim)) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        const a = t.arena.allocator();
        t.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const input_n = if (t.which == .embed) vocab * s.n else s.rows * s.n * dim;
        t.x = try a.alloc(u16, input_n);
        for (t.x, 0..) |*x, i| x.* = c.toBf16(if (t.which == .embed)
            @as(f32, @floatFromInt((i *% 1031) % 4096)) / 256 - 8
        else
            rng.float(f32) * 16 - 8);
        if (t.which == .embed) {
            t.ids = try a.alloc(i32, s.rows);
            for (t.ids, 0..) |*id, i| id.* = @intCast((i * 7919) % vocab);
            const samples = [_]i32{ 0, vocab / 2, vocab - 1, -1, vocab };
            for (samples, 0..) |id, i| if (i < t.ids.len) {
                t.ids[i] = id;
            };
            if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, samples[0..3], 0..) |x, id, i| {
                t.x[@as(usize, @intCast(id)) * s.n + i % s.n] = c.toBf16(x);
            };
            t.di = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        } else {
            t.positions = try a.alloc(i32, s.rows);
            for (t.positions, 0..) |*p, i| p.* = @intCast(if (t.which == .cache) context - 1 - i else i);
            if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |x, i| if (i < s.rows) {
                t.x[i * s.n * dim + i] = c.toBf16(x);
            };
            t.di = try t.rig.input(std.mem.sliceAsBytes(t.positions));
        }
        t.dx = try t.rig.input(std.mem.sliceAsBytes(t.x));
        const n = t.count();
        t.out = try t.rig.output(n * (if (t.which == .query) @as(usize, 4) else 2));
        if (t.which == .cache) {
            const initial = try a.alloc(u16, n);
            @memset(initial, c.toBf16(1.25));
            try t.rig.put(t.out, std.mem.sliceAsBytes(initial));
        }
        if (t.which != .embed) {
            t.low = try a.alloc(f64, n);
            t.high = try a.alloc(f64, n);
        }
        return true;
    }

    pub fn run(t: *Text) !void {
        var arena = hip.Arena.counting();
        const o: hip.ops.Ops = .{ .l = t.l, .bf16 = true, .stream = t.rig.s.handle, .arena = &arena };
        const s = t.shape;
        const out: hip.ops.Tensor = .{ .ptr = t.out, .kind = if (t.which == .query) .f32 else .bf16 };
        if (t.which == .embed) try hip.dense_text.embed(o, t.dx, t.di, out, s.rows, s.n, vocab) else try hip.dense_text.rope(o, .{ .ptr = t.dx, .kind = .bf16 }, t.di, out, s.rows, s.n, dim, 1000000, t.which == .cache, context);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Text) usize {
        return if (t.which == .embed) t.shape.rows * t.shape.n else (if (t.which == .cache) context else t.shape.rows) * t.shape.n * dim;
    }

    pub fn reference(t: *Text, want: []f64, mag: []f64) !void {
        const s = t.shape;
        if (t.which == .embed) {
            for (want, mag, 0..) |*y, *m, i| {
                const id = t.ids[i / s.n];
                y.* = if (id < 0 or id >= vocab) 0 else c.fromBf16(t.x[@as(usize, @intCast(id)) * s.n + i % s.n]);
                m.* = 0;
            }
            return;
        }
        @memset(want, 1.25);
        @memset(t.low, 1.25);
        @memset(t.high, 1.25);
        @memset(mag, 0);
        for (0..s.rows) |row| for (0..dim) |j| {
            const f = j % (dim / 2);
            const exponent = @as(f64, @floatFromInt(f)) / (dim / 2);
            const angle = @as(f64, @floatFromInt(t.positions[row])) / std.math.pow(f64, 1000000, exponent);
            const co = @cos(angle);
            const si = @sin(angle);
            // The exponent and positions are exact. Bound pow/reciprocal/angle rounding and sin/cos absolute error.
            const stray = (if (f == 0) 0 else 6 * 0x1p-24 * @abs(angle)) + 2 * 0x1p-24;
            const cos_range = iv.bf16(.{ co - stray, co + stray });
            const sin_range = iv.bf16(.{ si - stray, si + stray });
            const partner = if (j < dim / 2) j + dim / 2 else j - dim / 2;
            const sign: f64 = if (j < dim / 2) -1 else 1;
            for (0..s.n) |head| {
                const src = (row * s.n + head) * dim;
                const a: f64 = c.fromBf16(t.x[src + j]);
                const b: f64 = sign * c.fromBf16(t.x[src + partner]);
                const out_row: usize = if (t.which == .cache) @intCast(t.positions[row]) else row;
                const total: usize = if (t.which == .cache) context else s.rows;
                const dst = (head * total + out_row) * dim + j;
                want[dst] = bf(bf(a * bf(co)) + bf(b * bf(si)));
                const ar = iv.bf16(iv.mul(iv.point(a), cos_range, 0x1p-24));
                const br = iv.bf16(iv.mul(iv.point(b), sin_range, 0x1p-24));
                const ends = iv.bf16(iv.widen(.{ ar[0] + br[0], ar[1] + br[1] }, 0x1p-24));
                t.low[dst], t.high[dst] = ends;
            }
        };
    }

    pub fn bounds(t: *Text, want: []const f64, _: []const f64, allow: []f64) !void {
        if (t.which == .embed) {
            @memset(allow, 0);
            return;
        }
        for (want, allow, t.low, t.high) |w, *a, lo, hi| a.* = if (std.math.isFinite(w)) @max(@abs(w - lo), @abs(hi - w)) else 0;
    }

    pub fn intervals(t: *Text, low: []f64, high: []f64) !void {
        if (t.which == .embed) {
            const mag = try t.rig.a.alloc(f64, low.len);
            defer t.rig.a.free(mag);
            try t.reference(low, mag);
            @memcpy(high, low);
        } else {
            @memcpy(low, t.low);
            @memcpy(high, t.high);
        }
    }
};
