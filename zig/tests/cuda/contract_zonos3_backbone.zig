//! ZONOS3's summing backbone kernels under the contract: the embedding, AttnRes and the GDN's convolution.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_zonos3_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

const D = 1024;
const columns = 11;

fn name(comptime plain: []const u8, comptime broken: []const u8, fault: bool) []const u8 {
    return if (fault) "control: zonos3 " ++ broken else "cuda zonos3 " ++ plain;
}

fn elevenDeep(_: c.Shape) usize {
    return columns;
}

/// Frames x width x the frame's eleven columns.
pub fn embedSpec(fault: bool) c.Spec {
    return .{ .name = name("z3_embed", "z3_embed reading its tables one row late", fault), .accum = .f32, .depth = elevenDeep, .out = .f32, .served = &.{ .{ .rows = 64, .n = D, .k = columns }, .{ .rows = 1, .n = D, .k = columns }, .{ .rows = 256, .n = D, .k = columns } } };
}

/// A frame row's eleven embeddings summed in column order: bf16 tables, an fp32 chain.
pub const Embed = struct {
    f: cuda.Function,
    /// control: the device takes each column's table to start one row late
    fault: bool = false,
    shape: c.Shape = undefined,
    ids: []i32 = &.{},
    tables: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};
    const table_rows = 16;

    pub fn spec(self: *const Embed) c.Spec {
        return embedSpec(self.fault);
    }

    pub fn control(self: *const Embed) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Embed, a: std.mem.Allocator) void {
        a.free(self.ids);
        a.free(self.tables);
        self.ids = &.{};
        self.tables = &.{};
    }

    pub fn prepare(self: *Embed, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != columns) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.ids = try a.alloc(i32, shape.rows * columns);
        for (self.ids, 0..) |*v, i| v.* = if (fill == .stress) @intCast(i % columns) else @intCast(rng.uintLessThan(u32, table_rows));
        // a row more than the columns read, for the control's shifted tables
        self.tables = try ref.bfs(a, rng, (columns * table_rows + columns) * n, -1, 1);
        if (fill == .stress) for (0..n) |i| {
            var terms: [columns]f32 = undefined;
            ref.stress(&terms, rng);
            for (terms, 0..) |t, col| self.tables[(col * table_rows + col) * n + i] = c.toBf16(t);
        };
        if (fill == .nonfinite) {
            const put = [_]f32{ ref.nan, ref.inf, -ref.inf };
            for (put, 0..) |v, r| if (r < shape.rows) {
                const row: usize = @intCast(self.ids[r * columns + r]);
                self.tables[((r * table_rows) + row) * n + (r + 1) * n / 5] = c.toBf16(v);
            };
        }
        self.dev = .{ try ref.input(rig, self.ids), try ref.input(rig, self.tables) };
        self.out = try rig.output(shape.rows * n * 4);
        return true;
    }

    pub fn run(self: *Embed, rig: *rig_.Rig) !void {
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(self.dev[1]);
        a.add(ref.int(table_rows + ref.one(self.fault)));
        a.add(self.out);
        a.add(ref.int(self.shape.n));
        try ref.go(self.f, rig.s, self.shape.rows, 1, 256, 0, &a);
    }

    pub fn count(self: *const Embed) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Embed, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        var scratch: [columns]f64 = undefined;
        for (out, 0..) |*o, at| {
            var terms: [columns]Iv = undefined;
            for (&terms, 0..) |*t, col| {
                const row: usize = @intCast(self.ids[at / n * columns + col]);
                t.* = Iv.of(ref.bf(self.tables[(col * table_rows + row) * n + at % n]));
            }
            o.* = s.sum(&terms, columns, &scratch);
        }
    }
};

const eps: f32 = 1e-5;

/// A thread's fma chain over the width, the warp's and the block's butterflies; then the sources in order.
fn attnresDepth(s: c.Shape) usize {
    return @max((s.n + 255) / 256 + 10, s.k);
}

/// Rows x width x sources: the embedding alone, a layer's bank so far, the whole bank before the head.
pub fn attnresSpec(fault: bool) c.Spec {
    return .{ .name = name("z3_attnres", "z3_attnres mixing one source fewer", fault), .accum = .f32, .depth = attnresDepth, .out = .f32, .served = &.{ .{ .rows = 16, .n = D, .k = 16 }, .{ .rows = 1, .n = D, .k = 1 }, .{ .rows = 1, .n = D, .k = 23 }, .{ .rows = 64, .n = D, .k = 5 } } };
}

/// A row's sources mixed by the softmax of each source's rms-normed dot with the query.
pub const AttnRes = struct {
    f: cuda.Function,
    /// control: the device mixes the first k - 1 sources
    fault: bool = false,
    shape: c.Shape = undefined,
    bank: []f32 = &.{},
    query: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const AttnRes) c.Spec {
        return attnresSpec(self.fault);
    }

    pub fn control(self: *const AttnRes) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *AttnRes, a: std.mem.Allocator) void {
        a.free(self.bank);
        a.free(self.query);
        self.bank = &.{};
        self.query = &.{};
    }

    pub fn prepare(self: *AttnRes, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k > 64) return false;
        self.free(a);
        self.shape = shape;
        // gate 4: sixteen sources and a zero query, so every weight is 1/16 and the mix sums whole numbers
        if (fill == .stress) self.shape.k = 16;
        const S = self.shape.k;
        const rows = shape.rows;
        const n = shape.n;
        self.bank = try ref.values(a, rng, S * rows * n, .random, 2);
        self.query = try ref.bfs(a, rng, n, -1, 1);
        if (fill == .stress) {
            @memset(self.query, 0);
            for (0..rows * n) |at| ref.stressAxis(self.bank, at, rows * n, S, rng);
            for (self.bank) |*v| v.* *= 16;
        }
        if (fill == .nonfinite) ref.plant(self.bank[0 .. rows * n], rows, n);
        self.dev = .{ try ref.input(rig, self.bank), try ref.input(rig, self.query) };
        self.out = try rig.output(rows * n * 4);
        return true;
    }

    pub fn run(self: *AttnRes, rig: *rig_.Rig) !void {
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(ref.int(self.shape.k - ref.one(self.fault)));
        a.add(ref.int(self.shape.rows));
        a.add(self.dev[1]);
        a.add(self.out);
        a.add(ref.int(self.shape.n));
        a.add(eps);
        try ref.go(self.f, rig.s, self.shape.rows, 1, 256, 0, &a);
    }

    pub fn count(self: *const AttnRes) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const AttnRes, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const S = self.shape.k;
        const rows = self.shape.rows;
        const n = self.shape.n;
        const depth = attnresDepth(self.shape);
        const terms = try a.alloc(Iv, n);
        defer a.free(terms);
        const scratch = try a.alloc(f64, @max(n, S));
        defer a.free(scratch);
        var w: [64]Iv = undefined;
        var mix: [64]Iv = undefined;
        for (0..rows) |r| {
            for (w[0..S], 0..) |*score, src| {
                const x = self.bank[(src * rows + r) * n ..][0..n];
                for (terms, x) |*t, v| t.* = ref.square(Iv.of(v));
                const inv = ref.rmsScale(s.sum(terms, depth, scratch), n, eps);
                for (terms, x, self.query) |*t, v, q| t.* = Iv.of(v).mul(inv).mul(Iv.of(ref.bf(q)));
                score.* = s.sum(terms, depth, scratch);
            }
            const total = ref.softmaxParts(s, w[0..S], S, scratch);
            for (w[0..S]) |*v| v.* = v.div(total);
            for (0..n) |i| {
                for (mix[0..S], w[0..S], 0..) |*t, weight, src| t.* = Iv.of(self.bank[(src * rows + r) * n + i]).mul(weight);
                out[r * n + i] = s.sum(mix[0..S], S, scratch);
            }
        }
    }
};

const conv_width = 1536;
const taps = 4;

fn tapsDeep(_: c.Shape) usize {
    return taps;
}

/// Rows x the GDN's q | k | v channels x the kernel's four taps.
pub fn convSpec(fault: bool) c.Spec {
    return .{ .name = name("z3_conv_silu (state in place)", "z3_conv_silu launched one row short", fault), .accum = .f32, .depth = tapsDeep, .out = .f32, .in_place = true, .served = &.{ .{ .rows = 64, .n = conv_width, .k = taps }, .{ .rows = 1, .n = conv_width, .k = taps }, .{ .rows = 7, .n = conv_width, .k = taps } } };
}

/// The GDN's causal depthwise convolution and SiLU; the state keeps a channel's last three inputs.
pub const ConvSilu = struct {
    f: cuda.Function,
    /// control: the device runs rows - 1 rows
    fault: bool = false,
    shape: c.Shape = undefined,
    proj: []f32 = &.{},
    state: []f32 = &.{},
    w: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub const sums = {};
    const pad = 8;

    pub fn spec(self: *const ConvSilu) c.Spec {
        return convSpec(self.fault);
    }

    pub fn control(self: *const ConvSilu) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *ConvSilu, a: std.mem.Allocator) void {
        a.free(self.proj);
        a.free(self.state);
        a.free(self.w);
        self.proj = &.{};
        self.state = &.{};
        self.w = &.{};
    }

    pub fn prepare(self: *ConvSilu, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != taps) return false;
        self.free(a);
        self.shape = shape;
        const T = shape.rows;
        const C = shape.n;
        const ps = C + pad;
        self.proj = try ref.values(a, rng, T * ps, .random, 2);
        self.state = try ref.values(a, rng, C * 3, .random, 2);
        self.w = try ref.bfs(a, rng, C * taps, -1, 1);
        if (fill == .stress) {
            for (self.w) |*v| v.* = c.toBf16(1);
            var line: [3 + 4]f32 = undefined;
            for (0..C) |ch| {
                ref.stress(line[0 .. 3 + @min(T, 4)], rng);
                @memcpy(self.state[ch * 3 ..][0..3], line[0..3]);
                for (0..T) |t| self.proj[t * ps + ch] = if (t < 4) line[3 + t] else 1;
            }
        }
        if (fill == .nonfinite) ref.plant(self.proj, T, ps);
        self.dev = .{ try ref.input(rig, self.proj), try ref.input(rig, self.w) };
        self.out = .{ try rig.output(T * C * 4), try rig.output(C * 3 * 4) };
        return true;
    }

    pub fn run(self: *ConvSilu, rig: *rig_.Rig) !void {
        const C = self.shape.n;
        try ref.restore(rig, self.out[1], std.mem.sliceAsBytes(self.state));
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(ref.int(C + pad));
        a.add(ref.int(self.shape.rows - ref.one(self.fault)));
        a.add(self.dev[1]);
        a.add(self.out[1]);
        a.add(self.out[0]);
        a.add(ref.int(C));
        try ref.go(self.f, rig.s, (C + 255) / 256, 1, 256, 0, &a);
    }

    pub fn count(self: *const ConvSilu) usize {
        return self.shape.rows * self.shape.n + self.shape.n * 3;
    }

    /// Channel `ch`'s input `i` rows after the state's first: the three the state holds, then the rows.
    fn at(self: *const ConvSilu, ch: usize, i: usize) f32 {
        return if (i < 3) self.state[ch * 3 + i] else self.proj[(i - 3) * (self.shape.n + pad) + ch];
    }

    pub fn solve(self: *const ConvSilu, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const T = self.shape.rows;
        const C = self.shape.n;
        var scratch: [taps]f64 = undefined;
        for (0..C) |ch| {
            for (0..T) |t| {
                var terms: [taps]Iv = undefined;
                for (&terms, 0..) |*term, j| term.* = Iv.of(self.at(ch, t + j)).mul(Iv.of(ref.bf(self.w[ch * taps + j])));
                out[t * C + ch] = ref.silu(s.sum(&terms, taps, &scratch));
            }
            for (0..3) |j| out[T * C + ch * 3 + j] = Iv.of(self.at(ch, T + j));
        }
    }
};
