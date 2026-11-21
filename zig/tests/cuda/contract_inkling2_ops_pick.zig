//! Inkling 2's picks and moves under the contract: tf_ink2_top1, tf_ink2_race1, tf_ink2_tobf and tf_ink2_put_words.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");
const M = @import("contract_inkling2_model.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const ops = ref.ops;

/// A rank's view of the head (shape.k: 1 rank 0, 2 rank 1): its first id, its rows' stride, the real ids it holds.
const Rank = struct { base: usize, stride: usize, n: usize };

fn model(mini: bool) M.Model {
    return if (mini) M.mini else M.small;
}

/// Mini at one rank: rows of the whole 201,024, its 200,058 real ids from 0.
fn rank(s: c.Shape, mini: bool) Rank {
    const share = model(mini).share();
    return if (s.k == 2) .{ .base = share, .stride = share, .n = s.n } else .{ .base = 0, .stride = share, .n = s.n };
}

const ranks = [_]c.Shape{ .{ .rows = 16, .n = ref.share, .k = 1 }, .{ .rows = 1, .n = ref.share, .k = 1 }, .{ .rows = 128, .n = ref.share, .k = 1 }, .{ .rows = 16, .n = ref.share1_real, .k = 2 }, .{ .rows = 1, .n = ref.share1_real, .k = 2 } };
const mini_rows = [_]c.Shape{ .{ .rows = 16, .n = M.mini.real, .k = 1 }, .{ .rows = 1, .n = M.mini.real, .k = 1 }, .{ .rows = 128, .n = M.mini.real, .k = 1 } };

const top_rule = "the largest value and its lowest id plus the rank's base; a NaN never picked, -inf picked when all else is NaN, a row of NaNs (-inf, base)";

/// Rows x a rank's real ids x the rank: greedy draws of a window's rows, the drafter's single row.
pub fn top1Spec(short: bool, mini: bool) c.Spec {
    const name = if (short) M.named("control: inkling2 ops tf_ink2_top1 one id short", mini) else M.named("inkling2 ops tf_ink2_top1 (greedy pick, fp32 logits)", mini);
    const served: []const c.Shape = if (mini) &mini_rows else &ranks;
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = top_rule }, .served = if (short) served[0..2] else served };
}

/// Values on 1/8 steps of [-64, 64) and padding past the real ids at 1000: equal maxima, and a read past n shows.
fn logits(a: std.mem.Allocator, rng: std.Random, s: c.Shape, fill: g.Fill, mini: bool) ![]f32 {
    const rk = rank(s, mini);
    const x = try a.alloc(f32, s.rows * rk.stride);
    for (x, 0..) |*v, i| v.* = if (i % rk.stride < rk.n) ref.int(rng, -512, 511) / 8 else 1000;
    if (fill != .nonfinite) {
        x[rk.n - 1] = 100; // row 0's unique maximum at its last real id
        return x;
    }
    const st = rk.stride;
    for ([_]usize{ 0, rk.n / 2, std.mem.indexOfMax(f32, x[0..rk.n]) }) |i| x[i] = ref.nan;
    if (s.rows > 1) for ([_]usize{ rk.n / 2, rk.n / 3 }) |i| {
        x[st + i] = ref.inf;
    };
    if (s.rows > 2) for (0..rk.n / 2) |i| {
        x[2 * st + i] = -ref.inf;
    };
    if (s.rows > 3) for (0..rk.n) |i| {
        x[3 * st + i] = ref.nan;
    };
    return x;
}

/// The host's scan of the rule: the first id of the largest value, NaNs skipped, none (n) for a row of NaNs.
fn scan(row: []const f32) struct { v: f32, i: usize } {
    var best = -std.math.inf(f32);
    var id: usize = row.len;
    for (row, 0..) |v, i| if (v > best or (v == best and id == row.len)) {
        best = v;
        id = i;
    };
    return .{ .v = best, .i = if (id == row.len) 0 else id };
}

/// Each row's (value, id) against the host's scan.
pub const Top1 = struct {
    kern: *const ops.Kernels,
    short: bool = false,
    mini: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    dev: u64 = 0,
    out: [2]u64 = @splat(0),

    pub fn spec(self: *const Top1) c.Spec {
        return top1Spec(self.short, self.mini);
    }

    pub fn control(self: *const Top1) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *Top1, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }

    pub fn prepare(self: *Top1, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n < 4 or shape.n > model(self.mini).share()) return false;
        self.free(a);
        self.shape = shape;
        self.x = try logits(a, rng, shape, fill, self.mini);
        self.dev = try ref.input(rig, self.x);
        self.out = .{ try rig.output(shape.rows * 4), try rig.output(shape.rows * 4) };
        return true;
    }

    pub fn run(self: *Top1, rig: *rig_.Rig) !void {
        const rk = rank(self.shape, self.mini);
        try ops.top1(self.kern, rig.s, self.dev, self.shape.rows, rk.stride, rk.n - @intFromBool(self.short), rk.base, self.out[0], self.out[1]);
    }

    pub fn count(self: *const Top1) usize {
        return self.shape.rows * 8;
    }

    pub fn check(self: *const Top1, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const rk = rank(self.shape, self.mini);
        const rows = self.shape.rows;
        for (0..rows) |r| {
            const want = scan(self.x[r * rk.stride ..][0..rk.n]);
            const v = ref.readF32(got, r);
            const i = ref.readI32(got, rows + r);
            if (v != want.v or i != @as(i32, @intCast(want.i + rk.base))) return try std.fmt.bufPrint(msg, "row {d}: ({e}, {d}) where ({e}, {d}) is due", .{ r, v, i, want.v, want.i + rk.base });
        }
        return null;
    }
};

const seed: u64 = 0x1234_5678_9abc_def0;
const inv_t: f32 = 1.0 / 0.7;
const pos0 = 70_000;

const race_rule = "the largest bf16(x) / T + Gumbel(sample.cu's key at pos, id): within logf's error of the host's, none " ++
    "better beyond it; NaN never picked, the lowest +Inf first, a row of NaNs (-inf, base)";

/// One row a launch (the drafter's draw at a position) x a rank's real ids x the rank.
pub fn raceSpec(late: bool, mini: bool) c.Spec {
    const name = if (late) M.named("control: inkling2 ops tf_ink2_race1 keyed one position late", mini) else M.named("inkling2 ops tf_ink2_race1 (keyed Gumbel race, a launch a row)", mini);
    const served: []const c.Shape = if (mini) &.{ .{ .rows = 4, .n = M.mini.real, .k = 1 }, .{ .rows = 1, .n = M.mini.real, .k = 1 }, .{ .rows = 2, .n = M.mini.real, .k = 1 } } else &.{ .{ .rows = 4, .n = ref.share, .k = 1 }, .{ .rows = 1, .n = ref.share, .k = 1 }, .{ .rows = 2, .n = ref.share1_real, .k = 2 } };
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = race_rule }, .served = if (late) served[0..2] else served };
}

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// sample.cu's uniform at (seed, pos, id), bit for bit.
fn uniform(pos: u32, id: u32) f32 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (@as(u64, pos) *% 0xD1B54A32D192ED03));
    x = mix(x ^ @as(u64, id));
    const u = (@as(f32, @floatFromInt(@as(u32, @truncate(x >> 40)))) + 0.5) * (1.0 / 16777216.0);
    return @min(u, @as(f32, @bitCast(@as(u32, 0x3F7FFFFF))));
}

/// Each row's (score, id) against the host's scores in float64 and a margin for logf's two roundings.
pub const Race = struct {
    kern: *const ops.Kernels,
    late: bool = false,
    mini: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    dev: u64 = 0,
    out: [2]u64 = @splat(0),

    pub fn spec(self: *const Race) c.Spec {
        return raceSpec(self.late, self.mini);
    }

    pub fn control(self: *const Race) ?g.Gate {
        return if (self.late) .shapes else null;
    }

    pub fn free(self: *Race, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }

    pub fn prepare(self: *Race, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n < 4 or shape.n > model(self.mini).share()) return false;
        self.free(a);
        self.shape = shape;
        self.x = try logits(a, rng, shape, fill, self.mini);
        for (self.x) |*v| if (std.math.isFinite(v.*)) {
            v.* /= 32; // logits within a few Gumbel widths, so the keys decide
        };
        self.dev = try ref.input(rig, self.x);
        self.out = .{ try rig.output(shape.rows * 4), try rig.output(shape.rows * 4) };
        return true;
    }

    pub fn run(self: *Race, rig: *rig_.Rig) !void {
        const rk = rank(self.shape, self.mini);
        for (0..self.shape.rows) |r| {
            const pos = pos0 + r + @intFromBool(self.late);
            try ops.race1(self.kern, rig.s, self.dev + r * rk.stride * 4, rk.n, rk.base, seed, inv_t, pos, self.out[0] + r * 4, self.out[1] + r * 4);
        }
    }

    pub fn count(self: *const Race) usize {
        return self.shape.rows * 8;
    }

    /// The host's score of id i, and its margin: logf's error on both logs, the fp32 roundings around them.
    fn score(rk: Rank, row: []const f32, pos: u32, i: usize) [2]f64 {
        const a: f32 = c.fromBf16(c.toBf16(row[i])) * inv_t;
        const u: f64 = uniform(pos, @intCast(rk.base + i));
        const gum = -@log(-@log(u));
        return .{ a + gum, 0x1p-20 * (1 + @abs(a) + @abs(gum)) };
    }

    pub fn check(self: *const Race, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const rk = rank(self.shape, self.mini);
        const rows = self.shape.rows;
        for (0..rows) |r| {
            const row = self.x[r * rk.stride ..][0..rk.n];
            const pos: u32 = @intCast(pos0 + r);
            const v = ref.readF32(got, r);
            const id = ref.readI32(got, rows + r) - @as(i32, @intCast(rk.base));
            const top = scan(row);
            if (top.v == -std.math.inf(f32) and std.math.isNan(row[0])) {
                if (id != 0 or v != -std.math.inf(f32)) return try std.fmt.bufPrint(msg, "row {d} of NaNs: ({e}, {d})", .{ r, v, id });
                continue;
            }
            if (top.v == std.math.inf(f32)) {
                if (id != @as(i32, @intCast(top.i)) or v != top.v) return try std.fmt.bufPrint(msg, "row {d}: ({e}, {d}) where the first +Inf {d} is due", .{ r, v, id, top.i });
                continue;
            }
            if (id < 0 or id >= @as(i32, @intCast(rk.n)) or !std.math.isFinite(row[@intCast(id)])) return try std.fmt.bufPrint(msg, "row {d}: id {d} is not a finite candidate", .{ r, id });
            const mine = score(rk, row, pos, @intCast(id));
            if (@abs(v - mine[0]) > mine[1]) return try std.fmt.bufPrint(msg, "row {d}: id {d} scored {e}, the host {e}", .{ r, id, v, mine[0] });
            for (row, 0..) |x, i| if (std.math.isFinite(x)) {
                const s = score(rk, row, pos, i);
                if (s[0] - s[1] > v) return try std.fmt.bufPrint(msg, "row {d}: id {d} ({e}) beats the pick {d} ({e})", .{ r, i, s[0], id, v });
            };
        }
        return null;
    }
};

/// Rows x a rank's share x the rank: logits into the sampler's whole bf16 rows at the rank's columns (Mini: one row).
pub fn tobfSpec(unpadded: bool, mini: bool) c.Spec {
    const V = M.mini.vocab;
    const served: []const c.Shape = if (mini) &.{ .{ .rows = 16, .n = V, .k = 1 }, .{ .rows = 1, .n = V, .k = 1 }, .{ .rows = 128, .n = V, .k = 1 } } else &.{ .{ .rows = 16, .n = ref.share, .k = 2 }, .{ .rows = 1, .n = ref.share, .k = 1 }, .{ .rows = 128, .n = ref.share, .k = 1 }, .{ .rows = 1, .n = ref.share, .k = 2 } };
    const name = if (unpadded) (if (mini) M.named("control: inkling2 ops tf_ink2_tobf rounding the padding ids", true) else "control: inkling2 ops tf_ink2_tobf rounding rank 1's padding ids") else M.named("inkling2 ops tf_ink2_tobf (bf16 rows, padding ids -inf)", mini);
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "a rounding: each value to bf16 at nearest even, NaN to NaN" }, .served = if (unpadded) served[0..1] else served };
}

/// Each written element against the host's rounding, the padding ids -inf and the other rank's half untouched.
pub const Tobf = struct {
    kern: *const ops.Kernels,
    unpadded: bool = false,
    mini: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const Tobf) c.Spec {
        return tobfSpec(self.unpadded, self.mini);
    }

    pub fn control(self: *const Tobf) ?g.Gate {
        return if (self.unpadded) .shapes else null;
    }

    pub fn free(self: *Tobf, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }

    pub fn prepare(self: *Tobf, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != model(self.mini).share()) return false;
        self.free(a);
        self.shape = shape;
        self.x = try ref.f32s(a, rng, shape.rows * shape.n, -40, 40);
        if (fill == .nonfinite) ref.plant(f32, self.x, shape.rows, shape.n, ref.asF32);
        self.dev = try ref.input(rig, self.x);
        self.out = try rig.output(shape.rows * model(self.mini).vocab * 2);
        return true;
    }

    /// The rank's first column, and its real ids (the last rank's stop at unpadded_vocab_size).
    fn cols(self: *const Tobf) [2]usize {
        const m = model(self.mini);
        const col0: usize = if (self.shape.k == 2) m.share() else 0;
        return .{ col0, if (self.shape.k == m.ranks) m.lastReal() else self.shape.n };
    }

    pub fn run(self: *Tobf, rig: *rig_.Rig) !void {
        const at = self.cols();
        const n = if (self.unpadded) self.shape.n else at[1];
        try ops.tobf(self.kern, rig.s, self.dev, self.shape.rows, self.shape.n, n, self.out, model(self.mini).vocab, at[0], self.shape.n);
    }

    pub fn count(self: *const Tobf) usize {
        return self.shape.rows * model(self.mini).vocab * 2;
    }

    pub fn check(self: *const Tobf, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        const V = model(self.mini).vocab;
        const col0 = self.cols()[0];
        const real = self.cols()[1];
        for (0..self.shape.rows) |r| for (0..V) |col| {
            const have = std.mem.readInt(u16, got[2 * (r * V + col) ..][0..2], .little);
            const mine = col >= col0 and col < col0 + n;
            const i = col -% col0;
            const want: u16 = if (!mine) 0xffff else if (i < real) c.toBf16(self.x[r * n + i]) else 0xff80;
            const nan_ok = mine and i < real and std.math.isNan(self.x[r * n + i]) and std.math.isNan(c.fromBf16(have));
            if (have != want and !nan_ok) return try std.fmt.bufPrint(msg, "row {d} column {d}: {x} where {x} is due", .{ r, col, have, want });
        };
        return null;
    }
};

/// Launches x words x 1: a link step's header and up to 80 ids.
pub fn wordsSpec(short: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 2, .n = 17, .k = 1 }, .{ .rows = 1, .n = 16, .k = 1 }, .{ .rows = 4, .n = 96, .k = 1 }, .{ .rows = 1, .n = 80, .k = 1 } };
    const name = if (short) "control: inkling2 ops tf_ink2_put_words one word short" else "inkling2 ops tf_ink2_put_words (a step's words by value)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "a move: every bit copied, NaN payloads too" }, .served = if (short) served[0..2] else served };
}

/// The first n words of each launch's argument in its buffer, the rest untouched.
pub const Words = struct {
    kern: *const ops.Kernels,
    short: bool = false,
    shape: c.Shape = undefined,
    words: []ops.Words = &.{},
    out: u64 = 0,

    pub fn spec(self: *const Words) c.Spec {
        return wordsSpec(self.short);
    }

    pub fn control(self: *const Words) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *Words, a: std.mem.Allocator) void {
        a.free(self.words);
        self.words = &.{};
    }

    pub fn prepare(self: *Words, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 96) return false;
        self.free(a);
        self.shape = shape;
        self.words = try a.alloc(ops.Words, shape.rows);
        for (self.words) |*w| for (&w.w) |*v| {
            v.* = rng.int(u32);
        };
        if (fill == .nonfinite) self.words[0].w[1] = @bitCast(ref.nan);
        self.out = try rig.output(shape.rows * 96 * 4);
        return true;
    }

    pub fn run(self: *Words, rig: *rig_.Rig) !void {
        for (self.words, 0..) |w, j| try ops.putWords(self.kern, rig.s, self.out + j * 96 * 4, self.shape.n - @intFromBool(self.short), w);
    }

    pub fn count(self: *const Words) usize {
        return self.shape.rows * 96 * 4;
    }

    pub fn check(self: *const Words, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        for (self.words, 0..) |w, j| for (w.w, 0..) |v, i| {
            const want: u32 = if (i < self.shape.n) v else 0xffff_ffff;
            const have = std.mem.readInt(u32, got[(j * 96 + i) * 4 ..][0..4], .little);
            if (have != want) return try std.fmt.bufPrint(msg, "launch {d} word {d}: {x} where {x} is due", .{ j, i, have, want });
        };
        return null;
    }
};
