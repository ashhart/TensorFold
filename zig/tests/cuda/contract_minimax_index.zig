//! MiniMax-M3.1-Flash's CUDA indexer under the contract: block scores (chunks and windows) and selections.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mm3 = @import("minimax_m3_cuda");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;
const D = 128;
const topk = 16;

pub const Which = enum { tc2, tc4 };

fn heads(w: Which) usize {
    return if (w == .tc4) 4 else 2;
}

/// A key's 128 products; the block's max adds no rounding.
fn scoreDepth(_: c.Shape) usize {
    return D;
}

const max_nan = "a block's max is fmaxf from -inf: a NaN dot is left out, so a block with no other key scores -inf, never kept";

/// rows x (heads x blocks) x 128; served: the row kernel at 16 rows or fewer, the tensor units past them.
pub fn scoresSpec(comptime w: Which) c.Spec {
    return switch (w) {
        // prompt chunks and decode windows (1 and 16 rows) alike
        .tc2 => .{ .name = "cuda minimax tf_mm3_index_scores_tc, 2 index heads", .accum = .f32, .depth = scoreDepth, .out = .f32, .served = &.{ .{ .rows = 64, .n = 2 * 24, .k = D }, .{ .rows = 37, .n = 2 * 19, .k = D }, .{ .rows = 1, .n = 2 * 40, .k = D }, .{ .rows = 16, .n = 2 * 23, .k = D } }, .nan = .{ .stated = max_nan } },
        .tc4 => .{ .name = "cuda minimax tf_mm3_index_scores_tc, 4 index heads", .accum = .f32, .depth = scoreDepth, .out = .f32, .served = &.{.{ .rows = 48, .n = 4 * 18, .k = D }}, .nan = .{ .stated = max_nan } },
    };
}

/// Each (row, head)'s per-block score: max over the block's keys <= pos + row of iq . ik, in fp32.
pub const Scores = struct {
    rig: rig_.Rig,
    k: *const mm3.kernels.Kernels,
    which: Which,
    /// control: the keys read one row late, so every block's keys shift by one
    shifted: bool = false,
    shape: c.Shape = undefined,
    iq: []u16 = &.{},
    ik: []u16 = &.{},
    pos: usize = 0,
    nb: usize = 0,
    exact: bool = false, // stress: integer inputs, so fp32 sums are exact in any order
    in: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn deinit(self: *Scores) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Scores) void {
        const a = self.rig.gpu.gpa;
        a.free(self.iq);
        a.free(self.ik);
        self.iq = &.{};
        self.ik = &.{};
    }

    pub fn kernel(self: *Scores) g.Kernel {
        var declared = switch (self.which) {
            inline else => |w| scoresSpec(w),
        };
        if (self.shifted) declared.name = "control: MiniMax index scores read the keys one row late";
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.shifted) .shapes else null };
    }

    const vt = g.hooks(Scores);

    /// q, k in [-2, 2); stress: integers 1-8 (sums past 2,048); non-finite: NaN q, inf k; the last block 5 keys in.
    pub fn prepare(self: *Scores, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const h = heads(self.which);
        if (shape.k != D or shape.n % h != 0 or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.exact = fill == .stress;
        self.nb = shape.n / h;
        if (self.nb * 128 < shape.rows + 128) return false;
        self.pos = (self.nb - 1) * 128 + 5 - (shape.rows - 1);
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n_keys = self.pos + shape.rows + 1; // one past the last row's: the control's late read stays inside
        self.iq = try a.alloc(u16, shape.rows * h * D);
        self.ik = try a.alloc(u16, n_keys * D);
        for ([_][]u16{ self.iq, self.ik }) |x| for (x) |*v| {
            v.* = c.toBf16(if (fill == .stress) @floatFromInt(1 + rng.uintLessThan(u32, 8)) else rng.float(f32) * 4 - 2);
        };
        if (fill == .nonfinite) {
            for (self.iq[0..D]) |*v| v.* = c.toBf16(std.math.nan(f32)); // row 0, head 0: every dot NaN
            self.ik[(130) * D + 7] = c.toBf16(std.math.inf(f32));
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.iq));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.ik));
        self.out = try self.rig.output(shape.rows * h * self.nb * 4);
        return true;
    }

    pub fn run(self: *Scores) !void {
        const ik = self.in[1] + if (self.shifted) @as(u64, D * 2) else 0;
        const rows = self.shape.rows;
        const h = heads(self.which);
        var a: cuda.Args = .{};
        inline for (.{ self.in[0], ik, self.out }) |v| a.add(v);
        for ([_]usize{ rows, self.pos, h, self.nb }) |v| a.add(@as(c_int, @intCast(v)));
        const y: u32 = @intCast((rows * h + 63) / 64);
        try cuda.launch.launch(self.k.index_scores_tc, .{ .grid = .{ .x = @intCast(self.nb), .y = y }, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Scores) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Row r, head h, block b: the max visible non-NaN dot (-inf: none), sums in `acc`; `mag`: the largest sum |q k|.
    fn score(self: *const Scores, r: usize, h: usize, b: usize, acc: ?c.Accum, order: g.Order, mag: *f64) f64 {
        const hh = heads(self.which);
        const q = self.iq[(r * hh + h) * D ..][0..D];
        var best = -std.math.inf(f64);
        mag.* = 0;
        var terms: [D]f64 = undefined;
        const end = @min(b * 128 + 128, self.pos + r + 1);
        if (end <= b * 128) return best; // a block past the row's own: no key visible
        for (b * 128..end) |key| {
            var abs: f64 = 0;
            for (&terms, q, self.ik[key * D ..][0..D]) |*x, qv, kv| {
                x.* = @as(f64, c.fromBf16(qv)) * c.fromBf16(kv);
                abs += @abs(x.*);
            }
            const dot = if (acc) |k| sumIn(k, order, &terms) else sumIn(.none, .sequential, &terms);
            if (std.math.isFinite(abs)) mag.* = @max(mag.*, abs);
            if (!std.math.isNan(dot)) best = @max(best, dot);
        }
        return best;
    }

    pub fn reference(self: *Scores, want: []f64, mag: []f64) !void {
        const hh = heads(self.which);
        for (0..self.shape.rows) |r| for (0..hh) |h| for (0..self.nb) |b| {
            const i = (r * hh + h) * self.nb + b;
            want[i] = self.score(r, h, b, null, .sequential, &mag[i]);
        };
    }

    /// gamma(4 (D + 16)) of the terms (tensor units add in their own order); stress integers keep the exact bound.
    pub fn bounds(self: *Scores, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (self.exact) return;
        for (allow, want, mag) |*l, w, m| l.* = if (std.math.isFinite(w)) c.gamma(4 * (D + 16), 0x1p-24) * m + @abs(w) * 0x1p-24 + 0x1p-140 else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Scores, acc: c.Accum, order: g.Order, out: []f64) !void {
        const hh = heads(self.which);
        var m: f64 = 0;
        for (0..self.shape.rows) |r| for (0..hh) |h| for (0..self.nb) |b| {
            const v = self.score(r, h, b, acc, order, &m);
            out[(r * hh + h) * self.nb + b] = if (std.math.isFinite(v)) @as(f32, @floatCast(v)) else v;
        };
    }
};

/// rows x topk x the blocks the last row sees: the selection, then its bitmap.
pub fn selectSpec() c.Spec {
    return .{ .name = "cuda minimax tf_mm3_index_select with bitmaps", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "NaN and infinite scores are never kept; the row's own block always is" }, .served = &.{ .{ .rows = 64, .n = topk, .k = 40 }, .{ .rows = 4, .n = topk, .k = 200 }, .{ .rows = 8, .n = topk, .k = 12 } } };
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Each (row, head)'s kept blocks ascending, -1 past the last, and their bitmap.
pub const Select = struct {
    rig: rig_.Rig,
    k: *const mm3.kernels.Kernels,
    shape: c.Shape = undefined,
    s: []f32 = &.{},
    pos: usize = 0,
    words: usize = 0,
    in: u64 = 0,
    sel: u64 = 0,
    bits: u64 = 0,
    msg: [200]u8 = undefined,

    const hh = 2;

    pub fn deinit(self: *Select) void {
        self.rig.gpu.gpa.free(self.s);
        self.s = &.{};
        self.rig.deinit();
    }

    pub fn kernel(self: *Select) g.Kernel {
        return .{ .spec = selectSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Select);

    /// Scores on a coarse grid (ties everywhere: the low id must win them); non-finite: NaN and both infinities.
    pub fn prepare(self: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != topk or shape.rows == 0 or shape.k * 128 < shape.rows) return false;
        const a = self.rig.gpu.gpa;
        a.free(self.s);
        self.s = &.{};
        self.rig.reset();
        self.shape = shape;
        self.pos = shape.k * 128 - shape.rows;
        self.words = shape.k / 32 + 1;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.s = try a.alloc(f32, shape.rows * hh * shape.k);
        for (self.s) |*v| v.* = @floatFromInt(rng.uintLessThan(u32, 24));
        if (fill == .nonfinite) for (self.s, 0..) |*v, i| switch (i % 7) {
            1 => v.* = std.math.nan(f32),
            3 => v.* = std.math.inf(f32),
            5 => v.* = -std.math.inf(f32),
            else => {},
        };
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.s));
        self.sel = try self.rig.output(shape.rows * hh * topk * 4);
        self.bits = try self.rig.output(shape.rows * hh * self.words * 4);
        return true;
    }

    pub fn run(self: *Select) !void {
        try mm3.kernels.indexSelect(self.k, self.rig.s, self.in, self.sel, self.bits, self.shape.rows, self.pos, hh, self.shape.k, topk, self.words);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Select) usize {
        return self.shape.rows * hh * (topk + self.words) * 4;
    }

    pub fn reference(_: *Select, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Row r, head h's blocks: all while 16 or fewer, else its own and the 15 best finite (low id on ties), ascending.
    fn expected(self: *const Select, r: usize, h: usize, out: *[topk]i32) void {
        @memset(out, -1);
        const nb = (self.pos + r) / 128 + 1;
        if (nb <= topk) {
            for (0..nb) |b| out[b] = @intCast(b);
            return;
        }
        const s = self.s[(r * hh + h) * self.shape.k ..];
        out[0] = @intCast(nb - 1);
        var n: usize = 1;
        while (n < topk) : (n += 1) {
            var best: ?usize = null;
            for (0..nb - 1) |b| {
                if (!std.math.isFinite(s[b]) or std.mem.indexOfScalar(i32, out[0..n], @intCast(b)) != null) continue;
                if (best == null or s[b] > s[best.?]) best = b;
            }
            out[n] = @intCast(best orelse break);
        }
        std.mem.sort(i32, out[0..n], {}, std.sort.asc(i32));
    }

    pub fn check(self: *Select, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const rows = self.shape.rows;
        const got = try a.dupe(u8, try self.rig.bytes(self.sel, rows * hh * topk * 4));
        defer a.free(got);
        const bits = try self.rig.bytes(self.bits, rows * hh * self.words * 4);
        for (0..rows) |r| for (0..hh) |h| {
            var want: [topk]i32 = undefined;
            self.expected(r, h, &want);
            const p = r * hh + h;
            for (want, 0..) |w, i| {
                const v = std.mem.readInt(i32, got[(p * topk + i) * 4 ..][0..4], .little);
                if (v != w) return try std.fmt.bufPrint(&self.msg, "row {d} head {d} slot {d}: block {d} where {d} is due", .{ r, h, i, v, w });
            }
            for (0..self.words) |wd| {
                var m: u32 = 0;
                for (want) |b| if (b >= 0 and @as(usize, @intCast(b)) / 32 == wd) {
                    m |= @as(u32, 1) << @intCast(@as(usize, @intCast(b)) % 32);
                };
                const v = std.mem.readInt(u32, bits[(p * self.words + wd) * 4 ..][0..4], .little);
                if (v != m) return try std.fmt.bufPrint(&self.msg, "row {d} head {d} word {d}: {x} where {x} is due", .{ r, h, wd, v, m });
            }
        };
        return null;
    }
};
