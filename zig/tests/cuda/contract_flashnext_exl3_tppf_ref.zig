//! Host cases of Flash Next EXL3's TP prefill reduction launches: inputs, float64 and bit oracles, stress, faults.
const std = @import("std");
const core = @import("core");
const flash = @import("flashnext");

const c = core.contract.spec;
const g = core.contract.gates;
const canon = flash.exl3_canon;

pub const slots = canon.slots;
pub const dims = 2560;

/// A planted fault of a control kernel (contract_tppf_controls.cuh); none is the shipped kernel.
pub const Fault = enum { none, flat, nan_dropped, writes_input, bf16_adds, wrong };

pub fn controlGate(f: Fault) ?g.Gate {
    return switch (f) {
        .none => null,
        .flat, .wrong => .shapes,
        .nan_dropped => .nonfinite,
        .writes_input => .repeat,
        .bf16_adds => .sums,
    };
}

/// An output no launch wrote: the rig poisons outputs with 0xff bytes.
const unwritten: u32 = 0xffffffff;

/// Bit-equal to the host's result, or both NaN (a payload is not part of the contract).
pub fn sameBits(got: f32, want: f32) bool {
    return @as(u32, @bitCast(got)) == @as(u32, @bitCast(want)) or (std.math.isNan(got) and std.math.isNan(want));
}

pub fn bf16(x: f32) f32 {
    return c.fromBf16(c.toBf16(x));
}

fn bitsOf(out: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, out[i * 4 ..][0..4], .little));
}

pub fn put(out: []u8, i: usize, v: f32) void {
    std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
}

/// Fp32 values that make fp16 sums overflow and bf16 ones round away their low bits: 2^19 to 2^20, either sign.
fn big(rng: std.Random) f32 {
    const v: f32 = @floatFromInt((@as(u32, 1) << 19) + rng.uintLessThan(u32, 1 << 19));
    return if (rng.boolean()) v else -v;
}

const stated = "the bytes of the host canonical tree (NaN payloads aside) and float64's class on every output";

/// Each output against the host's bits, float64's class and bound (gate 1: some finite and some not).
fn verifyF32(msg: []u8, s: c.Spec, shape: c.Shape, gate: g.Gate, out: []const u8, want: []const f64, mag: []const f64, ctx: anytype, comptime expected: anytype) !?[]const u8 {
    var finite: usize = 0;
    for (want, mag, 0..) |w, m, i| {
        const got = bitsOf(out, i);
        const host = expected(ctx, i);
        if (!sameBits(got, host)) return try std.fmt.bufPrint(msg, "output {d}: {x:0>8} where the host canonical reduction gives {x:0>8}", .{ i, @as(u32, @bitCast(got)), @as(u32, @bitCast(host)) });
        const have: f64 = got;
        if (c.classOf(have) != c.classOf(w)) return try std.fmt.bufPrint(msg, "output {d} is {t} where float64 gives {t}", .{ i, c.classOf(have), c.classOf(w) });
        if (std.math.isFinite(w)) {
            finite += 1;
            if (@abs(have - w) > c.bound(s, shape, w, m)) return try std.fmt.bufPrint(msg, "output {d}: {d} is outside float64's bound around {d}", .{ i, have, w });
        }
    }
    if (gate == .nonfinite and (finite == 0 or finite == want.len)) return "the planted values leave all outputs finite or none";
    return null;
}

/// The planted non-finite rows of a 3-D f32 layout: a row each of NaN, +Inf, -Inf, both infinities and Inf x 0.
fn plantRows(rows: usize, n: usize, y: []f32, w: []f32, pieces: usize) void {
    for (0..rows) |r| for (0..n) |col| {
        if (col % 7 != 0) continue;
        const k = (col / 7) % pieces;
        const at = (r * pieces + k) * n + col;
        switch (r % 5) {
            0 => y[at] = std.math.nan(f32),
            1 => y[at] = std.math.inf(f32),
            2 => y[at] = -std.math.inf(f32),
            3 => {
                y[at] = std.math.inf(f32);
                y[(r * pieces + (k + 1) % pieces) * n + col] = -std.math.inf(f32);
            },
            else => {
                y[at] = std.math.inf(f32);
                w[r * pieces + k] = 0;
            },
        }
    };
}

/// A row's value of one block as the kernel's leaf chains and tree give it (absent leaves are +0, never -0).
pub fn blockSum(y: *const [slots]f32, w: *const [slots]f32, p: *const [slots]i32, start: usize, log: u8, fault: Fault) f32 {
    var leaf: [canon.leaves]f32 = @splat(0);
    const size = @as(usize, 1) << @intCast(log);
    var chain: f32 = 0;
    for (0..slots) |k| {
        if (p[k] < 0) continue;
        const l = canon.leafOf(@intCast(p[k]));
        if (l < start or l >= start + size) continue;
        if (fault == .nan_dropped and std.math.isNan(y[k] * w[k])) continue;
        chain = @mulAdd(f32, y[k], w[k], chain);
        leaf[l - start] = @mulAdd(f32, y[k], w[k], leaf[l - start]);
    }
    if (fault == .flat) return chain;
    var n = size;
    while (n > 1) : (n /= 2) for (0..n / 2) |i| {
        const s = leaf[2 * i] + leaf[2 * i + 1];
        leaf[i] = if (fault == .bf16_adds) bf16(s) else s;
    };
    return leaf[0];
}

/// The blocks rank `rank` of `world` computes, in leaf order.
pub fn rankBlocks(buf: *[canon.max_blocks]canon.Block, world: usize, rank: usize) usize {
    const r = canon.run(world, rank);
    return canon.blocks(buf, r.lo, r.hi);
}

/// A leaf's fma chain (at most 11 slots) and the 6 levels of the tree above it.
fn combineDepth(_: c.Shape) usize {
    return slots + canon.leaf_log;
}

/// Each piece meets at most 6 adds on its way up the tree.
fn finishDepth(_: c.Shape) usize {
    return canon.leaf_log;
}

pub const edges: [3]c.Tile = .{ .{}, .{ .size = 256, .step = 256, .min = 256 }, .{} };
const combine_region = [_]c.Shape{
    .{ .rows = 5, .n = dims, .k = 2 },
    .{ .rows = 1, .n = dims, .k = 1 },
    .{ .rows = 130, .n = dims, .k = 2 },
    .{ .rows = 1024, .n = dims, .k = 1 },
    .{ .rows = 2048, .n = dims, .k = 2 },
    .{ .rows = 17, .n = dims, .k = 3 },
    .{ .rows = 40, .n = dims, .k = 5 },
    .{ .rows = 6, .n = dims, .k = 8 },
};
const combine_blocks = [_]c.Shape{
    .{ .rows = 300, .n = dims, .k = 2 },
    .{ .rows = 512, .n = dims, .k = 2 },
    .{ .rows = 2048, .n = dims, .k = 2 },
    .{ .rows = 700, .n = dims, .k = 5 },
    .{ .rows = 257, .n = dims, .k = 8 },
};
pub const max_rows = 2048;
/// The all-gather's sub-block rows (the engine's chunk is a multiple of 256, one region of `sub` rows a block).
pub const sub_rows = 256;

/// Rank partials of one launch per rank: every rank of k launches into one gather buffer, as the all-gather lays it.
pub const Combine = struct {
    gpa: std.mem.Allocator,
    blocks: bool,
    fault: Fault = .none,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    m: usize = 1,
    sub: usize = 0,
    y: []f32 = &.{},
    w: []f32 = &.{},
    p: []i32 = &.{},
    msg: [200]u8 = undefined,

    pub const n_in = 3;
    pub const n_out = 1;

    pub fn deinit(self: *Combine) void {
        if (self.arena) |*a| a.deinit();
    }

    pub fn spec(self: *const Combine) c.Spec {
        const name = switch (self.fault) {
            .none => if (self.blocks) "cuda Flash Next EXL3 prefill tf_exl3pf_tp_combine (256-row sub-blocks)" else "cuda Flash Next EXL3 prefill tf_exl3pf_tp_combine (one region)",
            .flat => "control: tf_exl3pf_tp_combine as one chain, no tree",
            .nan_dropped => "control: tf_exl3pf_tp_combine with NaN products dropped",
            .writes_input => "control: tf_exl3pf_tp_combine writing its input",
            .bf16_adds => "control: tf_exl3pf_tp_combine with bf16 tree adds",
            else => unreachable,
        };
        return .{ .name = name, .accum = .f32, .depth = combineDepth, .out = .f32, .nan = .{ .stated = stated }, .tiles = edges, .served = if (self.blocks) &combine_blocks else &combine_region };
    }

    pub fn accepts(shape: c.Shape) bool {
        return shape.rows >= 1 and shape.rows <= max_rows and shape.n >= 256 and shape.n % 256 == 0 and shape.n <= dims and shape.k >= 1 and shape.k <= 8;
    }

    pub fn prepare(self: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!accepts(shape)) return false;
        self.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        self.m = canon.maxPartials(shape.k);
        self.sub = if (self.blocks and shape.rows > sub_rows) sub_rows else shape.rows;
        self.y = try a.alloc(f32, shape.rows * slots * shape.n);
        self.w = try a.alloc(f32, shape.rows * slots);
        self.p = try a.alloc(i32, shape.rows * slots);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x636f6d62);
        const rng = prng.random();
        if (fill == .stress) self.stress(rng) else self.random(rng, fill);
        return true;
    }

    pub fn inputs(self: *const Combine) [n_in][]const u8 {
        return .{ std.mem.sliceAsBytes(self.y), std.mem.sliceAsBytes(self.w), std.mem.sliceAsBytes(self.p) };
    }

    pub fn outputLens(self: *const Combine) [n_out]usize {
        return .{self.count() * 4};
    }

    pub fn count(self: *const Combine) usize {
        return self.shape.k * self.m * self.shape.rows * self.shape.n;
    }

    /// Router-like picks (clustered leaves, the shared expert last), some masked (-1) or past the table, small weights.
    fn random(self: *Combine, rng: std.Random, fill: g.Fill) void {
        for (0..self.shape.rows) |r| {
            const base = [2]u16{ rng.uintLessThan(u16, 64) * 8, rng.uintLessThan(u16, 64) * 8 };
            for (0..slots) |k| {
                var id: i32 = if (k == slots - 1) 512 else if (rng.boolean()) rng.uintLessThan(u16, 512) else base[rng.uintLessThan(usize, 2)] + rng.uintLessThan(u16, 8);
                if (rng.uintLessThan(u8, 5) == 0) id = -1 else if (k == slots - 1 and rng.uintLessThan(u8, 9) == 0) id = 513 + @as(i32, rng.uintLessThan(u8, 9));
                self.p[r * slots + k] = id;
            }
        }
        for (self.y) |*v| v.* = if (rng.uintLessThan(u8, 23) == 0) -0.0 else rng.floatNorm(f32) * 0.05;
        for (self.w) |*v| v.* = if (rng.uintLessThan(u8, 29) == 0) -0.0 else rng.floatNorm(f32) * 0.2;
        if (fill == .nonfinite) {
            plantRows(self.shape.rows, self.shape.n, self.y, self.w, slots);
        } else for (0..slots) |k| {
            self.y[k * self.shape.n] = @bitCast(@as(u32, 0x80000001));
            self.w[k] = 0x1p-4;
            self.p[k] = if (k == slots - 1) 512 else @intCast(256 + k);
        }
    }

    /// Integers: ten routed picks in rank 0's first block, mirrored (same y, negated weight), the shared slot 3 x 1.
    fn stress(self: *Combine, rng: std.Random) void {
        var bl: [canon.max_blocks]canon.Block = undefined;
        _ = rankBlocks(&bl, self.shape.k, 0);
        const first: usize = bl[0].start;
        const lo = first * canon.leaf_ids;
        const span = @min(@as(usize, 1) << @intCast(bl[0].log), canon.leaves - first) * canon.leaf_ids;
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            for (0..slots - 1) |k| self.p[r * slots + k] = @intCast(lo + rng.uintLessThan(usize, span));
            self.p[r * slots + slots - 1] = 512;
            for (0..(slots - 1) / 2) |k| {
                const wv: f32 = @floatFromInt(512 + rng.uintLessThan(u32, 512));
                self.w[r * slots + k] = wv;
                self.w[r * slots + slots - 2 - k] = -wv;
                for (0..n) |col| {
                    const yv: f32 = @floatFromInt(512 + rng.uintLessThan(u32, 512));
                    self.y[(r * slots + k) * n + col] = yv;
                    self.y[(r * slots + slots - 2 - k) * n + col] = yv;
                }
            }
            self.w[r * slots + slots - 1] = 1;
            for (0..n) |col| self.y[(r * slots + slots - 1) * n + col] = 3;
        }
    }

    /// Where output `i` of the gather layout comes from: its row, column and block (null: no rank writes it).
    const At = struct { row: usize, col: usize, block: ?canon.Block };

    fn locate(self: *const Combine, i: usize) At {
        const n = self.shape.n;
        const region = self.sub * self.shape.k * self.m * n;
        const b = i / region;
        const mb = @min(self.sub, self.shape.rows - b * self.sub);
        const rem = i - b * region;
        const slot = rem / (mb * n);
        const row = b * self.sub + (rem / n) % mb;
        var bl: [canon.max_blocks]canon.Block = undefined;
        const nb = rankBlocks(&bl, self.shape.k, slot / self.m);
        return .{ .row = row, .col = rem % n, .block = if (slot % self.m < nb) bl[slot % self.m] else null };
    }

    fn slotValues(self: *const Combine, at: At, y: *[slots]f32, w: *[slots]f32, p: *[slots]i32) void {
        for (0..slots) |k| {
            y[k] = self.y[(at.row * slots + k) * self.shape.n + at.col];
            w[k] = self.w[at.row * slots + k];
            p[k] = self.p[at.row * slots + k];
        }
    }

    fn expectedAs(self: *const Combine, i: usize, fault: Fault) f32 {
        const at = self.locate(i);
        const blk = at.block orelse return @bitCast(unwritten);
        var y: [slots]f32 = undefined;
        var w: [slots]f32 = undefined;
        var p: [slots]i32 = undefined;
        self.slotValues(at, &y, &w, &p);
        return blockSum(&y, &w, &p, blk.start, blk.log, fault);
    }

    fn expected(self: *const Combine, i: usize) f32 {
        return self.expectedAs(i, .none);
    }

    /// Float64's sum of the products of the block's slots in slot order, and the sum of their magnitudes.
    fn terms(self: *const Combine, i: usize, buf: *[slots]f64) ?usize {
        const at = self.locate(i);
        const blk = at.block orelse return null;
        var y: [slots]f32 = undefined;
        var w: [slots]f32 = undefined;
        var p: [slots]i32 = undefined;
        self.slotValues(at, &y, &w, &p);
        var m: usize = 0;
        for (0..slots) |k| {
            if (p[k] < 0) continue;
            const l = canon.leafOf(@intCast(p[k]));
            if (l < blk.start or l >= blk.start + (@as(usize, 1) << @intCast(blk.log))) continue;
            buf[m] = @as(f64, y[k]) * @as(f64, w[k]);
            m += 1;
        }
        return m;
    }

    pub fn reference(self: *const Combine, want: []f64, mag: []f64) !void {
        var t: [slots]f64 = undefined;
        for (want, mag, 0..) |*w, *m, i| {
            const cnt = self.terms(i, &t) orelse {
                w.* = std.math.nan(f64);
                m.* = 0;
                continue;
            };
            w.* = 0;
            m.* = 0;
            for (t[0..cnt]) |x| {
                w.* += x;
                m.* += @abs(x);
            }
        }
    }

    pub fn simulate(self: *const Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [slots]f64 = undefined;
        for (out, 0..) |*o, i| o.* = if (self.terms(i, &t)) |cnt| core.contract.rows.sumIn(acc, order, t[0..cnt]) else std.math.nan(f64);
    }

    pub fn verify(self: *Combine, out: []const u8, gate: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        return verifyF32(&self.msg, self.spec(), self.shape, gate, out, want, mag, self, Combine.expected);
    }

    /// The host's emulation of a launch (a planted fault's too), for the gates' own host tests.
    pub fn emulate(self: *Combine, fault: Fault, out: []u8) void {
        for (0..self.count()) |i| put(out, i, self.expectedAs(i, fault));
        if (fault == .writes_input) self.y[0] += 1;
    }
};

const finish_served = [_]c.Shape{
    .{ .rows = 5, .n = dims, .k = 2 },
    .{ .rows = 1, .n = dims, .k = 2 },
    .{ .rows = 44, .n = dims, .k = 2 },
    .{ .rows = 256, .n = dims, .k = 2 },
    .{ .rows = 4096, .n = dims, .k = 2 },
    .{ .rows = 7, .n = dims, .k = 3 },
    .{ .rows = 9, .n = dims, .k = 5 },
    .{ .rows = 6, .n = dims, .k = 8 },
};
pub const max_finish_rows = 4096;

/// The tree finished from the gathered partials of one region: world x M partial blocks a row, merged in leaf order.
pub const Finish = struct {
    gpa: std.mem.Allocator,
    fault: Fault = .none,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    m: usize = 1,
    ps: [canon.max_pieces]canon.Piece = undefined,
    np: usize = 0,
    vals: []f32 = &.{},
    msg: [200]u8 = undefined,

    pub const n_in = 1;
    pub const n_out = 1;

    pub fn deinit(self: *Finish) void {
        if (self.arena) |*a| a.deinit();
    }

    pub fn spec(self: *const Finish) c.Spec {
        const name = switch (self.fault) {
            .none => "cuda Flash Next EXL3 prefill tf_exl3pf_tp_finish",
            .flat => "control: tf_exl3pf_tp_finish as one chain, no tree",
            .bf16_adds => "control: tf_exl3pf_tp_finish with bf16 merges",
            .writes_input => "control: tf_exl3pf_tp_finish writing its input",
            else => unreachable,
        };
        return .{ .name = name, .accum = .f32, .depth = finishDepth, .out = .f32, .nan = .{ .stated = stated }, .tiles = edges, .served = &finish_served };
    }

    /// The gathered slot of a piece: rank * M + its index on the rank.
    fn slotOf(self: *const Finish, p: canon.Piece) usize {
        return @as(usize, p.rank) * self.m + p.idx;
    }

    pub fn accepts(shape: c.Shape) bool {
        return shape.rows >= 1 and shape.rows <= max_finish_rows and shape.n >= 256 and shape.n % 256 == 0 and shape.n <= dims and shape.k >= 1 and shape.k <= 8;
    }

    pub fn prepare(self: *Finish, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!accepts(shape)) return false;
        self.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        self.m = canon.maxPartials(shape.k);
        self.np = canon.pieces(&self.ps, shape.k);
        self.vals = try a.alloc(f32, shape.k * self.m * shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x66696e73);
        const rng = prng.random();
        if (fill == .stress) self.stress(rng) else self.random(rng, fill);
        self.poison();
        return true;
    }

    pub fn inputs(self: *const Finish) [n_in][]const u8 {
        return .{std.mem.sliceAsBytes(self.vals)};
    }

    pub fn outputLens(self: *const Finish) [n_out]usize {
        return .{self.count() * 4};
    }

    pub fn count(self: *const Finish) usize {
        return self.shape.rows * self.shape.n;
    }

    fn at(self: *const Finish, piece: usize, i: usize) usize {
        return self.slotOf(self.ps[piece]) * self.count() + i;
    }

    /// Partials of different magnitudes (an fp32 sum of them depends on the order of the merges), a few negative zeros.
    fn random(self: *Finish, rng: std.Random, fill: g.Fill) void {
        for (self.vals) |*v| {
            const scale: f32 = @floatFromInt(@as(u32, 1) << rng.uintLessThan(u5, 10));
            v.* = if (rng.uintLessThan(u8, 23) == 0) -0.0 else rng.floatNorm(f32) * scale;
        }
        if (fill != .nonfinite) return;
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |col| {
            if (col % 7 != 0) continue;
            const piece = (col / 7) % self.np;
            const i = r * n + col;
            switch (r % 5) {
                0 => self.vals[self.at(piece, i)] = std.math.nan(f32),
                1 => self.vals[self.at(piece, i)] = std.math.inf(f32),
                2 => self.vals[self.at(piece, i)] = -std.math.inf(f32),
                3 => {
                    self.vals[self.at(piece, i)] = std.math.inf(f32);
                    self.vals[self.at((piece + 1) % self.np, i)] = -std.math.inf(f32);
                },
                else => self.vals[self.at(self.np - 1, i)] = std.math.nan(f32),
            }
        };
    }

    /// Integer partials of 2^19 to 2^20 in shuffled negated pairs (odd one out 3): exact in fp32, ruinous in fp16/bf16.
    fn stress(self: *Finish, rng: std.Random) void {
        for (0..self.count()) |i| {
            var v: [canon.max_pieces]f32 = undefined;
            for (0..self.np / 2) |j| {
                v[2 * j] = big(rng);
                v[2 * j + 1] = -v[2 * j];
            }
            if (self.np % 2 == 1) v[self.np - 1] = 3;
            rng.shuffle(f32, v[0..self.np]);
            for (v[0..self.np], 0..) |x, piece| self.vals[self.at(piece, i)] = x;
        }
    }

    /// Gather slots no rank fills (a rank with fewer blocks than M) hold a value no merge may read.
    fn poison(self: *Finish) void {
        var bl: [canon.max_blocks]canon.Block = undefined;
        for (0..self.shape.k) |q| {
            const nb = rankBlocks(&bl, self.shape.k, q);
            for (nb..self.m) |j| @memset(self.vals[(q * self.m + j) * self.count() ..][0..self.count()], 1.0e30);
        }
    }

    fn pieceValues(self: *const Finish, i: usize, buf: *[canon.max_pieces]f32) []const f32 {
        for (0..self.np) |piece| buf[piece] = self.vals[self.at(piece, i)];
        return buf[0..self.np];
    }

    fn expectedAs(self: *const Finish, i: usize, fault: Fault) f32 {
        var buf: [canon.max_pieces]f32 = undefined;
        const v = self.pieceValues(i, &buf);
        if (fault == .flat) {
            var chain: f32 = 0;
            for (v) |x| chain += x;
            return chain;
        }
        if (fault != .bf16_adds) return canon.finish(self.ps[0..self.np], v);
        var top: usize = 0;
        var s: [canon.max_pieces]f32 = undefined;
        var lg: [canon.max_pieces]u8 = undefined;
        var st: [canon.max_pieces]u8 = undefined;
        for (self.ps[0..self.np], v) |p, x| {
            s[top] = x;
            lg[top] = p.log;
            st[top] = p.start;
            top += 1;
            while (top >= 2 and lg[top - 1] == lg[top - 2] and st[top - 2] % (@as(usize, 2) << @intCast(lg[top - 1])) == 0) {
                s[top - 2] = bf16(s[top - 2] + s[top - 1]);
                lg[top - 2] += 1;
                top -= 1;
            }
        }
        return s[0];
    }

    fn expected(self: *const Finish, i: usize) f32 {
        return self.expectedAs(i, .none);
    }

    pub fn reference(self: *const Finish, want: []f64, mag: []f64) !void {
        var buf: [canon.max_pieces]f32 = undefined;
        for (want, mag, 0..) |*w, *m, i| {
            w.* = 0;
            m.* = 0;
            for (self.pieceValues(i, &buf)) |v| {
                w.* += v;
                m.* += @abs(@as(f64, v));
            }
        }
    }

    pub fn simulate(self: *const Finish, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [canon.max_pieces]f32 = undefined;
        var t: [canon.max_pieces]f64 = undefined;
        for (out, 0..) |*o, i| {
            const v = self.pieceValues(i, &buf);
            for (v, t[0..v.len]) |x, *y| y.* = x;
            o.* = core.contract.rows.sumIn(acc, order, t[0..v.len]);
        }
    }

    pub fn verify(self: *Finish, out: []const u8, gate: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        return verifyF32(&self.msg, self.spec(), self.shape, gate, out, want, mag, self, Finish.expected);
    }

    pub fn emulate(self: *Finish, fault: Fault, out: []u8) void {
        for (0..self.count()) |i| put(out, i, self.expectedAs(i, fault));
        if (fault == .writes_input) self.vals[0] += 1;
    }
};
