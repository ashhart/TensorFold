//! Flash Next EXL3's canonical expert reduction (leaf tree and finish) against the host exl3_canon.zig, bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const canon = flash.exl3_canon;

const slots = canon.slots;

/// The subtree a launch computes: the leaves a rank of two owns, or all 64 (one rank, the whole tree).
pub const Split = enum { low, high, full };

/// A planted fault of a control kernel (contract_leaf_controls.cuh); none is the shipped kernel.
pub const Fault = enum { none, flat, nan_dropped, writes_input, bf16_adds };

fn subtreeOf(s: Split) struct { start: usize, log: u8 } {
    return switch (s) {
        .low => .{ .start = 0, .log = 5 },
        .high => .{ .start = 32, .log = 5 },
        .full => .{ .start = 0, .log = canon.leaf_log },
    };
}

fn controlGate(f: Fault) ?g.Gate {
    return switch (f) {
        .none => null,
        .flat => .shapes,
        .nan_dropped => .nonfinite,
        .writes_input => .repeat,
        .bf16_adds => .sums,
    };
}

/// A leaf's fma chain (at most 11 slots) and the 6 levels of the tree above it.
fn treeDepth(_: c.Shape) usize {
    return slots + canon.leaf_log;
}

/// Each piece meets at most 6 adds on its way up the tree.
fn finishDepth(_: c.Shape) usize {
    return canon.leaf_log;
}

fn treeName(s: Split, f: Fault) []const u8 {
    return switch (f) {
        .none => switch (s) {
            .low => "cuda Flash Next EXL3 fn_exl3_leaf_tree (leaves 0-31)",
            .high => "cuda Flash Next EXL3 fn_exl3_leaf_tree (leaves 32-63)",
            .full => "cuda Flash Next EXL3 fn_exl3_leaf_tree (all 64 leaves)",
        },
        .flat => "cuda Flash Next EXL3 fn_exl3_leaf_tree control: one chain, no tree",
        .nan_dropped => "cuda Flash Next EXL3 fn_exl3_leaf_tree control: NaN products dropped",
        .writes_input => "cuda Flash Next EXL3 fn_exl3_leaf_tree control: writes its input",
        .bf16_adds => "cuda Flash Next EXL3 fn_exl3_leaf_tree control: bf16 tree adds",
    };
}

fn finishName(f: Fault) []const u8 {
    return switch (f) {
        .none => "cuda Flash Next EXL3 fn_exl3_leaf_finish",
        .flat => "cuda Flash Next EXL3 fn_exl3_leaf_finish control: one chain, no tree",
        else => "cuda Flash Next EXL3 fn_exl3_leaf_finish control: bf16 merges",
    };
}

const stated_rule = "the bytes of the host canonical tree (NaN payloads aside) and float64's class on every output";

/// Rows 1, 2, 5 and 16 at the hidden width 2560 (the engine's launches, dims / 256 column blocks); k is unused.
const tree_served = [_]c.Shape{
    .{ .rows = 5, .n = 2560, .k = 0 },
    .{ .rows = 1, .n = 2560, .k = 0 },
    .{ .rows = 2, .n = 2560, .k = 0 },
    .{ .rows = 16, .n = 2560, .k = 0 },
};

fn treeSpec(s: Split, f: Fault) c.Spec {
    return .{
        .name = treeName(s, f),
        .accum = .f32,
        .depth = treeDepth,
        .out = .f32,
        .nan = .{ .stated = stated_rule },
        // a block takes 256 columns: one block, two, and the whole width
        .tiles = .{ .{}, .{ .size = 256, .step = 256, .min = 256 }, .{} },
        .served = &tree_served,
    };
}

/// k is the TP degree whose pieces the finish merges (2 and 8 give one piece a rank, 3 and 5 give uneven runs).
const finish_served = [_]c.Shape{
    .{ .rows = 5, .n = 2560, .k = 8 },
    .{ .rows = 1, .n = 2560, .k = 2 },
    .{ .rows = 2, .n = 2560, .k = 3 },
    .{ .rows = 16, .n = 2560, .k = 5 },
};

fn finishSpec(f: Fault) c.Spec {
    return .{
        .name = finishName(f),
        .accum = .f32,
        .depth = finishDepth,
        .out = .f32,
        .nan = .{ .stated = stated_rule },
        .tiles = .{ .{}, .{ .size = 256, .step = 256, .min = 256 }, .{} },
        .served = &finish_served,
    };
}

fn refused(shape: c.Shape) bool {
    return shape.rows == 0 or shape.rows > 64 or shape.n == 0 or shape.n % 256 != 0 or shape.n > 2560;
}

/// Bit-equal to the host's result, or both NaN (a payload is not part of the contract).
fn sameBits(got: f32, want: f32) bool {
    return @as(u32, @bitCast(got)) == @as(u32, @bitCast(want)) or (std.math.isNan(got) and std.math.isNan(want));
}

fn bitsOf(out: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, out[i * 4 ..][0..4], .little));
}

/// Fp32 values that make fp16 sums overflow and bf16 ones round away their low bits: 2^19 to 2^20, either sign.
fn big(rng: std.Random) f32 {
    const v: f32 = @floatFromInt((@as(u32, 1) << 19) + rng.uintLessThan(u32, 1 << 19));
    return if (rng.boolean()) v else -v;
}

pub const Tree = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    split: Split,
    fault: Fault,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    y: []f32 = &.{},
    w: []f32 = &.{},
    p: []i32 = &.{},
    out: u64 = 0,
    args: [3]u64 = .{ 0, 0, 0 },
    msg: [220]u8 = undefined,

    pub fn deinit(self: *Tree) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Tree);

    pub fn kernel(self: *Tree) g.Kernel {
        return .{ .spec = treeSpec(self.split, self.fault), .ctx = self, .vt = &vt, .control = controlGate(self.fault) };
    }

    fn inRange(self: *const Tree, id: i32) bool {
        const s = subtreeOf(self.split);
        const l = canon.leafOf(@intCast(id));
        return l >= s.start and l < s.start + (@as(usize, 1) << @intCast(s.log));
    }

    pub fn prepare(self: *Tree, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (refused(shape)) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        self.y = try a.alloc(f32, shape.rows * slots * shape.n);
        self.w = try a.alloc(f32, shape.rows * slots);
        self.p = try a.alloc(i32, shape.rows * slots);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6c656166);
        const rng = prng.random();
        if (fill == .stress) self.stress(rng) else self.random(rng, fill);
        const dy = try self.rig.input(std.mem.sliceAsBytes(self.y));
        const dw = try self.rig.input(std.mem.sliceAsBytes(self.w));
        const dp = try self.rig.input(std.mem.sliceAsBytes(self.p));
        self.out = try self.rig.output(self.count() * 4);
        self.args = .{ dy, dw, dp };
        return true;
    }

    /// Router-like picks (clustered leaves, the shared expert 512 last), small weights, a few negative zeros.
    fn random(self: *Tree, rng: std.Random, fill: g.Fill) void {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            const base = [2]u16{ rng.uintLessThan(u16, 64) * 8, rng.uintLessThan(u16, 64) * 8 };
            for (0..slots) |k| {
                const id: u16 = if (k == slots - 1) 512 else if (rng.boolean()) rng.uintLessThan(u16, 512) else base[rng.uintLessThan(usize, 2)] + rng.uintLessThan(u16, 8);
                self.p[r * slots + k] = id;
            }
        }
        for (self.y) |*v| v.* = if (rng.uintLessThan(u8, 23) == 0) -0.0 else rng.floatNorm(f32) * 0.05;
        for (self.w) |*v| v.* = if (rng.uintLessThan(u8, 29) == 0) -0.0 else rng.floatNorm(f32) * 0.2;
        if (fill != .nonfinite) {
            if (self.split == .full) for (0..slots) |k| {
                self.y[k * n] = @bitCast(@as(u32, 0x80000001));
                self.w[k] = 0x1p-4;
                self.p[k] = if (k == slots - 1) 512 else @intCast(256 + k);
            };
            return;
        }
        // a row each of NaN, +Inf, -Inf, both infinities and Inf x 0: every 7th column, a slot a column
        for (0..self.shape.rows) |r| for (0..n) |col| {
            if (col % 7 != 0) continue;
            const k = (col / 7) % slots;
            const at = (r * slots + k) * n + col;
            switch (r % 5) {
                0 => self.y[at] = std.math.nan(f32),
                1 => self.y[at] = std.math.inf(f32),
                2 => self.y[at] = -std.math.inf(f32),
                3 => {
                    self.y[at] = std.math.inf(f32);
                    self.y[(r * slots + (k + 1) % slots) * n + col] = -std.math.inf(f32);
                },
                else => {
                    self.y[at] = std.math.inf(f32);
                    self.w[r * slots + k] = 0;
                },
            }
        };
    }

    /// Integers: picks inside the subtree, routed slots mirrored (same y, negated weight), the shared slot 3 x 1.
    fn stress(self: *Tree, rng: std.Random) void {
        const s = subtreeOf(self.split);
        const lo: u16 = @intCast(s.start * canon.leaf_ids);
        const span: u16 = @intCast(@min((@as(usize, 1) << @intCast(s.log)) * canon.leaf_ids, 512 - s.start * canon.leaf_ids));
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            for (0..slots - 1) |k| self.p[r * slots + k] = lo + rng.uintLessThan(u16, span);
            self.p[r * slots + slots - 1] = if (s.start + (@as(usize, 1) << @intCast(s.log)) == canon.leaves) 512 else lo + rng.uintLessThan(u16, span);
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

    pub fn run(self: *Tree) !void {
        const s = subtreeOf(self.split);
        var a: cuda.Args = .{};
        for (self.args) |ptr| a.add(ptr);
        a.add(self.out);
        a.add(@as(c_int, @intCast(self.shape.n)));
        a.add(@as(c_int, slots));
        a.add(@as(c_int, @intCast(s.start)));
        a.add(@as(c_int, s.log));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.shape.n / 256) }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Tree) usize {
        return self.shape.rows * self.shape.n;
    }

    /// The host canonical subtree of one output (exl3_canon.subtree, the fp32 reference of this kernel).
    fn expected(self: *const Tree, i: usize) f32 {
        const r = i / self.shape.n;
        const col = i % self.shape.n;
        var y: [slots]f32 = undefined;
        var w: [slots]f32 = undefined;
        var p: [slots]u16 = undefined;
        for (0..slots) |k| {
            y[k] = self.y[(r * slots + k) * self.shape.n + col];
            w[k] = self.w[r * slots + k];
            p[k] = @intCast(self.p[r * slots + k]);
        }
        const s = subtreeOf(self.split);
        return canon.subtree(&y, &w, &p, s.start, s.log);
    }

    /// Float64's sum of the products of the slots whose leaf the subtree holds, and the sum of their magnitudes.
    fn terms(self: *const Tree, i: usize, buf: *[slots]f64) usize {
        const r = i / self.shape.n;
        const col = i % self.shape.n;
        var m: usize = 0;
        for (0..slots) |k| {
            if (!self.inRange(self.p[r * slots + k])) continue;
            buf[m] = @as(f64, self.y[(r * slots + k) * self.shape.n + col]) * @as(f64, self.w[r * slots + k]);
            m += 1;
        }
        return m;
    }

    pub fn reference(self: *Tree, want: []f64, mag: []f64) !void {
        var t: [slots]f64 = undefined;
        for (want, mag, 0..) |*w, *m, i| {
            const cnt = self.terms(i, &t);
            w.* = 0;
            m.* = 0;
            for (t[0..cnt]) |x| {
                w.* += x;
                m.* += @abs(x);
            }
        }
    }

    pub fn simulate(self: *Tree, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [slots]f64 = undefined;
        for (out, 0..) |*o, i| o.* = core.contract.rows.sumIn(acc, order, t[0..self.terms(i, &t)]);
    }

    pub fn check(self: *Tree, gate: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(self.y), std.mem.sliceAsBytes(self.w), std.mem.sliceAsBytes(self.p) };
        return verify(&self.rig, &self.msg, treeSpec(self.split, self.fault), self.shape, gate, &bytes, want, mag, self, Tree.expected);
    }
};

/// Guards, inputs, then each output against the host tree, float64's class and bound (gate 1: finite and not).
fn verify(rig: *rig_.Rig, msg: []u8, s: c.Spec, shape: c.Shape, gate: g.Gate, inputs: []const []const u8, want: []const f64, mag: []const f64, ctx: anytype, comptime expected: anytype) !?[]const u8 {
    if (!try rig.intact()) return "a payload guard changed";
    const seen = try rig.inputBytes();
    var at: usize = 0;
    for (inputs) |bytes| {
        if (!std.mem.eql(u8, seen[at..][0..bytes.len], bytes)) return "an immutable input changed";
        at += bytes.len;
    }
    const out = try rig.outputBytes();
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

pub const Finish = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    fault: Fault,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ps: [canon.max_pieces]canon.Piece = undefined,
    np: usize = 0,
    /// Piece i's rows are stored at slot np - 1 - i: the meta's source index is exercised, not the identity.
    vals: []f32 = &.{},
    meta: []i32 = &.{},
    out: u64 = 0,
    args: [2]u64 = .{ 0, 0 },
    msg: [220]u8 = undefined,

    pub fn deinit(self: *Finish) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Finish);

    pub fn kernel(self: *Finish) g.Kernel {
        return .{ .spec = finishSpec(self.fault), .ctx = self, .vt = &vt, .control = controlGate(self.fault) };
    }

    fn at(self: *const Finish, piece: usize, i: usize) usize {
        return (self.np - 1 - piece) * self.shape.rows * self.shape.n + i;
    }

    pub fn prepare(self: *Finish, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (refused(shape) or shape.k < 1 or shape.k > 8) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        self.np = canon.pieces(&self.ps, shape.k);
        const per = shape.rows * shape.n;
        self.vals = try a.alloc(f32, self.np * per);
        self.meta = try a.alloc(i32, 3 * self.np);
        for (self.ps[0..self.np], 0..) |p, i| self.meta[3 * i ..][0..3].* = .{ p.start, p.log, @intCast(self.np - 1 - i) };
        var prng = std.Random.DefaultPrng.init(seed ^ 0x66696e69);
        const rng = prng.random();
        switch (fill) {
            .stress => self.stress(rng),
            else => self.random(rng, fill),
        }
        const dg = try self.rig.input(std.mem.sliceAsBytes(self.vals));
        const dmeta = try self.rig.input(std.mem.sliceAsBytes(self.meta));
        self.args = .{ dg, dmeta };
        self.out = try self.rig.output(per * 4);
        return true;
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
        for (0..self.shape.rows * self.shape.n) |i| {
            var v: [canon.max_pieces]f32 = undefined;
            for (0..self.np / 2) |j| {
                v[2 * j] = big(rng);
                v[2 * j + 1] = -v[2 * j];
            }
            if (self.np % 2 == 1) v[self.np - 1] = 3;
            // a mirrored order would cancel its own roundings (halves and lanes add pairs symmetrically): break it
            rng.shuffle(f32, v[0..self.np]);
            for (v[0..self.np], 0..) |x, piece| self.vals[self.at(piece, i)] = x;
        }
    }

    pub fn run(self: *Finish) !void {
        var a: cuda.Args = .{};
        a.add(self.args[0]);
        a.add(self.out);
        a.add(self.args[1]);
        a.add(@as(c_int, @intCast(self.np)));
        a.add(@as(c_int, @intCast(self.shape.n)));
        a.add(@as(c_longlong, @intCast(self.shape.rows * self.shape.n)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.shape.n / 256) }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Finish) usize {
        return self.shape.rows * self.shape.n;
    }

    /// The pieces of one output, in the order the finish merges them.
    fn pieceValues(self: *const Finish, i: usize, buf: *[canon.max_pieces]f32) []const f32 {
        for (0..self.np) |piece| buf[piece] = self.vals[self.at(piece, i)];
        return buf[0..self.np];
    }

    fn expected(self: *const Finish, i: usize) f32 {
        var buf: [canon.max_pieces]f32 = undefined;
        return canon.finish(self.ps[0..self.np], self.pieceValues(i, &buf));
    }

    pub fn reference(self: *Finish, want: []f64, mag: []f64) !void {
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

    pub fn simulate(self: *Finish, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [canon.max_pieces]f32 = undefined;
        var t: [canon.max_pieces]f64 = undefined;
        for (out, 0..) |*o, i| {
            const v = self.pieceValues(i, &buf);
            for (v, t[0..v.len]) |x, *y| y.* = x;
            o.* = core.contract.rows.sumIn(acc, order, t[0..v.len]);
        }
    }

    pub fn check(self: *Finish, gate: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(self.vals), std.mem.sliceAsBytes(self.meta) };
        return verify(&self.rig, &self.msg, finishSpec(self.fault), self.shape, gate, &bytes, want, mag, self, Finish.expected);
    }
};

/// Both kernels, each fault of each as a control, over the flashnext_batch image and the controls image.
pub const Canons = struct {
    mod: cuda.Module,
    trees: [7]Tree,
    finishes: [3]Finish,

    const tree_plan = [7]struct { Split, Fault }{
        .{ .low, .none },         .{ .high, .none },         .{ .full, .none },      .{ .full, .flat },
        .{ .full, .nan_dropped }, .{ .full, .writes_input }, .{ .full, .bf16_adds },
    };
    const finish_plan = [3]Fault{ .none, .flat, .bf16_adds };

    fn treeSymbol(f: Fault) [:0]const u8 {
        return switch (f) {
            .none => "fn_exl3_leaf_tree",
            .flat => "tf_ctl_leaf_tree_flat",
            .nan_dropped => "tf_ctl_leaf_tree_nan_dropped",
            .writes_input => "tf_ctl_leaf_tree_writes_input",
            .bf16_adds => "tf_ctl_leaf_tree_bf16_adds",
        };
    }

    fn finishSymbol(f: Fault) [:0]const u8 {
        return switch (f) {
            .none => "fn_exl3_leaf_finish",
            .flat => "tf_ctl_leaf_finish_flat",
            else => "tf_ctl_leaf_finish_bf16_adds",
        };
    }

    pub fn init(gpu: check.Gpu, s: cuda.Stream, controls: cuda.Module) !Canons {
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
        errdefer mod.unload();
        var self: Canons = .{ .mod = mod, .trees = undefined, .finishes = undefined };
        for (tree_plan, &self.trees) |plan, *t| {
            const f = try (if (plan[1] == .none) mod else controls).function(treeSymbol(plan[1]));
            t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .split = plan[0], .fault = plan[1] };
        }
        for (finish_plan, &self.finishes) |fault, *x| {
            const f = try (if (fault == .none) mod else controls).function(finishSymbol(fault));
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .fault = fault };
        }
        return self;
    }

    pub fn deinit(self: *Canons) void {
        for (&self.trees) |*t| t.deinit();
        for (&self.finishes) |*x| x.deinit();
        self.mod.unload();
    }

    pub fn add(self: *Canons, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (&self.trees) |*t| try list.append(a, t.kernel());
        for (&self.finishes) |*x| try list.append(a, x.kernel());
    }
};
