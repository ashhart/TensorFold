//! Flash Next's indexer block scores under the contract on HIP: relu-summed heads against an fp64 reference.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const heads = 4;
const dims = 128;
const top = 4;

/// Fp32 sums over a head's dims, then over the heads.
fn depth(_: c.Shape) usize {
    return dims + heads;
}

const why = "a lane's dims 32 apart, the 32-lane xor tree, then the heads";
const rule = "a dot that is NaN adds 0 (relu by fmaxf); an infinite q reaches the blocks whose key carries its sign";

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = if (k == .bf16) "hip tf_fn_idx_scores (bf16 keys)" else "hip tf_fn_idx_scores (fp16 keys)",
        .accum = .f32,
        .depth = depth,
        .ops = 2, // the relu and the divide by sqrt(dims)
        .out = .f32,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 8, .min = 1 }, .{ .size = 8, .min = 1 }, .{} },
        // a 64-row slice over a 32K prompt's blocks, a tail of rows past one tile, and a deep cache
        .served = &.{
            .{ .rows = 64, .n = 8192, .k = dims },
            .{ .rows = 13, .n = 2051, .k = dims },
            .{ .rows = 8, .n = 32768, .k = dims },
        },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = why, .exact = false },
    };
}

/// Either the real launch or a control that drops the last head so gate 3 catches the missing sum.
pub const Fault = enum {
    none,
    one_head_short,

    fn name(f: Fault) ?[]const u8 {
        return switch (f) {
            .none => null,
            .one_head_short => "control: tf_fn_idx_scores leaves out each row's last head",
        };
    }
};

/// Rows of q against a cache of pooled block keys of the kind; every row scores every block.
pub const IdxScores = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    fault: Fault,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    q: []f32 = &.{},
    keys: []f64 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, f: Fault) IdxScores {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .fault = f, .arena = .init(a) };
    }

    pub fn deinit(self: *IdxScores) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    /// The control keeps the spec under its own name and must go red at gate 3 (shapes).
    pub fn kernel(self: *IdxScores) g.Kernel {
        var s = spec(self.k);
        const control = self.fault.name();
        if (control) |name| s.name = name;
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (control != null) .shapes else null };
    }

    const vt = g.hooks(IdxScores);

    /// Random: values in [-1, 1) exact in the kind. Stress: small integers, so every sum is exact.
    pub fn prepare(self: *IdxScores, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dims or shape.rows < 1 or shape.n <= top or shape.n > 40000) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const stress = fill == .stress;
        self.q = try a.alloc(f32, shape.rows * heads * dims);
        for (self.q) |*x| {
            x.* = if (stress) @floatFromInt(rng.intRangeAtMost(i32, -3, 3)) else @floatCast(rng.float(f64) * 2 - 1);
        }
        self.keys, const kb = try kit.values(a, rng, self.k, shape.n * dims, -1, 1);
        if (stress) for (self.keys, 0..) |*x, i| {
            x.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
            kit.put(self.k, kb, i, x.*);
        };
        if (fill == .nonfinite) self.q[heads * dims] = std.math.inf(f32); // row 1, head 0, dim 0
        const complete = try a.alloc(i32, shape.rows);
        @memset(complete, @intCast(shape.n));
        self.dev = .{
            try self.rig.input(std.mem.sliceAsBytes(self.q)),
            try self.rig.input(kb),
            try self.rig.input(std.mem.sliceAsBytes(complete)),
        };
        self.out = try self.rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(self: *IdxScores) !void {
        const s = self.shape;
        const d = self.dev;
        const run_heads: c_int = if (self.fault == .one_head_short) heads - 1 else heads;
        try self.l.fn_idx_scores(
            @ptrFromInt(d[0]),
            @ptrFromInt(d[1]),
            @ptrFromInt(d[2]),
            @ptrFromInt(self.out),
            @backingInt(self.k),
            @intCast(s.rows),
            run_heads,
            dims,
            @intCast(s.n),
            @intCast(s.n),
            top,
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    pub fn count(self: *IdxScores) usize {
        return self.shape.rows * self.shape.n;
    }

    /// fp64 scores: each head's dot over the dims, a NaN dot as 0, the rest relu'd, summed over heads, / sqrt(dims).
    pub fn reference(self: *IdxScores, want: []f64, mag: []f64) !void {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |b| {
            var total: f64 = 0;
            var size: f64 = 0;
            for (0..heads) |h| {
                var dot: f64 = 0;
                var m: f64 = 0;
                for (0..dims) |d| {
                    const p = @as(f64, self.q[(r * heads + h) * dims + d]) * self.keys[b * dims + d];
                    dot += p;
                    m += @abs(p);
                }
                if (!std.math.isNan(dot) and dot > 0) total += dot;
                size += m;
            }
            want[r * n + b] = total / @sqrt(@as(f64, dims));
            mag[r * n + b] = size / @sqrt(@as(f64, dims));
        };
    }

    /// Every sum is exact on the stress fill's integers, in any order.
    pub fn simulate(self: *IdxScores, _: c.Accum, _: g.Order, out: []f64) !void {
        const mag = try self.arena.allocator().alloc(f64, out.len);
        try self.reference(out, mag);
    }
};
