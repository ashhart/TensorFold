//! Flash Next's routed prompt expert tiles under the contract on HIP (gfx11 matrix cores): the matrix tile's bytes.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const dense_k = @import("contract_wmma_dense.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const tiles = hip.launches.affine;
const Arg = dense_k.Arg;

/// gate+up: silu(gate) * up of each pair in bf16; down: each pair's fp32 sums.
pub const Part = enum { gu, down };

/// The experts stacked in a test's bank and the pairs a row: every expert takes each row once.
pub const experts = 2;
pub const slots = 2;
/// The most pairs an item of a prompt's plan holds (moe_route's tile).
pub const item_pairs = 128;
/// The largest |silu'(x)|: 1.0998 at x = 2.3994.
const slope_max = 1.1;

pub fn shapeOf(part: Part) tiles.Kernels.MoeShape {
    return if (part == .gu) tiles.Kernels.moe_gu_shape else tiles.Kernels.moe_down_shape;
}

/// Outputs a pair: the expert width for gate+up (its rows hold gate then up), the hidden width for down.
pub fn width(part: Part) usize {
    const n: usize = @intCast(shapeOf(part).n);
    return if (part == .gu) n / 2 else n;
}

/// Row r's slot s picks expert (r + s) % experts.
pub fn pick(p: usize) usize {
    return (p / slots + p % slots) % experts;
}

/// A prompt's plan as moe_route writes it: each expert's pairs in order, items of at most 128, two empty items past.
pub const Plan = struct { items: []i32, members: []i32, count: usize };

pub fn plan(a: std.mem.Allocator, pairs: usize) !Plan {
    const members = try a.alloc(i32, pairs);
    const items = try a.alloc(i32, 3 * (pairs + experts + 2));
    @memset(items, 0);
    var used: usize = 0;
    var at: usize = 0;
    for (0..experts) |e| {
        var run: usize = 0;
        for (0..pairs) |p| {
            if (pick(p) != e) continue;
            if (run == 0) {
                items[3 * used ..][0..3].* = .{ @intCast(e), @intCast(at), 0 };
                used += 1;
            }
            members[at] = @intCast(p);
            at += 1;
            items[3 * (used - 1) + 2] += 1;
            run = if (run + 1 == item_pairs) 0 else run + 1;
        }
    }
    return .{ .items = items[0 .. 3 * (used + 2)], .members = members, .count = used + 2 };
}

/// A bank of `experts` stacked expert matrices, the plan over `rows` rows' pairs, and the call's argument.
pub const Bank = struct {
    f: Affine,
    plan: Plan,
    arg: Arg,
    pairs: usize,

    /// x is the rows (gate+up) or the pairs (down); null where the shape is not the tile's.
    pub fn load(r: *rig_.Rig, a: std.mem.Allocator, part: Part, shape: c.Shape, fill: g.Fill, seed: u64) !?Bank {
        const sh = shapeOf(part);
        if (shape.rows == 0 or shape.n != width(part) or shape.k != @as(usize, @intCast(sh.k))) return null;
        const pairs = shape.rows * slots;
        const n: usize = @intCast(sh.n);
        const f = try Affine.init(a, .{ .rows = if (part == .gu) shape.rows else pairs, .n = experts * n, .k = shape.k }, dense_k.bits, dense_k.group);
        try f.fill(fill, seed);
        const pl = try plan(a, pairs);
        var arg = try dense_k.upload(r, a, f);
        const out = try r.output(pairs * shape.n * @as(usize, if (part == .gu) 2 else 4));
        if (part == .gu) arg.out16 = out else arg.out = out;
        arg.m = item_pairs;
        arg.n = sh.n;
        arg.route = .{
            .items = try r.input(std.mem.sliceAsBytes(pl.items)),
            .members = try r.input(std.mem.sliceAsBytes(pl.members)),
            .x_div = if (part == .gu) slots else 1,
        };
        return .{ .f = f, .plan = pl, .arg = arg, .pairs = pairs };
    }

    /// Each pair's outputs from the bank's columns y [x rows][experts n] (and their term magnitudes `ym`).
    pub fn gather(b: Bank, part: Part, y: []const f64, ym: ?[]const f64, want: []f64, mag: ?[]f64) void {
        const n = width(part);
        const per: usize = @intCast(shapeOf(part).n);
        for (0..b.pairs) |p| {
            const row = if (part == .gu) p / slots else p;
            const at = row * b.f.shape.n + pick(p) * per;
            for (0..n) |col| {
                const o = p * n + col;
                if (part == .down) {
                    want[o] = y[at + col];
                    if (mag) |m| m[o] = ym.?[at + col];
                    continue;
                }
                const gate = y[at + col];
                const up = y[at + n + col];
                want[o] = kit.silu(gate) * up;
                // an error e in gate moves silu(gate) * up by at most 1.1 e (|up| + up's error)
                if (mag) |m| m[o] = slope_max * ym.?[at + col] * (@abs(up) + ym.?[at + n + col]) + @abs(kit.silu(gate)) * ym.?[at + n + col];
            }
        }
    }
};

/// A group's 16 dot2 steps from zero (2 u each: an ulp of the step's largest addend), then its scale and bias fmas.
fn depth(s: c.Shape) usize {
    return 2 * 16 + 2 * (s.k / dense_k.group) + 8;
}

pub fn spec(part: Part) c.Spec {
    return .{
        .name = if (part == .gu) "hip affine_wmma_moe_gu<4> (Flash Next's expert gate and up, silu(g) u, bf16 out)" else "hip affine_wmma_moe_down<4> (Flash Next's expert down, fp32 out)",
        .accum = .f32,
        .depth = depth,
        // gate+up's epilogue: expf within 2 ulps, then the add, the quotient and the product
        .ops = if (part == .gu) 7 else 0,
        .out = if (part == .gu) .bf16 else .f32,
        .replaces = if (part == .gu) "affine_wmma_gemm over the plan, then tf_moe_act" else "affine_wmma_gemm over the plan",
        // a 64-row pass of an item; 300 rows make three items an expert, the last short
        .tiles = .{ .{ .size = 64 }, .{}, .{} },
        .served = if (part == .gu) &.{.{ .rows = 300, .n = 640, .k = 2560 }} else &.{.{ .rows = 300, .n = 2560, .k = 640 }},
    };
}

/// The plan's pairs through one expert bank: gate+up (bf16 activations) or down (fp32).
pub const Moe = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    part: Part,
    bank: ?Bank = null,
    sum: u64 = 0,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, part: Part) Moe {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a), .part = part };
    }

    pub fn deinit(self: *Moe) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn ready(self: *const Moe) bool {
        return (if (self.part == .gu) self.l.affine.moe_gu else self.l.affine.moe_down) != null;
    }

    pub fn kernel(self: *Moe) g.Kernel {
        return .{ .spec = spec(self.part), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Moe);

    pub fn prepare(self: *Moe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const b = try Bank.load(&self.rig, self.arena.allocator(), self.part, shape, fill, seed) orelse return false;
        self.bank = b;
        self.sum = try self.rig.scratch(b.pairs * @as(usize, @intCast(shapeOf(self.part).n)) * 4);
        self.base = if (self.part == .gu) try self.rig.scratch(b.pairs * width(.gu) * 2) else 0;
        return true;
    }

    pub fn run(self: *Moe) !void {
        const b = self.bank.?;
        const k = &self.l.affine;
        const items: c_int = @intCast(b.plan.count);
        const st = self.rig.s.handle;
        const took = if (self.part == .gu) try k.moeGateUp(self.l.r, b.arg, items, 0, st) else try k.moeDown(self.l.r, b.arg, items, st);
        if (!took) return error.TileRefused;
        try self.rig.s.synchronize();
    }

    /// The routed matrix tile the registry took before (fp32 sums), then moe_act for gate+up (no clamp, as served).
    pub fn replaced(self: *Moe) ![]const u8 {
        const b = self.bank.?;
        var m = b.arg;
        m.out = self.sum;
        m.out16 = 0;
        const st = self.rig.s.handle;
        try self.l.affine.blockGo(self.l.r, m, st, @intCast(b.plan.count), .matrix);
        const n = width(self.part);
        if (self.part == .down) return self.rig.bytes(self.sum, b.pairs * n * 4);
        const both = kit.ptr(?[*]const f32, self.sum);
        try self.l.tf_moe_act(both, kit.ptr(?*anyopaque, self.base), @backingInt(kit.Kind.bf16), @intCast(b.pairs), @intCast(n), 0, st);
        return self.rig.bytes(self.base, b.pairs * n * 2);
    }

    pub fn count(self: *Moe) usize {
        return self.bank.?.pairs * width(self.part);
    }

    /// MLX's factored formula in float64 for each pair's expert, then silu(gate) * up for gate+up.
    pub fn reference(self: *Moe, want: []f64, mag: []f64) !void {
        const b = self.bank.?;
        const a = self.arena.allocator();
        const y = try a.alloc(f64, 2 * b.f.outputs());
        try b.f.reference(y[0..b.f.outputs()], y[b.f.outputs()..]);
        b.gather(self.part, y[0..b.f.outputs()], y[b.f.outputs()..], want, mag);
    }

    pub fn simulate(self: *Moe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const b = self.bank.?;
        const y = try self.rig.a.alloc(f64, b.f.outputs());
        defer self.rig.a.free(y);
        try b.f.simulate(acc, order, y);
        b.gather(self.part, y, null, out, null);
    }
};
