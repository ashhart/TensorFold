//! tf_heads_rows under the contract on HIP: the causal kernel's fp32 (heads, len, d) rows to row-major o_proj input.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = if (k == .bf16) "hip tf_heads_rows (bf16)" else "hip tf_heads_rows (fp16)",
        .accum = .none,
        .depth = none,
        .out = kit.elem(k),
        // rows: a prompt block (1, a mid one, a whole 2048); n: query heads; k: head width (Qwen3-8B: 32 of 128)
        .served = &.{ .{ .rows = 1, .n = 32, .k = 128 }, .{ .rows = 300, .n = 32, .k = 128 }, .{ .rows = 2048, .n = 32, .k = 128 } },
    };
}

/// shape.n heads of shape.rows rows of shape.k fp32 values, each to (row, head, j) of the activation kind.
pub const HeadsRows = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    att: []f64 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) HeadsRows {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *HeadsRows) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *HeadsRows) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(HeadsRows);

    pub fn prepare(self: *HeadsRows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const n = shape.rows * shape.n * shape.k;
        // fp32 attention outputs: spread past the activation's step so the rounding is exercised everywhere
        self.att = try kit.floats(a, prng.random(), n, -40, 40);
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, i| {
            self.att[(i * 97 + 3) % n] = v;
        };
        self.in = try self.rig.input(try kit.f32Bytes(a, self.att));
        self.out = try self.rig.output(n * kit.size(self.k));
        return true;
    }

    pub fn run(self: *HeadsRows) !void {
        const s = self.shape;
        try self.l.tf_heads_rows(kit.ptr(?[*]const f32, self.in), kit.ptr(?*anyopaque, self.out), @backingInt(self.k), @intCast(s.rows), @intCast(s.n), @intCast(s.k), self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *HeadsRows) usize {
        return self.shape.rows * self.shape.n * self.shape.k;
    }

    /// Output (r, h, j) is input (h, r, j), exactly; the bound adds the output's one rounding.
    pub fn reference(self: *HeadsRows, want: []f64, mag: []f64) !void {
        const s = self.shape;
        for (0..s.rows) |r| for (0..s.n) |h| for (0..s.k) |j| {
            const v = self.att[(h * s.rows + r) * s.k + j];
            want[(r * s.n + h) * s.k + j] = v;
            mag[(r * s.n + h) * s.k + j] = @abs(v);
        };
    }
};
