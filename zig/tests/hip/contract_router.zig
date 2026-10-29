//! The router over the stored bf16 rows on HIP: the fp32-data router's logits, bit for bit.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_k = contract.rows;

/// A lane's groups of four, 128 values apart, then the wave's five-step tree.
fn depth(s: c.Shape) usize {
    return (s.k + 127) / 128 + 5;
}

const names = [2][]const u8{
    "hip tf_router_bf16_f16 (the stored bf16 rows)",
    "hip tf_router_bf16 (the stored bf16 rows)",
};

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = names[@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .replaces = if (k == .bf16) "tf_router_decode_bf16" else "tf_router_decode_f16",
        // a round's rows: 16 a block of row groups, and a lane's group of four over the hidden width
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 128, .step = 4, .min = 4 } },
        .served = &.{
            .{ .rows = 16, .n = 513, .k = 2560 },
            .{ .rows = 4, .n = 513, .k = 2560 },
            .{ .rows = 1, .n = 513, .k = 2560 },
        },
    };
}

/// shape.rows rows of shape.k, and shape.n router rows as a checkpoint holds them (bf16) and widened to fp32.
pub const Router = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    w: []f64 = &.{},
    in: u64 = 0,
    rows16: u64 = 0,
    rows32: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    terms: []f64 = &.{},

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, k: kit.Kind) Router {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = k, .arena = .init(a) };
    }

    pub fn deinit(self: *Router) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Router) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Router);

    /// The rows of the router the engine serves: the experts and the shared expert's gate.
    fn experts(self: *const Router) usize {
        return self.shape.n;
    }

    pub fn prepare(self: *Router, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.k < 4 or shape.k % 4 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.k;
        const e = shape.n;
        self.x = try a.alloc(f64, n);
        self.w = try a.alloc(f64, e * shape.k);
        self.terms = try a.alloc(f64, @max(shape.k, 1));
        // random weights under 1/8 keep the logits small; the stress rows are integers gate 4 can sum exactly
        for (self.w) |*v| v.* = switch (fill) {
            .stress => 0,
            else => kit.round(.bf16, (rng.float(f64) * 2 - 1) / 8),
        };
        for (self.x) |*v| v.* = switch (fill) {
            .stress => 0,
            else => kit.round(self.k, rng.float(f64) * 2 - 1),
        };
        if (fill == .stress) self.stress(rng) else if (fill == .nonfinite) self.plant();
        const xb = try a.alloc(u8, n * kit.size(self.k));
        for (self.x, 0..) |v, i| kit.put(self.k, xb, i, v);
        const w16 = try a.alloc(u8, 2 * e * shape.k);
        const w32 = try a.alloc(u8, 4 * e * shape.k);
        for (self.w, 0..) |v, i| {
            const h = c.toBf16(@floatCast(v));
            std.mem.writeInt(u16, w16[2 * i ..][0..2], h, .little);
            std.mem.writeInt(u32, w32[4 * i ..][0..4], @as(u32, h) << 16, .little);
        }
        self.in = try self.rig.input(xb);
        self.rows16 = try self.rig.input(w16);
        self.rows32 = try self.rig.input(w32);
        self.out = try self.rig.output(shape.rows * e * 4);
        self.base = try self.rig.scratch(shape.rows * e * 4);
        return true;
    }

    /// Gate 4's rows: integers a lane sums exactly in fp32, mirrored so the row's own result stays small.
    fn stress(self: *Router, rng: std.Random) void {
        const k = self.shape.k;
        const h = k / 2;
        for (0..self.shape.rows) |r| for (0..h) |i| {
            const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 128));
            self.x[r * k + i] = v;
            self.x[r * k + h + i] = v + @as(f64, @floatFromInt(@intFromBool(i < 8 and k > 16)));
        };
        if (k % 2 == 1) for (0..self.shape.rows) |r| {
            self.x[r * k + k - 1] = 0;
        };
        for (0..self.shape.n) |e| for (0..h) |i| {
            const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            self.w[e * k + i] = v;
            self.w[e * k + h + i] = -v;
        };
        if (k % 2 == 1) for (0..self.shape.n) |e| {
            self.w[e * k + k - 1] = 0;
        };
    }

    /// Gate 1: a NaN expert, an infinite expert each way, and a NaN input row.
    fn plant(self: *Router) void {
        const k = self.shape.k;
        const e = self.shape.n;
        self.w[k / 3] = std.math.nan(f64);
        self.w[1 * k + k / 2] = std.math.inf(f64);
        if (e > 2 and k / 2 + 1 < k) self.w[2 * k + k / 2 + 1] = -std.math.inf(f64);
        if (self.shape.rows > 1) self.x[k] = std.math.nan(f64);
    }

    /// One row's terms against one expert's row, into `t`.
    fn termsOf(self: *Router, r: usize, e: usize, t: []f64) void {
        const k = self.shape.k;
        for (t, 0..) |*o, i| o.* = self.x[r * k + i] * self.w[e * k + i];
    }

    pub fn run(self: *Router) !void {
        const s = self.shape;
        const ran = try self.l.routerBf16(
            @ptrFromInt(self.in),
            @ptrFromInt(self.rows16),
            @backingInt(self.k),
            @ptrFromInt(self.out),
            @intCast(s.rows),
            @intCast(s.k),
            @intCast(s.n),
            self.rig.s.handle,
        );
        if (!ran) return error.BadShape;
        try self.rig.s.synchronize();
    }

    /// The router over the fp32 rows widened from the same bf16 values.
    pub fn replaced(self: *Router) ![]const u8 {
        const s = self.shape;
        const ran = try self.l.routerWindow(
            @ptrFromInt(self.in),
            @backingInt(self.k),
            @ptrFromInt(self.rows32),
            @ptrFromInt(self.base),
            @intCast(s.rows),
            @intCast(s.k),
            @intCast(s.n),
            self.rig.s.handle,
        );
        if (!ran) return error.BadShape;
        return self.rig.bytes(self.base, s.rows * s.n * 4);
    }

    pub fn count(self: *Router) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Router, want: []f64, mag: []f64) !void {
        const t = self.terms;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |e| {
            self.termsOf(r, e, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            want[r * self.shape.n + e] = sum;
            mag[r * self.shape.n + e] = abs;
        };
    }

    /// The logits if each lane's sums ran in `acc` in `order`, the wave's tree exact.
    pub fn simulate(self: *Router, acc: c.Accum, order: rows_k.Order, out: []f64) !void {
        const t = self.terms;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |e| {
            self.termsOf(r, e, t);
            out[r * self.shape.n + e] = rows_k.sumIn(acc, order, t);
        };
    }
};
