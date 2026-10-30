//! tf_moe_combine_gate under the contract on HIP: a shared expert beside the stack, bit-equal to tf_moe_combine.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// The combine's output kind: fp16, bf16 or fp32.
pub const Which = enum { gate_f16, gate_bf16, gate_f32 };

pub fn kindOf(w: Which) kit.Kind {
    return switch (w) {
        .gate_f16 => .f16,
        .gate_bf16 => .bf16,
        .gate_f32 => .f32,
    };
}

fn none(_: c.Shape) usize {
    return 0;
}

/// Each routed slot's product and add in order, then the shared expert's product and add.
fn slotChain(s: c.Shape) usize {
    return 2 * s.k + 1;
}

const odd = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
const names = [3][]const u8{
    "hip tf_moe_combine_gate (fp16)",
    "hip tf_moe_combine_gate (bf16)",
    "hip tf_moe_combine_gate (fp32)",
};

/// Flash Next's 2,560-wide hidden at its 10 picks plus a shared expert, a decode row to a prompt chunk.
pub fn spec(which: Which) c.Spec {
    const sums = &[_]c.Shape{
        .{ .rows = 1, .n = 2560, .k = 11 },
        .{ .rows = 16, .n = 2560, .k = 11 },
        .{ .rows = 512, .n = 2560, .k = 11 },
    };
    return .{
        .name = names[@backingInt(which)],
        .accum = .f32,
        .depth = slotChain,
        .out = kit.elem(kindOf(which)),
        .replaces = "tf_moe_combine",
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = sums,
    };
}

/// shape.rows rows of shape.k slots (shape.k - 1 routed and the shared expert) over shape.n columns.
pub const Gate = struct {
    rig: rig_.Rig,
    which: Which,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{}, // y [rows][slots][n]: the routed slots, the last one never read
    w: []f64 = &.{}, // weights [rows][slots], the shared expert's last
    shared: []f64 = &.{}, // the shared expert's rows [rows][n]
    in: [3]u64 = @splat(0), // y, wts, shared
    stack: u64 = 0, // y with the shared expert in its last slot: tf_moe_combine's input
    out: u64 = 0,
    base: u64 = 0,
    bytes: []u8 = &.{}, // y's bytes as the kind's width, so both kernels read the same routed rows

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Gate {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .which = w, .l = l, .k = kindOf(w), .arena = .init(a) };
    }

    pub fn deinit(self: *Gate) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Gate) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Gate);

    pub fn prepare(self: *Gate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k < 2) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const s = shape;
        self.x = try a.alloc(f64, s.rows * s.k * s.n);
        self.w = try a.alloc(f64, s.rows * s.k);
        self.shared = try a.alloc(f64, s.rows * s.n);
        for (self.x) |*v| v.* = @as(f32, @floatCast(rng.float(f64) * 4 - 2));
        for (self.w) |*v| v.* = @as(f32, @floatCast(rng.float(f64)));
        for (self.shared) |*v| v.* = @as(f32, @floatCast(rng.float(f64) * 2 - 1));
        if (fill == .stress) self.stress(rng);
        if (fill == .nonfinite) self.plant();
        // y's last slot holds the shared expert for the combine; the gate's y keeps its own there, unread
        const stack = try a.dupe(f64, self.x);
        for (0..s.rows) |r| for (0..s.n) |col| {
            stack[(r * s.k + s.k - 1) * s.n + col] = self.shared[r * s.n + col];
        };
        self.bytes = try kit.f32Bytes(a, self.x);
        self.in[0] = try self.rig.input(self.bytes);
        self.in[1] = try self.rig.input(try kit.f32Bytes(a, self.w));
        self.in[2] = try self.rig.input(try kit.f32Bytes(a, self.shared));
        self.stack = try self.rig.input(try kit.f32Bytes(a, stack));
        self.out = try self.rig.output(s.rows * s.n * kit.size(self.k));
        self.base = try self.rig.scratch(s.rows * s.n * kit.size(self.k));
        return true;
    }

    /// Gate 4: integers whose routed slots undo each other in pairs, so only fp32 sums hold the bound.
    fn stress(self: *Gate, rng: std.Random) void {
        const s = self.shape;
        const half = (s.k - 1) / 2;
        for (0..s.rows) |r| {
            // weights repeat every `half` slots, so slot j and slot j + half cancel at the same weight
            for (0..s.k) |j| self.w[r * s.k + j] = @floatFromInt(200 + j % half);
            for (0..s.n) |col| {
                self.shared[r * s.n + col] = @floatFromInt(rng.uintLessThan(u32, 8));
                for (0..s.k - 1) |j| {
                    const y = &self.x[(r * s.k + j) * s.n + col];
                    const nudge: f64 = if (j == half) 1 else 0;
                    const big: f64 = @floatFromInt(500 + rng.uintLessThan(u32, 600));
                    y.* = if (j < half) big else nudge - self.x[(r * s.k + j - half) * s.n + col];
                }
            }
        }
    }

    /// NaN in a routed slot of row 0, +Inf and -Inf in two more, and an infinite shared row in the last row.
    fn plant(self: *Gate) void {
        const s = self.shape;
        const at = [_]usize{ 0, s.n + 3, 2 * s.n + 9 }; // (slot * n + column): all of row 0's three slots
        for (odd, at) |v, i| self.x[i] = v;
        const last = self.shared.len - 1;
        self.shared[last] = std.math.inf(f64);
    }

    pub fn run(self: *Gate) !void {
        const s = self.shape;
        try self.l.tf_moe_combine_gate(
            kit.ptr(?[*]const f32, self.in[0]),
            kit.ptr(?[*]const f32, self.in[1]),
            kit.ptr(?[*]const f32, self.in[2]),
            kit.ptr(?*anyopaque, self.out),
            @backingInt(self.k),
            @intCast(s.rows),
            @intCast(s.k),
            @intCast(s.n),
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    /// tf_moe_combine on the same rows with the shared expert stacked as y's last slot.
    pub fn replaced(self: *Gate) ![]const u8 {
        const s = self.shape;
        try self.l.tf_moe_combine(
            kit.ptr(?[*]const f32, self.stack),
            kit.ptr(?[*]const f32, self.in[1]),
            kit.ptr(?*anyopaque, self.base),
            @backingInt(self.k),
            @intCast(s.rows),
            @intCast(s.k),
            @intCast(s.n),
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, s.rows * s.n * kit.size(self.k));
    }

    pub fn count(self: *Gate) usize {
        return self.shape.rows * self.shape.n;
    }

    /// The routed slots' sum plus the shared expert's, in float64, with the terms' magnitude for the bound.
    pub fn reference(self: *Gate, want: []f64, mag: []f64) !void {
        const s = self.shape;
        @memset(mag, 0);
        for (0..s.rows) |r| for (0..s.n) |col| {
            var acc: f64 = 0;
            var abs: f64 = 0;
            for (0..s.k - 1) |j| {
                const t = self.x[(r * s.k + j) * s.n + col] * self.w[r * s.k + j];
                acc += t;
                abs += @abs(t);
            }
            const sh = self.shared[r * s.n + col] * self.w[r * s.k + s.k - 1];
            want[r * s.n + col] = acc + sh;
            mag[r * s.n + col] = abs + @abs(sh);
        };
    }

    pub fn simulate(self: *Gate, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        var t: [16]f64 = undefined;
        for (0..s.rows) |r| for (0..s.n) |col| {
            for (0..s.k - 1) |j| t[j] = self.x[(r * s.k + j) * s.n + col] * self.w[r * s.k + j];
            t[s.k - 1] = self.shared[r * s.n + col] * self.w[r * s.k + s.k - 1];
            out[r * s.n + col] = rows.sumIn(acc, order, t[0..s.k]);
        };
    }
};
