//! tf_tail under the contract on HIP: the decode tail's residual add (a row or the slots' sum) and the norm after it.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// What the residual adds: a row of the kind (slots 0), or a MoE layer's slots' fp32 rows times their weights.
pub const Mode = enum { plain, moe };

/// Qwen3.6-35B-A3B's MoE tail: eight routed experts and the shared one.
const slots = 9;
const threads = 512;
const widest = 8192;
const eps: f32 = 1e-6;

/// A term's square, its thread's strided chain, its wave's xor tree, then the block's 16 wave partials in order.
fn depthPlain(s: c.Shape) usize {
    return 1 + (s.k + threads - 1) / threads + 5 + threads / 32;
}

/// The slots' products and adds, in slot order, before the norm's sum.
fn depthMoe(s: c.Shape) usize {
    return depthPlain(s) + 2 * slots;
}

/// The 35B's 2,048 at a decode row and a window's rows, the 27B's 5,120, and the widest row the launch takes.
const served_plain = [_]c.Shape{
    .{ .rows = 1, .n = 2048, .k = 2048 },
    .{ .rows = 16, .n = 2048, .k = 2048 },
    .{ .rows = 1, .n = 5120, .k = 5120 },
    .{ .rows = 16, .n = 5120, .k = 5120 },
    .{ .rows = 3, .n = widest, .k = widest },
};

/// The 35B's MoE tail at a decode row and a window's rows, and the widest row the launch takes.
const served_moe = [_]c.Shape{
    .{ .rows = 1, .n = 2048, .k = 2048 },
    .{ .rows = 16, .n = 2048, .k = 2048 },
    .{ .rows = 3, .n = widest, .k = widest },
};

const why = "a 16-bit output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const why_bf16 = why ++ "; fp16's overflowing sums give 0, which the bound's residual roundings put at 51x to 85x";

pub fn spec(mode: Mode, k: kit.Kind) c.Spec {
    const names = [2][2][]const u8{
        .{ "hip tf_tail fp16 (y fp16)", "hip tf_tail bf16 (y bf16)" },
        .{ "hip tf_tail fp16 (9 fp32 slots)", "hip tf_tail bf16 (9 fp32 slots)" },
    };
    return .{
        .name = names[@backingInt(mode)][@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = if (mode == .moe) &depthMoe else &depthPlain,
        // the residual's add, the mean, eps, rsqrtf (2 ulps), the products by inv and by the weight
        .ops = 7,
        // the residual rounded before the norm reads it, directly and through the sum; the slots' sum rounded too
        .mids = if (mode == .moe) 4 else 2,
        .out = kit.elem(k),
        // a pass over the row is 4 elements a thread of 512, its last ragged
        .tiles = .{ .{}, .{}, .{ .size = 4 * threads } },
        .served = if (mode == .moe) &served_moe else &served_plain,
        .sums = if (k == .bf16) .{ .f16 = 40, .bf16 = 0, .why = why_bf16 } else .{ .bf16 = 0, .why = why },
    };
}

/// shape.rows rows of shape.k: x [rows][k] of the kind updated in place, normed [rows][k] of the kind.
pub const Tail = struct {
    rig: rig_.Rig,
    mode: Mode,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    y: []f64 = &.{}, // [rows][k], or the slots' [rows][slots][k]
    wts: []f64 = &.{}, // [rows][slots]
    w: []f64 = &.{},
    x_bytes: []u8 = &.{},
    in: [3]u64 = @splat(0), // y, wts (slots only), the norm's weight
    out: [2]u64 = @splat(0), // x, normed

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, m: Mode) Tail {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .mode = m, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Tail) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Tail) g.Kernel {
        return .{ .spec = spec(self.mode, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Tail);

    fn signed(rng: std.Random, lo: f64, hi: f64) f64 {
        const v = lo + rng.float(f64) * (hi - lo);
        return if (rng.boolean()) v else -v;
    }

    /// x in +-[1, 2) and an addend of x's sign at about a quarter to a half of it, so the add never cancels.
    pub fn prepare(self: *Tail, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.k > widest) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = self.k;
        const n = shape.rows * shape.k;
        const moe = self.mode == .moe;
        self.x = try a.alloc(f64, n);
        self.y = try a.alloc(f64, if (moe) n * slots else n);
        self.wts = try a.alloc(f64, if (moe) shape.rows * slots else 0);
        self.w = try kit.floats(a, rng, shape.k, 0.75, 1.25);
        for (self.x) |*v| v.* = kit.round(k, signed(rng, 1, 2));
        for (self.wts) |*v| v.* = @as(f32, @floatCast(0.05 + rng.float(f64) * 0.1)); // nine weights near 0.1 each
        for (0..n) |i| {
            const sign: f64 = if (self.x[i] < 0) -1 else 1;
            if (!moe) {
                self.y[i] = kit.round(k, sign * (0.25 + rng.float(f64) * 0.25) * @abs(self.x[i]));
                continue;
            }
            const r = i / shape.k;
            for (0..slots) |j| {
                const part = sign * (0.25 + rng.float(f64) * 0.25) * @abs(self.x[i]) / (slots * 0.1);
                self.y[(r * slots + j) * shape.k + i % shape.k] = @as(f32, @floatCast(part));
            }
        }
        if (fill == .stress) try self.stress(seed);
        if (fill == .nonfinite) self.plant();
        self.x_bytes = try a.alloc(u8, n * kit.size(k));
        for (self.x, 0..) |v, i| kit.put(k, self.x_bytes, i, v);
        const yb = if (moe) try kit.f32Bytes(a, self.y) else blk: {
            const b = try a.alloc(u8, n * kit.size(k));
            for (self.y, 0..) |v, i| kit.put(k, b, i, v);
            break :blk b;
        };
        self.in[0] = try self.rig.input(yb);
        self.in[1] = if (moe) try self.rig.input(try kit.f32Bytes(a, self.wts)) else 0;
        self.in[2] = try self.rig.input(try kit.f32Bytes(a, self.w));
        for (&self.out) |*at| at.* = try self.rig.output(self.x_bytes.len);
        return true;
    }

    /// Gate 4: the norm's rows of integers (rows.zig's) and nothing added, so its sums are exact in fp32.
    fn stress(self: *Tail, seed: u64) !void {
        const s = self.shape;
        var r = try rows.Rows.init(self.arena.allocator(), .rms, .{ .rows = s.rows, .n = s.k, .k = s.k });
        r.fill(.stress, seed);
        for (self.x, r.x) |*v, b| v.* = c.fromBf16(b);
        @memset(self.y, 0);
    }

    /// NaN in row 0's x, +Inf in row 1's addend and -Inf in row 2's (a slot's row), an overflowing sum in row 3.
    fn plant(self: *Tail) void {
        const s = self.shape;
        const big: f64 = if (self.k == .f16) 60000 else 3e38;
        if (5 < self.x.len) self.x[5] = std.math.nan(f64);
        const at = [_]usize{ s.k + 9, 2 * s.k + 1 };
        for ([_]f64{ std.math.inf(f64), -std.math.inf(f64) }, at) |v, i| if (i < self.x.len) switch (self.mode) {
            .moe => self.y[(i / s.k * slots + 1) * s.k + i % s.k] = v,
            .plain => self.y[i] = v,
        };
        if (3 * s.k + 2 < self.x.len) self.x[3 * s.k + 2] = kit.round(self.k, big);
        if (3 * s.k + 2 < self.x.len and self.mode == .plain) self.y[3 * s.k + 2] = kit.round(self.k, big);
    }

    fn ptr(at: u64) ?*anyopaque {
        return @ptrFromInt(at);
    }

    /// x back to the prepared rows, then the launch as the engine makes it (no weights for a plain row).
    pub fn run(self: *Tail) !void {
        try self.rig.put(self.out[0], self.x_bytes);
        const s = self.shape;
        const wts: ?[*]const f32 = if (self.mode == .moe) @ptrFromInt(self.in[1]) else null;
        try self.l.tf_tail(
            ptr(self.out[0]),
            ptr(self.in[0]),
            wts,
            @ptrFromInt(self.in[2]),
            ptr(self.out[1]),
            @backingInt(self.k),
            @intCast(s.rows),
            if (self.mode == .moe) slots else 0,
            @intCast(s.k),
            eps,
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Tail) usize {
        return 2 * self.x.len;
    }

    /// Element i's addend in float64, and the magnitude of the terms the slots' sum carries into it.
    fn addend(self: *Tail, i: usize) struct { f64, f64 } {
        const s = self.shape;
        if (self.mode == .plain) return .{ self.y[i], 0 };
        const r = i / s.k;
        var acc: f64 = 0;
        var abs: f64 = 0;
        for (0..slots) |j| {
            const t = self.y[(r * slots + j) * s.k + i % s.k] * self.wts[r * slots + j];
            acc += t;
            abs += @abs(t);
        }
        return .{ acc, abs };
    }

    /// x + the addend, then rms(x + addend) * w, in float64; the norm's magnitude as rows.zig's.
    pub fn reference(self: *Tail, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const n = self.x.len;
        for (0..n) |i| {
            const b, const m = self.addend(i);
            const v = self.x[i] + b;
            want[i] = if (std.math.isInf(kit.round(self.k, v))) kit.round(self.k, v) else v; // past the kind's range
            mag[i] = m;
        }
        for (0..s.rows) |r| {
            const row = want[r * s.k ..][0..s.k];
            var ss: f64 = 0;
            for (row) |v| ss += v * v;
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(s.k)) + eps);
            for (row, 0..) |v, j| {
                const i = n + r * s.k + j;
                want[i] = v * inv * self.w[j];
                mag[i] = @abs(want[i]) / 2 + mag[r * s.k + j] * inv * self.w[j];
            }
        }
    }

    /// Gate 4's rows, nothing added: the norm's sum of squares in `acc` and `order`, the rest exact.
    pub fn simulate(self: *Tail, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        const n = self.x.len;
        const t = try self.rig.a.alloc(f64, s.k);
        defer self.rig.a.free(t);
        for (0..s.rows) |r| {
            const row = self.x[r * s.k ..][0..s.k];
            for (t, row) |*v, x| v.* = x * x;
            const ss = rows.sumIn(acc, order, t);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(s.k)) + eps);
            for (row, 0..) |x, j| {
                out[r * s.k + j] = x;
                out[n + r * s.k + j] = x * inv * self.w[j];
            }
        }
    }
};
