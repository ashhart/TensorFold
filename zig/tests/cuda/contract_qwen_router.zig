//! The Qwen CUDA engine's prompt router, tf_moe_router_tiles, under the contract in place of tf_moe_router.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

/// One fma chain over the row (two a bf16 word pair), as contract_router.zig counts a Nemotron slice.
fn chain(s: c.Shape) usize {
    return s.k + 2;
}

/// The 35B-A3B's router (2,048 inputs, 256 experts and the shared gate's row) on calls from 257 rows.
pub fn spec() c.Spec {
    return .{ .name = "cuda qwen router logits tf_moe_router_tiles", .accum = .f32, .depth = chain, .out = .f32, .replaces = "tf_moe_router", .tiles = .{ .{ .size = 128 }, .{ .size = 128, .step = 16, .min = 16 }, .{} }, .served = &.{ .{ .rows = 257, .n = 257, .k = 2048 }, .{ .rows = 300, .n = 257, .k = 2048 }, .{ .rows = 4100, .n = 257, .k = 2048 } } };
}

/// x [rows, D] bf16 times the router's W [E, D] bf16: fp32 logits [rows][E]; `replaced` runs tf_moe_router.
pub const Logits = struct {
    rig: rig_.Rig,
    router: cuda.Function,
    tiles: cuda.Function,
    ctl: ?cuda.Function = null, // tf_ctl_moe_router_tiles_row, which drops the last live row and must fail gate 3
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Logits) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Logits) g.Kernel {
        var k: g.Kernel = .{ .spec = spec(), .ctx = self, .vt = &vt, .control = if (self.ctl != null) .shapes else null };
        if (self.ctl != null) k.spec.name = "cuda qwen router logits control: tiles that drop the last live row";
        return k;
    }

    const vt = g.hooks(Logits);

    pub fn prepare(self: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 32 != 0 or shape.n == 0 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x51);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * 4);
        self.base = try self.rig.scratch(d.outputs() * 4);
        return true;
    }

    pub fn run(self: *Logits) !void {
        try self.launch(true, self.out);
        try self.rig.s.synchronize();
    }

    /// tf_moe_router_tiles' 128-row tiles, or tf_moe_router's 16, on the prepared rows (rows a row apart, no cap).
    fn launch(self: *Logits, tiles: bool, out: u64) !void {
        const d = self.d.?;
        var a: cuda.Args = .{};
        a.add(self.x);
        a.add(@as(c_int, @intCast(d.shape.k)));
        a.add(self.w);
        a.add(out);
        for ([_]usize{ d.shape.rows, d.shape.k, d.shape.n }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(@as(u64, 0));
        const t: usize = if (tiles) 128 else 16;
        const grid: cuda.Dim3 = .{ .x = @intCast((d.shape.rows + t - 1) / t), .y = @intCast((d.shape.n + t - 1) / t), .z = 1 };
        const f = if (!tiles) self.router else self.ctl orelse self.tiles;
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 }, .shared = if (tiles) cuda.moe.router_tiles_shared else 0 }, self.rig.s, &a);
    }

    pub fn replaced(self: *Logits) ![]const u8 {
        try self.launch(false, self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Logits) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Logits, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Logits, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};
