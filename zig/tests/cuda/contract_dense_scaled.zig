//! dense_rows' scaled epilogue (tf_dense_rows_scaled_*) under the contract: bit-equal to its unscaled tile, scaled.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const moe_k = @import("contract_moe.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const bfr = core.bf16_round;

pub const tiles = moe_k.dense_tiles;

fn chain(shape: c.Shape) usize {
    return moe_k.denseSpec(0).depth(shape);
}

/// Dory's head (131,072 x 2,560) cut to an odd 4,103 columns, at the row counts the wrapper sends each tile.
pub fn spec(i: usize) c.Spec {
    const names = [_][]const u8{ "cuda dense rows tf_dense_rows_scaled_1 (Dory's logits)", "cuda dense rows tf_dense_rows_scaled_4", "cuda dense rows tf_dense_rows_scaled_16", "cuda dense rows tf_dense_rows_scaled_32" };
    const m = [_][2]usize{ .{ 1, 1 }, .{ 4, 3 }, .{ 16, 9 }, .{ 32, 40 } };
    const served = [_][2]c.Shape{
        .{ .{ .rows = m[0][0], .n = 4103, .k = 2560 }, .{ .rows = m[0][1], .n = 384, .k = 2560 } },
        .{ .{ .rows = m[1][0], .n = 4103, .k = 2560 }, .{ .rows = m[1][1], .n = 384, .k = 2560 } },
        .{ .{ .rows = m[2][0], .n = 4103, .k = 2560 }, .{ .rows = m[2][1], .n = 384, .k = 2560 } },
        .{ .{ .rows = m[3][0], .n = 4103, .k = 2560 }, .{ .rows = m[3][1], .n = 384, .k = 2560 } },
    };
    return .{
        .name = names[i],
        .accum = .f32,
        .depth = chain,
        .ops = 2, // the scale's product, after the bf16 rounding (mids)
        .mids = 2,
        .out = .f32,
        .replaces = "its tf_dense_rows tile's fp32 outputs, rounded to bf16 and times the scale in fp32 on the host",
        .tiles = .{ .{ .size = tiles[i] }, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } },
        .served = &served[i],
    };
}

pub const Scaled = struct {
    rig: rig_.Rig,
    tile: usize,
    g: *const cuda.dense_rows.DenseRows,
    d: ?Dense = null,
    scale: []f32 = &.{},
    dev: [4]u64 = @splat(0),
    base: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Scaled) void {
        if (self.d) |d| d.deinit();
        self.rig.gpu.gpa.free(self.scale);
        self.base.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Scaled) g.Kernel {
        return .{ .spec = spec(self.tile), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Scaled);

    pub fn prepare(self: *Scaled, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 8 != 0 or shape.k == 0 or shape.n == 0 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x77);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        a.free(self.scale);
        self.scale = try a.alloc(f32, shape.n);
        // logit_scale / std: Dory's runs about 20 to 120; a NaN and an infinity with the non-finite fill
        for (self.scale) |*v| v.* = 20 + 100 * rng.float(f32);
        if (fill == .nonfinite and shape.n > 7) {
            self.scale[3] = std.math.nan(f32);
            self.scale[7] = std.math.inf(f32);
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.scale));
        self.dev[3] = try self.rig.output(d.outputs() * 4);
        return true;
    }

    /// The wrapper's arguments at this entry's row tile, through `f` (scale `with`: the scaled kernel's).
    fn launch(self: *Scaled, f: cuda.Function, with: bool, out: u64) !void {
        const sh = self.d.?.shape;
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(@as(i64, @intCast(sh.k)));
        a.add(self.dev[1]);
        if (with) a.add(self.dev[2]);
        a.add(out);
        a.add(@as(i64, @intCast(sh.n)));
        for ([_]usize{ sh.rows, sh.n, sh.k }) |v| a.add(@as(i32, @intCast(v)));
        const rt = tiles[self.tile];
        const grid: cuda.Dim3 = .{ .x = @intCast((sh.n + 7) / 8), .y = @intCast((sh.rows + rt - 1) / rt), .z = 1 };
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Scaled) !void {
        try self.launch(self.g.scaled_fns[self.tile], true, self.dev[3]);
    }

    pub fn count(self: *Scaled) usize {
        return self.d.?.outputs();
    }

    fn scaled(self: *Scaled, out: []f64) void {
        const n = self.d.?.shape.n;
        for (out, 0..) |*v, i| v.* = @as(f32, @floatCast(bfr.bf16(v.*))) * self.scale[i % n];
    }

    /// float64's sums, rounded to bf16 and scaled in fp32 (the epilogue's steps on an exact sum).
    pub fn reference(self: *Scaled, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
        const n = self.d.?.shape.n;
        for (mag, 0..) |*m, i| m.* *= @abs(self.scale[i % n]);
        self.scaled(want);
    }

    pub fn simulate(self: *Scaled, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
        self.scaled(out);
    }

    /// The unscaled tile on the same inputs, its fp32 outputs rounded to bf16 and scaled on the host.
    pub fn replaced(self: *Scaled) ![]const u8 {
        const a = self.rig.gpu.gpa;
        const n = self.count();
        var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, n * 4);
        defer tmp.free();
        try self.launch(self.g.fns[self.tile], false, tmp.ptr);
        try self.base.resize(a, n * 4);
        try tmp.download(0, self.base.items);
        const v = std.mem.bytesAsSlice(f32, @as([]align(1) u8, self.base.items));
        const cols = self.d.?.shape.n;
        for (0..n) |i| v[i] = bfr.round32(v[i]) * self.scale[i % cols];
        return self.base.items;
    }
};
