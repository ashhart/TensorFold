//! Contract suite for `hy4_rows_wide` and `hy4_rows_narrow`: bit-equal to `tf_dense_rows_1` per row.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const Tile = hr.Tile;

fn chain(s: c.Shape) usize {
    return s.k / 32 + 5;
}

const served = [2][2]c.Shape{
    .{ .{ .rows = 16, .n = 4103, .k = 3072 }, .{ .rows = 200, .n = 512, .k = 3072 } },
    .{ .{ .rows = 16, .n = 512, .k = 3072 }, .{ .rows = 5, .n = 512, .k = 3072 } },
};

pub const Case = struct {
    rig: rig_.Rig,
    fn_handle: cuda.Function,
    dense: *const cuda.dense_rows.DenseRows,
    tile: Tile,
    ctl: ?g.Gate,
    name: []const u8,
    d: ?Dense = null,
    dev: [3]u64 = @splat(0),
    runs: i32 = 0,
    base: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        if (self.d) |d| d.deinit();
        self.base.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const i = @backingInt(self.tile);
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = chain,
                .out = .f32,
                .replaces = "tf_dense_rows_1 launched a row at a time",
                .tiles = .{ .{ .size = hr.tile_rows[i] }, .{ .size = hr.warps * hr.tile_cols[i] }, .{ .size = 256, .step = 8, .min = 8 } },
                .served = &served[i],
            },
            .ctx = self,
            .vt = &vt,
            .control = self.ctl,
        };
    }

    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
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
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.dev[2] = try self.rig.output(d.outputs() * 4);
        return true;
    }

    pub fn run(self: *Case) !void {
        const sh = self.d.?.shape;
        self.runs += 1;
        try hr.launchTiled(self.fn_handle, self.tile, self.runs, self.rig.s, self.dev[0], sh.k, self.dev[1], self.dev[2], sh.n, sh.rows, sh.n, sh.k);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Case) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }

    pub fn replaced(self: *Case) ![]const u8 {
        const sh = self.d.?.shape;
        const n = self.count();
        var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, n * 4);
        defer tmp.free();
        for (0..sh.rows) |r| try self.dense.rows(self.rig.s, self.dev[0] + r * sh.k * 2, sh.k, self.dev[1], tmp.ptr + r * sh.n * 4, sh.n, 1, sh.n, sh.k);
        try self.rig.s.synchronize();
        try self.base.resize(self.rig.gpu.gpa, n * 4);
        try tmp.download(0, self.base.items);
        return self.base.items;
    }
};
