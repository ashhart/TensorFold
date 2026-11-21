//! GLM-5.3-Flash's router partials under the contract: tf_glm_router_tile, the shared mma tiles in 8 K slices.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const glue = glm.glue;

/// One fp32 MMA chain over a K slice from zero, as contract_route_rows.zig counts a whole row.
fn chain(s: c.Shape) usize {
    return s.k / glue.router_ks / 16 + 16;
}

/// The MoE router (288 experts) and the indexer gate (128) over the 4,096-wide hidden rows.
pub fn spec() c.Spec {
    return .{ .name = "cuda glm router partials tf_glm_router_tile", .accum = .f32, .depth = chain, .out = .f32, .replaces = "tf_glm_router_part", .tiles = .{ .{ .size = 64 }, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 4096, .step = 512, .min = 512 } }, .served = &.{ .{ .rows = 300, .n = 288, .k = 4096 }, .{ .rows = 64, .n = 128, .k = 4096 }, .{ .rows = 1, .n = 288, .k = 4096 } } };
}

/// x [rows, D] bf16 times the router's W [E, D] bf16 in 8 K slices: fp32 partials [slice][rows][E].
pub const Tile = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    old: cuda.Function, // tf_ref_glm_router_part: the removed kernel, the bytes every partial must equal
    ctl: ?cuda.Function = null, // tf_ctl_glm_router_tile_slice, which leaves the last slice unwritten and must fail gate 3
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Tile) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Tile) g.Kernel {
        var k: g.Kernel = .{ .spec = spec(), .ctx = self, .vt = &vt, .control = if (self.ctl != null) .shapes else null };
        if (self.ctl != null) k.spec.name = "cuda glm router partials control: tiles that leave the last K slice unwritten";
        return k;
    }

    const vt = g.hooks(Tile);

    pub fn prepare(self: *Tile, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % (glue.router_ks * 64) != 0 or shape.n == 0 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, glue.router_ks);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x61);
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

    pub fn run(self: *Tile) !void {
        try self.launch(self.ctl orelse self.f, 64, glue.router_tile_smem, self.out);
        try self.rig.s.synchronize();
    }

    /// One launch over the prepared rows: `rows_a_block` 64 (the tiles, the control) or 16 (the removed kernel).
    fn launch(self: *Tile, f: cuda.Function, rows_a_block: usize, shared: u32, out: u64) !void {
        const d = self.d.?;
        var a: cuda.Args = .{};
        inline for (.{ self.x, self.w, out }) |p| a.add(p);
        inline for (.{ d.shape.rows, d.shape.k, d.shape.k, d.shape.n }) |v| a.add(@as(i64, @intCast(v)));
        a.add(@as(c_int, glue.router_ks));
        const grid: cuda.Dim3 = .{ .x = @intCast((d.shape.rows + rows_a_block - 1) / rows_a_block), .y = @intCast((d.shape.n + 31) / 32), .z = glue.router_ks };
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 128 }, .shared = shared }, self.rig.s, &a);
    }

    pub fn replaced(self: *Tile) ![]const u8 {
        try self.launch(self.old, 16, 0, self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Tile) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Tile, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Tile, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

/// The tiles against the removed kernel, and the planted control, from the glm_glue and controls images.
pub fn pair(rig: rig_.Rig, glm_glue: cuda.Module, controls: cuda.Module) ![2]Tile {
    const f = try glm_glue.function("tf_glm_router_tile");
    const old = try controls.function("tf_ref_glm_router_part");
    return .{ .{ .rig = rig, .f = f, .old = old }, .{ .rig = rig, .f = f, .old = old, .ctl = try controls.function("tf_ctl_glm_router_tile_slice") } };
}
