//! Optional FP8 tiles replace the original products byte for byte through the real Native wrapper.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Base = @import("contract_fp8_prompt.zig");
const Rig = @import("contract_rig.zig").Rig;
const Gpu = @import("check.zig").Gpu;
const tile = cuda.fp8_prefill.tile_geometry;

pub const Config = struct { mode: tile.Mode, base: Base.Config, label: []const u8 };
pub const entries = blk: {
    var configs: [18]Config = undefined;
    var i = 0;
    for ([_]tile.Mode{ .tile128x64, .tile128x128, .tile64x64 }) |mode| {
        const prefix = "cuda fp8 " ++ @tagName(mode);
        for ([_]Base.Kind{ .calibrated, .row }) |kind| for ([_]cuda.qmmf.Out{ .bf16, .f32 }) |out| {
            configs[i] = .{ .mode = mode, .base = .{ .kind = kind, .out = out }, .label = prefix ++ " " ++ @tagName(kind) ++ " " ++ @tagName(out) };
            i += 1;
        };
        configs[i] = .{ .mode = mode, .base = .{ .kind = .row, .out = .f32, .fault = .missing_tail }, .label = prefix ++ " CONTROL omitted final row" };
        i += 1;
        configs[i] = .{ .mode = mode, .base = .{ .kind = .row, .out = .f32, .fault = .narrow_sum }, .label = prefix ++ " CONTROL bf16 accumulator replay" };
        i += 1;
    }
    break :blk configs;
};

pub fn spec(config: Config) c.Spec {
    var result = Base.spec(config.base);
    result.name = config.label;
    result.replaces = Base.spec(.{ .kind = config.base.kind, .out = config.base.out }).name;
    const geometry = tile.geometry(config.mode);
    result.tiles[0].size = geometry.rows;
    result.tiles[1].size = geometry.columns;
    return result;
}

pub const Case = struct {
    base: Base.Case = undefined,
    rig: *Rig = undefined,
    config: Config = undefined,
    baseline: u64 = 0,

    pub fn init(self: *Case, gpu: Gpu, stream: cuda.Stream, config: Config) void {
        self.* = .{ .base = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = config.base }, .config = config };
        self.rig = &self.base.rig;
    }
    pub fn deinit(self: *Case) void {
        self.base.deinit();
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.base.prepare(shape, fill, seed)) return false;
        const native = &self.base.native.?;
        if (native.tiles) |*old| old.unload();
        native.tiles = null;
        native.tiles = try tile.Tiles.load(self.rig.gpu.d, self.config.mode);
        self.baseline = try self.rig.scratch(self.count() * c.size(spec(self.config).out));
        return true;
    }
    pub fn run(self: *Case) !void {
        try self.base.run();
    }
    pub fn count(self: *const Case) usize {
        return self.base.count();
    }
    pub fn reference(self: *Case, want: []f64, magnitude: []f64) !void {
        try self.base.reference(want, magnitude);
    }
    pub fn simulate(self: *Case, accum: c.Accum, order: g.Order, out: []f64) !void {
        try self.base.simulate(accum, order, out);
    }
    pub fn replaced(self: *Case) ![]const u8 {
        // Borrow all operands and scratch, but run the original products into a different guarded buffer.
        var original = self.base.native.?;
        original.tiles = null;
        const b = &self.base;
        switch (self.config.base.kind) {
            .calibrated => try original.matmul(self.rig.s, b.dx, b.shape.k + 3, b.shape.rows, b.weight, self.baseline, self.config.base.out),
            .row => try original.rowScaled(self.rig.s, b.dx, b.shape.rows, b.weight, self.baseline, self.config.base.out),
            .affine4 => return error.InvalidFp8TileContract,
        }
        if (!try self.rig.intact()) return error.Fp8TileBaselineGuardWrite;
        return self.rig.bytes(self.baseline, self.count() * c.size(spec(self.config).out));
    }
    fn release(ctx: *anyopaque) void {
        const self: *Case = @ptrCast(@alignCast(ctx));
        const original = self.base.kernel();
        original.vt.release.?(original.ctx);
    }
    const vt = blk: {
        var hooks = g.hooks(Case);
        hooks.release = release;
        break :blk hooks;
    };
    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.config), .ctx = self, .vt = &vt, .control = switch (self.config.base.fault) {
            .none => null,
            .missing_tail => .shapes,
            .narrow_sum => .sums,
            else => unreachable,
        } };
    }
};

pub const Cases = struct {
    items: [entries.len]Case = undefined,
    initialized: usize = 0,
    pub fn append(self: *Cases, list: *std.ArrayList(g.Kernel), gpu: Gpu, stream: cuda.Stream, sm: u32) !void {
        if (sm != 120 and sm != 121) return;
        for (&self.items, entries) |*item, config| {
            item.init(gpu, stream, config);
            self.initialized += 1;
            try list.append(gpu.gpa, item.kernel());
        }
    }
    pub fn deinit(self: *Cases) void {
        for (self.items[0..self.initialized]) |*item| item.deinit();
    }
};

pub fn run(gpu: Gpu) !void {
    const sm = try gpu.ctx.capability();
    if (sm != 120 and sm != 121) return error.UnsupportedNativeFp8Prefill;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: Cases = .{};
    defer cases.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try cases.append(&kernels, gpu, stream, sm);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0xf8a8128, &tally);
    std.debug.print("SUMMARY optional FP8 tiles: {d} kernels pass, {d} fail, {d} controls caught, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    if (tally.problems() != 0) return error.TestFailed;
}

test "FP8 prompt tile contracts retain original replacement and independent accumulation hooks" {
    std.mem.doNotOptimizeAway(&run);
    for (entries) |config| {
        const s = spec(config);
        try std.testing.expect(s.replaces != null);
        try std.testing.expectEqual(c.Accum.f32, s.accum);
        try std.testing.expectEqual(@as(usize, tile.geometry(config.mode).rows), s.tiles[0].size);
        try std.testing.expectEqual(@as(usize, tile.geometry(config.mode).columns), s.tiles[1].size);
    }
    try std.testing.expect(Case.vt.replaced != null);
    try std.testing.expect(Case.vt.simulate != null);
}
