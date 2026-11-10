//! Prepared affine codes retain the original quantizer and replace the original FP8 products byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Base = @import("contract_fp8_prompt.zig");
const Rig = @import("contract_rig.zig").Rig;
const Gpu = @import("check.zig").Gpu;
const packing = cuda.fp8_prefill.affine_packing;
pub const entries = [_]Base.Config{
    .{ .kind = .affine4, .out = .bf16 },
    .{ .kind = .affine4, .out = .f32 },
    .{ .kind = .affine4, .out = .f32, .fault = .missing_tail },
    .{ .kind = .affine4, .out = .f32, .fault = .narrow_sum },
};

pub fn spec(config: Base.Config) c.Spec {
    var result = Base.spec(config);
    result.name = switch (config.fault) {
        .none => if (config.out == .f32) "cuda prepared affine FP8 f32" else "cuda prepared affine FP8 bf16",
        .missing_tail => "cuda prepared affine FP8 CONTROL omitted final row",
        .narrow_sum => "cuda prepared affine FP8 CONTROL bf16 accumulator replay",
        else => unreachable,
    };
    result.replaces = Base.spec(.{ .kind = .affine4, .out = config.out }).name;
    return result;
}

pub const Case = struct {
    base: Base.Case = undefined,
    rig: *Rig = undefined,
    prepared: ?packing.Prepared = null,
    baseline: u64 = 0,

    pub fn init(self: *Case, gpu: Gpu, stream: cuda.Stream, config: Base.Config) void {
        self.* = .{ .base = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = config } };
        self.rig = &self.base.rig;
    }
    fn clear(self: *Case) void {
        if (self.prepared) |*p| p.unload();
        self.prepared = null;
        const old = self.base.kernel();
        old.vt.release.?(old.ctx);
    }
    pub fn deinit(self: *Case) void {
        self.clear();
        self.base.deinit();
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.clear();
        if (!try self.base.prepare(shape, fill, seed)) return false;
        self.prepared = try packing.Prepared.load(self.rig.gpu.d);
        const bytes = packing.required(self.base.affine) orelse return error.InvalidFp8PromptShape;
        self.prepared.?.buffer = .{ .d = self.rig.gpu.d, .ptr = try self.rig.scratch(bytes), .len = bytes, .borrowed = true };
        self.baseline = try self.rig.scratch(self.count() * c.size(spec(self.base.config).out));
        return true;
    }
    pub fn run(self: *Case) !void {
        const fault = self.base.config.fault;
        self.base.config.fault = .none;
        defer self.base.config.fault = fault;
        try self.base.run();
        const output: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.base.out, .len = self.count() * c.size(spec(self.base.config).out), .borrowed = true };
        try output.fill8(0xff, self.rig.s.handle);
        const rows = self.base.shape.rows - @as(usize, @intFromBool(fault == .missing_tail and self.base.shape.rows > 1));
        const codes = self.base.native.?.scratch.ptr;
        const scales = codes + self.base.shape.rows * self.base.shape.k;
        const sums = scales + self.base.shape.rows * (self.base.shape.k / 64) * 4;
        // Lower row counts test kernel edges; production selection independently retains its measured row gate.
        if (!try self.prepared.?.matmul(self.rig.s, codes, scales, sums, rows, self.base.affine, self.base.out, self.base.config.out)) return error.AffinePreparedRefused;
        try self.rig.s.synchronize();
        if (fault == .narrow_sum) {
            self.base.config.fault = fault;
            try self.base.run();
        }
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
        var original = self.base.native.?;
        original.affine_prepared = null;
        try original.affine4(self.rig.s, self.base.dx, self.base.shape.rows, self.base.affine, self.baseline, self.base.config.out);
        if (!try self.rig.intact()) return error.AffinePreparedBaselineGuardWrite;
        return self.rig.bytes(self.baseline, self.count() * c.size(spec(self.base.config).out));
    }
    fn release(ctx: *anyopaque) void {
        const self: *Case = @ptrCast(@alignCast(ctx));
        self.clear();
    }
    const vt = blk: {
        var hooks = g.hooks(Case);
        hooks.release = release;
        break :blk hooks;
    };
    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.base.config), .ctx = self, .vt = &vt, .control = switch (self.base.config.fault) {
            .none => null,
            .missing_tail => .shapes,
            .narrow_sum => .sums,
            else => unreachable,
        } };
    }
};

pub fn run(gpu: Gpu) !void {
    const sm = try gpu.ctx.capability();
    if (sm != 120 and sm != 121) return error.UnsupportedNativeFp8Prefill;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: [entries.len]Case = undefined;
    var kernels: [entries.len]g.Kernel = undefined;
    for (&cases, &kernels, entries) |*item, *kernel, config| {
        item.init(gpu, stream, config);
        kernel.* = item.kernel();
    }
    defer for (&cases) |*item| item.deinit();
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, &kernels, 0xaf8001, &tally);
    std.debug.print("SUMMARY prepared affine FP8: {d} pass, {d} fail, {d} controls caught, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    if (tally.problems() != 0) return error.TestFailed;
}

test "FP8 prompt prepared affine contract exposes original-byte and accumulation checks" {
    std.mem.doNotOptimizeAway(&run);
    for (entries) |config| {
        try std.testing.expect(spec(config).replaces != null);
        try std.testing.expectEqual(c.Accum.f32, spec(config).accum);
    }
    try std.testing.expect(Case.vt.replaced != null and Case.vt.simulate != null);
}
