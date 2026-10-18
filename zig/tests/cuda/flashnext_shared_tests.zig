//! Independent stage-wise FP64 references for the separate MXFP8 shared expert and its reserved slot.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const d = 2560;
const w = 640;
const most = 17;
const slots = 11;

const Weight = struct {
    codes: []u8,
    scales: []u8,
    n: usize,
    k: usize,

    fn init(a: std.mem.Allocator, n: usize, k: usize, salt: usize) !Weight {
        const codes = try a.alloc(u8, n * k);
        const scales = try a.alloc(u8, n * k / 32);
        for (codes, 0..) |*v, i| v.* = @intCast(0x20 + ((i * 13 + salt * 7) % 32) + (if ((i + salt) % 3 == 0) @as(usize, 128) else 0));
        for (scales, 0..) |*v, i| v.* = @intCast(121 + (i + salt) % 3);
        return .{ .codes = codes, .scales = scales, .n = n, .k = k };
    }

    fn value(self: Weight, n: usize, k: usize) f64 {
        const exponent: i32 = @as(i32, self.scales[n * (self.k / 32) + k / 32]) - 127;
        return core.quant.fp8.e4m3(self.codes[n * self.k + k]) * std.math.pow(f64, 2, @floatFromInt(exponent));
    }

    fn upload(self: Weight, a: std.mem.Allocator, driver: *const cuda.Driver) !flash.projection.Projection {
        return flash.projection.mxfp8(a, driver, tensor(.f8_e4m3, self.codes, self.n, self.k), tensor(.u8, self.scales, self.n, self.k / 32));
    }
};

fn tensor(dtype: core.safetensors.DType, bytes: []const u8, n: usize, k: usize) core.safetensors.Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes };
}

fn projectionReference(r: *rig.Rig, name: []const u8, x: []const u16, weight: Weight, out: anytype, rows: usize, comptime bf16: bool) !void {
    for (0..rows) |row| for (0..weight.n) |n| {
        var total: f64 = 0;
        var magnitude: f64 = 0;
        for (0..weight.k) |k| {
            const term = rig.bf(x[row * weight.k + k]) * weight.value(n, k);
            total += term;
            magnitude += @abs(term);
        }
        const at = row * weight.n + n;
        if (bf16) {
            try r.within(name, rig.bf(out[at]), rig.rbf(total), rig.ulps(rig.rbf(total), 1));
        } else try r.within(name, out[at], total, rig.sumTol(magnitude, weight.k) + 1e-6);
    };
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(0x5a4ed);
    var r = try rig.Rig.init(gpu, a, rng.random());
    defer r.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    const gw = try Weight.init(a, w, d, 1);
    const uw = try Weight.init(a, w, d, 2);
    const dw = try Weight.init(a, d, w, 3);
    var gate = try gw.upload(a, gpu.d);
    defer gate.free();
    var up = try uw.upload(a, gpu.d);
    defer up.free();
    var down = try dw.upload(a, gpu.d);
    defer down.free();
    const x = r.bfs(most * d, 0.125);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const g = try r.out(most * w * 2);
    const u = try r.out(most * w * 2);
    const act = try r.out(most * w * 2);
    const out = try r.out(most * d * 4);
    const temp = try r.out(8 * most * d * 4);
    const sentinel = try a.alloc(f32, most * slots * d);
    @memset(sentinel, 0.375);
    const y = try r.dev(std.mem.sliceAsBytes(sentinel));
    var all_g: []u16 = undefined;
    var all_u: []u16 = undefined;
    var all_act: []u16 = undefined;
    var all_out: []f32 = undefined;
    for ([_]usize{ most, 1, 7 }) |rows| {
        try lane.matmul(r.s, dx, d, rows, gate.native.?, g, temp);
        try lane.matmul(r.s, dx, d, rows, up.native.?, u, temp);
        try glue.sharedSwiGLU(r.s, g, u, act, rows, w);
        try lane.matmulF32(r.s, act, w, rows, down.native.?, out, temp);
        try glue.sharedScatter(r.s, out, y, rows, d, slots);
        if (rows == most) {
            all_g = try r.back(u16, g, most * w);
            all_u = try r.back(u16, u, most * w);
            all_act = try r.back(u16, act, most * w);
            all_out = try r.back(f32, out, most * d);
        } else {
            try r.sameRows("shared gate rows", u16, g, all_g[0 .. rows * w]);
            try r.sameRows("shared up rows", u16, u, all_u[0 .. rows * w]);
            try r.sameRows("shared activation rows", u16, act, all_act[0 .. rows * w]);
            try r.sameRows("shared FP32 output rows", f32, out, all_out[0 .. rows * d]);
        }
    }
    try projectionReference(&r, "shared MXFP8 gate", x, gw, all_g, most, true);
    try projectionReference(&r, "shared MXFP8 up", x, uw, all_u, most, true);
    for (all_g, all_u, all_act) |gate_value, up_value, actual| {
        const f = rig.bf(gate_value);
        const want = rig.rbf(rig.rbf(f / (1 + @exp(-f))) * rig.bf(up_value));
        try r.within("shared SwiGLU rounding", rig.bf(actual), want, rig.ulps(want, 1));
    }
    try projectionReference(&r, "shared MXFP8 FP32 down", all_act, dw, all_out, most, false);
    const scattered = try r.back(f32, y, most * slots * d);
    for (scattered, 0..) |actual, i| {
        const slot = (i / d) % slots;
        const want = if (slot == slots - 1) all_out[(i / (slots * d)) * d + i % d] else @as(f32, 0.375);
        try check.expect(actual == want, "shared scatter preserves routed slots and copies exact FP32 values", .{});
    }
    try std.testing.expectError(error.RowShape, glue.sharedScatter(r.s, out, y, most, d, 0));
    r.report("MXFP8 shared expert, BF16 projection/activation seams, FP32 down, reserved-slot scatter; rows 1/7/17 equal");
}
