//! Inert embedded assets exercise release admission and capabilities with recording driver callbacks only.
const std = @import("std");
const cuda = @import("cuda");
const caps = @import("src/server/cli_capabilities.zig");
const device = @import("src/native/cuda_device.zig");
const t = std.testing;
const logs = @import("qwen_weights_diagnostics");

const Fake = struct {
    var capability: u32 = 90;
    var retains: usize = 0;
    var loads: usize = 0;
    fn get(out: *cuda.abi.Device, _: c_int) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn attr(out: *c_int, field: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = @intCast(if (field == .compute_capability_major) capability / 10 else capability % 10);
        return 0;
    }
    fn name(out: [*]u8, _: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        const text = switch (capability) {
            90 => "NVIDIA H100",
            89 => "NVIDIA GeForce RTX 4090",
            87 => "NVIDIA Jetson AGX Orin",
            75 => "NVIDIA T4",
            100 => "NVIDIA B200",
            else => "NVIDIA GB10",
        };
        @memcpy(out[0..text.len], text);
        out[text.len] = 0;
        return 0;
    }
    fn retain(out: *cuda.abi.Context, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        retains += 1;
        out.* = @ptrFromInt(0x611);
        return 0;
    }
    fn set(_: cuda.abi.Context) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn release(_: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        retains -= 1;
        return 0;
    }
    fn sync() callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn load(_: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        loads += 1;
        return 209;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.func_name = null;
        d.api.cuDeviceGet = get;
        d.api.cuDeviceGetAttribute = attr;
        d.api.cuDeviceGetName = name;
        d.api.cuDevicePrimaryCtxRetain = retain;
        d.api.cuCtxSetCurrent = set;
        d.api.cuDevicePrimaryCtxRelease_v2 = release;
        d.api.cuCtxSynchronize = sync;
        d.api.cuModuleLoadData = load;
        return d;
    }
};

test "CUDA release SM: unsupported GPU is named before context retain or module load, supported GPU starts" {
    const prior = cuda.copy_source.current();
    defer cuda.copy_source.setCurrent(prior);
    var d = Fake.driver();
    Fake.capability = 90;
    Fake.retains = 0;
    Fake.loads = 0;
    logs.used = 0;
    try t.expectError(error.WrongGpu, cuda.Context.init(&d, 0));
    try t.expectEqual(@as(usize, 0), Fake.retains);
    try t.expectEqual(@as(usize, 0), Fake.loads);
    const text = logs.message[0..logs.used];
    try t.expect(std.mem.indexOf(u8, text, "NVIDIA H100") != null);
    try t.expect(std.mem.indexOf(u8, text, "9.0 (sm_90)") != null);
    var list: std.Io.Writer.Allocating = .init(t.allocator);
    defer list.deinit();
    for (cuda.kernels.target_sms, 0..) |sm, i| try list.writer.print("{s}{d}", .{ if (i == 0) "" else ",", sm });
    try t.expect(std.mem.indexOf(u8, text, list.written()) != null);
    Fake.capability = 121;
    logs.used = 0;
    var ctx = try cuda.Context.init(&d, 0);
    try t.expectEqual(@as(usize, 1), Fake.retains);
    try t.expectEqual(@as(usize, 0), Fake.loads);
    try t.expectEqual(@as(usize, 0), logs.used);
    ctx.deinit();
    try t.expectEqual(@as(usize, 0), Fake.retains);
}

test "CUDA release SM: sm_80 and sm_86 images serve later 8.x minors; another major or a later 12.x minor is refused" {
    const prior = cuda.copy_source.current();
    defer cuda.copy_source.setCurrent(prior);
    var d = Fake.driver();
    for ([_]u32{ 80, 86, 87, 89, 120 }) |sm| {
        Fake.capability = sm;
        Fake.retains = 0;
        Fake.loads = 0;
        logs.used = 0;
        var ctx = try cuda.Context.init(&d, 0);
        try t.expectEqual(@as(usize, 1), Fake.retains);
        try t.expectEqual(@as(usize, 0), logs.used);
        ctx.deinit();
        try t.expectEqual(@as(usize, 0), Fake.retains);
    }
    for ([_]u32{ 75, 90, 100, 122 }) |sm| {
        Fake.capability = sm;
        Fake.retains = 0;
        logs.used = 0;
        try t.expectError(error.WrongGpu, cuda.Context.init(&d, 0));
        try t.expectEqual(@as(usize, 0), Fake.retains);
        try t.expect(logs.used > 0);
        logs.used = 0; // the runner fails a test that leaves a diagnostic unread
    }
    try t.expect(cuda.kernels.covers(&.{ 80, 86 }, 89));
    try t.expect(!cuda.kernels.covers(&.{89}, 86));
    try t.expect(!cuda.kernels.covers(&.{ 80, 86 }, 90));
    // arch-specific images (sm_120a from sm_120 on) run on that SM alone
    try t.expect(!cuda.kernels.covers(&.{120}, 121));
    try t.expect(!cuda.kernels.covers(&.{121}, 122));
}

test "CUDA release SM: capabilities JSON reports build coverage for both detected capabilities" {
    var d = Fake.driver();
    for ([_]u32{ 90, 121, 89 }, [_]bool{ false, true, true }) |sm, expected| {
        Fake.capability = sm;
        const support = device.kernelSupportOn(&d, 0);
        var out: std.Io.Writer.Allocating = .init(t.allocator);
        defer out.deinit();
        try caps.capabilities(&out.writer, .{ .version = "test", .kernels = support.kernels, .kernel_sms = support.sms });
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, out.written(), .{});
        defer parsed.deinit();
        try t.expectEqual(expected, parsed.value.object.get("kernels").?.bool);
        const values = parsed.value.object.get("kernel_sms").?.array.items;
        try t.expectEqual(cuda.kernels.target_sms.len, values.len);
        for (values, cuda.kernels.target_sms) |value, target| try t.expectEqual(@as(i64, target), value.integer);
    }
}
