//! The CUDA device a serve opens: its chip class and its name, and the variables that pick it.
const std = @import("std");
const cuda = @import("cuda");
const Allocator = std.mem.Allocator;

/// The chip class gate entries name ("nvidia-sm121" for a GB10) of the device TF_CUDA_DEVICE picks; null without one.
pub fn chip(a: Allocator) ?[]const u8 {
    const ordinal = @import("sys").environment.device(null, getenv("TF_CUDA_DEVICE")) catch return null;
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    if (ordinal >= driver.deviceCount() catch return null) return null;
    return chipClass(a, driver.capability(@intCast(ordinal)) catch return null);
}

/// The name of the GPU a serve opens ("NVIDIA GB10"): --device, else TF_CUDA_DEVICE, else 0; null without one.
pub fn deviceName(a: Allocator, flag: ?u32) ?[]const u8 {
    const ordinal = @import("sys").environment.device(flag, getenv("TF_CUDA_DEVICE")) catch return null;
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    var buf: [256]u8 = undefined;
    const text = driver.deviceName(std.math.cast(c_int, ordinal) orelse return null, &buf) catch return null;
    return a.dupe(u8, text) catch null;
}

/// The SMs this build's kernels are for: none without kernels, the recorded list or -Dgpu's SM, else null (unknown).
pub const kernel_sms: ?[]const u32 = if (!cuda.kernels.available)
    &.{}
else if (@hasDecl(cuda.kernels, "target_sms") and cuda.kernels.target_sms.len > 0)
    cuda.kernels.target_sms
else if (cuda.kernels.built_sm) |sm|
    &[_]u32{sm}
else
    null;

/// One GPU as the driver lists it, for `tensorfold device`.
pub const Listed = struct { name: []const u8, sm: u32, memory: u64, integrated: bool };

/// Every GPU the driver lists, read without making a context; null when the driver does not load or answer.
pub fn list(a: Allocator) ?[]const Listed {
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    const n = driver.deviceCount() catch return null;
    const out = a.alloc(Listed, @intCast(@max(n, 0))) catch return null;
    for (out, 0..) |*g, i| {
        const ordinal: c_int = @intCast(i);
        var dev: cuda.abi.Device = 0;
        if (driver.api.cuDeviceGet(&dev, ordinal) != cuda.abi.success) return null;
        var integrated: c_int = 0;
        if (driver.api.cuDeviceGetAttribute(&integrated, .integrated, dev) != cuda.abi.success) integrated = 0;
        var total: usize = 0;
        if (driver.api.cuDeviceTotalMem_v2(&total, dev) != cuda.abi.success) total = 0;
        var buf: [256]u8 = undefined;
        const name = driver.deviceName(ordinal, &buf) catch "NVIDIA GPU";
        g.* = .{ .name = a.dupe(u8, name) catch return null, .sm = driver.capability(ordinal) catch 0, .memory = total, .integrated = integrated != 0 };
    }
    return out;
}

fn chipClass(a: Allocator, capability: u32) ?[]const u8 {
    return std.fmt.allocPrint(a, "nvidia-sm{d}", .{capability}) catch null;
}

pub fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The same strict setting contract as the direct CLI, with flags taking precedence over environment text.
pub fn setting(a: Allocator, flag: ?u32, comptime name: [:0]const u8, default: u32, problem: *[]const u8) !?u32 {
    const settings = @import("sys").environment;
    comptime settings.registered(name);
    if (flag) |v| return v;
    const raw = getenv(name);
    const n = settings.parse(name, raw) catch {
        problem.* = try settings.describe(a, name, raw.?);
        return null;
    };
    return @intCast(n orelse default);
}

test "chip classes name the compute capability as gate entries do" {
    const a = std.testing.allocator;
    const name = chipClass(a, 121).?;
    defer a.free(name);
    try std.testing.expectEqualStrings("nvidia-sm121", name);
}

test "variables: device settings share raw parsing without opening the driver" {
    const settings = @import("sys").environment;
    try std.testing.expectEqual(@as(?u64, null), try settings.parse("TF_CUDA_DEVICE", null));
    try std.testing.expectEqual(@as(?u64, null), try settings.parse("TF_CUDA_DEVICE", ""));
    try std.testing.expectEqual(@as(?u64, 3), try settings.parse("TF_CUDA_DEVICE", " 3"));
    try std.testing.expectError(error.InvalidEnvironment, settings.parse("TF_CUDA_DEVICE", "one"));
    try std.testing.expectError(error.InvalidEnvironment, settings.parse("TF_CUDA_DEVICE", "-1"));
}
