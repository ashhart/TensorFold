//! The CUDA device a serve opens: its chip class and its name, and the variables that pick it.
const std = @import("std");
const cuda = @import("cuda");
const Allocator = std.mem.Allocator;

/// The chip class gate entries name ("nvidia-sm121" for a GB10) of the device TF_CUDA_DEVICE picks; null without one.
pub fn chip(a: Allocator) ?[]const u8 {
    const ordinal = envNumber(getenv("TF_CUDA_DEVICE")) catch return null;
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    if (ordinal orelse 0 >= driver.deviceCount() catch return null) return null;
    return chipClass(a, driver.capability(@intCast(ordinal orelse 0)) catch return null);
}

/// The name of the GPU a serve opens ("NVIDIA GB10"): --device, else TF_CUDA_DEVICE, else 0; null without one.
pub fn deviceName(a: Allocator, flag: ?u32) ?[]const u8 {
    const ordinal = flag orelse (envNumber(getenv("TF_CUDA_DEVICE")) catch return null) orelse 0;
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    var buf: [256]u8 = undefined;
    const text = driver.deviceName(std.math.cast(c_int, ordinal) orelse return null, &buf) catch return null;
    return a.dupe(u8, text) catch null;
}

fn chipClass(a: Allocator, capability: u32) ?[]const u8 {
    return std.fmt.allocPrint(a, "nvidia-sm{d}", .{capability}) catch null;
}

pub fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A variable's whole number: null when unset or empty, an error for anything else that is not one.
fn envNumber(text: ?[]const u8) error{Invalid}!?u32 {
    const t = std.mem.trim(u8, text orelse return null, " ");
    if (t.len == 0) return null;
    return std.fmt.parseInt(u32, t, 10) catch error.Invalid;
}

/// The flag's value, else the variable's, else `default`; `problem` names a variable that is not a whole number.
pub fn setting(a: Allocator, flag: ?u32, name: [:0]const u8, default: u32, problem: *[]const u8) !?u32 {
    if (flag) |v| return v;
    const v = envNumber(getenv(name)) catch {
        problem.* = try std.fmt.allocPrint(a, "{s}={s}: a whole number", .{ name, getenv(name).? });
        return null;
    };
    return v orelse default;
}

test "chip classes name the compute capability as gate entries do" {
    const a = std.testing.allocator;
    const name = chipClass(a, 121).?;
    defer a.free(name);
    try std.testing.expectEqualStrings("nvidia-sm121", name);
}

test "variables: unset or empty means the default, anything but a whole number is refused" {
    try std.testing.expectEqual(@as(?u32, null), try envNumber(null));
    try std.testing.expectEqual(@as(?u32, null), try envNumber(""));
    try std.testing.expectEqual(@as(?u32, 3), try envNumber(" 3"));
    try std.testing.expectError(error.Invalid, envNumber("one"));
    try std.testing.expectError(error.Invalid, envNumber("-1"));
}
