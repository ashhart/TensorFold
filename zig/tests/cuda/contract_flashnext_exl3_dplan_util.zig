//! Shared pieces of the Flash Next EXL3 device-plan contract runners.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;

pub const experts = 512;
pub const recent_rows_words = 256 * (experts / 32);
pub const trace_words = 1024 * 1024 / 4;

pub fn stateWords(lines: usize) usize {
    return 2 * lines + 3 * experts + recent_rows_words + 10;
}

pub fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn put(rig: *rig_.Rig, ptr: u64, bytes: []const u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = rig.gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
}

pub fn fetch(rig: *rig_.Rig, comptime T: type, ptr: u64, n: usize) ![]T {
    const out = try rig.gpu.gpa.alloc(T, n);
    errdefer rig.gpu.gpa.free(out);
    try rig.s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = rig.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
    try b.download(0, std.mem.sliceAsBytes(out));
    return out;
}

/// `got` (a device buffer's `n` elements) equals `want`, or what first differs.
pub fn same(rig: *rig_.Rig, comptime T: type, ptr: u64, want: []const T, what: []const u8, msg: []u8) !?[]const u8 {
    const got = try fetch(rig, T, ptr, want.len);
    defer rig.gpu.gpa.free(got);
    for (got, want, 0..) |x, y, i| if (x != y) return try std.fmt.bufPrint(msg, "{s}[{d}] is {d}, not {d}", .{ what, i, x, y });
    return null;
}
