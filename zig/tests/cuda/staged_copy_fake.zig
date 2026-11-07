//! A successful fake integrated readback deliberately misses every pageable destination.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const stage = @import("cuda").memory;
const download_stage = @import("cuda").download_stage;
const t = std.testing;
var pinned: [4 << 20]u8 align(16) = undefined;
var source: [8192]u8 align(16) = undefined;
var output: [8192]u8 align(16) = undefined;
var integrated_flag: c_int = 0;
var allocations: usize = 0;
var copies: usize = 0;
var last_destination: usize = 0;
var last_source: u64 = 0;
var last_size: usize = 0;

pub const Fake = struct {
    d: cuda.Driver,
    src: u64,
    before: u64,

    pub fn init(integrated: bool) Fake {
        download_stage.drop();
        download_stage.forget();
        integrated_flag = @intFromBool(integrated);
        allocations = 0;
        copies = 0;
        for (&source, 0..) |*b, i| b.* = @truncate(i *% 37 +% (i / 251));
        @memset(&output, 0xaa);
        var d: cuda.Driver = undefined;
        d.api.cuCtxGetDevice = device;
        d.api.cuDeviceGetAttribute = attribute;
        d.api.cuMemHostAlloc = alloc;
        d.api.cuMemFreeHost = free;
        d.api.cuMemcpyDtoH_v2 = copy;
        return .{ .d = d, .src = @intFromPtr(&source[32]), .before = stage.usage(false).host };
    }

    pub fn deinit(self: *Fake) void {
        download_stage.drop();
        download_stage.forget();
        std.debug.assert(stage.usage(false).host == self.before);
    }

    pub fn dest(_: Fake, comptime T: type, n: usize) []T {
        std.debug.assert(n * @sizeOf(T) <= output.len - 32);
        return @as([*]T, @ptrCast(@alignCast(&output[16])))[0..n];
    }

    pub fn check(_: Fake, src: u64, out: []u8) !void {
        try t.expectEqualSlices(u8, @as([*]const u8, @ptrFromInt(src))[0..out.len], out);
        try t.expectEqual(@as(usize, 1), copies);
        try t.expectEqual(src, last_source);
        try t.expectEqual(out.len, last_size);
        if (integrated_flag != 0) {
            try t.expectEqual(@as(usize, 1), allocations);
            try t.expectEqual(@intFromPtr(&pinned), last_destination);
            try t.expect(last_destination != @intFromPtr(out.ptr));
        } else {
            try t.expectEqual(@as(usize, 0), allocations);
            try t.expectEqual(@intFromPtr(out.ptr), last_destination);
        }
        const start = @intFromPtr(out.ptr);
        const base = @intFromPtr(&output);
        if (start >= base and start + out.len <= base + output.len) {
            const lo = start - base;
            try t.expect(std.mem.allEqual(u8, output[0..lo], 0xaa));
            try t.expect(std.mem.allEqual(u8, output[lo + out.len ..], 0xaa));
        } else try t.expect(std.mem.allEqual(u8, &output, 0xaa));
        for (source, 0..) |b, i| try t.expectEqual(@as(u8, @truncate(i *% 37 +% (i / 251))), b);
    }
};

fn device(out: *abi.Device) callconv(.c) abi.Result {
    out.* = 0;
    return 0;
}
fn attribute(out: *c_int, _: abi.DeviceAttribute, _: abi.Device) callconv(.c) abi.Result {
    out.* = integrated_flag;
    return 0;
}
fn alloc(out: ?*?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
    if (n > pinned.len) return abi.error_not_ready;
    allocations += 1;
    out.?.* = &pinned;
    return 0;
}
fn free(_: ?*anyopaque) callconv(.c) abi.Result {
    return 0;
}
fn copy(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) abi.Result {
    copies += 1;
    last_destination = @intFromPtr(dst.?);
    last_source = src;
    last_size = n;
    if (integrated_flag == 0 or dst == @as(*anyopaque, @ptrCast(&pinned))) {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
    }
    return 0;
}
