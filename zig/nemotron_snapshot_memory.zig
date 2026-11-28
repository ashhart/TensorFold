//! Host storage for the real Nemotron cache and snapshot copier; no Metal driver or shader arithmetic.
const std = @import("std");
const Self = @This();
pub const ResourceOptions = struct {
    pub const shared = 1;
    pub const untracked = 2;
};
pub const Buffer = struct {
    data: []align(16) u8,
    pub fn contents(b: Buffer) [*]u8 {
        return b.data.ptr;
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(b.data.ptr)))[0..n];
    }
    pub fn deinit(b: Buffer) void {
        std.testing.allocator.free(b.data);
    }
};
pub const Device = struct {
    pub fn buffer(_: Device, n: usize, _: usize) !Buffer {
        const data = try std.testing.allocator.alignedAlloc(u8, .@"16", n);
        @memset(data, 0);
        return .{ .data = data };
    }
};
pub const ResidencySet = struct {
    pub const Follow = struct {
        ctx: *anyopaque,
        join: *const fn (*anyopaque, Buffer) void,
        leave: *const fn (*anyopaque, Buffer) void,
    };
};
pub const resident = struct {
    pub const fields = @import("src/metal/resident_fields.zig").Fields(Self);
};
