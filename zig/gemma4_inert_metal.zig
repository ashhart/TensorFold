//! Inert Metal allocations and release identities; no device, Objective-C calls or shader execution.
const std = @import("std");
pub const ResourceOptions = @import("src/metal/types.zig").ResourceOptions;
pub const Size = @import("src/metal/types.zig").Size;
var data: [16][4194304]u8 align(256) = undefined;
pub var made: usize = 0;
pub var releases: [16]usize = @splat(0);
pub var libraries: usize = 0;
pub var pipes: usize = 0;
pub var queues: usize = 0;
pub fn reset() void {
    made = 0;
    releases = @splat(0);
    libraries = 0;
    pipes = 0;
    queues = 0;
}
pub const Buffer = struct {
    id: usize,
    bytes: usize,
    pub fn contents(b: @This()) [*]u8 {
        return &data[b.id];
    }
    pub fn length(b: @This()) usize {
        return b.bytes;
    }
    pub fn slice(b: @This(), comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(b.contents())))[0..n];
    }
    pub fn deinit(b: @This()) void {
        releases[b.id] += 1;
    }
};
pub const Device = struct {
    pub fn buffer(_: @This(), n: usize, _: usize) !Buffer {
        std.debug.assert(made < data.len and n <= data[0].len);
        const i = made;
        made += 1;
        return .{ .id = i, .bytes = n };
    }
    pub fn tensorUnits(_: @This()) bool {
        return true;
    }
    pub fn queue(_: @This()) !Queue {
        queues += 1;
        return .{};
    }
};
pub const CompileOptions = struct {
    pub fn mlx() @This() {
        return .{};
    }
};
pub const Library = struct {
    pub fn fromSource(_: Device, _: []const u8, _: CompileOptions) !@This() {
        libraries += 1;
        return .{};
    }
    pub fn deinit(_: @This()) void {
        libraries -= 1;
    }
};
pub const Pipeline = struct {
    pub fn init(_: Device, _: Library, _: []const u8, _: bool) !@This() {
        pipes += 1;
        return .{};
    }
    pub fn deinit(_: @This()) void {
        pipes -= 1;
    }
    pub fn maxThreads(_: @This()) usize {
        return 1024;
    }
    pub fn simdWidth(_: @This()) usize {
        return 32;
    }
};
pub const Queue = struct {
    pub fn commandBuffer(_: @This()) CommandBuffer {
        return .{};
    }
    pub fn deinit(_: @This()) void {
        queues -= 1;
    }
};
pub const CommandBuffer = struct {
    pub fn compute(_: @This(), _: enum { serial, concurrent }) ComputeEncoder {
        return .{};
    }
    pub fn commit(_: @This()) void {}
    pub fn wait(_: @This()) void {}
    pub fn failure(_: @This()) ?[]const u8 {
        return null;
    }
};
pub const ComputeEncoder = struct {
    pub fn barrier(_: @This()) void {}
    pub fn setPipeline(_: @This(), _: Pipeline) void {}
    pub fn setBuffer(_: @This(), _: Buffer, _: usize, _: usize) void {}
    pub fn setValue(_: @This(), _: anytype, _: usize) void {}
    pub fn dispatchGroups(_: @This(), _: Size, _: Size) void {}
    pub fn dispatchThreads(_: @This(), _: Size, _: Size) void {}
    pub fn end(_: @This()) void {}
};
pub const objc = struct {
    pub const Pool = struct {
        pub fn push() @This() {
            return .{};
        }
        pub fn pop(_: @This()) void {}
    };
};
