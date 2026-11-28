//! Recording-only Metal boundary for the real DFlash2 matcher, with no device or shader compiler.
const std = @import("std");
pub const Pipeline = u32;
pub const Buffer = struct {
    id: usize,
    storage: []align(8) u8,
    pub fn length(b: @This()) usize {
        return b.storage.len;
    }
    pub fn slice(b: @This(), comptime T: type, count: usize) []T {
        return @ptrCast(@alignCast(b.storage[0 .. count * @sizeOf(T)]));
    }
};
pub const Size = struct {
    width: usize = 1,
    height: usize = 1,
    depth: usize = 1,
    pub fn of(x: usize, y: usize, z: usize) @This() {
        return .{ .width = x, .height = y, .depth = z };
    }
};
pub const ResourceOptions = struct {
    pub const shared: u8 = 1;
    pub const untracked: u8 = 2;
};
pub const Pass = struct { pipe: Pipeline = 0, grid: Size = .{}, threads: Size = .{} };
pub const Record = struct { current: Pass = .{}, passes: [8]Pass = undefined, n: usize = 0 };
pub const ComputeEncoder = struct {
    r: *Record,
    pub fn setPipeline(e: @This(), p: Pipeline) void {
        e.r.current.pipe = p;
    }
    pub fn setBuffer(_: @This(), _: Buffer, _: usize, _: usize) void {}
    pub fn setValue(_: @This(), _: anytype, _: usize) void {}
    pub fn dispatchGroups(e: @This(), grid: Size, threads: Size) void {
        e.r.current.grid = grid;
        e.r.current.threads = threads;
        e.r.passes[e.r.n] = e.r.current;
        e.r.n += 1;
    }
    pub fn barrier(_: @This()) void {}
};
