//! Recording-only Metal boundary: no Objective-C, device discovery or shader compilation.
const std = @import("std");
pub const Pipeline = u32;
pub const Device = struct {
    tensor: bool = true,
    pub fn tensorUnits(d: @This()) bool {
        return d.tensor;
    }
};
pub const Buffer = struct {
    id: usize,
    bytes: usize = 1 << 30,
    pub fn length(b: @This()) usize {
        return b.bytes;
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
pub const Bound = struct { id: usize = 0, off: usize = 0 };
pub const Pass = struct {
    pipe: u32 = 0,
    buffers: [16]Bound = @splat(.{}),
    values: [16][64]u8 = @splat(@splat(0)),
    lengths: [16]usize = @splat(0),
    grid: Size = .{},
    threads: Size = .{},
    direct: bool = false,
};
pub const Record = struct { current: Pass = .{}, passes: [16]Pass = undefined, n: usize = 0 };
pub const ComputeEncoder = struct {
    r: *Record,
    pub fn setPipeline(e: @This(), p: Pipeline) void {
        e.r.current.pipe = p;
    }
    pub fn setBuffer(e: @This(), b: Buffer, off: usize, i: usize) void {
        e.r.current.buffers[i] = .{ .id = b.id, .off = off };
    }
    pub fn setValue(e: @This(), v: anytype, i: usize) void {
        const bytes = std.mem.asBytes(&v);
        e.r.current.lengths[i] = bytes.len;
        @memcpy(e.r.current.values[i][0..bytes.len], bytes);
    }
    fn dispatch(e: @This(), g: Size, t: Size, direct: bool) void {
        e.r.current.grid = g;
        e.r.current.threads = t;
        e.r.current.direct = direct;
        e.r.passes[e.r.n] = e.r.current;
        e.r.n += 1;
    }
    pub fn dispatchGroups(e: @This(), g: Size, t: Size) void {
        e.dispatch(g, t, false);
    }
    pub fn dispatchThreads(e: @This(), g: Size, t: Size) void {
        e.dispatch(g, t, true);
    }
    pub fn barrier(_: @This()) void {}
};
