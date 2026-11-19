//! Recording-only Metal boundary: no Objective-C, device discovery or shader compilation.
const std = @import("std");
pub var builds: usize = 0;
pub var scalar_limit: usize = 512;
pub var tile_limit: usize = 512;
pub const Pipeline = struct {
    id: u64 = 1,
    scalar: bool = false,
    pub fn init(_: Device, _: Library, name: []const u8, _: bool) !Pipeline {
        builds += 1;
        return .{ .id = std.hash.Wyhash.hash(0, name), .scalar = std.mem.startsWith(u8, name, "tf_sqw_scalar_") };
    }
    pub fn maxThreads(p: Pipeline) usize {
        return if (p.scalar) scalar_limit else tile_limit;
    }
    pub fn deinit(_: Pipeline) void {}
};
pub const Library = struct {
    pub fn fromSource(_: Device, _: []const u8, _: CompileOptions) !Library {
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const CompileOptions = struct {
    pub fn mlx() @This() {
        return .{};
    }
};
pub const objc = struct {
    pub const Pool = struct {
        pub fn push() @This() {
            return .{};
        }
        pub fn pop(_: @This()) void {}
    };
};
pub const Queue = struct {};
pub const ResourceOptions = struct {
    pub const shared = 1;
    pub const untracked = 2;
};
pub const ResidencySet = struct {
    pub const Follow = struct {
        ctx: *anyopaque,
        join: *const fn (*anyopaque, Buffer) void,
        leave: *const fn (*anyopaque, Buffer) void,
    };
};
pub const Device = struct {
    tensor: bool = true,
    pub fn generation(_: @This()) u32 {
        return 13;
    }
    pub fn buffer(_: @This(), _: usize, _: usize) !Buffer {
        return error.HostMustNotAllocateMetal;
    }
    pub fn tensorUnits(d: @This()) bool {
        return d.tensor;
    }
};
pub const Buffer = struct {
    id: usize,
    bytes: usize = 1 << 30,
    pub fn deinit(_: @This()) void {}
    pub fn contents(_: @This()) [*]u8 {
        unreachable;
    }
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
    pipe: Pipeline = .{},
    buffers: [20]Bound = @splat(.{}),
    values: [20][64]u8 = @splat(@splat(0)),
    lengths: [20]usize = @splat(0),
    grid: Size = .{},
    threads: Size = .{},
    direct: bool = false,
};
pub const Record = struct { current: Pass = .{}, passes: [64]Pass = undefined, n: usize = 0 };
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
    pub fn setBytes(e: @This(), v: []const u8, i: usize) void {
        e.r.current.lengths[i] = v.len;
        @memcpy(e.r.current.values[i][0..v.len], v);
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
