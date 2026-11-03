//! Record actual pocket bindings and ownership on host buffers; no shader arithmetic is executed.
const std = @import("std");
const real = @import("real_metal");
pub const ResourceOptions = real.ResourceOptions;
pub const Size = real.Size;
pub const MappedFile = real.MappedFile;
pub const Queue = real.Queue;
pub const objc = real.objc;
pub const Bound = struct { buffer: Buffer, offset: usize };
pub const Call = struct { input: usize, output: usize, weights: usize, rows: usize, shape: [4]u32 };
pub const Record = struct {
    a: std.mem.Allocator,
    allocated: usize = 0,
    live: usize = 0,
    libraries: usize = 0,
    pipelines: usize = 0,
    fail_pipeline: bool = false,
    bound: [3]Bound = undefined,
    shape: [4]u32 = undefined,
    calls: [256]Call = undefined,
    count: usize = 0,
    pub fn device(r: *Record) Device {
        return .{ .r = r };
    }
    pub fn encoder(r: *Record) ComputeEncoder {
        return .{ .r = r };
    }
};
pub const Buffer = struct {
    r: *Record,
    data: []align(256) u8,
    pub fn contents(b: Buffer) [*]u8 {
        return b.data.ptr;
    }
    pub fn length(b: Buffer) usize {
        return b.data.len;
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(b.contents())))[0..n];
    }
    pub fn deinit(b: Buffer) void {
        b.r.a.free(b.data);
        b.r.live -= 1;
    }
};
pub const Device = struct {
    r: *Record,
    pub fn tensorUnits(_: Device) bool {
        return false;
    }
    pub fn buffer(d: Device, bytes: usize, _: usize) !Buffer {
        const data = try d.r.a.alignedAlloc(u8, .fromByteUnits(256), bytes);
        @memset(data, 0);
        d.r.allocated += 1;
        d.r.live += 1;
        return .{ .r = d.r, .data = data };
    }
};
pub const CompileOptions = struct {
    pub fn mlx() @This() {
        return .{};
    }
};
pub const Library = struct {
    pub fn fromSource(d: Device, _: []const u8, _: CompileOptions) !Library {
        d.r.libraries += 1;
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    pub fn init(d: Device, _: Library, name: []const u8, _: bool) !Pipeline {
        if (d.r.fail_pipeline) return error.PlantedPipelineFailure;
        if (!std.mem.eql(u8, name, "tf_q27_pocket_rows")) return error.WrongPocketSymbol;
        d.r.pipelines += 1;
        return .{};
    }
    pub fn maxThreads(_: Pipeline) usize {
        return 256;
    }
    pub fn deinit(_: Pipeline) void {}
};
pub const ComputeEncoder = struct {
    r: *Record,
    pub fn setPipeline(_: ComputeEncoder, _: Pipeline) void {}
    pub fn setBuffer(e: ComputeEncoder, b: Buffer, offset: usize, at: usize) void {
        e.r.bound[at] = .{ .buffer = b, .offset = offset };
    }
    pub fn setValue(e: ComputeEncoder, value: anytype, at: usize) void {
        std.debug.assert(at == 3);
        e.r.shape = @bitCast(value);
    }
    pub fn dispatchGroups(e: ComputeEncoder, grid: Size, group: Size) void {
        std.debug.assert(group.width == 256 and e.r.count < e.r.calls.len);
        e.r.calls[e.r.count] = .{ .input = e.r.bound[0].offset, .output = e.r.bound[2].offset, .weights = @intFromPtr(e.r.bound[1].buffer.contents()), .rows = grid.width, .shape = e.r.shape };
        e.r.count += 1;
    }
    pub fn barrier(_: ComputeEncoder) void {}
};
pub const ResidencySet = struct {
    pub const Follow = struct { ctx: *anyopaque, join: *const fn (*anyopaque, Buffer) void, leave: *const fn (*anyopaque, Buffer) void };
};
