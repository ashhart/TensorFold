//! Host recording boundary. Tokens below are width sentinels, not Metal arithmetic results.
const std = @import("std");
pub const flashnext_recording = true;
pub const ResourceOptions = @import("src/metal/types.zig").ResourceOptions;
pub const Size = @import("src/metal/types.zig").Size;
pub var allocator: std.mem.Allocator = undefined;
pub var wrong: usize = 0;
pub var calls: [4]usize = @splat(0);
pub var expected: [4]usize = .{ 4, 64, 4, 64 };
pub var scale_calls: usize = 0;
pub var first_blit: ?struct { offset: usize, bytes: usize } = null;
pub var blits: usize = 0;
pub var draws: usize = 0;
pub const Buffer = struct {
    id: usize,
    data: []align(256) u8,
    pub fn contents(b: Buffer) [*]u8 {
        return b.data.ptr;
    }
    pub fn length(b: Buffer) usize {
        return b.data.len;
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(b.data.ptr)))[0..n];
    }
    pub fn deinit(_: Buffer) void {}
};
pub const Device = struct {
    pub fn allocated(_: Device) u64 {
        return 0;
    }
    pub fn buffer(_: Device, n: usize, _: usize) !Buffer {
        const data = try allocator.alignedAlloc(u8, .fromByteUnits(256), n);
        @memset(data, 0);
        return .{ .id = @intFromPtr(data.ptr), .data = data };
    }
    pub fn tensorUnits(_: Device) bool {
        return true;
    }
    pub fn generation(_: Device) u32 {
        return 14;
    }
};
pub const CompileOptions = struct {
    pub fn mlx() @This() {
        return .{};
    }
};
pub const Library = struct {
    id: usize = 0,
    text: []const u8,
    pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !Library {
        return .{ .text = text };
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    id: usize = 0,
    name: []const u8 = "noop",
    text: []const u8 = "",
    pub fn init(_: Device, lib: Library, name: []const u8, _: bool) !Pipeline {
        return .{ .name = name, .text = lib.text };
    }
    pub fn maxThreads(_: Pipeline) usize {
        return 1024;
    }
    pub fn simdWidth(_: Pipeline) usize {
        return 32;
    }
    pub fn deinit(_: Pipeline) void {}
};
pub const Bound = struct { b: Buffer, off: usize };
var pipe: Pipeline = .{};
var bound: [32]Bound = undefined;
pub const ComputeEncoder = struct {
    pub fn setPipeline(_: ComputeEncoder, p: Pipeline) void {
        pipe = p;
    }
    pub fn setBuffer(_: ComputeEncoder, b: Buffer, off: usize, at: usize) void {
        bound[at] = .{ .b = b, .off = off };
    }
    pub fn setValue(_: ComputeEncoder, _: anytype, _: usize) void {}
    pub fn setBytes(_: ComputeEncoder, _: []const u8, _: usize) void {}
    fn value(text: []const u8, name: []const u8) usize {
        const at = (std.mem.indexOf(u8, text, name) orelse unreachable) + name.len;
        const stop = at + (std.mem.indexOfScalar(u8, text[at..], ';') orelse unreachable);
        return std.fmt.parseInt(usize, text[at..stop], 10) catch unreachable;
    }
    fn check(kind: usize, names: []const []const u8, wanted: []const usize) void {
        calls[kind] += 1;
        for (names, wanted) |name, want| if (value(pipe.text, name) != want) {
            wrong += 1;
        };
    }
    pub fn dispatchThreads(_: ComputeEncoder, grid: Size, _: Size) void {
        if (std.mem.eql(u8, pipe.name, "fz_xgu") or std.mem.eql(u8, pipe.name, "fz_xdown")) {
            calls[if (std.mem.eql(u8, pipe.name, "fz_xgu")) @as(usize, 2) else 3] += 1;
            for (expected, [_]usize{ 6, 32, 6, 32 }) |want, actual| if (want != actual) {
                wrong += 1;
            };
            return;
        }
        if (std.mem.eql(u8, pipe.name, "draw")) {
            draws += 1;
            return;
        }
        if (std.mem.indexOf(u8, pipe.name, "qa_embed_rows_") != null) check(0, &.{ "constexpr int BITS = ", "constexpr int GS = " }, expected[0..2]) else if (std.mem.indexOf(u8, pipe.name, "qa_ple_lookup_") != null) check(1, &.{ "constexpr int BITS = ", "constexpr int GS = " }, expected[0..2]) else if (std.mem.indexOf(u8, pipe.name, "qa_expert_gateup_") != null) check(2, &.{ "constexpr int WB = ", "constexpr int WG = ", "constexpr int SWB = ", "constexpr int SWG = " }, &expected) else if (std.mem.indexOf(u8, pipe.name, "qa_expert_down_y_") != null) check(3, &.{ "constexpr int WB = ", "constexpr int WG = ", "constexpr int SWB = ", "constexpr int SWG = " }, &expected) else if (std.mem.eql(u8, pipe.name, "scale")) {
            scale_calls += 1;
        } else if (std.mem.eql(u8, pipe.name, "argmax")) {
            const b = bound[1];
            const out = @as([*]u32, @ptrCast(@alignCast(b.b.contents() + b.off)));
            for (0..grid.width / 1024) |i| out[i] = if (wrong == 0) 7 else 9;
        }
    }
    pub fn dispatchGroups(e: ComputeEncoder, grid: Size, threads: Size) void {
        e.dispatchThreads(grid, threads);
    }
    pub fn barrier(_: ComputeEncoder) void {}
    pub fn end(_: ComputeEncoder) void {}
};
pub const Queue = struct {
    pub fn deinit(_: @This()) void {}
    pub fn commandBuffer(_: Queue) CommandBuffer {
        return .{};
    }
};
pub const CommandBuffer = struct {
    pub fn compute(_: CommandBuffer, _: enum { serial, concurrent }) ComputeEncoder {
        return .{};
    }
    pub fn blit(_: CommandBuffer) BlitEncoder {
        return .{};
    }
    pub fn commit(_: CommandBuffer) void {}
    pub fn wait(_: CommandBuffer) void {}
    pub fn gpuSeconds(_: CommandBuffer) f64 {
        return 0;
    }
    pub fn failure(_: CommandBuffer) ?[]const u8 {
        return null;
    }
};
pub const BlitEncoder = struct {
    pub fn copy(_: BlitEncoder, a: Buffer, ao: usize, b: Buffer, bo: usize, n: usize) void {
        _ = a;
        _ = b;
        _ = bo;
        if (first_blit == null) first_blit = .{ .offset = ao, .bytes = n };
        blits += 1;
    }
    pub fn end(_: BlitEncoder) void {}
};
pub const ResidencySet = struct {
    pub const Follow = struct { ctx: *anyopaque, join: *const fn (*anyopaque, Buffer) void, leave: *const fn (*anyopaque, Buffer) void };
};
pub const SharedEvent = struct {
    id: usize = 0,
    pub fn deinit(_: @This()) void {}
};
pub const objc = struct {
    pub const Id = usize;
    pub fn release(_: Id) void {}
    pub const Pool = struct {
        pub fn push() @This() {
            return .{};
        }
        pub fn pop(_: @This()) void {}
    };
};
pub const clock = struct {
    pub fn seconds() f64 {
        return 0;
    }
};

pub const MappedFile = struct {
    bytes: [*]const u8,
    size: usize,
    pub fn open(_: []const u8) !MappedFile {
        return error.UnexpectedFile;
    }
};
