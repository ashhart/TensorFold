//! Real recurrence launch adapters under owned recording objects; this does not execute CUDA or Metal arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const family = @import("qwen4").family;
const gdn = family.cuda_gdn;
const wo = @import("core").weight_only;
const T = std.testing;
const Cu = gdn.Kernels(cuda);
const Trace = struct {
    var live: usize = 0;
    var launches: usize = 0;
    var max_threads: c_int = 1024;
    var fails_drain = false;
    var expected_stride: u64 = 32768;
    var ints: [4]c_int = undefined;
    var pointers: [10]u64 = undefined;
    fn load(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        live += 1;
        out.* = @ptrFromInt(0x100);
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        std.debug.assert(std.mem.eql(u8, std.mem.span(name), "qwen4_gdn_chain"));
        out.* = @ptrFromInt(0x200);
        return 0;
    }
    fn attr(out: *c_int, _: cuda.abi.FunctionAttribute, _: cuda.abi.Function) callconv(.c) cuda.abi.Result {
        out.* = max_threads;
        return 0;
    }
    fn drain(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (fails_drain) cuda.abi.error_not_ready else 0;
    }
    fn arg(comptime V: type, args: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(_: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(gx == 2 and gy == 1 and gz == 1 and bx == 1024 and by == 1 and bz == 1 and shared == 0);
        std.debug.assert(arg(c_int, args, 8) == 3 and arg(f32, args, 7) == @as(f32, 1e-6));
        for (pointers[0..7], 0..) |*p, i| p.* = arg(u64, args, i);
        for (pointers[7..], 9..) |*p, i| p.* = arg(u64, args, i);
        for (&ints, 12..) |*v, i| v.* = arg(c_int, args, i);
        std.debug.assert(arg(u64, args, 16) == expected_stride);
        launches += 1;
        return 0;
    }
};
fn driver() cuda.Driver {
    var out: cuda.Driver = .{ .lib = undefined, .api = undefined };
    out.api.cuModuleLoadData = Trace.load;
    out.api.cuModuleUnload = Trace.unload;
    out.api.cuModuleGetFunction = Trace.function;
    out.api.cuFuncGetAttribute = Trace.attr;
    out.api.cuStreamSynchronize = Trace.drain;
    out.api.cuLaunchKernel = Trace.launch;
    return out;
}
test "Qwen4 GDN launch: CUDA ABI and refusal before launch, failed drain retains owner" {
    var d = driver();
    const stream = cuda.Stream.borrowed(&d, null);
    const image: [8]u8 align(8) = @splat(1);
    var kernels = try Cu.load(&d, &image);
    const p: gdn.Params = .{ .rows = 3, .keys = 1, .values = 2, .eps = 1e-6 };
    const s = try gdn.sizes(p);
    const needs = [_]usize{ s.projection, s.conv_state, s.conv_weight, s.state, s.scalar, s.scalar, s.norm, s.output, s.sums, s.snapshots };
    var ranges: [10]wo.Range = undefined;
    for (&ranges, needs, 0..) |*r, n, i| r.* = .{ .address = (i + 1) * 0x1000000, .len = n };
    try kernels.encode(stream, p, ranges);
    try T.expectEqualSlices(c_int, &.{ 1, 2, 0, 3 }, &Trace.ints);
    for (ranges, Trace.pointers) |range, pointer| try T.expectEqual(range.address, pointer);
    for ([_]u4{ gdn.weightFlags(true, true), gdn.weightFlags(true, false), gdn.weightFlags(false, false) }) |flags| {
        var weighted = p;
        weighted.weight_flags = flags;
        try kernels.encode(stream, weighted, ranges);
        try T.expectEqual(@as(c_int, flags), Trace.ints[3]);
    }
    var strided = p;
    strided.row_stride = 262144;
    var spaced = ranges;
    spaced[9].len = (try gdn.sizes(strided)).snapshots;
    Trace.expected_stride = 65536;
    try kernels.encode(stream, strided, spaced);
    Trace.expected_stride = 32768;
    const before = Trace.launches;
    ranges[7] = ranges[0];
    try T.expectError(error.Qwen4GdnAlias, kernels.encode(stream, p, ranges));
    try T.expectEqual(before, Trace.launches);
    Trace.fails_drain = true;
    try T.expectError(error.NotReady, kernels.close(stream));
    try T.expectEqual(@as(usize, 1), Trace.live);
    Trace.fails_drain = false;
    try kernels.close(stream);
    Trace.max_threads = 512;
    try T.expectError(error.Qwen4GdnThreads, Cu.load(&d, &image));
    Trace.max_threads = 1024;
    try T.expectEqual(@as(usize, 0), Trace.live);
}
const Recording = struct {
    var repeat_value: ?u32 = null;
    pub const Size = struct {
        width: usize,
        height: usize,
        depth: usize,
        pub fn of(x: usize, y: usize, z: usize) @This() {
            return .{ .width = x, .height = y, .depth = z };
        }
    };
    pub var live: usize = 0;
    pub var calls: usize = 0;
    pub var bindings: [15]?usize = @splat(null);
    pub var rows: c_int = 0;
    pub var epsilon: f32 = 0;
    pub var expected_stride: u64 = 0;
    pub var expected_norm_round: u32 = 1;
    pub var expected_conv_round: u32 = 0;
    pub var max_threads: usize = 1024;
    pub const Device = struct { id: usize = 1 };
    pub const Buffer = struct {
        bytes: []u8,
        owner: usize = 1,
        pub fn deviceId(b: Buffer) usize {
            return b.owner;
        }
        pub fn length(b: Buffer) usize {
            return b.bytes.len;
        }
        pub fn gpuAddress(b: Buffer) u64 {
            return @intFromPtr(b.bytes.ptr);
        }
    };
    pub const CompileOptions = struct {
        pub fn mlx() @This() {
            return .{};
        }
    };
    pub const Library = struct {
        pub fn fromSource(_: Device, _: []const u8, _: CompileOptions) !Library {
            return .{};
        }
        pub fn deinit(_: Library) void {}
    };
    pub const Pipeline = struct {
        pub fn init(_: Device, _: Library, name: []const u8, _: bool) !Pipeline {
            std.debug.assert(std.mem.eql(u8, name, "qwen4_gdn_chain"));
            live += 1;
            return .{};
        }
        pub fn maxThreads(_: Pipeline) usize {
            return max_threads;
        }
        pub fn simdWidth(_: Pipeline) usize {
            return 32;
        }
        pub fn deinit(_: Pipeline) void {
            live -= 1;
        }
    };
    pub const ComputeEncoder = struct {
        owner: usize = 1,
        pub fn deviceId(e: ComputeEncoder) usize {
            return e.owner;
        }
        pub fn setPipeline(_: ComputeEncoder, _: Pipeline) void {
            calls += 1;
        }
        pub fn setBuffer(_: ComputeEncoder, b: Buffer, off: usize, i: usize) void {
            bindings[i] = @intFromPtr(b.bytes.ptr) + off;
            calls += 1;
        }
        pub fn setBytes(_: ComputeEncoder, bytes: []const u8, i: usize) void {
            if (i == family.metal_control.slot) {
                std.debug.assert(bytes.len == @sizeOf(family.metal_control.Params));
                repeat_value = std.mem.bytesToValue(family.metal_control.Params, bytes).repeat;
            }
            if (i == 7) epsilon = std.mem.bytesToValue(f32, bytes[0..4]);
            if (i == 8) rows = std.mem.bytesToValue(c_int, bytes[0..4]);
            if (i == 14) std.debug.assert(std.mem.bytesToValue(u64, bytes[0..8]) == expected_stride);
            if (i == 15) std.debug.assert(std.mem.bytesToValue(u32, bytes[0..4]) == expected_norm_round);
            if (i == 16) std.debug.assert(std.mem.bytesToValue(u32, bytes[0..4]) == expected_conv_round);
            if (i == 12) std.debug.assert(std.mem.bytesToValue(c_int, bytes[0..4]) == 0);
            calls += 1;
        }
        pub fn dispatchGroups(_: ComputeEncoder, grid: Size, threads: Size) void {
            std.debug.assert(grid.width == 2 and grid.height == 1 and threads.width == 1024);
            calls += 1;
        }
        pub fn barrier(_: ComputeEncoder) void {
            calls += 1;
        }
    };
};
test "Qwen4 GDN launch: Metal binding layout and cross-device refusal before binding" {
    const M = family.metal_gdn.Kernels(Recording);
    var arena: std.heap.ArenaAllocator = .init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var kernels = try M.load(.{}, "recording", 1, 2, .silu);
    const p: gdn.Params = .{ .rows = 3, .keys = 1, .values = 2, .eps = 1e-6 };
    const s = try gdn.sizes(p);
    const needs = [_]usize{ s.projection, s.conv_state, s.state, s.conv_weight, s.scalar, s.scalar, s.norm, s.output, p.rows * s.conv_state, p.rows * s.state };
    var views: [10]M.View = undefined;
    for (&views, needs) |*view, n| view.* = .{ .buffer = .{ .bytes = (try a.alignedAlloc(u8, .@"16", n)) }, .len = n };
    try kernels.encode(.{}, p, views);
    try T.expectEqual(@as(?u32, null), Recording.repeat_value);
    kernels.control = .repeat;
    try kernels.testEncode(.{}, p, views, 9);
    try T.expectEqual(@as(?u32, 9), Recording.repeat_value);
    kernels.control = .production;
    var f32_norm = p;
    f32_norm.weight_flags = 7;
    Recording.expected_norm_round = 0;
    try kernels.encode(.{}, f32_norm, views);
    Recording.expected_norm_round = 1;
    var rounded_conv = p;
    rounded_conv.weight_flags = gdn.weightFlags(true, true);
    Recording.expected_conv_round = 1;
    try kernels.encode(.{}, rounded_conv, views);
    rounded_conv.weight_flags = gdn.weightFlags(true, false);
    Recording.expected_norm_round = 0;
    try kernels.encode(.{}, rounded_conv, views);
    Recording.expected_norm_round = 1;
    Recording.expected_conv_round = 0;
    try T.expectEqual(@as(c_int, 3), Recording.rows);
    try T.expectEqual(p.eps, Recording.epsilon);
    for (views[7..], 9..) |view, i| try T.expectEqual(@intFromPtr(view.buffer.bytes.ptr), Recording.bindings[i].?);
    var strided = p;
    strided.row_stride = 262144;
    var spaced = views;
    spaced[8].buffer.bytes = try a.alignedAlloc(u8, .@"16", try gdn.extent(strided, s.conv_state));
    spaced[8].len = spaced[8].buffer.bytes.len;
    spaced[9].buffer.bytes = try a.alignedAlloc(u8, .@"16", (try gdn.sizes(strided)).snapshots);
    spaced[9].len = spaced[9].buffer.bytes.len;
    Recording.expected_stride = 262144;
    try kernels.encode(.{}, strided, spaced);
    Recording.expected_stride = 0;
    // The actual state owner interleaves both outputs in complete frames of one allocation.
    const joined = try a.alignedAlloc(u8, .@"16", 2 * strided.row_stride + 4096 + s.state);
    spaced[8] = .{ .buffer = .{ .bytes = joined }, .len = try gdn.extent(strided, s.conv_state) };
    spaced[9] = .{ .buffer = .{ .bytes = joined }, .offset = 4096, .len = try gdn.extent(strided, s.state) };
    Recording.expected_stride = 262144;
    try kernels.encode(.{}, strided, spaced);
    Recording.expected_stride = 0;
    const before = Recording.calls;
    var collision = spaced;
    collision[9].offset = 16;
    try T.expectError(error.Qwen4GdnAlias, kernels.encode(.{}, strided, collision));
    var misaligned = views;
    misaligned[2].offset = 4;
    misaligned[2].len -= 4;
    // Keep the logical view long enough to test the native float4 alignment separately from bounds.
    misaligned[2].buffer.bytes = try a.alignedAlloc(u8, .@"16", needs[2] + 16);
    misaligned[2].len = needs[2];
    try T.expectError(error.Qwen4GdnAlignment, kernels.encode(.{}, p, misaligned));
    try T.expectError(error.Qwen4GdnContext, kernels.encode(.{ .owner = 2 }, p, views));
    views[0].buffer.owner = 2;
    try T.expectError(error.Qwen4GdnContext, kernels.encode(.{}, p, views));
    try T.expectEqual(before, Recording.calls);
    kernels.deinit();
    Recording.max_threads = 512;
    try T.expectError(error.Qwen4GdnThreads, M.load(.{}, "recording", 1, 2, .silu));
    Recording.max_threads = 1024;
    try T.expectEqual(@as(usize, 0), Recording.live);
}
