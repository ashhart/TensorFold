//! The tower's upload through a fake driver, stopping before device arithmetic.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const source = cuda.copy_source;
const Tower = @import("tower.zig").Tower;

const Fake = struct {
    var allocs: usize = 0;
    var frees: usize = 0;
    var waits: usize = 0;
    var flight: ?[*]const u8 = null;
    var expected: []const u8 = &.{};
    var held: []const u8 = &.{};
    var blocks: std.ArrayList([]align(16) u8) = .empty;
    fn alloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        const b = t.allocator.alignedAlloc(u8, .@"16", len) catch unreachable;
        blocks.append(t.allocator, b) catch unreachable;
        allocs += 1;
        out.* = b.ptr;
        return 0;
    }
    fn free(ptr: ?*anyopaque) callconv(.c) abi.Result {
        std.debug.assert(flight == null);
        for (blocks.items, 0..) |b, i| if (@intFromPtr(b.ptr) == @intFromPtr(ptr)) {
            t.allocator.free(b);
            _ = blocks.swapRemove(i);
            frees += 1;
            return 0;
        };
        unreachable;
    }
    fn copy(_: u64, ptr: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(flight == null);
        const bytes: [*]const u8 = @ptrCast(ptr.?);
        std.debug.assert(std.mem.eql(u8, expected, bytes[0..len]));
        flight = bytes;
        held = expected;
        return 0;
    }
    fn create(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn query(_: abi.Event) callconv(.c) abi.Result {
        return if (flight == null) 0 else abi.error_not_ready;
    }
    fn wait(_: abi.Event) callconv(.c) abi.Result {
        if (flight) |ptr| std.debug.assert(std.mem.eql(u8, held, ptr[0..held.len]));
        flight = null;
        waits += 1;
        return 0;
    }
    fn destroy(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return wait(null);
    }
    fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        return abi.error_not_ready;
    }
};

test "RADIO upload reuses its pinned pixels without freeing between images" {
    Fake.allocs = 0;
    Fake.frees = 0;
    Fake.waits = 0;
    Fake.flight = null;
    defer Fake.blocks.deinit(t.allocator);
    var d: cuda.Driver = undefined;
    d.api.cuMemHostAlloc = Fake.alloc;
    d.api.cuMemFreeHost = Fake.free;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.copy;
    d.api.cuEventCreate = Fake.create;
    d.api.cuEventRecord = Fake.record;
    d.api.cuEventQuery = Fake.query;
    d.api.cuEventSynchronize = Fake.wait;
    d.api.cuEventDestroy_v2 = Fake.destroy;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuLaunchKernel = Fake.launch;
    source.setCurrent(null);
    source.setSparesForTest(.{});
    defer source.setSparesForTest(.{ .count = 0, .bytes = 0 });
    defer source.dropSpares(&d, null);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var tower: Tower = undefined;
    tower.c = @import("video_test.zig").cfg;
    tower.b = .{ .px = 0, .h = 0, .x = 0, .wide = 0, .a = 0, .acc = 256 };
    tower.stage = .{};
    defer tower.stage.deinit();
    const len = tower.c.max_patches * 3 * tower.c.patch * tower.c.patch;
    const images = try t.allocator.alloc(f32, 2 * len);
    defer t.allocator.free(images);
    @memset(images[0..len], 0.5);
    @memset(images[len..], -0.5);
    defer source.synchronizeStream(&d, null) catch unreachable;
    tower.scratch = .{ .d = &d, .ptr = 256, .len = len * @sizeOf(f32) };
    tower.k.cast_fn = .{ .d = &d, .handle = null };
    _ = try tower.stage.take(&d, len * @sizeOf(f32));
    for (0..4) |i| {
        const pixels = images[i % 2 * len ..][0..len];
        Fake.expected = std.mem.sliceAsBytes(pixels);
        try t.expectError(error.NotReady, tower.features(stream, .{ .pixels = pixels, .gh = 128, .gw = 104, .tokens = 3328, .at = 0 }));
        try t.expectEqual(@as(usize, 1), Fake.allocs);
        try t.expectEqual(@as(usize, 0), Fake.frees);
    }
    try t.expectEqual(@as(usize, 3), Fake.waits);
    try tower.stage.close();
    try t.expectEqual(@as(usize, 4), Fake.waits);
    try t.expectEqual(@as(usize, 1), Fake.frees);
}
