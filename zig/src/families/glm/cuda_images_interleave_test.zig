//! A/B/A crosses actual follower admission and splice dispatch without running a tower or CUDA arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const Images = @import("cuda_images.zig").Images;
const Fake = struct {
    var launches: usize = 0;
    var feature: u64 = 0;
    var next: u64 = 0x10000;
    var ordered_uploads: usize = 0;
    var upload_stream: cuda.abi.Stream = null;
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += n + 256;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        return 0;
    }
    fn upload(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn uploadAsync(_: u64, _: ?*const anyopaque, _: usize, stream: cuda.abi.Stream) callconv(.c) c_int {
        ordered_uploads += 1;
        upload_stream = stream;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        launches += 1;
        feature = @as(*const u64, @ptrCast(@alignCast(args.?[0].?))).*;
        return 0;
    }
};
fn run(other_media: bool) !void {
    const a = std.testing.allocator;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    d.api.cuMemcpyHtoD_v2 = Fake.upload;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.uploadAsync;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x55) };
    var map: [2049]i32 = undefined;
    var v: Images = undefined;
    v.gpa = a;
    v.d = &d;
    v.limits = .{ .image_tokens = 4, .max_images = 2 };
    v.video_ids = null;
    v.spans = .empty;
    v.slot = 0;
    v.saved = .empty;
    v.rows = .{ .d = &d, .ptr = 0x1000, .len = 4096, .borrowed = true };
    v.map = .{ .d = &d, .ptr = 0x2000, .len = 8196, .borrowed = true };
    v.host_map = &map;
    v.run.stream = stream;
    v.run.f.inject = .{ .d = &d, .handle = null };
    defer {
        var it = v.saved.valueIterator();
        while (it.next()) |state| {
            state.rows.free();
            state.spans.deinit(a);
        }
        v.saved.deinit(a);
        v.rows.free();
        v.spans.deinit(a);
    }
    Fake.launches = 0;
    Fake.ordered_uploads = 0;
    try v.take(7, &.{ 2048, 2, 0 });
    try v.splice(stream, 7, 0, 2048, 0x3000, 4);
    try std.testing.expectEqual(@as(usize, 0), Fake.launches);
    // A's image is past its first 2048-row chunk. B is a shorter text or different-media prompt.
    try v.take(8, if (other_media) &.{ 1, 1, 0 } else &.{});
    try v.splice(stream, 7, 2048, 2050, 0x3000, 4);
    try std.testing.expectEqual(@as(usize, 1), Fake.launches);
    try std.testing.expectEqual(@as(usize, 1), Fake.ordered_uploads);
    try std.testing.expectEqual(stream.handle, Fake.upload_stream);
    const a_rows = Fake.feature;
    if (other_media) {
        try v.splice(stream, 8, 1, 2, 0x3000, 4);
        try std.testing.expect(Fake.feature != a_rows);
        try v.splice(stream, 7, 2048, 2050, 0x3000, 4);
        try std.testing.expectEqual(a_rows, Fake.feature);
    }
    v.release(7);
    try v.take(7, &.{ 2050, 1, 0 });
    try v.splice(stream, 7, 2050, 2051, 0x3000, 4);
    try std.testing.expect(Fake.feature != a_rows);
    try std.testing.expectEqual(@as(i32, 0), map[0]);
    // A restored prefix needs only the suffix's spans; a failed B admission does not clear A.
    try std.testing.expectError(error.ImageRows, v.take(8, &.{ 1, 2, 4 }));
    try v.splice(stream, 7, 2048, 2051, 0x3000, 4);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0 }, map[0..3]);
}
test "an image stream's rows survive a shorter text stream between its prompt chunks" {
    try run(false);
}
test "a stream's media rows survive another stream's different media between its prompt chunks" {
    try run(true);
}

test "slot 0's stream released while another slot is bound leaves that slot's own release whole" {
    const a = std.testing.allocator;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    d.api.cuMemcpyHtoD_v2 = Fake.upload;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.uploadAsync;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x55) };
    var map: [2049]i32 = undefined;
    var v: Images = undefined;
    v.gpa = a;
    v.d = &d;
    v.limits = .{ .image_tokens = 4, .max_images = 2 };
    v.video_ids = null;
    v.spans = .empty;
    v.slot = 0;
    v.saved = .empty;
    v.rows = .{ .d = &d, .ptr = 0x1000, .len = 4096, .borrowed = true };
    v.map = .{ .d = &d, .ptr = 0x2000, .len = 8196, .borrowed = true };
    v.host_map = &map;
    v.run.stream = stream;
    v.run.f.inject = .{ .d = &d, .handle = null };
    defer {
        var it = v.saved.valueIterator();
        while (it.next()) |state| {
            state.rows.free();
            state.spans.deinit(a);
        }
        v.saved.deinit(a);
        v.rows.free();
        v.spans.deinit(a);
    }
    try v.take(0, &.{ 1, 2, 0 });
    try v.take(5, &.{ 3, 1, 0 }); // slot 0's rows and spans are put aside
    v.release(0); // slot 0's stream ends first: its rows go
    v.release(5); // then slot 5's: its buffer is slot 0's from here
    try std.testing.expectEqual(@as(u32, 0), v.slot);
    try std.testing.expectEqual(@as(usize, 0), v.spans.items.len);
    try std.testing.expectEqual(@as(usize, 0), v.saved.count());
    try v.take(0, &.{ 1, 2, 0 });
    try v.splice(stream, 0, 0, 4, 0x3000, 1);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, -1 }, map[0..4]);
}
