//! A recording driver delays readback until completion and can plant a late device-weight byte change.
const std = @import("std");
const cuda = @import("cuda");
const digest = @import("cuda_weight_digest.zig");
const t = std.testing;
const Fake = struct {
    var bytes: []u8 = &.{};
    var pin: ?[]align(16) u8 = null;
    var pending: ?struct { dst: []u8, at: usize } = null;
    var fail = false;
    var copies: usize = 0;
    var early: usize = 0;
    fn alloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        const b = t.allocator.alignedAlloc(u8, .@"16", n) catch return 1;
        pin = b;
        out.* = b.ptr;
        return 0;
    }
    fn free(p: ?*anyopaque) callconv(.c) cuda.abi.Result {
        if (pending != null) early += 1;
        if (pin) |b| if (@intFromPtr(p) == @intFromPtr(b.ptr)) {
            t.allocator.free(b);
            pin = null;
        };
        return 0;
    }
    fn download(p: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (pin == null or @intFromPtr(p) != @intFromPtr(pin.?.ptr)) return 1;
        pending = .{ .dst = @as([*]u8, @ptrCast(p.?))[0..n], .at = @intCast(src - 0x1000) };
        copies += 1;
        return 0;
    }
    fn sync() callconv(.c) cuda.abi.Result {
        if (fail) return cuda.abi.error_not_ready;
        if (pending) |p| @memcpy(p.dst, bytes[p.at..][0..p.dst.len]);
        pending = null;
        return 0;
    }
    fn streamSync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return sync();
    }
    fn fill(p: u64, b: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memset(bytes[@intCast(p - 0x1000)..][0..n], b);
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .api = undefined, .lib = undefined };
        d.api.cuMemHostAlloc = alloc;
        d.api.cuMemFreeHost = free;
        d.api.cuMemcpyDtoHAsync_v2 = download;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuCtxSynchronize = sync;
        d.api.cuMemsetD8Async = fill;
        return d;
    }
};
test "weight digest: final device bytes include a piece tail and a late planted write is detected" {
    const data = try t.allocator.alloc(u8, digest.piece_bytes + 17);
    defer t.allocator.free(data);
    for (data, 0..) |*b, i| b.* = @intCast(i % 251);
    Fake.bytes = data;
    Fake.copies = 0;
    Fake.fail = false;
    var d = Fake.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(51));
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const spans = [_]digest.Span{.{ .name = "weight", .ptr = 0x1000, .len = data.len }};
    const good = try digest.collect(t.allocator, &d, s, &spans, false);
    defer t.allocator.free(good);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &sha, .{});
    try t.expectEqualStrings(&std.fmt.bytesToHex(sha, .lower), &good[0].sha256);
    try t.expectEqual(@as(usize, 2), Fake.copies);
    const bad = try digest.collect(t.allocator, &d, s, &spans, true);
    defer t.allocator.free(bad);
    try t.expect(!std.mem.eql(u8, &good[0].sha256, &bad[0].sha256));
    try t.expectEqual(@as(u8, 1), data[0]);
    try t.expectEqual(@as(usize, 0), Fake.early);
}
test "weight digest: a failed DtoH completion retains its pinned destination until recovery" {
    var data: [16]u8 = @splat(0x33);
    Fake.bytes = &data;
    Fake.fail = true;
    Fake.early = 0;
    var d = Fake.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(52));
    const before = cuda.copy_source.retained();
    try t.expectError(error.NotReady, digest.collect(t.allocator, &d, .{ .d = &d, .handle = null }, &.{.{ .name = "w", .ptr = 0x1000, .len = 16 }}, false));
    try t.expect(Fake.pin != null);
    try t.expectEqual(before + 1, cuda.copy_source.retained());
    Fake.fail = false;
    try cuda.copy_source.synchronizeContext(&d);
    try t.expect(Fake.pin == null);
    try t.expectEqual(before, cuda.copy_source.retained());
    try t.expectEqual(@as(usize, 0), Fake.early);
}
