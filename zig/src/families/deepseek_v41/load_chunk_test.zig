const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const mock = @import("load_mock.zig");
const named = @import("named.zig");
const Weights = @import("cuda_weights.zig").Weights;
const chunk = 64 << 20;
const first = chunk + 8192;
var host_alloc: @FieldType(abi.Api, "cuMemHostAlloc") = undefined;
var copied: usize = 0;
var calls: usize = 0;
var fail_second = false;
var valid = true;
fn pinned(out: *?*anyopaque, n: usize, flags: c_uint) callconv(.c) abi.Result {
    if (n > chunk) return abi.error_not_ready;
    return host_alloc(out, n, flags);
}
fn copy(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    valid = valid and !mock.pending and dst == 256 + copied and n <= chunk;
    const bytes = @as([*]const u8, @ptrCast(src.?))[0..n];
    const boundary = @min(n, first -| copied);
    valid = valid and std.mem.allEqual(u8, bytes[0..boundary], 0x12);
    valid = valid and std.mem.allEqual(u8, bytes[boundary..], 0x87);
    copied += n;
    calls += 1;
    mock.pending = true;
    if (fail_second and calls == 2) {
        mock.fail_sync = true;
        return abi.error_not_ready;
    }
    return 0;
}
test "large paired weights upload in bounded chunks without changing addresses or source custody" {
    var b: named.Builder = .{ .gpa = t.allocator, .io = t.io, .pack = undefined };
    defer b.deinit();
    for ([_][]const u8{ "L0.moe.w1.suh", "L0.moe.shared.0.w1.suh" }, 0..) |name, i| {
        const bytes = try t.allocator.alignedAlloc(u8, .@"16", if (i == 0) first else 8192);
        @memset(bytes, if (i == 0) 0x12 else 0x87);
        try b.out.append(t.allocator, .{ .name = try t.allocator.dupe(u8, name), .kind = .bf16, .shape = .{ bytes.len / 2, 1, 1, 1 }, .rank = 1, .bytes = bytes });
    }
    for ([_]bool{ false, true }) |fail| {
        var driver = mock.init();
        host_alloc = driver.api.cuMemHostAlloc;
        driver.api.cuMemHostAlloc = pinned;
        driver.api.cuMemcpyHtoDAsync_v2 = copy;
        copied = 0;
        calls = 0;
        valid = true;
        fail_second = fail;
        var weights: Weights = .{ .gpa = t.allocator, .pair_stacks = true };
        defer {
            mock.fail_sync = false;
            weights.close() catch unreachable;
        }
        if (fail) {
            try t.expectError(error.NotReady, weights.upload(&driver, &b));
            try t.expectError(error.NotReady, weights.close());
            try t.expect(weights.get(b.out.items[0].name) == null);
            try t.expect(mock.pinned != null);
            try t.expectEqual(@as(usize, 0), mock.freed);
            try t.expectEqual(@as(usize, 2), calls);
        } else {
            try weights.upload(&driver, &b);
            try t.expectEqual(@as(usize, 3), calls);
            try t.expectEqual(@as(usize, first + 8192), copied);
            try t.expectEqual(@as(usize, 1), weights.bufs.items.len);
            const left = weights.get(b.out.items[0].name).?;
            const right = weights.get(b.out.items[1].name).?;
            try t.expectEqual(left.ptr + first, right.ptr);
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(b.out.items[0].bytes, &digest, .{});
            try t.expectEqual(digest, left.sha256);
            try t.expect(mock.pinned == null);
        }
        try t.expect(valid);
        try t.expectEqual(@as(usize, 0), mock.early_free);
        try t.expectEqual(@as(usize, 0), mock.double_free);
    }
}
test "large persistent forms share bounded upload and retain a failed later chunk" {
    const bytes = try t.allocator.alloc(u8, first);
    defer t.allocator.free(bytes);
    @memset(bytes, 0x12);
    for ([_]bool{ false, true }) |fail| {
        var driver = mock.init();
        host_alloc = driver.api.cuMemHostAlloc;
        driver.api.cuMemHostAlloc = pinned;
        driver.api.cuMemcpyHtoDAsync_v2 = copy;
        copied = 0;
        calls = 0;
        valid = true;
        fail_second = fail;
        var runner: @import("run.zig").Runner = .{ .gpa = t.allocator, .d = &driver, .stream = .{ .d = &driver, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
        defer runner.deinit();
        var cfg: @import("config.zig").Config = .{};
        var widths: @import("block.zig").Widths = .{};
        var f = @import("forward.zig").Forward.init(t.allocator, &cfg, &widths, .{}, &runner, undefined);
        defer f.arena.deinit();
        var dst = try cuda.DeviceBuffer.alloc(&driver, bytes.len);
        defer {
            mock.fail_sync = false;
            f.uploads.close() catch unreachable;
            dst.free();
        }
        if (fail) {
            try t.expectError(error.NotReady, f.upload(dst.ptr, bytes));
            try t.expectError(error.NotReady, f.uploads.close());
            try t.expect(mock.pinned != null);
        } else {
            try f.upload(dst.ptr, bytes);
            try t.expect(!f.uploads.pending);
        }
        try t.expectEqual(@as(usize, 2), calls);
        try t.expectEqual(bytes.len, copied);
        try t.expect(valid);
        try t.expectEqual(@as(usize, 0), mock.early_free);
    }
}
