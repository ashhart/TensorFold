//! #2102 on a recording driver: which host pointer DeviceBuffer.download hands cuMemcpyDtoH_v2, with no GPU.
const std = @import("std");
const t = std.testing;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const pinned = @import("pinned_download.zig");
const copy_source = @import("copy_source.zig");
const env = @import("sys").env;

const name = "TF_CUDA_PINNED_DOWNLOAD";
const base: abi.DevicePtr = 0x7000_0000;

/// Records each copy and pinned allocation; a copy writes the low byte of each source address it reads.
const Fake = struct {
    const Copy = struct { dst: usize, src: abi.DevicePtr, n: usize };
    var copies: std.ArrayList(Copy) = .empty;
    var blocks: std.ArrayList([]align(16) u8) = .empty;
    var allocs: usize = 0;
    var frees: usize = 0;
    var copy_result: abi.Result = abi.success;

    fn reset() void {
        copies.clearRetainingCapacity();
        allocs = 0;
        frees = 0;
        copy_result = abi.success;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        const b = t.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        blocks.append(t.allocator, b) catch return abi.error_not_ready;
        allocs += 1;
        out.* = b.ptr;
        return abi.success;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        for (blocks.items, 0..) |b, i| if (@intFromPtr(b.ptr) == @intFromPtr(ptr)) {
            t.allocator.free(b);
            _ = blocks.swapRemove(i);
            frees += 1;
            return abi.success;
        };
        return abi.error_not_ready;
    }
    fn dtoh(dst: ?*anyopaque, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
        copies.append(t.allocator, .{ .dst = @intFromPtr(dst), .src = src, .n = n }) catch return abi.error_not_ready;
        if (copy_result != abi.success) return copy_result;
        const bytes: [*]u8 = @ptrCast(dst.?);
        for (bytes[0..n], 0..) |*b, i| b.* = @truncate(src + i);
        return abi.success;
    }
    fn sync() callconv(.c) abi.Result {
        return abi.success;
    }
    fn release(_: abi.Device) callconv(.c) abi.Result {
        return abi.success;
    }
    fn driver() Driver {
        var d: Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemcpyDtoH_v2 = dtoh;
        d.api.cuCtxSynchronize = sync;
        return d;
    }
    fn pinnedAt(i: usize) usize {
        return @intFromPtr(blocks.items[i].ptr);
    }
    fn deinit() void {
        copies.deinit(t.allocator);
        blocks.deinit(t.allocator);
        copies = .empty;
        blocks = .empty;
    }
};

/// Sets or unsets the switch and has the next download read it, as a fresh process would.
fn set(value: ?[:0]const u8) void {
    _ = if (value) |v| env.setenv(name, v, 1) else env.unsetenv(name);
    pinned.rereadForTest();
}

/// The bytes a download of `out.len` from `offset` must deliver: each source address's low byte.
fn expectDelivered(out: []const u8, offset: usize) !void {
    for (out, 0..) |b, i| if (b != @as(u8, @truncate(base + offset + i))) {
        std.debug.print("byte {d} of {d}: {d}\n", .{ i, out.len, b });
        return error.TestUnexpectedResult;
    };
}

test "pinned download: with the switch unset or not exactly 1, the recorded call is main's, into the caller's slice" {
    defer Fake.deinit();
    defer set(null);
    var d = Fake.driver();
    const buf: memory.DeviceBuffer = .{ .d = &d, .ptr = base, .len = 4 << 20, .borrowed = true };
    for ([_]?[:0]const u8{ null, "0", "", "true", " 1", "01" }) |raw| {
        set(raw);
        try t.expect(!pinned.enabled());
        Fake.reset();
        for ([_]usize{ 4, 32 << 10, 303_104 }) |len| {
            const out = try t.allocator.alloc(u8, len);
            defer t.allocator.free(out);
            @memset(out, 0xAA);
            try buf.download(16, out);
            const c = Fake.copies.items[Fake.copies.items.len - 1];
            try t.expectEqual(@intFromPtr(out.ptr), c.dst);
            try t.expectEqual(base + 16, c.src);
            try t.expectEqual(len, c.n);
            try expectDelivered(out, 16);
        }
        try t.expectEqual(@as(usize, 3), Fake.copies.items.len);
        try t.expectEqual(@as(usize, 0), Fake.allocs);
        try t.expectEqual(@as(usize, 0), pinned.idleCount(&d));
    }
}

test "pinned download: TF_CUDA_PINNED_DOWNLOAD=1 copies into the driver's pinned allocation, then into the slice" {
    defer Fake.deinit();
    defer set(null);
    set("1");
    try t.expect(pinned.enabled());
    Fake.reset();
    var d = Fake.driver();
    const before = memory.usage(false).host;
    const buf: memory.DeviceBuffer = .{ .d = &d, .ptr = base, .len = 8 << 20, .borrowed = true };
    var out: [303_104]u8 = undefined;
    @memset(&out, 0xAA);
    try buf.download(64, &out);
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    const first = Fake.pinnedAt(0);
    try t.expectEqual(Fake.Copy{ .dst = first, .src = base + 64, .n = out.len }, Fake.copies.items[0]);
    try t.expect(first != @intFromPtr(&out));
    try expectDelivered(&out, 64);
    try t.expectEqual(before + out.len, memory.usage(false).host);

    var small: [4]u8 = @splat(0xAA);
    try buf.download(4096, &small); // a smaller copy reuses the bounce
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    try t.expectEqual(Fake.Copy{ .dst = first, .src = base + 4096, .n = 4 }, Fake.copies.items[1]);
    try expectDelivered(&small, 4096);

    const large = try t.allocator.alloc(u8, 4 << 20);
    defer t.allocator.free(large);
    @memset(large, 0xAA);
    try buf.download(0, large); // a larger one replaces it with one sized to the new largest copy
    try t.expectEqual(@as(usize, 2), Fake.allocs);
    try t.expectEqual(@as(usize, 1), Fake.frees);
    try t.expectEqual(Fake.Copy{ .dst = Fake.pinnedAt(0), .src = base, .n = large.len }, Fake.copies.items[2]);
    try expectDelivered(large, 0);
    try t.expectEqual(before + large.len, memory.usage(false).host);
    try t.expectEqual(@as(usize, 1), pinned.idleCount(&d));

    pinned.drop(&d, copy_source.current(), false);
    try t.expectEqual(@as(usize, 2), Fake.frees);
    try t.expectEqual(@as(usize, 0), pinned.idleCount(&d));
    try t.expectEqual(before, memory.usage(false).host);
}

test "pinned download: the switch is read once, and a later change reaches no download until it is read again" {
    defer Fake.deinit();
    defer set(null);
    set(null);
    Fake.reset();
    var d = Fake.driver();
    defer pinned.drop(&d, null, true);
    const buf: memory.DeviceBuffer = .{ .d = &d, .ptr = base, .len = 4096, .borrowed = true };
    var out: [64]u8 = undefined;
    try buf.download(0, &out);
    _ = env.setenv(name, "1", 1); // no reread: the unset switch's first read stands, and the call stays main's
    try buf.download(0, &out);
    try t.expectEqual(@as(usize, 0), Fake.allocs);
    try t.expectEqual(@intFromPtr(&out), Fake.copies.items[1].dst);
    pinned.rereadForTest();
    try buf.download(0, &out);
    try t.expectEqual(Fake.pinnedAt(0), Fake.copies.items[2].dst);
    _ = env.unsetenv(name); // no reread: the exact 1 stands
    try buf.download(0, &out);
    try t.expectEqual(Fake.pinnedAt(0), Fake.copies.items[3].dst);
    try expectDelivered(&out, 0);
}

test "pinned download: a copy longer than the bounce goes through it in pieces, the bounce capped" {
    defer Fake.deinit();
    defer set(null);
    set("1");
    Fake.reset();
    var d = Fake.driver();
    defer pinned.drop(&d, null, true);
    const buf: memory.DeviceBuffer = .{ .d = &d, .ptr = base, .len = pinned.chunk + 8, .borrowed = true };
    const out = try t.allocator.alloc(u8, pinned.chunk + 3);
    defer t.allocator.free(out);
    try buf.download(5, out);
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    try t.expectEqual(pinned.chunk, Fake.blocks.items[0].len);
    const p = Fake.pinnedAt(0);
    try t.expectEqualSlices(Fake.Copy, &.{ .{ .dst = p, .src = base + 5, .n = pinned.chunk }, .{ .dst = p, .src = base + 5 + pinned.chunk, .n = 3 } }, Fake.copies.items);
    try expectDelivered(out, 5);
}

test "pinned download: bounces belong to their context, and the context's end frees its own" {
    defer Fake.deinit();
    defer set(null);
    set("1");
    Fake.reset();
    var d = Fake.driver();
    d.api.cuDevicePrimaryCtxRelease_v2 = Fake.release;
    const previous = copy_source.current();
    defer copy_source.setCurrent(previous);
    defer pinned.drop(&d, null, true);
    const buf: memory.DeviceBuffer = .{ .d = &d, .ptr = base, .len = 4096, .borrowed = true };
    var out: [256]u8 = undefined;
    for ([_]usize{ 31, 32, 31 }) |context| {
        copy_source.setCurrent(@ptrFromInt(context));
        try buf.download(0, &out);
        try expectDelivered(&out, 0);
    }
    try t.expectEqual(@as(usize, 2), Fake.allocs); // context 31's second download reused its own bounce
    try t.expectEqual(Fake.copies.items[0].dst, Fake.copies.items[2].dst);
    try t.expect(Fake.copies.items[0].dst != Fake.copies.items[1].dst);
    var context: @import("context.zig").Context = .{ .d = &d, .device = 0, .handle = @ptrFromInt(31) };
    context.deinit();
    try t.expectEqual(@as(usize, 1), Fake.frees);
    try t.expectEqual(@as(usize, 1), pinned.idleCount(&d));
    try t.expectEqual(@intFromPtr(Fake.blocks.items[0].ptr), Fake.copies.items[1].dst); // context 32's bounce stays
}

test "pinned download: a failed copy quarantines its bounce until a later wait succeeds" {
    defer Fake.deinit();
    defer set(null);
    set("1");
    Fake.reset();
    var d = Fake.driver();
    const retained = copy_source.retained();
    const buf: memory.DeviceBuffer = .{ .d = &d, .ptr = base, .len = 4096, .borrowed = true };
    var out: [64]u8 = undefined;
    Fake.copy_result = abi.error_not_ready; // fails the call without the driver's error log, which would fail the test
    try t.expectError(error.NotReady, buf.download(0, &out));
    try t.expectEqual(@as(usize, 0), pinned.idleCount(&d));
    try t.expectEqual(retained + 1, copy_source.retained());
    try t.expectEqual(@as(usize, 0), Fake.frees);
    try copy_source.synchronizeContext(&d);
    try t.expectEqual(retained, copy_source.retained());
    try t.expectEqual(@as(usize, 1), Fake.frees);
}
