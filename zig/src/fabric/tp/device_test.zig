//! The `Collective` contract on in-process ranks of one device, over a fake driver whose copies move host memory.
const std = @import("std");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const device = @import("device.zig");
const contract = @import("host_test.zig");
const Collective = collective.Collective;
const abi = cuda.abi;

fn addr(b: []u8) collective.DevicePtr {
    return @intFromPtr(b.ptr);
}

const Fake = struct {
    var capturing: bool = false;

    fn bytes(p: usize, n: usize) []u8 {
        return @as([*]u8, @ptrFromInt(p))[0..n];
    }
    fn dtod(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        @memcpy(bytes(dst, n), bytes(src, n));
        return abi.success;
    }
    fn dtoh(dst: ?*anyopaque, src: abi.DevicePtr, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        @memcpy(bytes(@intFromPtr(dst), n), bytes(src, n));
        return abi.success;
    }
    fn htod(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        @memcpy(bytes(dst, n), bytes(@intFromPtr(src), n));
        return abi.success;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }
    fn isCapturing(_: abi.Stream, status: *abi.CaptureStatus) callconv(.c) abi.Result {
        status.* = if (capturing) .active else .none;
        return abi.success;
    }
    fn setCurrent(_: abi.Context) callconv(.c) abi.Result {
        return abi.success;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemcpyDtoDAsync_v2 = dtod;
        d.api.cuMemcpyDtoHAsync_v2 = dtoh;
        d.api.cuMemcpyHtoDAsync_v2 = htod;
        d.api.cuStreamSynchronize = sync;
        d.api.cuStreamIsCapturing = isCapturing;
        d.api.cuCtxSetCurrent = setCurrent;
        return d;
    }
};

fn context(d: *const cuda.Driver) cuda.Context {
    return .{ .d = d, .device = 0, .handle = null };
}

test "every call on two, three and four ranks of one device equals the reference" {
    var d = Fake.driver();
    const cx = context(&d);
    for ([_]u32{ 2, 3, 4 }) |w| try device.run(&cx, std.testing.allocator, w, .{}, {}, contract.Calls.go);
}

test "exchange, send and recv between two ranks of one device, and in pairs of four" {
    var d = Fake.driver();
    const cx = context(&d);
    for ([_]u32{ 2, 4 }) |w| try device.run(&cx, std.testing.allocator, w, .{}, {}, contract.Pairwise.go);
}

const OddPairs = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var mine: [64]u8 = @splat(@intCast(rank + 1));
        var got: [64]u8 = @splat(0);
        // the last rank of an odd world has no partner and goes straight on, as the GPU checks pair ranks
        const peer = rank ^ 1;
        if (peer < c.world()) {
            try c.exchange(addr(&mine), addr(&got), 64, .u8, peer, null);
            try std.testing.expectEqualSlices(u8, &@as([64]u8, @splat(@intCast(peer + 1))), &got);
        }
        var all: [64 * device.max_ranks]u8 = undefined;
        try c.allGather(addr(&mine), addr(&all), 64, .u8, null);
        for (0..c.world()) |r| try std.testing.expectEqualSlices(u8, &@as([64]u8, @splat(@intCast(r + 1))), all[r * 64 ..][0..64]);
    }
};

test "pairs at an odd world: the rank without a partner meets the others at the next collective" {
    var d = Fake.driver();
    const cx = context(&d);
    for ([_]u32{ 3, 5 }) |w| try device.run(&cx, std.testing.allocator, w, .{}, {}, OddPairs.go);
}

const BackToBack = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        if (rank > 1) return;
        for (0..200) |k| {
            var sent: [8]u8 = @splat(@truncate(k));
            if (rank == 0) {
                try c.send(addr(&sent), 8, .u8, 1, null);
            } else {
                var got: [8]u8 = @splat(0xff);
                try c.recv(addr(&got), 8, .u8, 0, null);
                try std.testing.expectEqualSlices(u8, &sent, &got);
            }
        }
    }
};

test "back-to-back sends: each recv takes the next post, never the previous one" {
    var d = Fake.driver();
    const cx = context(&d);
    for ([_]u32{ 2, 3 }) |w| try device.run(&cx, std.testing.allocator, w, .{}, {}, BackToBack.go);
}

test "a desynchronised call or lengths that differ are refused on every rank" {
    var d = Fake.driver();
    const cx = context(&d);
    for ([_]u32{ 2, 3 }) |w| try device.run(&cx, std.testing.allocator, w, .{}, {}, contract.Desync.go);
    try device.run(&cx, std.testing.allocator, 2, .{}, {}, contract.VarDesync.go);
}

test "abort frees a rank of one device waiting in a collective" {
    var d = Fake.driver();
    const cx = context(&d);
    try device.run(&cx, std.testing.allocator, 2, .{}, {}, contract.Aborts.go);
}

test "a variable-length all-gather on one device moves each rank's length, and the ranks agree once" {
    var d = Fake.driver();
    const cx = context(&d);
    for ([_]u32{ 2, 3 }) |w| {
        try device.run(&cx, std.testing.allocator, w, .{}, @as(u32, 0), contract.VarGather.go);
        try device.run(&cx, std.testing.allocator, w, .{ .var_refuse = 2 }, @as(u32, 2), contract.VarGather.go);
    }
}

const Sizes = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var a: [16]u8 = @splat(1);
        if (rank == 0) return std.testing.expectError(error.Aborted, c.send(addr(&a), 8, .u8, 1, null));
        if (rank == 1) return std.testing.expectError(error.Invalid, c.recv(addr(&a), 16, .u8, 0, null));
    }
};

test "a recv of another size than its send is refused, and the waiting sender is freed" {
    var d = Fake.driver();
    const cx = context(&d);
    for ([_]u32{ 2, 3 }) |w| try device.run(&cx, std.testing.allocator, w, .{}, {}, Sizes.go);
}

const Capture = struct {
    fn go(c: Collective, _: u32, _: void) !void {
        var a: [16]u8 = @splat(0);
        var b: [32]u8 = undefined;
        try std.testing.expectError(error.Unsupported, c.allGather(addr(&a), addr(&b), 16, .u8, null));
    }
};

test "a capturing stream is refused before any rank waits" {
    var d = Fake.driver();
    const cx = context(&d);
    Fake.capturing = true;
    defer Fake.capturing = false;
    try device.run(&cx, std.testing.allocator, 2, .{}, {}, Capture.go);
}
