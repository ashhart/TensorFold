//! Staging over fake devices: the `Collective` contract with copies done at once, then streams that run later.
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const stage = @import("stage.zig");
const tcp = @import("tcp.zig");
const contract = @import("host_test.zig");
const Collective = collective.Collective;
const DevicePtr = collective.DevicePtr;
const Stream = collective.Stream;
const Error = collective.Error;

const gpa = std.testing.allocator;
const net: tcp.Options = .{ .timeout_ns = 5 * std.time.ns_per_s };
/// Parts of 64 bytes: the contract's calls go in many parts.
const tiny: stage.Options = .{ .part_bytes = 64 };

fn addr(b: []u8) DevicePtr {
    return @intFromPtr(b.ptr);
}

fn bytes(p: DevicePtr, n: usize) []u8 {
    return @as([*]u8, @ptrFromInt(p))[0..n];
}

/// A device whose memory is host memory and whose stream is the caller: each operation is done when it returns.
const Now = struct {
    capturing: bool = false,
    /// A stream whose words cannot be written, and a device address that cannot be copied, as device errors.
    const failing: Stream = @ptrFromInt(0xbad0);
    const unreadable: DevicePtr = 0xbad0;

    fn device(f: *Now) stage.Device {
        return .{ .ptr = f, .vtable = &.{ .download = download, .upload = upload, .write = write, .wait = wait, .capturing = capture, .alloc = alloc, .free = free } };
    }
    fn download(_: *anyopaque, host: [*]u8, dev: DevicePtr, n: usize) Error!void {
        if (dev == unreadable) return error.BackendFailed;
        @memcpy(host[0..n], bytes(dev, n));
    }
    fn upload(_: *anyopaque, dev: DevicePtr, host: [*]const u8, n: usize) Error!void {
        @memcpy(bytes(dev, n), host[0..n]);
    }
    fn write(_: *anyopaque, word: *u64, value: u64, s: Stream) Error!void {
        if (s == failing) return error.BackendFailed;
        @atomicStore(u64, word, value, .release);
    }
    fn wait(_: *anyopaque, word: *u64, value: u64, _: Stream) Error!void {
        while (@atomicLoad(u64, word, .acquire) < value) std.Thread.yield() catch {};
    }
    fn capture(ptr: *anyopaque, _: Stream) Error!bool {
        return @as(*Now, @ptrCast(@alignCast(ptr))).capturing;
    }
    fn alloc(_: *anyopaque, n: usize) Error![]align(64) u8 {
        return gpa.alignedAlloc(u8, .@"64", n) catch error.BackendFailed;
    }
    fn free(_: *anyopaque, slab: []align(64) u8) void {
        gpa.free(slab);
    }

    /// The same device for each of `world` ranks.
    fn all(f: *Now, world: usize) []const stage.Device {
        const S = struct {
            var devs: [tcp.max_ranks]stage.Device = undefined;
        };
        for (S.devs[0..world]) |*d| d.* = f.device();
        return S.devs[0..world];
    }
};

test "every call on two, three and four staged ranks equals the reference, whole and in parts" {
    var f: Now = .{};
    for ([_]u32{ 2, 3, 4 }) |w| {
        try stage.run(gpa, net, f.all(w), .{}, {}, contract.Calls.go);
        try stage.run(gpa, net, f.all(w), tiny, {}, contract.Calls.go);
    }
}

test "staged exchange, send and recv between two ranks, and in pairs of four" {
    var f: Now = .{};
    for ([_]u32{ 2, 4 }) |w| {
        try stage.run(gpa, net, f.all(w), .{}, {}, contract.Pairwise.go);
        try stage.run(gpa, net, f.all(w), tiny, {}, contract.Pairwise.go);
    }
}

test "a desynchronised staged call is refused on every rank, and abort frees a waiting peer" {
    var f: Now = .{};
    try stage.run(gpa, net, f.all(2), .{}, {}, contract.Desync.go);
    try stage.run(gpa, net, f.all(2), .{}, {}, contract.Aborts.go);
}

const KindOk = struct {
    fn go(c: Collective, _: u32, _: void) !void {
        try std.testing.expectEqual(collective.Kind.tcp, c.kind());
        try c.barrier();
    }
};

test "staging is the TCP Kind" {
    var f: Now = .{};
    try stage.run(gpa, net, f.all(2), .{}, {}, KindOk.go);
}

/// Rank 0 makes one call of 128 bytes, rank 1 two of 64: in 64-byte parts the counts agree, the part numbers do not.
const Parts = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var a: [128]u8 = @splat(1);
        var b: [128]u8 = undefined;
        const r = if (rank == 0) c.allReduce(addr(&a), addr(&b), 128, .u8, .sum, null) else blk: {
            c.allReduce(addr(&a), addr(&b), 64, .u8, .sum, null) catch |e| break :blk e;
            break :blk c.allReduce(addr(a[64..]), addr(b[64..]), 64, .u8, .sum, null);
        };
        if (r) |_| return error.ShouldHaveFailed else |e| if (e != error.Invalid and e != error.Aborted) return e;
    }
};

test "one call in two parts against two whole calls is refused, though every part's count agrees" {
    var f: Now = .{};
    try stage.run(gpa, net, f.all(2), tiny, {}, Parts.go);
}

const Opens = struct {
    fn go(_: Collective, _: u32, _: void) !void {
        return error.ShouldHaveFailed;
    }
};

test "ranks that differ in their part size refuse to open" {
    var f: Now = .{};
    const fds = try tcp.loopback(2);
    var nets = [2]tcp.Rank{ tcp.Rank.init(gpa, 0, 2, &.{ -1, fds[0][1] }, net), tcp.Rank.init(gpa, 1, 2, &.{ fds[1][0], -1 }, net) };
    defer for (&nets) |*n| n.deinit();
    const Open = struct {
        fn go(n: *tcp.Rank, d: stage.Device, part: usize, out: *?anyerror) void {
            var r: stage.Rank = undefined;
            r.open(n, d, .{ .part_bytes = part }) catch |e| {
                out.* = e;
                return;
            };
            r.deinit();
        }
    };
    var errs: [2]?anyerror = .{ null, null };
    const t = try std.Thread.spawn(.{}, Open.go, .{ &nets[1], f.device(), 128, &errs[1] });
    Open.go(&nets[0], f.device(), 64, &errs[0]);
    t.join();
    try std.testing.expectEqual(@as(?anyerror, error.Invalid), errs[0]);
    try std.testing.expectEqual(@as(?anyerror, error.Invalid), errs[1]);
    try std.testing.expectError(error.Invalid, stage.run(gpa, net, f.all(2), .{ .part_bytes = 8 }, {}, Opens.go));
}

const Capture = struct {
    fn go(c: Collective, _: u32, _: void) !void {
        var a: [16]u8 = @splat(0);
        var b: [32]u8 = undefined;
        try std.testing.expectError(error.Unsupported, c.allGather(addr(&a), addr(&b), 16, .u8, null));
    }
};

test "a capturing stream is refused before anything is queued" {
    var f: Now = .{ .capturing = true };
    try stage.run(gpa, net, f.all(2), .{}, {}, Capture.go);
}

/// No variable-length gather of its own: `allGatherV` moves whole strides, and the ranks agree it does.
const Whole = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        try std.testing.expect(!c.varFits(200));
        try std.testing.expect(!try c.varAgreed());
        var ins: [2][200]u8 = undefined;
        for (0..2) |r| reference.fill(&ins[r], .u8, @intCast(r), 11);
        var out: [400]u8 = @splat(0xEE);
        const lens = [2]i32{ 17, 200 };
        try c.allGatherV(addr(&ins[rank]), addr(&out), 200, @intFromPtr(&lens), .u8, null);
        try std.testing.expectEqualSlices(u8, &ins[0], out[0..200]);
        try std.testing.expectEqualSlices(u8, &ins[1], out[200..]);
    }
};

test "a staged variable-length all-gather moves whole strides" {
    var f: Now = .{};
    try stage.run(gpa, net, f.all(2), tiny, {}, Whole.go);
}

/// Rank 1's stream cannot take its word (`host`), or its proxy cannot copy the input (`proxy`).
const Broken = struct {
    fn go(c: Collective, rank: u32, proxy: bool) !void {
        var a: [16]u8 = @splat(0);
        var b: [32]u8 = undefined;
        if (rank == 1) {
            const r = if (proxy) c.allGather(Now.unreadable, addr(&b), 16, .u8, null) else c.allGather(addr(&a), addr(&b), 16, .u8, Now.failing);
            try std.testing.expectError(error.BackendFailed, r);
            return std.testing.expectError(error.BackendFailed, c.check());
        }
        const r = c.allGather(addr(&a), addr(&b), 16, .u8, null);
        if (r) |_| return error.ShouldHaveFailed else |e| if (e != error.Aborted and e != error.PeerFailed) return e;
    }
};

test "a device call that fails, on the stream or in the proxy, is BackendFailed here and ends the peer's call" {
    var f: Now = .{};
    try stage.run(gpa, net, f.all(2), .{}, false, Broken.go);
    try stage.run(gpa, net, f.all(2), .{}, true, Broken.go);
}

test {
    _ = @import("stage_later_test.zig");
}
