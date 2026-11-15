//! Staging HIP memory on a real GPU: two ranks of one device over loopback, each on its own stream; skips without HIP.
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const sock = @import("sock.zig");
const stage = @import("stage.zig");
const tcp = @import("tcp.zig");
const Hip = @import("stage_hip.zig").Hip;
const Rig = @import("hip_rig.zig").Rig;
const Collective = collective.Collective;
const DType = collective.DType;
const DevicePtr = collective.DevicePtr;

const gpa = std.testing.allocator;
const net: tcp.Options = .{ .timeout_ns = 10 * std.time.ns_per_s };
/// 64 KiB parts: the larger calls below move in several.
const small: stage.Options = .{ .part_bytes = 64 << 10 };
const max_bytes = 1 << 21;

/// The runtime, and each rank's HIP device with the stream its proxy copies on.
const Gpu = struct {
    rig: Rig,
    hips: [2]Hip,

    fn open() !Gpu {
        var rig = Rig.open() catch return error.SkipZigTest;
        errdefer rig.close();
        var a = Hip.init(&rig.lib) catch |e| return if (e == error.Unsupported) error.SkipZigTest else e;
        errdefer a.deinit();
        return .{ .rig = rig, .hips = .{ a, Hip.init(&rig.lib) catch |e| return if (e == error.Unsupported) error.SkipZigTest else e } };
    }

    fn close(g: *Gpu) void {
        for (&g.hips) |*h| h.deinit();
        g.rig.close();
    }

    fn run(g: *Gpu, opts: stage.Options, comptime body: fn (Collective, u32, *Gpu) anyerror!void) !void {
        const devs = [2]stage.Device{ g.hips[0].device(), g.hips[1].device() };
        try stage.run(gpa, net, &devs, opts, g, body);
    }
};

/// A rank's stream and buffers: `in` and `out` on the device, `host` and `want` to fill and compare.
const Bufs = struct {
    s: collective.Stream,
    in: DevicePtr,
    out: DevicePtr,
    host: []u8,
    want: []u8,

    fn init(rig: *Rig) !Bufs {
        const s = try rig.stream();
        return .{ .s = s, .in = try rig.alloc(max_bytes), .out = try rig.alloc(2 * max_bytes), .host = try gpa.alloc(u8, 2 * max_bytes), .want = try gpa.alloc(u8, 3 * max_bytes) };
    }

    fn deinit(b: *Bufs, rig: *Rig) void {
        rig.free(b.in);
        rig.free(b.out);
        rig.destroy(b.s);
        gpa.free(b.host);
        gpa.free(b.want);
    }
};

/// Sums, gathers, trades and sends of 0 B to 1 MiB, each checked bit for bit after its stream synchronised.
const Calls = struct {
    fn go(c: Collective, rank: u32, g: *Gpu) !void {
        const rig = &g.rig;
        var b = try Bufs.init(rig);
        defer b.deinit(rig);
        const peer = rank ^ 1;
        const counts = [_]usize{ 0, 1, 7, 96, 4099, 65536 + 3, 262144 };
        for (counts, 0..) |n, k| inline for (.{ DType.f32, DType.bf16, DType.f16, DType.i32 }) |t| {
            const len = n * t.size();
            reference.fill(b.host[0..len], t, rank, k);
            try rig.put(b.in, b.host[0..len]);
            try c.allReduce(b.in, b.out, n, t, .sum, b.s);
            try c.allGather(b.in, b.out + max_bytes, n, t, b.s);
            try rig.sync(b.s);
            try c.check();
            var views: [2][]const u8 = undefined;
            for (0..2) |r| {
                reference.fill(b.want[(r + 1) * max_bytes ..][0..len], t, @intCast(r), k);
                views[r] = b.want[(r + 1) * max_bytes ..][0..len];
            }
            reference.reduce(b.want[0..len], &views, t, .sum);
            try rig.get(b.host[0..len], b.out);
            try std.testing.expectEqualSlices(u8, b.want[0..len], b.host[0..len]);
            try rig.get(b.host[0 .. 2 * len], b.out + max_bytes);
            for (0..2) |r| try std.testing.expectEqualSlices(u8, views[r], b.host[r * len ..][0..len]);
            // in place, then a trade and a send beside it
            try rig.put(b.out, views[rank]);
            try c.allReduce(b.out, b.out, n, t, .sum, b.s);
            try c.exchange(b.in, b.out + max_bytes, n, t, peer, b.s);
            if (rank == 0) try c.send(b.in, n, t, 1, b.s) else try c.recv(b.out + max_bytes + len, n, t, 0, b.s);
            try rig.sync(b.s);
            try c.check();
            try rig.get(b.host[0..len], b.out);
            try std.testing.expectEqualSlices(u8, b.want[0..len], b.host[0..len]);
            try rig.get(b.host[0..len], b.out + max_bytes);
            try std.testing.expectEqualSlices(u8, views[peer], b.host[0..len]);
            if (rank == 1) {
                try rig.get(b.host[0..len], b.out + max_bytes + len);
                try std.testing.expectEqualSlices(u8, views[0], b.host[0..len]);
            }
        };
        try c.barrier();
        try c.check();
    }
};

test "every staged call on HIP memory equals the reference, whole and in parts" {
    var g = try Gpu.open();
    defer g.close();
    try g.run(.{}, Calls.go);
    try g.run(small, Calls.go);
}

const rounds = 20;

/// The stream waits on a gate word first: every call returns while nothing has run, and then all land.
const Held = struct {
    fn go(c: Collective, rank: u32, g: *Gpu) !void {
        const rig = &g.rig;
        var b = try Bufs.init(rig);
        defer b.deinit(rig);
        const dev = g.hips[rank].device();
        const gate = try dev.vtable.alloc(dev.ptr, 64);
        defer dev.vtable.free(dev.ptr, gate);
        const word: *u64 = @ptrCast(gate.ptr);
        @atomicStore(u64, word, 0, .release);
        const n = 24576; // 96 KiB of f32: two parts a call at 64 KiB
        const len = n * 4;
        @memset(b.host[0 .. 2 * max_bytes], 0xEE);
        try rig.put(b.out, b.host[0 .. 2 * max_bytes]);
        reference.fill(b.host[0..len], .f32, rank, 5);
        try rig.put(b.in, b.host[0..len]);
        try dev.vtable.wait(dev.ptr, word, 1, b.s);
        defer @atomicStore(u64, word, 1, .release); // a failed check below still lets the stream go before cleanup
        for (0..rounds) |k| try c.allGather(b.in, b.out + k * 2 * len, n, .f32, b.s);
        sock.sleepNs(50 * std.time.ns_per_ms);
        try rig.get(b.host[0 .. rounds * 2 * len], b.out);
        try std.testing.expect(std.mem.allEqual(u8, b.host[0 .. rounds * 2 * len], 0xEE));
        @atomicStore(u64, word, 1, .release);
        try rig.sync(b.s);
        try c.check();
        try rig.get(b.host[0 .. rounds * 2 * len], b.out);
        for (0..2) |r| reference.fill(b.want[r * len ..][0..len], .f32, @intCast(r), 5);
        for (0..rounds) |k| try std.testing.expectEqualSlices(u8, b.want[0 .. 2 * len], b.host[k * 2 * len ..][0 .. 2 * len]);
    }
};

test "staged calls on a HIP stream return before the GPU reaches them" {
    var g = try Gpu.open();
    defer g.close();
    try g.run(small, Held.go);
}

/// Rank 1 aborts instead of calling; rank 0's stream must still finish its queued calls.
const Freed = struct {
    fn go(c: Collective, rank: u32, g: *Gpu) !void {
        if (rank == 1) {
            sock.sleepNs(20 * std.time.ns_per_ms);
            return c.abort();
        }
        const rig = &g.rig;
        var b = try Bufs.init(rig);
        defer b.deinit(rig);
        for (0..rounds) |k| c.allGather(b.in, b.out + k * 8192, 1024, .f32, b.s) catch |e| try std.testing.expectEqual(error.Aborted, e);
        try rig.sync(b.s);
        try std.testing.expectError(error.Aborted, c.check());
    }
};

test "an abort lets a HIP stream with staged calls finish" {
    var g = try Gpu.open();
    defer g.close();
    try g.run(small, Freed.go);
}

const Capture = struct {
    fn go(c: Collective, _: u32, g: *Gpu) !void {
        const rig = &g.rig;
        var b = try Bufs.init(rig);
        defer b.deinit(rig);
        try rig.capture(b.s);
        const r = c.allGather(b.in, b.out, 16, .f32, b.s);
        try rig.endCapture(b.s);
        try std.testing.expectError(error.Unsupported, r);
    }
};

test "a capturing HIP stream is refused" {
    var g = try Gpu.open();
    defer g.close();
    try g.run(.{}, Capture.go);
}

/// A HIP runtime whose copies queue and whose stream finishes only when told: the proxy's wait without a GPU.
const Stuck = struct {
    var finished = false;

    fn down(_: ?*anyopaque, _: ?*anyopaque, _: usize, _: collective.Stream) callconv(.c) c_int {
        return 0;
    }

    fn up(_: ?*anyopaque, _: ?*const anyopaque, _: usize, _: collective.Stream) callconv(.c) c_int {
        return 0;
    }

    fn query(_: collective.Stream) callconv(.c) c_int {
        return if (finished) 0 else 600; // hipErrorNotReady
    }

    fn hip(limit_ns: u64) Hip {
        var h: Hip = .{ .api = undefined, .limit_ns = limit_ns };
        h.api.hipMemcpyDtoHAsync = &down;
        h.api.hipMemcpyHtoDAsync = &up;
        h.api.hipStreamQuery = &query;
        return h;
    }
};

test "a proxy copy still queued past its limit fails its call instead of waiting forever" {
    var buf: [64]u8 = undefined;
    var h = Stuck.hip(5 * std.time.ns_per_ms);
    const d = h.device();
    Stuck.finished = false;
    const t0 = sock.nowNs();
    try std.testing.expectError(error.BackendFailed, d.vtable.download(d.ptr, &buf, 0x1000, buf.len));
    try std.testing.expectError(error.BackendFailed, d.vtable.upload(d.ptr, 0x1000, &buf, buf.len));
    try std.testing.expect(sock.nowNs() - t0 < 2 * std.time.ns_per_s);
    Stuck.finished = true;
    try d.vtable.download(d.ptr, &buf, 0x1000, buf.len);
}
