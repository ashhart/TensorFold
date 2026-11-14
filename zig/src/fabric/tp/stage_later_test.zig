//! Staging over streams that run later: calls return before their bytes move, and a failure still frees a held stream.
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const sock = @import("sock.zig");
const stage = @import("stage.zig");
const tcp = @import("tcp.zig");
const Collective = collective.Collective;
const DevicePtr = collective.DevicePtr;
const Stream = collective.Stream;
const Error = collective.Error;

const gpa = std.testing.allocator;
const net: tcp.Options = .{ .timeout_ns = 5 * std.time.ns_per_s };
const tiny: stage.Options = .{ .part_bytes = 64 };

fn addr(b: []u8) DevicePtr {
    return @intFromPtr(b.ptr);
}

/// A stream: its own thread runs the words queued on it in order, once it is no longer held; copies are the proxy's.
const Later = struct {
    const Item = union(enum) {
        write: struct { word: *u64, value: u64 },
        wait: struct { word: *u64, value: u64 },
    };
    items: [1 << 12]Item = undefined,
    head: std.atomic.Value(usize) = .init(0),
    tail: std.atomic.Value(usize) = .init(0),
    held: std.atomic.Value(bool) = .init(true),
    quit: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn of(s: Stream) *Later {
        return @ptrCast(@alignCast(s.?));
    }

    fn stream(l: *Later) Stream {
        return @ptrCast(l);
    }

    fn push(l: *Later, it: Item) void {
        const t = l.tail.raw;
        while (t - l.head.load(.acquire) >= l.items.len) std.Thread.yield() catch {};
        l.items[t % l.items.len] = it;
        l.tail.store(t + 1, .release);
    }

    fn loop(l: *Later) void {
        while (true) {
            const h = l.head.raw;
            if (l.held.load(.acquire) or h == l.tail.load(.acquire)) {
                if (l.quit.load(.acquire)) return;
                std.Thread.yield() catch {};
                continue;
            }
            switch (l.items[h % l.items.len]) {
                .write => |w| @atomicStore(u64, w.word, w.value, .release),
                .wait => |w| while (@atomicLoad(u64, w.word, .acquire) < w.value) std.Thread.yield() catch {},
            }
            l.head.store(h + 1, .release);
        }
    }

    fn sync(l: *Later) void {
        while (l.head.load(.acquire) != l.tail.load(.acquire)) std.Thread.yield() catch {};
    }

    const devs = [2]stage.Device{ device(), device() };

    fn device() stage.Device {
        return .{ .ptr = undefined, .vtable = &.{ .download = download, .upload = upload, .write = write, .wait = wait, .capturing = capture, .alloc = alloc, .free = free } };
    }
    fn download(_: *anyopaque, host: [*]u8, dev: DevicePtr, n: usize) Error!void {
        @memcpy(host[0..n], @as([*]const u8, @ptrFromInt(dev))[0..n]);
    }
    fn upload(_: *anyopaque, dev: DevicePtr, host: [*]const u8, n: usize) Error!void {
        @memcpy(@as([*]u8, @ptrFromInt(dev))[0..n], host[0..n]);
    }
    fn write(_: *anyopaque, word: *u64, value: u64, s: Stream) Error!void {
        of(s).push(.{ .write = .{ .word = word, .value = value } });
    }
    fn wait(_: *anyopaque, word: *u64, value: u64, s: Stream) Error!void {
        of(s).push(.{ .wait = .{ .word = word, .value = value } });
    }
    fn capture(_: *anyopaque, _: Stream) Error!bool {
        return false;
    }
    fn alloc(_: *anyopaque, n: usize) Error![]align(64) u8 {
        return gpa.alignedAlloc(u8, .@"64", n) catch error.BackendFailed;
    }
    fn free(_: *anyopaque, slab: []align(64) u8) void {
        gpa.free(slab);
    }
};

const rounds = 10;
const count = 96;

/// Ten rounds of a sum, a gather and a trade queued on a held stream, each moved in 64-byte parts.
const Queued = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var l: Later = .{};
        l.thread = try std.Thread.spawn(.{}, Later.loop, .{&l});
        defer {
            l.held.store(false, .release);
            l.sync();
            l.quit.store(true, .release);
            l.thread.?.join();
        }
        const peer = rank ^ 1;
        var ins: [rounds][count * 4]u8 = undefined;
        var sums: [rounds][count * 4]u8 = @splat(@splat(0xEE));
        var all: [rounds][2 * count * 4]u8 = @splat(@splat(0xEE));
        var trades: [rounds][count * 4]u8 = @splat(@splat(0xEE));
        for (0..rounds) |k| {
            reference.fill(&ins[k], .f32, rank, k);
            try c.allReduce(addr(&ins[k]), addr(&sums[k]), count, .f32, .sum, l.stream());
            try c.allGather(addr(&ins[k]), addr(&all[k]), count, .f32, l.stream());
            try c.exchange(addr(&ins[k]), addr(&trades[k]), count, .f32, peer, l.stream());
        }
        // every call returned while its stream was held, and a while later still no byte has moved
        sock.sleepNs(50 * std.time.ns_per_ms);
        for (0..rounds) |k| {
            try std.testing.expect(std.mem.allEqual(u8, &sums[k], 0xEE));
            try std.testing.expect(std.mem.allEqual(u8, &all[k], 0xEE));
            try std.testing.expect(std.mem.allEqual(u8, &trades[k], 0xEE));
        }
        l.held.store(false, .release);
        l.sync();
        try c.check();
        for (0..rounds) |k| {
            var each: [2][count * 4]u8 = undefined;
            for (0..2) |r| reference.fill(&each[r], .f32, @intCast(r), k);
            var want: [count * 4]u8 = undefined;
            reference.reduce(&want, &[_][]const u8{ &each[0], &each[1] }, .f32, .sum);
            try std.testing.expectEqualSlices(u8, &want, &sums[k]);
            try std.testing.expectEqualSlices(u8, &(each[0] ++ each[1]), &all[k]);
            try std.testing.expectEqualSlices(u8, &each[peer], &trades[k]);
        }
    }
};

test "staged calls return before their stream runs, and the stream then gives every one the reference's bytes" {
    try stage.run(gpa, net, &Later.devs, tiny, {}, Queued.go);
}

/// Rank 0 queues calls on a held stream; rank 1 aborts instead of calling. The stream must still drain.
const Freed = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        if (rank == 1) {
            sock.sleepNs(20 * std.time.ns_per_ms);
            return c.abort();
        }
        var l: Later = .{};
        l.thread = try std.Thread.spawn(.{}, Later.loop, .{&l});
        defer {
            l.quit.store(true, .release);
            l.thread.?.join();
        }
        var a: [count * 4]u8 = @splat(1);
        var b: [rounds][2 * count * 4]u8 = undefined;
        for (0..rounds) |k| c.allGather(addr(&a), addr(&b[k]), count, .f32, l.stream()) catch |e| try std.testing.expectEqual(error.Aborted, e);
        sock.sleepNs(50 * std.time.ns_per_ms);
        l.held.store(false, .release);
        l.sync();
        try std.testing.expectError(error.Aborted, c.check());
    }
};

test "an abort while staged calls wait on a held stream lets the stream drain" {
    try stage.run(gpa, net, &Later.devs, tiny, {}, Freed.go);
}
