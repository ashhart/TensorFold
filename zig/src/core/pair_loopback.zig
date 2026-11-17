//! The full GLM pair's real GPU producers, a host loopback, and consumers on fresh command buffers.
const std = @import("std");
const mtl = @import("metal");

fn Worker(comptime Pair: type) type {
    return struct {
        pair: *Pair,
        queue: mtl.Queue,
        h: mtl.Buffer,
        w: mtl.Buffer,
        y: mtl.Buffer,
        count: usize = 3,
        err: ?anyerror = null,

        fn run(self: *@This()) void {
            self.work() catch |err| {
                self.err = err;
            };
        }
        fn work(self: *@This()) !void {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            var enc = self.queue.commandBuffer().compute(.serial);
            for (0..self.count) |_| {
                try self.pair.addRms(&enc, .{ .buf = self.h }, .{ .buf = self.w }, .{ .buf = self.y }, 1, 6144, 1e-5);
                enc.barrier();
            }
            const last = enc.commandBuffer();
            enc.end();
            last.commit();
            last.wait();
            if (last.failure() != null) return error.GpuFailure;
        }
    };
}

pub fn ordered(comptime Pair: type, comptime Fixture: type) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const p = [_]*Pair{ try Fixture.create(device, 0), try Fixture.create(device, 1) };
    defer for (p) |pair| Fixture.drop(pair);
    const queues = [_]mtl.Queue{ try device.queue(), try device.queue() };
    defer for (queues) |q| q.deinit();
    const h = [_]mtl.Buffer{ try device.buffer(6144 * 4, 0), try device.buffer(6144 * 4, 0) };
    defer for (h) |b| b.deinit();
    const y = [_]mtl.Buffer{ try device.buffer(6144 * 4, 0), try device.buffer(6144 * 4, 0) };
    defer for (y) |b| b.deinit();
    const w = try device.buffer(6144 * 2, 0);
    defer w.deinit();
    @memset(w.slice(u16, 6144), 0x3f80);
    var workers: [2]Worker(Pair) = undefined;
    var threads: [2]?std.Thread = .{ null, null };
    // Failure releases every host waiter before its buffers and callback owner go away.
    defer {
        for (p) |pair| pair.stopFollowing();
        for (threads) |thread| if (thread) |t| t.join();
    }
    for (p, 0..) |pair, rank| {
        @memset(pair.partial.slice(f32, 6144), @floatFromInt(rank + 1));
        @memset(h[rank].slice(f32, 6144), 0);
        @memset(pair.wbuf.slice(f32, 2 * pair.l.slot / 4), 91);
        workers[rank] = .{ .pair = pair, .queue = queues[rank], .h = h[rank], .w = w, .y = y[rank] };
        threads[rank] = try std.Thread.spawn(.{}, Worker(Pair).run, .{&workers[rank]});
    }
    for (1..4) |x| {
        for (p, h) |pair, hb| {
            try std.testing.expect(pair.handoff.posted.wait(x, 1000));
            for (hb.slice(f32, 6144)) |v| try std.testing.expectEqual(@as(f32, @floatFromInt((x - 1) * 3)), v);
            try std.testing.expectEqual(x - 1, pair.handoff.ready.slice(u64, 1)[0]);
        }
        // Six seconds of cumulative peer delay exceed the GPU watchdog budget, with no GPU waiter submitted.
        std.Io.sleep(std.testing.io, .fromMilliseconds(2000), .awake) catch {};
        for (h) |hb| for (hb.slice(f32, 6144)) |v| {
            try std.testing.expectEqual(@as(f32, @floatFromInt((x - 1) * 3)), v);
        };
        // An event alone cannot substitute a previous slot's bytes for this exchange.
        for (p, 0..) |pair, rank| @memcpy(Fixture.receive(pair, x), Fixture.send(p[1 - rank], x));
        for (p) |pair| {
            _ = pair.handoff.arm(x, std.time.ns_per_s, error.PairPeerSilent) orelse return error.LoopbackFailed;
            try std.testing.expect(pair.handoff.publish(x));
        }
    }
    for (&threads, &workers, h, y) |*thread, *worker, hb, yb| {
        thread.*.?.join();
        thread.* = null;
        if (worker.err) |err| return err;
        for (hb.slice(f32, 6144)) |v| try std.testing.expectEqual(@as(f32, 9), v);
        for (yb.slice(f32, 6144)) |v| try std.testing.expectApproxEqAbs(@as(f32, 1), v, 1e-6);
    }
    try std.testing.expectEqualSlices(u8, h[0].contents()[0 .. 6144 * 4], h[1].contents()[0 .. 6144 * 4]);
}

pub fn stale(comptime Pair: type, comptime Fixture: type) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const p = try Fixture.create(device, 0);
    defer Fixture.drop(p);
    const queue = try device.queue();
    defer queue.deinit();
    const h = try device.buffer(6144 * 4, 0);
    defer h.deinit();
    const y = try device.buffer(6144 * 4, 0);
    defer y.deinit();
    const w = try device.buffer(6144 * 2, 0);
    defer w.deinit();
    @memset(w.slice(u16, 6144), 0x3f80);
    @memset(h.slice(f32, 6144), 1);
    @memset(y.slice(f32, 6144), 77);
    @memset(p.partial.slice(f32, 6144), 2);
    @memset(p.win, 0x44);
    p.x = 0x1_0000_0003;
    p.handoff.ready.slice(u64, 1)[0] = 3; // identical low word, different full epoch
    var worker: Worker(Pair) = .{ .pair = p, .queue = queue, .h = h, .w = w, .y = y, .count = 1 };
    var thread: ?std.Thread = try std.Thread.spawn(.{}, Worker(Pair).run, .{&worker});
    defer if (thread) |t| {
        p.stopFollowing();
        t.join();
    };
    try std.testing.expect(p.handoff.posted.wait(0x1_0000_0004, 1000));
    p.stopFollowing();
    thread.?.join();
    thread = null;
    const err = worker.err orelse return error.ExpectedLinkFailure;
    try std.testing.expect(err == error.PairLinkFailed or err == error.PairStopped);
    // The terminal signal cannot publish the missing epoch or start a consumer.
    for (h.slice(f32, 6144)) |v| try std.testing.expectEqual(@as(f32, 1), v);
    for (y.slice(f32, 6144)) |v| try std.testing.expectEqual(@as(f32, 77), v);
    try std.testing.expect(std.mem.allEqual(u8, p.win, 0x44));
    try std.testing.expectEqual(@as(u64, 3), p.handoff.ready.slice(u64, 1)[0]);
}
