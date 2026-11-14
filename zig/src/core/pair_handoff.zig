//! Command-buffer handoff and host publication for a pair; peer waits run only on the host.
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");
const Fence = @import("fence.zig").Fence;
pub const deadlines = @import("pair_deadline.zig");

pub const Handoff = struct {
    posted: mtl.SharedEvent,
    landed: mtl.SharedEvent,
    ready: mtl.Buffer,
    fence: Fence,
    failed: *std.atomic.Value(bool),
    stop: *std.atomic.Value(bool),
    rank: u32,
    sequence: u64 = 0,
    gpu_seconds: f64 = 0,
    lock: std.atomic.Mutex = .unlocked,
    deadline: deadlines.State = .{},
    native: deadlines.State = .{},

    pub fn init(device: mtl.Device, failed: *std.atomic.Value(bool), stop: *std.atomic.Value(bool), rank: u32) !Handoff {
        const posted = try device.sharedEvent();
        errdefer posted.deinit();
        const landed = try device.sharedEvent();
        errdefer landed.deinit();
        const ready = try device.buffer(16, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer ready.deinit();
        @memset(ready.contents()[0..16], 0);
        return .{ .posted = posted, .landed = landed, .ready = ready, .fence = try Fence.init(device), .failed = failed, .stop = stop, .rank = rank };
    }

    pub fn deinit(h: *Handoff) void {
        h.fence.deinit();
        h.ready.deinit();
        h.landed.deinit();
        h.posted.deinit();
    }

    /// Retire this producer before the host waits for its peer, then start the consumer in a new command buffer.
    pub fn seam(h: *Handoff, enc: *mtl.ComputeEncoder, seq: u64) !void {
        const cb = enc.commandBuffer();
        const mode = enc.dispatchType();
        const queue: mtl.Queue = .{ .id = mtl.objc.msg(mtl.objc.Id, cb.id, "commandQueue", .{}) };
        h.fence.update(enc.*);
        enc.end();
        cb.commit();
        cb.wait();
        h.gpu_seconds += cb.gpuSeconds();
        if (cb.failure()) |msg| {
            h.fail(seq, error.GpuFailure);
            std.log.err("pair rank {d}: producer GPU command failed: {s}", .{ h.rank, msg });
            return error.GpuFailure;
        }
        h.posted.set(seq);
        try h.waitPeer(seq);
        enc.* = queue.commandBuffer().compute(mode);
        h.fence.wait(enc.*);
    }

    fn waitPeer(h: *Handoff, seq: u64) !void {
        const now = fabric.words.checkedNowNs() catch |err| {
            h.fail(seq, err);
            return err;
        };
        const until = now +| (deadlines.send_budget_ns + deadlines.arrival_budget_ns + std.time.ns_per_s);
        while (true) {
            if (h.failed.load(.acquire)) return error.PairLinkFailed;
            if (h.stop.load(.acquire)) return error.PairStopped;
            if (h.landed.value() >= seq) {
                h.acquire();
                defer h.lock.unlock();
                if (h.failed.load(.acquire)) return error.PairLinkFailed;
                if (h.stop.load(.acquire)) return error.PairStopped;
                if (h.ready.slice(u64, 1)[0] != seq) {
                    h.failLocked(seq, error.PairOutOfStep);
                    return error.PairOutOfStep;
                }
                return;
            }
            const at = fabric.words.checkedNowNs() catch |err| {
                h.fail(seq, err);
                return err;
            };
            if (at >= until) {
                h.fail(seq, error.PairPeerSilent);
                return error.PairPeerSilent;
            }
            fabric.words.pause(20_000);
        }
    }

    pub fn bind(h: *const Handoff, enc: mtl.ComputeEncoder, seq: u64, first: usize) void {
        enc.setBuffer(h.ready, 0, first);
        enc.setValue(seq, first + 1);
    }

    fn acquire(h: *Handoff) void {
        while (!h.lock.tryLock()) std.atomic.spinLoopHint();
    }

    fn check(h: *const Handoff, now: u64) !void {
        try h.deadline.check(now);
        try h.native.check(now);
    }

    /// Heartbeat native calls have a separate watchdog and cannot replace an exchange's deadline.
    pub fn armNative(h: *Handoff, until: u64) bool {
        h.acquire();
        defer h.lock.unlock();
        if (h.failed.load(.acquire)) return false;
        const now = fabric.words.checkedNowNs() catch |err| {
            h.failLocked(h.sequence, err);
            return false;
        };
        h.check(now) catch |err| {
            h.failLocked(h.sequence, err);
            return false;
        };
        if (now >= until) {
            h.failLocked(h.sequence, error.PairHeartbeatTimeout);
            return false;
        }
        _ = h.native.arm(now, until - now, error.PairHeartbeatTimeout) catch |err| {
            h.failLocked(h.sequence, err);
            return false;
        };
        return true;
    }

    pub fn finishNative(h: *Handoff) bool {
        h.acquire();
        defer h.lock.unlock();
        if (h.failed.load(.acquire)) return false;
        const now = fabric.words.checkedNowNs() catch |err| {
            h.failLocked(h.sequence, err);
            return false;
        };
        h.check(now) catch |err| {
            h.failLocked(h.sequence, err);
            return false;
        };
        h.native.clear();
        return true;
    }

    pub fn arm(h: *Handoff, seq: u64, budget: u64, expired: anyerror) ?u64 {
        h.acquire();
        defer h.lock.unlock();
        if (h.failed.load(.acquire)) return null;
        h.sequence = seq;
        const now = fabric.words.checkedNowNs() catch |err| {
            h.failLocked(seq, err);
            return null;
        };
        h.check(now) catch |err| {
            h.failLocked(seq, err);
            return null;
        };
        return h.deadline.arm(now, budget, expired) catch |err| {
            h.failLocked(seq, err);
            return null;
        };
    }

    pub fn publish(h: *Handoff, seq: u64) bool {
        h.acquire();
        defer h.lock.unlock();
        if (h.failed.load(.acquire)) return false;
        const now = fabric.words.checkedNowNs() catch |err| {
            h.failLocked(seq, err);
            return false;
        };
        h.check(now) catch |err| {
            h.failLocked(seq, err);
            return false;
        };
        h.ready.slice(u64, 1)[0] = seq;
        h.landed.set(seq);
        h.deadline.clear();
        return true;
    }

    pub fn fail(h: *Handoff, seq: u64, err: anyerror) void {
        h.acquire();
        defer h.lock.unlock();
        h.failLocked(seq, err);
    }

    pub fn cancel(h: *Handoff) void {
        h.acquire();
        defer h.lock.unlock();
        h.failed.store(true, .release);
        h.deadline.clear();
        h.native.clear();
        h.landed.set(std.math.maxInt(u64));
    }

    fn failLocked(h: *Handoff, seq: u64, err: anyerror) void {
        if (!h.failed.swap(true, .acq_rel)) {
            h.deadline.clear();
            h.native.clear();
            h.landed.set(std.math.maxInt(u64));
            std.log.err("pair rank {d}: exchange {d} failed: {s}", .{ h.rank, seq, @errorName(err) });
        }
    }

    /// Independent of native transport calls, so even a blocked send cannot trap the host handoff.
    pub fn watch(h: *Handoff) void {
        _ = @import("sys").c.pthread_set_qos_class_self_np(.USER_INTERACTIVE, 0);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        while (!h.stop.load(.acquire) and !h.failed.load(.acquire)) {
            fabric.words.pause(std.time.ns_per_ms);
            h.acquire();
            defer h.lock.unlock();
            const now = fabric.words.checkedNowNs() catch |err| return h.failLocked(h.sequence, err);
            h.check(now) catch |err| return h.failLocked(h.sequence, err);
        }
    }
};

test "pair deadline: heartbeat native expiry cannot hide behind an exchange deadline" {
    var h: Handoff = undefined;
    h.deadline = .{};
    h.native = .{};
    _ = try h.deadline.arm(100, 100, error.PairPeerSilent);
    _ = try h.native.arm(100, 10, error.PairHeartbeatTimeout);
    try h.check(109);
    try std.testing.expectError(error.PairHeartbeatTimeout, h.check(110));
    try std.testing.expectEqual(@as(u64, 200), h.deadline.pending.?.deadline);
    try std.testing.expectEqual(@as(u64, 110), h.native.pending.?.deadline);
}

test "pair deadline: an exchange expiry remains active during a heartbeat native call" {
    var h: Handoff = undefined;
    h.deadline = .{};
    h.native = .{};
    _ = try h.deadline.arm(100, 10, error.PairPeerSilent);
    _ = try h.native.arm(100, 100, error.PairHeartbeatTimeout);
    try std.testing.expectError(error.PairPeerSilent, h.check(110));
    try std.testing.expectEqual(@as(u64, 110), h.deadline.pending.?.deadline);
}
