//! Metal event boundaries publish host arrivals; a terminal release never publishes a valid payload epoch.
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");
const platform = @import("sys");
const Fence = @import("../../core/fence.zig").Fence;
pub const deadlines = @import("tp_deadline.zig");

pub const Handoff = struct {
    posted: mtl.SharedEvent,
    landed: mtl.SharedEvent,
    ready: mtl.Buffer,
    fence: Fence,
    failed: *std.atomic.Value(bool),
    stop: *std.atomic.Value(bool),
    rank: u32,
    sequence: u64 = 0,
    lock: std.atomic.Mutex = .unlocked,
    deadline: deadlines.State = .{},

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
        h.posted.deinit();
        h.landed.deinit();
        h.ready.deinit();
        h.fence.deinit();
    }

    /// The core pair_rows seam pattern, with full-width job epochs and Flash Next's existing window protocol.
    pub fn seam(h: *Handoff, enc: *mtl.ComputeEncoder, seq: u64) void {
        const cb = enc.commandBuffer();
        const mode = enc.dispatchType();
        h.fence.update(enc.*);
        enc.end();
        cb.signal(h.posted, seq);
        cb.waitFor(h.landed, seq);
        enc.* = cb.compute(mode);
        h.fence.wait(enc.*);
    }

    pub fn bind(h: *const Handoff, enc: mtl.ComputeEncoder, seq: u64, first: usize) void {
        enc.setBuffer(h.ready, 0, first);
        enc.setValue(seq, first + 1);
    }

    fn acquire(h: *Handoff) void {
        while (!h.lock.tryLock()) std.atomic.spinLoopHint();
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
        h.deadline.check(now) catch |err| {
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
        h.landed.set(std.math.maxInt(u64));
    }

    fn failLocked(h: *Handoff, seq: u64, err: anyerror) void {
        if (!h.failed.swap(true, .acq_rel)) {
            h.deadline.clear();
            h.landed.set(std.math.maxInt(u64));
            std.log.err("TP=2 rank {d}: sequence {d} failed: {s}", .{ h.rank, seq, @errorName(err) });
        }
    }

    /// A separate host thread releases GPU waiters even if the fabric is blocked inside a native call.
    pub fn watch(h: *Handoff) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        while (!h.stop.load(.acquire) and !h.failed.load(.acquire)) {
            sleep(std.time.ns_per_ms);
            h.acquire();
            defer h.lock.unlock();
            if (h.failed.load(.acquire) or h.stop.load(.acquire)) return;
            const now = fabric.words.checkedNowNs() catch |err| return h.failLocked(h.sequence, err);
            h.deadline.check(now) catch |err| return h.failLocked(h.sequence, err);
        }
    }
};

pub fn sleep(ns: u64) void {
    const ts: platform.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = platform.c.nanosleep(&ts, null);
}
