//! A lane host's part of the load ledger: its counts and counters, published at each unlock that changed them.
const std = @import("std");
const ledger = @import("load_ledger.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

const Published = struct { requests: ?ledger.Requests = null, work: ?ledger.Work = null, capacity: ?ledger.Capacity = null };

/// The host's ledger state; counters move under the host mutex, the producer and `last` under `mutex`.
pub const State = struct {
    /// The producer and the values last sent; taken after the host mutex, before the ledger's.
    mutex: std.Io.Mutex = .init,
    producer: ?ledger.Producer = null,
    /// What the ledger holds from this host; null makes the next unlock publish.
    last: ?Published = null,
    started: u64 = 0,
    requeued: u64 = 0,
    yields: u64 = 0,
};

/// Engine.load: the server's producer at attach (published at once), null at detach.
pub fn attach(ctx: *anyopaque, producer: ?ledger.Producer) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    if (producer == null) {
        h.load.mutex.lockUncancelable(h.io);
        defer h.load.mutex.unlock(h.io);
        h.load.producer = null;
        return;
    }
    h.lock();
    h.load.mutex.lockUncancelable(h.io);
    h.load.producer = producer;
    h.load.last = null;
    h.load.mutex.unlock(h.io);
    h.unlock(); // its publish sends the first values
}

/// Sends the host's groups when they changed since the last send; the caller holds the host mutex.
pub fn publish(h: *LaneHost) void {
    const s = &h.load;
    s.mutex.lockUncancelable(h.io);
    defer s.mutex.unlock(h.io);
    const p = s.producer orelse return;
    const now: Published = .{ .requests = requests(h) };
    if (s.last) |last| if (std.meta.eql(last, now)) return;
    s.last = now;
    p.publish(now.requests, now.work, now.capacity);
}

/// A background pass gave its lane up to foreground work.
pub fn yielded(h: *LaneHost) void {
    h.lock();
    defer h.unlock();
    h.load.yields += 1;
}

/// The `requests` group from the host's lists; the caller holds the host mutex.
fn requests(h: *const LaneHost) ledger.Requests {
    var waiting_tokens: u64 = 0;
    for (h.queued.items) |job| waiting_tokens += job.request.prompt.len;
    const left = filling(h);
    return .{
        .running = @intCast(h.admitted.items.len),
        .waiting = @intCast(h.queued.items.len),
        .waiting_prompt_tokens = waiting_tokens,
        .filling_left = left,
        .fill_unknown = left == null,
        .started = h.load.started,
        .requeued = h.load.requeued,
        .yields = h.load.yields,
    };
}

/// Prompt tokens left in the foreground passes filling now: 0 for none, null when the backend reports no progress.
fn filling(h: *const LaneHost) ?u64 {
    var left: u64 = 0;
    var any = false;
    for (h.admitted.items) |job| {
        if (!job.fill_on.load(.acquire) or job.request.background) continue; // as the dashboard's fill totals
        any = true;
        left += job.request.prompt.len -| job.stream.fill_at.load(.monotonic);
    }
    if (any and !h.core.backend.vtable.reports_fill) return null;
    return left;
}

test {
    _ = @import("lane_host_load_test.zig");
}
