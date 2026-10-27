//! A lane host stages disk images and indexes completed writes between rounds, and bounds the writer on a stop.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const rows = @import("prompt_cache.zig").rows;

/// A planned stop's time to finish writing what waits.
pub const stop_ns: i128 = 20 * std.time.ns_per_s;

pub const Disk = api.ImprintDisk;

/// Engine.drain, and stop's first half: no more admissions, what is left cancelled, writes finished, the thread joined.
pub fn drain(ctx: *anyopaque) void {
    drainThread(@ptrCast(@alignCast(ctx)));
}

pub fn drainThread(h: *LaneHost) void {
    h.lock();
    h.closing = true;
    h.wake.broadcast(h.io);
    h.unlock();
    if (h.thread) |t| t.join();
    h.thread = null;
}

/// Engine.imprint: an ask queued for the engine thread; false when the host is closing.
pub fn ask(ctx: *anyopaque, a: *api.ImprintAsk) bool {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.lock();
    defer h.unlock();
    if (h.closing or !h.disk.push(h.gpa, a)) return false;
    h.wake.signal(h.io);
    return true;
}

/// Every queued ask answered (between rounds, on the engine thread).
pub fn answer(h: *LaneHost) void {
    if (h.cache) |store| if (store.disk) |r| {
        if (!r.sync_writes) _ = rows.write.step(r);
    };
    h.lock();
    const list = h.disk.take(h.gpa);
    h.unlock();
    defer h.gpa.free(list);
    for (list) |a| a.answer(h.cache, .{ .spans = h.decoded_rows }); // keyed as its lookups are
}

/// Asks left when the engine stops: answered as having no disk states.
pub fn close(h: *LaneHost) void {
    h.disk.close(h.gpa);
}

/// Idle work waits; the synchronous writer also waits for the admission quiet (under the lock).
pub fn ready(h: *LaneHost) bool {
    const store = h.cache orelse return false;
    const r = store.disk orelse return false;
    if (!rows.write.ready(r)) return false;
    if (h.queued.items.len > 0 or h.admitted.items.len > 0 or h.cancels.items.len > 0 or h.closing) return false;
    return !r.sync_writes or h.disk.quiet(h.admissions, std.Io.Clock.awake.now(h.io).toNanoseconds());
}

/// Stage one queued image or advance the synchronous writer, outside the lock.
pub fn step(h: *LaneHost) void {
    _ = rows.write.step(h.cache.?.disk.?);
}

/// The engine loop's idle turn (under the lock): when `ready`, the lock let go and one block written; true if so.
pub fn idle(h: *LaneHost) bool {
    if (!ready(h)) return false;
    h.unlock();
    step(h);
    return true;
}

/// The engine thread's last act on a stop: what waits is written under `stop_ns`, and --learn's half-made cut settled.
pub fn flush(h: *LaneHost) void {
    const store = h.cache orelse return;
    if (store.imprint) |im| @import("prompt_learn.zig").settle(store, im);
    const r = store.disk orelse return;
    _ = rows.write.drain(r, rows.write.nowNs() + stop_ns);
}

/// A prompt pass's tokens and time, for the store's choice between a disk read and prefilling.
pub fn prefilled(h: *LaneHost, tokens: usize, ns: i96) void {
    const store = h.cache orelse return;
    const r = store.disk orelse return;
    r.prefilled(tokens, ns);
}
