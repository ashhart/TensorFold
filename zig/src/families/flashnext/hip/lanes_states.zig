//! The prompt cache's hooks for Flash Next's lanes: a kept state's bytes, its save, restore and drop on every rank.
const std = @import("std");
const cache = @import("cache.zig");
const snapshot = @import("snapshot.zig");
const worker = @import("worker.zig");
const lanes = @import("lanes.zig");
const pc = @import("engine_api").prompt_cache;

const Fx = lanes.Fx;
const Lane = lanes.Lane;

pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const f: *Fx = @ptrCast(@alignCast(ptr));
    var buf: [64]bool = undefined;
    const kinds = f.e.kinds(&buf) catch return std.math.maxInt(u64);
    const main = cache.Layout.of(f.e.weights.model.dims, kinds, 64, f.e.act).stateBytes(at);
    const head = if (f.e.weights.model.mtp) |h|
        cache.Layout.of(h.dims, &.{false}, 64, f.e.act).stateBytes(at)
    else
        0;
    return main + head;
}

pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
    const f = Fx.of(ptr);
    const lane: *Lane = @ptrCast(@alignCast(owner orelse return error.NoOwner));
    const saved = try snapshot.save(
        f.gpa,
        &f.e.driver,
        f.e.stream,
        &lane.caches,
        if (lane.head) |*h| h else null,
        at,
        f.e.cfg.index_ratio,
    );
    saved.slot = f.next_slot;
    f.next_slot += 1;
    if (f.link != null) try f.send(&.{ @backingInt(worker.Op.snap), lane.id, saved.slot, at });
    return saved;
}

/// Every rank resumes from the state, or every rank starts the prompt over; a group that cannot do either stops.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
    const f = Fx.of(ptr);
    const lane: *Lane = @ptrCast(@alignCast(owner orelse return error.NoOwner));
    const s: *snapshot.Saved = @ptrCast(@alignCast(saved));
    // a follower that got the step waits for every rank's word on it: a rank 0 that cannot send stops too
    if (f.link != null) f.send(&.{ @backingInt(worker.Op.restore), lane.id, s.slot }) catch |err| {
        f.e.stopped = true;
        return err;
    };
    snapshot.restoreAgreed(
        f.gpa,
        &f.e.driver,
        f.e.stream,
        s,
        &lane.caches,
        if (lane.head) |*h| h else null,
        f.e.cfg.index_ratio,
        f.e.eos,
        f.link,
    ) catch |err| {
        if (err == error.GroupStopped) f.e.stopped = true;
        return err;
    };
}

pub fn drop(ptr: *anyopaque, saved: pc.Saved) void {
    const f: *Fx = @ptrCast(@alignCast(ptr));
    const s: *snapshot.Saved = @ptrCast(@alignCast(saved));
    if (f.link != null) f.send(&.{ @backingInt(worker.Op.drop), s.slot }) catch {};
    snapshot.drop(f.gpa, s);
}
