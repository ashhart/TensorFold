//! The prompt cache's hooks for Flash Next's lanes: a kept state's bytes, its save, restore and drop on every rank.
const std = @import("std");
const cache = @import("cache.zig");
const snapshot = @import("snapshot.zig");
const worker = @import("worker.zig");
const lanes = @import("lanes.zig");
const Stream = @import("lanes").Stream;
const pc = @import("engine_api").prompt_cache;

const Fx = lanes.Fx;
const Lane = lanes.Lane;

/// A prompt chunk's kept rows not saved yet: its marks and the row at its end.
pub const Keeps = struct {
    lane: *Lane,
    s: *Stream,
    marks: [cache.max_window]lanes.Marked,
    n: usize,
    live: ?usize,
};

/// The keeps a prompt chunk left (Fx.keeps) saved, before any step that writes a stream's caches or window.
pub fn settle(f: *Fx) void {
    const k = f.keeps orelse return;
    f.keeps = null;
    if (f.lanes.get(k.s) != k.lane) return; // its stream gone without a release: nothing to save from
    const hook = k.s.reuse.hook orelse return;
    for (k.marks[0..k.n]) |mk| {
        f.marked = mk;
        hook.at(hook.ptr, k.s, @intCast(mk.at));
    }
    f.marked = null; // a kept row at the chunk's end saves the live caches
    if (k.live) |at| hook.at(hook.ptr, k.s, @intCast(at));
}

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
    const stream: *Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const f = Fx.of(ptr);
    const lane = f.lanes.get(stream) orelse return error.UnknownStream;
    const head = if (lane.head) |*h| h else null;
    const saved = if (f.marked) |mk| if (mk.at == at)
        try snapshot.saveMarked(f.gpa, f.copy(), f.e.stream, &lane.caches, head, at, f.e.cfg.index_ratio, mk.side, mk.history)
    else
        return error.SnapshotNotAtMark else if ((if (lane.fill) |fill| fill.at else stream.prompt_len) != at)
        return error.NotAtMark
    else
        try snapshot.save(f.gpa, f.copy(), f.e.stream, &lane.caches, head, at, f.e.cfg.index_ratio);
    saved.slot = f.next_slot;
    f.next_slot += 1;
    if (f.link != null) try f.send(&.{ @backingInt(worker.Op.snap), lane.id, saved.slot, at });
    return saved;
}

/// Every rank resumes from the state, or every rank starts the prompt over; a group that cannot do either stops.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
    const stream: *Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const f = Fx.of(ptr);
    const lane = f.lanes.get(stream) orelse return error.UnknownStream;
    const s: *snapshot.Saved = @ptrCast(@alignCast(saved));
    // a follower that got the step waits for every rank's word on it: a rank 0 that cannot send stops too
    if (f.link != null) f.send(&.{ @backingInt(worker.Op.restore), lane.id, s.slot }) catch |err| {
        f.e.stopped = true;
        return err;
    };
    snapshot.restoreAgreed(
        f.gpa,
        f.copy(),
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
