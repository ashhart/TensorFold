//! Inkling 2 Mini's kept prompt states: a stream's exact state at a length and, with DSpark, its draft head's context.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const fl = tf.inkling2_metal;
const Target = @import("inkling2_backend.zig").Target;

/// What `Model.save` returns: one stream's exact state at a length (3 loop passes of KV and convolution history).
pub const Snap = @typeInfo(@typeInfo(@TypeOf(fl.Model.save)).@"fn".return_type.?).error_union.payload;

pub const Kept = struct {
    target: *const Target,
    snap: Snap,
    /// The head's context after the same rows; null: the stream had none whole (no drafts, or it began past a prefix).
    head: ?fl.draft.Snapshot,
};

/// The prompt cache's copies of a stream's state at a chunk end; the prompt pass restores them itself.
pub fn snapshots(t: *Target) api.prompt_cache.Snapshots {
    return .{ .ptr = t, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .charged = charged } };
}

fn of(ptr: *anyopaque) *Target {
    return @ptrCast(@alignCast(ptr));
}

fn headBytes(t: *const Target, at: u32) u64 {
    return if (t.head != null) fl.draft.Snapshot.bytes(t.model.config.mtp.?, at) else 0;
}

fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t = of(ptr);
    return Snap.bytes(&t.model.config, at) + headBytes(t, at);
}

fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const t = of(ptr);
    const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const state = try t.stateOf(stream);
    if (state.len != at) return error.SnapshotOutOfStep;
    const kept = try t.gpa.create(Kept);
    errdefer t.gpa.destroy(kept);
    kept.* = .{ .target = t, .snap = try t.model.save(state), .head = null };
    errdefer kept.snap.deinit();
    if (t.head) |h| kept.head = try h.save(stream, at);
    return kept;
}

fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
    return error.BackendRestores; // the prompt pass restores, before its first chunk
}

/// The bytes this copy holds: the head's part only where the stream had one.
fn charged(ptr: *anyopaque, saved: *anyopaque) u64 {
    const t = of(ptr);
    const kept: *const Kept = @ptrCast(@alignCast(saved));
    return Snap.bytes(&t.model.config, kept.snap.len) + if (kept.head != null) headBytes(t, kept.snap.len) else 0;
}

fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const kept: *Kept = @ptrCast(@alignCast(saved));
    kept.snap.deinit();
    if (kept.head) |*h| h.deinit();
    of(ptr).gpa.destroy(kept);
}
