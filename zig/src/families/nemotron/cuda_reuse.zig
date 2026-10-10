//! Nemotron CUDA prompt states for the shared cache: the engine's caches and MTP head, copied on its stream.

const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;

/// What one kept state holds. The token position lives beside the device copy.
pub const Copy = struct {
    state: cuda.DeviceBuffer,
    head: ?cuda.DeviceBuffer = null,
    pos: usize,
    parity: usize,
    prev_keep: usize,
    rows: usize,
    head_pos: usize = 0,
};

/// The engine and head a cache save copies. The native host keeps one of these for the loaded model.
pub const Target = struct {
    e: *Engine,
    head: ?*Head,
};

/// Bytes a save at any position takes: the caches are sized for the whole window, not for `at`.
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    _ = at;
    const t: *Target = @ptrCast(@alignCast(ptr));
    var n: u64 = t.e.b.stateBytes();
    if (t.head) |h| n += h.snapshotBytes();
    return n;
}

/// Copy the live state after `at` prompt tokens. The pass is standing there.
pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const t: *Target = @ptrCast(@alignCast(ptr));
    const e = t.e;
    if (e.pos != at) return error.NotAtMark;
    if (t.head) |h| if (h.pos != at) return error.NotAtMark;
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    copy.* = .{ .state = undefined, .pos = e.pos, .parity = e.parity, .prev_keep = e.prev_keep, .rows = e.rows };
    copy.state = try e.b.snapshot(e.ops());
    errdefer copy.state.free();
    if (t.head) |h| {
        copy.head = try h.snapshot();
        copy.head_pos = h.pos;
    }
    return copy;
}

/// Make the bound sequence `saved`'s. The next prompt chunk starts at its position.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = owner;
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    const e = t.e;
    if (t.head != null and copy.head == null) return error.NoHead; // checked before any copy: nothing half restored
    try e.b.restore(e.ops(), copy.state);
    e.pos = copy.pos;
    e.parity = copy.parity;
    e.prev_keep = copy.prev_keep;
    e.rows = copy.rows;
    if (t.head) |h| try h.restore(copy.head.?, copy.head_pos);
}

/// Free a kept state's device copy.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    copy.state.free();
    if (copy.head) |*buf| buf.free();
    t.e.gpa.destroy(copy);
}
