//! Retained target features let a resumed DFlash stream rebuild its full historical context.
const std = @import("std");
const taps = @import("cuda_prompt_taps.zig");

pub const Layout = struct {
    rows: usize = 0,
    row_bytes: usize = 0,
    taps_hash: u64 = 0,
    pub fn bytes(self: Layout) usize {
        return self.rows * self.row_bytes;
    }
};

pub fn current(e: anytype) Layout {
    if (e.n_taps == 0 or !e.prompt_taps.cache) return .{};
    return .{ .rows = if (e.prompt_taps.buffer != null) e.prompt_taps.rows else e.tap_capacity, .row_bytes = e.n_taps * e.g.hidden * 2, .taps_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(e.taps[0..e.n_taps])) };
}

pub fn same(e: anytype, saved: Layout) bool {
    return std.meta.eql(current(e), saved);
}

/// A save includes only a proven current ring, never another stream's shared prompt features.
pub fn source(e: anytype, c: anytype, at: usize, layout: Layout) !u64 {
    if (!same(e, layout)) return error.BadReuseFeatures;
    if (layout.bytes() == 0) return 0;
    if (at == 0 or c.len != at or c.written != at) return error.NotAtMark;
    if (e.prompt_taps.buffer) |b| {
        const held = e.prompt_taps.held orelse return error.RowsNotHeld;
        const n = @min(at, layout.rows);
        if (e.prompt_taps.active or held.owner != @intFromPtr(c) or held.epoch != c.tap_epoch or
            !held.range.contains(at - n, n) or b.len < layout.bytes()) return error.RowsNotHeld;
        return b.ptr;
    }
    const b = c.tap orelse return error.NoTaps;
    if (b.len < layout.bytes()) return error.BadReuseFeatures;
    return b.ptr;
}

/// Invalidate advertised rows before an asynchronous restore can overwrite the old shared ring.
pub fn destination(e: anytype, c: anytype, layout: Layout) !u64 {
    if (!same(e, layout)) return error.BadReuseFeatures;
    if (layout.bytes() == 0) return 0;
    if (e.prompt_taps.active) return error.BadPromptTapState;
    c.tap_begin = 0;
    c.tap_end = 0;
    if (e.prompt_taps.buffer) |b| {
        if (b.len < layout.bytes()) return error.BadReuseFeatures;
        e.prompt_taps.held = null;
        e.prompt_taps.previous = null;
        return b.ptr;
    }
    const b = c.tap orelse return error.NoTaps;
    if (b.len < layout.bytes()) return error.BadReuseFeatures;
    return b.ptr;
}

/// Publish only after the caller's existing completion wait proves the copied features are ready.
pub fn restored(e: anytype, c: anytype, at: usize, layout: Layout) !void {
    if (!same(e, layout)) return error.BadReuseFeatures;
    if (layout.bytes() == 0) return;
    if (e.prompt_taps.buffer != null) return taps.restored(e, c, at);
    c.tap_begin = at -| layout.rows;
    c.tap_end = at;
}
