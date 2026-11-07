//! Ling's kept prompt states: a copy of the sequence's state (KDA part, then each MLA layer's rows below its position).
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const fw = @import("forward.zig");
const weights = @import("weights.zig");

const Engine = engine.Engine;
const cache_row = @import("state.zig").cache_width * 2;

/// A kept state: its KDA part and cache rows in `buf`, and the host fields beside it.
pub const Snap = struct { buf: cuda.DeviceBuffer, pos: usize, cur: []u1 };

/// Bytes a State kept at `at` holds: conv windows, each KDA layer's committed state, the rows below `at`.
pub fn bytes(st: *const fw.State, at: usize) usize {
    return st.keptHead() + st.mla_layers * at * cache_row;
}

/// The State's KDA part into `kept` (out) or back from it: conv windows, the KDA copies `cur` names.
fn headCopy(e: *const Engine, st: *const fw.State, kept: u64, cur: []const u1, out: bool) !void {
    const conv = st.kda_layers * 3 * st.convWidth() * 2;
    if (conv > 0) try (if (out) e.copy(kept, st.conv, conv) else e.copy(st.conv, kept, conv));
    for (0..st.kda_layers) |li| {
        const k = kept + conv + li * st.rec_layer;
        try (if (out) e.copy(k, st.recAt(cur[li], li), st.rec_layer) else e.copy(st.recAt(cur[li], li), k, st.rec_layer));
    }
}

/// MLA rows below `at` between a State's caches and a kept buffer, after its head.
fn rowsCopy(e: *const Engine, st: *const fw.State, kept: u64, at: usize, out: bool) !void {
    const rows = at * cache_row;
    if (rows == 0) return;
    for (0..st.mla_layers) |mi| {
        const dst = kept + st.keptHead() + mi * rows;
        try (if (out) e.copy(dst, st.cacheAt(mi), rows) else e.copy(st.cacheAt(mi), dst, rows));
    }
}

/// A copy of the bound sequence's state under a new id (the pass standing at a mark).
pub fn keep(e: *Engine) !u64 {
    const q = e.cur;
    if (q.st.pending != 0) return error.NotCommitted;
    try e.sendOn(e.slot, .{ .kind = @backingInt(@import("tp.zig").Kind.keep_state), .a = e.next_snap }, &.{});
    var buf = try cuda.DeviceBuffer.alloc(e.d, bytes(&q.st, q.st.pos));
    errdefer if (weights.drains(e.stream)) buf.free(); // else leaked: the copy into it may be queued
    try headCopy(e, &q.st, buf.ptr, q.st.cur, true);
    try rowsCopy(e, &q.st, buf.ptr, q.st.pos, true);
    try e.stream.synchronize();
    const cur = try e.gpa.dupe(u1, q.st.cur);
    errdefer e.gpa.free(cur);
    const id = e.next_snap;
    try e.snaps.put(e.gpa, id, .{ .buf = buf, .pos = q.st.pos, .cur = cur });
    e.next_snap += 1;
    return id;
}

/// The bound sequence made kept state `id`'s: its KDA part and rows copied back.
pub fn restore(e: *Engine, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    const q = e.cur;
    try headCopy(e, &q.st, s.buf.ptr, s.cur, false);
    try rowsCopy(e, &q.st, s.buf.ptr, s.pos, false);
    @memcpy(q.st.cur, s.cur);
    q.st.pending = 0;
    q.st.prompt = false;
    try q.st.setPos(e.stream, s.pos);
    try e.stream.synchronize();
}
