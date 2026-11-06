//! Ling's kept prompt states: a copy of the sequence's state, or (past the native window) its KDA part, rows in place.

// The rows stay in the slot's caches until it writes below the state's position (dropped) or is freed (copied out).
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const fw = @import("forward.zig");
const weights = @import("weights.zig");

const Engine = engine.Engine;
const Seq = engine.Seq;
const cache_row = @import("state.zig").cache_width * 2;

/// A kept state: its KDA part (and with `src` null, its cache rows) in `buf` and `hd`, and the host fields beside it.
pub const Snap = struct { buf: cuda.DeviceBuffer, hd: ?cuda.DeviceBuffer, pos: usize, hd_pos: usize, cur: []u1, src: ?*Seq = null, ids: []u32 = &.{}, hd_room: usize = 0 };

/// Kept states keep their cache rows in place past the native window (a full copy does not fit there).
pub fn inplaceFor(context: usize) bool {
    return context > engine.split_from;
}

/// Bytes a State kept at `at` holds: conv windows, each KDA layer's committed state, and the rows unless in place.
pub fn bytes(st: *const fw.State, at: usize, inplace: bool) usize {
    return st.keptHead() + if (inplace) 0 else st.mla_layers * st.below(at) * cache_row;
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

/// MLA rows below `at` between a State's caches and a kept buffer of `room` positions a layer, after its head.
fn rowsCopy(e: *const Engine, st: *const fw.State, kept: u64, at: usize, room: usize, out: bool) !void {
    const rows = st.below(at) * cache_row;
    if (rows == 0) return;
    for (0..st.mla_layers) |mi| {
        const dst = kept + st.keptHead() + mi * st.below(room) * cache_row;
        try (if (out) e.copy(dst, st.cacheAt(mi), rows) else e.copy(st.cacheAt(mi), dst, rows));
    }
}

/// Each MLA layer's rows below `at` from one State's caches into another's.
fn rowsAcross(e: *const Engine, from: *const fw.State, to: *const fw.State, at: usize) !void {
    const rows = from.below(at) * cache_row;
    if (rows == 0) return;
    for (0..from.mla_layers) |mi| try e.copy(to.cacheAt(mi), from.cacheAt(mi), rows);
}

/// A copy of the bound sequence's state under a new id (the pass standing at a mark).
pub fn keep(e: *Engine) !u64 {
    const q = e.cur;
    if (q.st.pending != 0 or (q.hd != null and q.in_head != 0)) return error.NotCommitted;
    try e.sendOn(e.slot, .{ .kind = @backingInt(@import("tp.zig").Kind.keep_state), .a = e.next_snap }, &.{});
    const in = e.inplace;
    var buf = try cuda.DeviceBuffer.alloc(e.d, bytes(&q.st, q.st.pos, in));
    errdefer if (weights.drains(e.stream)) buf.free(); // else leaked: the copy into it may be queued
    try headCopy(e, &q.st, buf.ptr, q.st.cur, true);
    if (!in) try rowsCopy(e, &q.st, buf.ptr, q.st.pos, q.st.pos, true);
    var hd: ?cuda.DeviceBuffer = null;
    errdefer if (hd) |*x| if (weights.drains(e.stream)) x.free();
    if (q.hd) |h| {
        hd = try cuda.DeviceBuffer.alloc(e.d, bytes(&h, h.pos, in));
        try headCopy(e, &h, hd.?.ptr, h.cur, true);
        if (!in) try rowsCopy(e, &h, hd.?.ptr, h.pos, h.pos, true);
    }
    try e.stream.synchronize();
    const cur = try e.gpa.dupe(u1, q.st.cur);
    errdefer e.gpa.free(cur);
    const ids: []u32 = if (in and q.ids.items.len == q.st.pos) try e.gpa.dupe(u32, q.ids.items) else &.{}; // in place
    errdefer e.gpa.free(ids);
    const id = e.next_snap;
    try e.snaps.put(e.gpa, id, .{ .buf = buf, .hd = hd, .pos = q.st.pos, .hd_pos = if (q.hd) |h| h.pos else 0, .cur = cur, .src = if (in) q else null, .ids = ids });
    e.next_snap += 1;
    return id;
}

/// The bound sequence made kept state `id`'s: rows copied back, or from the slot holding them (none if it is this one).
pub fn restore(e: *Engine, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    const q = e.cur;
    try headCopy(e, &q.st, s.buf.ptr, s.cur, false);
    if (s.src) |src| {
        if (src != q) try rowsAcross(e, &src.st, &q.st, s.pos);
    } else try rowsCopy(e, &q.st, s.buf.ptr, s.pos, s.pos, false);
    @memcpy(q.st.cur, s.cur);
    q.st.pending = 0;
    q.st.prompt = false;
    try q.st.setPos(e.stream, s.pos);
    if (q.hd) |*h| {
        const hb = s.hd orelse return error.NoKeptState;
        try headCopy(e, h, hb.ptr, h.cur, false);
        if (s.src) |src| {
            if (src != q) try rowsAcross(e, &src.hd.?, h, s.hd_pos);
        } else try rowsCopy(e, h, hb.ptr, s.hd_pos, hdRoom(s), false);
        h.pending = 0;
        try h.setPos(e.stream, s.hd_pos);
    }
    q.in_head = 0;
    q.ids.clearRetainingCapacity(); // the ids below the restored position, when the state names them
    if (s.ids.len == s.pos) try q.ids.appendSlice(e.gpa, s.ids);
    try e.stream.synchronize();
    if (e.inplace and s.src == null and s.ids.len == 0) try settleInPlace(e, s, q);
}

/// A copied state (one read back from disk) restored where rows stay in place: its rows now lead `q`'s caches.
fn settleInPlace(e: *Engine, s: *Snap, q: *Seq) !void {
    var buf = try cuda.DeviceBuffer.alloc(e.d, q.st.keptHead());
    errdefer if (weights.drains(e.stream)) buf.free();
    try e.copy(buf.ptr, s.buf.ptr, q.st.keptHead());
    var hd: ?cuda.DeviceBuffer = null;
    if (q.hd) |h| {
        hd = try cuda.DeviceBuffer.alloc(e.d, h.keptHead());
        if (h.keptHead() > 0) try e.copy(hd.?.ptr, s.hd.?.ptr, h.keptHead());
    }
    try e.stream.synchronize();
    s.buf.free();
    s.buf = buf;
    if (s.hd) |*x| x.free();
    s.hd = hd;
    s.hd_room = 0;
    s.src = q;
}

/// Positions a layer the kept MTP buffer holds rows for: its own position, or a blank's room.
fn hdRoom(s: *const Snap) usize {
    return if (s.hd_room > 0) s.hd_room else s.hd_pos;
}

/// A new copied state of `at` positions under the next id, filled from disk (disk.zig), restored as any other.
pub fn blank(e: *Engine, at: usize) !u64 {
    const q = e.seqs.items[0] orelse return error.NoSequence;
    try e.sendOn(e.slot, .{ .kind = @backingInt(@import("tp.zig").Kind.blank), .a = e.next_snap, .seed = at }, &.{});
    var buf = try cuda.DeviceBuffer.alloc(e.d, bytes(&q.st, at, false));
    errdefer if (weights.drains(e.stream)) buf.free();
    var hd: ?cuda.DeviceBuffer = null;
    errdefer if (hd) |*x| if (weights.drains(e.stream)) x.free();
    if (q.hd) |h| hd = try cuda.DeviceBuffer.alloc(e.d, bytes(&h, at + 1, false));
    const cur = try e.gpa.alloc(u1, q.st.cur.len);
    errdefer e.gpa.free(cur);
    @memset(cur, 0);
    const id = e.next_snap;
    try e.snaps.put(e.gpa, id, .{ .buf = buf, .hd = hd, .pos = at, .hd_pos = at + 1, .cur = cur, .hd_room = at + 1 });
    e.next_snap += 1;
    return id;
}

/// Before a prompt chunk writes `new` from `at`: in-place states past `at` stay only if their ids match.
pub fn guardChunk(e: *Engine, q: *Seq, at: usize, new: []const u32) void {
    if (at == 0 or q.ids.items.len != at) q.ids.clearRetainingCapacity();
    const known = q.ids.items.len == at;
    var gone: [16]u64 = undefined;
    var n: usize = 0;
    var it = e.snaps.iterator();
    while (it.next()) |kv| {
        const s = kv.value_ptr;
        if (s.src != q or s.pos <= at) continue;
        const k = @min(s.pos - at, new.len);
        const same = known and s.ids.len == s.pos and std.mem.eql(u32, s.ids[0..at], q.ids.items) and std.mem.eql(u32, s.ids[at..][0..k], new[0..k]);
        if (!same and n < gone.len) {
            gone[n] = kv.key_ptr.*;
            n += 1;
        }
    }
    for (gone[0..n]) |id| e.dropState(id);
    if (n == gone.len) return guardChunk(e, q, at, new); // more than a batch: look again
    if (known) q.ids.appendSlice(e.gpa, new) catch q.ids.clearRetainingCapacity();
}

/// Before `q` writes decode rows from its position on: its in-place states standing past them are dropped.
pub fn guard(e: *Engine, q: *Seq) void {
    q.ids.shrinkRetainingCapacity(@min(q.ids.items.len, q.st.pos));
    dropWhere(e, q, false);
}

/// Every in-place state of `q` dropped (its rows could not be copied out).
fn dropWhere(e: *Engine, q: *Seq, all: bool) void {
    var gone: [16]u64 = undefined;
    while (true) {
        var n: usize = 0;
        var it = e.snaps.iterator();
        while (it.next()) |kv| {
            const s = kv.value_ptr;
            if (s.src != q) continue;
            const past = all or s.pos > q.st.pos or (if (q.hd) |h| s.hd_pos > h.pos else false);
            if (past and n < gone.len) {
                gone[n] = kv.key_ptr.*;
                n += 1;
            }
        }
        for (gone[0..n]) |id| e.dropState(id);
        if (n < gone.len) return;
    }
}

/// Before slot sequence `q` is freed: its in-place states get their rows copied out (they were always this big before).
pub fn orphan(e: *Engine, q: *Seq) void {
    copyOut(e, q) catch |err| {
        std.log.warn("kept states of a freed slot dropped, rows not copied out: {s}", .{@errorName(err)});
        dropWhere(e, q, true);
    };
}

fn copyOut(e: *Engine, q: *Seq) !void {
    var it = e.snaps.iterator();
    while (it.next()) |kv| {
        const s = kv.value_ptr;
        if (s.src != q) continue;
        var buf = try cuda.DeviceBuffer.alloc(e.d, bytes(&q.st, s.pos, false));
        errdefer if (weights.drains(e.stream)) buf.free();
        try e.copy(buf.ptr, s.buf.ptr, q.st.keptHead());
        try rowsCopy(e, &q.st, buf.ptr, s.pos, s.pos, true);
        var hd: ?cuda.DeviceBuffer = null;
        if (q.hd) |*h| {
            hd = try cuda.DeviceBuffer.alloc(e.d, bytes(h, s.hd_pos, false));
            try e.copy(hd.?.ptr, s.hd.?.ptr, h.keptHead());
            try rowsCopy(e, h, hd.?.ptr, s.hd_pos, s.hd_pos, true);
        }
        try e.stream.synchronize();
        s.buf.free();
        s.buf = buf;
        if (s.hd) |*x| x.free();
        s.hd = hd;
        s.src = null;
    }
}
