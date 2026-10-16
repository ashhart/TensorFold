//! Qwen3.8-27B CUDA prompt states for the shared cache: DeltaNet recurrences, conv windows and attention rows on host.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const c = @import("shape.zig");
const st = @import("state.zig");
const Engine = @import("engine.zig").Engine;

/// One kept state: the copy and the position it holds.
pub const Copy = struct {
    host: cuda.HostBuffer,
    at: usize,
};

/// The engine and the streams' states a save copies. The native host keeps one of these for the loaded model.
pub const Target = struct {
    e: *Engine,
    seqs: *std.AutoHashMapUnmanaged(*lanes.Stream, *st.Seq),
};

/// One piece of a stream's state: its device address and length.
const Piece = struct { dev: u64, len: usize };

/// A state at `at`'s pieces in copy order (DeltaNet state, conv windows, `at` K/V rows, kv8 scales); seq null: lengths.
fn pieces(g: c.Geometry, kv8: bool, seq: ?*const st.Seq, at: usize, out: []Piece) []Piece {
    var n: usize = 0;
    const row: usize = g.kvInner() * @as(usize, if (kv8) 1 else 2);
    for (0..g.layers) |i| {
        if (c.linear(i)) {
            out[n] = .{ .dev = if (seq) |s| s.rec[i] else 0, .len = g.deltaBytes() };
            out[n + 1] = .{ .dev = if (seq) |s| s.conv[i] else 0, .len = g.convBytes() };
            n += 2;
            continue;
        }
        out[n] = .{ .dev = if (seq) |s| s.keys[i] else 0, .len = at * row };
        out[n + 1] = .{ .dev = if (seq) |s| s.values[i] else 0, .len = at * row };
        n += 2;
        if (kv8) {
            out[n] = .{ .dev = if (seq) |s| s.kscale[i] else 0, .len = at * g.kv_heads * 4 };
            out[n + 1] = .{ .dev = if (seq) |s| s.vscale[i] else 0, .len = at * g.kv_heads * 4 };
            n += 2;
        }
    }
    return out[0..n];
}

const max_pieces = 4 * 128;

fn total(g: c.Geometry, kv8: bool, at: usize) u64 {
    var buf: [max_pieces]Piece = undefined;
    var n: u64 = 0;
    for (pieces(g, kv8, null, at, &buf)) |p| n += p.len;
    return n;
}

/// Host bytes a save at `at` takes.
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    return total(t.e.w.g, t.e.kv8, at);
}

/// Copy the stream's state after `at` prompt tokens. Its pass stands there.
pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const seq = t.seqs.get(s) orelse return error.UnknownStream;
    if (seq.pos != at or seq.rope_delta != 0) return error.NotAtMark;
    const e = t.e;
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    copy.* = .{ .host = try cuda.HostBuffer.alloc(e.ctx.d, total(e.w.g, e.kv8, at)), .at = at };
    errdefer copy.host.free();
    var buf: [max_pieces]Piece = undefined;
    var off: usize = 0;
    for (pieces(e.w.g, e.kv8, seq, at, &buf)) |p| {
        try e.ops().download(copy.host.bytes[off..][0..p.len], p.dev);
        off += p.len;
    }
    try e.s.synchronize();
    return copy;
}

/// Make `seq` the kept state's: the next prompt chunk starts at its position.
pub fn restoreSeq(e: *Engine, seq: *st.Seq, saved: *anyopaque) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    if (copy.at >= seq.capacity) return error.BadReuse;
    var buf: [max_pieces]Piece = undefined;
    var off: usize = 0;
    for (pieces(e.w.g, e.kv8, seq, copy.at, &buf)) |p| {
        try e.ops().upload(p.dev, copy.host.bytes[off..][0..p.len]);
        off += p.len;
    }
    try e.s.synchronize(); // the host copy may be dropped once this returns
    seq.pos = copy.at;
    seq.rope_delta = 0;
    seq.media = null;
}

/// The store's restore: the owner's state made the kept one's.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    return restoreSeq(t.e, t.seqs.get(s) orelse return error.UnknownStream, saved);
}

/// Free a kept state's host copy.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    copy.host.free();
    t.e.gpa.destroy(copy);
}

test "a kept state holds the recurrences whole and the attention rows it read" {
    const g = c.the_27b;
    var buf: [max_pieces]Piece = undefined;
    try std.testing.expectEqual(@as(usize, 48 * 2 + 16 * 2), pieces(g, false, null, 1, &buf).len);
    try std.testing.expectEqual(@as(usize, 48 * 2 + 16 * 4), pieces(g, true, null, 1, &buf).len);
    const fixed = total(g, false, 0);
    try std.testing.expectEqual(fixed, 48 * (g.deltaBytes() + g.convBytes()));
    try std.testing.expectEqual(fixed + 16 * 2 * 1000 * g.kvInner() * 2, total(g, false, 1000));
    try std.testing.expectEqual(fixed + 16 * 2 * 1000 * (g.kvInner() + g.kv_heads * 4), total(g, true, 1000));
}
