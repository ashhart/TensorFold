//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` tokens, copied on the device.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;

const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
const state_bytes = wt.state_n * 4;
const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;

/// One attention cache's first `at` key, value and indexer-key rows, and the pooled blocks they complete.
fn attnBytes(at: usize) usize {
    return at * (2 * prompt.kv_row_bytes + index_row) + at / wt.index_ratio * index_row;
}

/// Device bytes a state after `at` tokens takes: the DeltaNet's are fixed, each attention cache grows by the token.
pub fn bytes(at: usize, head: bool) usize {
    return wt.linear_n * (conv_bytes + state_bytes) + (wt.attn_n + @intFromBool(head)) * attnBytes(at) + tail_bytes;
}

/// One kept state; `head`: it holds the MTP head's cache over the same rows, which a drafting stream needs.
pub const Snap = struct {
    rank_id: u64 = 0,
    buf: cuda.DeviceBuffer,
    at: usize,
    history: [2]i64,
    head: bool,
};

const Way = enum { save, restore };

const Piece = struct { bufs: []cuda.DeviceBuffer, n: usize };

/// An attention cache's pieces for `at` rows: complete pooled blocks only (the rest pool again as later rows land).
fn attnPieces(kc: []cuda.DeviceBuffer, vc: []cuda.DeviceBuffer, ikc: []cuda.DeviceBuffer, pooled: []cuda.DeviceBuffer, at: usize) [4]Piece {
    return .{
        .{ .bufs = kc, .n = at * prompt.kv_row_bytes },
        .{ .bufs = vc, .n = at * prompt.kv_row_bytes },
        .{ .bufs = ikc, .n = at * index_row },
        .{ .bufs = pooled, .n = at / wt.index_ratio * index_row },
    };
}

/// Each piece of the sequence (and the head's cache, with `d`) to or from its place in the buffer at `base`.
fn move(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, base: u64, at: usize, way: Way) !void {
    const dr = stream.d;
    var off: u64 = 0;
    var pieces: [11]Piece = undefined;
    pieces[0] = .{ .bufs = seq.conv, .n = conv_bytes };
    pieces[1] = .{ .bufs = seq.state, .n = state_bytes };
    pieces[2..6].* = attnPieces(seq.kc, seq.vc, seq.ikc, seq.pooled, at);
    pieces[6] = .{ .bufs = (&seq.tail)[0..1], .n = tail_bytes };
    var count: usize = 7;
    if (d) |h| {
        pieces[7..11].* = attnPieces((&h.kc)[0..1], (&h.vc)[0..1], (&h.ikc)[0..1], (&h.pooled)[0..1], at);
        count = 11;
    }
    for (pieces[0..count]) |p| for (p.bufs) |b| {
        if (p.n == 0) continue;
        const dst, const src = if (way == .save) .{ base + off, b.ptr } else .{ b.ptr, base + off };
        try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, p.n, stream.handle), "cuMemcpyDtoDAsync");
        off += p.n;
    };
    std.debug.assert(off == bytes(at, d != null));
}

/// A copy of `seq` (and of the head's cache, with `d`) as it stands after `seq.fed` tokens.
pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft) !*Snap {
    const at = seq.fed;
    if (d) |h| if (h.len != at or h.drafted != 0) return error.NotAtMark;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = try cuda.DeviceBuffer.alloc(driver, bytes(at, d != null)), .at = at, .history = seq.history, .head = d != null };
    errdefer snap.buf.free();
    try move(stream, seq, d, snap.buf.ptr, at, .save);
    try stream.synchronize();
    return snap;
}

/// `seq` (and the head's cache, with `d`) becomes `snap`'s state; refused before any copy when it cannot hold it.
pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (d) |h| if (!snap.head or snap.at > h.cap) return error.NoHeadState;
    // a state with the head's rows serves a stream without drafts too: those rows stay in the buffer
    try move(stream, seq, d, snap.buf.ptr, snap.at, .restore);
    try stream.synchronize();
    seq.history = snap.history;
    seq.fed = snap.at;
    if (d) |h| {
        h.reset();
        h.len = snap.at;
    }
}

pub fn drop(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    stream.synchronize() catch {};
    snap.buf.free();
    gpa.destroy(snap);
}

test "a state's bytes: about 110 MiB of DeltaNet states and the n-gram tail, then 27 KiB a token" {
    try std.testing.expectEqual(@as(usize, 12 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, false) - bytes(0, false));
    // a complete block of 4 adds its pooled row
    try std.testing.expectEqual(@as(usize, 12 * (4 * (2 * 2 * 256 * 2 + 128 * 2) + 128 * 2)), bytes(4, false) - bytes(0, false));
    // the head's cache is one more attention layer's rows
    try std.testing.expectEqual(@as(usize, 13 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true) - bytes(0, true));
    try std.testing.expect(bytes(0, false) > 100 << 20 and bytes(0, false) < 120 << 20);
}
