//! Decode on each Mac alone: both ranks' halves of every split stage made here, joined as the exchanges join them.
const std = @import("std");
const mtl = @import("metal");
const tp_seq = @import("tp_seq.zig");
const D = 2560;
const MAXR = 16;

/// The local launches' guard word: no published epoch reaches it, so a ready word bound to it never passes.
pub const PASS: u64 = std.math.maxInt(u64);
/// Both ranks' branch rows follow the guard word, each rank's at its first row, as its send slot holds them.
pub const ROWS_AT = 256;
/// The peer rank's fp32 out-projection partial, as its reduce delivers it; this rank's stays in the pair's partial.
pub const PARTIAL_AT = ROWS_AT + MAXR * D * 2;
/// This rank's argmax part (value, index a row), then the peer rank's, as the pick's send and receive slots hold them.
pub const PICKS_AT = PARTIAL_AT + MAXR * D * 4;
pub const PICK = 256;
pub const BYTES = PICKS_AT + 2 * PICK;

/// Write the guard word once; no kernel writes below ROWS_AT.
pub fn arm(b: mtl.Buffer) void {
    b.slice(u64, 1)[0] = PASS;
}

fn guard(t: anytype, enc: mtl.ComputeEncoder, first: usize) void {
    enc.setBuffer(t.branches, 0, first);
    enc.setValue(PASS, first + 1);
}

/// Each rank's owned rows' MoE branch with the exchange's kernel and arguments, into this Mac's branch rows.
pub fn branch(t: anytype, enc: mtl.ComputeEncoder, y: anytype, wts: anytype, lg: anytype, rows: usize) void {
    for (0..2) |k| {
        const own = tp_seq.ownRows(@intCast(k), rows);
        if (own[1] == own[0]) continue;
        const own2 = [2]u32{ @intCast(own[0]), @intCast(own[1]) };
        enc.setPipeline(t.branch_pipe);
        for ([_]@TypeOf(y){ y, wts, lg }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.setBuffer(t.branches, ROWS_AT + own[0] * D * 2, 3);
        enc.setBytes(std.mem.asBytes(&own2), 4);
        guard(t, enc, 5);
        enc.dispatchThreads(mtl.Size.of(D, own[1] - own[0], 1), mtl.Size.of(256, 1, 1));
    }
    if (enc.dispatchType() == .concurrent) enc.barrier();
}

/// The exchange's stream update with this rank's split, both ranks' rows read from this Mac's branch rows.
pub fn merge(t: anytype, enc: mtl.ComputeEncoder, h: anytype, inj: anytype, out: anytype, ssp: anytype, rows_buf: anytype, rows: usize) void {
    const own = tp_seq.ownRows(t.rank, rows);
    const peer_lo: usize = if (t.rank == 0) own[1] else 0;
    const split = [4]u32{ @intCast(own[0]), @intCast(own[1]), @intCast(peer_lo), 0 };
    enc.setPipeline(t.merge_pipe);
    guard(t, enc, 0);
    enc.setBuffer(rows_buf.b, rows_buf.off, 2);
    enc.setBuffer(h.b, h.off, 3);
    enc.setBuffer(inj.b, inj.off, 4);
    enc.setBuffer(t.branches, ROWS_AT + own[0] * D * 2, 5);
    enc.setBuffer(t.branches, ROWS_AT + peer_lo * D * 2, 6);
    enc.setBytes(std.mem.asBytes(&split), 7);
    enc.setBuffer(out.b, out.off, 8);
    enc.setBuffer(ssp.b, ssp.off, 9);
    enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
}

/// The reduce's stream update with this rank's arguments: its own partial as mine, the peer rank's from here.
pub fn plain(t: anytype, enc: mtl.ComputeEncoder, h: anytype, inj: anytype, out: anytype, ssp: anytype, rows_buf: anytype, rows: usize) void {
    const mine = t.partNext();
    enc.setPipeline(t.plain_pipe);
    guard(t, enc, 0);
    enc.setBuffer(rows_buf.b, rows_buf.off, 2);
    enc.setBuffer(mine.b, mine.off, 3);
    enc.setBuffer(t.branches, PARTIAL_AT, 4);
    enc.setBytes(std.mem.asBytes(&t.rank), 5);
    enc.setBuffer(h.b, h.off, 6);
    enc.setBuffer(inj.b, inj.off, 7);
    enc.setBuffer(out.b, out.off, 8);
    enc.setBuffer(ssp.b, ssp.off, 9);
    enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
}

/// Both ranks' argmax parts over their own {first column, columns} in `parts`, then the swap's pick on this rank.
pub fn argmax(t: anytype, enc: mtl.ComputeEncoder, logits: anytype, vocab: usize, parts: [2][2]usize, picks: anytype, rows_buf: anytype, rows: usize) void {
    for ([2]u32{ t.rank, 1 - t.rank }, 0..) |k, j| {
        const dims = [4]u32{ @intCast(vocab), @intCast(parts[k][0]), @intCast(parts[k][1]), 0 };
        enc.setPipeline(t.argmax_pipe);
        enc.setBuffer(logits.b, logits.off, 0);
        enc.setBuffer(t.branches, PICKS_AT + j * PICK, 1);
        enc.setBytes(std.mem.asBytes(&dims), 2);
        guard(t, enc, 3);
        enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }
    if (enc.dispatchType() == .concurrent) enc.barrier();
    enc.setPipeline(t.pick_pipe);
    guard(t, enc, 0);
    enc.setBuffer(rows_buf.b, rows_buf.off, 2);
    enc.setBuffer(t.branches, PICKS_AT, 3);
    enc.setBuffer(t.branches, PICKS_AT + PICK, 4);
    enc.setBuffer(picks.b, picks.off, 5);
    enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
}

/// After `argmax` into `pick_tmp`: the merged draft pick's token id into `out`, as mapIds makes it.
pub fn ids(t: anytype, enc: mtl.ComputeEncoder, ids_buf: anytype, out: anytype) void {
    enc.setPipeline(t.ids_pipe);
    enc.setBuffer(t.pick_tmp, 0, 0);
    enc.setBuffer(ids_buf.b, ids_buf.off, 1);
    enc.setBuffer(out.b, out.off, 2);
    guard(t, enc, 3);
    enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
}

test "local branch rows fit after the guard word and each rank's rows start at its first row" {
    try std.testing.expect(ROWS_AT >= @sizeOf(u64) and ROWS_AT % 256 == 0);
    for (1..MAXR + 1) |rows| for (0..2) |k| {
        const own = tp_seq.ownRows(@intCast(k), rows);
        try std.testing.expect(ROWS_AT + own[1] * D * 2 <= PARTIAL_AT);
    };
}

test "the peer's partial and both argmax parts hold the widest window apart from the branch rows" {
    try std.testing.expectEqual(ROWS_AT + MAXR * D * 2, PARTIAL_AT);
    try std.testing.expectEqual(PARTIAL_AT + MAXR * D * 4, PICKS_AT);
    try std.testing.expect(MAXR * 2 * @sizeOf(u32) <= PICK and PARTIAL_AT % 256 == 0 and PICKS_AT % 256 == 0);
    try std.testing.expectEqual(PICKS_AT + 2 * PICK, BYTES);
}
