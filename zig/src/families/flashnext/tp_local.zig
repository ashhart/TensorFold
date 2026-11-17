//! Decode's MoE stage on each Mac alone: both ranks' branch rows made here, merged as the exchange would merge them.
const std = @import("std");
const mtl = @import("metal");
const tp_seq = @import("tp_seq.zig");
const D = 2560;
const MAXR = 16;

/// The local launches' guard word: no published epoch reaches it, so a ready word bound to it never passes.
pub const PASS: u64 = std.math.maxInt(u64);
/// Both ranks' branch rows follow the guard word, each rank's at its first row, as its send slot holds them.
pub const ROWS_AT = 256;
pub const BYTES = ROWS_AT + MAXR * D * 2;

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

test "local branch rows fit after the guard word and each rank's rows start at its first row" {
    try std.testing.expect(ROWS_AT >= @sizeOf(u64) and ROWS_AT % 256 == 0);
    for (1..MAXR + 1) |rows| for (0..2) |k| {
        const own = tp_seq.ownRows(@intCast(k), rows);
        try std.testing.expect(ROWS_AT + own[1] * D * 2 <= BYTES);
    };
}
