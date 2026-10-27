//! The identity of a captured EXL3 decode graph: its key, the shape that may be updated in place, the rendezvous hash.

const std = @import("std");
const wt = @import("../../cuda_prompt_wt.zig");
const grebind = @import("cuda_exl3_graph_rebind.zig");

pub const max_lanes = 32;

pub const Key = struct {
    /// 1: a decode round; 2: a window.
    kind: u8,
    rows: usize,
    /// Per lane (a window: lane 0): the sequence's buffers, its attention chunks and score tiles at the step's end.
    seq: [max_lanes][5]u64 = @splat(@splat(0)),
    paging: [max_lanes][6]u64 = @splat(@splat(0)),
    paging_logical: ?[max_lanes][2]u64 = null,
    chunks: [max_lanes]u32 = @splat(0),
    score_tiles: [max_lanes]u32 = @splat(0),
    /// The pass's scratch (its first buffer and its row capacity, which sets every other buffer's offset).
    scratch: u64,
    scratch_cap: usize = 0,
    scores: u64,
    output: u64,
    streams: u64 = 0,
    keep: u64 = 0,
    keep_nonstate: u64 = 0,
    /// The device planner's state (a new planner is new memory).
    plan: u64 = 0,

    pub fn seams(end: usize, chunks: *u32, tiles: *u32) void {
        chunks.* = @intCast((@min(end, wt.dense_keys) + 511) / 512);
        tiles.* = if (end / wt.index_ratio > wt.index_top) @intCast(((end + 3) / 4 + 63) / 64) else 0;
    }

    /// The DeltaNet state set is in the key: a drafting stream swaps it with its window's spare set (Draft.settle).
    pub fn seqOf(q: anytype) [5]u64 {
        return .{ if (q.kc.len > 0) q.kc[0].ptr else 0, if (q.conv.len > 0) q.conv[0].ptr else 0, q.tail.ptr, q.cap, if (q.state.len > 0) q.state[0].ptr else 0 };
    }
};

/// The key without the addresses a cudaGraphExecUpdate may change: equal shapes capture the same topology.
pub fn sameShape(a: Key, b: Key) bool {
    return a.kind == b.kind and a.rows == b.rows and std.mem.eql(u32, &a.chunks, &b.chunks) and
        std.mem.eql(u32, &a.score_tiles, &b.score_tiles) and a.scratch_cap == b.scratch_cap and
        (a.keep != 0) == (b.keep != 0) and (a.streams != 0) == (b.streams != 0);
}

pub fn graphSignature(key: Key, pos0: usize) u64 {
    var hash = std.hash.Wyhash.init(0);
    std.hash.autoHash(&hash, .{ key.kind, key.rows, key.chunks, key.score_tiles, key.scratch_cap, key.keep != 0, key.streams != 0, pos0 });
    std.hash.autoHash(&hash, grebind.stateRebindEnabled());
    for (key.seq, key.paging, 0..) |seq, paging, index| {
        const identity: [2]u64 = if (key.paging_logical) |logical| logical[index] else .{ paging[4], paging[5] };
        std.hash.autoHash(&hash, .{ seq[3], identity[0], identity[1] });
    }
    return hash.final();
}
