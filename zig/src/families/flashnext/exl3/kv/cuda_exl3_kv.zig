//! KV memory of an EXL3 pack's streams (K10): bytes by piece, the scratch format, where kept states live.

const std = @import("std");
const wt = @import("../../cuda_prompt_wt.zig");
const gdn = @import("../../cuda_gdn.zig");

const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const index_row: usize = wt.index_dim * 2;

/// One token's K and V in one attention layer, bf16.
pub fn kvRowBytes() usize {
    return 2 * kv_heads * head_dim * 2;
}

/// Bytes a sequence of `cap` tokens holds, by piece.
pub const SeqBytes = struct {
    gdn: u64,
    kv: u64,
    ikc: u64,
    pooled: u64,
    tail: u64,

    pub fn device(s: SeqBytes) u64 {
        return s.gdn + s.kv + s.ikc + s.pooled + s.tail;
    }
};

pub fn pooledBytes(cap: usize) u64 {
    return (cap + wt.index_ratio - 1) / wt.index_ratio * index_row;
}

/// cuda_prompt.zig seqBytes split by piece (a host test checks that the pieces sum to upstream's number).
pub fn seqBytes(cap: usize) SeqBytes {
    const attn: u64 = wt.attn_n;
    return .{
        .gdn = wt.linear_n * ((gdn.taps - 1) * gdn.conv_dim * 2 + wt.state_n * 4),
        .kv = attn * cap * kvRowBytes(),
        .ikc = attn * cap * index_row,
        .pooled = attn * pooledBytes(cap),
        .tail = wt.ple_tail_n * wt.wide * 2,
    };
}

pub const gib: f64 = 1024.0 * 1024.0 * 1024.0;

/// Widest logits pass the server runs (a decode round of 32 streams, an MTP window): `head_local`'s rows.
pub const head_local_serve_rows: usize = 64;

/// A full-vocabulary rank still stages every batched MTP lane's BF16 scores in the shared head buffer.
pub fn headLocalRows(full_vocab: bool) usize {
    return if (full_vocab) 2 * @import("../../cuda_mtp_ops.zig").max_lanes else head_local_serve_rows;
}
/// K10 lever 3: the pass's chunk for an EXL3 checkpoint unless TF_FLASHNEXT_CHUNK says.
pub const exl3_chunk: usize = 1024;

/// The server's load hook (cuda_lanes_load.zig, right after the engine loads, before any scratch or sequence).
pub fn setup(e: anytype, exl3: bool) !void {
    if (!exl3) return;
    if (e.batch_s != null) return error.ScratchAlreadyReserved;
    const decoder_exl3 = e.n_layers != 0 and for (e.layers[0..e.n_layers]) |layer| {
        if (layer.exl3 == null) break false;
    } else true;
    const draft_exl3 = @import("../pack/cuda_exl3_load.zig").hasExperts(&e.store, "language_model.mtp.layers.0.mlp");
    e.scratch_format = if (decoder_exl3 and draft_exl3) .exl3 else .classic;
    if (std.c.getenv("TF_FLASHNEXT_CHUNK") == null) e.chunk = exl3_chunk;
    e.head_local_rows = headLocalRows(e.head_q.n == wt.vocab);
}

test "exl3: full-vocabulary scratch admits every shared MTP score row" {
    const state = @import("../../cuda_batch_state.zig");
    const lanes = @import("../../cuda_mtp_ops.zig").max_lanes;
    const empty = state.Scratch.bytesForHead(exl3_chunk, .exl3, 0);
    const full = state.Scratch.bytesForHead(exl3_chunk, .exl3, headLocalRows(true));
    try std.testing.expect(full - empty + 256 >= lanes * wt.vocab * 2);
    try std.testing.expect(headLocalRows(false) >= headLocalRows(true));
}

/// One log line: a stream's device bytes at the window by piece.
pub fn logStream(e: anytype, window: usize, stream_bytes: usize) !void {
    const b = seqBytes(window);
    std.log.info("flashnext: a stream at a {d}-token window holds {d:.3} GiB on the device (K/V {d:.3} bf16, indexer keys {d:.3}, pooled {d:.3}, DeltaNet {d:.3}, draft and the rest {d:.3}); chunk {d} rows, scratch {d:.3} GiB", .{
        window,
        @as(f64, @floatFromInt(stream_bytes)) / gib,
        @as(f64, @floatFromInt(b.kv)) / gib,
        @as(f64, @floatFromInt(b.ikc)) / gib,
        @as(f64, @floatFromInt(b.pooled)) / gib,
        @as(f64, @floatFromInt(b.gdn)) / gib,
        @as(f64, @floatFromInt(stream_bytes -| b.device())) / gib,
        e.chunk,
        @as(f64, @floatFromInt(@import("../../cuda_batch_state.zig").Scratch.bytesForHead(e.chunk, e.scratch_format, e.head_local_rows))) / gib,
    });
}

/// Whether kept prompt states go to pinned host memory (cuda_snap.on_host, set here):
pub fn snapHost(context: usize, exl3: bool) bool {
    var on = exl3 and context > 32768;
    if (std.c.getenv("TF_FLASHNEXT_SNAP_HOST")) |v| on = std.mem.eql(u8, std.mem.span(v), "1");
    @import("../../cuda_snap.zig").on_host = on;
    return on;
}

/// cuda_lanes.zig rankRoom: the ranks agree how many STREAMS fit (each its own stream_bytes: rank 1 holds no MTP head).
pub fn rankRoom(r: anytype, local: u64, stream_bytes: usize) !u64 {
    if (stream_bytes == 0) return r.room(local);
    const sb: u64 = stream_bytes;
    const q = (local / sb) << 16 | ((local % sb) << 16) / sb;
    const m = try r.room(q);
    return (m >> 16) * sb + ((m & 0xffff) * sb >> 16);
}
