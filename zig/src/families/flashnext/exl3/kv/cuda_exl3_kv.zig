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

/// K10 lever 3: the pass's chunk for an EXL3 checkpoint unless TF_FLASHNEXT_CHUNK says.
pub const exl3_chunk: usize = 1024;

/// One prompt cut whatever else decodes (chunks never depend on batching): busy and solo passes take the same rows.
pub fn cut(e: anytype, chunk_set: bool) void {
    if (!chunk_set) e.chunk = exl3_chunk;
    e.busy_chunk = e.chunk;
}

/// TF_FLASHNEXT_CHUNK_SOLO would cut a lone stream's prompt apart from a batched one's.
pub fn refuseSolo(exl3: bool, solo_set: bool) !void {
    if (exl3 and solo_set) return error.Exl3ChunkSoloUnsupported;
}

/// The server's load hook (cuda_lanes_load.zig, right after the engine loads, before any scratch or sequence).
pub fn setup(e: anytype, exl3: bool) !void {
    if (!exl3) return;
    if (e.batch_s != null) return error.ScratchAlreadyReserved;
    const decoder_exl3 = e.n_layers != 0 and for (e.layers[0..e.n_layers]) |layer| {
        if (layer.exl3 == null) break false;
    } else true;
    e.scratch_format = if (decoder_exl3) .exl3 else .classic;
    cut(e, std.c.getenv("TF_FLASHNEXT_CHUNK") != null);
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
        @as(f64, @floatFromInt(@import("../../cuda_batch_state.zig").Scratch.bytesForFormat(e.chunk, e.scratch_format))) / gib,
    });
}

/// Whether kept prompt states go to pinned host memory (cuda_snap.on_host, set here):
pub fn snapHost(context: usize, exl3: bool) bool {
    var on = exl3 and context > 32768;
    if (std.c.getenv("TF_FLASHNEXT_SNAP_HOST")) |v| on = std.mem.eql(u8, std.mem.span(v), "1");
    @import("../../cuda_snap.zig").on_host = on;
    return on;
}
