//! DeepSeek V4.1's indexer scores on Metal: the pipeline, its arguments and the chunk a launch cuts its keys into.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const heads = 32; // index_n_heads
pub const dim = 128; // index_head_dim
pub const block = 8; // candidate_block_size
/// Simdgroups a threadgroup carries (the source's IXR), each scoring a block of 8 rows.
pub const simdgroups = 4;
/// Rows a threadgroup scores.
pub const rows_group = 8 * simdgroups;
pub const max_rows = 8192;
pub const max_keys = 1048576; // max_position_embeddings

/// A call: `rows` queries of `keys` keys; a weight row is w_stride bf16 elements apart, a key row ik_stride bytes.
pub const Call = struct {
    rows: u32,
    keys: u32,
    start: u32 = 1, // a row r reaches (start + r) / ratio keys: 1 is prefill, keys is decode (all visible)
    ratio: u32 = 1,
    chunk: u32 = 0, // keys a simdgroup scores before moving on; 0: chunkOf. No bit depends on it.
    w_stride: u32 = 32,
    ik_stride: u32 = 68, // 64 FP4 bytes then 4 e8m0 scales
    mask_stride: u32 = 0, // 0: no candidate mask; else one byte a block a row
    w_scale: f32 = 0x1p-6, // softmax_scale * heads^-0.5 for 128 dims, 32 heads: exactly 2^-6, applied in the kernel
};

const Args = extern struct {
    rows: u32,
    keys: u32,
    start: u32,
    ratio: u32,
    chunk: u32,
    out_stride: u32,
    w_stride: u32,
    ik_stride: u32,
    mask_stride: u32,
    w_scale: f32,
};

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

pub const Pipelines = struct {
    score: mtl.Pipeline,
    /// decode's few rows: eight simdgroups a group over one decoded query (the same bits)
    dec: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.deepseek_v41_index);
    }

    /// The same kernel from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{ .score = try mtl.Pipeline.init(device, lib, "tf_dsv41_ix_scores", false), .dec = try mtl.Pipeline.init(device, lib, "tf_dsv41_ix_scores_dec", false) };
    }

    pub fn deinit(p: *Pipelines) void {
        p.score.deinit();
        p.dec.deinit();
    }
};

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.rows > max_rows) return error.BadIxRows;
    if (c.keys == 0 or c.keys > max_keys) return error.BadIxKeys;
    if (c.ratio == 0 or c.chunk > c.keys) return error.BadIxChunk;
    if (c.mask_stride != 0 and c.mask_stride * 8 < c.keys) return error.BadIxMask;
}

/// Threadgroups a launch aims for, so a launch of few rows still fills the GPU.
const target_groups = 4000;

/// The chunk one launch cuts a row's keys into: enough chunks for `target_groups`, between 32 and 1024 keys.
pub fn chunkOf(c: Call) u32 {
    if (c.chunk != 0) return c.chunk;
    const groups = (c.rows + rows_group - 1) / rows_group;
    const chunks = target_groups / groups + 1;
    return @min(c.keys, std.math.clamp((c.keys / chunks + 15) / 16 * 16, 32, 1024));
}

/// Q [rows][2048] FP4, QS [rows][128] e8m0, W bf16, IK [keys][ik_stride], S [rows][keys] fp32; CM if mask_stride != 0.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, qs: Ref, w: Ref, ik: Ref, s: Ref, cm: Ref, c: Call) !void {
    try check(c);
    const args = Args{
        .rows = c.rows,
        .keys = c.keys,
        .start = c.start,
        .ratio = c.ratio,
        .chunk = chunkOf(c),
        .out_stride = c.keys,
        .w_stride = c.w_stride,
        .ik_stride = c.ik_stride,
        .mask_stride = c.mask_stride,
        .w_scale = c.w_scale,
    };
    e.setPipeline(p.score);
    e.setBuffer(q.buf, q.off, 0);
    e.setBuffer(qs.buf, qs.off, 1);
    e.setBuffer(w.buf, w.off, 2);
    e.setBuffer(ik.buf, ik.off, 3);
    e.setBuffer(s.buf, s.off, 4);
    e.setBuffer(cm.buf, cm.off, 5);
    e.setValue(args, 6);
    const chunks = (c.keys + args.chunk - 1) / args.chunk;
    e.dispatchGroups(mtl.Size.of(chunks, (c.rows + rows_group - 1) / rows_group, 1), mtl.Size.of(32 * simdgroups, 1, 1));
}

/// Decode's threadgroup: eight simdgroups taking a chunk's keys in turn over one decoded query.
const dec_threads = 256;

/// `encode` for decode's rows (a launch of one slot): groups of 32 keys a row, the same bits.
pub fn encodeDecode(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, qs: Ref, w: Ref, ik: Ref, s: Ref, cm: Ref, c: Call) !void {
    try check(c);
    const chunk = if (c.chunk != 0) c.chunk else @min(c.keys, 32);
    const args = Args{ .rows = c.rows, .keys = c.keys, .start = c.start, .ratio = c.ratio, .chunk = chunk, .out_stride = c.keys, .w_stride = c.w_stride, .ik_stride = c.ik_stride, .mask_stride = c.mask_stride, .w_scale = c.w_scale };
    e.setPipeline(p.dec);
    for ([_]Ref{ q, qs, w, ik, s, cm }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(args, 6);
    e.dispatchGroups(mtl.Size.of((c.keys + chunk - 1) / chunk, c.rows, 1), mtl.Size.of(dec_threads, 1, 1));
}
