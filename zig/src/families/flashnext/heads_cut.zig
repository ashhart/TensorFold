//! What one rank keeps of a layer under the head split: stitched in-projection rows, conv channels and per-head values.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const gdn = @import("cuda_gdn.zig");
const wt = @import("cuda_prompt_wt.zig");
const Heads = @import("cuda_heads.zig").Heads;
const load = @import("cuda_projection_load.zig");
const Cut = @import("cuda_modelopt_experts.zig").Cut;
const Ranges = load.Ranges;

/// The stitched parts of each class, in the order their rows are kept.
pub const delta_parts = [_][]const u8{ ".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a" };
pub const attention_parts = [_][]const u8{ ".q_proj", ".k_proj", ".v_proj", ".indexer.index_qk_proj" };

/// One rank's cut of a stitched in-projection: each part's kept rows, the local width, the whole shape's K slices.
pub const Stitch = struct { ranges: [4]Ranges, n: usize, sk: u32 };

/// `h` divides every count of the full heads by one world, and `rank` is one of its ranks.
fn check(h: Heads, rank: usize) error{UnevenHeadSplit}!void {
    const f = Heads.full;
    if (h.nk == 0 or f.nk % h.nk != 0) return error.UnevenHeadSplit;
    const w: usize = f.nk / h.nk;
    if (f.nv != w * h.nv or f.q != w * h.q or f.kv != w * h.kv or rank >= w) return error.UnevenHeadSplit;
}

fn span(from: usize, width: usize) Cut {
    return .{ .from = from, .to = from + width };
}

/// The q, k and v channels of the rank's key and value heads, in the layout `in_proj_qkv` and conv1d share.
pub fn convChannels(h: Heads, rank: usize) !Ranges {
    try check(h, rank);
    const qk = h.nk * gdn.dk;
    const v = h.nv * gdn.dv;
    const whole_qk = Heads.full.nk * gdn.dk;
    return Ranges.of(&.{ span(rank * qk, qk), span(whole_qk + rank * qk, qk), span(2 * whole_qk + rank * v, v) });
}

/// The rank's value heads, the rows of `in_proj_b` and `in_proj_a` and the entries of `A_log` and `dt_bias`.
pub fn valueHeads(h: Heads, rank: usize) !Cut {
    try check(h, rank);
    return span(rank * h.nv, h.nv);
}

/// The DeltaNet in-projection: q, k and v channels, z of each value head, then b and a.
pub fn delta(h: Heads, rank: usize) !Stitch {
    const vh = try valueHeads(h, rank);
    const z = h.nv * gdn.dv;
    const s: Stitch = .{
        .ranges = .{ try convChannels(h, rank), Ranges.of(&.{span(rank * z, z)}), Ranges.of(&.{vh}), Ranges.of(&.{vh}) },
        .n = h.projWidth(),
        .sk = cuda.qmmf.splitK(Heads.full.projWidth(), wt.dims),
    };
    std.debug.assert(rows(s.ranges) == s.n);
    return s;
}

/// The attention in-projection: each query head's q and gate rows, k and v of the KV head, the indexer whole.
pub fn attention(h: Heads, rank: usize) !Stitch {
    try check(h, rank);
    const q = h.q * 2 * wt.head_dim;
    const kv = h.kv * wt.head_dim;
    const s: Stitch = .{
        .ranges = .{
            Ranges.of(&.{span(rank * q, q)}),
            Ranges.of(&.{span(rank * kv, kv)}),
            Ranges.of(&.{span(rank * kv, kv)}),
            Ranges.of(&.{span(0, wt.index_heads * wt.index_dim + wt.index_dim)}),
        },
        .n = h.projN(),
        .sk = cuda.qmmf.splitK(Heads.full.projN(), wt.dims),
    };
    std.debug.assert(rows(s.ranges) == s.n);
    return s;
}

fn rows(ranges: [4]Ranges) usize {
    var n: usize = 0;
    for (ranges) |r| n += r.rows();
    return n;
}

/// The rank's channels of a BF16 conv1d weight (channel-major, `taps` per channel), copied before the store moves on.
pub fn convRows(a: std.mem.Allocator, t: core.safetensors.Tensor, h: Heads, rank: usize) ![]u8 {
    const kept = try convChannels(h, rank);
    const row = gdn.taps * 2;
    if (t.dtype != .bf16 or t.rank == 0 or t.dim(0) != Heads.full.convDim() or t.bytes.len != t.dim(0) * row)
        return error.UnexpectedTensor;
    const out = try a.alloc(u8, h.convDim() * row);
    errdefer a.free(out);
    if (try load.copyRanges(out, t.bytes, row, kept) != out.len) return error.UnexpectedTensor;
    return out;
}

/// The rank's entries of a per-value-head vector (`A_log`, `dt_bias`) of the full heads.
pub fn valueSlice(values: []const f32, h: Heads, rank: usize) ![]const f32 {
    if (values.len != Heads.full.nv) return error.UnexpectedTensor;
    const c = try valueHeads(h, rank);
    return values[c.from..c.to];
}
