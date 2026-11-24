//! W8A16 tensor-parallel partitioning (world 1/2/4 on 128-row FP8 blocks) and per-rank memory floor for Hy4-flash.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const engram = @import("engram.zig");

const Text = config.Text;
const Why = core.modelopt.Why;

pub const max_world = 4;
pub const block = core.quant.fp8.block;

pub const Share = struct {
    lo: usize,
    len: usize,

    pub fn scaled(s: Share, by: usize) Share {
        return .{ .lo = s.lo * by, .len = s.len * by };
    }
};

/// Partition `count` items into `world` (1, 2 or 4) equal shares aligned to `unit`.
pub fn even(count: usize, unit: usize, rank: usize, world: usize) !Share {
    if ((world != 1 and world != 2 and world != 4) or rank >= world) return error.UnsupportedWorld;
    if (unit == 0 or count % (world * unit) != 0) return error.SplitNotOnBlocks;
    const len = count / world;
    return .{ .lo = rank * len, .len = len };
}

/// Per-rank attention slices: whole heads with q_b, linear_gate and o_proj aligned to 128-element FP8 blocks.
pub const Attn = struct {
    heads: Share,
    q_rows: Share,
    gate_rows: Share,
    kv_rows: Share,
    o_cols: Share,

    pub fn of(text: Text, layer: usize, rank: usize, world: usize) !Attn {
        const h = text.headShape(@intCast(layer));
        const heads = try even(h.heads, 1, rank, world);
        const a: Attn = .{
            .heads = heads,
            .q_rows = heads.scaled(@as(usize, h.non_rotary) + h.rotary),
            .gate_rows = heads.scaled(h.value),
            .kv_rows = heads.scaled(@as(usize, h.non_rotary) + h.value),
            .o_cols = heads.scaled(h.value),
        };
        for ([_]Share{ a.q_rows, a.gate_rows, a.o_cols }) |s| {
            if (s.lo % block != 0 or s.len % block != 0) return error.SplitNotOnBlocks;
        }
        return a;
    }
};

/// Per-rank FFN intermediate slice aligned to 128-row FP8 blocks.
pub fn ffn(width: usize, rank: usize, world: usize) !Share {
    return even(width, block, rank, world);
}

/// Per-rank routed expert intermediate slice (every rank holds every expert's inner slice).
pub fn expert(text: Text, rank: usize, world: usize) !Share {
    return ffn(text.moe_intermediate_size, rank, world);
}

/// Per-rank Engram slice: whole hash heads, their contiguous table rows and projection input columns.
pub const Engram = struct {
    heads: Share,
    cols: Share,
    bias: bool,

    pub fn of(text: Text, rank: usize, world: usize) !Engram {
        const heads = try even(engram.columns, 1, rank, world);
        const head_dim = text.engram_n_embed_per_ngram / text.engram_n_head_per_ngram;
        return .{ .heads = heads, .cols = heads.scaled(head_dim), .bias = rank == 0 };
    }

    /// Contiguous table rows owned by this rank's heads in one-based Engram `layer_id`.
    pub fn rows(e: Engram, text: Text, layer_id: u32) !Share {
        var key_buf: [10]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{d}", .{layer_id});
        const groups = text.engram_layer_head_vocab_sizes.map.get(key) orelse return error.BadEngramGeometry;
        var flat: [engram.columns]u32 = undefined;
        var at: usize = 0;
        for (groups) |g| for (g) |prime| {
            if (at >= flat.len) return error.BadEngramGeometry;
            flat[at] = prime;
            at += 1;
        };
        if (at != flat.len) return error.BadEngramGeometry;
        var lo: usize = 0;
        for (flat[0..e.heads.lo]) |p| lo += p;
        var len: usize = 0;
        for (flat[e.heads.lo..][0..e.heads.len]) |p| len += p;
        return .{ .lo = lo, .len = len };
    }
};

/// Per-rank lm_head vocabulary slice.
pub fn vocab(text: Text, rank: usize, world: usize) !Share {
    return even(text.vocab_size, 1, rank, world);
}

/// Validate every TP split required at `world` across all ranks and layers before loading weights.
pub fn check(text: Text, world: usize, why: *Why) !void {
    if (world != 1 and world != 2 and world != 4) {
        why.set("world {d} is not supported (expected 1, 2 or 4)", .{world});
        return error.UnsupportedWorld;
    }
    const shared_w: usize = @as(usize, text.moe_intermediate_size) * text.n_shared_experts;
    for (0..world) |r| {
        for (0..text.num_hidden_layers) |l| _ = Attn.of(text, l, r, world) catch |e| {
            why.set("layer {d}: {d} heads do not split over {d} ranks on {d}-row blocks", .{ l, text.headShape(@intCast(l)).heads, world, block });
            return e;
        };
        _ = expert(text, r, world) catch |e| return refuse(why, "routed expert width", text.moe_intermediate_size, world, e);
        _ = ffn(shared_w, r, world) catch |e| return refuse(why, "shared expert width", shared_w, world, e);
        _ = ffn(text.intermediate_size, r, world) catch |e| return refuse(why, "dense FFN width", text.intermediate_size, world, e);
        _ = Engram.of(text, r, world) catch |e| return refuse(why, "Engram heads", engram.columns, world, e);
        _ = vocab(text, r, world) catch |e| return refuse(why, "vocabulary", text.vocab_size, world, e);
    }
}

fn refuse(why: *Why, what: []const u8, n: usize, world: usize, e: anyerror) anyerror {
    why.set("{s} {d} does not split over {d} ranks", .{ what, n, world });
    return e;
}

/// Bytes per 128x128 FP8 block scale: bf16 as stored on disk, or fp32 in packed GPU buffers.
pub const Scale = enum(usize) { stored = 2, packed_ = 4 };
pub const Format = enum { bf16, fp8 };

/// Projection [n, k] byte size in bf16 or e4m3 + block scales.
pub fn projBytes(f: Format, n: usize, k: usize, scale: Scale) usize {
    return switch (f) {
        .bf16 => n * k * 2,
        .fp8 => n * k + core.quant.fp8.blocks(n) * core.quant.fp8.blocks(k) * @backingInt(scale),
    };
}

fn attnBytes(text: Text, layer: usize, rank: usize, world: usize, scale: Scale) !usize {
    const d: usize = text.hidden_size;
    const h = text.headShape(@intCast(layer));
    const a = try Attn.of(text, layer, rank, world);
    const f: Format = if (layer == 0) .bf16 else .fp8;
    var n: usize = 2 * (@as(usize, text.q_lora_rank) * d + text.q_lora_rank + (@as(usize, text.kv_lora_rank) + h.rotary) * d + text.kv_lora_rank + d);
    n += 2 * a.kv_rows.len * text.kv_lora_rank + 4 * a.heads.len;
    n += projBytes(f, a.q_rows.len, text.q_lora_rank, scale) + projBytes(f, a.gate_rows.len, d, scale) + projBytes(f, d, a.o_cols.len, scale);
    if (text.layer_types[layer] == .deepseek_sparse_attention) {
        const t: usize = text.compressed_indexer_head_dim;
        const ih: usize = text.compressed_indexer_num_attention_heads;
        n += 2 * (ih * d + ih * t * text.q_lora_rank + t + t * d) + 4 * (@as(usize, text.compressed_indexer_compress_ratio) * t + 2 * t * t + t);
    }
    return n;
}

fn ffnBytes(text: Text, layer: usize, rank: usize, world: usize, scale: Scale) !usize {
    const d: usize = text.hidden_size;
    if (text.mlp_layer_types[layer] == .dense) return 2 * 3 * (try ffn(text.intermediate_size, rank, world)).len * d + 2 * d;
    const w = (try expert(text, rank, world)).len;
    const l: usize = text.routed_expert_hidden_size;
    const shared_w: usize = @as(usize, text.moe_intermediate_size) * text.n_shared_experts;
    var n: usize = 2 * @as(usize, text.n_routed_experts) * d + 4 * @as(usize, text.n_routed_experts) + 2 * 2 * l * d;
    n += 2 * 3 * (try ffn(shared_w, rank, world)).len * d + 2 * d;
    return n + @as(usize, text.n_routed_experts) * (projBytes(.fp8, 2 * w, l, scale) + projBytes(.fp8, l, w, scale));
}

fn engramBytes(text: Text, rank: usize, world: usize) !usize {
    const d: usize = text.hidden_size;
    const head_dim: usize = text.engram_n_embed_per_ngram / text.engram_n_head_per_ngram;
    const e = try Engram.of(text, rank, world);
    const projections: usize = 1 + text.hc_mult;
    return 4 * @as(usize, text.engram_lookup_table_size) + 2 * e.heads.len * head_dim + projections * 2 * (d * e.cols.len + d) + 2 * d * (3 * @as(usize, text.hc_mult) + 1);
}

/// One decoder layer's weight bytes on a rank, excluding the large Engram embedding table.
pub fn layerBytes(text: Text, layer: usize, rank: usize, world: usize, scale: Scale) !usize {
    const d: usize = text.hidden_size;
    const hc: usize = text.hc_mult;
    var n = 2 * 2 * d + 2 * (4 * 2 * hc * hc * d + 4 * 2 * hc + 4 * 2);
    n += try attnBytes(text, layer, rank, world, scale) + try ffnBytes(text, layer, rank, world, scale);
    if (std.mem.indexOfScalar(u32, text.engram_layer_ids, @intCast(layer + 1)) != null) n += try engramBytes(text, rank, world);
    return n;
}

/// Top-level weights on a rank: replicated embeddings, vocabulary-split lm_head, final norm and hc_head.
pub fn topBytes(text: Text, rank: usize, world: usize) !usize {
    const d: usize = text.hidden_size;
    const hc: usize = text.hc_mult;
    return 2 * @as(usize, text.vocab_size) * d + 2 * (try vocab(text, rank, world)).len * d + 2 * d + 4 * (hc * hc * d + hc + 1);
}

/// Total GPU-resident backbone weight bytes on a rank (excluding Engram tables, DFly2 and vision).
pub fn weightBytes(text: Text, rank: usize, world: usize, scale: Scale) !usize {
    var n = try topBytes(text, rank, world);
    for (0..text.num_hidden_layers) |l| n += try layerBytes(text, l, rank, world, scale);
    return n;
}

/// A rank's Engram table bytes across all Engram layers: bf16 as stored (192 B/row) or packed FP8 (100 B/row).
pub fn engramTableBytes(text: Text, rank: usize, world: usize, row: enum { stored, packed_ }) !usize {
    const e = try Engram.of(text, rank, world);
    const head_dim: usize = text.engram_n_embed_per_ngram / text.engram_n_head_per_ngram;
    const per_row = if (row == .stored) 2 * head_dim else head_dim + 4;
    var n: usize = 0;
    for (text.engram_layer_ids) |layer_id| n += (try e.rows(text, layer_id)).len * per_row;
    return n;
}

/// BF16 KV cache bytes per token on each rank across all sparse-indexed layers.
pub fn tokenBytes(text: Text) usize {
    var n: usize = 0;
    for (text.layer_types) |k| if (k == .deepseek_sparse_attention) {
        n += 2 * (@as(usize, text.kv_lora_rank) + text.qk_rope_head_dim + text.compressed_indexer_head_dim);
    };
    return n;
}

/// Per-stream BF16 sliding-window ring bytes across all sliding attention layers.
pub fn ringBytes(text: Text) usize {
    var n: usize = 0;
    for (text.layer_types) |k| if (k == .sliding_attention) {
        n += 2 * (@as(usize, text.kv_lora_rank) + text.swa_qk_rope_head_dim) * text.sliding_window;
    };
    return n;
}
