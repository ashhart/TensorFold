//! Hy4-flash's weight map: a checkpoint tensor's name to its part, stored format and shape, from the config alone.
const std = @import("std");
const config = @import("config.zig");

/// A tensor's stored element type, as safetensors names it.
pub const Format = enum { bf16, f32, e4m3, u8, i32 };

/// Where a tensor belongs. Drafter and vision tensors are named here and not checked yet.
pub const Part = enum { top, attention, indexer, streams, dense_mlp, router, experts, shared, engram, drafter, vision };

/// A tensor's expected shape; a 0 stands for a size the config does not fix (an Engram table's rows).
pub const Dims = struct {
    n: u8 = 0,
    d: [3]u64 = .{ 0, 0, 0 },

    pub fn matches(want: Dims, got: []const u64) bool {
        if (want.n != got.len) return false;
        for (want.d[0..want.n], got) |w, g| if (w != 0 and w != g) return false;
        return true;
    }
};

/// `plain`: the same tensor stored unquantized (BF16 at these dims), as the BF16 release and transformers' exports keep it.
pub const Expected = struct { part: Part, format: Format, dims: Dims, checked: bool = true, plain: ?Dims = null };

fn vec(part: Part, format: Format, a: u64) Expected {
    return .{ .part = part, .format = format, .dims = .{ .n = 1, .d = .{ a, 0, 0 } } };
}

fn mat(part: Part, format: Format, a: u64, b: u64) Expected {
    return .{ .part = part, .format = format, .dims = .{ .n = 2, .d = .{ a, b, 0 } } };
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// `name` without `prefix`, or null.
fn after(name: []const u8, prefix: []const u8) ?[]const u8 {
    return if (std.mem.startsWith(u8, name, prefix)) name[prefix.len..] else null;
}

/// A leading index and the rest after its dot: "12.mlp.gate.weight" gives 12 and "mlp.gate.weight".
fn indexed(name: []const u8) ?struct { i: u32, rest: []const u8 } {
    const dot = std.mem.indexOfScalar(u8, name, '.') orelse return null;
    const i = std.fmt.parseInt(u32, name[0..dot], 10) catch return null;
    return .{ .i = i, .rest = name[dot + 1 ..] };
}

/// One ModelOpt NVFP4 linear of `out` by `in`: e2m1 codes two to a byte, an e4m3 scale every 16 inputs, two scalars.
fn nvfp4(part: Part, leaf: []const u8, out: u64, in: u64) ?Expected {
    if (eq(leaf, "weight")) {
        var e = mat(part, .u8, out, in / 2);
        e.plain = .{ .n = 2, .d = .{ out, in, 0 } };
        return e;
    }
    if (eq(leaf, "weight_scale")) return mat(part, .e4m3, out, in / 16);
    if (eq(leaf, "weight_scale_2") or eq(leaf, "input_scale")) return .{ .part = part, .format = .f32, .dims = .{} };
    return null;
}

/// The three projections of a SwiGLU MLP of width `inter` over `width`, each an NVFP4 linear.
fn swiglu(part: Part, rest: []const u8, width: u64, inter: u64) ?Expected {
    if (after(rest, "gate_proj.")) |leaf| return nvfp4(part, leaf, inter, width);
    if (after(rest, "up_proj.")) |leaf| return nvfp4(part, leaf, inter, width);
    if (after(rest, "down_proj.")) |leaf| return nvfp4(part, leaf, width, inter);
    return null;
}

fn attention(c: *const config.Text, i: u32, rest: []const u8) ?Expected {
    const h = c.heads(i);
    const H = c.hidden;
    if (eq(rest, "q_a_proj.weight")) return mat(.attention, .bf16, c.q_rank, H);
    if (eq(rest, "q_a_layernorm.weight")) return vec(.attention, .bf16, c.q_rank);
    if (eq(rest, "q_b_proj.weight")) return mat(.attention, .bf16, h.q * (h.nope + h.rope), c.q_rank);
    if (eq(rest, "kv_a_proj_with_mqa.weight")) return mat(.attention, .bf16, c.kv_rank + h.rope, H);
    if (eq(rest, "kv_a_layernorm.weight")) return vec(.attention, .bf16, c.kv_rank);
    if (eq(rest, "kv_b_proj.weight")) return mat(.attention, .bf16, h.q * (h.nope + h.v), c.kv_rank);
    if (eq(rest, "linear_gate.weight")) return mat(.attention, .bf16, h.q * h.v, H);
    if (eq(rest, "o_proj.weight")) return mat(.attention, .bf16, H, h.q * h.v);
    if (eq(rest, "post_proj_norm.weight")) return vec(.attention, .bf16, H);
    if (eq(rest, "learnable_sink_param")) return vec(.attention, .f32, h.q);
    const ix = after(rest, "compressed_indexer.") orelse return null;
    if (c.kinds[i] != .sparse) return null; // sliding layers have no indexer
    const D = c.index_dim;
    if (eq(ix, "intra_group_ape")) return mat(.indexer, .f32, c.compress, D);
    if (eq(ix, "key_gate_projection.weight") or eq(ix, "key_value_projection.weight")) return mat(.indexer, .f32, D, D);
    if (eq(ix, "head_weight_projection.weight")) return mat(.indexer, .bf16, c.index_heads, H);
    if (eq(ix, "query_projection.weight")) return mat(.indexer, .bf16, c.index_heads * D, c.q_rank);
    if (eq(ix, "token_key_projection.weight")) return mat(.indexer, .bf16, D, H);
    if (eq(ix, "key_norm.weight") or eq(ix, "token_key_norm.weight")) return vec(.indexer, .bf16, D);
    return null;
}

/// One hyper-connection read (before attention or before the MLP): 2 * hc rows over the hc streams.
fn streams(c: *const config.Text, rest: []const u8) ?Expected {
    if (eq(rest, "hc_base")) return vec(.streams, .f32, 2 * c.hc);
    if (eq(rest, "hc_fn")) return mat(.streams, .f32, 2 * c.hc, @as(u64, c.hc) * c.hidden);
    if (eq(rest, "hc_scale")) return vec(.streams, .f32, 2);
    return null;
}

fn mlp(c: *const config.Text, i: u32, rest: []const u8) ?Expected {
    const H = c.hidden;
    if (!c.moe[i]) {
        if (eq(rest, "post_fc2_norm.weight")) return vec(.dense_mlp, .bf16, H);
        return swiglu(.dense_mlp, rest, H, c.dense_inter);
    }
    if (eq(rest, "gate.weight")) return mat(.router, .bf16, c.experts, H);
    if (eq(rest, "gate.e_score_correction_bias")) return vec(.router, .f32, c.experts);
    if (eq(rest, "routed_expert_down_proj.weight")) return mat(.experts, .bf16, c.latent, H);
    if (eq(rest, "routed_expert_up_proj.weight")) return mat(.experts, .bf16, H, c.latent);
    if (eq(rest, "combine_output_norm.weight")) return vec(.experts, .bf16, H);
    if (after(rest, "shared_experts.")) |s| return swiglu(.shared, s, H, @as(u64, c.expert_inter) * c.shared);
    const fused: Dims = .{ .n = 3, .d = .{ c.experts, 2 * @as(u64, c.expert_inter), c.latent } };
    if (eq(rest, "experts.gate_up_proj")) return .{ .part = .experts, .format = .bf16, .dims = fused };
    if (eq(rest, "experts.down_proj")) return .{ .part = .experts, .format = .bf16, .dims = .{ .n = 3, .d = .{ c.experts, c.latent, c.expert_inter } } };
    const e = indexed(after(rest, "experts.") orelse return null) orelse return null;
    if (e.i >= c.experts) return null;
    return swiglu(.experts, e.rest, c.latent, c.expert_inter);
}

fn engram(c: *const config.Config, rest: []const u8) ?Expected {
    const H = c.text.hidden;
    const g = c.engram;
    const heads: u64 = @as(u64, g.max_ngram - 1) * g.heads; // every n-gram size has `heads` hash heads
    if (eq(rest, "multi_head_embedding.embedding.weight")) {
        // Tencent's own releases keep BF16 rows under engram_fp8_embedding; the reference rounds them to fp8 as it runs
        var e = mat(.engram, if (g.fp8) .e4m3 else .bf16, 0, g.dim);
        e.plain = e.dims;
        return e;
    }
    if (eq(rest, "multi_head_embedding.embedding.weight_scale_inv")) return if (g.fp8) vec(.engram, .bf16, 0) else null;
    if (eq(rest, "lookup_table")) return vec(.engram, .i32, 0);
    if (eq(rest, "value_proj.weight")) return mat(.engram, .bf16, H, heads * g.dim);
    if (eq(rest, "value_proj.bias") or eq(rest, "unified_pre_gamma")) return vec(.engram, .bf16, H);
    if (after(rest, "lookup_per_head_norms.")) |r| {
        const x = indexed(r) orelse return null;
        return if (x.i < heads and eq(x.rest, "weight")) vec(.engram, .bf16, g.dim) else null;
    }
    // one of each for every residual stream
    inline for (.{ "key_projs.", "norm_key.", "norm_query.", "unified_per_hc_scales." }, 0..) |prefix, which| {
        if (after(rest, prefix)) |r| {
            if (which == 3) {
                const s = std.fmt.parseInt(u32, r, 10) catch return null;
                return if (s < c.text.hc) vec(.engram, .bf16, H) else null;
            }
            const x = indexed(r) orelse return null;
            if (x.i >= c.text.hc) return null;
            if (which == 0 and eq(x.rest, "weight")) return mat(.engram, .bf16, H, heads * g.dim);
            if (which == 0 and eq(x.rest, "bias")) return vec(.engram, .bf16, H);
            if (which != 0 and eq(x.rest, "weight")) return vec(.engram, .bf16, H);
            return null;
        }
    }
    return null;
}

fn backbone(c: *const config.Config, i: u32, rest: []const u8) ?Expected {
    const x = &c.text;
    if (i >= x.layers) return null;
    if (eq(rest, "input_layernorm.weight") or eq(rest, "post_attention_layernorm.weight")) return vec(.top, .bf16, x.hidden);
    if (after(rest, "self_attn.")) |r| return attention(x, i, r);
    if (after(rest, "hc_attn_layer.hc_pre.")) |r| return streams(x, r);
    if (after(rest, "hc_mlp_layer.hc_pre.")) |r| return streams(x, r);
    if (after(rest, "mlp.")) |r| return mlp(x, i, r);
    if (after(rest, "engram_layer.")) |r| return if (x.engram[i]) engram(c, r) else null;
    return null;
}

/// What the config says `name` must be, or null for a tensor the family does not know.
pub fn expect(c: *const config.Config, name: []const u8) ?Expected {
    const x = &c.text;
    if (eq(name, "lm_head.weight")) {
        var e = mat(.top, .f32, x.vocab, x.hidden);
        e.plain = e.dims;
        return e;
    }
    if (after(name, "vit.")) |_| return if (c.has_vision) .{ .part = .vision, .format = .bf16, .dims = .{}, .checked = false } else null;
    const m = after(name, "model.") orelse return null;
    if (eq(m, "embed_tokens.weight")) return mat(.top, .bf16, x.vocab, x.hidden);
    if (eq(m, "norm.weight")) return vec(.top, .bf16, x.hidden);
    if (eq(m, "hc_head.hc_head_base")) return vec(.streams, .f32, x.hc);
    if (eq(m, "hc_head.hc_head_fn")) return mat(.streams, .f32, x.hc, @as(u64, x.hc) * x.hidden);
    if (eq(m, "hc_head.hc_head_scale")) return vec(.streams, .f32, 1);
    if (after(m, "dfly_layers.")) |_| return if (c.drafter != null) .{ .part = .drafter, .format = .bf16, .dims = .{}, .checked = false } else null;
    const l = indexed(after(m, "layers.") orelse return null) orelse return null;
    return backbone(c, l.i, l.rest);
}

/// safetensors' dtype string as a Format, or null for one the family stores nothing in.
pub fn formatOf(dtype: []const u8) ?Format {
    inline for (.{ .{ "BF16", .bf16 }, .{ "F32", .f32 }, .{ "F8_E4M3", .e4m3 }, .{ "U8", .u8 }, .{ "I32", .i32 } }) |p| {
        if (eq(dtype, p[0])) return p[1];
    }
    return null;
}

/// One header entry against the config: an unknown name, another format and another shape each refuse by name.
pub fn check(c: *const config.Config, name: []const u8, dtype: []const u8, dims: []const u64, why: *config.Why) !Part {
    const want = expect(c, name) orelse {
        why.set("{s}: not a Hy4-flash tensor for this config", .{name});
        return error.UnknownTensor;
    };
    if (!want.checked) return want.part;
    const f = formatOf(dtype);
    if (want.plain) |plain| if (f == .bf16 and f != want.format) {
        if (plain.matches(dims)) return want.part;
        why.set("{s} has another shape than its config gives it", .{name});
        return error.TensorShape;
    };
    if (f != want.format) {
        why.set("{s} is stored as {s}: the family reads {s} there", .{ name, dtype, @tagName(want.format) });
        return error.TensorFormat;
    }
    if (!want.dims.matches(dims)) {
        why.set("{s} has another shape than its config gives it", .{name});
        return error.TensorShape;
    }
    return want.part;
}
