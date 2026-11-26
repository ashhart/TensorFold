//! ERNIE 5.5's weight map: each tensor the config implies, with dtype and shape, checked against safetensors headers.
const std = @import("std");
const st = @import("safetensors");
const config = @import("config.zig");
const plan = @import("plan.zig");

const Dtype = st.DType;
const Config = config.Config;

/// A dimension the config does not give (the VHA post-mix rank, MultiMax's pieces): any size is accepted there.
pub const any = std.math.maxInt(u64);
/// The side of the 128x128 blocks an FP8 weight's F32 scales cover.
pub const fp8_block = 128;
/// The inputs one MXFP4 expert scale covers.
pub const mxfp4_group = 32;

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

fn put(sink: anytype, comptime fmt: []const u8, args: anytype, dtype: Dtype, dims: []const u64) !void {
    var buf: [160]u8 = undefined;
    try sink.put(try name(&buf, fmt, args), dtype, dims);
}

/// An FP8 E4M3 weight [rows, cols] and its F32 scales, one per 128x128 block; `scale` is the scale's name suffix.
fn fp8(sink: anytype, comptime fmt: []const u8, args: anytype, comptime scale: []const u8, rows: u64, cols: u64) !void {
    try put(sink, fmt, args, .f8_e4m3, &.{ rows, cols });
    try put(sink, fmt ++ scale, args, .f32, &.{ std.math.divCeil(u64, rows, fp8_block) catch unreachable, std.math.divCeil(u64, cols, fp8_block) catch unreachable });
}

fn weight(sink: anytype, comptime fmt: []const u8, args: anytype, rows: u64, cols: u64) !void {
    try fp8(sink, fmt ++ ".weight", args, "_scale_inv", rows, cols);
}

/// The two hyper-connection mixers around a block: pre and post weights per stream and a stream-by-stream residual.
fn hyper(c: *const Config, sink: anytype, comptime pre: []const u8, layer: usize) !void {
    const args = .{layer};
    const s: u64 = c.hc.streams;
    const mix = 2 * s + s * s;
    inline for (.{ "self_attention_hyper_connection", "mlp_hyper_connection" }) |hc| {
        inline for (.{ "alpha_pre", "alpha_post", "alpha_res" }) |a| try put(sink, pre ++ hc ++ "." ++ a, args, .f32, &.{1});
        try put(sink, pre ++ hc ++ ".bias", args, .f32, &.{mix});
        try put(sink, pre ++ hc ++ ".mapping_proj.weight", args, .f32, &.{ s * c.hidden, mix });
    }
}

/// An HCA layer's attention; the compressor is the target's, the draft layers attend in their window alone.
fn hca(c: *const Config, sink: anytype, comptime pre: []const u8, layer: usize, ratio: ?u32) !void {
    const args = .{layer};
    const h: u64 = c.hidden;
    const a = c.hca;
    const grouped: u64 = @as(u64, a.o_groups) * a.o_lora;
    try weight(sink, pre ++ "self_attn.linear_q_down_proj", args, a.q_lora, h);
    try put(sink, pre ++ "self_attn.q_layernorm.weight", args, .bf16, &.{a.q_lora});
    try weight(sink, pre ++ "self_attn.linear_q_up_proj", args, @as(u64, a.heads) * a.head_dim, a.q_lora);
    try weight(sink, pre ++ "self_attn.linear_kv_proj", args, a.head_dim, h);
    try put(sink, pre ++ "self_attn.kv_layernorm.weight", args, .bf16, &.{a.head_dim});
    try fp8(sink, pre ++ "self_attn.linear_o_group_proj", args, "_scale_inv", grouped, @as(u64, a.heads / a.o_groups) * a.head_dim);
    try weight(sink, pre ++ "self_attn.o_proj", args, h, grouped);
    try weight(sink, pre ++ "self_attn.gate_proj", args, grouped, a.q_lora);
    try put(sink, pre ++ "self_attn.core_attention.attn_sink", args, .f32, &.{a.heads});
    inline for (.{ "U", "V" }) |m| try put(sink, pre ++ "self_attn.vha_postmix_" ++ m, args, .bf16, &.{ a.heads, any });
    const r = ratio orelse return;
    try put(sink, pre ++ "self_attn.core_attention.compressor.ape", args, .f32, &.{ r, a.head_dim });
    inline for (.{ "linear_wkv", "linear_wgate" }) |w| try put(sink, pre ++ "self_attn.core_attention.compressor." ++ w ++ ".weight", args, .bf16, &.{ a.head_dim, h });
    try put(sink, pre ++ "self_attn.core_attention.compressor.norm.weight", args, .bf16, &.{a.head_dim});
}

/// A DSA layer's MLA attention with its indexer.
fn dsa(c: *const Config, sink: anytype, comptime pre: []const u8, layer: usize) !void {
    const args = .{layer};
    const h: u64 = c.hidden;
    const m = c.mla;
    const ix = c.index;
    try weight(sink, pre ++ "self_attn.q_a_proj", args, m.q_lora, h);
    try put(sink, pre ++ "self_attn.q_a_layernorm.weight", args, .bf16, &.{m.q_lora});
    try weight(sink, pre ++ "self_attn.q_b_proj", args, @as(u64, m.heads) * (@as(u64, m.nope) + m.rope), m.q_lora);
    try weight(sink, pre ++ "self_attn.kv_a_proj_with_mqa", args, @as(u64, m.kv_lora) + m.rope, h);
    try put(sink, pre ++ "self_attn.kv_a_layernorm.weight", args, .bf16, &.{m.kv_lora});
    try weight(sink, pre ++ "self_attn.kv_b_proj", args, @as(u64, m.heads) * (@as(u64, m.nope) + m.value), m.kv_lora);
    try weight(sink, pre ++ "self_attn.o_proj", args, h, @as(u64, m.heads) * m.value);
    try weight(sink, pre ++ "self_attn.gate_proj", args, @as(u64, m.heads) * m.value, m.q_lora);
    try put(sink, pre ++ "self_attn.core_attention.softmax_offset", args, .bf16, &.{m.heads});
    inline for (.{ "U", "V" }) |x| try put(sink, pre ++ "self_attn.vha_postmix_" ++ x, args, .bf16, &.{ m.heads, any });
    const idx = pre ++ "self_attn.core_attention.indexer.";
    try weight(sink, idx ++ "wq_b", args, @as(u64, ix.heads) * ix.dim, m.q_lora);
    try put(sink, idx ++ "wk.weight", args, .bf16, &.{ ix.dim, h });
    inline for (.{ "weight", "bias" }) |p| try put(sink, idx ++ "k_norm." ++ p, args, .bf16, &.{ix.dim});
    try put(sink, idx ++ "weights_proj.weight", args, .bf16, &.{ ix.heads, h });
}

/// The routed experts (MXFP4 in the latent space), the shared expert and the latent projections, routers and norm.
fn moe(c: *const Config, sink: anytype, comptime pre: []const u8, layer: usize) !void {
    const args = .{layer};
    const h: u64 = c.hidden;
    const m = c.moe;
    const e: u64 = m.experts;
    const moe_pre = pre ++ "block_sparse_moe.";
    inline for (.{ "gate.weight", "gate.weight_1" }) |g| try put(sink, moe_pre ++ g, args, .bf16, &.{ e, h });
    try put(sink, moe_pre ++ "gate.routed_scaling_factor_param", args, .f32, &.{e});
    try put(sink, moe_pre ++ "e_score_correction_bias", args, .f32, &.{e});
    try weight(sink, moe_pre ++ "fc1_latent_proj", args, m.latent, h);
    try weight(sink, moe_pre ++ "fc2_latent_proj", args, h, m.latent);
    try put(sink, moe_pre ++ "latent_norm.weight", args, .bf16, &.{m.latent});
    inline for (.{ "w1", "w3" }) |w| try weight(sink, moe_pre ++ "shared_experts." ++ w, args, m.inter, h);
    try weight(sink, moe_pre ++ "shared_experts.w2", args, h, m.inter);
    for (0..m.experts) |x| {
        const ex = .{ layer, x };
        const rows = [3][2]u64{ .{ m.inter, m.latent }, .{ m.latent, m.inter }, .{ m.inter, m.latent } };
        inline for (.{ "w1", "w2", "w3" }, 0..) |w, n| {
            const r = rows[n][0];
            const k = rows[n][1];
            try put(sink, moe_pre ++ "experts.{d}." ++ w ++ ".weight", ex, .i8, &.{ r, k / 2 });
            try put(sink, moe_pre ++ "experts.{d}." ++ w ++ ".weight_scale_inv", ex, .f8_e8m0, &.{ r, k / mxfp4_group });
        }
    }
}

fn dense(c: *const Config, sink: anytype, comptime pre: []const u8, layer: usize) !void {
    const args = .{layer};
    const h: u64 = c.hidden;
    inline for (.{ "gate_proj", "up_proj" }) |w| try weight(sink, pre ++ "mlp." ++ w, args, c.moe.dense_inter, h);
    try weight(sink, pre ++ "mlp.down_proj", args, h, c.moe.dense_inter);
}

/// The head that folds the streams into one: per-stream weights from a projection of all of them.
fn contract(c: *const Config, sink: anytype, comptime pre: []const u8) !void {
    const s: u64 = c.hc.streams;
    try put(sink, pre ++ "mhc_contract.hc_head_fn", .{}, .f32, &.{ s * c.hidden, s });
    try put(sink, pre ++ "mhc_contract.hc_head_base", .{}, .f32, &.{s});
    try put(sink, pre ++ "mhc_contract.hc_head_scale", .{}, .f32, &.{1});
}

fn block(c: *const Config, sink: anytype, comptime pre: []const u8, layer: usize) !void {
    const args = .{layer};
    inline for (.{ "input_layernorm", "post_attention_layernorm" }) |n| try put(sink, pre ++ n ++ ".weight", args, .bf16, &.{c.hidden});
    try hyper(c, sink, pre, layer);
}

/// The bundled DSpark draft: window-attention layers with routed experts, the taps' projection and its Markov head.
fn draft(c: *const Config, p: *const plan.Plan, sink: anytype) !void {
    const h: u64 = c.hidden;
    const d = c.dspark;
    for (0..p.draft_layers) |i| {
        try block(c, sink, "model.mtp.{d}.", i);
        try hca(c, sink, "model.mtp.{d}.", i, null);
        try moe(c, sink, "model.mtp.{d}.", i);
    }
    try weight(sink, "model.mtp.feature_projection", .{}, h, d.taps_n * h);
    try put(sink, "model.mtp.feature_norm.weight", .{}, .bf16, &.{h});
    try put(sink, "model.mtp.norm.weight", .{}, .bf16, &.{h});
    try put(sink, "model.mtp.markov_head.markov_w1.weight", .{}, .bf16, &.{ d.rank, c.vocab });
    try put(sink, "model.mtp.markov_head.markov_w2.weight", .{}, .bf16, &.{ c.vocab, d.rank });
    try put(sink, "model.mtp.confidence_head.projection.weight", .{}, .bf16, &.{ 1, h + d.rank });
    try contract(c, sink, "model.mtp.");
}

/// Every tensor of the release's model shards the config implies, as sink.put(name, dtype, dims).
pub fn all(c: *const Config, sink: anytype) !void {
    const p = plan.build(c);
    const h: u64 = c.hidden;
    try put(sink, "model.embed_tokens.weight", .{}, .bf16, &.{ c.vocab, h });
    try put(sink, "lm_head.weight", .{}, .bf16, &.{ c.vocab, h });
    try put(sink, "model.norm.weight", .{}, .bf16, &.{h});
    inline for (.{ "ranges", "ts" }) |m| try put(sink, "model.lm_head.multimax_" ++ m, .{}, .bf16, &.{any});
    try contract(c, sink, "model.");
    for (p.layers[0..p.n], 0..) |l, i| {
        try block(c, sink, "model.layers.{d}.", i);
        switch (l.attention) {
            .hca => try hca(c, sink, "model.layers.{d}.", i, l.ratio),
            .dsa => try dsa(c, sink, "model.layers.{d}.", i),
        }
        switch (l.ffn) {
            .dense => try dense(c, sink, "model.layers.{d}.", i),
            .moe => try moe(c, sink, "model.layers.{d}.", i),
        }
    }
    try draft(c, &p, sink);
}

/// Every tensor of `from` added to `into`, for a checkpoint held in several files; a name in both is refused.
pub fn merge(arena: std.mem.Allocator, into: *st.Header, from: st.Header) !void {
    for (from.keys(), from.values()) |n, entry| {
        const slot = try into.getOrPut(arena, n);
        if (slot.found_existing) return error.DuplicateName;
        slot.value_ptr.* = entry;
    }
}

/// The map against a header: tensors it expects, those missing or held in another dtype or shape, and extras.
pub const Report = struct {
    expected: usize = 0,
    missing: std.ArrayList([]const u8) = .empty,
    wrong: std.ArrayList([]const u8) = .empty,
    unknown: std.ArrayList([]const u8) = .empty,

    pub fn clean(r: *const Report) bool {
        return r.missing.items.len + r.wrong.items.len + r.unknown.items.len == 0;
    }
};

const Checker = struct {
    arena: std.mem.Allocator,
    entries: st.Header,
    seen: []bool,
    report: *Report,

    pub fn put(k: *Checker, n: []const u8, dtype: Dtype, dims: []const u64) !void {
        k.report.expected += 1;
        const at = k.entries.getIndex(n) orelse return k.report.missing.append(k.arena, try k.arena.dupe(u8, n));
        if (k.seen[at]) return error.DuplicateName;
        k.seen[at] = true;
        const e = k.entries.values()[at];
        var same = e.dtype == dtype and e.rank == dims.len;
        if (same) for (dims, e.shape[0..e.rank]) |want, got| {
            if (want != any and want != got) same = false;
        };
        if (!same) try k.report.wrong.append(k.arena, try k.arena.dupe(u8, n));
    }
};

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// `entries` (the model shards' headers merged) against the map; the lists live in `arena`.
pub fn check(arena: std.mem.Allocator, c: *const Config, entries: st.Header) !Report {
    const seen = try arena.alloc(bool, entries.count());
    @memset(seen, false);
    var r: Report = .{};
    var k: Checker = .{ .arena = arena, .entries = entries, .seen = seen, .report = &r };
    try all(c, &k);
    for (entries.keys(), seen) |n, was| if (!was) try r.unknown.append(arena, n);
    std.mem.sort([]const u8, r.unknown.items, {}, lessThan);
    return r;
}
