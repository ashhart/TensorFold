//! GigaChat 4's weight map: every tensor the config implies, with dtype and shape, checked against safetensors headers.
const std = @import("std");
const st = @import("safetensors");
const config = @import("config.zig");
const norms = @import("norms.zig");
const plan = @import("plan.zig");

const Dtype = st.DType;
const Config = config.Config;

/// The side of the 128x128 blocks one F32 `weight_scale_inv` entry covers.
pub const fp8_block = 128;
/// The most expected tensors a report may hold (the 708B release has 95,616).
const max_expected = 1_000_000;

fn put(sink: anytype, comptime fmt: []const u8, args: anytype, dtype: Dtype, dims: []const u64) !void {
    var buf: [160]u8 = undefined;
    try sink.put(try std.fmt.bufPrint(&buf, fmt, args), dtype, dims);
}

fn blocks(n: u64) u64 {
    return std.math.divCeil(u64, n, fp8_block) catch unreachable;
}

/// A projection the FP8 release quantizes: E4M3 [rows, cols] with F32 scales per block, or BF16 [rows, cols] alone.
fn proj(c: *const Config, sink: anytype, comptime fmt: []const u8, args: anytype, rows: u64, cols: u64) !void {
    switch (c.storage) {
        .fp8 => {
            try put(sink, fmt ++ ".weight", args, .f8_e4m3, &.{ rows, cols });
            try put(sink, fmt ++ ".weight_scale_inv", args, .f32, &.{ blocks(rows), blocks(cols) });
        },
        .bf16 => try put(sink, fmt ++ ".weight", args, .bf16, &.{ rows, cols }),
    }
}

/// A tensor both releases keep in BF16.
fn bf(sink: anytype, comptime fmt: []const u8, args: anytype, dims: []const u64) !void {
    try put(sink, fmt, args, .bf16, dims);
}

/// GatedNorm over `n`: its weight and the two gate projections of norms.gate_rank.
fn gated(sink: anytype, comptime fmt: []const u8, args: anytype, n: u64) !void {
    try bf(sink, fmt ++ ".weight", args, &.{n});
    try bf(sink, fmt ++ ".gate_up_projection.weight", args, &.{ norms.gate_rank, n });
    try bf(sink, fmt ++ ".gate_down_projection.weight", args, &.{ n, norms.gate_rank });
}

const layer = "model.layers.{d}.";

/// The four GatedNorms of the pre_post layout around the mixer and the feed-forward.
fn block(c: *const Config, sink: anytype, i: u64) !void {
    inline for (.{ "input_layernorm", "post_self_attn_layernorm", "post_attention_layernorm", "post_feedforward_layernorm" }) |n|
        try gated(sink, layer ++ n, .{i}, c.hidden);
}

fn deltanet(c: *const Config, sink: anytype, i: u64) !void {
    const d = c.delta;
    const h: u64 = c.hidden;
    const keys: u64 = @as(u64, d.key_heads) * d.key_dim;
    const values: u64 = @as(u64, d.value_heads) * d.value_dim;
    const a = layer ++ "self_attn.";
    try proj(c, sink, a ++ "in_proj_qkvz", .{i}, 2 * keys + 2 * values, h);
    try bf(sink, a ++ "in_proj_ba.weight", .{i}, &.{ 2 * @as(u64, d.value_heads), h });
    try bf(sink, a ++ "conv1d.weight", .{i}, &.{ 2 * keys + values, 1, d.conv });
    try bf(sink, a ++ "A_log", .{i}, &.{d.value_heads});
    try bf(sink, a ++ "dt_bias", .{i}, &.{d.value_heads});
    try bf(sink, a ++ "norm.weight", .{i}, &.{d.value_dim});
    try proj(c, sink, a ++ "out_proj", .{i}, h, values);
}

/// MLA with its sigmoid output gate; `indexer` adds the DSA indexer's four tensors.
fn mla(c: *const Config, sink: anytype, i: u64, indexer: bool) !void {
    const m = c.mla;
    const h: u64 = c.hidden;
    const heads: u64 = m.heads;
    const a = layer ++ "self_attn.";
    try proj(c, sink, a ++ "q_a_proj", .{i}, m.q_lora, h);
    try gated(sink, a ++ "q_a_layernorm", .{i}, m.q_lora);
    try proj(c, sink, a ++ "q_b_proj", .{i}, heads * (@as(u64, m.nope) + m.rope), m.q_lora);
    try proj(c, sink, a ++ "kv_a_proj_with_mqa", .{i}, @as(u64, m.kv_lora) + m.rope, h);
    try gated(sink, a ++ "kv_a_layernorm", .{i}, m.kv_lora);
    try proj(c, sink, a ++ "kv_b_proj", .{i}, heads * (@as(u64, m.nope) + m.value), m.kv_lora);
    try proj(c, sink, a ++ "o_proj", .{i}, h, heads * m.value);
    try bf(sink, a ++ "gate_proj.weight", .{i}, &.{ heads * m.value, h });
    if (!indexer) return;
    const x = c.index;
    try proj(c, sink, a ++ "indexer.wq_b", .{i}, @as(u64, x.heads) * x.dim, m.q_lora);
    try bf(sink, a ++ "indexer.wk.weight", .{i}, &.{ x.dim, h });
    try bf(sink, a ++ "indexer.k_norm.weight", .{i}, &.{x.dim});
    try bf(sink, a ++ "indexer.weights_proj.weight", .{i}, &.{ x.heads, h });
}

fn dense(c: *const Config, sink: anytype, i: u64) !void {
    const h: u64 = c.hidden;
    const w: u64 = c.moe.dense_inter;
    try bf(sink, layer ++ "mlp.gate_proj.weight", .{i}, &.{ w, h });
    try bf(sink, layer ++ "mlp.up_proj.weight", .{i}, &.{ w, h });
    try bf(sink, layer ++ "mlp.down_proj.weight", .{i}, &.{ h, w });
}

/// The sigmoid router with its correction bias, the shared expert (BF16) and the routed experts (quantized).
fn moe(c: *const Config, sink: anytype, i: u64) !void {
    const h: u64 = c.hidden;
    const m = c.moe;
    const shared: u64 = @as(u64, m.inter) * m.shared;
    try bf(sink, layer ++ "mlp.gate.weight", .{i}, &.{ m.experts, h });
    try bf(sink, layer ++ "mlp.gate.e_score_correction_bias", .{i}, &.{m.experts});
    try bf(sink, layer ++ "mlp.shared_experts.gate_proj.weight", .{i}, &.{ shared, h });
    try bf(sink, layer ++ "mlp.shared_experts.up_proj.weight", .{i}, &.{ shared, h });
    try bf(sink, layer ++ "mlp.shared_experts.down_proj.weight", .{i}, &.{ h, shared });
    for (0..m.experts) |e| {
        const ex = .{ i, @as(u64, e) };
        try proj(c, sink, layer ++ "mlp.experts.{d}.gate_proj", ex, m.inter, h);
        try proj(c, sink, layer ++ "mlp.experts.{d}.up_proj", ex, m.inter, h);
        try proj(c, sink, layer ++ "mlp.experts.{d}.down_proj", ex, h, m.inter);
    }
}

/// The embeddings, final GatedNorm, LM head and every decoder layer.
pub fn backbone(c: *const Config, sink: anytype) !void {
    const p = plan.build(c);
    var moe_layers: u64 = 0;
    for (p.layers[0..p.n]) |l| moe_layers += @intFromBool(l.ffn == .moe);
    if (moe_layers * @as(u64, c.moe.experts) * 6 > max_expected) return error.WeightReportTooLarge;
    const h: u64 = c.hidden;
    try bf(sink, "model.embed_tokens.weight", .{}, &.{ c.vocab, h });
    try bf(sink, "lm_head.weight", .{}, &.{ c.vocab, h });
    try gated(sink, "model.norm", .{}, h);
    for (p.layers[0..p.n], 0..) |l, k| {
        const i: u64 = k;
        try block(c, sink, i);
        switch (l.mixer) {
            .gated_deltanet => try deltanet(c, sink, i),
            .mla_dsa => try mla(c, sink, i, true),
            .mla_dense => try mla(c, sink, i, false),
        }
        switch (l.ffn) {
            .dense => try dense(c, sink, i),
            .moe => try moe(c, sink, i),
        }
    }
}

/// The MTP layer: a decoder block (MLA keeping its indexer, a dense MLP), eh_proj, its norms, embedding and head.
pub fn mtp(c: *const Config, sink: anytype) !void {
    const i: u64 = plan.build(c).mtp.index;
    const h: u64 = c.hidden;
    try block(c, sink, i);
    try mla(c, sink, i, true);
    try dense(c, sink, i);
    try bf(sink, layer ++ "eh_proj.weight", .{i}, &.{ h, 2 * h });
    try gated(sink, layer ++ "enorm", .{i}, h);
    try gated(sink, layer ++ "hnorm", .{i}, h);
    try bf(sink, layer ++ "embed_tokens.weight", .{i}, &.{ c.vocab, h });
    try gated(sink, layer ++ "shared_head.norm", .{i}, h);
    try bf(sink, layer ++ "shared_head.head.weight", .{i}, &.{ c.vocab, h });
}

/// Every tensor of the release's model shards, as sink.put(name, dtype, dims); dflash_draft/ is not a model shard.
pub fn all(c: *const Config, sink: anytype) !void {
    try backbone(c, sink);
    try mtp(c, sink);
}

/// Every tensor of `from` added to `into`, for a checkpoint held in several files; a name in both is refused.
pub fn merge(arena: std.mem.Allocator, into: *st.Header, from: st.Header) !void {
    for (from.keys(), from.values()) |n, entry| {
        const slot = try into.getOrPut(arena, n);
        if (slot.found_existing) return error.DuplicateName;
        slot.value_ptr.* = entry;
    }
}

/// The map against a header: the tensors it expects, those missing or of another dtype or shape, and the unknown.
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
        if (k.report.expected >= max_expected) return error.WeightReportTooLarge;
        k.report.expected += 1;
        const at = k.entries.getIndex(n) orelse return k.report.missing.append(k.arena, try k.arena.dupe(u8, n));
        if (k.seen[at]) return error.DuplicateName;
        k.seen[at] = true;
        const e = k.entries.values()[at];
        const same = e.dtype == dtype and e.rank == dims.len and for (dims, e.shape[0..e.rank]) |want, got| {
            if (want != got) break false;
        } else true;
        if (!same) try k.report.wrong.append(k.arena, try k.arena.dupe(u8, n));
    }
};

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// `entries` (the model shards' headers merged) against the map; the lists live in `arena`, the unknown sorted.
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

/// The map's stored bytes, summed as sink.put sees them: the size of each tensor in its dtype.
pub const Sizer = struct {
    bytes: u64 = 0,
    tensors: u64 = 0,

    pub fn put(s: *Sizer, _: []const u8, dtype: Dtype, dims: []const u64) !void {
        var n: u64 = dtype.size();
        for (dims) |d| n = try std.math.mul(u64, n, d);
        s.bytes = try std.math.add(u64, s.bytes, n);
        s.tensors += 1;
    }
};
