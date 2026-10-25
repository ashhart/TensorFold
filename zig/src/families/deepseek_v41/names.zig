//! The pack's tensor names, each one's TP split and expected shape, plus the names the text engine leaves unread.

const std = @import("std");
const Config = @import("config.zig").Config;
const List = @import("config.zig").List;
const Split = @import("core").exl3_layout.Split;
const DType = @import("core").safetensors.DType;

/// "layers.L" for the backbone, "mtp.i" for DSpark block i (layer = layers + i).
pub fn blockPrefix(c: *const Config, layer: u32, buf: []u8) ![]const u8 {
    return if (c.isBackbone(layer)) std.fmt.bufPrint(buf, "layers.{d}", .{layer}) else std.fmt.bufPrint(buf, "mtp.{d}", .{layer - c.layers});
}

/// An EXL3 projection of a block, by the forward's name for it.
pub const Proj = enum { wq_a, wkv, wq_b, wo_a, wo_b, comp_wkv, comp_wgate, ix_wq_b, ix_wk, shared_w1, shared_w2, shared_w3, engram_wkv };

/// A block's EXL3 group: its name under the block prefix, how ranks split it, and its K x N (0: not checked).
pub const GroupSpec = struct {
    proj: Proj,
    suffix: []const u8,
    /// wo_a: the global group index this slice is
    slice: u32 = 0,
    split: Split,
    k: u32 = 0,
    n: u32 = 0,
};

pub const Groups = List(GroupSpec, 32);

/// The block's EXL3 groups this rank loads, in the loader's order (wo_a: only this rank's groups).
pub fn blockGroups(c: *const Config, layer: u32, rank: u32, world: u32, has_shared: bool) !Groups {
    var g: Groups = .{};
    const per_group_k = c.heads / c.o_groups * c.head_dim;
    try g.push(.{ .proj = .wq_a, .suffix = "attn.wq_a", .split = .whole, .k = c.hidden, .n = c.q_lora });
    try g.push(.{ .proj = .wkv, .suffix = "attn.wkv", .split = .whole, .k = c.hidden, .n = c.head_dim });
    try g.push(.{ .proj = .wq_b, .suffix = "attn.wq_b", .split = .col, .k = c.q_lora, .n = c.heads * c.head_dim });
    const gl = c.o_groups / world;
    var s = rank * gl;
    while (s < (rank + 1) * gl) : (s += 1) try g.push(.{ .proj = .wo_a, .suffix = "attn.wo_a.slice", .slice = s, .split = .whole, .k = per_group_k, .n = c.o_lora });
    try g.push(.{ .proj = .wo_b, .suffix = "attn.wo_b", .split = .row, .k = c.o_groups * c.o_lora, .n = c.hidden });
    if (c.isBackbone(layer) and c.isKvSource(layer)) {
        try g.push(.{ .proj = .comp_wkv, .suffix = "attn.compressor.wkv", .split = .whole, .k = c.hidden, .n = c.head_dim });
        if (c.compressRatio(layer) > 1) try g.push(.{ .proj = .comp_wgate, .suffix = "attn.compressor.wgate", .split = .whole, .k = c.hidden, .n = c.head_dim });
        try g.push(.{ .proj = .ix_wk, .suffix = "attn.indexer.wk", .split = .whole, .k = c.head_dim, .n = c.index_dim });
    }
    if (c.isBackbone(layer) and c.isIndexSource(layer))
        try g.push(.{ .proj = .ix_wq_b, .suffix = "attn.indexer.wq_b", .split = .whole, .k = c.q_lora, .n = c.index_heads * c.index_dim });
    if (has_shared) {
        try g.push(.{ .proj = .shared_w1, .suffix = "ffn.shared_experts.w1", .split = .col, .k = c.hidden, .n = c.expert_width });
        try g.push(.{ .proj = .shared_w2, .suffix = "ffn.shared_experts.w2", .split = .row, .k = c.expert_width, .n = c.hidden });
        try g.push(.{ .proj = .shared_w3, .suffix = "ffn.shared_experts.w3", .split = .col, .k = c.hidden, .n = c.expert_width });
    }
    if (c.isEngram(layer)) try g.push(.{ .proj = .engram_wkv, .suffix = "engram.wkv", .split = .whole, .k = c.hashCols() * c.engram_head_dim, .n = (c.engram_max_ngram + 1) * c.hidden });
    return g;
}

/// The group's full name under `block`.
pub fn groupName(spec: GroupSpec, block: []const u8, buf: []u8) ![]const u8 {
    return if (spec.proj == .wo_a) std.fmt.bufPrint(buf, "{s}.{s}.{d}", .{ block, spec.suffix, spec.slice }) else std.fmt.bufPrint(buf, "{s}.{s}", .{ block, spec.suffix });
}

/// Routed experts' projections: gate (w1) and up (w3) split by columns, down (w2) by rows.
pub const ExpertProj = enum {
    w1,
    w2,
    w3,

    pub fn split(p: ExpertProj) Split {
        return if (p == .w2) .row else .col;
    }

    /// (K, N) of one expert's full matrix.
    pub fn kn(p: ExpertProj, c: *const Config) [2]u32 {
        return if (p == .w2) .{ c.expert_width, c.hidden } else .{ c.hidden, c.expert_width };
    }
};

pub fn expertName(block: []const u8, e: u32, p: ExpertProj, buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.ffn.experts.{d}.{t}", .{ block, e, p });
}

/// Which dtypes a native tensor may be stored in; the loader converts to what the forward holds.
pub const Store = enum {
    param,
    /// norms: bf16 in the pack (fp32 in the forward), fp32 accepted
    norm,
    /// fp32 only (sinks, hc_*)
    f32,
    /// router bias: fp32, or fp16 / bf16 widened exactly to fp32 (some packs store the MTP biases in fp16)
    bias,
    /// fp16 or bf16, rounded to bf16 once (router gate, indexer weights_proj, Markov heads)
    half,

    pub fn accepts(s: Store, d: DType) bool {
        return switch (s) {
            .param => d == .bf16 or d == .f32 or d == .f16,
            .norm => d == .bf16 or d == .f32,
            .f32 => d == .f32,
            .bias => d == .f32 or d == .f16 or d == .bf16,
            .half => d == .bf16 or d == .f16,
        };
    }
};

/// How a rank holds a native tensor.
pub const Slice = enum { whole, heads, vocab_rows };

pub const NativeSpec = struct {
    suffix: []const u8,
    store: Store,
    shape: [2]u32,
    rank: u8,
    slice: Slice = .whole,

    fn v(suffix: []const u8, store: Store, n: u32) NativeSpec {
        return .{ .suffix = suffix, .store = store, .shape = .{ n, 0 }, .rank = 1 };
    }

    fn m(suffix: []const u8, store: Store, r: u32, cols: u32) NativeSpec {
        return .{ .suffix = suffix, .store = store, .shape = .{ r, cols }, .rank = 2 };
    }

    pub fn dims(s: *const NativeSpec) []const u32 {
        return s.shape[0..s.rank];
    }
};

pub const Natives = List(NativeSpec, 24);

/// A block's native tensors: norms, mHC mixers, the sink, compressor / indexer pieces, the router, Engram's q.k.
pub fn blockNatives(c: *const Config, layer: u32) !Natives {
    var n: Natives = .{};
    const hc = c.hc_mult;
    const mix = (2 + hc) * hc; // pre (hc), post (hc) and the hc x hc residual mix
    try n.push(.v("attn_norm.weight", .norm, c.hidden));
    try n.push(.v("ffn_norm.weight", .norm, c.hidden));
    inline for (.{ "attn", "ffn" }) |w| {
        try n.push(.m("hc_" ++ w ++ "_fn", .f32, mix, hc * c.hidden));
        try n.push(.v("hc_" ++ w ++ "_base", .f32, mix));
        try n.push(.v("hc_" ++ w ++ "_scale", .f32, 3));
    }
    try n.push(.v("attn.q_norm.weight", .norm, c.q_lora));
    try n.push(.v("attn.kv_norm.weight", .norm, c.head_dim));
    var sink = NativeSpec.v("attn.attn_sink", .f32, c.heads);
    sink.slice = .heads;
    try n.push(sink);
    if (c.isBackbone(layer) and c.isKvSource(layer)) {
        try n.push(.v("attn.compressor.norm.weight", .norm, c.head_dim));
        try n.push(.v("attn.indexer.k_norm.weight", .norm, c.index_dim));
    }
    if (c.isBackbone(layer) and c.isIndexSource(layer)) try n.push(.m("attn.indexer.weights_proj.weight", .half, c.index_heads, c.hidden));
    const ex = c.expertsOf(layer);
    try n.push(.m("ffn.gate.weight", .half, ex.count, c.hidden));
    try n.push(.v("ffn.gate.bias", .bias, ex.count));
    if (c.isEngram(layer)) {
        try n.push(.m("engram.q_weight", .param, hc, c.hidden));
        try n.push(.m("engram.k_weight", .param, hc, c.hidden));
    }
    return n;
}

/// Model-level tensors outside the blocks: this rank's vocabulary rows, the final norm (the head is an EXL3 group).
pub fn topNatives(c: *const Config) !Natives {
    var n: Natives = .{};
    var embed = NativeSpec.m("embed.weight", .half, c.vocab, c.hidden);
    embed.slice = .vocab_rows;
    try n.push(embed);
    try n.push(.v("norm.weight", .norm, c.hidden));
    return n;
}

pub const head = GroupSpec{ .proj = .wq_a, .suffix = "head", .split = .col };

/// DSpark's heads beside its blocks (the latest mtp.i that has each).
pub fn dsparkNatives(c: *const Config) !Natives {
    var n: Natives = .{};
    try n.push(.v("main_norm.weight", .norm, c.hidden));
    try n.push(.v("norm.weight", .norm, c.hidden));
    try n.push(.m("markov_head.embed.weight", .half, c.vocab, c.dspark_markov_rank));
    try n.push(.m("markov_head.head.weight", .half, c.vocab, c.dspark_markov_rank));
    return n;
}

/// The optional confidence head: fp32 values [D + r], any shape with that many.
pub const dspark_confidence = "confidence_head.proj.weight";
pub const dspark_main_proj = "main_proj";

/// Names the text engine never reads: vision tower, image markers, image routing bias, bf16 copies of EXL3 groups.
pub fn ignored(name: []const u8, has: anytype) bool {
    for ([_][]const u8{ "vision.", "aligner.", "image_" }) |p| if (std.mem.startsWith(u8, name, p)) return true;
    if (std.mem.endsWith(u8, name, ".ffn.gate.bias_vl")) return true;
    if (std.mem.endsWith(u8, name, ".weight")) {
        var buf: [256]u8 = undefined;
        const base = name[0 .. name.len - ".weight".len];
        const sib = std.fmt.bufPrint(&buf, "{s}.trellis", .{base}) catch return false;
        return has.has(sib);
    }
    return false;
}

const testing = std.testing;

test "the release's block 2: a ratio-2 KV and index source with its compressor gate" {
    const c: Config = .{};
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("mtp.1", try blockPrefix(&c, 41, &buf));
    const g = try blockGroups(&c, 2, 1, 2, true);
    var names: [32][]const u8 = undefined;
    var nbuf: [32][96]u8 = undefined;
    for (g.items(), 0..) |s, i| names[i] = try groupName(s, "layers.2", &nbuf[i]);
    // wo_a: rank 1 of 2 holds groups 4..7
    try testing.expectEqualStrings("layers.2.attn.wo_a.slice.4", names[3]);
    try testing.expectEqual(@as(usize, 3 + 4 + 1 + 3 + 1 + 3), g.len);
    try testing.expectEqual(@as(u32, 4096), g.items()[3].k);
    try testing.expectEqual(@as(u32, 8192), g.items()[7].k);
    // layer 20: ratio 1, no compressor gate; layer 24: reindex (own wq_b only); layer 30: reuse; DSpark blocks: neither
    const g20 = try blockGroups(&c, 20, 0, 2, true);
    for (g20.items()) |s| try testing.expect(s.proj != .comp_wgate);
    try testing.expectEqual(@as(usize, 3 + 4 + 1 + 1), (try blockGroups(&c, 24, 0, 2, false)).len);
    try testing.expectEqual(@as(usize, 3 + 4 + 1), (try blockGroups(&c, 30, 0, 2, false)).len);
    try testing.expectEqual(@as(usize, 3 + 4 + 1 + 3), (try blockGroups(&c, 41, 0, 2, true)).len);
    try testing.expect((try blockGroups(&c, 14, 0, 2, true)).items()[15].proj == .engram_wkv);
    try testing.expectEqualStrings("layers.3.ffn.experts.17.w2", try expertName("layers.3", 17, .w2, &buf));
}

test "natives and the ignore list" {
    const c: Config = .{};
    const n = try blockNatives(&c, 41);
    for (n.items()) |s| if (std.mem.eql(u8, s.suffix, "ffn.gate.weight")) try testing.expectEqual(@as(u32, 128), s.shape[0]);
    try testing.expectEqual(@as(u32, 24), (try blockNatives(&c, 0)).items()[2].shape[0]);
    const Has = struct {
        pub fn has(_: @This(), name: []const u8) bool {
            return std.mem.eql(u8, name, "layers.2.attn.indexer.wk.trellis");
        }
    };
    try testing.expect(ignored("layers.2.attn.indexer.wk.weight", Has{}));
    try testing.expect(!ignored("layers.2.attn.kv_norm.weight", Has{}));
    try testing.expect(ignored("vision.blocks.3.mlp.w1.weight", Has{}));
    try testing.expect(ignored("layers.9.ffn.gate.bias_vl", Has{}));
    try testing.expect(Store.half.accepts(.f16) and !Store.f32.accepts(.bf16));
    try testing.expect(Store.bias.accepts(.f16) and Store.bias.accepts(.bf16) and !Store.f32.accepts(.f16));
}
