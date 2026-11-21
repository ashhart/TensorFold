//! Inkling 2 weight maps: each tensor the config implies, with dtype and shape, checked against headers.
const std = @import("std");
const st = @import("safetensors");
const config = @import("config.zig");
const plan = @import("plan.zig");

const Dtype = st.DType;
const Text = config.Text;

/// A dimension the map does not derive from the config: any size is accepted there.
pub const any = std.math.maxInt(u64);

/// Which file's tensors: the backbone and vision tower (model.safetensors), or the draft head (mtp.safetensors).
pub const File = enum { model, mtp };

fn put(sink: anytype, comptime fmt: []const u8, args: anytype, dtype: Dtype, dims: []const u64) !void {
    var buf: [128]u8 = undefined;
    try sink.put(try std.fmt.bufPrint(&buf, fmt, args), dtype, dims);
}

fn attention(c: *const Text, i: u32, sink: anytype) !void {
    const h: u64 = c.hidden;
    const hd = c.heads(i);
    const q: u64 = @as(u64, hd.q) * hd.dim;
    const kv: u64 = @as(u64, hd.kv) * hd.dim;
    const k: u64 = config.conv_taps;
    const L = .{i};
    try put(sink, "model.llm.layers.{d}.attn.wq_du.weight", L, .bf16, &.{ q, h });
    try put(sink, "model.llm.layers.{d}.attn.wk_dv.weight", L, .bf16, &.{ kv, h });
    try put(sink, "model.llm.layers.{d}.attn.wv_dv.weight", L, .bf16, &.{ kv, h });
    try put(sink, "model.llm.layers.{d}.attn.wr_du.weight", L, .bf16, &.{ @as(u64, hd.q) * c.d_rel, h });
    try put(sink, "model.llm.layers.{d}.attn.wo_ud.weight", L, .bf16, &.{ h, q });
    try put(sink, "model.llm.layers.{d}.attn.gate_attn.weight", L, .bf16, &.{ q, h });
    try put(sink, "model.llm.layers.{d}.attn.attn_sink_proj.weight", L, .bf16, &.{ hd.q, h });
    try put(sink, "model.llm.layers.{d}.attn.q_l2_clip.scale", L, .bf16, &.{ hd.q, hd.dim });
    try put(sink, "model.llm.layers.{d}.attn.rel_l2_clip.scale", L, .bf16, &.{ hd.q, c.d_rel });
    try put(sink, "model.llm.layers.{d}.attn.rel_logits_proj.proj", L, .bf16, &.{ c.d_rel, c.extent(i) });
    inline for (.{ "k", "v" }) |kind| {
        if (config.isGlobal(i)) {
            try put(sink, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.static_w", L, .bf16, &.{ k, kv });
            try put(sink, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight_proj", L, .bf16, &.{ @as(u64, hd.kv) * k, h });
        } else {
            try put(sink, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight", L, .bf16, &.{ kv, 1, k });
        }
    }
    if (config.isGlobal(i)) {
        inline for (.{ "q", "k", "v" }) |kind| try put(sink, "model.llm.layers.{d}." ++ kind ++ "_input_norm.weight", L, .bf16, &.{h});
    } else {
        try put(sink, "model.llm.layers.{d}.attn_norm.weight", L, .bf16, &.{h});
    }
}

fn experts(c: *const Text, i: u32, prec: plan.Precision, sink: anytype) !void {
    const h: u64 = c.hidden;
    const m: u64 = c.expert_inter;
    const e: u64 = c.experts;
    const L = .{i};
    try put(sink, "model.llm.layers.{d}.mlp.gate.weight", L, .bf16, &.{ e, h });
    try put(sink, "model.llm.layers.{d}.mlp.gate.bias", L, .f32, &.{e});
    // gate and up are one tensor, w13: 2 * intermediate rows
    const rows = [2][2]u64{ .{ 2 * m, h }, .{ h, m } };
    inline for (.{ "w13", "w2" }, 0..) |w, n| {
        const r = rows[n][0];
        const k = rows[n][1];
        switch (prec) {
            .nvfp4 => {
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight", L, .u8, &.{ e, r, k / 2 });
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_scale", L, .f8_e4m3, &.{ e, r, k / 16 });
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_amax", L, .f32, &.{e});
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_outer_scale_denominator", L, .f32, &.{1});
            },
            .mxfp8 => {
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight", L, .f8_e4m3, &.{ e, r, k });
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_scale", L, .u8, &.{ e, r, k / 32 });
            },
        }
    }
}

fn controllers(c: *const Text, i: u32, sink: anytype) !void {
    const loop = c.loop orelse return;
    if (i < loop.core_start or i >= loop.core_end) return;
    const rank: u64 = loop.gate_rank;
    const h: u64 = c.hidden;
    const L = .{i};
    inline for (.{ "attn", "k", "mlp", "v" }) |kind| {
        try put(sink, "model.llm.layers.{d}.controllers." ++ kind ++ ".down_weight", L, .bf16, &.{ rank, 2 * h });
        try put(sink, "model.llm.layers.{d}.controllers." ++ kind ++ ".forget_weight", L, .bf16, &.{ h, rank });
        try put(sink, "model.llm.layers.{d}.controllers." ++ kind ++ ".gain_weight", L, .bf16, &.{ h, rank });
    }
}

fn backbone(c: *const Text, prec: []const plan.Precision, sink: anytype) !void {
    const h: u64 = c.hidden;
    try put(sink, "model.llm.embed.weight", .{}, .bf16, &.{ c.vocab, h });
    try put(sink, "model.llm.embed_norm.weight", .{}, .bf16, &.{h});
    try put(sink, "model.llm.norm.weight", .{}, .bf16, &.{h});
    try put(sink, "model.llm.unembed.weight", .{}, .bf16, &.{ c.vocab, h });
    try put(sink, "model.llm.attn_res.output_query", .{}, .f32, &.{h});
    if (c.loop) |loop| {
        const rows = 2 * @as(u64, loop.core_end - loop.core_start);
        try put(sink, "model.llm.middle_cycle_cross_pass_queries", .{}, .f32, &.{ rows, h });
        try put(sink, "model.llm.middle_cycle_cross_pass_kv_queries", .{}, .f32, &.{ rows, h });
    }
    for (0..c.layers) |k| {
        const i: u32 = @intCast(k);
        try put(sink, "model.llm.attn_res.mlp_queries.{d}", .{i}, .f32, &.{h});
        // queries.j feeds the attention input of layer j + 1: layer 0 reads the embedding alone
        if (i + 1 < c.layers) {
            const dims: []const u64 = if (config.isGlobal(i + 1)) &.{ 3, h } else &.{h};
            try put(sink, "model.llm.attn_res.attn_queries.{d}", .{i}, .f32, dims);
        }
        try attention(c, i, sink);
        try controllers(c, i, sink);
        try put(sink, "model.llm.layers.{d}.attn_sconv.weight", .{i}, .bf16, &.{ h, 1, config.conv_taps });
        try put(sink, "model.llm.layers.{d}.mlp_norm.weight", .{i}, .bf16, &.{h});
        try put(sink, "model.llm.layers.{d}.mlp_sconv.weight", .{i}, .bf16, &.{ h, 1, config.conv_taps });
        try experts(c, i, prec[i], sink);
    }
}

fn vision(v: config.Vision, sink: anytype) !void {
    const h: u64 = v.hidden;
    const ff: u64 = h * v.mlp_ratio;
    try put(sink, "model.visual.backbone.embeddings.class_embedding", .{}, .bf16, &.{h});
    try put(sink, "model.visual.backbone.embeddings.cls_position_embedding", .{}, .bf16, &.{h});
    try put(sink, "model.visual.backbone.embeddings.patch_embedding.weight", .{}, .bf16, &.{ h, 3 * @as(u64, v.patch) * v.patch });
    try put(sink, "model.visual.backbone.embeddings.position_embedding", .{}, .bf16, &.{ @as(u64, v.grid) * v.grid, h });
    inline for (.{ "weight", "bias" }) |p| try put(sink, "model.visual.backbone.ln_pre." ++ p, .{}, .bf16, &.{h});
    for (0..v.layers) |k| {
        const L = .{k};
        const pre = "model.visual.backbone.layers.{d}.";
        inline for (.{ "ln_1", "ln_2" }) |n| inline for (.{ "weight", "bias" }) |p| try put(sink, pre ++ n ++ "." ++ p, L, .bf16, &.{h});
        inline for (.{ "ls_1", "ls_2" }) |n| try put(sink, pre ++ n ++ ".gamma", L, .bf16, &.{h});
        inline for (.{ "q_proj", "k_proj", "v_proj", "out_proj" }) |n| {
            try put(sink, pre ++ "self_attn." ++ n ++ ".weight", L, .bf16, &.{ h, h });
            try put(sink, pre ++ "self_attn." ++ n ++ ".bias", L, .bf16, &.{h});
        }
        try put(sink, pre ++ "mlp.c_fc.weight", L, .bf16, &.{ ff, h });
        try put(sink, pre ++ "mlp.c_fc.bias", L, .bf16, &.{ff});
        try put(sink, pre ++ "mlp.c_proj.weight", L, .bf16, &.{ h, ff });
        try put(sink, pre ++ "mlp.c_proj.bias", L, .bf16, &.{h});
    }
    try put(sink, "model.visual.proj_in.weight", .{}, .bf16, &.{ v.proj_hidden, h * v.merge * v.merge });
    try put(sink, "model.visual.proj_out.weight", .{}, .bf16, &.{ v.text_hidden, v.proj_hidden });
}

fn draft(m: config.Mtp, text: *const Text, sink: anytype) !void {
    const h: u64 = text.hidden;
    const q: u64 = @as(u64, m.q_heads) * m.head_dim;
    const kv: u64 = @as(u64, m.kv_heads) * m.head_dim;
    const rank: u64 = m.rank;
    // One fusion row per draft layer plus the Markov head, one column per target visit tap.
    try put(sink, "fuse_weight", .{}, .bf16, &.{ m.layers + 1, m.taps_n });
    try put(sink, "mask_embed", .{}, .bf16, &.{h});
    try put(sink, "norm.weight", .{}, .bf16, &.{h});
    try put(sink, "markov_head.dflare_norm.weight", .{}, .bf16, &.{h});
    try put(sink, "markov_head.hidden_out.bias", .{}, .bf16, &.{h});
    try put(sink, "markov_head.hidden_out.weight", .{}, .bf16, &.{ h, m.rank });
    try put(sink, "markov_head.markov_w1.weight", .{}, .bf16, &.{ text.vocab, m.rank });
    try put(sink, "markov_head.rnn_proj.bias", .{}, .bf16, &.{3 * rank});
    try put(sink, "markov_head.rnn_proj.weight", .{}, .bf16, &.{ 3 * rank, 2 * h + 2 * rank });
    for (0..m.layers) |k| {
        const L = .{k};
        const pre = "layers.{d}.";
        inline for (.{ "dflare_norm", "input_layernorm", "post_attention_layernorm" }) |n| try put(sink, pre ++ n ++ ".weight", L, .bf16, &.{h});
        inline for (.{ "k_norm", "k_norm_ctx", "q_norm" }) |n| try put(sink, pre ++ "self_attn." ++ n ++ ".weight", L, .bf16, &.{m.head_dim});
        inline for (.{ "k_proj", "k_proj_ctx", "v_proj", "v_proj_ctx" }) |n| try put(sink, pre ++ "self_attn." ++ n ++ ".weight", L, .bf16, &.{ kv, h });
        try put(sink, pre ++ "self_attn.q_proj.weight", L, .bf16, &.{ q, h });
        try put(sink, pre ++ "self_attn.o_proj.weight", L, .bf16, &.{ h, q });
        try put(sink, pre ++ "mlp.gate_proj.weight", L, .bf16, &.{ m.inter, h });
        try put(sink, pre ++ "mlp.up_proj.weight", L, .bf16, &.{ m.inter, h });
        try put(sink, pre ++ "mlp.down_proj.weight", L, .bf16, &.{ h, m.inter });
    }
}

/// Every tensor of `file` the config implies, as sink.put(name, dtype, dims); `prec` is one precision per layer.
pub fn all(c: *const config.Config, prec: []const plan.Precision, file: File, sink: anytype) !void {
    switch (file) {
        .model => {
            try backbone(&c.text, prec, sink);
            try vision(c.vision, sink);
        },
        .mtp => try draft(c.mtp orelse return error.NoDraftHead, &c.text, sink),
    }
}

/// Each layer's expert precision from its w13 tensor: packed U8 is NVFP4, FP8 is MXFP8, anything else is refused.
pub fn precisions(entries: st.Header, layers: u32, out: []plan.Precision) !void {
    if (out.len < layers) return error.PrecisionCount;
    for (0..layers) |i| {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.w13_weight", .{i});
        const hit = entries.get(name) orelse return error.NoExperts;
        out[i] = switch (hit.dtype) {
            .u8 => .nvfp4,
            .f8_e4m3 => .mxfp8,
            else => return error.ExpertFormat,
        };
    }
}

/// Every tensor of `from` added to `into`, for a checkpoint held in several files; a name in both is refused.
pub fn merge(arena: std.mem.Allocator, into: *st.Header, from: st.Header) !void {
    for (from.keys(), from.values()) |name, entry| {
        const slot = try into.getOrPut(arena, name);
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

    pub fn put(k: *Checker, name: []const u8, dtype: Dtype, dims: []const u64) !void {
        k.report.expected += 1;
        const at = k.entries.getIndex(name) orelse return k.report.missing.append(k.arena, try k.arena.dupe(u8, name));
        if (k.seen[at]) return error.DuplicateName;
        k.seen[at] = true;
        const e = k.entries.values()[at];
        var same = e.dtype == dtype and e.rank == dims.len;
        if (same) for (dims, e.shape[0..e.rank]) |want, got| {
            if (want != any and want != got) same = false;
        };
        if (!same) try k.report.wrong.append(k.arena, try k.arena.dupe(u8, name));
    }
};

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// `entries` against the map of `file`; expert precisions come from the entries themselves (lists live in `arena`).
pub fn check(arena: std.mem.Allocator, c: *const config.Config, entries: st.Header, file: File) !Report {
    var prec: [config.max_layers]plan.Precision = undefined;
    if (file == .model) try precisions(entries, c.text.layers, &prec);
    const seen = try arena.alloc(bool, entries.count());
    @memset(seen, false);
    var r: Report = .{};
    var k: Checker = .{ .arena = arena, .entries = entries, .seen = seen, .report = &r };
    try all(c, &prec, file, &k);
    for (entries.keys(), seen) |name, was| if (!was) try r.unknown.append(arena, name);
    std.mem.sort([]const u8, r.unknown.items, {}, lessThan);
    return r;
}
