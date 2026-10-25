//! Shape and format ownership for dense Qwen4. Readers retain the checkpoint's codes and scales.
const std = @import("std");
const cfg = @import("config.zig");
const quant = @import("quant.zig");
const st = @import("core").safetensors;
const ngram = @import("hybrid_ngram");
pub const Namespace = enum {
    causal,
    conditional,
    mlx,
    pub fn prefix(n: Namespace) []const u8 {
        return switch (n) {
            .causal => "model.",
            .conditional => "model.language_model.",
            .mlx => "language_model.model.",
        };
    }
};
pub const Shape = struct { rank: u8, dims: [3]usize = @splat(1) };
pub const Role = enum { matrix, norm, conv, scalar_vector, embedding, integer_vector };
pub const Want = struct { role: Role, shape: Shape, recipe: quant.Recipe = .{} };
pub const Stats = struct { tensors: usize = 0, matrices: usize = 0 };

/// `sink.take(name, want)` receives every required tensor stem. The sink must finish before names are reused.
pub fn walk(c: *const cfg.Config, ns: Namespace, sink: anytype, p: *cfg.Problem) !void {
    var w = Walker(@TypeOf(sink)){ .c = c, .sink = sink, .p = p };
    const pre = ns.prefix();
    var b: [256]u8 = undefined;
    try w.matrix(try std.fmt.bufPrint(&b, "{s}embed_tokens", .{pre}), c.vocab, c.hidden, .embedding);
    if (!c.tie_embeddings) try w.matrix(if (ns == .mlx) "language_model.lm_head" else "lm_head", c.vocab, c.hidden, .matrix);
    try w.hc(try std.fmt.bufPrint(&b, "{s}hyper_connection_mixer", .{pre}), false);
    // The supplied model has a final grouped HC norm in the mixer, no separate transformer RMSNorm.
    for (0..c.layers) |i| {
        var lb: [128]u8 = undefined;
        const layer = try std.fmt.bufPrint(&lb, "{s}layers.{d}", .{ pre, i });
        for ([_][]const u8{ "attn_hyper_connection", "mlp_hyper_connection" }) |part|
            try w.hc(try std.fmt.bufPrint(&b, "{s}.{s}", .{ layer, part }), true);
        try w.proj(layer, "mlp.gate_proj", c.intermediate, c.hidden);
        try w.proj(layer, "mlp.up_proj", c.intermediate, c.hidden);
        try w.proj(layer, "mlp.down_proj", c.hidden, c.intermediate);
        switch (c.kinds[i]) {
            .linear_attention => {
                try w.proj(layer, "linear_attn.in_proj_qkv", c.convWidth(), c.hidden);
                try w.proj(layer, "linear_attn.in_proj_z", c.valueWidth(), c.hidden);
                try w.proj(layer, "linear_attn.in_proj_b", c.value_heads, c.hidden);
                try w.proj(layer, "linear_attn.in_proj_a", c.value_heads, c.hidden);
                try w.proj(layer, "linear_attn.out_proj", c.hidden, c.valueWidth());
                try w.raw(layer, "linear_attn.conv1d.weight", .conv, &.{ c.convWidth(), 1, c.conv });
                try w.raw(layer, "linear_attn.A_log", .scalar_vector, &.{c.value_heads});
                try w.raw(layer, "linear_attn.dt_bias", .scalar_vector, &.{c.value_heads});
                try w.raw(layer, "linear_attn.norm.weight", .norm, &.{c.value_dim});
            },
            .indexed_attention => {
                try w.proj(layer, "self_attn.q_proj", 2 * c.heads * c.head_dim, c.hidden);
                try w.proj(layer, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden);
                try w.proj(layer, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden);
                try w.proj(layer, "self_attn.o_proj", c.hidden, c.heads * c.head_dim);
                try w.proj(layer, "self_attn.indexer.index_qk_proj", (c.index_heads + 1) * c.index_dim, c.hidden);
                for ([_][]const u8{ "self_attn.q_norm.weight", "self_attn.k_norm.weight" }) |name|
                    try w.raw(layer, name, .norm, &.{c.head_dim});
                for ([_][]const u8{ "self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight" }) |name|
                    try w.raw(layer, name, .norm, &.{c.index_dim});
            },
        }
        if (c.ple[i]) {
            try w.ple(layer);
            w.ple_ordinal += 1;
        }
    }
}
fn Walker(comptime Sink: type) type {
    return struct {
        const W = @This();
        c: *const cfg.Config,
        sink: Sink,
        p: *cfg.Problem,
        ple_ordinal: usize = 0,
        fn matrix(w: *W, name: []const u8, n: usize, k: usize, role: Role) !void {
            const recipe = try w.c.quant.of(name, w.p);
            try w.sink.take(name, Want{ .role = role, .shape = .{ .rank = 2, .dims = .{ n, k, 1 } }, .recipe = recipe });
        }
        fn proj(w: *W, stem: []const u8, tail: []const u8, n: usize, k: usize) !void {
            var b: [256]u8 = undefined;
            try w.matrix(try std.fmt.bufPrint(&b, "{s}.{s}", .{ stem, tail }), n, k, .matrix);
        }
        fn raw(w: *W, stem: []const u8, tail: []const u8, role: Role, dims: []const usize) !void {
            var b: [256]u8 = undefined;
            var shape: Shape = .{ .rank = @intCast(dims.len) };
            @memcpy(shape.dims[0..dims.len], dims);
            try w.sink.take(try std.fmt.bufPrint(&b, "{s}.{s}", .{ stem, tail }), Want{ .role = role, .shape = shape });
        }
        fn hc(w: *W, stem: []const u8, inject: bool) !void {
            const c = w.c;
            try w.raw(stem, "hc_norm.weight", .norm, &.{c.wide()});
            try w.proj(stem, "input_mix_weight_down", c.lowrank, c.wide());
            try w.proj(stem, "input_mix_weight_up", c.wide(), c.lowrank);
            if (inject) try w.proj(stem, "block_inject_weight", c.streams, c.wide());
        }
        fn ple(w: *W, stem: []const u8) !void {
            const c = w.c;
            var b: [256]u8 = undefined;
            const pre = try std.fmt.bufPrint(&b, "{s}.ple", .{stem});
            try w.proj(pre, "key_proj", c.wide(), c.ple_dim);
            try w.proj(pre, "value_proj", c.hidden, c.ple_dim);
            for ([_][]const u8{ "norm_key.weight", "norm_query.weight", "norm_conv.weight" }) |name|
                try w.raw(pre, name, .norm, &.{c.wide()});
            try w.raw(pre, "conv1d.weight", .conv, &.{ c.wide(), 1, c.ple_conv });
            const heads = (c.ngram - 1) * c.ngram_heads;
            try w.raw(pre, "ple_embedding.layer_multipliers", .integer_vector, &.{c.ngram});
            try w.raw(pre, "ple_embedding.ngram_heads_vocab_sizes", .integer_vector, &.{heads});
            try w.raw(pre, "ple_embedding.ngram_heads_offsets", .integer_vector, &.{heads});
            const rows = try ngram.tableRows(c.ngram_vocab, heads, w.ple_ordinal, c.ngram_divisor);
            try w.matrix(try std.fmt.bufPrint(&b, "{s}.ple.ple_embedding.ngram_embedding", .{stem}), rows, c.ple_dim / heads, .embedding);
        }
    };
}

fn same(e: st.Entry, dtype: st.DType, dims: []const usize) bool {
    if (e.dtype != dtype or e.rank != dims.len) return false;
    for (dims, 0..) |d, i| if (e.shape[i] != d) return false;
    var bytes: usize = dtype.size();
    for (dims) |d| {
        bytes = std.math.mul(usize, bytes, d) catch return false;
    }
    return e.end >= e.begin and e.end - e.begin == bytes;
}
/// Validates stored headers only; does not dequantize, reinterpret, transpose or change the checkpoint.
pub fn headers(name: []const u8, w: Want, map: st.Header, p: *cfg.Problem) !void {
    var b: [320]u8 = undefined;
    if (w.role != .matrix and w.role != .embedding) {
        const e = map.get(name) orelse return p.reject(name);
        if (w.role == .integer_vector) {
            if (!same(e, .i64, w.shape.dims[0..w.shape.rank])) return p.reject(name);
        } else if (!same(e, .bf16, w.shape.dims[0..w.shape.rank]) and
            !same(e, .f32, w.shape.dims[0..w.shape.rank])) return p.reject(name);
        return;
    }
    const n = w.shape.dims[0];
    const k = w.shape.dims[1];
    const weight = try std.fmt.bufPrint(&b, "{s}.weight", .{name});
    const e = map.get(weight) orelse return p.reject(weight);
    switch (w.recipe.format) {
        .bf16 => if (!same(e, .bf16, &.{ n, k })) return p.reject(weight),
        .mlx4, .mlx8 => {
            const bits: usize = if (w.recipe.format == .mlx4) 4 else 8;
            const g = w.recipe.group;
            if (g == 0 or k % g != 0 or k % (32 / bits) != 0 or !same(e, .u32, &.{ n, k / (32 / bits) })) return p.reject(weight);
            const scale_name = try std.fmt.bufPrint(&b, "{s}.scales", .{name});
            const s = map.get(scale_name) orelse return p.reject(scale_name);
            const bias = map.get(try std.fmt.bufPrint(&b, "{s}.biases", .{name})) orelse return p.reject(name);
            if ((s.dtype != .bf16 and s.dtype != .f16 and s.dtype != .f32) or !same(s, s.dtype, &.{ n, k / g }) or !same(bias, s.dtype, &.{ n, k / g })) return p.reject(name);
        },
        .fp8 => {
            if (!same(e, .f8_e4m3, &.{ n, k })) return p.reject(weight);
            try scalar(map, try std.fmt.bufPrint(&b, "{s}.weight_scale", .{name}), p);
        },
        .nvfp4 => {
            if (k % 16 != 0 or !same(e, .u8, &.{ n, k / 2 })) return p.reject(weight);
            const scale_name = try std.fmt.bufPrint(&b, "{s}.weight_scale", .{name});
            const scale = map.get(scale_name) orelse return p.reject(scale_name);
            if (!same(scale, .f8_e4m3, &.{ n, k / 16 })) return p.reject(scale_name);
            try scalar(map, try std.fmt.bufPrint(&b, "{s}.weight_scale_2", .{name}), p);
        },
    }
}
fn scalar(map: st.Header, name: []const u8, p: *cfg.Problem) !void {
    const e = map.get(name) orelse return p.reject(name);
    if (!same(e, .f32, &.{}) and !same(e, .f32, &.{1})) return p.reject(name);
}

/// Every header must have a role. A partial load cannot silently omit another projection or head.
pub fn admit(a: std.mem.Allocator, c: *const cfg.Config, ns: Namespace, map: st.Header, p: *cfg.Problem) !Stats {
    var sink = HeaderSink{ .a = a, .map = map, .p = p };
    defer sink.used.deinit(a);
    defer sink.projections.deinit(a);
    try walk(c, ns, &sink, p);
    try c.quant.validateUsed(sink.projections, p);
    for (map.keys()) |name| if (!sink.used.contains(name)) return p.reject(name);
    return sink.stats;
}
const HeaderSink = struct {
    a: std.mem.Allocator,
    map: st.Header,
    p: *cfg.Problem,
    used: std.StringHashMapUnmanaged(void) = .empty,
    projections: std.StringHashMapUnmanaged(void) = .empty,
    stats: Stats = .{},
    fn mark(s: *@This(), name: []const u8) !void {
        // Keys live in the validated header, never in the walker's reused name buffer.
        const i = s.map.getIndex(name) orelse return s.p.reject(name);
        const key = s.map.keys()[i];
        if ((try s.used.getOrPut(s.a, key)).found_existing) return s.p.reject(name);
        s.stats.tensors += 1;
    }
    pub fn take(s: *@This(), name: []const u8, w: Want) !void {
        try headers(name, w, s.map, s.p);
        if (w.role != .matrix and w.role != .embedding) return s.mark(name);
        s.stats.matrices += 1;
        var keybuf: [320]u8 = undefined;
        const index = s.map.getIndex(try std.fmt.bufPrint(&keybuf, "{s}.weight", .{name})).?;
        const stored = s.map.keys()[index];
        try s.projections.put(s.a, stored[0 .. stored.len - 7], {});
        var b: [320]u8 = undefined;
        const tails: []const []const u8 = switch (w.recipe.format) {
            .bf16 => &.{"weight"},
            .mlx4, .mlx8 => &.{ "weight", "scales", "biases" },
            .fp8 => &.{ "weight", "weight_scale" },
            .nvfp4 => &.{ "weight", "weight_scale", "weight_scale_2" },
        };
        for (tails) |tail| try s.mark(try std.fmt.bufPrint(&b, "{s}.{s}", .{ name, tail }));
        // ModelOpt calibration can accompany A16 weights; it never enables activation quantization.
        if (w.recipe.format == .fp8 or w.recipe.format == .nvfp4) for ([_][]const u8{ "input_scale", "input_scale_2" }) |tail| {
            const calibration = try std.fmt.bufPrint(&b, "{s}.{s}", .{ name, tail });
            if (s.map.contains(calibration)) {
                try scalar(s.map, calibration, s.p);
                try s.mark(calibration);
            }
        };
    }
};
