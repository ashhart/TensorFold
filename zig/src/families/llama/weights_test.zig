//! Llama's loader on a checkpoint's names, dtypes and shapes alone: tied heads, MLX's triples, refusals by name.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const t = std.testing;

/// Tensors by name with a dtype, a shape and their bytes, no storage: what the loader checks before a kernel reads.
const Fake = struct {
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMapUnmanaged(ck.Tensor) = .empty,
    end: usize = 0, // the next tensor's offset, as if every one followed the last in one shard

    fn init() Fake {
        return .{ .arena = std.heap.ArenaAllocator.init(t.allocator) };
    }

    fn deinit(f: *Fake) void {
        f.arena.deinit();
    }

    fn put(f: *Fake, n: []const u8, dtype: ck.DType, dims: []const usize) !void {
        const a = f.arena.allocator();
        var x: ck.Tensor = .{ .buffer = undefined, .offset = f.end, .bytes = dtype.size(), .dtype = dtype, .rank = dims.len };
        for (dims, 0..) |d, i| {
            x.shape[i] = d;
            x.bytes *= d;
        }
        f.end = std.mem.alignForward(usize, f.end + x.bytes, 256);
        try f.map.put(a, try a.dupe(u8, n), x);
    }

    pub fn get(f: *const Fake, n: []const u8) !ck.Tensor {
        return f.map.get(n) orelse error.MissingTensor;
    }

    pub fn has(f: *const Fake, n: []const u8) bool {
        return f.map.contains(n);
    }

    /// `base` [n, k] as the config's weights store it: bf16 rows, or MLX's words and `tables`-typed scales and biases.
    fn matrix(f: *Fake, c: *const cfg.Config, base: []const u8, n: usize, k: usize, tables: ck.DType) !void {
        var buf: [128]u8 = undefined;
        const w = c.weights.mlx() orelse return f.put(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), .bf16, &.{ n, k });
        try f.put(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), .u32, &.{ n, k * w.bits / 32 });
        try f.put(try std.fmt.bufPrint(&buf, "{s}.scales", .{base}), tables, &.{ n, k / w.group });
        try f.put(try std.fmt.bufPrint(&buf, "{s}.biases", .{base}), tables, &.{ n, k / w.group });
    }

    /// Every tensor `c` names; an lm_head when untied or `head` says so.
    fn model(f: *Fake, c: *const cfg.Config, head: bool, tables: ck.DType) !void {
        const h = c.hidden;
        try f.matrix(c, "model.embed_tokens", c.vocab, h, tables);
        if (head) try f.matrix(c, "lm_head", c.vocab, h, tables);
        try f.put("model.norm.weight", .bf16, &.{h});
        for (0..c.layers) |i| {
            var pb: [64]u8 = undefined;
            const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
            var nb: [128]u8 = undefined;
            for ([_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |n| try f.put(try std.fmt.bufPrint(&nb, "{s}{s}.weight", .{ p, n }), .bf16, &.{h});
            const shapes = [_]struct { []const u8, usize, usize }{
                .{ "self_attn.q_proj", c.queryWidth(), h }, .{ "self_attn.k_proj", c.kvWidth(), h }, .{ "self_attn.v_proj", c.kvWidth(), h },
                .{ "self_attn.o_proj", h, c.queryWidth() }, .{ "mlp.gate_proj", c.intermediate, h }, .{ "mlp.up_proj", c.intermediate, h },
                .{ "mlp.down_proj", h, c.intermediate },
            };
            for (shapes) |s| try f.matrix(c, try std.fmt.bufPrint(&nb, "{s}{s}", .{ p, s[0] }), s[1], s[2], tables);
        }
    }
};

/// A small Llama: 2 layers, 2 query heads over 1 key head of 128 (a 256-wide query row), hidden 512.
const small = "{\"model_type\": \"llama\", \"hidden_size\": 512, \"intermediate_size\": 768, \"num_hidden_layers\": 2," ++
    " \"num_attention_heads\": 2, \"num_key_value_heads\": 1, \"head_dim\": 128, \"vocab_size\": 1024, \"rms_norm_eps\": 1e-5," ++
    " \"max_position_embeddings\": 4096, \"eos_token_id\": 2";

fn config(extra: []const u8) !cfg.Config {
    const text = try std.mem.concat(t.allocator, u8, &.{ small, extra, "}" });
    defer t.allocator.free(text);
    return cfg.parse(t.allocator, text);
}

fn same(a: weights.Dense, b: weights.Dense) bool {
    return a.off == b.off and a.n == b.n and a.k == b.k and (a.affine == null) == (b.affine == null);
}

test "llama weights: a tied bf16 head is the embedding table, an untied one lm_head, a tied one beside lm_head refused" {
    const c = try config(", \"tie_word_embeddings\": true");
    var f = Fake.init();
    defer f.deinit();
    try f.model(&c, false, .bf16);
    var w = try weights.load(t.allocator, &f, &c);
    defer w.deinit(t.allocator);
    try t.expect(same(w.head, w.embed));
    try t.expectEqual(@as(usize, 1024), w.head.n);
    try t.expectEqual(@as(usize, 2), w.layers.len);
    try t.expectEqual(@as(usize, 256), w.layers[0].attn.o.k);
    // the untied config reads the stored head, and without one the load stops at the missing tensor
    const u = try config("");
    try t.expectError(error.MissingTensor, weights.load(t.allocator, &f, &u));
    try f.matrix(&u, "lm_head", u.vocab, u.hidden, .bf16);
    var v = try weights.load(t.allocator, &f, &u);
    defer v.deinit(t.allocator);
    try t.expectEqual(@as(usize, 1024), v.head.n);
    try t.expectError(error.LlamaHeadNotTied, weights.load(t.allocator, &f, &c));
}

test "llama weights: an MLX checkpoint's projections, embedding and tied head are its triples at the config's width" {
    const widths = [_]struct { []const u8, u8, usize }{ .{ "4, \"group_size\": 64", 4, 64 }, .{ "8, \"group_size\": 32", 8, 32 } };
    for (widths) |x| {
        var qb: [128]u8 = undefined;
        const c = try config(try std.fmt.bufPrint(&qb, ", \"tie_word_embeddings\": true, \"quantization\": {{\"bits\": {s}}}", .{x[0]}));
        try t.expectEqual(x[1], c.weights.mlx().?.bits);
        var f = Fake.init();
        defer f.deinit();
        try f.model(&c, false, .bf16);
        var w = try weights.load(t.allocator, &f, &c);
        defer w.deinit(t.allocator);
        for ([_]weights.Dense{ w.embed, w.head, w.layers[1].attn.q, w.layers[1].attn.o, w.layers[0].mlp.down }) |d| {
            const a = d.affine.?;
            try t.expectEqual(x[1], a.bits);
            try t.expectEqual(x[2], a.group);
            try t.expectEqual(d.n, a.n);
            try t.expectEqual(d.k, a.k);
        }
        try t.expect(same(w.head, w.embed));
        try t.expectEqual(@as(usize, 768), w.layers[0].mlp.down.k);
    }
}

test "llama weights: an MLX conversion with float16 tables is refused by name, its untied head too" {
    for ([_]bool{ true, false }) |tied| {
        const c = try config(if (tied) ", \"tie_word_embeddings\": true, \"quantization\": {\"bits\": 4, \"group_size\": 64}" else ", \"quantization\": {\"bits\": 4, \"group_size\": 64}");
        var f = Fake.init();
        defer f.deinit();
        try f.model(&c, !tied, .f16);
        try t.expectError(error.UnsupportedLlamaTables, weights.load(t.allocator, &f, &c));
    }
    // bf16 tables but one float16 projection: still refused, at that projection
    const c = try config(", \"quantization\": {\"bits\": 4, \"group_size\": 64}");
    var f = Fake.init();
    defer f.deinit();
    try f.model(&c, true, .bf16);
    try f.put("model.layers.1.mlp.up_proj.biases", .f16, &.{ 768, 512 / 64 });
    try t.expectError(error.UnsupportedLlamaTables, weights.load(t.allocator, &f, &c));
}
