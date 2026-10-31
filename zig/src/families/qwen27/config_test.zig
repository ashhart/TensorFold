const std = @import("std");
const cfg = @import("config.zig");
const parse = cfg.parse;
const Config = cfg.Config;
const Kind = cfg.Kind;
const Span = cfg.Span;
const Mode = cfg.Mode;
const split = cfg.split;
const object = cfg.object;

const fixture =
    \\{"model_type":"qwen3_5","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":97,
    \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,
    \\"linear_value_head_dim":16,"linear_conv_kernel_dim":4,"eos_token_id":[2,11]}
;

fn edit(a: std.mem.Allocator, key: []const u8, value: std.json.Value) ![]u8 {
    var p = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer p.deinit();
    try p.value.object.put(p.arena.allocator(), key, value);
    return std.json.Stringify.valueAlloc(a, p.value, .{});
}

test "flat config preserves dimensions and interval schedule" {
    const c = try parse(std.testing.allocator, fixture, null);
    try std.testing.expectEqual(@as(usize, 128), c.qDim());
    try std.testing.expectEqual(@as(usize, 32), c.kvDim());
    try std.testing.expectEqual(@as(usize, 128), c.gdnQkvDim());
    try std.testing.expectEqual(Kind.linear, c.kind(0));
    try std.testing.expectEqual(Kind.attention, c.kind(3));
}

test "nested config merges generation EOS in stable order" {
    const a = std.testing.allocator;
    const text = try std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_5\",\"eos_token_id\":[11,2],\"text_config\":{s}}}", .{fixture});
    defer a.free(text);
    const c = try parse(a, text, "{\"eos_token_id\":[2,3,11]}");
    try std.testing.expectEqualSlices(u32, &.{ 11, 2, 3 }, c.eos[0..c.eos_count]);
    try std.testing.expect(c.isEos(3) and !c.isEos(4));
}

test "explicit layer types preserve their order" {
    const a = std.testing.allocator;
    var p = try std.json.parseFromSlice(std.json.Value, a, "[\"full_attention\",\"linear_attention\",\"full_attention\",\"linear_attention\"]", .{});
    defer p.deinit();
    const json = try edit(a, "layer_types", p.value);
    defer a.free(json);
    const c = try parse(a, json, null);
    try std.testing.expectEqualSlices(Kind, &.{ .attention, .linear, .attention, .linear }, c.kinds[0..4]);
}

test "rope parameters preserve precision and section metadata" {
    const a = std.testing.allocator;
    const p = try std.json.parseFromSlice(std.json.Value, a, "{\"rope_theta\":1234567.25,\"partial_rotary_factor\":0.5,\"mrope_section\":[2,1,1]}", .{});
    defer p.deinit();
    const json = try edit(a, "rope_parameters", p.value);
    defer a.free(json);
    const c = try parse(a, json, null);
    try std.testing.expectEqual(@as(f64, 1234567.25), c.rope_theta);
    try std.testing.expectEqual(@as(usize, 8), c.rope_dims);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 1 }, &c.mrope_section);
}

test "a routed config reads its experts, the shared expert's width as the layer's and the routing flag" {
    const a = std.testing.allocator;
    const json =
        \\{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":3072,"intermediate_size":1024,
        \\"moe_intermediate_size":1024,"shared_expert_intermediate_size":1024,"num_experts":256,"num_experts_per_tok":8,
        \\"num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":2,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":64,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}}
    ;
    const c = try parse(a, json, null);
    try std.testing.expect(c.moe());
    try std.testing.expectEqual(@as(usize, 256), c.experts);
    try std.testing.expectEqual(@as(usize, 8), c.top_k);
    try std.testing.expectEqual(@as(usize, 1024), c.moe_intermediate);
    try std.testing.expectEqual(@as(usize, 1024), c.intermediate);
    try std.testing.expect(c.renorm); // absent: renormalized
    try std.testing.expectEqual(Kind.attention, c.kind(3));
    try std.testing.expectEqual(Kind.linear, c.kind(2));
    // a routed model type without experts, and a dense one with them, are refused
    const dense_typed = try std.mem.replaceOwned(u8, a, json, "\"num_experts\":256", "\"num_experts\":0");
    defer a.free(dense_typed);
    try std.testing.expectError(error.NotQwenDense, parse(a, dense_typed, null));
}

test "a routed config refuses a rank split" {
    var c = model27();
    c.experts = 256;
    c.top_k = 8;
    c.moe_intermediate = c.intermediate;
    try std.testing.expectError(error.BadShard, split(&c, 0, 2, .full));
    try std.testing.expectError(error.BadShard, split(&c, 1, 2, .mlp));
}

test "dense config refuses MoE and tied embedding heads" {
    const a = std.testing.allocator;
    const moe = try edit(a, "num_experts", .{ .integer = 8 });
    defer a.free(moe);
    try std.testing.expectError(error.NotQwenDense, parse(a, moe, null));
    const tied = try edit(a, "tie_word_embeddings", .{ .bool = true });
    defer a.free(tied);
    try std.testing.expectError(error.TiedEmbeddingHead, parse(a, tied, null));
}

test "invalid scalar shapes and types fail without a cast trap" {
    const a = std.testing.allocator;
    for ([_]std.json.Value{ .{ .integer = -1 }, .{ .integer = 0 }, .{ .bool = true }, .{ .integer = 4294967296 } }) |v| {
        const json = try edit(a, "hidden_size", v);
        defer a.free(json);
        try std.testing.expectError(error.BadConfig, parse(a, json, null));
    }
    const json = try edit(a, "num_hidden_layers", .{ .integer = 129 });
    defer a.free(json);
    try std.testing.expectError(error.BadConfig, parse(a, json, null));
}

test "missing and out of vocabulary EOS fail" {
    const a = std.testing.allocator;
    const missing = try edit(a, "eos_token_id", .null);
    defer a.free(missing);
    try std.testing.expectError(error.MissingEos, parse(a, missing, null));
    const c = try parse(a, missing, "{\"eos_token_id\":7}");
    try std.testing.expect(c.isEos(7));
    const wrong = try edit(a, "eos_token_id", .{ .integer = 97 });
    defer a.free(wrong);
    try std.testing.expectError(error.BadEos, parse(a, wrong, null));
}

test "malformed layer and rope metadata fail" {
    const a = std.testing.allocator;
    const layers = try edit(a, "layer_types", .{ .bool = true });
    defer a.free(layers);
    try std.testing.expectError(error.BadConfig, parse(a, layers, null));
    const rope = try edit(a, "partial_rotary_factor", .{ .float = 2 });
    defer a.free(rope);
    try std.testing.expectError(error.BadConfig, parse(a, rope, null));
    try std.testing.expectError(error.BadConfig, parse(a, "[]", null));
}

test "head grouping requires complete key and value groups" {
    const a = std.testing.allocator;
    const attention = try edit(a, "num_key_value_heads", .{ .integer = 3 });
    defer a.free(attention);
    try std.testing.expectError(error.BadConfig, parse(a, attention, null));
    const gdn = try edit(a, "linear_num_value_heads", .{ .integer = 3 });
    defer a.free(gdn);
    try std.testing.expectError(error.BadConfig, parse(a, gdn, null));
}

test "text execution refuses unsupported gates and activation contracts" {
    const a = std.testing.allocator;
    const gate = try edit(a, "attn_output_gate", .{ .bool = false });
    defer a.free(gate);
    try std.testing.expectError(error.UnsupportedAttentionGate, parse(a, gate, null));
    const act = try edit(a, "output_gate_type", .{ .string = "sigmoid" });
    defer a.free(act);
    try std.testing.expectError(error.UnsupportedActivation, parse(a, act, null));
}

/// The real 27B's shapes, so the shard arithmetic is pinned against the checkpoint rather than a toy config.
fn model27() Config {
    return .{
        .hidden = 5120,
        .intermediate = 17408,
        .layers = 64,
        .vocab = 248320,
        .heads = 24,
        .kv_heads = 4,
        .head_dim = 256,
        .k_heads = 16,
        .v_heads = 48,
        .dk = 128,
        .dv = 128,
        .conv_kernel = 4,
        .eps = 1e-6,
        .rope_dims = 64,
        .rope_theta = 1e7,
    };
}

fn expectSpan(span: Span, from: usize, len: usize) !void {
    try std.testing.expectEqual(from, span.from);
    try std.testing.expectEqual(len, span.len);
}

test "a full shard takes whole heads and boundary-aligned row blocks" {
    var d = model27();
    const s = try split(&d, 1, 2, .full);
    try std.testing.expectEqual(@as(usize, 64), s.local.layers);
    try std.testing.expectEqual(@as(usize, 12), s.local.heads);
    try std.testing.expectEqual(@as(usize, 2), s.local.kv_heads);
    try std.testing.expectEqual(@as(usize, 8), s.local.k_heads);
    try std.testing.expectEqual(@as(usize, 24), s.local.v_heads);
    try std.testing.expectEqual(@as(usize, 8704), s.local.intermediate);
    try std.testing.expectEqual(@as(usize, 124160), s.local.vocab);
    try std.testing.expect(s.splitsAttention());
    // q and its gate interleave every head: rank 1 starts at head 12, a 2*head_dim (512) boundary
    try expectSpan(s.qRows(), 6144, 6144);
    try std.testing.expectEqual(@as(usize, 0), s.qRows().from % (2 * s.global.head_dim));
    try expectSpan(s.kvRows(), 512, 512);
    // GDN q | k | v is one tensor of three blocks: q and k split 8 of 16 heads, v splits 24 of 48
    const blocks = s.qkvSpans();
    try expectSpan(blocks[0], 1024, 1024);
    try expectSpan(blocks[1], 3072, 1024);
    try expectSpan(blocks[2], 7168, 3072);
    try std.testing.expectEqual(@as(usize, 5120), blocks[0].len + blocks[1].len + blocks[2].len);
    const conv = s.convSpans();
    for (blocks, conv) |want, got| {
        try expectSpan(got, want.from, want.len);
    }
    try expectSpan(s.zRows(), 3072, 3072);
    try expectSpan(s.valueRows(), 24, 24);
    try expectSpan(s.abRows(), 0, 48);
    try expectSpan(s.intermediateRows(), 8704, 8704);
    try expectSpan(s.vocabRows(), 124160, 124160);
    // o_proj, out_proj and down_proj own rows of the hidden width, each over the whole input K
    try expectSpan(s.outputRows(), 2560, 2560);
}

test "the embedding bound is the whole table on a pair rank, not the shard's width" {
    var d = model27();
    const s = try split(&d, 1, 2, .full);
    // the table keeps 248320 rows on both ranks while the head's shard is 124160 wide
    try std.testing.expectEqual(@as(usize, 124160), s.local.vocab);
    try std.testing.expectEqual(@as(u32, 248320), try cfg.embedBound(&s.local, 248320));
    try std.testing.expectEqual(@as(u32, 248320), try cfg.embedBound(&s.global, 248320));
    // a bound below the shard's own width is a table that lost rows, so it stays an error
    try std.testing.expectError(error.BadEmbedding, cfg.embedBound(&s.local, 124159));
}

test "rank 0's blocks start at the tensor's front" {
    var d = model27();
    const s = try split(&d, 0, 2, .full);
    try expectSpan(s.qRows(), 0, 6144);
    try expectSpan(s.kvRows(), 0, 512);
    const blocks = s.qkvSpans();
    try expectSpan(blocks[0], 0, 1024);
    try expectSpan(blocks[1], 2048, 1024);
    try expectSpan(blocks[2], 4096, 3072);
    try expectSpan(s.zRows(), 0, 3072);
    try expectSpan(s.valueRows(), 0, 24);
    try expectSpan(s.outputRows(), 0, 2560);
}

test "the GDN qkv blocks tile the global tensor exactly across ranks" {
    var seen: [10240]bool = @splat(false);
    for (0..2) |r| {
        var d = model27();
        const s = try split(&d, @intCast(r), 2, .full);
        for (s.qkvSpans()) |span| {
            try std.testing.expect(span.end() <= seen.len);
            for (span.from..span.end()) |i| {
                try std.testing.expect(!seen[i]);
                seen[i] = true;
            }
        }
    }
    for (seen) |hit| try std.testing.expect(hit);
}

test "an mlp shard replicates attention and splits only the mlp and the head" {
    var d = model27();
    const s = try split(&d, 1, 2, .mlp);
    try std.testing.expect(!s.splitsAttention());
    try std.testing.expectEqual(@as(usize, 24), s.local.heads);
    try std.testing.expectEqual(@as(usize, 4), s.local.kv_heads);
    try std.testing.expectEqual(@as(usize, 16), s.local.k_heads);
    try std.testing.expectEqual(@as(usize, 48), s.local.v_heads);
    try std.testing.expectEqual(@as(usize, 8704), s.local.intermediate);
    try std.testing.expectEqual(@as(usize, 124160), s.local.vocab);
    try expectSpan(s.qRows(), 0, 12288);
    try expectSpan(s.kvRows(), 0, 1024);
    const blocks = s.qkvSpans();
    try expectSpan(blocks[0], 0, 2048);
    try expectSpan(blocks[1], 2048, 2048);
    try expectSpan(blocks[2], 4096, 6144);
    try expectSpan(s.zRows(), 0, 6144);
    try expectSpan(s.abRows(), 0, 48);
    // attention is replicated, so every value head is this rank's: A_log and dt_bias stay whole too
    try expectSpan(s.valueRows(), 0, 48);
    try expectSpan(s.intermediateRows(), 8704, 8704);
    // an mlp shard keeps the mixers and down_proj replicated, so their output rows are the whole width
    try expectSpan(s.outputRows(), 0, 5120);
}

test "the fused zba row keeps the tiled projection and each rank's a and b" {
    for ([_]Mode{ .full, .mlp }) |mode| {
        var d = model27();
        const s = try split(&d, 1, 2, mode);
        const c = s.local;
        try std.testing.expectEqual(@as(usize, 0), c.zbaDim() % 32);
        const whole_attention = mode == .mlp;
        const ranks: usize = c.ranks;
        const b_width = c.v_heads * (if (whole_attention) 1 else ranks);
        const head0 = if (whole_attention) 0 else c.v_heads;
        try std.testing.expectEqual(c.gdnValueDim() + 2 * b_width, c.zbaDim());
        try std.testing.expectEqual(c.gdnValueDim() + head0, c.zbaB());
        try std.testing.expectEqual(c.gdnValueDim() + b_width + head0, c.zbaA());
        try std.testing.expectEqual(c.zbaDim(), c.zbaA() + c.v_heads);
        var e = model27();
        const z = try split(&e, 0, 2, mode);
        try std.testing.expectEqual(z.local.gdnValueDim(), z.local.zbaB());
        try std.testing.expectEqual(z.local.gdnValueDim() + 48, z.local.zbaA());
    }
    // one Mac is today's layout: z | b | a, each value head once
    const u = model27();
    try std.testing.expectEqual(@as(usize, 6144), u.zbaB());
    try std.testing.expectEqual(@as(usize, 6192), u.zbaA());
    try std.testing.expectEqual(@as(usize, 6240), u.zbaDim());
}
