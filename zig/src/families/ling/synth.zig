//! A tiny Ling-3.1 config: the real config.json's schema with fewer heads, experts and dims, in any loader format.
const std = @import("std");
const config = @import("config.zig");

pub const real_config = @embedFile("fixtures/config.json");
/// Ling-3.0-tiny's config.json, the base of `--tiny` checkpoints: its real widths with a 1,024-token vocabulary.
pub const tiny_config = @embedFile("fixtures/tiny_config.json");

/// Which model's schema a synthetic checkpoint follows: Ling-3.1-flash's, shrunk, or Ling-3.0-tiny's own widths.
pub const Base = enum { flash, tiny };

/// The shrunk widths; KDA's head dim 128, MLA's 512 latent, 64 rope and 128-wide heads stay as the kernels take them.
pub const shape = .{
    .hidden = 512,
    .heads = 4,
    .dense_inter = 256,
    .moe_inter = 128,
    .experts = 16,
    .topk = 4,
    .n_group = 4,
    .topk_group = 2,
    .q_lora = 512, // a width glue.rmsnorm takes (512, 1536, 4096)
    .vocab = 1024,
    .group = 2, // KDA, MLA, KDA, MLA, ...
};

/// The tiny config for `layers` backbone layers (at least 2: layer 0 KDA + dense, 1 MLA + MoE) and one MTP layer.
pub fn configJson(gpa: std.mem.Allocator, format: config.Quant, layers: u32) ![]u8 {
    return configJsonOf(gpa, .flash, format, layers);
}

/// `configJson` for either base; tiny's is its real widths (hidden 1536, 128 experts, q_lora_rank 256) and has no MTP.
pub fn configJsonOf(gpa: std.mem.Allocator, base: Base, format: config.Quant, layers: u32) ![]u8 {
    if (layers < 2 or layers > config.max_layers - 1) return error.BadLayerCount;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root = try std.json.parseFromSliceLeaky(std.json.Value, a, if (base == .tiny) tiny_config else real_config, .{});
    const o = &root.object;
    if (base == .tiny) {
        const tiny_ints = .{ .{ "vocab_size", shape.vocab }, .{ "eos_token_id", shape.vocab - 1 }, .{ "pad_token_id", shape.vocab - 2 }, .{ "num_hidden_layers", layers } };
        inline for (tiny_ints) |kv| try o.put(a, kv[0], .{ .integer = kv[1] });
        const group: u32 = @intCast(o.get("layer_group_size").?.integer);
        switch (format) {
            .bf16 => {},
            .fp8_block128 => try o.put(a, "quantization_config", try quant(a, fp8_block, layers, group, false)),
            .nvfp4 => try o.put(a, "quantization_config", try quant(a, modelopt_block, layers, group, false)),
        }
        return std.json.Stringify.valueAlloc(gpa, root, .{ .whitespace = .indent_2 });
    }
    const ints = .{
        .{ "hidden_size", shape.hidden },            .{ "num_attention_heads", shape.heads },       .{ "num_key_value_heads", shape.heads },
        .{ "intermediate_size", shape.dense_inter }, .{ "moe_intermediate_size", shape.moe_inter }, .{ "moe_shared_expert_intermediate_size", shape.moe_inter },
        .{ "num_experts", shape.experts },           .{ "num_experts_per_tok", shape.topk },        .{ "n_group", shape.n_group },
        .{ "topk_group", shape.topk_group },         .{ "q_lora_rank", shape.q_lora },              .{ "vocab_size", shape.vocab },
        .{ "eos_token_id", shape.vocab - 1 },        .{ "pad_token_id", shape.vocab - 2 },          .{ "layer_group_size", shape.group },
        .{ "first_k_dense_replace", 1 },             .{ "num_hidden_layers", layers },              .{ "num_nextn_predict_layers", 1 },
    };
    inline for (ints) |kv| try o.put(a, kv[0], .{ .integer = kv[1] });
    // clamps on the last layer only: routed and shared, as layers 65-71 carry them in the real file
    inline for (.{ "expert_swiglu_limit_list", "share_expert_swiglu_limit_list" }) |key| {
        var list = std.json.Array.init(a);
        for (0..layers) |i| try list.append(.{ .integer = if (i + 1 == layers) 10 else 0 });
        try o.put(a, key, .{ .array = list });
    }
    switch (format) {
        .bf16 => _ = o.orderedRemove("quantization_config"),
        .fp8_block128 => try o.put(a, "quantization_config", try quant(a, fp8_block, layers, shape.group, true)),
        .nvfp4 => try o.put(a, "quantization_config", try quant(a, modelopt_block, layers, shape.group, true)),
    }
    return std.json.Stringify.valueAlloc(gpa, root, .{ .whitespace = .indent_2 });
}

const fp8_block =
    \\{"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128],
    \\ "modules_to_not_convert": ["lm_head", "model.word_embeddings", "b_proj", "mlp.gate"]}
;

/// ModelOpt NVFP4 as nvidia's checkpoints write it; `*` globs, and layer 0's dense MLP left in bf16 to mix formats.
const modelopt_block =
    \\{"quant_method": "modelopt", "quant_algo": "NVFP4", "config_groups": {"group_0": {"targets": ["Linear"],
    \\ "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
    \\ "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16}}},
    \\ "ignore": ["lm_head", "model.word_embeddings", "*.b_proj", "*.mlp.gate", "model.layers.0.mlp*"]}
;

/// The block with the real file's per-layer exceptions added: each MLA layer's g_proj and the MTP layer's eh_proj.
fn quant(a: std.mem.Allocator, block: []const u8, layers: u32, group: u32, mtp: bool) !std.json.Value {
    var q = try std.json.parseFromSliceLeaky(std.json.Value, a, block, .{});
    const key = if (q.object.contains("ignore")) "ignore" else "modules_to_not_convert";
    const list = &q.object.getPtr(key).?.array;
    var i: u32 = group - 1;
    while (i < layers) : (i += group) try list.append(.{ .string = try std.fmt.allocPrint(a, "model.layers.{d}.attention.g_proj", .{i}) });
    if (mtp) try list.append(.{ .string = try std.fmt.allocPrint(a, "model.layers.{d}.eh_proj", .{layers}) });
    return q;
}

test "the tiny configs parse in every format: KDA dense 0, MLA MoE 1, two groups, the MTP layer, the exceptions" {
    const gpa = std.testing.allocator;
    for ([_]config.Quant{ .bf16, .fp8_block128, .nvfp4 }) |f| {
        const json = try configJson(gpa, f, 4);
        defer gpa.free(json);
        var why: config.Why = .{};
        const c = config.parse(gpa, json, &why) catch |e| {
            std.debug.print("{t}: {s}\n", .{ f, why.text() });
            return e;
        };
        try std.testing.expectEqual(f, c.quant);
        try std.testing.expect(c.kind(0) == .kda and c.kind(1) == .mla and c.kind(2) == .kda and c.kind(4) == .mla);
        try std.testing.expect(!c.isMoe(0) and c.isMoe(1) and c.mtp == 1 and c.n_group == 4 and c.topk_group == 2);
        try std.testing.expectEqual(@as(f32, 10), c.expert_limit[3]);
        try std.testing.expectEqual(@as(u32, 512), c.kv_lora);
        try std.testing.expectEqual(config.Quant.bf16, c.format("model.layers.3.attention.g_proj"));
        try std.testing.expectEqual(config.Quant.bf16, c.format("model.layers.2.attention.b_proj"));
        try std.testing.expectEqual(config.Quant.bf16, c.format("model.layers.4.eh_proj"));
        try std.testing.expectEqual(f, c.format("model.layers.2.attention.g_proj"));
        try std.testing.expectEqual(f, c.format("model.layers.1.mlp.experts.3.down_proj"));
        try std.testing.expectEqual(if (f == .nvfp4) .bf16 else f, c.format("model.layers.0.mlp.up_proj"));
    }
}

test "tiny-shaped configs parse in every format: groups of 4, 128 experts, q_lora_rank 256, no MTP, g_proj on layer 3" {
    const gpa = std.testing.allocator;
    for ([_]config.Quant{ .bf16, .fp8_block128, .nvfp4 }) |f| {
        const json = try configJsonOf(gpa, .tiny, f, 4);
        defer gpa.free(json);
        var why: config.Why = .{};
        const c = config.parse(gpa, json, &why) catch |e| {
            std.debug.print("{t}: {s}\n", .{ f, why.text() });
            return e;
        };
        try std.testing.expectEqual(f, c.quant);
        try std.testing.expect(c.kind(2) == .kda and c.kind(3) == .mla and !c.isMoe(0) and c.isMoe(1) and c.mtp == 0);
        try std.testing.expect(c.experts == 128 and c.q_lora == 256 and c.hidden == 1536 and c.layers == 4);
        try std.testing.expectEqual(config.Quant.bf16, c.format("model.layers.3.attention.g_proj"));
        try std.testing.expectEqual(f, c.format("model.layers.2.attention.g_proj"));
    }
}
