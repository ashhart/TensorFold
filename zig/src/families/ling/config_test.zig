//! Ling-3.1-flash's own config.json (fixtures/config.json) parsed, and each refusal on a one-edit copy.
const std = @import("std");
const config = @import("config.zig");

const real = @embedFile("fixtures/config.json");
const t = std.testing;

fn load() !config.Config {
    var why: config.Why = .{};
    return config.parse(t.allocator, real, &why);
}

/// The fixture with `from` (which must occur exactly once) replaced by `to`.
fn edit(from: []const u8, to: []const u8) ![]u8 {
    if (std.mem.count(u8, real, from) != 1) return error.EditAnchor;
    return std.mem.replaceOwned(u8, t.allocator, real, from, to);
}

fn refused(from: []const u8, to: []const u8, want: anyerror, says: []const u8) !void {
    const json = try edit(from, to);
    defer t.allocator.free(json);
    var why: config.Why = .{};
    try t.expectError(want, config.parse(t.allocator, json, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), says) != null);
}

test "the checkpoint's config.json parses to Ling-3.1-flash's shape" {
    const c = try load();
    try t.expectEqual(@as(u32, 4096), c.hidden);
    try t.expectEqual(@as(u32, 72), c.layers);
    try t.expectEqual(@as(u32, 157184), c.vocab);
    try t.expectEqual(@as(f32, 1e-6), c.eps);
    try t.expectEqual(@as(f64, 6000000), c.rope_theta);
    try t.expectEqual(@as(u32, 64), c.rope_dim);
    try t.expectEqual(config.RopeStyle.gptj, c.rope_style);
    try t.expectEqual(@as(u32, 8), c.group);
    try t.expectEqual(@as(u32, 4), c.dense_layers);
    try t.expectEqual(@as(u32, 10240), c.dense_inter);
    try t.expectEqual(@as(u32, 64), c.kda_heads);
    try t.expectEqual(@as(u32, 128), c.kda_dim);
    try t.expectEqual(@as(u32, 4), c.conv);
    try t.expectEqual(@as(f32, -5.0), c.lower_bound);
    try t.expectEqual(@as(u32, 64), c.mla_heads);
    try t.expectEqual(@as(u32, 1536), c.q_lora);
    try t.expectEqual(@as(u32, 512), c.kv_lora);
    try t.expectEqual(@as(u32, 128), c.nope);
    try t.expectEqual(@as(u32, 128), c.v_dim);
    try t.expectEqual(@as(u32, 192), c.qkDim());
    try t.expectEqual(@as(f32, @floatCast(1.0 / @sqrt(192.0))), c.smScale());
    try t.expect(c.head_gate);
    try t.expectEqual(@as(u32, 512), c.experts);
    try t.expectEqual(@as(u32, 8), c.topk);
    try t.expectEqual(@as(u32, 8), c.n_group);
    try t.expectEqual(@as(u32, 4), c.topk_group);
    try t.expectEqual(@as(f32, 2.5), c.routed_scale);
    try t.expect(c.norm_topk and c.expert_bias);
    try t.expectEqual(@as(u32, 1280), c.moe_inter);
    try t.expectEqual(@as(u32, 1280), c.shared_inter);
    try t.expectEqual(@as(u32, 1), c.mtp);
    try t.expect(!c.tie);
    try t.expectEqual(@as(u32, 1), c.eos_n);
    try t.expectEqual(@as(u32, 156895), c.eos[0]);
    try t.expect(c.bos == null); // config.json names no bos_token_id; the tokenizer adds none either
    try t.expectEqual(config.Quant.fp8_block128, c.quant);
    try t.expectEqual(@as(u32, 14), c.plain_n);
}

test "the schedule: MLA exactly on 7, 15, ..., 71 (9 layers), KDA on the other 63, dense 0-3, the MTP layer MLA + MoE" {
    const c = try load();
    var mla: [9]u32 = undefined;
    var n: usize = 0;
    for (0..c.layers) |i| if (c.kind(@intCast(i)) == .mla) {
        mla[n] = @intCast(i);
        n += 1;
    };
    try t.expectEqual(@as(usize, 9), n);
    try t.expectEqualSlices(u32, &.{ 7, 15, 23, 31, 39, 47, 55, 63, 71 }, &mla);
    try t.expectEqual(@as(u32, 9), c.countKind(.mla));
    try t.expectEqual(@as(u32, 63), c.countKind(.kda));
    for (0..c.layers) |i| try t.expectEqual(i >= 4, c.isMoe(@intCast(i)));
    try t.expectEqual(config.Kind.mla, c.kind(72));
    try t.expect(c.isMoe(72));
}

test "a tail past the last full group is MLA, as the modeling's schedule says" {
    var c = try load();
    c.layers = 10;
    c.group = 4;
    for ([_]config.Kind{ .kda, .kda, .kda, .mla, .kda, .kda, .kda, .mla, .mla, .mla }, 0..) |k, i| try t.expectEqual(k, c.kind(@intCast(i)));
}

test "SwiGLU limits from the file: 10 on routed experts of layers 65-71 and shared ones of 64-71, else 0" {
    const c = try load();
    for (0..c.layers) |i| {
        try t.expectEqual(@as(f32, if (i >= 65) 10 else 0), c.expert_limit[i]);
        try t.expectEqual(@as(f32, if (i >= 64) 10 else 0), c.shared_limit[i]);
    }
}

test "modules_to_not_convert matches whole dotted tails: mlp.gate is the router, not gate_proj" {
    const c = try load();
    try t.expect(c.unconverted("model.layers.4.mlp.gate"));
    try t.expect(!c.unconverted("model.layers.4.mlp.gate_proj"));
    try t.expect(!c.unconverted("model.layers.4.mlp.shared_experts.gate_proj"));
    try t.expect(c.unconverted("model.layers.0.attention.b_proj"));
    try t.expect(!c.unconverted("model.layers.7.attention.q_b_proj"));
    try t.expect(c.unconverted("model.layers.7.attention.g_proj"));
    try t.expect(!c.unconverted("model.layers.72.attention.g_proj"));
    try t.expect(c.unconverted("model.layers.72.eh_proj"));
    try t.expect(c.unconverted("lm_head") and !c.scaled("lm_head"));
    try t.expect(c.scaled("model.layers.0.attention.q_proj"));
}

test "refusals: one edit each, with the reason" {
    try refused("\"BailingMoeV3ForCausalLM\"", "\"BailingMoeV2ForCausalLM\"", error.NotLing, "not a Ling-3.1");
    try refused("\"score_function\": \"sigmoid\"", "\"score_function\": \"softmax\"", error.ScoreFunction, "score_function");
    try refused("\"scoring_func\": \"sigmoid\"", "\"scoring_func\": \"softmax\"", error.ScoreFunction, "scoring_func");
    try refused("\"no_kda_lora\": true", "\"no_kda_lora\": false", error.KdaLora, "no_kda_lora");
    try refused("\"use_kda_lora\": false", "\"use_kda_lora\": true", error.KdaLora, "no_kda_lora");
    try refused("\"kda_safe_gate\": true", "\"kda_safe_gate\": false", error.KdaGate, "kda_safe_gate");
    try refused("\"head_wise\"", "\"element_wise\"", error.GateGranularity, "element_wise");
    try refused("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"yarn\", \"factor\": 4}", error.RopeScaling, "rope_scaling");
    try refused("\"n_group\": 8", "\"n_group\": 7", error.Routing, "groups");
    try refused("\"topk_group\": 4", "\"topk_group\": 9", error.Routing, "groups");
    try refused("\"num_experts\": 512", "\"num_experts\": 1024", error.Routing, "glue.select");
    try refused("\"norm_topk_prob\": true", "\"norm_topk_prob\": false", error.Routing, "norm_topk_prob");
    try refused("\"expert_swiglu_limit_list\": [\n    0,\n", "\"expert_swiglu_limit_list\": [\n", error.LimitList, "71 entries for 72");
    try refused("\"share_expert_swiglu_limit_list\": [\n    0,\n", "\"share_expert_swiglu_limit_list\": [\n    0,\n    0,\n", error.LimitList, "73 entries");
    try refused("\"share_expert_swiglu_limit_list\": [\n    0,\n", "\"share_expert_swiglu_limit_list\": [\n    -1,\n", error.LimitList, "[0]");
    try refused("\"expert_swiglu_limit_list\": [\n    0,\n", "\"expert_swiglu_limit_list\": [\n    7,\n", error.LimitList, "layer 0 is dense");
    try refused("\"quant_method\": \"fp8\"", "\"quant_method\": \"gptq\"", error.Quant, "gptq");
    try refused("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 2", error.Mtp, "MTP");
    try refused("\"mtp_use_kda\": false", "\"mtp_use_kda\": true", error.Mtp, "MTP");
    try refused("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", error.Activation, "hidden_act");
    try refused("\"use_qkv_bias\": false", "\"use_qkv_bias\": true", error.Bias, "bias");
    try refused("\"eos_token_id\": 156895", "\"eos_token_id\": 157184", error.BadConfig, "eos_token_id");
    try refused("\"num_hidden_layers\": 72", "\"num_hidden_layers\": 73", error.LimitList, "72 entries for 73");
    try refused("\"hidden_size\": 4096", "\"hidden_sz\": 4096", error.BadConfig, "hidden_size");
}

test "optional keys: no quantization reads bf16, a null q_lora_rank one q_proj" {
    const b = try edit("\"q_lora_rank\": 1536", "\"q_lora_rank\": null");
    defer t.allocator.free(b);
    const at = std.mem.indexOf(u8, b, "\"quantization_config\"").?;
    const end = std.mem.indexOf(u8, b[at..], "  },\n").? + at + 5;
    const c_json = try std.mem.concat(t.allocator, u8, &.{ b[0..at], b[end..] });
    defer t.allocator.free(c_json);
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, c_json, &why);
    try t.expectEqual(config.Quant.bf16, c.quant);
    try t.expectEqual(@as(u32, 0), c.q_lora);
    try t.expectEqual(config.RopeStyle.gptj, c.rope_style);
    try t.expect(!c.scaled("model.layers.0.attention.q_proj"));
}

/// The fixture without its quantization_config block.
fn plainJson() ![]u8 {
    const at = std.mem.indexOf(u8, real, "\"quantization_config\"").?;
    const end = std.mem.indexOf(u8, real[at..], "  },\n").? + at + 5;
    return std.mem.concat(t.allocator, u8, &.{ real[0..at], real[end..] });
}

test "ModelOpt NVFP4 from quantization_config: ignore globs and names pick the bf16 modules, the rest is nvfp4" {
    const json = try edit("\"quant_method\": \"fp8\",", "\"quant_method\": \"modelopt\", \"quant_algo\": \"NVFP4\", \"ignore\": [\"*.attention.b_proj\", \"model.layers.7.mlp*\"],");
    defer t.allocator.free(json);
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, json, &why);
    try t.expectEqual(config.Quant.nvfp4, c.quant);
    try t.expectEqual(config.Quant.nvfp4, c.format("model.layers.0.attention.q_proj"));
    try t.expectEqual(config.Quant.bf16, c.format("model.layers.30.attention.b_proj"));
    try t.expectEqual(config.Quant.bf16, c.format("model.layers.7.mlp.experts.3.up_proj"));
    try t.expectEqual(config.Quant.nvfp4, c.format("model.layers.17.mlp.experts.3.up_proj"));
    try t.expectEqual(config.Quant.bf16, c.format("lm_head")); // modules_to_not_convert stays an exception list too
}

test "hf_quant_config.json's NVFP4 block when config.json has none; groups other than 16 and other algorithms refused" {
    const json = try plainJson();
    defer t.allocator.free(json);
    var why: config.Why = .{};
    const hf = "{\"producer\": {\"name\": \"modelopt\"}, \"quantization\": {\"quant_algo\": \"NVFP4\", \"group_size\": 16, \"exclude_modules\": [\"lm_head\", \"model.layers.*.mlp.gate\"]}}";
    const c = try config.parseWith(t.allocator, json, hf, &why);
    try t.expectEqual(config.Quant.nvfp4, c.quant);
    try t.expectEqual(config.Quant.bf16, c.format("model.layers.40.mlp.gate"));
    try t.expectEqual(config.Quant.nvfp4, c.format("model.layers.40.mlp.gate_proj"));
    const g32 = "{\"quantization\": {\"quant_algo\": \"NVFP4\", \"group_size\": 32}}";
    try t.expectError(error.Quant, config.parseWith(t.allocator, json, g32, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "groups of 16") != null);
    const fp8 = "{\"quantization\": {\"quant_algo\": \"FP8\"}}";
    try t.expectError(error.Quant, config.parseWith(t.allocator, json, fp8, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "ModelOpt FP8") != null);
    try t.expectEqual(config.Quant.bf16, (try config.parseWith(t.allocator, json, null, &why)).quant);
}

test "EXL3 is refused naming #832; a glob matches whole names only" {
    try refused("\"quant_method\": \"fp8\"", "\"quant_method\": \"exl3\"", error.Exl3, "#832");
    try t.expect(config.glob("*.mlp.gate", "model.layers.4.mlp.gate"));
    try t.expect(!config.glob("*.mlp.gate", "model.layers.4.mlp.gate_proj"));
    try t.expect(config.glob("model.layers.*.attention*", "model.layers.12.attention.q_proj"));
    try t.expect(!config.glob("model.layers.1*", "model.layers.0.mlp"));
    try t.expect(config.glob("a*b*c", "a.b.x.c") and !config.glob("a*b*c", "a.c"));
}

const synth = @import("synth.zig");

/// The tiny config (synth.zig, FP8) with `key` set to `value` (JSON text) and `also`'s pair, parsed.
fn tiny(key: []const u8, value: []const u8, also: ?[2][]const u8, why: *config.Why) !config.Config {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var root = try std.json.parseFromSliceLeaky(std.json.Value, a, try synth.configJson(a, .fp8_block128, 4), .{});
    try root.object.put(a, key, try std.json.parseFromSliceLeaky(std.json.Value, a, value, .{}));
    if (also) |kv| try root.object.put(a, kv[0], try std.json.parseFromSliceLeaky(std.json.Value, a, kv[1], .{}));
    return config.parse(t.allocator, try std.json.Stringify.valueAlloc(a, root, .{}), why);
}

const Case = struct { key: []const u8, value: []const u8, err: anyerror, says: []const u8, also: ?[2][]const u8 = null };

test "geometry the kernels do not take is refused at load, one value at a time on the tiny config, with the reason" {
    const cases = [_]Case{
        .{ .key = "head_dim", .value = "64", .err = error.KdaGeometry, .says = "head_dim 64" },
        .{ .key = "head_dim", .value = "256", .err = error.KdaGeometry, .says = "128-wide heads" },
        .{ .key = "short_conv_kernel_size", .value = "3", .err = error.KdaGeometry, .says = "short_conv_kernel_size 3" },
        .{ .key = "short_conv_kernel_size", .value = "5", .err = error.KdaGeometry, .says = "4 taps" },
        .{ .key = "num_attention_heads", .value = "0", .err = error.KdaGeometry, .says = "0 attention heads" },
        .{ .key = "num_attention_heads", .value = "0", .err = error.KdaGeometry, .says = "0 attention heads, 0", .also = .{ "num_key_value_heads", "0" } },
        .{ .key = "num_key_value_heads", .value = "2", .err = error.KdaGeometry, .says = "2 key/value heads" },
        .{ .key = "kv_lora_rank", .value = "256", .err = error.MlaGeometry, .says = "kv_lora_rank 256" },
        .{ .key = "kv_lora_rank", .value = "1024", .err = error.MlaGeometry, .says = "take 512" },
        .{ .key = "qk_nope_head_dim", .value = "64", .err = error.MlaGeometry, .says = "qk_nope_head_dim 64" },
        .{ .key = "qk_nope_head_dim", .value = "192", .err = error.MlaGeometry, .says = "qk_nope_head_dim 192" },
        .{ .key = "v_head_dim", .value = "256", .err = error.MlaGeometry, .says = "v_head_dim 256" },
        .{ .key = "v_head_dim", .value = "64", .err = error.MlaGeometry, .says = "v_head_dim 64" },
        .{ .key = "qk_rope_head_dim", .value = "32", .err = error.MlaGeometry, .says = "qk_rope_head_dim 32" },
        .{ .key = "qk_rope_head_dim", .value = "128", .err = error.MlaGeometry, .says = "qk_rope_head_dim 128" },
        .{ .key = "rope_interleave", .value = "false", .err = error.MlaGeometry, .says = "GPT-J" },
        .{ .key = "q_lora_rank", .value = "768", .err = error.MlaGeometry, .says = "q_lora_rank 768" },
        .{ .key = "q_lora_rank", .value = "1024", .err = error.MlaGeometry, .says = "q_lora_rank 1024" },
        .{ .key = "hidden_size", .value = "1024", .err = error.Width, .says = "hidden_size 1024" },
        .{ .key = "hidden_size", .value = "640", .err = error.Width, .says = "hidden_size 640" },
        .{ .key = "intermediate_size", .value = "200", .err = error.Width, .says = "intermediate_size 200" },
        .{ .key = "intermediate_size", .value = "0", .err = error.Width, .says = "intermediate_size 0" },
        .{ .key = "moe_intermediate_size", .value = "96", .err = error.Width, .says = "moe_intermediate_size 96" },
        .{ .key = "moe_intermediate_size", .value = "192", .err = error.Width, .says = "moe_intermediate_size 192" },
        .{ .key = "moe_shared_expert_intermediate_size", .value = "64", .err = error.Width, .says = "shared expert's width 64" },
        .{ .key = "tie_word_embeddings", .value = "true", .err = error.TiedHead, .says = "lm_head" },
    };
    for (cases) |x| {
        var why: config.Why = .{};
        const got = tiny(x.key, x.value, x.also, &why);
        if (got) |_| {
            std.debug.print("{s} {s}: accepted\n", .{ x.key, x.value });
            return error.TestUnexpectedResult;
        } else |e| if (e != x.err or std.mem.indexOf(u8, why.text(), x.says) == null) {
            std.debug.print("{s} {s}: {t}, \"{s}\"\n", .{ x.key, x.value, e, why.text() });
            return error.TestUnexpectedResult;
        }
    }
}

test "the shapes the kernels take are accepted: the tiny config as written and with these values, the real one" {
    var why: config.Why = .{};
    const ok = [_][2][]const u8{
        .{ "head_dim", "128" },         .{ "short_conv_kernel_size", "4" },   .{ "num_key_value_heads", "4" },
        .{ "q_lora_rank", "1536" },     .{ "q_lora_rank", "null" },           .{ "hidden_size", "4096" },
        .{ "hidden_size", "1536" },     .{ "moe_intermediate_size", "1280" }, .{ "intermediate_size", "10240" },
        .{ "rope_interleave", "true" },
    };
    for (ok) |kv| _ = tiny(kv[0], kv[1], null, &why) catch |e| {
        std.debug.print("{s} {s}: {t}, \"{s}\"\n", .{ kv[0], kv[1], e, why.text() });
        return e;
    };
    const c = try load();
    try t.expect(c.kda_dim == config.fixed.kda_dim and c.conv == config.fixed.conv and c.kv_lora == config.fixed.kv_lora);
}

test "routing groups: groups of one and a single pick refused (HF's rule differs there); one group, all groups kept, groups of two accepted" {
    const refusals = [_]Case{
        .{ .key = "n_group", .value = "16", .err = error.Routing, .says = "16 groups of 1 experts", .also = .{ "topk_group", "4" } },
        .{ .key = "n_group", .value = "16", .err = error.Routing, .says = "best two", .also = .{ "topk_group", "16" } },
        .{ .key = "num_experts_per_tok", .value = "1", .err = error.Routing, .says = "num_experts_per_tok 1" },
        .{ .key = "n_group", .value = "1", .err = error.Routing, .says = "past glue.select's bounds" }, // keeping 2 of 1
    };
    for (refusals) |x| {
        var why: config.Why = .{};
        const got = tiny(x.key, x.value, x.also, &why);
        if (got) |_| {
            std.debug.print("{s} {s}: accepted\n", .{ x.key, x.value });
            return error.TestUnexpectedResult;
        } else |e| if (e != x.err or std.mem.indexOf(u8, why.text(), x.says) == null) {
            std.debug.print("{s} {s}: {t}, \"{s}\"\n", .{ x.key, x.value, e, why.text() });
            return error.TestUnexpectedResult;
        }
    }
    const ok = [_][4][]const u8{
        .{ "n_group", "1", "topk_group", "1" }, // one group: glue.select skips the group pass, HF keeps every expert
        .{ "n_group", "4", "topk_group", "4" }, // every group kept
        .{ "n_group", "8", "topk_group", "2" }, // groups of two
        .{ "n_group", "8", "topk_group", "8" },
    };
    for (ok) |kv| {
        var why: config.Why = .{};
        const c = tiny(kv[0], kv[1], .{ kv[2], kv[3] }, &why) catch |e| {
            std.debug.print("{s} {s}, {s} {s}: {t}, \"{s}\"\n", .{ kv[0], kv[1], kv[2], kv[3], e, why.text() });
            return e;
        };
        try t.expect(c.experts / c.n_group >= 2 and c.topk_group <= c.n_group);
    }
}

test "fixtures/config.json is inclusionAI/Ling-3.1-flash's at a6943f4b, byte for byte (oracle/ling_hf.py pins the same)" {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(real, &h, .{});
    try t.expectEqualStrings("6f3773e52a12509e772e698aacacaa156d2402ebb8250812a1da52a03d92f084", &std.fmt.bytesToHex(h, .lower));
}
