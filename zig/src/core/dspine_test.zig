//! DSpine admission: Tencent's 27B config parses to its shape; another second pass, decode, write or layout is refused.
const std = @import("std");
const dspine = @import("dspine.zig");
const a = std.testing.allocator;
const t = std.testing;

// tencent/Qwen3.8-27B-DSpine-b16's config.json (rev ea909c0f5f) without its training-only keys.
const fixture =
    \\{"architectures":["DFlashDraftModel"],"attention_bias":false,"attention_dropout":0.0,"block_size":16,
    \\"code_bridge_second_pass":{"first_exact":4,"head_topk":16,"mode":"serial_table"},"dflash_config":{
    \\"draft_block_causal":true,"mask_token_id":248070,"selected_prefix_code_bridge":{"code_size":1024,"eps":1e-06,
    \\"kappa":10.0,"rank":512},"selected_prefix_decode_mode":"serial","selected_prefix_feature_layers":[1,2,3,4,5],
    \\"selected_prefix_feature_write_mode":"independent","selected_prefix_feature_writes":true,
    \\"selected_prefix_last_write_refine":{"head_topk":16,"kind":"exact_sender","version":1,"weight":1.0},
    \\"selected_prefix_terminal":false,"target_layer_ids":[1,16,31,46,61]},"dtype":"bfloat16","head_dim":128,
    \\"hidden_act":"silu","hidden_size":5120,"intermediate_size":17408,"is_causal":true,"layer_types":[
    \\"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
    \\"model_type":"qwen3","num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,
    \\"num_target_layers":64,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000000,"rope_type":"default"},
    \\"rope_scaling":null,"sliding_window":8192,"tie_word_embeddings":false,"use_sliding_window":true,
    \\"vocab_size":248320}
;

fn parsed(text: []const u8) !dspine.Shape {
    const doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    return dspine.parse(doc.value);
}

test "DSpine config: Tencent's 27B drafter, its taps, bridge widths and 8,192 window" {
    const s = try parsed(fixture);
    try t.expectEqualSlices(u32, &.{ 1, 16, 31, 46, 61 }, s.tapIds());
    try t.expectEqual(@as(usize, 1024), s.code);
    try t.expectEqual(@as(usize, 512), s.rank);
    try t.expectEqual(@as(usize, 16), s.topk);
    try t.expectEqual(@as(usize, 16), s.block);
    try t.expectEqual(@as(usize, 8192), s.window);
    try t.expectEqual(@as(u32, 248070), s.mask);
    try t.expectEqual(@as(f32, 1e-6), s.bridge_eps);
    const doc = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer doc.deinit();
    try t.expect(dspine.declared(doc.value));
}

test "DSpine config: another second pass, decode mode, write mode or a non-causal block is refused" {
    const edits = [_][2][]const u8{
        .{ "\"mode\":\"serial_table\"", "\"mode\":\"parallel\"" },
        .{ "\"kind\":\"exact_sender\"", "\"kind\":\"mean_sender\"" },
        .{ "\"selected_prefix_decode_mode\":\"serial\"", "\"selected_prefix_decode_mode\":\"parallel\"" },
        .{ "\"selected_prefix_feature_write_mode\":\"independent\"", "\"selected_prefix_feature_write_mode\":\"shared\"" },
        .{ "\"draft_block_causal\":true", "\"draft_block_causal\":false" },
        .{ "\"selected_prefix_terminal\":false", "\"selected_prefix_terminal\":true" },
        .{ "\"head_topk\":16,\"mode\"", "\"head_topk\":8,\"mode\"" },
        .{ "\"block_size\":16", "\"block_size\":8" },
        .{ "\"sliding_window\":8192", "\"sliding_window\":16384" },
        .{ "[1,2,3,4,5]", "[1,2,3,4]" },
        .{ "\"DFlashDraftModel\"", "\"DFlash2DraftModel\"" },
        .{ "\"rope_scaling\":null", "\"rope_scaling\":{\"type\":\"yarn\"}" },
    };
    for (edits) |e| {
        const bad = try std.mem.replaceOwned(u8, a, fixture, e[0], e[1]);
        defer a.free(bad);
        try t.expectError(error.UnsupportedDraftConfig, parsed(bad));
    }
    const missing = try std.mem.replaceOwned(u8, a, fixture, "\"selected_prefix_feature_writes\":true,", "");
    defer a.free(missing);
    try t.expectError(error.BadDraftConfig, parsed(missing));
}

test "DSpine config: a DFlash2 config is not declared DSpine" {
    const doc = try std.json.parseFromSlice(std.json.Value, a, "{\"dflash_config\":{\"block_size\":8}}", .{});
    defer doc.deinit();
    try t.expect(!dspine.declared(doc.value));
}

test "DSpine table: the walk follows the sender's row; the sender is the chosen candidate's index" {
    const cand = [_]u32{ 10, 11, 20, 21 };
    // [row][sender][candidate]: row 1 after 10 prefers 21, after 11 prefers 20
    const table = [_]f32{ 1, 0, 1, 0, 0, 2, 3, 0 };
    var out: [2]u32 = undefined;
    dspine.chain(&cand, &table, 2, &out);
    try t.expectEqualSlices(u32, &.{ 10, 21 }, &out);
    try t.expectEqual(@as(?usize, 1), dspine.sender(1, cand[0..2], 11));
    try t.expectEqual(@as(?usize, 0), dspine.sender(0, cand[0..2], 99));
    try t.expectEqual(@as(?usize, null), dspine.sender(1, cand[0..2], 99));
}

test "DSpine checkpoint: 71 tensors for five layers, each name found once with its dims; others are not" {
    const s = try parsed(fixture);
    const names = try dspine.names(a, s.layers);
    defer dspine.freeNames(a, names);
    try t.expectEqual(@as(usize, 71), names.len);
    for (names) |name| try t.expect(dspine.find(name, s.layers) != null);
    var dims: [3]usize = undefined;
    const early = dspine.find("selected_prefix_bridge.early_read_weights", s.layers).?;
    try t.expectEqualSlices(usize, &.{ 4, 1024, 5120 }, dspine.dims(s, early, &dims));
    try t.expectEqualSlices(usize, &.{ 5120, 25600 }, dspine.dims(s, dspine.find("fc.weight", s.layers).?, &dims));
    try t.expectEqual(dspine.Kind.i64, dspine.find("draft_block_causal_version", s.layers).?.kind);
    try t.expectEqual(@as(usize, 0), dspine.dims(s, dspine.find("selected_prefix_bridge.terminal_disabled_version", s.layers).?, &dims).len);
    for ([_][]const u8{ "layers.5.mlp.up_proj.weight", "layers.01.mlp.up_proj.weight", "codebook", "selected_prefix_bridge.fc.weight", "lm_head.weight" }) |name| {
        try t.expect(dspine.find(name, s.layers) == null);
    }
}

test "DSpine TG: the device head input for column k is bf16(norm * W[:, k])" {
    const bf = @import("bf16_round");
    // hidden 2, rank 2: W [hidden][rank]
    const norm = [_]u16{ bf.bits(2), bf.bits(0.5) };
    const write = [_]u16{ bf.bits(1), bf.bits(3), bf.bits(4), bf.bits(-2) };
    var input: [2]u16 = undefined;
    dspine.tgInput(&norm, &write, 2, 1, &input);
    try t.expectEqualSlices(u16, &.{ bf.bits(6), bf.bits(-1) }, &input);
}
