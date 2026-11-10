//! DSpine on CUDA: Tencent's 27B config is the bridge kind with its taps, window and attention chunks.
const std = @import("std");
const Config = @import("cuda_draft_config.zig").Config;
const Geometry = @import("cuda_engine.zig").Geometry;
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

fn target() Geometry {
    return .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
}

test "DSpine CUDA config: Tencent's 27B drafter is the bridge kind with its taps, 8,192 window and 33 chunks" {
    const c = try Config.parse(a, fixture, target());
    try t.expectEqual(Config.Kind.dspine, c.kind);
    try t.expectEqualSlices(u32, &.{ 1, 16, 31, 46, 61 }, c.taps[0..c.n_taps]);
    try t.expectEqual(@as(usize, 1024), c.code);
    try t.expectEqual(@as(usize, 512), c.rank);
    try t.expectEqual(@as(usize, 16), c.topk);
    try t.expectEqual(@as(usize, 16), c.block);
    try t.expectEqual(@as(usize, 8192), c.window);
    try t.expectEqual(@as(usize, 33), c.chunks());
    try t.expectEqual(@as(f32, 1e-6), c.bridge_eps);
}

test "DSpine CUDA config: a target of another width or depth refuses it before any weights load" {
    var g = target();
    g.layers = 32;
    try t.expectError(error.UnsupportedDraftConfig, Config.parse(a, fixture, g));
}

test "DSpine CUDA config: a served window narrows the checkpoint's 8,192 rows, and one outside a block..its own is refused" {
    var doc = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer doc.deinit();
    const dspine = @import("cuda_dspine_config.zig");
    const own = try dspine.parseWindow(doc.value, target(), 0);
    try t.expectEqual(@as(usize, 8192), own.window);
    const narrow = try dspine.parseWindow(doc.value, target(), 4096);
    try t.expectEqual(@as(usize, 4096), narrow.window);
    try t.expectEqual(@as(usize, 4096 + 16), narrow.ringRows());
    try t.expectEqual(@as(usize, 17), narrow.chunks());
    try t.expectEqual(@as(usize, 16), (try dspine.parseWindow(doc.value, target(), 16)).window);
    try t.expectError(error.DSpineWindowOutOfRange, dspine.parseWindow(doc.value, target(), 15));
    try t.expectError(error.DSpineWindowOutOfRange, dspine.parseWindow(doc.value, target(), 8193));
}

test "DSpine CUDA config: TF_DSPINE_WINDOW is read, unset or blank keeps the checkpoint's, a wider one is refused" {
    var doc = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer doc.deinit();
    const dspine = @import("cuda_dspine_config.zig");
    try t.expectEqual(@as(usize, 4096), (try dspine.parseRaw(doc.value, target(), "4096")).window);
    try t.expectEqual(@as(usize, 16), (try dspine.parseRaw(doc.value, target(), "16")).window);
    try t.expectEqual(@as(usize, 8192), (try dspine.parseRaw(doc.value, target(), null)).window);
    try t.expectEqual(@as(usize, 8192), (try dspine.parseRaw(doc.value, target(), " ")).window);
    try t.expectError(error.InvalidEnvironment, dspine.parseRaw(doc.value, target(), "15"));
    try t.expectError(error.InvalidEnvironment, dspine.parseRaw(doc.value, target(), "8193"));
    const own_2048 = try std.mem.replaceOwned(u8, a, fixture, "\"sliding_window\":8192", "\"sliding_window\":2048");
    defer a.free(own_2048);
    var narrow = try std.json.parseFromSlice(std.json.Value, a, own_2048, .{});
    defer narrow.deinit();
    try t.expectEqual(@as(usize, 2048), (try dspine.parseRaw(narrow.value, target(), null)).window);
    try t.expectError(error.DSpineWindowOutOfRange, dspine.parseRaw(narrow.value, target(), "4096"));
}
