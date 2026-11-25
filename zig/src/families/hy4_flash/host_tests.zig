const std = @import("std");
const core = @import("core");
const engram = @import("engram.zig");
const engram_rows = @import("engram_rows.zig");
const loader = @import("loader.zig");

test {
    _ = @import("config.zig");
    _ = @import("attention_slots.zig");
    _ = @import("mla_attention.zig");
    _ = @import("index_groups.zig");
    _ = @import("rope.zig");
    _ = @import("rank_weights.zig");
    _ = @import("cuda_model.zig");
    _ = @import("index_recall.zig");
    _ = @import("forward.zig");
    _ = @import("exl3_weights.zig");
    _ = @import("fp8_weights_test.zig");
    _ = @import("nvfp4_weights_test.zig");
    _ = @import("fp8_pack_test.zig");
    _ = @import("vision_config_test.zig");
    _ = @import("hc_cuda.zig");
    _ = @import("ihc_fused_cuda.zig");
    _ = @import("route_cuda.zig");
    _ = @import("rows_cuda.zig");
    _ = @import("dfly2_glue_cuda.zig");
    _ = @import("engram_fp8_cuda.zig");
    if (@import("builtin").os.tag == .linux) _ = @import("bf16_rows_test.zig");
}

fn tensor(dtype: core.checkpoint_host.DType, rows: usize, width: usize, bytes: []const u8) core.checkpoint_host.Tensor {
    var result: core.checkpoint_host.Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(0), .bytes = bytes };
    result.shape[0..2].* = .{ rows, width };
    return result;
}

test "Engram gathers the selected affine rows in token and head order" {
    const n = engram.columns * 2;
    var words: [n][4]u32 = undefined;
    var offsets: [engram.columns]u32 = undefined;
    for (&words, 0..) |*row, i| row.* = @splat(if (i % 2 == 0) 0x11111111 else 0xeeeeeeee);
    for (&offsets, 0..) |*offset, i| offset.* = @intCast(i * 2);
    const scales: [n]u16 = @splat(0x3f80); // BF16 1.0
    const biases: [n]u16 = @splat(0xbf80); // BF16 -1.0
    const lookup = [2]u32{ 0, 1 };
    const table: engram_rows.Table = .{
        .map = .{ .bytes = std.mem.asBytes(&lookup), .pad = 0 },
        .hash = .{ .multipliers = .{ 3, 5, 7 }, .primes = @splat(2), .offsets = offsets },
        .affine = .{
            .words = tensor(.u32, n, 4, std.mem.asBytes(&words)),
            .scales = tensor(.bf16, n, 1, std.mem.asBytes(&scales)),
            .biases = tensor(.bf16, n, 1, std.mem.asBytes(&biases)),
            .bits = 4,
            .group = 32,
        },
        .width = 32,
    };
    const a = std.testing.allocator;
    const window0 = [3]u32{ 1, 0, 1 };
    const window1 = [3]u32{ 0xffffffff, 0, 1 };
    var out: [2 * engram.columns * 32]u16 = undefined;
    try table.gather(a, &.{ window0, window1 }, &out);
    for (out, 0..) |word, i| {
        const head = (i / 32) % engram.columns;
        const from_first = i / 32 < engram.columns;
        const on = from_first == (head < engram.heads);
        try std.testing.expectEqual(@as(u16, if (on) 0x4150 else 0), word); // BF16 13.0 or 0.0
    }
    try std.testing.expectError(error.BadEngramOutput, table.gather(a, &.{window0}, out[1..]));
}

const auxiliary_data = [2]u8{ 0, 0 };
const auxiliary_entries = [_]core.checkpoint_fixture.Entry{
    .{ .name = "model.norm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.token_key_projection.weight", .dtype = "U32", .shape = &.{1}, .bytes = &auxiliary_wide_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.token_key_projection.scales", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.token_key_projection.biases", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.query_projection.weight", .dtype = "U32", .shape = &.{1}, .bytes = &auxiliary_wide_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.query_projection.scales", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.query_projection.biases", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.head_weight_projection.weight", .dtype = "U32", .shape = &.{1}, .bytes = &auxiliary_wide_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.head_weight_projection.scales", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.head_weight_projection.biases", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.key_value_projection.weight", .dtype = "F32", .shape = &.{1}, .bytes = &auxiliary_wide_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.key_gate_projection.weight", .dtype = "F32", .shape = &.{1}, .bytes = &auxiliary_wide_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.intra_group_ape", .dtype = "F32", .shape = &.{1}, .bytes = &auxiliary_wide_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.key_norm.weight", .dtype = "F32", .shape = &.{1}, .bytes = &auxiliary_wide_data },
    .{ .name = "model.layers.0.self_attn.compressed_indexer.token_key_norm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.final_norm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.confidence_head.proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.confidence_head.proj.bias", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.mtp_fusion.enorm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.mtp_fusion.hnorm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.mtp_fusion.gate", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.mtp_fusion.eh_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.hidden_correction.embed_norm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.hidden_correction.hidden_norm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.hidden_correction.gate_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.hidden_correction.up_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.hidden_correction.down_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.input_layernorm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.post_attention_layernorm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.post_proj_norm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.q_a_layernorm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.kv_a_layernorm.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.q_a_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.q_b_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.kv_a_proj_with_mqa.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.kv_b_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.linear_gate.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.o_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.self_attn.learnable_sink_param", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.mlp.gate_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.mlp.up_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "model.dfly_layers.0.mlp.down_proj.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
    .{ .name = "vit.embeddings.patch_embedding.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data },
};

const auxiliary_wide_data = [4]u8{ 0, 0, 0, 0 };

test "the CUDA loader owns complete auxiliary tensors but refuses a stray" {
    const fixture = core.checkpoint_fixture;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const parsed = try @import("config.zig").parse(a, @embedFile("config_fixture.json"));
    defer parsed.deinit();
    var model = parsed.value;
    model.text_config.num_hidden_layers = 1;
    model.text_config.layer_types = &.{.deepseek_sparse_attention};
    model.text_config.mlp_layer_types = &.{.dense};
    model.text_config.engram_layer_ids = &.{};
    model.dfly_num_hidden_layers = 1;
    model.dfly_target_index_layer = &.{0};
    model.dfly_target_kv_layers = &.{0};
    for ([_]bool{ false, true }) |with_stray| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var entries: std.ArrayList(fixture.Entry) = .empty;
        defer entries.deinit(a);
        try entries.appendSlice(a, &auxiliary_entries);
        if (with_stray) try entries.append(a, .{ .name = "model.dfly_layers.stray.weight", .dtype = "BF16", .shape = &.{1}, .bytes = &auxiliary_data });
        try fixture.write(a, io, tmp.dir, "model.safetensors", entries.items);
        const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        defer a.free(dir);
        var ck = try core.Checkpoint.openModel(a, io, dir);
        defer ck.close();
        _ = try ck.expect("model.norm.weight", .bf16, &.{1});
        if (with_stray) {
            try std.testing.expectError(error.UnusedHy4Tensors, loader.finishCuda(&ck, model));
        } else {
            try loader.finishCuda(&ck, model);
        }
    }
}

test "a direct Engram table loads, while a stray hy4 tensor refuses at the checkpoint's last read site" {
    const fixture = @import("core").checkpoint_fixture;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data: [4]u16 = @splat(0x3f80);
    const table_name = "model.layers.0.engram_layer.multi_head_embedding.embedding.weight";
    const entries = [_]fixture.Entry{
        .{ .name = "model.norm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = table_name, .dtype = "BF16", .shape = &.{ 1, 4 }, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = "model.stray.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
    };
    try fixture.write(a, io, tmp.dir, "model.safetensors", &entries);
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var ck = try core.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    _ = try ck.expect("model.norm.weight", .bf16, &.{4});
    try ck.skip(table_name); // openDirect owns the read; only its exact table name is outside the used-set audit.
    try std.testing.expectError(error.UnusedHy4Tensors, @import("loader.zig").refuseLeftovers(&ck, error.UnusedHy4Tensors));
    _ = try ck.expect("model.stray.weight", .bf16, &.{4});
    try @import("loader.zig").refuseLeftovers(&ck, error.UnusedHy4Tensors);
}
