//! The DFlash drafter's config parse on hostile values: errors, never a trap.
const std = @import("std");
const cfg = @import("cuda_config.zig");
const dfl = @import("cuda_dflash.zig");
const core = @import("core");

const stock =
    \\{"architectures":["DFlashDraftModel"],"dflash_config":{"mask_token_id":4,"target_layer_ids":[1,12,23,35,46,57]},"block_size":16,
    \\"hidden_size":5376,"intermediate_size":10752,"num_hidden_layers":5,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,
    \\"sliding_window":2048,"rms_norm_eps":1e-6,"rope_theta":1000000}
;

fn refused(from: []const u8, to: []const u8) !void {
    const text = try std.mem.replaceOwned(u8, std.testing.allocator, stock, from, to);
    defer std.testing.allocator.free(text);
    try std.testing.expectError(error.BadDraftConfig, dfl.Config.parse(std.testing.allocator, text, cfg.dense31b));
}

test "a zero, negative or oversized count is a bad config, not a trap" {
    try refused("\"num_key_value_heads\":8", "\"num_key_value_heads\":0");
    try refused("\"hidden_size\":5376", "\"hidden_size\":-1");
    try refused("\"block_size\":16", "\"block_size\":1099511627776");
    try refused("\"num_attention_heads\":64", "\"num_attention_heads\":9223372036854775807");
    try refused("\"sliding_window\":2048", "\"sliding_window\":0");
    try refused("\"mask_token_id\":4", "\"mask_token_id\":-4");
}

test "a stray draft tensor refuses; the shared dir's model and vision names never index" {
    const fixture = @import("core").checkpoint_fixture;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data: [4]u16 = @splat(0x3f80);
    const entries = [_]fixture.Entry{
        .{ .name = "fc.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = "layers.0.self_attn.q_proj.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = "model.embed_tokens.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = "vision_tower.patch_embedder.input_proj.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = "layers.1.stray.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
    };
    try fixture.write(a, io, tmp.dir, "model.safetensors", &entries);
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var ck = try core.checkpoint.Checkpoint.openModelPrefixes(a, io, dir, &.{ "fc.", "hidden_norm.", "norm.", "layers." });
    defer ck.close();
    _ = try ck.expect("fc.weight", .bf16, &.{4});
    _ = try ck.expect("layers.0.self_attn.q_proj.weight", .bf16, &.{4});
    try std.testing.expectError(error.UnusedDFlashTensors, dfl.refuseLeftovers(&ck, error.UnusedDFlashTensors));
    _ = try ck.expect("layers.1.stray.weight", .bf16, &.{4});
    try dfl.refuseLeftovers(&ck, error.UnusedDFlashTensors); // model./vision_ names never indexed: nothing left over
}
