//! The DFlash drafter's config parse on hostile values: errors, never a trap.
const std = @import("std");
const cfg = @import("cuda_config.zig");
const dfl = @import("cuda_dflash.zig");

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
