//! Tensors the engine reads raw (experts, embedding, n-gram tables) checked at pack build against declared widths.
const std = @import("std");
const st = @import("../../core/safetensors.zig");
const ckpt = @import("../../core/checkpoint.zig");
const config_mod = @import("config.zig");
const Widths = @import("widths.zig").Widths;

fn entry(t: st.Tensor) st.Entry {
    return .{ .dtype = t.dtype, .rank = t.rank, .shape = t.shape, .begin = 0, .end = t.bytes.len };
}

/// Module `name`'s stored affine triple against the width declared for `key` (unquantized: admission's to judge).
pub fn check(ck: *ckpt.Checkpoint, w: *const Widths, key: []const u8, name: []const u8) !void {
    var buf: [3][256]u8 = undefined;
    var e: [3]st.Entry = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }) |p| if (!ck.has(try std.fmt.bufPrint(&buf[0], "{s}.{s}", .{ name, p }))) return;
    for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |p, i| e[i] = entry(try ck.get(try std.fmt.bufPrint(&buf[i], "{s}.{s}", .{ name, p })));
    _ = w.shaped(key, e[0], e[1], e[2]) catch |err| {
        std.log.warn("{s}: not stored at its declared width: {s}", .{ name, @errorName(err) });
        return err;
    };
}

const projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };

/// Every raw-read module: each layer's and the MTP head's experts, the embedding and the n-gram shards.
pub fn checkAll(ck: *ckpt.Checkpoint, ckm: *ckpt.Checkpoint, w: *const Widths, cfg: *const config_mod.Config, mtp_root: []const u8, spelling: []const u8) !void {
    var buf: [256]u8 = undefined;
    var key: [256]u8 = undefined;
    for (0..cfg.layers) |i| for (projs) |p| {
        const name = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.mlp.{s}", .{ i, p });
        try check(ck, w, name, name);
    };
    if (cfg.mtp.layers > 0) for (projs) |p| {
        try check(ckm, w, try std.fmt.bufPrint(&key, "language_model.mtp.layers.0.mlp.{s}", .{p}), try std.fmt.bufPrint(&buf, "{s}.layers.0.mlp.{s}", .{ mtp_root, p }));
    };
    try check(ck, w, "language_model.model.embed_tokens", "language_model.model.embed_tokens");
    for (0..cfg.layers) |i| if (cfg.ple[i]) for (0..cfg.ngram_shards) |k| {
        const name = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.{s}{d}", .{ i, spelling, k });
        try check(ck, w, name, name);
    };
}
