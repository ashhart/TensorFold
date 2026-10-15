//! The checkpoint's config.json against the shapes the CUDA path is built for: a different one is refused at load.

const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");

/// One config field and the value the kernels and buffers assume.
const Want = struct { name: []const u8, value: i64 };

/// The text config's shapes the CUDA path's constants hold.
const shapes = [_]Want{
    .{ .name = "hidden_size", .value = wt.dims },
    .{ .name = "num_hidden_layers", .value = wt.layers_n },
    .{ .name = "full_attention_interval", .value = 4 },
    .{ .name = "vocab_size", .value = wt.vocab },
    .{ .name = "num_attention_heads", .value = wt.q_heads },
    .{ .name = "num_key_value_heads", .value = wt.kv_heads },
    .{ .name = "head_dim", .value = wt.head_dim },
    .{ .name = "num_experts", .value = wt.experts_n },
    .{ .name = "num_experts_per_tok", .value = wt.slots - 1 },
    .{ .name = "moe_intermediate_size", .value = wt.moe_w },
    .{ .name = "shared_expert_intermediate_size", .value = wt.moe_w },
    .{ .name = "hc_count", .value = wt.streams },
    .{ .name = "hc_lowrank", .value = wt.low },
    .{ .name = "linear_num_key_heads", .value = gdn.nk },
    .{ .name = "linear_num_value_heads", .value = gdn.nv },
    .{ .name = "linear_key_head_dim", .value = gdn.dk },
    .{ .name = "linear_value_head_dim", .value = gdn.dv },
    .{ .name = "linear_conv_kernel_dim", .value = gdn.taps },
    .{ .name = "indexer_n_heads", .value = wt.index_heads },
    .{ .name = "indexer_head_dim", .value = wt.index_dim },
    .{ .name = "indexer_compress_ratio", .value = wt.index_ratio },
    .{ .name = "mtp_num_hidden_layers", .value = 1 },
};

fn int(o: std.json.ObjectMap, name: []const u8) ?i64 {
    const v = o.get(name) orelse return null;
    return if (v == .integer) v.integer else null;
}

/// The first config value that differs from what the CUDA path serves, or null when every one matches.
pub fn mismatch(text: std.json.ObjectMap, quant: ?std.json.ObjectMap) ?[]const u8 {
    const q = quant orelse return "quantization";
    if (int(q, "bits") != 4) return "quantization.bits";
    if (int(q, "group_size") != 32) return "quantization.group_size";
    if (q.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return "quantization.mode";
    for (shapes) |w| if (int(text, w.name) != w.value) return w.name;
    return null;
}

/// Reads `dir`/config.json; error.UnsupportedCheckpoint names the field in a log line when one differs.
pub fn check(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.UnsupportedCheckpoint;
    const root = parsed.value.object;
    const text = if (root.get("text_config")) |v| (if (v == .object) v.object else root) else root;
    const quant = if (root.get("quantization")) |v| (if (v == .object) v.object else null) else null;
    if (mismatch(text, quant)) |field| {
        std.log.err("flashnext: config.json's {s} differs from the MLX 4-bit group-32 Flash Next the CUDA path serves", .{field});
        return error.UnsupportedCheckpoint;
    }
}

test "the served shapes pass and a 6-bit checkpoint or another width is named" {
    const gpa = std.testing.allocator;
    var text: std.json.ObjectMap = .empty;
    defer text.deinit(gpa);
    for (shapes) |w| try text.put(gpa, w.name, .{ .integer = w.value });
    var quant: std.json.ObjectMap = .empty;
    defer quant.deinit(gpa);
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try quant.put(gpa, "group_size", .{ .integer = 32 });
    try std.testing.expectEqual(@as(?[]const u8, null), mismatch(text, quant));
    try quant.put(gpa, "bits", .{ .integer = 6 });
    try std.testing.expectEqualStrings("quantization.bits", mismatch(text, quant).?);
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try text.put(gpa, "hidden_size", .{ .integer = 4096 });
    try std.testing.expectEqualStrings("hidden_size", mismatch(text, quant).?);
    try std.testing.expectEqualStrings("quantization", mismatch(text, null).?);
}
