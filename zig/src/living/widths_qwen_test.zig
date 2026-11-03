//! The supplied dense checkpoint config and independent served matrix identities exercise learner width admission.
const std = @import("std");
const t = std.testing;
const widths = @import("widths.zig");
const family = @import("families.zig").qwen3_5;
const config = @embedFile("testdata/qwen27-mlx-config.json");
fn root(a: std.mem.Allocator) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, config, .{})).object;
}
test "Dense Qwen widths: supplied 27B MLX config admits before any model or device access" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    try t.expect((try widths.other(arena.allocator(), family, try root(arena.allocator()))) == null);
}
const linear = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj" };
const attention = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj" };
const mlp = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
fn override(a: std.mem.Allocator, doc: *std.json.ObjectMap, name: []const u8, bits: i64, group: i64) !void {
    var v: std.json.ObjectMap = .{};
    try v.put(a, "bits", .{ .integer = bits });
    try v.put(a, "group_size", .{ .integer = group });
    try doc.getPtr("quantization").?.object.put(a, name, .{ .object = v });
}
fn bad(key: []const u8, named: []const u8, bits: i64, group: i64) !void {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var doc = try root(a);
    try override(a, &doc, key, bits, group);
    const mismatch = try widths.other(a, family, doc);
    try t.expect(mismatch != null);
    const m = mismatch.?;
    try t.expectEqualStrings(named, m.name);
    try t.expectEqual(bits, @as(i64, m.quant.bits));
    try t.expectEqual(group, @as(i64, @intCast(m.quant.group)));
}
test "Dense Qwen widths: every served matrix rejects a mixed width by its checkpoint name" {
    try bad("language_model.model.embed_tokens", "language_model.model.embed_tokens", 8, 64);
    try bad("language_model.lm_head", "language_model.lm_head", 4, 32);
    var count: usize = 2;
    for (0..64) |i| {
        const mixer: []const []const u8 = if ((i + 1) % 4 == 0) &attention else &linear;
        for ([_][]const []const u8{ mixer, &mlp }) |members| for (members) |member| {
            var buf: [128]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.{s}", .{ i, member });
            try bad(name, name, 8, 64);
            count += 1;
        };
    }
    try t.expectEqual(@as(usize, 498), count);
}
test "Dense Qwen widths: native prefix aliases and weight suffixes cannot bypass admission" {
    for ([_][]const u8{ "", "model.", "language_model.", "language_model.model.", "model.language_model.", "model.language_model.model.", "text_model.", "text_model.model." }) |prefix| {
        for ([_][]const u8{ "", ".weight" }) |suffix| {
            for ([_][2][]const u8{ .{ "layers.0.linear_attn.in_proj_b", "language_model.model.layers.0.linear_attn.in_proj_b" }, .{ "layers.3.self_attn.v_proj", "language_model.model.layers.3.self_attn.v_proj" }, .{ "embed_tokens", "language_model.model.embed_tokens" }, .{ "lm_head", "language_model.lm_head" } }) |pair| {
                var buf: [160]u8 = undefined;
                try bad(try std.fmt.bufPrint(&buf, "{s}{s}{s}", .{ prefix, pair[0], suffix }), pair[1], 6, 64);
            }
        }
    }
}
test "Dense Qwen widths: MTP, vision and the other mixer kind are not served matrices" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var doc = try root(a);
    for ([_][]const u8{ "language_model.mtp.layers.0.mlp.gate_proj", "vision_tower.blocks.0.attn.qkv", "language_model.model.layers.0.self_attn.q_proj", "language_model.model.layers.3.linear_attn.in_proj_a", "language_model.model.layers.64.mlp.gate_proj" }) |name| try override(a, &doc, name, 8, 64);
    try t.expect((try widths.other(a, family, doc)) == null);
}
test "Dense Qwen widths: explicit layer types precede interval and missing types use the native default" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var doc = try root(a);
    const text = &doc.getPtr("text_config").?.object;
    try text.put(a, "full_attention_interval", .{ .integer = 1 });
    try override(a, &doc, "language_model.model.layers.0.self_attn.q_proj", 8, 64);
    try t.expect((try widths.other(a, family, doc)) == null);
    try t.expect(text.swapRemove("layer_types"));
    try t.expectEqualStrings("language_model.model.layers.0.self_attn.q_proj", (try widths.other(a, family, doc)).?.name);
    try t.expect(text.swapRemove("full_attention_interval"));
    try t.expect((try widths.other(a, family, doc)) == null);
    try override(a, &doc, "language_model.model.layers.3.self_attn.o_proj", 4, 128);
    try t.expectEqualStrings("language_model.model.layers.3.self_attn.o_proj", (try widths.other(a, family, doc)).?.name);
}
