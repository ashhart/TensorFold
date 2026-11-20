//! The weight map against the NVFP4 checkpoint's own tensor headers (fixtures/header.txt) and its refusals.
const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");

const real = @embedFile("fixtures/config.json");
const header = @embedFile("fixtures/header.txt");
const t = std.testing;

fn load() !config.Config {
    var why: config.Why = .{};
    return config.parse(t.allocator, real, &why);
}

/// "3072x1536" to its sizes; "scalar" to none.
fn sizes(text: []const u8, out: *[4]u64) ![]const u64 {
    if (std.mem.eql(u8, text, "scalar")) return out[0..0];
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, 'x');
    while (it.next()) |part| : (n += 1) out[n] = try std.fmt.parseInt(u64, part, 10);
    return out[0..n];
}

test "every tensor of the checkpoint header is known, in its format and shape" {
    const c = try load();
    var seen = std.EnumArray(names.Part, u32).initFill(0);
    var lines = std.mem.tokenizeScalar(u8, header, '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next().?;
        const dtype = f.next().?;
        var buf: [4]u64 = undefined;
        const dims = try sizes(f.next().?, &buf);
        var why: config.Why = .{};
        const part = names.check(&c, name, dtype, dims, &why) catch |err| {
            std.debug.print("{s}: {s}\n", .{ line, why.text() });
            return err;
        };
        seen.set(part, seen.get(part) + 1);
    }
    // layers 0, 1 and 2, experts 0 and 511 of the two MoE layers, drafter layer 0, the vision tower's first block
    for (std.enums.values(names.Part)) |p| try t.expect(seen.get(p) > 0);
    try t.expectEqual(@as(u32, 3 * 10), seen.get(.attention));
    try t.expectEqual(@as(u32, 2 * 8), seen.get(.indexer)); // layers 0 and 1 are sparse
    try t.expectEqual(@as(u32, 2 * (2 * 12 + 3)), seen.get(.experts)); // two experts and the latent parts, twice
}

fn refused(name: []const u8, dtype: []const u8, dims: []const u64, want: anyerror, says: []const u8) !void {
    const c = try load();
    var why: config.Why = .{};
    try t.expectError(want, names.check(&c, name, dtype, dims, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), says) != null);
}

test "a sliding layer has 64 heads and no indexer; a sparse layer has 32 and one" {
    const c = try load();
    try t.expectEqual(@as(u64, 64 * 256), names.expect(&c, "model.layers.2.self_attn.q_b_proj.weight").?.dims.d[0]);
    try t.expectEqual(@as(u64, 32 * 256), names.expect(&c, "model.layers.0.self_attn.q_b_proj.weight").?.dims.d[0]);
    try t.expect(names.expect(&c, "model.layers.2.self_attn.compressed_indexer.key_norm.weight") == null);
    try t.expect(names.expect(&c, "model.layers.0.self_attn.compressed_indexer.key_norm.weight") != null);
}

test "Engram tensors exist on the Engram layers only, and the dense MLP on layer 0 only" {
    const c = try load();
    try t.expect(names.expect(&c, "model.layers.1.engram_layer.lookup_table") != null);
    try t.expect(names.expect(&c, "model.layers.2.engram_layer.lookup_table") == null);
    try t.expect(names.expect(&c, "model.layers.0.mlp.gate_proj.weight") != null);
    try t.expect(names.expect(&c, "model.layers.1.mlp.gate_proj.weight") == null);
    try t.expect(names.expect(&c, "model.layers.0.mlp.gate.weight") == null);
}

test "an unknown name, another format and another shape are refused by name" {
    try refused("model.layers.42.input_layernorm.weight", "BF16", &.{3072}, error.UnknownTensor, "not a Hy4-flash tensor");
    try refused("model.layers.1.mlp.experts.512.gate_proj.weight", "U8", &.{ 1536, 768 }, error.UnknownTensor, "not a Hy4-flash tensor");
    try refused("model.layers.1.mlp.experts.0.gate_proj.weight", "F32", &.{ 1536, 1536 }, error.TensorFormat, "the family reads u8 there");
    try refused("model.layers.1.mlp.experts.0.gate_proj.weight", "BF16", &.{ 1536, 768 }, error.TensorShape, "another shape");
    try refused("model.layers.1.mlp.experts.0.gate_proj.weight_scale", "F8_E4M3", &.{ 1536, 48 }, error.TensorShape, "another shape");
    try refused("model.layers.1.engram_layer.multi_head_embedding.embedding.weight", "F32", &.{ 164994380, 96 }, error.TensorFormat, "reads e4m3 there");
    try refused("model.layers.1.engram_layer.multi_head_embedding.embedding.weight", "BF16", &.{ 164994380, 128 }, error.TensorShape, "another shape");
    try refused("model.layers.2.self_attn.learnable_sink_param", "F32", &.{32}, error.TensorShape, "another shape");
}

test "transformers' tiny export (BF16, fused experts, indexed_attention) maps with the same rules" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, @embedFile("fixtures/tiny_config.json"), &why);
    try t.expectEqual(@as(u32, 3), c.text.layers);
    try t.expectEqual(@as(u32, 2), c.text.count(.sparse));
    var lines = std.mem.tokenizeScalar(u8, @embedFile("fixtures/tiny_header.txt"), '\n');
    var n: u32 = 0;
    while (lines.next()) |line| : (n += 1) {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next().?;
        const dtype = f.next().?;
        var buf: [4]u64 = undefined;
        const dims = try sizes(f.next().?, &buf);
        _ = names.check(&c, name, dtype, dims, &why) catch |err| {
            std.debug.print("{s}: {s}\n", .{ line, why.text() });
            return err;
        };
    }
    try t.expect(n > 100);
}
