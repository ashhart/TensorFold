//! Header-only tests cover the exact tensor map before a backend opens or allocates.
const std = @import("std");
const weights = @import("weights.zig");
const cfg = @import("config.zig");
const st = @import("core").safetensors;
const a = std.testing.allocator;
fn entry(dtype: st.DType, dims: []const usize) st.Entry {
    var e: st.Entry = .{ .dtype = dtype, .rank = @intCast(dims.len), .shape = @splat(1), .begin = 0, .end = dtype.size() };
    for (dims, 0..) |d, i| {
        e.shape[i] = d;
        e.end *= d;
    }
    return e;
}
const Sink = struct {
    a: std.mem.Allocator,
    map: st.Header = .empty,
    pub fn take(s: *@This(), name: []const u8, w: weights.Want) !void {
        const matrix = w.role == .matrix or w.role == .embedding;
        const key = try std.fmt.allocPrint(s.a, "{s}{s}", .{ name, if (matrix) ".weight" else "" });
        try s.map.put(s.a, key, entry(if (w.role == .integer_vector) .i64 else .bf16, w.shape.dims[0..w.shape.rank]));
    }
};
test "Qwen4 weights: exact dense HC GDN QSA PLE names and dimensions, leftovers refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var sink = Sink{ .a = arena.allocator() };
    try weights.walk(&c, .causal, &sink, &p);
    const stats = try weights.admit(a, &c, .causal, sink.map, &p);
    try std.testing.expectEqual(sink.map.count(), stats.tensors);
    try std.testing.expectEqualSlices(usize, &.{ 32, 16 }, sink.map.get("model.layers.0.mlp.gate_proj.weight").?.shape[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 32, 16 }, sink.map.get("model.layers.1.self_attn.q_proj.weight").?.shape[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 88, 4 }, sink.map.get("model.layers.0.ple.ple_embedding.ngram_embedding.weight").?.shape[0..2]);
    try sink.map.put(arena.allocator(), "model.layers.0.mlp.experts.weight", entry(.bf16, &.{ 2, 32, 16 }));
    try std.testing.expectError(error.UnsupportedQwen4, weights.admit(a, &c, .causal, sink.map, &p));
    try std.testing.expectEqualStrings("model.layers.0.mlp.experts.weight", p.name());
}
test "Qwen4 weights: stored MLX and ModelOpt layouts cannot be reinterpreted" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var map: st.Header = .empty;
    var p: cfg.Problem = .{};
    var w: weights.Want = .{ .role = .matrix, .shape = .{ .rank = 2, .dims = .{ 8, 64, 1 } }, .recipe = .{ .format = .mlx4, .group = 32 } };
    try map.put(g, "p.weight", entry(.u32, &.{ 8, 8 }));
    try map.put(g, "p.scales", entry(.bf16, &.{ 8, 2 }));
    try map.put(g, "p.biases", entry(.bf16, &.{ 8, 2 }));
    try weights.headers("p", w, map, &p);
    w.recipe.format = .mlx8;
    try std.testing.expectError(error.UnsupportedQwen4, weights.headers("p", w, map, &p));
    try map.put(g, "p.weight", entry(.u32, &.{ 8, 16 }));
    try weights.headers("p", w, map, &p);
    w.recipe = .{ .format = .fp8 };
    try map.put(g, "p.weight", entry(.f8_e4m3, &.{ 8, 64 }));
    try map.put(g, "p.weight_scale", entry(.f32, &.{}));
    try weights.headers("p", w, map, &p);
    w.recipe = .{ .format = .nvfp4, .group = 16 };
    try map.put(g, "p.weight", entry(.u8, &.{ 8, 32 }));
    try map.put(g, "p.weight_scale", entry(.f8_e4m3, &.{ 8, 4 }));
    try map.put(g, "p.weight_scale_2", entry(.f32, &.{1}));
    try weights.headers("p", w, map, &p);
    try map.put(g, "p.weight_scale", entry(.f8_e8m0, &.{ 8, 4 }));
    try std.testing.expectError(error.UnsupportedQwen4, weights.headers("p", w, map, &p));
    try std.testing.expectEqualStrings("p.weight_scale", p.name());
}
