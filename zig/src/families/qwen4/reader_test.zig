//! The payload constants must match the config; header shape checks alone cannot prove their hashes.
const std = @import("std");
const cfg = @import("config.zig");
const Reader = @import("reader.zig").Reader;
const fixture = @import("reader_fixture.zig");
test "Qwen4 reader: real mapped checkpoint owns its storage and checks PLE hash payloads" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try fixture.write(tmp, arena.allocator(), &c, false);
    var reader = try Reader.open(a, std.testing.io, dir, &c, &p);
    try std.testing.expect(reader.stats.matrices > 20);
    const t = try reader.get("model.layers.0.mlp.gate_proj.weight");
    try std.testing.expect(t.is(.bf16, &.{ 32, 16 }));
    reader.close();
    try fixture.write(tmp, arena.allocator(), &c, true);
    try std.testing.expectError(error.UnsupportedQwen4, Reader.open(a, std.testing.io, dir, &c, &p));
    try std.testing.expectEqualStrings("model.layers.0.ple.ple_embedding.layer_multipliers", p.name());
}
test "Qwen4 reader: an MLX namespace needs an explicit normalization convention, never a value-based guess" {
    const a = std.testing.allocator;
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    const check = @import("reader.zig").normPolicy;
    try check(.causal, &c, &p);
    try std.testing.expectError(error.UnsupportedQwen4, check(.mlx, &c, &p));
    try std.testing.expectEqualStrings("norm_weight_mode", p.name());
    c.norm_explicit = true;
    c.norm_weights = .scale;
    try check(.mlx, &c, &p);
    c.norm_weights = .hf_gamma;
    try check(.mlx, &c, &p);
}
