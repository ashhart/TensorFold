//! Inventory admission and complete local checkpoint header checks for dots.mocr.
const std = @import("std");
const weights = @import("weights.zig");
const cfg = @import("config.zig");
const cases = @import("config_test.zig");
const st = @import("core").safetensors;

test "dots: inventory covers the released parameter count and biased projections" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try cfg.Config.parse(a, cases.sample, cases.generation);
    const specs = try weights.inventory(a, c);
    try std.testing.expectEqual(@as(usize, 643), specs.len);
    var entries: st.Header = .empty;
    var bytes: usize = 0;
    for (specs) |s| {
        var e: st.Entry = .{ .dtype = .bf16, .rank = @intCast(s.rank), .shape = @splat(1), .begin = bytes, .end = bytes + s.bytes() };
        @memcpy(e.shape[0..s.rank], s.shape[0..s.rank]);
        try std.testing.expect(!entries.contains(s.name));
        try entries.put(a, s.name, e);
        bytes += s.bytes();
    }
    try std.testing.expectEqual(@as(usize, 6078358528), try weights.check(a, c, entries));
    try std.testing.expectEqual(@as(usize, 256), entries.get("model.layers.0.self_attn.k_proj.bias").?.shape[0]);
    try std.testing.expectEqual(@as(usize, 6144), entries.get("vision_tower.merger.mlp.0.bias").?.shape[0]);
    try std.testing.expect(entries.contains("vision_tower.patch_embed.patchifier.proj.bias"));
    const key = "model.layers.0.self_attn.q_proj.bias";
    const original = entries.get(key).?;
    entries.getPtr(key).?.dtype = .f16;
    try std.testing.expectError(error.DotsDType, weights.check(a, c, entries));
    entries.getPtr(key).?.* = original;
    entries.getPtr(key).?.shape[0] -= 1;
    try std.testing.expectError(error.DotsShape, weights.check(a, c, entries));
    entries.getPtr(key).?.* = original;
    entries.getPtr(key).?.end -= 2;
    try std.testing.expectError(error.DotsExtent, weights.check(a, c, entries));
    _ = entries.swapRemove(key);
    try entries.put(a, "model.layers.0.self_attn.o_proj.bias", original);
    try std.testing.expectError(error.DotsMissingTensor, weights.check(a, c, entries));
}

test "dots: existing local checkpoint headers agree with index and every graph tensor" {
    const dir = @import("sys").getenv("TF_DOTS_DIR") orelse return error.SkipZigTest;
    const c = try cfg.Config.read(std.testing.allocator, std.testing.io, dir);
    const s = try weights.inspect(std.testing.allocator, std.testing.io, dir, c);
    try std.testing.expectEqual(@as(usize, 643), s.tensors);
    try std.testing.expectEqual(@as(usize, 2), s.shards);
    try std.testing.expectEqual(@as(usize, 6078358528), s.weight_bytes);
    try std.testing.expectEqual(@as(usize, 6078431736), s.file_bytes);
}
