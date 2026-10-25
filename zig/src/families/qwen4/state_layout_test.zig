//! Prefix plans keep only complete pooled blocks and count every linear and lexical frame before allocation.
const std = @import("std");
const cfg = @import("config.zig");
const layout = @import("state_layout.zig");
test "Qwen4 state layout: every persistent component and speculative row is counted without overlap" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    const l = try layout.Layout.of(&c, 32);
    try std.testing.expectEqual(@as(usize, 96), l.layers[0].conv.bytes);
    try std.testing.expectEqual(@as(usize, 128), l.layers[0].recurrent.bytes);
    try std.testing.expect(l.layers[0].ple.bytes > 0);
    try std.testing.expectEqual(@as(usize, 0), l.layers[0].key.bytes);
    try std.testing.expectEqual(@as(usize, 0), l.layers[1].conv.bytes);
    try std.testing.expectEqual(@as(usize, 512), l.layers[1].key.bytes);
    try std.testing.expectEqual(l.kv_bytes + 6 * l.state_bytes, try l.deviceBytes(4));
    for (0..32) |at| {
        const before = try l.savedBytes(&c, at);
        const after = try l.savedBytes(&c, at + 1);
        const ordinary = (2 * c.kv_heads * c.head_dim + c.index_dim) * 2;
        try std.testing.expectEqual(ordinary + (if ((at + 1) % c.index_ratio == 0) c.index_dim * 2 else 0), after - before);
    }
    try std.testing.expectError(error.Qwen4StateCapacity, layout.Layout.of(&c, 65));
    try std.testing.expectError(error.Qwen4Window, l.deviceBytes(0));
    try std.testing.expectError(error.Qwen4StateCapacity, l.savedBytes(&c, 33));
    for (l.layers[0..c.layers]) |layer| {
        const state = [_]layout.Range{ layer.conv, layer.recurrent, layer.ple };
        const kv = [_]layout.Range{ layer.key, layer.value, layer.index, layer.pooled };
        for ([_][]const layout.Range{ &state, &kv }) |ranges| for (ranges, 0..) |x, i| for (ranges[i + 1 ..]) |y| {
            if (x.bytes == 0 or y.bytes == 0) continue;
            try std.testing.expect(x.offset + x.bytes <= y.offset or y.offset + y.bytes <= x.offset);
        };
    }
}
