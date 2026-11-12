//! GGUF sources' F32 q/k norm gains reach both attention head bindings; native checkpoints keep BF16.
const std = @import("std");
const attention = @import("attention.zig");
const glue = @import("glue.zig");

test "GGUF attention: native gain defaults to BF16 and exact F32 gains reach both head bindings" {
    var p: attention.Preprocess = .{ .qg = undefined, .key = undefined, .q_gain = undefined, .k_gain = undefined, .norm_q = undefined, .norm_k = undefined, .query = undefined, .rotated_key = undefined, .positions = undefined, .rows = 3, .eps = 1e-6, .theta = 1e7 };
    try std.testing.expectEqual(glue.Float.bf16, attention.prepHead(p, 24, 512).gain);
    p.gain = .f32;
    const q = attention.prepHead(p, 24, 512);
    const k = attention.prepHead(p, 4, 256);
    try std.testing.expectEqual(glue.Float.f32, q.gain);
    try std.testing.expectEqual(glue.Float.f32, k.gain);
    try std.testing.expectEqual(@as(u32, 72), q.rows);
    try std.testing.expectEqual(@as(u32, 12), k.rows);
    try std.testing.expectEqual(@as(u32, 256), q.width);
    try std.testing.expectEqual(@as(u32, 512), q.stride);
}
