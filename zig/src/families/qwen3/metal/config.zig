//! Qwen3's dense Metal geometry, read through the shared Qwen3 config admission.
const std = @import("std");
const common = @import("../config.zig");
const llama = @import("../../llama/config.zig");
pub const Spec = common.Spec;
pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Spec {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    const s = try common.parse(gpa, bytes);
    _ = try dense(s);
    return s;
}
/// The existing 128-wide GQA/SwiGLU forward, with Qwen3's separate Q/K norms.
pub fn dense(s: Spec) !llama.Config {
    if (s.head_dim != 128 or s.hidden % 256 != 0 or s.inter % 256 != 0 or
        s.layers > llama.max_layers or s.heads / s.kv_heads > llama.max_group) return error.UnsupportedQwen3MetalGeometry;
    return .{ .hidden = s.hidden, .intermediate = s.inter, .layers = s.layers, .heads = s.heads, .kv_heads = s.kv_heads, .vocab = s.vocab, .eps = @floatCast(s.eps), .theta = s.theta, .context = s.max_positions };
}
test "Qwen3 embedding sizes include query widths different from hidden" {
    const sizes = [_][6]usize{ .{ 1024, 3072, 28, 16, 151669, 32768 }, .{ 2560, 9728, 36, 32, 151665, 40960 }, .{ 4096, 12288, 36, 32, 151665, 40960 } };
    for (sizes) |v| {
        var s: Spec = .{ .hidden = v[0], .inter = v[1], .layers = v[2], .heads = v[3], .kv_heads = 8, .head_dim = 128, .vocab = v[4], .eps = 1e-6, .theta = 1e6, .max_positions = v[5], .tied = true };
        const c = try dense(s);
        try std.testing.expectEqual(v[3] * 128, c.queryWidth());
        s.head_dim = 64;
        try std.testing.expectError(error.UnsupportedQwen3MetalGeometry, dense(s));
    }
}
