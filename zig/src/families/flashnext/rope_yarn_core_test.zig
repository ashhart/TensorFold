//! core.rope_yarn (the Qwen CUDA path's table) against this family's own, which its receipts pin by bytes.
const std = @import("std");
const core = @import("core");
const rotary = @import("rotary_config.zig");

test "core's YaRN table equals Flash Next's at Qwen's config and Flash Next's own" {
    for ([_]f64{ 1e6, 1e7 }) |base| for ([_]f64{ 2, 4, 8 }) |factor| for ([_]usize{ 32768, 262144 }) |trained| {
        const a = core.rope_yarn.yarn(base, factor, trained);
        const b = rotary.yarn(base, factor, trained);
        try std.testing.expectEqualSlices(u32, &@as([32]u32, @bitCast(b.freq)), &@as([32]u32, @bitCast(a.freq)));
        try std.testing.expectEqual(b.scale, a.scale);
    };
}
