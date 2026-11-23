//! Ling MLA extras (tf_ling_mla_rope_cache, tf_ling_mla_rope_q, tf_ling_mla_gate_expand).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal ling mla tf_ling_mla_rope_cache", .threads = 32, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal ling mla tf_ling_mla_rope_q", .threads = 32, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal ling mla tf_ling_mla_gate_expand", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "the Ling MLA launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
