//! Nemotron's paged K/V twins (kernels/metal/nemotron/paged.zig) as the 3b encoders will launch them, main's shapes.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "nemotron paged write nemotron_kv_write_paged", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "nemotron paged compact nemotron_kv_compact_paged", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "nemotron paged put nemotron_kv_put_paged", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
} ++ partials ++ [_]M{
    .{ .name = "nemotron paged tail nemotron_tree_tail_paged", .threads = 32, .tg_memory = 18432, .offline = false },
    .{ .name = "nemotron paged prompt nemotron_attention_paged_n_n", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "nemotron paged prompt nemotron_attention_paged_n_t", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "nemotron paged prompt nemotron_attention_paged_t_n", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "nemotron paged prompt nemotron_attention_paged_t_t", .threads = 128, .tg_memory = 0, .reserved = 128 },
};

/// The decode partials at 1 to 16 tiles (32 threads and 2 KiB of scores a tile); past 8 they need 288 to 512 threads.
const partials = blk: {
    var out: [16]M = undefined;
    for (&out, 1..) |*r, n| r.* = .{ .name = std.fmt.comptimePrint("nemotron paged partial nemotron_attn_partial_paged_{d}", .{n}), .threads = 32 * n, .tg_memory = 2048 * n, .offline = false, .unbuildable = n > 8 };
    break :blk out;
};

test "the paged Nemotron launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
