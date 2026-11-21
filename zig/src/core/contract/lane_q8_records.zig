//! Byte-lane launch envelopes; the standalone sums compiled for M1/M2 even when tensor ops cannot.
const M = @import("limits.zig").MetalLaunch;
pub const Bad = enum { none, group, stride };
pub const Source = struct { name: []const u8, n: usize, k: usize, sk: usize, group: usize = 64, sums: bool = false, pf: usize = 1, f32_out: bool = false, bad: Bad = .none };
pub const sources = [_]Source{
    .{ .name = "qwen27 q8 exact fp32 sk1 tf_lane", .n = 512, .k = 512, .sk = 1, .sums = true, .f32_out = true },
    .{ .name = "qwen27 q8 exact fp32 sk8 tf_lane", .n = 512, .k = 512, .sk = 8, .sums = true, .f32_out = true },
    .{ .name = "qwen27 q8 direct head tf_lane", .n = 248320, .k = 5120, .sk = 1 },
    .{ .name = "qwen27 q8 direct mlp tf_lane", .n = 34816, .k = 5120, .sk = 2 },
    .{ .name = "qwen27 q8 direct qkv tf_lane", .n = 10240, .k = 5120, .sk = 4 },
    .{ .name = "qwen27 q8 direct down tf_lane", .n = 5120, .k = 17408, .sk = 8 },
    .{ .name = "qwen27 q8 prepared head tf_lane", .n = 248320, .k = 5120, .sk = 1, .sums = true },
    .{ .name = "qwen27 q8 prepared mlp tf_lane", .n = 34816, .k = 5120, .sk = 2, .sums = true },
    .{ .name = "qwen27 q8 prepared qkv tf_lane", .n = 10240, .k = 5120, .sk = 4, .sums = true },
    .{ .name = "qwen27 q8 prepared z tf_lane", .n = 12288, .k = 5120, .sk = 4, .sums = true },
    .{ .name = "qwen27 q8 prepared out tf_lane", .n = 5120, .k = 6144, .sk = 8, .sums = true },
    .{ .name = "qwen27 q8 prepared down tf_lane", .n = 5120, .k = 17408, .sk = 8, .sums = true },
    .{ .name = "qwen27 q8 prepared ab tf_lane", .n = 2048, .k = 5120, .sk = 8, .sums = true },
    .{ .name = "qwen27 q8 prepared gdn tf_lane", .n = 6240, .k = 5120, .sk = 8, .sums = true },
    .{ .name = "qwen27 q8 exact bad-group tf_lane", .n = 10240, .k = 5120, .sk = 4, .sums = true, .bad = .group },
    .{ .name = "qwen27 q8 exact bad-stride tf_lane", .n = 10240, .k = 5120, .sk = 4, .sums = true, .bad = .stride },
    .{ .name = "qwen27 q8 exact g32 tf_lane", .n = 512, .k = 512, .sk = 1, .group = 32 },
    .{ .name = "qwen27 q8 exact g128 tf_lane", .n = 512, .k = 512, .sk = 4, .group = 128 },
    .{ .name = "qwen27 q8 exact pf2 tf_lane", .n = 512, .k = 512, .sk = 2, .sums = true, .pf = 2 },
    .{ .name = "qwen27 q8 exact fallback tf_lane", .n = 512, .k = 512, .sk = 2, .sums = true },
    .{ .name = "qwen27 q8 exact fallback tf_lane_sums", .n = 512, .k = 512, .sk = 2, .sums = true },
    .{ .name = "qwen27 q8 prepared norm tf_lane_sums", .n = 10240, .k = 5120, .sk = 4, .sums = true },
    .{ .name = "qwen27 q8 prepared out tf_lane_sums", .n = 5120, .k = 6144, .sk = 8, .sums = true },
    .{ .name = "qwen27 q8 prepared down tf_lane_sums", .n = 5120, .k = 17408, .sk = 8, .sums = true },
};
pub const records = blk: {
    var out: [sources.len]M = undefined;
    for (sources, 0..) |s, i| {
        const sum = @import("std").mem.endsWith(u8, s.name, "tf_lane_sums");
        const groups: usize = @max(@as(usize, s.sk) -| 1, @as(usize, 1));
        const threads: usize = if (sum) 256 else 32 * s.sk;
        const memory: usize = if (sum) 0 else groups * 16 * 32 * 4;
        out[i] = .{ .name = s.name, .threads = @intCast(threads), .tg_memory = @intCast(memory), .m1 = if (sum) 1024 else 0, .m2 = if (sum) 1024 else 0, .offline = sum };
    }
    break :blk out;
};

test "byte lane records: standalone sums have measured offline limits; tensor ops stay unmeasured" {
    const std = @import("std");
    var count: usize = 0;
    for (records) |r| {
        const sum = std.mem.endsWith(u8, r.name, "tf_lane_sums");
        try std.testing.expectEqual(sum, r.offline);
        try std.testing.expectEqual(@as(u32, if (sum) 1024 else 0), r.m1);
        try std.testing.expectEqual(r.m1, r.m2);
        if (sum) {
            count += 1;
            try std.testing.expectEqual(@as(u32, 256), r.threads);
            try std.testing.expectEqual(@as(u32, 0), r.tg_memory);
        }
    }
    try std.testing.expectEqual(@as(usize, 4), count);
}
