//! Hy4-flash's tensor-unit lane projections at each served width: reg kernel, inline-sums twin and sums pass.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

/// [n, k, K splits]: the dense projections, then the q/kv inputs and the queries at 64 and 32 heads.
pub const shapes = [_][3]u32{
    .{ 3072, 16384, 8 }, .{ 3072, 8192, 8 },  .{ 1536, 3072, 4 },   .{ 3072, 1536, 2 },
    .{ 13824, 3072, 4 }, .{ 3072, 13824, 8 }, .{ 120832, 3072, 4 }, .{ 17984, 3072, 4 },
    .{ 9792, 3072, 4 },  .{ 16384, 1024, 2 }, .{ 8192, 1024, 2 },
};

/// Reg threadgroup bytes: later splits' 16 x 32 fp32 partials, plus inline sums' K splits x groups a split x 16 rows.
fn regBytes(k: u32, sk: u32, sums: bool) u32 {
    const groups = k / 64;
    return (sk - 1) * 2048 + if (sums) sk * ((groups + sk - 1) / sk) * 16 * 4 else 0;
}

pub const records = blk: {
    @setEvalBranchQuota(20000);
    var out: [shapes.len * 2 * 3]M = undefined;
    var at = 0;
    for ([_]u32{ 4, 8 }) |bits| for (shapes) |s| {
        const head = std.fmt.comptimePrint("hy4 units lane q{d} n{d} k{d} sk{d}", .{ bits, s[0], s[1], s[2] });
        out[at] = .{ .name = head ++ " tf_lane_reg", .threads = 32 * s[2], .tg_memory = regBytes(s[1], s[2], false), .offline = false };
        out[at + 1] = .{ .name = head ++ " inline tf_lane_reg", .threads = 32 * s[2], .tg_memory = regBytes(s[1], s[2], true), .offline = false };
        out[at + 2] = .{ .name = head ++ " tf_lane_sums", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 };
        at += 3;
    };
    break :blk out;
};
