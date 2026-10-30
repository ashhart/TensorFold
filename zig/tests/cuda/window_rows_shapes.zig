//! Every admitted wide GQA ratio, including partial and idle head groups, in both prompt modes and decode.
const std = @import("std");

pub fn cases(comptime Cfg: type) [96]Cfg {
    @setEvalBranchQuota(200000);
    var all: [96]Cfg = undefined;
    var at = 0;
    for ([_]usize{ 256, 512 }) |d| for (1..17) |ratio| for (0..3) |mode| {
        const h = 2 * ratio;
        const prompt = mode != 0;
        const linear = d == 512;
        all[at] = .{
            .name = std.fmt.comptimePrint("cuda wide branch dim{d} GQA{d} mode{d}, tail17 and wrapped/image masks", .{ d, ratio, mode }),
            .h = h,
            .hk = 2,
            .d = d,
            .win = if (linear) 0 else 1024,
            .ring = if (linear) (if (prompt) 0 else 3000) else 1536,
            .streams = 2,
            .prompt = prompt,
            .p32 = mode == 2,
            .image = prompt,
            .served = &.{.{ .rows = 17, .n = h, .k = 3000 }},
        };
        at += 1;
    };
    return all;
}
