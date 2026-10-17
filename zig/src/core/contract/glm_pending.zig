//! GLM matmul launches, pending compiler resource measurements on every supported SM.
const std = @import("std");
const R = @import("registry.zig").CudaRecord;
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });

pub const mm: [22]R = blk: {
    var out: [22]R = undefined;
    var at: usize = 0;
    for ([_]u8{ 16, 4, 6, 8 }) |bits| {
        for ([_]usize{ 16, 32, 64, 128 }) |bm| {
            const words: usize = if (bits == 16) 8 else bits / 2;
            const shared = 3 * (bm * 128 + 8 * 32 * words * 4 + 128);
            const symbol = std.fmt.comptimePrint("glm_mm{d}_{d}", .{ bits, bm });
            out[at] = .{ .launch = .{ .name = symbol, .symbol = symbol, .image = "glm_mm", .threads = if (bm <= 32) 128 else 256, .dynamic = shared, .opt_in = true, .min_sm = 80 }, .usage = none };
            at += 1;
        }
        const tile = std.fmt.comptimePrint("glm_mm_tile{d}", .{bits});
        out[at] = .{ .launch = .{ .name = tile, .symbol = tile, .image = "glm_mm", .threads = 256, .min_sm = 80 }, .usage = none };
        at += 1;
    }
    out[at] = .{ .launch = .{ .name = "glm_mm16_rows", .symbol = "glm_mm16_rows", .image = "glm_mm", .threads = 256, .dynamic = 74112, .opt_in = true, .min_sm = 80 }, .usage = none };
    out[at + 1] = .{ .launch = .{ .name = "glm_mm_reduce", .symbol = "glm_mm_reduce", .image = "glm_mm", .threads = 256, .min_sm = 80 }, .usage = none };
    break :blk out;
};

test "GLM pending records cannot pass the measured resource gate" {
    const registry = @import("registry.zig");
    for (mm) |r| {
        try std.testing.expectEqual(none, r.usage);
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
