//! Actual GLM load, transport-cap and kernel-catalog boundaries, without loading a GPU.
const std = @import("std");
const glm = @import("glm");
const registry = @import("core").contract.registry;
const t = std.testing;

test "glm packed: invalid RoCE cap text and overflow are refused" {
    for ([_][]const u8{ "", "abc", "-1", "0", "4194304", "18446744073709551615" }) |v|
        try t.expectError(error.BadRoceCap, glm.native.roceBytes(v));
    try t.expectEqual(@as(usize, 4096 << 10), try glm.native.roceBytes(null));
    try t.expectEqual(@as(usize, 1024), try glm.native.roceBytes("1"));
    try t.expectEqual(@as(usize, 4194303) * 1024, try glm.native.roceBytes("4194303"));
}

fn record(symbol: []const u8) !registry.CudaRecord {
    for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) return r;
    return error.MissingGlmLaunch;
}

test "glm packed: every loaded matmul bucket and tiler has an unmeasured resource record" {
    inline for (.{ 16, 4, 6, 8 }) |bits| {
        inline for (.{ 16, 32, 64, 128 }) |bm| {
            const r = try record(std.fmt.comptimePrint("glm_mm{d}_{d}", .{ bits, bm }));
            try t.expectEqualStrings("glm_mm", r.launch.image);
            try t.expectEqual(@as(u32, if (bm <= 32) 128 else 256), r.launch.threads);
            try t.expectEqual(glm.mm.sharedBytes(bits, bm), r.launch.dynamic);
            for (r.usage) |u| try t.expectEqual(@as(u32, 0), u.regs);
        }
        _ = try record(std.fmt.comptimePrint("glm_mm_tile{d}", .{bits}));
    }
    _ = try record("glm_mm16_rows");
    _ = try record("glm_mm_reduce");
}
