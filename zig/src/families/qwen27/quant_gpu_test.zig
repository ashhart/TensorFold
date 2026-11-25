//! quant_gpu's pure helpers: the long-prompt choice and the specialized sources' group sums.
const std = @import("std");
const q = @import("quant_gpu.zig");

test "long prompt dispatches only on an Ultra with tensor units" {
    try std.testing.expect(q.longPrompt("Apple M5 Ultra", true));
    try std.testing.expect(!q.longPrompt("Apple M5 Max", true));
    try std.testing.expect(!q.longPrompt("Apple M5 Pro", true));
    try std.testing.expect(!q.longPrompt("Apple M3 Ultra", false));
}

test "only prompt chunks past 128 rows take the long key" {
    try std.testing.expect(q.longKey(true, true, 129));
    try std.testing.expect(q.longKey(true, true, 2048));
    try std.testing.expect(!q.longKey(true, true, 128));
    try std.testing.expect(!q.longKey(true, false, 2048));
    try std.testing.expect(!q.longKey(false, true, 2048));
}

test "specialized source selects zero, 64, and 128 group sums" {
    const a = std.testing.allocator;
    for ([_]struct { xs: bool, xg: u16, want: []const u8 }{
        .{ .xs = false, .xg = 64, .want = "#define Q27_XS 0\n" },
        .{ .xs = true, .xg = 64, .want = "#define Q27_XS 64\n" },
        .{ .xs = true, .xg = 128, .want = "#define Q27_XS 128\n" },
    }) |case| {
        for ([_]q.Kind{ .norm, .input_norm, .mlp }) |kind| {
            const src = try q.specializedSource(a, .{ .kind = kind, .k = 4096, .n = 12288, .xs = case.xs, .xs_group = case.xg });
            defer a.free(src);
            try std.testing.expect(std.mem.startsWith(u8, src, case.want));
        }
    }
}

test "the steered norm keys on main's group sums and refuses SenseNova's 128-column ones" {
    const a = std.testing.allocator;
    for ([_]struct { xg: u16, want: []const u8 }{
        .{ .xg = 0, .want = "#define Q27_XS 0\n" },
        .{ .xg = 64, .want = "#define Q27_XS 64\n" },
    }) |case| {
        const key = try q.steerKey(5120, case.xg);
        try std.testing.expectEqual(case.xg != 0, key.xs);
        try std.testing.expectEqual(case.xg, key.xs_group);
        const src = try q.specializedSource(a, key);
        defer a.free(src);
        try std.testing.expect(std.mem.startsWith(u8, src, case.want));
    }
    try std.testing.expectError(error.SteerXsGroup, q.steerKey(5120, 128));
}
