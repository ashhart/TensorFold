const std = @import("std");
const gpu = @import("quant_gpu.zig");

test "specialized source selects zero, 64, and 128 group sums" {
    const a = std.testing.allocator;
    for ([_]struct { xs: bool, xg: u16, want: []const u8 }{
        .{ .xs = false, .xg = 64, .want = "#define Q27_XS 0\n" },
        .{ .xs = true, .xg = 64, .want = "#define Q27_XS 64\n" },
        .{ .xs = true, .xg = 128, .want = "#define Q27_XS 128\n" },
    }) |case| {
        for ([_]gpu.Kind{ .norm, .input_norm, .mlp }) |kind| {
            const src = try gpu.specializedSource(a, .{ .kind = kind, .k = 4096, .n = 12288, .xs = case.xs, .xs_group = case.xg });
            defer a.free(src);
            try std.testing.expect(std.mem.startsWith(u8, src, case.want));
        }
    }
}
