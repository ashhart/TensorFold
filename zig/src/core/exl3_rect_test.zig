const std = @import("std");
const exl3 = @import("exl3_format.zig");

test "rectangular trellis places every tile at its output-column stride" {
    const gpa = std.testing.allocator;
    const width: exl3.Width = .{ .halves = 8 };
    for ([_][2]usize{ .{ 8, 16 }, .{ 16, 8 }, .{ 8, 24 } }) |shape| {
        const k = shape[0] * 16;
        const n = shape[1] * 16;
        const words = width.tileWords();
        const trellis = try gpa.alloc(i16, shape[0] * shape[1] * words);
        defer gpa.free(trellis);
        for (trellis, 0..) |*v, i| v.* = @bitCast(@as(u16, @truncate(i *% 173 +% 37)));
        const out = try gpa.alloc(f16, k * n);
        defer gpa.free(out);
        @memset(out, std.math.nan(f16));
        exl3.unpack(width, .mcg, trellis, shape[0], shape[1], out);
        var placement: [256]exl3.Placement = undefined;
        exl3.tilePlacement(&placement);
        for (0..shape[0]) |tk| for (0..shape[1]) |tn| {
            var states: [256]u32 = undefined;
            exl3.tileStates(width, trellis[(tk * shape[1] + tn) * words ..][0..words], &states);
            for (states, placement) |state, at| {
                const want: u16 = @bitCast(exl3.codebookValue(.mcg, @intCast(state)));
                try std.testing.expectEqual(want, @as(u16, @bitCast(out[(tk * 16 + at.row) * n + tn * 16 + at.col])));
            }
        };
    }
}
