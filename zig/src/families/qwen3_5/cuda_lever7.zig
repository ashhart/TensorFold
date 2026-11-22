//! Default-off REV295 CUDA switches and the live group-sums pointer selection.
const std = @import("std");

pub const Options = struct {
    skip_xs: bool = false,
    active_tables: bool = false,
    capture: bool = false,

    pub fn load() !Options {
        const environment = @import("sys").environment;
        return .{
            .skip_xs = try environment.read(bool, "TF_Q35_SKIP_XS", false),
            .active_tables = try environment.read(bool, "TF_Q35_ACTIVE_TABLES", false),
            .capture = try environment.read(bool, "TF_Q35_CAPTURE", false),
        };
    }
};

/// ModelOpt owns group sums only after its MoE gate. Prompt rows retain the ordinary gate.
pub fn groupSums(e: anytype, prompt: bool, total: usize) u64 {
    const skip = (prompt and total > 16) or (e.lever7.skip_xs and e.mo != null and e.learning == null);
    return if (skip) 0 else e.xs;
}

/// Uniform draw rows have one fixed launch topology; mixed sampling stays eager.
pub fn uniformDraws(pk: anytype) bool {
    if (pk.len == 0) return false;
    const sampled = pk[0].rule != null;
    for (pk[1..]) |p| if ((p.rule != null) != sampled) return false;
    return true;
}

fn fakeGlue(xs: u64, xs_bytes: []u8, output: []u8) !void {
    if (xs != 0) @memset(xs_bytes, 0x11);
    @memset(output, 0x42);
}

test "REV295 skip-xs leaves poisoned group sums unread and preserves dense rows" {
    const Fake = struct {
        lever7: Options,
        mo: ?u8,
        learning: ?u8,
        xs: u64,
    };
    for ([_]bool{ false, true }) |skip| {
        var xs_bytes: [32]u8 = @splat(0xa5);
        var output: [32]u8 = @splat(0xa5);
        var e: Fake = .{ .lever7 = .{ .skip_xs = skip }, .mo = 1, .learning = null, .xs = 7 };
        try fakeGlue(groupSums(&e, false, 8), &xs_bytes, &output);
        if (skip) {
            try std.testing.expectEqualSlices(u8, &@as([32]u8, @splat(0xa5)), &xs_bytes);
        } else {
            try std.testing.expectEqualSlices(u8, &@as([32]u8, @splat(0x11)), &xs_bytes);
        }
        try std.testing.expectEqualSlices(u8, &@as([32]u8, @splat(0x42)), &output);
    }
}

test "REV295 prompt cutoff, learning, and ModelOpt absence retain group sums" {
    const Fake = struct { lever7: Options, mo: ?u8, learning: ?u8, xs: u64 };
    var e: Fake = .{ .lever7 = .{ .skip_xs = true }, .mo = 1, .learning = null, .xs = 9 };
    try std.testing.expectEqual(@as(u64, 0), groupSums(&e, true, 17));
    for ([_]usize{ 1, 8, 16, 17 }) |rows| try std.testing.expectEqual(@as(u64, 0), groupSums(&e, false, rows));
    e.learning = 1;
    try std.testing.expectEqual(@as(u64, 9), groupSums(&e, false, 17));
    e.learning = null;
    e.mo = null;
    try std.testing.expectEqual(@as(u64, 9), groupSums(&e, false, 17));
}
