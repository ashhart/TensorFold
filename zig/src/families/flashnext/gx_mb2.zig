//! TF_FN_GX_GATEUP_MB2: the grouped gate/up's source and its pairing of list slots, the rule fn_gx_mb2.metal applies.
const std = @import("std");
const ks = @import("kernel_sources");

pub const function: [:0]const u8 = "gx_gateup_mb2";
const indexed_define = "#define GX_MB2_INDEXED 1\n";

/// `base` (fn_gx.metal as the loader patched it) with the MB=2 kernel after it, reading through the index if `indexed`.
pub fn text(a: std.mem.Allocator, base: []const u8, indexed: bool) ![]const u8 {
    return std.mem.concat(a, u8, &.{ base, "\n", if (indexed) indexed_define else "", ks.flashnext_gx_mb2 });
}

pub const Launch = enum { skip, one, two };

/// What slot `y` of `keys` (each slot's expert, -1 for a shared row) launches.
pub fn launch(keys: []const i32, y: usize) Launch {
    if (y % 2 == 1 and keys[y - 1] == keys[y]) return .skip;
    return if (y % 2 == 0 and y + 1 < keys.len and keys[y + 1] == keys[y]) .two else .one;
}

fn expectCovered(keys: []const i32) !void {
    var runs: [64]u8 = @splat(0);
    for (keys, 0..) |_, y| switch (launch(keys, y)) {
        .skip => {},
        .one => runs[y] += 1,
        .two => {
            try std.testing.expectEqual(keys[y], keys[y + 1]);
            runs[y] += 1;
            runs[y + 1] += 1;
        },
    };
    for (runs[0..keys.len]) |count| try std.testing.expectEqual(@as(u8, 1), count);
}

test "MB=2 launches every list slot exactly once, pairing one expert's slots only" {
    try expectCovered(&.{ 3, 3, 3, 3 });
    try expectCovered(&.{ 3, 3, 3 });
    try expectCovered(&.{ 1, 2, 2, 3, 3, 3, -1, -1, -1 });
    try expectCovered(&.{ 7, -1, -1, -1, -1 });
    var random: std.Random.DefaultPrng = .init(1486);
    var keys: [64]i32 = undefined;
    for (0..2000) |_| {
        const n = random.random().intRangeAtMost(usize, 1, keys.len);
        for (keys[0..n]) |*k| k.* = random.random().intRangeAtMost(i32, -1, 3);
        try expectCovered(keys[0..n]);
    }
}

test "a lone trailing slot takes the one-pair path" {
    try std.testing.expectEqual(Launch.two, launch(&.{ 5, 5, 5 }, 0));
    try std.testing.expectEqual(Launch.skip, launch(&.{ 5, 5, 5 }, 1));
    try std.testing.expectEqual(Launch.one, launch(&.{ 5, 5, 5 }, 2));
}

test "the default source is untouched and the MB=2 source holds its kernel once" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, ks.flashnext_gx, function));
    for ([_]bool{ false, true }) |indexed| {
        const out = try text(arena.allocator(), ks.flashnext_gx, indexed);
        try std.testing.expect(std.mem.startsWith(u8, out, ks.flashnext_gx));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "void gx_gateup_mb2("));
        try std.testing.expectEqual(indexed, std.mem.indexOf(u8, out, indexed_define) != null);
    }
}
