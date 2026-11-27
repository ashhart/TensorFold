//! Host unit tests and macOS GPU conformance tests for Inkling 2 Mini tree verify.
const std = @import("std");
const builtin = @import("builtin");
const tp = @import("inkling2_tree_plan");
const ap = @import("attention_plan.zig");

test "validateParents and isChain enforce DAG and detect branches" {
    var depths: [tp.window_rows]u32 = undefined;
    const chain = [_]i32{ -1, 0, 1, 2 };
    try tp.validateParents(&chain, depths[0..chain.len]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, depths[0..chain.len]);
    try std.testing.expect(tp.isChain(&chain));

    const branch = [_]i32{ -1, 0, 0, 1, 2, 4 };
    try tp.validateParents(&branch, depths[0..branch.len]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2, 2, 3 }, depths[0..branch.len]);
    try std.testing.expect(!tp.isChain(&branch));

    try std.testing.expectError(error.BadParents, tp.validateParents(&.{}, depths[0..0]));
    try std.testing.expectError(error.BadParents, tp.validateParents(&.{0}, depths[0..1]));
    try std.testing.expectError(error.BadParents, tp.validateParents(&.{ -1, 1 }, depths[0..2]));
    try std.testing.expectError(error.BadParents, tp.validateParents(&.{ -1, -1 }, depths[0..2]));
}

test "validateKeepPath accepts valid tree paths and rejects cross-branch jumps" {
    const branch = [_]i32{ -1, 0, 0, 1, 2, 4 };
    try tp.validateKeepPath(6, &branch, &.{0});
    try tp.validateKeepPath(6, &branch, &.{ 0, 1, 3 });
    try tp.validateKeepPath(6, &branch, &.{ 0, 2, 4, 5 });
    try tp.validateKeepPath(6, null, &.{ 0, 1, 2 });

    try std.testing.expectError(error.UnsupportedInklingWindow, tp.validateKeepPath(6, &branch, &.{}));
    try std.testing.expectError(error.UnsupportedInklingWindow, tp.validateKeepPath(6, &branch, &.{1}));
    try std.testing.expectError(error.UnsupportedInklingWindow, tp.validateKeepPath(6, &branch, &.{ 0, 2, 3 }));
    try std.testing.expectError(error.UnsupportedInklingWindow, tp.validateKeepPath(6, &branch, &.{ 0, 2, 4, 5, 5 }));
    try std.testing.expectError(error.UnsupportedInklingWindow, tp.validateKeepPath(6, null, &.{ 0, 2 }));
}

test "buildConvSrc, buildTreePaths, and convHistorySrc trace exact ancestor taps" {
    const branch = [_]i32{ -1, 0, 0, 1, 2, 4 };
    var depths: [6]u32 = undefined;
    try tp.validateParents(&branch, &depths);

    var conv_src: [6 * 4]u32 = undefined;
    tp.buildConvSrc(&branch, &conv_src);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, conv_src[0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, conv_src[4..8]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5 }, conv_src[8..12]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4, 6 }, conv_src[12..16]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 5, 7 }, conv_src[16..20]);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5, 7, 8 }, conv_src[20..24]);

    var paths: [6 * tp.window_rows]u32 = undefined;
    tp.buildTreePaths(&branch, &depths, &paths);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, paths[0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 0, 0 }, paths[2 * tp.window_rows ..][0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4, 5 }, paths[5 * tp.window_rows ..][0..4]);

    try std.testing.expectEqual([3]u32{ 1, 2, 3 }, tp.convHistorySrc(&.{0}));
    try std.testing.expectEqual([3]u32{ 2, 3, 5 }, tp.convHistorySrc(&.{ 0, 2 }));
    try std.testing.expectEqual([3]u32{ 3, 5, 7 }, tp.convHistorySrc(&.{ 0, 2, 4 }));
    try std.testing.expectEqual([3]u32{ 5, 7, 8 }, tp.convHistorySrc(&.{ 0, 2, 4, 5 }));
}

test "attention_plan.attendTree computes exact prefix/tail grids across tile and chunk boundaries" {
    const g = ap.Layer.global(32768);
    const l0 = try ap.attendTree(.{ .rows = 6, .last_pos = 3 }, g, 4, 0);
    try std.testing.expectEqual(@as(u32, 0), l0.p_tile);
    try std.testing.expectEqual(@as(u32, 0), l0.c_off);
    try std.testing.expectEqual([3]u32{ 1, 8, 0 }, l0.prefix_grid);
    try std.testing.expectEqual([3]u32{ 6, 8, 1 }, l0.tail_grid);

    const l64 = try ap.attendTree(.{ .rows = 6, .last_pos = 67 }, g, 4, 64);
    try std.testing.expectEqual(@as(u32, 64), l64.p_tile);
    try std.testing.expectEqual(@as(u32, 0), l64.c_off);
    try std.testing.expectEqual([3]u32{ 1, 8, 1 }, l64.prefix_grid);
    try std.testing.expectEqual([3]u32{ 6, 8, 1 }, l64.tail_grid);

    const lcross = try ap.attendTree(.{ .rows = 6, .last_pos = 16385 }, g, 4, 16382);
    try std.testing.expectEqual(@as(u32, 16320), lcross.p_tile);
    try std.testing.expectEqual(@as(u32, 0), lcross.c_off);
    try std.testing.expectEqual([3]u32{ 1, 8, 1 }, lcross.prefix_grid);
    try std.testing.expectEqual([3]u32{ 6, 8, 2 }, lcross.tail_grid);

    const lnext = try ap.attendTree(.{ .rows = 6, .last_pos = 16390 }, g, 4, 16384);
    try std.testing.expectEqual(@as(u32, 16384), lnext.p_tile);
    try std.testing.expectEqual(@as(u32, 1), lnext.c_off);
    try std.testing.expectEqual([3]u32{ 1, 8, 1 }, lnext.prefix_grid);
    try std.testing.expectEqual([3]u32{ 6, 8, 1 }, lnext.tail_grid);
}

test {
    if (builtin.os.tag.isDarwin()) _ = @import("tree_gpu_test.zig");
}
