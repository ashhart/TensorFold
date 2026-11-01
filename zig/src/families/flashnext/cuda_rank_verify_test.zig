//! The shared verify frame without ranks: round trips, every refusal, and two-phase window creation.
const std = @import("std");
const v = @import("cuda_rank_verify.zig");
const mtp = @import("cuda_mtp.zig");
const wt = @import("cuda_prompt_wt.zig");
const t = std.testing;

test "shared verify frame: one and 32 windows round trip in window order" {
    var raw: [v.max_bytes]u8 = undefined;
    const one = try v.encode(&.{7}, &.{&.{ 1, 2, 3 }}, 64, &raw);
    const f1 = try v.decode(one, 1, 64);
    var row: [mtp.max_rows]u32 = undefined;
    try t.expectEqual(@as(u64, 7), f1.ids[0]);
    try t.expectEqualSlices(u32, &.{ 1, 2, 3 }, f1.window(0, &row));
    var ids: [v.max_windows]u64 = undefined;
    var rows: [v.max_windows][2]u32 = undefined;
    var idss: [v.max_windows][]const u32 = undefined;
    for (0..v.max_windows) |w| {
        ids[w] = 100 + w;
        rows[w] = .{ @intCast(w), @intCast(wt.vocab - 1 - w) };
        idss[w] = &rows[w];
    }
    const all = try v.encode(&ids, &idss, 64, &raw);
    try t.expectEqual(@as(usize, 12 * 32 + 4 * 64), all.len);
    const f = try v.decode(all, v.max_windows, 64);
    for (0..v.max_windows) |w| {
        try t.expectEqual(ids[w], f.ids[w]);
        try t.expectEqualSlices(u32, &rows[w], f.window(w, &row));
    }
}

test "shared verify frame: refused before any state moves" {
    var raw: [v.max_bytes]u8 = undefined;
    const good = try v.encode(&.{ 1, 2 }, &.{ &.{5}, &.{ 6, 7 } }, 64, &raw);
    try t.expectError(error.BadRankFrame, v.decode(good, 0, 64));
    try t.expectError(error.BadRankFrame, v.decode(good, v.max_windows + 1, 64));
    try t.expectError(error.BadRankFrame, v.decode(good[0 .. good.len - 4], 2, 64));
    var long: [v.max_bytes]u8 = undefined;
    @memcpy(long[0..good.len], good);
    try t.expectError(error.BadRankFrame, v.decode(long[0 .. good.len + 4], 2, 64));
    // the total row bound: three windows of max_rows past a 64-row pass
    var wide: [mtp.max_rows]u32 = @splat(1);
    try t.expectError(error.BadRankFrame, v.encode(&.{ 1, 2, 3 }, &.{ &wide, &wide, &wide }, 64, &raw));
    try t.expectError(error.BadRankFrame, v.encode(&.{1}, &.{&.{}}, 64, &raw));
    var too_wide: [mtp.max_rows + 1]u32 = @splat(1);
    try t.expectError(error.BadRankFrame, v.encode(&.{1}, &.{&too_wide}, 1024, &raw));
    try t.expectError(error.DuplicateRankSequence, v.encode(&.{ 4, 4 }, &.{ &.{1}, &.{1} }, 64, &raw));
    try t.expectError(error.InvalidToken, v.encode(&.{1}, &.{&.{wt.vocab}}, 64, &raw));
    try t.expectError(error.BadRankFrame, v.encode(&.{0}, &.{&.{1}}, 64, &raw));
}

test "shared verify frame: ensure makes every missing window before any pointer is taken" {
    var map: std.AutoHashMapUnmanaged(u64, [64]u8) = .empty;
    defer map.deinit(t.allocator);
    var keys: [40]u64 = undefined;
    for (&keys, 0..) |*k, i| k.* = 1000 + i;
    const Make = struct {
        fn make(fail_at: usize, i: usize) ![64]u8 {
            if (i == fail_at) return error.OutOfDeviceMemory;
            return @splat(@intCast(i));
        }
    };
    try v.ensure(&map, t.allocator, &keys, @as(usize, std.math.maxInt(usize)), Make.make);
    var seen: [40]*[64]u8 = undefined;
    for (keys, 0..) |k, i| {
        seen[i] = map.getPtr(k).?;
        try t.expectEqual(@as(u8, @intCast(i)), seen[i][0]);
        for (seen[0..i]) |p| try t.expect(p != seen[i]);
    }
    // a failed make leaves its key absent and the four made before it in place
    var fresh: std.AutoHashMapUnmanaged(u64, [64]u8) = .empty;
    defer fresh.deinit(t.allocator);
    try t.expectError(error.OutOfDeviceMemory, v.ensure(&fresh, t.allocator, keys[0..8], @as(usize, 4), Make.make));
    try t.expectEqual(@as(u32, 4), fresh.count());
    try t.expect(fresh.get(keys[4]) == null);
    for (keys[0..4], 0..) |k, i| try t.expectEqual(@as(u8, @intCast(i)), fresh.get(k).?[0]);
}
