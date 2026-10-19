const std = @import("std");
const kv = @import("kv_state.zig");
const t = std.testing;
const Fixture = struct {
    live: [32]u8 = @splat(0),
    backup: [32]u8 = undefined,
    state: kv.State = .{},
    fn regions(f: *Fixture) [2]kv.Region {
        return .{
            .{ .live = @intFromPtr(&f.live), .backup = @intFromPtr(&f.backup), .row_bytes = 2, .rows = 8, .kind = .ring },
            .{ .live = @intFromPtr(&f.live[16]), .backup = @intFromPtr(&f.backup[16]), .row_bytes = 4, .rows = 1, .kind = .carry },
        };
    }
    fn copy(_: *anyopaque, dst: u64, src: u64, bytes: usize) !void {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..bytes], @as([*]const u8, @ptrFromInt(src))[0..bytes]);
    }
    fn copier(f: *Fixture) kv.Copy {
        return .{ .ctx = f, .run = copy };
    }
    // A deterministic recurrent KV writer. Position and the previous carry affect every written byte.
    fn step(f: *Fixture, pos: u64, token: u8) void {
        const at: usize = @intCast(pos % 8 * 2);
        const carry = f.live[16];
        f.live[at] = token +% carry;
        f.live[at + 1] = @truncate(pos);
        f.live[16] = carry *% 3 +% token;
    }
    fn append(f: *Fixture, token: u8) void {
        f.step(f.state.position, token);
        f.state.position += 1;
    }
    fn window(f: *Fixture, ids: []const u8) ![4]u8 {
        try f.state.begin(&f.regions(), f.copier(), @intCast(ids.len));
        var carries: [4]u8 = undefined;
        for (ids, 0..) |id, i| {
            f.step(f.state.position + i, id);
            carries[i] = f.live[16];
        }
        return carries;
    }
};
test "Snapshot state: partial acceptance restores rejected ring rows, including wrap, at every accepted depth" {
    for (0..5) |accepted| {
        var f: Fixture = .{};
        for (0..15) |i| f.append(@intCast(i + 1));
        var fresh = f;
        const ids = [_]u8{ 41, 42, 43, 44 };
        const carries = try f.window(&ids);
        if (accepted > 0) f.live[16] = carries[accepted - 1];
        try f.state.keep(&f.regions(), f.copier(), @intCast(accepted));
        for (ids[0..accepted]) |id| fresh.append(id);
        try t.expectEqual(fresh.state.position, f.state.position);
        try t.expectEqualSlices(u8, &fresh.live, &f.live);
        f.append(77);
        fresh.append(77);
        try t.expectEqualSlices(u8, &fresh.live, &f.live);
    }
}
test "Snapshot state: resume at odd compression and window seams equals fresh KV, and dropping verification restores carry" {
    for ([_]usize{ 1, 7, 8, 9, 15, 16, 17 }) |seam| {
        var f: Fixture = .{};
        for (0..seam) |i| f.append(@intCast(i + 1));
        const snapshot = f;
        _ = try f.window(&.{ 90, 91, 92 });
        try f.state.rollback(&f.regions(), f.copier());
        try t.expectEqualSlices(u8, &snapshot.live, &f.live);
        // Session payload owns every mutable byte, and the logical position. No process addresses are serialized.
        var restored: Fixture = .{};
        restored.live = snapshot.live;
        restored.state.position = snapshot.state.position;
        for (seam..40) |i| restored.append(@intCast(i + 1));
        var fresh: Fixture = .{};
        for (0..40) |i| fresh.append(@intCast(i + 1));
        try t.expectEqualSlices(u8, &fresh.live, &restored.live);
    }
}
const FailingCopy = struct {
    fail: bool = true,
    fn copy(ptr: *anyopaque, dst: u64, src: u64, bytes: usize) !void {
        const c: *FailingCopy = @ptrCast(@alignCast(ptr));
        if (c.fail) return error.CopyFailed;
        try Fixture.copy(ptr, dst, src, bytes);
    }
};
test "Snapshot state: a failed rollback retains the transaction and retries before cursor publication" {
    var f: Fixture = .{};
    for (0..9) |i| f.append(@intCast(i + 1));
    const before = f.live;
    _ = try f.window(&.{ 81, 82 });
    var fc: FailingCopy = .{};
    const copier: kv.Copy = .{ .ctx = &fc, .run = FailingCopy.copy };
    try t.expectError(error.CopyFailed, f.state.rollback(&f.regions(), copier));
    try t.expectEqual(@as(?u32, 2), f.state.pending);
    try t.expectEqual(@as(u64, 9), f.state.position);
    fc.fail = false;
    try f.state.rollback(&f.regions(), copier);
    try t.expectEqualSlices(u8, &before, &f.live);
    try t.expectError(error.NoCacheTransaction, f.state.keep(&f.regions(), copier, 1));
}
