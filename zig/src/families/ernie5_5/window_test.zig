//! window.zig against vLLM-ERNIE's merge_swa_topk, and its refusals.
const std = @import("std");
const win = @import("window.zig");

/// merge_swa_topk as the vendor's Triton kernel writes a column: its n from the position, not from picked.
fn vendor(topk: []const i32, k: usize, window: usize, seq: i64, col: usize) i32 {
    const n: i64 = @min(@max(seq - @as(i64, @intCast(window)), 0), @as(i64, @intCast(k)));
    const sw: i64 = @min(seq, @as(i64, @intCast(window)));
    const c: i64 = @intCast(col);
    if (c < n) return topk[col];
    return if (c < n + sw) @intCast(seq - sw + c - n) else -1;
}

fn run(a: std.mem.Allocator, sh: win.Shape, pos0: i64, picks: []const i32, picked: []const i32) !struct { []i32, []i32 } {
    const tok = try a.alloc(i32, sh.rows * sh.w);
    const cnt = try a.alloc(i32, sh.rows);
    win.merge(sh, picks, picked, pos0, tok, cnt);
    return .{ tok, cnt };
}

test "with picked as top-k gives it, each list is the vendor's, across the window's and the picks' edges" {
    const a = std.testing.allocator;
    inline for (.{ .{ 4, 3, 9 }, .{ win.topk, win.window, win.topk + win.window + 64 } }) |g| {
        const k: usize = g[0];
        const sh: win.Shape = .{ .rows = 2 * (k + g[1]) + 3, .k = k, .w = g[2], .window = g[1] };
        const picks = try a.alloc(i32, sh.rows * k);
        defer a.free(picks);
        const picked = try a.alloc(i32, sh.rows);
        defer a.free(picked);
        for (0..sh.rows) |r| {
            const seq: i64 = @intCast(r + 1);
            const n: usize = @intCast(@min(@max(seq - @as(i64, @intCast(sh.window)), 0), @as(i64, @intCast(k))));
            picked[r] = @intCast(n);
            // n ascending picks spread below seq - window, as topk_rows leaves them, then -1
            const older = seq - @as(i64, @intCast(sh.window));
            for (picks[r * k ..][0..k], 0..) |*p, i| p.* = if (i < n) @intCast(@divFloor(@as(i64, @intCast(i)) * older, @as(i64, @intCast(n)))) else -1;
        }
        const tok, const cnt = try run(a, sh, 0, picks, picked);
        defer a.free(tok);
        defer a.free(cnt);
        for (0..sh.rows) |r| {
            const seq: i64 = @intCast(r + 1);
            var valid: i32 = 0;
            for (0..sh.w) |c| {
                const want = vendor(picks[r * k ..][0..k], k, sh.window, seq, c);
                try std.testing.expectEqual(want, tok[r * sh.w + c]);
                valid += @intFromBool(want != -1);
            }
            try std.testing.expectEqual(valid, cnt[r]);
            try std.testing.expectEqual(@as(i32, @intCast(seq - 1)), tok[r * sh.w + @as(usize, @intCast(cnt[r])) - 1]);
        }
    }
}

test "rows from a later position: the window ends at each row's own position" {
    const a = std.testing.allocator;
    const sh: win.Shape = .{ .rows = 3, .k = 2, .w = 6, .window = 3 };
    const picks = [_]i32{ 5, 90, 7, 91, 8, 92 };
    const tok, const cnt = try run(a, sh, 99, &picks, &.{ 2, 2, 2 });
    defer a.free(tok);
    defer a.free(cnt);
    try std.testing.expectEqualSlices(i32, &.{ 5, 90, 97, 98, 99, -1, 7, 91, 98, 99, 100, -1, 8, 92, 99, 100, 101, -1 }, tok);
    try std.testing.expectEqualSlices(i32, &.{ 5, 5, 5 }, cnt);
}

test "picked is held to [0, k]: a negative one lists the window alone, one past k lists k picks" {
    const a = std.testing.allocator;
    const sh: win.Shape = .{ .rows = 2, .k = 2, .w = 5, .window = 2 };
    const tok, const cnt = try run(a, sh, 9, &.{ 1, 2, 3, 4 }, &.{ -3, 7 });
    defer a.free(tok);
    defer a.free(cnt);
    try std.testing.expectEqualSlices(i32, &.{ 8, 9, -1, -1, -1, 3, 4, 9, 10, -1 }, tok);
    try std.testing.expectEqualSlices(i32, &.{ 2, 4 }, cnt);
}

test "check refuses lists narrower than k + window, no window, and sizes past i32" {
    try win.check(.{ .rows = 16, .k = win.topk, .w = win.topk + win.window });
    try std.testing.expectError(error.WindowShape, win.check(.{ .rows = 1, .k = 4, .w = 6, .window = 3 }));
    try std.testing.expectError(error.WindowShape, win.check(.{ .rows = 1, .k = 4, .w = 6, .window = 0 }));
    try std.testing.expectError(error.WindowShape, win.check(.{ .rows = 1 << 20, .k = win.topk, .w = 4096 }));
}
