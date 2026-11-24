//! TF_Q27_DFLASH_ADAPTIVE_BLOCK: a lone stream's DFlash2 block rows from its landed depth; the target is unchanged.
const std = @import("std");

pub const wide: u32 = 16;
pub const narrow: u32 = 8;
/// Consecutive rounds landing at most `calm_depth` tree drafts before the block narrows.
pub const calm_rounds: u32 = 8;
pub const calm_depth: u32 = 4;
/// A round landing this many tree drafts is near the narrow block's depth of 7, so the block widens at once.
pub const deep_depth: u32 = 6;

/// The block rule: start wide, narrow after a calm run, widen on one deep round.
pub const Rule = struct {
    rows: u32 = wide,
    calm: u32 = 0,

    pub fn observe(r: *Rule, landed_drafts: u32) void {
        if (landed_drafts >= deep_depth) {
            r.* = .{};
        } else if (landed_drafts > calm_depth) {
            r.calm = 0;
        } else {
            r.calm = @min(r.calm + 1, calm_rounds);
            if (r.calm == calm_rounds) r.rows = narrow;
        }
    }
};

/// One lone stream's rule and its proposal counts; another stream starts the rule again.
pub const Lone = struct {
    enabled: bool = false,
    owner: usize = 0,
    rule: Rule = .{},
    narrow_rounds: u64 = 0,
    wide_rounds: u64 = 0,

    pub fn rows(l: *Lone, stream: usize) u32 {
        if (!l.enabled) return wide;
        if (l.owner != stream) l.start(stream);
        if (l.rule.rows == narrow) l.narrow_rounds += 1 else l.wide_rounds += 1;
        return l.rule.rows;
    }

    pub fn observe(l: *Lone, stream: usize, landed_drafts: u32) void {
        if (l.enabled and l.owner == stream) l.rule.observe(landed_drafts);
    }

    pub fn drop(l: *Lone) void {
        l.start(0);
    }

    fn start(l: *Lone, stream: usize) void {
        l.* = .{ .enabled = l.enabled, .owner = stream };
    }
};

/// Kept path rows that were tree drafts: row 0 is the pending token and rows past `tree_rows` are copies.
pub fn landed(path: []const u32, tree_rows: usize) u32 {
    var n: u32 = 0;
    for (path) |row| n += @intFromBool(row >= 1 and row <= tree_rows);
    return n;
}

fn replay(history: []const u32) Rule {
    var r: Rule = .{};
    for (history) |depth| r.observe(depth);
    return r;
}

test "qwen27 startup: draft block: stays wide until eight consecutive calm rounds" {
    try std.testing.expectEqual(wide, replay(&.{ 2, 3, 1, 4, 0, 2, 3 }).rows);
    try std.testing.expectEqual(narrow, replay(&.{ 2, 3, 1, 4, 0, 2, 3, 4 }).rows);
    try std.testing.expectEqual(wide, replay(&.{ 2, 3, 1, 5, 0, 2, 3, 4 }).rows);
}

test "qwen27 startup: draft block: a depth of five holds the block and one deep round widens it" {
    const calm: [8]u32 = @splat(2);
    const seven: [7]u32 = @splat(1);
    const eight: [8]u32 = @splat(1);
    try std.testing.expectEqual(narrow, replay(&(calm ++ [_]u32{5})).rows);
    try std.testing.expectEqual(wide, replay(&(calm ++ [_]u32{6})).rows);
    try std.testing.expectEqual(wide, replay(&(calm ++ [_]u32{6} ++ seven)).rows);
    try std.testing.expectEqual(narrow, replay(&(calm ++ [_]u32{6} ++ eight)).rows);
}

test "qwen27 startup: draft block: off is always the wide block and counts nothing" {
    var l: Lone = .{};
    for (0..32) |_| {
        try std.testing.expectEqual(wide, l.rows(7));
        l.observe(7, 0);
    }
    try std.testing.expectEqual(@as(u64, 0), l.narrow_rounds + l.wide_rounds);
    try std.testing.expectEqual(wide, l.rule.rows);
}

test "qwen27 startup: draft block: a new stream starts wide and another stream's rounds do not train it" {
    var l: Lone = .{ .enabled = true };
    _ = l.rows(1);
    for (0..8) |_| l.observe(1, 1);
    try std.testing.expectEqual(narrow, l.rows(1));
    l.observe(2, 9);
    try std.testing.expectEqual(narrow, l.rows(1));
    try std.testing.expectEqual(wide, l.rows(2));
    l.drop();
    try std.testing.expectEqual(@as(usize, 0), l.owner);
    try std.testing.expect(l.enabled);
}

test "qwen27 startup: draft block: landed counts tree rows only" {
    try std.testing.expectEqual(@as(u32, 2), landed(&.{ 0, 1, 3, 9 }, 4));
    try std.testing.expectEqual(@as(u32, 0), landed(&.{0}, 15));
}
