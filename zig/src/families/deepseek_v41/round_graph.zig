//! TF_DSV41_ROUND_GRAPH: the decode round's launches without the host's idle gap before the next window's graph.
const std = @import("std");
const calls = @import("block_calls.zig");

pub const Settings = @import("forward_types.zig").RoundGraph;

pub const Get = *const fn ([:0]const u8) ?[]const u8;

pub fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.trim(u8, std.mem.span(std.c.getenv(name) orelse return null), " \t");
    return if (v.len == 0) null else v;
}

/// TF_DSV41_ROUND_GRAPH = 0 | 1 | words from {split, gate} (comma separated); TF_DSV41_ROUND_GRAPH_HEAD = layers >= 1.
pub fn settings(get: Get) !Settings {
    var s: Settings = .{};
    if (get("TF_DSV41_ROUND_GRAPH")) |raw| {
        if (std.mem.eql(u8, raw, "1")) {
            s.split = true;
            s.gate = true;
        } else if (!std.mem.eql(u8, raw, "0")) {
            var it = std.mem.tokenizeAny(u8, raw, ", ");
            while (it.next()) |w| {
                if (std.mem.eql(u8, w, "split")) s.split = true else if (std.mem.eql(u8, w, "gate")) s.gate = true else return error.BadRoundGraph;
            }
        }
    }
    if (get("TF_DSV41_ROUND_GRAPH_HEAD")) |v| s.head = std.fmt.parseInt(u32, v, 10) catch return error.BadRoundGraph;
    if (s.head == 0) return error.BadRoundGraph;
    return s;
}

var cached: ?Settings = null;

/// The process's settings (read once; every rank reads the same environment).
pub fn current() !Settings {
    if (cached) |s| return s;
    const s = try settings(env);
    cached = s;
    return s;
}

/// Where a window's program splits: the first call of layer `head`, moved to one with no side stream open; null: none.
pub fn splitAt(cs: []const calls.Call, head: u32) ?usize {
    var layer: u32 = 0;
    var side = false;
    var dfr = false;
    var due = false;
    for (cs, 0..) |c, i| {
        if (c.begin == .layer) {
            layer += 1;
            if (layer == head) due = true;
        }
        if (due and !side and !dfr) return if (i == 0) null else i;
        if (c.join) side = false;
        if (c.fork) side = true;
        if (c.defer_join) dfr = false;
        if (c.defer_side) dfr = true;
    }
    return null;
}

fn call(begin: calls.Begin, flags: struct { fork: bool = false, join: bool = false, side: bool = false, defer_side: bool = false, defer_join: bool = false }) calls.Call {
    return .{ .triton = false, .name = "x.y", .args = &.{}, .begin = begin, .fork = flags.fork, .join = flags.join, .side = flags.side, .defer_side = flags.defer_side, .defer_join = flags.defer_join };
}

test "the knob's words" {
    const T = struct {
        var v: ?[]const u8 = null;
        var h: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DSV41_ROUND_GRAPH")) v else h;
        }
    };
    try std.testing.expectEqual(Settings{}, try settings(T.get));
    T.v = "1";
    try std.testing.expectEqual(Settings{ .split = true, .gate = true }, try settings(T.get));
    T.v = "gate";
    try std.testing.expectEqual(Settings{ .gate = true }, try settings(T.get));
    T.v = "split, gate";
    T.h = "3";
    try std.testing.expectEqual(Settings{ .split = true, .gate = true, .head = 3 }, try settings(T.get));
    T.v = "0";
    T.h = null;
    try std.testing.expectEqual(Settings{}, try settings(T.get));
    T.v = "chain";
    try std.testing.expectError(error.BadRoundGraph, settings(T.get));
    T.v = "1";
    T.h = "0";
    try std.testing.expectError(error.BadRoundGraph, settings(T.get));
}

test "the split: the head's layers, never inside an open side or deferred fork" {
    // layer 0: 0..2, layer 1: 3..6 (its side fork 3 joined at 5), layer 2: 7..
    const cs = [_]calls.Call{
        call(.window, .{}),
        call(.none, .{}),
        call(.none, .{}),
        call(.layer, .{}),
        call(.none, .{ .fork = true, .side = true }),
        call(.none, .{ .join = true }),
        call(.none, .{}),
        call(.layer, .{}),
        call(.none, .{}),
    };
    try std.testing.expectEqual(@as(?usize, 3), splitAt(&cs, 1));
    try std.testing.expectEqual(@as(?usize, 7), splitAt(&cs, 2));
    try std.testing.expectEqual(@as(?usize, null), splitAt(&cs, 3));
    // a side fork opened in layer 0 and joined in layer 1: the split waits for its join
    var c2 = cs;
    c2[2] = call(.none, .{ .fork = true, .side = true });
    c2[3] = call(.layer, .{});
    c2[4] = call(.none, .{ .join = true });
    try std.testing.expectEqual(@as(?usize, 5), splitAt(&c2, 1));
    // the deferred mHC stream likewise
    var c3 = cs;
    c3[1] = call(.none, .{ .defer_side = true });
    c3[3] = call(.layer, .{});
    c3[6] = call(.none, .{ .defer_join = true });
    try std.testing.expectEqual(@as(?usize, 7), splitAt(&c3, 1));
    // a join and a fork on one call (the join first, as runner.issue): open after it, until layer 1's join at 5
    var c4 = cs;
    c4[2] = call(.none, .{ .join = true, .fork = true, .side = true });
    try std.testing.expectEqual(@as(?usize, 6), splitAt(&c4, 1));
}
