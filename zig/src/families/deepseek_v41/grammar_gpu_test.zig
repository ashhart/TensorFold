//! Structured output's host side on the GPU forward: staged rows, the follower's message and a rank's column mask.
const std = @import("std");
const gg = @import("grammar_gpu.zig");
const iface = @import("draft/iface.zig");

const gpa = std.testing.allocator;
const all = std.math.maxInt(u32);

test "a staged window keeps its constrained rows at their places, and a follower decodes the same rows" {
    // two segments in one forward: 3 rows (the middle one unconstrained), then 2 rows without a grammar, then 2 rows
    const a = [_]u32{ 0b101, 7, all, all, 1 << 31, 0 };
    const c = [_]u32{ 1, 2, 3, 4 };
    const segs = [_]iface.Segment{
        .{ .slot = 0, .start = 5, .tokens = &.{ 1, 2, 3 }, .parents = null, .draws = &.{ 6, 7, 8 }, .sampling = null, .masks = &a },
        .{ .slot = 1, .start = 9, .tokens = &.{ 4, 5 }, .parents = null, .draws = &.{ 10, 11 }, .sampling = null },
        .{ .slot = 2, .start = 0, .tokens = &.{ 6, 7 }, .parents = null, .draws = &.{ 1, 2 }, .sampling = null, .masks = &c },
    };
    var buf: [gg.max_rows]gg.Row = undefined;
    const rows = try gg.segmentRows(&buf, &segs);
    try std.testing.expectEqual(@as(usize, 4), rows.len);
    for (rows, [_]u32{ 0, 2, 5, 6 }) |r, want| try std.testing.expectEqual(want, r.row);
    var st: gg.Staged = .{};
    defer st.deinit(gpa);
    try st.set(gpa, rows);
    var msg: std.ArrayList(i64) = .empty;
    defer msg.deinit(gpa);
    try st.encode(gpa, &msg);
    try std.testing.expectEqual(@as(usize, 3 + 4 + 4), msg.items.len); // 8 words, two to an i64
    var got: gg.Staged = .{};
    defer got.deinit(gpa);
    try got.decode(gpa, msg.items);
    try std.testing.expectEqualSlices(u32, st.rows.items, got.rows.items);
    try std.testing.expectEqualSlices(u32, st.bits.items, got.bits.items);
    try std.testing.expectEqualSlices(u32, &.{ 0b101, 7, 1 << 31, 0, 1, 2, 3, 4 }, got.bits.items);
    // an odd word count packs its last word alone; nothing staged clears the follower's rows
    try st.set(gpa, &.{.{ .row = 3, .bits = &.{9} }});
    try st.encode(gpa, &msg);
    try got.decode(gpa, msg.items);
    try std.testing.expectEqualSlices(u32, &.{9}, got.bits.items);
    try st.set(gpa, &.{.{ .row = 1, .bits = &.{ all, all } }});
    try std.testing.expectEqual(@as(usize, 0), st.count());
    try st.encode(gpa, &msg);
    try got.decode(gpa, msg.items);
    try std.testing.expectEqual(@as(usize, 0), got.count());
}

test "malformed masks and messages are refused before anything changes" {
    var buf: [gg.max_rows]gg.Row = undefined;
    const bad = [_]iface.Segment{.{ .slot = 0, .start = 0, .tokens = &.{ 1, 2 }, .parents = null, .draws = &.{ 1, 2 }, .sampling = null, .masks = &.{ 1, 2, 3 } }};
    try std.testing.expectError(error.InvalidGrammarMasks, gg.segmentRows(&buf, &bad));
    var st: gg.Staged = .{};
    defer st.deinit(gpa);
    try st.set(gpa, &.{.{ .row = 0, .bits = &.{ 1, 2 } }});
    try std.testing.expectError(error.InvalidGrammarMasks, st.set(gpa, &.{ .{ .row = 0, .bits = &.{1} }, .{ .row = 1, .bits = &.{ 1, 2 } } }));
    try std.testing.expectEqual(@as(usize, 1), st.count()); // the earlier rows stay staged
    const op = @import("forward.zig").Forward.op_grammar;
    for ([_][]const i64{ &.{op}, &.{ op, 1, 0 }, &.{ op, 1, 2, 0 }, &.{ op, 1, 2, -1, 5 }, &.{ op, 65, 1 } }) |m| try std.testing.expectError(error.BadPlan, st.decode(gpa, m));
    try std.testing.expectEqual(@as(usize, 1), st.count());
}

test "a rank masks its own columns: column c is token lo + c, tokens past the bits are disallowed" {
    var row: [96]f32 = undefined;
    for (&row, 0..) |*v, i| v.* = @floatFromInt(i);
    // rank 1 of a 256-token vocabulary: tokens 128..223 in its 96 columns; the bits cover tokens 0..191 only
    var bits: [6]u32 = @splat(0);
    bits[4] = (1 << 3) | (1 << 31); // tokens 131 and 159
    bits[5] = all; // tokens 160..191
    gg.maskRow(&row, 128, &bits);
    for (row, 0..) |v, c| {
        const t = 128 + c;
        const on = t == 131 or t == 159 or (t >= 160 and t < 192);
        try std.testing.expectEqual(on, v == @as(f32, @floatFromInt(c)));
        if (!on) try std.testing.expect(v == -std.math.inf(f32));
    }
}

test "a pick copies the masked rows, not their span: adjacent rows go as one run into slots in staged order" {
    const b = [_]u32{1};
    var st: gg.Staged = .{};
    defer st.deinit(gpa);
    try st.set(gpa, &.{ .{ .row = 0, .bits = &b }, .{ .row = 1, .bits = &b }, .{ .row = 2, .bits = &b }, .{ .row = 5, .bits = &b }, .{ .row = 63, .bits = &b } });
    var buf: [gg.max_rows]gg.Run = undefined;
    try std.testing.expectEqualDeep(@as([]const gg.Run, &.{ .{ .at = 0, .row = 0, .len = 3 }, .{ .at = 3, .row = 5, .len = 1 }, .{ .at = 4, .row = 63, .len = 1 } }), gg.runs(&st, 0, 64, &buf));
    // a pick of rows [1, 6): rows 1, 2 and 5 only
    try std.testing.expectEqualDeep(@as([]const gg.Run, &.{ .{ .at = 1, .row = 1, .len = 2 }, .{ .at = 3, .row = 5, .len = 1 } }), gg.runs(&st, 1, 5, &buf));
    // rows an earlier pick of the forward masked are skipped, and split a run
    st.done.items[1] = true;
    try std.testing.expectEqualDeep(@as([]const gg.Run, &.{ .{ .at = 0, .row = 0, .len = 1 }, .{ .at = 2, .row = 2, .len = 1 }, .{ .at = 3, .row = 5, .len = 1 }, .{ .at = 4, .row = 63, .len = 1 } }), gg.runs(&st, 0, 64, &buf));
}
