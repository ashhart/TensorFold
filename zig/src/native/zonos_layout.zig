//! A ZONOS3 request's prompt layout and its streamed chunks, as the host reads them.
const std = @import("std");
const z3 = @import("zonos3");

const cfg = z3.config;
const C = cfg.codebooks;
const W = cfg.frame_width;

/// What a request's prompt rows name: [reference prefix | speaker row] conditions, the text from BOS to EOS, silence.
pub const Layout = struct {
    condition_row: usize,
    prefill: usize,

    pub fn of(rows: []const [W]i32) !Layout {
        var first: ?usize = null;
        var last: ?usize = null;
        for (rows, 0..) |row, i| {
            if (row[C] == z3.prompt.bos and first == null) first = i;
            if (row[C] == z3.prompt.eos) last = i;
        }
        const b = first orelse return error.BadPromptLayout;
        const e = last orelse return error.BadPromptLayout;
        if (b < cfg.conditions or e <= b) return error.BadPromptLayout;
        return .{ .condition_row = b - cfg.conditions, .prefill = rows.len - 1 - e };
    }
};

/// Frames a streamed chunk waits for after `spoken` generated frames went out: 1, 2, 4, 8, then 15 (a second).
pub fn chunkFrames(spoken: usize) usize {
    return @min(15, 1 + spoken);
}

test "chunkFrames: a streamed reply's chunks double to a second" {
    const t = std.testing;
    var spoken: usize = 0;
    for ([_]usize{ 1, 2, 4, 8, 15, 15 }) |want| {
        try t.expectEqual(want, chunkFrames(spoken));
        spoken += want;
    }
}

test "Layout: the condition rows sit before the text and the silence rows after it" {
    const t = std.testing;
    var p = try z3.prompt.build(t.allocator, &.{ 9, 8 }, .embedding, @splat(1), 3);
    defer p.deinit(t.allocator);
    const l = try Layout.of(p.rows);
    try t.expectEqual(p.condition_row, l.condition_row);
    try t.expectEqual(@as(usize, 3), l.prefill);
    try t.expectError(error.BadPromptLayout, Layout.of(p.rows[0..3]));
}
