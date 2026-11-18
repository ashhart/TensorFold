//! Speech-edit prompts, fills and splices (zonos3.tts.edit): the rows after a prompt's text, one masked span a marker.
const std = @import("std");
const cfg = @import("config.zig");

const C = cfg.codebooks;
const W = cfg.frame_width;
const pad = cfg.audio_pad_id;

/// The edit adapter's marker rows: the most regions an edit fills.
pub const marker_rows = 3;

/// A masked region of a clip, in codec frames.
pub const Span = struct { start: usize, len: usize };

/// validate_edit_spans: one to `max` spans, ascending, a frame or more apart, inside the audio.
pub fn valid(spans: []const Span, total: usize, max: usize) bool {
    if (spans.len == 0 or spans.len > max) return false;
    var after: usize = 0;
    for (spans, 0..) |s, k| {
        if (s.len == 0 or s.start + s.len > total or (k > 0 and s.start <= after)) return false;
        after = s.start + s.len;
    }
    return true;
}

/// sheared_block: `frames` (then one row of `end`, when given) delayed a row a codebook, with the K - 1 rows of flush.
fn block(gpa: std.mem.Allocator, out: *std.ArrayList([W]i32), frames: []const [C]i32, end: ?i32) !void {
    const n = frames.len + @intFromBool(end != null);
    for (0..n + C - 1) |i| {
        var r: [W]i32 = @splat(pad);
        r[C] = cfg.text_vocab;
        for (0..C) |j| if (i >= j and i - j < n) {
            r[j] = if (i - j < frames.len) frames[i - j][j] else end.?;
        };
        try out.append(gpa, r);
    }
}

/// edit_prompt_tokens: the clip around its spans, a marker a span; fills end at span-end; the next marker last.
pub fn layout(gpa: std.mem.Allocator, codes: []const [C]i32, spans: []const Span, fills: []const []const [C]i32, marker: i32) ![][W]i32 {
    if (!valid(spans, codes.len, spans.len) or fills.len >= spans.len) return error.BadEditSpans;
    for (codes) |row| for (row) |code| if (code < 0 or code >= cfg.codebook_size) return error.BadEditCode;
    var out: std.ArrayList([W]i32) = .empty;
    errdefer out.deinit(gpa);
    var mark: [W]i32 = @splat(pad);
    var cursor: usize = 0;
    for (spans, 0..) |s, j| {
        if (s.start > cursor) try block(gpa, &out, codes[cursor..s.start], null);
        mark[C] = marker + @as(i32, @intCast(j));
        try out.append(gpa, mark);
        cursor = s.start + s.len;
    }
    try block(gpa, &out, codes[cursor..], cfg.eoa_id);
    for (fills, 0..) |fill, j| {
        mark[C] = marker + @as(i32, @intCast(j));
        try out.append(gpa, mark);
        try block(gpa, &out, fill, cfg.span_end_id);
    }
    mark[C] = marker + @as(i32, @intCast(fills.len));
    try out.append(gpa, mark);
    return out.toOwnedSlice(gpa);
}

pub const Fill = struct {
    /// The span's new frames (caller frees).
    frames: [][C]i32,
    /// Whether the fill ended on span-end, with every frame before it a codec frame.
    stopped: bool,
};

/// extract_fill: `rows` are a fill's generated rows with the finishing row; `end_frame` its aligned span-end.
pub fn extract(gpa: std.mem.Allocator, rows: []const [W]i32, end_frame: ?usize) !Fill {
    const n = rows.len -| 1;
    const whole = if (end_frame) |f| @min(f, n) else n -| (C - 1);
    var keep: usize = 0;
    scan: while (keep < whole) : (keep += 1) for (0..C) |cb| {
        const code = if (keep + cb < n) rows[keep + cb][cb] else pad;
        if (code < 0 or code >= cfg.codebook_size) break :scan;
    };
    const frames = try gpa.alloc([C]i32, keep);
    for (frames, 0..) |*f, i| for (f, 0..) |*code, cb| {
        code.* = rows[i + cb][cb];
    };
    return .{ .frames = frames, .stopped = end_frame != null and keep == whole };
}

/// splice_edit: the clip with each span replaced by its fill (caller frees).
pub fn splice(gpa: std.mem.Allocator, codes: []const [C]i32, spans: []const Span, fills: []const []const [C]i32) ![][C]i32 {
    if (!valid(spans, codes.len, spans.len) or fills.len != spans.len) return error.BadEditSpans;
    var out: std.ArrayList([C]i32) = .empty;
    errdefer out.deinit(gpa);
    var cursor: usize = 0;
    for (spans, fills) |s, fill| {
        try out.appendSlice(gpa, codes[cursor..s.start]);
        try out.appendSlice(gpa, fill);
        cursor = s.start + s.len;
    }
    try out.appendSlice(gpa, codes[cursor..]);
    return out.toOwnedSlice(gpa);
}

test "edit layout: context around the spans, a marker a span, earlier fills ended by span-end, the next marker last" {
    const t = std.testing;
    var codes: [6][C]i32 = undefined;
    for (&codes, 0..) |*f, i| f.* = @splat(@intCast(100 + i));
    const spans = [_]Span{ .{ .start = 1, .len = 2 }, .{ .start = 4, .len = 1 } };
    const fill = [_][C]i32{@splat(7)};
    const rows = try layout(t.allocator, &codes, &spans, &.{&fill}, 9000);
    defer t.allocator.free(rows);
    // frame 0 (10 rows), marker 0, frame 3 (10), marker 1, frame 5 and end (11), marker 0, the fill and its end (11)
    try t.expectEqual(@as(usize, 10 + 1 + 10 + 1 + 11 + 1 + 11 + 1), rows.len);
    try t.expectEqual(@as(i32, 100), rows[0][0]);
    try t.expectEqual(@as(i32, pad), rows[0][1]);
    try t.expectEqual(@as(i32, 100), rows[9][9]);
    try t.expectEqual(@as(i32, 9000), rows[10][C]);
    try t.expectEqual(@as(i32, 103), rows[11][0]);
    try t.expectEqual(@as(i32, 9001), rows[21][C]);
    try t.expectEqual(@as(i32, 105), rows[22][0]);
    try t.expectEqual(@as(i32, cfg.eoa_id), rows[23][0]);
    try t.expectEqual(@as(i32, cfg.eoa_id), rows[32][9]);
    try t.expectEqual(@as(i32, 9000), rows[33][C]);
    try t.expectEqual(@as(i32, 7), rows[34][0]);
    try t.expectEqual(@as(i32, cfg.span_end_id), rows[35][0]);
    try t.expectEqual(@as(i32, 9001), rows[rows.len - 1][C]);
    try t.expect(!valid(&.{ .{ .start = 1, .len = 2 }, .{ .start = 3, .len = 1 } }, 6, 3));
    try t.expect(!valid(&spans, 6, 1));
    try t.expectError(error.BadEditSpans, layout(t.allocator, &codes, &spans, &.{ &fill, &fill }, 9000));
}

test "edit fills: the frames before span-end, cut at a frame that holds no code; the splice replaces each span" {
    const t = std.testing;
    // three frames, then span-end from row 3 on in codebook 0, flushed through the delayed codebooks
    var rows: [3 + C + 1][W]i32 = undefined;
    for (&rows, 0..) |*r, i| {
        for (r[0..C], 0..) |*code, cb| code.* = if (i >= cb and i - cb < 3) @intCast(10 * (i - cb) + cb) else if (i < cb) pad else cfg.span_end_id;
        r[C] = cfg.text_vocab;
    }
    const got = try extract(t.allocator, &rows, 3);
    defer t.allocator.free(got.frames);
    try t.expect(got.stopped);
    try t.expectEqual(@as(usize, 3), got.frames.len);
    try t.expectEqual(@as(i32, 29), got.frames[2][9]);
    // no end: the delay tail is dropped; a pad inside a frame cuts the fill there
    const open = try extract(t.allocator, rows[0..8], null);
    defer t.allocator.free(open.frames);
    try t.expect(!open.stopped);
    try t.expectEqual(@as(usize, 0), open.frames.len);
    rows[1][0] = cfg.eoa_id;
    const cut = try extract(t.allocator, &rows, 3);
    defer t.allocator.free(cut.frames);
    try t.expect(!cut.stopped and cut.frames.len == 1);
    var codes: [5][C]i32 = undefined;
    for (&codes, 0..) |*f, i| f.* = @splat(@intCast(i));
    const two = [_][C]i32{ @splat(50), @splat(51) };
    const spliced = try splice(t.allocator, &codes, &.{.{ .start = 1, .len = 3 }}, &.{&two});
    defer t.allocator.free(spliced);
    try t.expectEqual(@as(usize, 4), spliced.len);
    try t.expectEqual(@as(i32, 51), spliced[2][0]);
    try t.expectEqual(@as(i32, 4), spliced[3][0]);
}
