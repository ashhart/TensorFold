//! A request's prompt rows in the training layout: [reference prefix | speaker row] [condition rows] text [silence].
const std = @import("std");
const cfg = @import("config.zig");

const C = cfg.codebooks;
const W = cfg.frame_width;
const pad = cfg.audio_pad_id;

pub const bos = 2;
pub const eos = 3;
pub const token_offset = 4; // SuperBPE id + 4 is the text id

/// The voice: a 192-d embedding in the speaker row, or reference codes [frames][10] as a delayed prefix.
pub const Speaker = union(enum) {
    none,
    embedding,
    reference: []const [C]i32,
};

pub const Prompt = struct {
    rows: [][W]i32,
    speaker_row: ?usize, // the row the speaker projection replaces
    condition_row: usize, // the first of the condition rows
    prefill: usize, // silence rows at the end: decoded as left context with the generated frames

    pub fn deinit(p: *Prompt, gpa: std.mem.Allocator) void {
        gpa.free(p.rows);
    }
};

fn blank() [W]i32 {
    var r: [W]i32 = @splat(pad);
    r[C] = cfg.text_vocab;
    return r;
}

/// Row `i` of `frames` code frames delayed a row a codebook: codebook j holds frame i - j, padding outside.
fn delayed(codes: []const [C]i32, i: usize) [W]i32 {
    var r = blank();
    for (0..C) |j| if (i >= j and i - j < codes.len) {
        r[j] = codes[i - j][j];
    };
    return r;
}

/// `ids`: the text's SuperBPE ids. `silence`: the codec's silence frame, `silence_frames` of it prefilled (0: none).
pub fn build(gpa: std.mem.Allocator, ids: []const u32, speaker: Speaker, silence: [C]i32, silence_frames: usize) !Prompt {
    if (ids.len == 0) return error.EmptyText;
    var rows: std.ArrayList([W]i32) = .empty;
    errdefer rows.deinit(gpa);
    var speaker_row: ?usize = null;
    switch (speaker) {
        .none => {},
        .embedding => {
            speaker_row = 0;
            try rows.append(gpa, blank());
        },
        .reference => |codes| {
            if (codes.len == 0) return error.EmptyReference;
            for (codes) |row| for (row) |code| if (code < 0 or code >= cfg.codebook_size) return error.BadReferenceCode;
            // the delayed codes and their flushed tail
            for (0..codes.len + C - 1) |i| try rows.append(gpa, delayed(codes, i));
        },
    }
    const condition_row = rows.items.len;
    for (0..cfg.conditions) |_| try rows.append(gpa, blank());
    var text = blank();
    text[C] = bos;
    try rows.append(gpa, text);
    for (ids) |id| {
        if (id + token_offset >= cfg.text_vocab) return error.BadTextToken;
        text[C] = @intCast(id + token_offset);
        try rows.append(gpa, text);
    }
    text[C] = eos;
    try rows.append(gpa, text);
    // the silence frame repeated, delayed, without its flushed tail: generation continues the delay pattern
    for (0..silence_frames) |i| {
        var r = blank();
        for (0..@min(i + 1, C)) |j| r[j] = silence[j];
        try rows.append(gpa, r);
    }
    return .{ .rows = try rows.toOwnedSlice(gpa), .speaker_row = speaker_row, .condition_row = condition_row, .prefill = silence_frames };
}

fn isSpace(c: u21) bool {
    return switch (c) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// A run of whitespace that holds anything but plain spaces becomes one space (caller frees).
pub fn collapseLineBreaks(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    const view = try std.unicode.Utf8View.init(text);
    var it = view.iterator();
    var run_start: ?usize = null;
    var run_break = false;
    var at: usize = 0;
    while (it.nextCodepointSlice()) |slice| : (at += slice.len) {
        const c = std.unicode.utf8Decode(slice) catch unreachable;
        if (isSpace(c)) {
            if (run_start == null) run_start = at;
            run_break = run_break or c != ' ';
            continue;
        }
        if (run_start) |s| {
            try out.appendSlice(gpa, if (run_break) " " else text[s..at]);
            run_start = null;
            run_break = false;
        }
        try out.appendSlice(gpa, slice);
    }
    if (run_start) |s| try out.appendSlice(gpa, if (run_break) " " else text[s..]);
    return out.toOwnedSlice(gpa);
}

test "prompt: an embedding voice gives the speaker row, three condition rows, the text and the silence rows" {
    const t = std.testing;
    const silence: [C]i32 = .{ 1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337 };
    var p = try build(t.allocator, &.{ 1138, 7 }, .embedding, silence, 3);
    defer p.deinit(t.allocator);
    try t.expectEqual(@as(usize, 1 + 3 + 4 + 3), p.rows.len);
    try t.expectEqual(@as(?usize, 0), p.speaker_row);
    try t.expectEqual(@as(usize, 1), p.condition_row);
    try t.expectEqual(@as(i32, bos), p.rows[4][C]);
    try t.expectEqual(@as(i32, 1142), p.rows[5][C]);
    try t.expectEqual(@as(i32, eos), p.rows[7][C]);
    try t.expectEqualSlices(i32, &.{ 1918, pad, pad }, p.rows[8][0..3]);
    try t.expectEqualSlices(i32, &.{ 1918, 746, 3544, pad }, p.rows[10][0..4]);
    try t.expectEqual(@as(i32, cfg.text_vocab), p.rows[10][C]);
}

test "prompt: reference codes are delayed a row a codebook and flushed with padding" {
    const t = std.testing;
    var codes: [2][C]i32 = undefined;
    for (&codes, 0..) |*row, f| for (row, 0..) |*code, j| {
        code.* = @intCast(100 * (f + 1) + j);
    };
    var p = try build(t.allocator, &.{5}, .{ .reference = &codes }, @splat(0), 0);
    defer p.deinit(t.allocator);
    try t.expectEqual(@as(usize, 2 + C - 1 + 3 + 3), p.rows.len);
    try t.expectEqual(@as(?usize, null), p.speaker_row);
    try t.expectEqual(@as(usize, 11), p.condition_row);
    try t.expectEqualSlices(i32, &.{ 100, pad }, p.rows[0][0..2]);
    try t.expectEqualSlices(i32, &.{ 200, 101, pad }, p.rows[1][0..3]);
    try t.expectEqualSlices(i32, &.{ pad, 201, 102 }, p.rows[2][0..3]);
    try t.expectEqual(@as(i32, 209), p.rows[10][9]);
}

test "collapseLineBreaks: breaks become a space, runs of plain spaces stay" {
    const t = std.testing;
    const got = try collapseLineBreaks(t.allocator, "one  two\n\nthree \t four\n");
    defer t.allocator.free(got);
    try t.expectEqualStrings("one  two three four ", got);
}
