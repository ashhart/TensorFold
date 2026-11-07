//! A page's blocks as plain-text lines with estimated quads, and a line's estimated words (design 5.4).
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const types = @import("types.zig");
const geometry = @import("geometry.zig");
const markdown = @import("markdown.zig");
const Quad = api.ocr.Quad;
const OcrLine = types.OcrLine;

/// One whitespace-separated word of a line and its estimated quad (null when the line has none).
pub const Word = struct { text: []const u8, quad: ?Quad };

/// The page's lines in reading order, allocated in ``a``; error.Unreadable only for a block quad that is not finite.
pub fn fromBlocks(a: Allocator, page: *const types.OcrPage, geo: api.ocr.Geometry) (Allocator.Error || error{Unreadable})![]OcrLine {
    var out: std.ArrayList(OcrLine) = .empty;
    for (page.blocks, 0..) |b, i| {
        if (b.kind == .picture) continue;
        const block: u32 = @intCast(i);
        const texts = try markdown.plainLines(a, b.content);
        if (texts.len == 0) continue;
        const quad = if (geo == .none) null else b.quad;
        if (geo == .line) {
            const text = if (texts.len == 1) texts[0] else try std.mem.join(a, " ", texts);
            try out.append(a, .{ .text = text, .quad = try check(quad, page.width, page.height), .block = block });
            continue;
        }
        const n: f32 = @floatFromInt(texts.len);
        for (texts, 0..) |text, k| {
            const band: ?Quad = if (quad) |q| blk: {
                const t0 = @as(f32, @floatFromInt(k)) / n;
                const t1 = @as(f32, @floatFromInt(k + 1)) / n;
                break :blk .{ lerp(q[0], q[3], t0), lerp(q[1], q[2], t0), lerp(q[1], q[2], t1), lerp(q[0], q[3], t1) };
            } else null;
            try out.append(a, .{ .text = text, .quad = try check(band, page.width, page.height), .block = block });
        }
    }
    return out.toOwnedSlice(a);
}

/// The line's words, its quad cut along TL→TR and BL→BR in proportion to each word's Unicode scalar count.
pub fn words(a: Allocator, line: OcrLine, width: u32, height: u32) (Allocator.Error || error{Unreadable})![]Word {
    var out: std.ArrayList(Word) = .empty;
    var total: usize = 0;
    var it: Split = .{ .s = line.text };
    while (it.next()) |w| {
        try out.append(a, .{ .text = w, .quad = null });
        total += scalars(w);
    }
    const q = line.quad orelse return out.toOwnedSlice(a);
    const sum: f32 = @floatFromInt(total);
    var done: usize = 0;
    for (out.items) |*w| {
        const t0 = @as(f32, @floatFromInt(done)) / sum;
        done += scalars(w.text);
        const t1 = @as(f32, @floatFromInt(done)) / sum;
        w.quad = try check(.{ lerp(q[0], q[1], t0), lerp(q[0], q[1], t1), lerp(q[3], q[2], t1), lerp(q[3], q[2], t0) }, width, height);
    }
    return out.toOwnedSlice(a);
}

fn check(q: ?Quad, w: u32, h: u32) error{Unreadable}!?Quad {
    return if (q) |v| try geometry.validate(v, w, h) else null;
}

/// The point at ``t`` from ``p`` to ``r``; exactly ``p`` at 0 and ``r`` at 1.
fn lerp(p: [2]f32, r: [2]f32, t: f32) [2]f32 {
    return .{ (1 - t) * p[0] + t * r[0], (1 - t) * p[1] + t * r[1] };
}

/// Words between runs of the whitespace Python's ``str.split()`` splits on; a byte that is not UTF-8 is a word byte.
const Split = struct {
    s: []const u8,
    i: usize = 0,

    fn next(it: *Split) ?[]const u8 {
        while (it.i < it.s.len) {
            const n = space(it.s[it.i..]);
            if (n == 0) break;
            it.i += n;
        }
        if (it.i == it.s.len) return null;
        const start = it.i;
        while (it.i < it.s.len and space(it.s[it.i..]) == 0) it.i += charLen(it.s[it.i..]);
        return it.s[start..it.i];
    }

    /// The byte length of the whitespace character that starts ``s``, or 0.
    fn space(s: []const u8) usize {
        const n = charLen(s);
        if (n == 1 and s[0] >= 0x80) return 0; // a stray byte, not U+0080..U+00FF
        const cp: u21 = if (n == 1) s[0] else std.unicode.utf8Decode(s[0..n]) catch return 0;
        return switch (cp) {
            0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => n,
            else => 0,
        };
    }

    /// The bytes of the UTF-8 character that starts ``s``; 1 for a byte that does not start a whole one.
    fn charLen(s: []const u8) usize {
        const n = std.unicode.utf8ByteSequenceLength(s[0]) catch return 1;
        if (n > s.len) return 1;
        _ = std.unicode.utf8Decode(s[0..n]) catch return 1;
        return n;
    }
};

/// Unicode scalar values in ``s``; bytes when it is not valid UTF-8.
fn scalars(s: []const u8) usize {
    return std.unicode.utf8CountCodepoints(s) catch s.len;
}
