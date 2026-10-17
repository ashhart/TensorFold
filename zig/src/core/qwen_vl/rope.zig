//! Qwen-VL's M-RoPE positions: each prompt row's (t, h, w) from its images, and the shift its later rows carry.
const std = @import("std");

/// Images one request holds at most (the server's --vision-max-images bound).
pub const max_images = 64;

/// One image's rows from `at`: `rows` x `cols` merged places, all at text position `base` plus their row and column.
pub const Span = struct { at: u32, rows: u32, cols: u32, base: i32 };

/// The tokens around an image's rows, and the one each row holds.
pub const Ids = struct { start: u32, image: u32, end: u32 };

/// get_rope_index for one stream: text counts up, an image sits at one position, text resumes past its larger side.
pub const Layout = struct {
    spans: [max_images]Span = undefined,
    count: u32 = 0,
    shift: i32 = 0, // a row past every image sits at its index plus this (decode rows, drafts, later turns)

    /// The layout of `prompt` for `images` (prompt order; fields at, gh, gw, tokens), each placeholder run checked.
    pub fn init(prompt: []const u32, images: anytype, ids: Ids, merge: u32) error{BadImagePlaceholders}!Layout {
        var l: Layout = .{};
        if (images.len > max_images) return error.BadImagePlaceholders;
        var cursor: usize = 0;
        for (images) |img| {
            if (merge == 0 or img.gh % merge != 0 or img.gw % merge != 0) return error.BadImagePlaceholders;
            const rows = img.gh / merge;
            const cols = img.gw / merge;
            const n: usize = @as(usize, rows) * cols;
            const at: usize = img.at;
            if (n == 0 or n != img.tokens or at == 0 or at < cursor + 1 or at + n >= prompt.len) return error.BadImagePlaceholders;
            if (prompt[at - 1] != ids.start or prompt[at + n] != ids.end) return error.BadImagePlaceholders;
            for (prompt[at..][0..n]) |t| if (t != ids.image) return error.BadImagePlaceholders;
            const base = @as(i32, @intCast(at)) + l.shift;
            l.spans[l.count] = .{ .at = img.at, .rows = rows, .cols = cols, .base = base };
            l.count += 1;
            l.shift = base + @as(i32, @intCast(@max(rows, cols))) - @as(i32, @intCast(at + n));
            cursor = at + n;
        }
        return l;
    }

    /// Row `p`'s (t, h, w): an image row's place in its grid, any other row its index shifted past earlier images.
    pub fn position(l: *const Layout, p: u32) [3]i32 {
        var shift: i32 = 0;
        for (l.spans[0..l.count]) |s| {
            if (p < s.at) break;
            const k = p - s.at;
            if (k < s.rows * s.cols) return .{ s.base, s.base + @as(i32, @intCast(k / s.cols)), s.base + @as(i32, @intCast(k % s.cols)) };
            shift = s.base + @as(i32, @intCast(@max(s.rows, s.cols))) - @as(i32, @intCast(s.at + s.rows * s.cols));
        }
        const t = @as(i32, @intCast(p)) + shift;
        return .{ t, t, t };
    }

    /// Whether rows [first, first + n) hold an image row.
    pub fn holds(l: *const Layout, first: u32, n: u32) bool {
        for (l.spans[0..l.count]) |s| if (s.at < first + n and first < s.at + s.rows * s.cols) return true;
        return false;
    }
};

/// The axis (0 t, 1 h, 2 w) of rotary pair `d` under interleaved sections: h at d % 3 == 1, w at d % 3 == 2, inside.
pub fn axis(d: u32, sections: [3]u32) u32 {
    if (d % 3 == 1 and d < 3 * sections[1]) return 1;
    if (d % 3 == 2 and d < 3 * sections[2]) return 2;
    return 0;
}

const Img = struct { at: u32, gh: u32, gw: u32, tokens: u32 };
const marks: Ids = .{ .start = 50, .image = 51, .end = 52 };

test "text counts up, an image sits at one position with its grid, text resumes past its larger side" {
    const prompt = [_]u32{ 7, 50, 51, 51, 51, 51, 51, 51, 52, 9 };
    const l = try Layout.init(&prompt, &[_]Img{.{ .at = 2, .gh = 4, .gw = 6, .tokens = 6 }}, marks, 2);
    try std.testing.expectEqual(@as(i32, -3), l.shift);
    var t: [10]i32 = undefined;
    var h: [10]i32 = undefined;
    var w: [10]i32 = undefined;
    for (0..10) |p| {
        const x = l.position(@intCast(p));
        t[p] = x[0];
        h[p] = x[1];
        w[p] = x[2];
    }
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6 }, &t);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 3, 3, 3, 5, 6 }, &h);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 2, 3, 4, 5, 6 }, &w);
    try std.testing.expectEqual([3]i32{ 97, 97, 97 }, l.position(100)); // a decode row
    try std.testing.expect(l.holds(0, 3) and l.holds(7, 1) and !l.holds(8, 2) and !l.holds(0, 2));
}

test "two images shift the text after each, and an empty layout is the identity" {
    // a tall image (3 x 1 merged) then a square one (2 x 2)
    const prompt = [_]u32{ 50, 51, 51, 51, 52, 8, 50, 51, 51, 51, 51, 52, 9 };
    const l = try Layout.init(&prompt, &[_]Img{ .{ .at = 1, .gh = 6, .gw = 2, .tokens = 3 }, .{ .at = 7, .gh = 4, .gw = 4, .tokens = 4 } }, marks, 2);
    try std.testing.expectEqual([3]i32{ 1, 3, 1 }, l.position(3)); // the first image's last row
    try std.testing.expectEqual([3]i32{ 4, 4, 4 }, l.position(4)); // its end marker: base 1 + 3 rows
    try std.testing.expectEqual([3]i32{ 7, 8, 7 }, l.position(9)); // the second image's second row, first column
    try std.testing.expectEqual([3]i32{ 9, 9, 9 }, l.position(11)); // base 7 + 2
    try std.testing.expectEqual([3]i32{ 10, 10, 10 }, l.position(12));
    try std.testing.expectEqual(@as(i32, -2), l.shift);
    const none: Layout = .{};
    try std.testing.expectEqual([3]i32{ 41, 41, 41 }, none.position(41));
}

test "placeholders that do not match their images are refused" {
    const good = [_]u32{ 50, 51, 51, 52 };
    const one = [_]Img{.{ .at = 1, .gh = 2, .gw = 4, .tokens = 2 }};
    _ = try Layout.init(&good, &one, marks, 2);
    const cases = [_][]const u32{ &.{ 49, 51, 51, 52 }, &.{ 50, 51, 51, 53 }, &.{ 50, 51, 7, 52 }, &.{ 50, 51, 51 } };
    for (cases) |p| try std.testing.expectError(error.BadImagePlaceholders, Layout.init(p, &one, marks, 2));
    try std.testing.expectError(error.BadImagePlaceholders, Layout.init(&good, &[_]Img{.{ .at = 1, .gh = 2, .gw = 4, .tokens = 3 }}, marks, 2));
    try std.testing.expectError(error.BadImagePlaceholders, Layout.init(&good, &[_]Img{.{ .at = 0, .gh = 2, .gw = 4, .tokens = 2 }}, marks, 2));
}

test "the 27B's sections [11, 11, 10] interleave over 32 pairs" {
    var count: [3]u32 = @splat(0);
    for (0..32) |d| count[axis(@intCast(d), .{ 11, 11, 10 })] += 1;
    try std.testing.expectEqual([3]u32{ 11, 11, 10 }, count);
    try std.testing.expectEqual(@as(u32, 1), axis(31, .{ 11, 11, 10 }));
    try std.testing.expectEqual(@as(u32, 0), axis(30, .{ 11, 11, 10 }));
    try std.testing.expectEqual(@as(u32, 2), axis(29, .{ 11, 11, 10 }));
}
