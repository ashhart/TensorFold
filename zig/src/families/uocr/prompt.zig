//! A request's ids and each position's rotary coordinates and router blocks, as measured black-box on the release.
const std = @import("std");
const cfg = @import("config.zig");

/// One position's rotary coordinates and router candidates: `mask` is a row of `Prompt.masks`, or -1 for every block.
pub const Coord = struct { tok: u32, page: u32, mask: i32 };

pub const Prompt = struct {
    ids: []u32,
    /// Per position: its place inside its page (tok) and its page, the rotary's two axes.
    tok: []u32,
    page: []u32,
    /// Per position: the candidate row its router reads, or -1 for every remote block.
    mask: []i32,
    /// Candidate rows of `words` u32 (bit b: block b may be read): the prompt's `pages` rows, then decode's.
    masks: []u32,
    words: usize,
    pages: usize,
    /// Tokens each rotary page held up to the last image token (page 0 first).
    counts: []u32,
    /// Where the prompt left the coordinates, for the reply's positions.
    seed: Step,

    pub fn deinit(p: *Prompt, a: std.mem.Allocator) void {
        a.free(p.ids);
        a.free(p.tok);
        a.free(p.page);
        a.free(p.mask);
        a.free(p.masks);
        a.free(p.counts);
        p.* = undefined;
    }

    pub fn len(p: Prompt) usize {
        return p.ids.len;
    }

    /// The candidate rows a launch reads, at least one row so the buffer is never empty.
    pub fn maskRows(p: Prompt) usize {
        return @max(p.masks.len / @max(p.words, 1), 1);
    }
};

/// The prompt for pages of `image_tokens` image tokens each, then `instruction`: [BOS] ([PAGE] [image] x n)... text.
pub fn build(a: std.mem.Allocator, image_tokens: []const usize, instruction: []const u32) !Prompt {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);
    try ids.append(a, cfg.bos);
    for (image_tokens) |n| {
        try ids.append(a, cfg.page_token);
        try ids.appendNTimes(a, cfg.image_token, n);
    }
    try ids.appendSlice(a, instruction);
    return positions(a, try ids.toOwnedSlice(a));
}

/// Reply tokens: page 0's count until a <PAGE>; the k-th goes to page min(k, last); rows read pages spanning k - 1.
pub const Step = struct {
    counts: []const u32 = &.{},
    page: u32 = 0,
    tok: u32 = 0,
    replies: u32 = 0,
    /// The rows decode's candidates start at, and how many there are.
    rows_at: u32 = 0,
    rows: u32 = 0,

    pub fn init(p: Prompt) Step {
        return p.seed;
    }

    pub fn next(s: *Step, id: u32) Coord {
        if (id == cfg.page_token) {
            s.replies += 1;
            s.page = @min(s.replies, @as(u32, @intCast(s.counts.len)) - 1);
            s.tok = s.counts[s.page];
        } else s.tok +%= 1;
        return .{ .tok = s.tok, .page = s.page, .mask = if (s.replies >= 1 and s.replies - 1 < s.rows) @intCast(s.rows_at + s.replies - 1) else -1 };
    }
};

/// A prompt from its ids (owning `ids`); without an image token it is refused (the release turns to 1-D rotary).
pub fn positions(a: std.mem.Allocator, ids: []u32) !Prompt {
    errdefer a.free(ids);
    const n = ids.len;
    const last = std.mem.lastIndexOfScalar(u32, ids, cfg.image_token) orelse return error.UocrNoImage;
    const tok = try a.alloc(u32, n);
    errdefer a.free(tok);
    const page = try a.alloc(u32, n);
    errdefer a.free(page);
    const mask = try a.alloc(i32, n);
    errdefer a.free(mask);

    // up to the last image token: page k from the k-th <PAGE>, each page counting its own tokens from 0
    var counts: std.ArrayList(u32) = .empty;
    errdefer counts.deinit(a);
    try counts.append(a, 0);
    for (ids[0 .. last + 1], 0..) |id, i| {
        if (id == cfg.page_token) try counts.append(a, 0);
        const cur = counts.items.len - 1;
        tok[i] = counts.items[cur];
        page[i] = @intCast(cur);
        counts.items[cur] += 1;
        mask[i] = @as(i32, @intCast(cur)) - 1;
    }
    const pages = counts.items.len - 1;
    const nblk = std.math.divCeil(usize, n, cfg.block) catch unreachable;
    const words = std.math.divCeil(usize, nblk, 32) catch unreachable;
    const decode_rows = decodePages(ids, pages);
    const masks = try a.alloc(u32, (pages + decode_rows) * words);
    errdefer a.free(masks);
    @memset(masks, 0);
    fillMasks(ids, masks, words, pages, decode_rows);

    const held = try counts.toOwnedSlice(a);
    var s: Step = .{ .counts = held, .tok = held[0] -% 1, .rows_at = @intCast(pages), .rows = @intCast(decode_rows) };
    for (ids[last + 1 ..], last + 1..) |id, i| {
        const c = s.next(id);
        tok[i] = c.tok;
        page[i] = c.page;
        // a prompt row reads the blocks holding its page's images, as every prompt position does
        mask[i] = if (s.replies >= 1 and s.replies - 1 < pages) @intCast(s.replies - 1) else -1;
    }
    return .{ .ids = ids, .tok = tok, .page = page, .mask = mask, .masks = masks, .words = words, .pages = pages, .counts = held, .seed = s };
}

/// Decode's page of each image token: marker-numbered, or (count - 1) / 273 before the first marker.
fn decodePage(cum: usize, run: usize) usize {
    return if (cum > 0) cum - 1 else (run - 1) / cfg.page_image_tokens;
}

/// Decode rows: one for every page an image token takes under decode's numbering.
fn decodePages(ids: []const u32, pages: usize) usize {
    var cum: usize = 0;
    var run: usize = 0;
    var most = pages;
    for (ids) |id| {
        cum += @intFromBool(id == cfg.page_token);
        if (id != cfg.image_token) continue;
        run += 1;
        most = @max(most, decodePage(cum, run) + 1);
    }
    return most;
}

fn setBit(row: []u32, b: usize) void {
    row[b / 32] |= @as(u32, 1) << @intCast(b % 32);
}

/// Prompt rows: blocks holding page k's images. Decode rows: blocks whose image pages (decode numbering) span k.
fn fillMasks(ids: []const u32, masks: []u32, words: usize, pages: usize, decode_rows: usize) void {
    var cum: usize = 0;
    var run: usize = 0;
    var b: usize = 0;
    while (b * cfg.block < ids.len) : (b += 1) {
        var lo: usize = std.math.maxInt(usize);
        var hi: usize = 0;
        for (ids[b * cfg.block .. @min(ids.len, (b + 1) * cfg.block)]) |id| {
            cum += @intFromBool(id == cfg.page_token);
            if (id != cfg.image_token) continue;
            run += 1;
            if (cum > 0) setBit(masks[(cum - 1) * words ..][0..words], b);
            const dp = decodePage(cum, run);
            lo = @min(lo, dp);
            hi = @max(hi, dp);
        }
        if (lo > hi) continue;
        for (lo..@min(hi, decode_rows - 1) + 1) |k| setBit(masks[(pages + k) * words ..][0..words], b);
    }
}

test "uocr: page coordinates, a page's markers and the reply's pages" {
    const a = std.testing.allocator;
    const P = cfg.page_token;
    const I = cfg.image_token;
    // [B P I I t P I I I t t]: two pages, a text token inside page 1, the instruction after the last image
    var p = try positions(a, try a.dupe(u32, &.{ 0, P, I, I, 9, P, I, I, I, 10, 11 }));
    defer p.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2 }, p.tok);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0 }, p.page);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1 }, p.mask);
    var s = Step.init(p);
    const want = [_]Coord{ .{ .tok = 3, .page = 0, .mask = -1 }, .{ .tok = 4, .page = 1, .mask = 2 }, .{ .tok = 5, .page = 1, .mask = 2 }, .{ .tok = 4, .page = 2, .mask = 3 }, .{ .tok = 4, .page = 2, .mask = -1 } };
    for ([_]u32{ 12, P, 13, P, P }, want) |id, w| try std.testing.expectEqual(w, s.next(id));
}

test "uocr: a marker in the instruction starts the reply's pages, and images before any marker are page 0's" {
    const a = std.testing.allocator;
    const P = cfg.page_token;
    const I = cfg.image_token;
    var p = try positions(a, try a.dupe(u32, &.{ 0, P, I, I, 7, P, 8, 9 }));
    defer p.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 2, 1, 3, 4, 5 }, p.tok);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 1, 0, 1, 1, 1 }, p.page);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 0, -1, 0, 0, 0 }, p.mask);
    var q = try positions(a, try a.dupe(u32, &.{ 0, I, I, I, P, I, I, 7, 8 }));
    defer q.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 0, 1, 2, 4, 5 }, q.tok);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0, 1, 1, 1, 0, 0 }, q.page);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, -1, -1, 0, 0, 0, -1, -1 }, q.mask);
    try std.testing.expectError(error.UocrNoImage, positions(a, try a.dupe(u32, &.{ 0, P, 5 })));
}
