//! A request's ids, page2d rotary coordinates and router page ids as the publisher lays them out, and their decode.
const std = @import("std");
const cfg = @import("config.zig");

/// The prompt's tokens and what the decoder needs per position and per 32-key block.
pub const Prompt = struct {
    ids: []u32,
    /// Rotary coordinates: the count within a page, and the page (1-based; 0 for BOS and the instruction).
    tok: []u32,
    page: []u32,
    /// The router's 0-based query page (-1: none) and image page of each image token (-1 elsewhere).
    qpage: []i32,
    image_page: []i32,
    /// Prompt tokens on each rotary page up to the last image, and the page count (max page + 1).
    pf_cnt: []u32,
    n_pg: u32,
    /// Pages in the prompt (<PAGE> markers before the last image).
    pages: u32,

    pub fn deinit(p: *Prompt, a: std.mem.Allocator) void {
        for ([_][]u32{ p.ids, p.tok, p.page, p.pf_cnt }) |s| a.free(s);
        a.free(p.qpage);
        a.free(p.image_page);
        p.* = undefined;
    }

    pub fn len(p: Prompt) usize {
        return p.ids.len;
    }

    /// The image pages a block of the prompt holds: min and max, or -1 for a block without image tokens.
    pub fn blockPages(p: Prompt, c: usize) [2]i32 {
        var lo: i32 = -1;
        var hi: i32 = -1;
        const end = @min((c + 1) * cfg.block, p.ids.len);
        if (c * cfg.block < end) for (p.image_page[c * cfg.block .. end]) |ip| {
            if (ip < 0) continue;
            lo = if (lo < 0) ip else @min(lo, ip);
            hi = @max(hi, ip);
        };
        return .{ lo, hi };
    }
};

/// `<PAGE>` and its image tokens for each page, then the instruction: [0] + ([PAGE] + [IMG] * T_k) * n + tail.
pub fn build(a: std.mem.Allocator, image_tokens: []const usize, instruction: []const u32) !Prompt {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);
    try ids.append(a, cfg.bos);
    for (image_tokens) |t| {
        try ids.append(a, cfg.page_token);
        try ids.appendNTimes(a, cfg.image_token, t);
    }
    try ids.appendSlice(a, instruction);
    return positions(a, try ids.toOwnedSlice(a));
}

/// The positions of a prompt's ids (owning `ids`), as _build_page_rope_pos and the page-aware candidates compute them.
pub fn positions(a: std.mem.Allocator, ids: []u32) !Prompt {
    const n = ids.len;
    var p: Prompt = .{ .ids = ids, .tok = try a.alloc(u32, n), .page = try a.alloc(u32, n), .qpage = try a.alloc(i32, n), .image_page = try a.alloc(i32, n), .pf_cnt = &.{}, .n_pg = 1, .pages = 0 };
    var last_img: ?usize = null;
    for (ids, 0..) |id, i| if (id == cfg.image_token) {
        last_img = i;
    };
    var cum: u32 = 0;
    var pages: u32 = 0;
    for (ids, 0..) |id, i| {
        cum += @intFromBool(id == cfg.page_token);
        if (last_img) |l| if (i == l) {
            pages = cum;
        };
    }
    p.pages = if (last_img != null) pages else 0;
    cum = 0;
    var max_page: u32 = 0;
    for (ids, 0..) |id, i| {
        cum += @intFromBool(id == cfg.page_token);
        const before = if (last_img) |l| i <= l else false;
        const pg: u32 = if (last_img == null) 0 else if (before) cum else cum -| p.pages;
        p.page[i] = pg;
        max_page = @max(max_page, pg);
        const q: i32 = if (before) @as(i32, @intCast(cum)) - 1 else @as(i32, @intCast(cum)) - 1 - @as(i32, @intCast(p.pages));
        p.qpage[i] = if (last_img == null or q < 0) -1 else q;
        p.image_page[i] = if (id == cfg.image_token and cum > 0) @as(i32, @intCast(cum)) - 1 else -1;
    }
    p.n_pg = max_page + 1;
    p.pf_cnt = try a.alloc(u32, p.n_pg);
    @memset(p.pf_cnt, 0);
    const counts = try a.alloc(u32, p.n_pg);
    defer a.free(counts);
    @memset(counts, 0);
    for (p.page, 0..) |pg, i| {
        p.tok[i] = counts[pg];
        counts[pg] += 1;
        if (last_img) |l| if (i <= l) {
            p.pf_cnt[pg] += 1;
        };
    }
    return p;
}

/// The decode side of both machines, seeded from the prompt and stepped with each token fed back.
pub const Step = struct {
    tok: u32,
    page: u32,
    marks: u32,
    pages: u32,
    pf_cnt: []const u32,
    n_pg: u32,

    pub fn init(p: Prompt) Step {
        var marks: u32 = 0;
        for (p.ids) |id| marks += @intFromBool(id == cfg.page_token);
        return .{ .tok = p.tok[p.len() - 1], .page = p.page[p.len() - 1], .marks = marks, .pages = p.pages, .pf_cnt = p.pf_cnt, .n_pg = p.n_pg };
    }

    /// Advance for `id`, the token about to be fed: its rotary (tok, page) and its router query page.
    pub fn next(s: *Step, id: u32) struct { tok: u32, page: u32, qpage: i32 } {
        if (id == cfg.page_token) {
            s.page = @min(s.page + 1, s.n_pg - 1);
            s.tok = s.pf_cnt[s.page];
            s.marks += 1;
        } else s.tok += 1;
        const q = @as(i32, @intCast(s.marks)) - 1 - @as(i32, @intCast(s.pages));
        return .{ .tok = s.tok, .page = s.page, .qpage = if (q < 0) -1 else q };
    }
};

test "uocr: a two-page base prompt's ids, coordinates and page ids, and the decode machines" {
    const a = std.testing.allocator;
    var p = try build(a, &.{ 3, 2 }, &cfg.instruction);
    defer p.deinit(a);
    const P = cfg.page_token;
    const I = cfg.image_token;
    try std.testing.expectEqualSlices(u32, &.{ 0, P, I, I, I, P, I, I, 37460, 4366, 76466, 16 }, p.ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0 }, p.page);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 2, 3, 0, 1, 2, 1, 2, 3, 4 }, p.tok);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1 }, p.qpage);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, 0, 0, -1, 1, 1, -1, -1, -1, -1 }, p.image_page);
    try std.testing.expectEqualSlices(u32, &.{ 1, 4, 3 }, p.pf_cnt);
    try std.testing.expectEqual(@as(u32, 2), p.pages);
    try std.testing.expectEqual([2]i32{ 0, 1 }, p.blockPages(0));
    var s = Step.init(p);
    const t0 = s.next(1234); // text before the first <PAGE>: page 0 continues
    try std.testing.expectEqual(@as(u32, 5), t0.tok);
    try std.testing.expectEqual(@as(i32, -1), t0.qpage);
    const t1 = s.next(P);
    try std.testing.expectEqual(@as(u32, 4), t1.tok);
    try std.testing.expectEqual(@as(u32, 1), t1.page);
    try std.testing.expectEqual(@as(i32, 0), t1.qpage);
    _ = s.next(P);
    const t3 = s.next(P); // past the last page: the page stays, the query page runs on
    try std.testing.expectEqual(@as(u32, 2), t3.page);
    try std.testing.expectEqual(@as(u32, 3), t3.tok);
    try std.testing.expectEqual(@as(i32, 2), t3.qpage);
}
