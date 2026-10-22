//! An image prompt in the replay engine: the tower's rows for its images, where they sit, and the frame they rotate in.
//! Two frames: the shifted one (`frame`) moves every row below the last image's end by the images' shift, so the rows
//! past it rotate at their slot; the standard one (`standardFrame`, the shared sessions') keeps get_rope_index's own
//! positions, so a prefix's rows sit where any longer prompt puts them and its state resumes as text does, and the rows
//! past the last image rotate at their slot plus `delta` (at most 0).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const lanes = @import("lanes");

/// An image's tokens: the prompt slot of its first, how many, its merged grid, and its first row of the tower's rows.
pub const Span = struct { at: u32, n: u32, mh: u32, mw: u32, row: u32 };

pub const VisPrompt = struct {
    rows: mtl.Buffer, // [tokens, 4, 2560] bf16: the tower's rows in prompt order, once a stream as the embedding
    spans: []const Span,
    pos3: []const i32, // [frame_end * 3]: each slot's t, h, w below frame_end
    frame_end: u32,
    delta: i32 = 0, // the standard frame: rows past the last image rotate at their slot plus this (0: the shifted frame)
    prep: mtl.Pipeline, // fn_attn_prep_mrope

    /// Slot `slot`'s three positions: the frame's below its end, else the slot plus `delta` on all three.
    pub fn at(v: *const VisPrompt, slot: usize) [3]i32 {
        return if (slot < v.frame_end) v.pos3[slot * 3 ..][0..3].* else @splat(@as(i32, @intCast(slot)) + v.delta);
    }

    /// Rows [pos, pos + n): the recorded kernels' position (t) and the multimodal kernel's three.
    pub fn positions(v: *const VisPrompt, pos: usize, n: usize, out_pos: []i32, out3: []i32) void {
        for (0..n) |i| {
            const p = v.at(pos + i);
            out_pos[i] = p[0];
            @memcpy(out3[i * 3 ..][0..3], &p);
        }
    }

    /// The indexer's blocks [from, to): each block's three positions (its first row's) at `out[b * 3 ..]`, as fz_idx_pool_pos
    /// reads them (a text block's three alike, so it keeps q4_idx_pool's bits at that position).
    pub fn blockPositions(v: *const VisPrompt, out: []i32, from: usize, to: usize) void {
        for (from..to) |b| @memcpy(out[b * 3 ..][0..3], &v.at(4 * b));
    }

    /// The image rows among prompt rows [pos, pos + n): (row in the chunk, row in `rows`, count) a span.
    pub fn overlap(s: Span, pos: usize, n: usize) ?[3]usize {
        const lo = @max(pos, s.at);
        const hi = @min(pos + n, @as(usize, s.at) + s.n);
        if (lo >= hi) return null;
        return .{ lo - pos, s.row + (lo - s.at), hi - lo };
    }
};

/// fn_attn_prep_mrope compiled as the recorded kernels are (MLX's options); `lib` outlives `prep`.
pub const Prep = struct {
    lib: mtl.Library,
    prep: mtl.Pipeline,

    pub fn init(device: mtl.Device) !Prep {
        return initCached(device, std.heap.page_allocator, false);
    }

    pub fn initCached(device: mtl.Device, a: std.mem.Allocator, enabled: bool) !Prep {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const source = try @import("rotary_cache.zig").source(arena.allocator(), ks.flashnext_mrope, enabled);
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        return .{ .lib = lib, .prep = try mtl.Pipeline.init(device, lib, "fn_attn_prep_mrope", false) };
    }

    pub fn deinit(p: Prep) void {
        p.prep.deinit();
        p.lib.deinit();
    }
};

/// The total every image's tokens run past max(rows, cols): the frame moves by it.
pub fn shiftOf(spans: []const Span) i64 {
    var shift: i64 = 0;
    for (spans) |s| shift += @as(i64, s.n) - @max(s.mh, s.mw);
    return shift;
}

/// A request's frame for the cache key: its shift (0, the text frame, for none), each image's merge read from its grid.
pub fn frameKey(images: []const lanes.stream.Image) u32 {
    var shift: u64 = 0;
    for (images) |img| {
        if (img.tokens == 0) continue;
        const merge = std.math.sqrt(img.gh * img.gw / img.tokens);
        if (merge == 0) continue;
        shift += img.tokens - @max(img.gh / merge, img.gw / merge);
    }
    return @truncate(shift);
}

/// get_rope_index's positions below the last image's end, shifted so the text after it rotates at its own slot.
pub fn frame(gpa: std.mem.Allocator, spans: []const Span) ![]i32 {
    return frameBy(gpa, spans, shiftOf(spans));
}

/// get_rope_index's own positions below the last image's end (the standard frame): a row's depend only on the images
/// before it; the rows past the last image rotate at their slot minus shiftOf(spans) (VisPrompt.delta).
pub fn standardFrame(gpa: std.mem.Allocator, spans: []const Span) ![]i32 {
    return frameBy(gpa, spans, 0);
}

fn frameBy(gpa: std.mem.Allocator, spans: []const Span, shift: i64) ![]i32 {
    if (spans.len == 0) return &.{};
    const last = spans[spans.len - 1];
    const end: usize = @as(usize, last.at) + last.n;
    const out = try gpa.alloc(i32, end * 3);
    errdefer gpa.free(out);
    var cur: i64 = 0; // transformers' current_pos
    var slot: usize = 0;
    for (spans) |s| {
        if (s.at < slot or s.n != s.mh * s.mw) return error.BadImageSpans;
        while (slot < s.at) : (slot += 1) {
            @memset(out[slot * 3 ..][0..3], @intCast(cur + shift));
            cur += 1;
        }
        for (0..s.n) |k| {
            const t = cur + shift;
            const row: i64 = @intCast(k / s.mw);
            const col: i64 = @intCast(k % s.mw);
            out[(slot + k) * 3 ..][0..3].* = .{ @intCast(t), @intCast(t + row), @intCast(t + col) };
        }
        slot += s.n;
        cur += @max(s.mh, s.mw);
    }
    return out;
}

test "the shifted frame: text after the last image at its slot, the image's tokens on its grid" {
    const a = std.testing.allocator;
    // 3 text tokens, a 2x3 image (6 tokens), then text: transformers gives the image (3, 3+r, 3+c) and the next text 6
    const p = try frame(a, &.{.{ .at = 3, .n = 6, .mh = 2, .mw = 3, .row = 0 }});
    defer a.free(p);
    try std.testing.expectEqual(@as(usize, 9 * 3), p.len);
    // shift 6 - 3 = 3: text 0, 1, 2 at 3, 4, 5; the image from t = 6; the next text (slot 9) at 6 + 3 = 9, its slot
    try std.testing.expectEqualSlices(i32, &.{ 3, 3, 3, 4, 4, 4, 5, 5, 5 }, p[0..9]);
    try std.testing.expectEqualSlices(i32, &.{ 6, 6, 6, 6, 6, 7, 6, 6, 8, 6, 7, 6, 6, 7, 7, 6, 7, 8 }, p[9..27]);
    const v: VisPrompt = .{ .rows = undefined, .spans = &.{}, .pos3 = p, .frame_end = 9, .prep = undefined };
    var t: [2]i32 = undefined;
    var t3: [6]i32 = undefined;
    v.positions(9, 2, &t, &t3); // past the frame: the slot itself on every axis
    try std.testing.expectEqualSlices(i32, &.{ 9, 10 }, &t);
    try std.testing.expectEqualSlices(i32, &.{ 9, 9, 9, 10, 10, 10 }, &t3);
}

test "two images: each starts where transformers' running position left the text" {
    const a = std.testing.allocator;
    const spans = [_]Span{ .{ .at = 1, .n = 4, .mh = 2, .mw = 2, .row = 0 }, .{ .at = 7, .n = 2, .mh = 1, .mw = 2, .row = 4 } };
    const p = try frame(a, &spans);
    defer a.free(p);
    // shift (4 - 2) + (2 - 2) = 2; text slot 0 at 2; image 1 from 3; text slots 5, 6 at 5, 6; image 2 from 7
    try std.testing.expectEqualSlices(i32, &.{ 2, 2, 2 }, p[0..3]);
    try std.testing.expectEqualSlices(i32, &.{ 3, 3, 3, 3, 3, 4, 3, 4, 3, 3, 4, 4 }, p[3..15]);
    try std.testing.expectEqualSlices(i32, &.{ 5, 5, 5, 6, 6, 6 }, p[15..21]);
    try std.testing.expectEqualSlices(i32, &.{ 7, 7, 7, 7, 7, 8 }, p[21..27]);
    try std.testing.expectError(error.BadImageSpans, frame(a, &.{.{ .at = 0, .n = 5, .mh = 2, .mw = 2, .row = 0 }}));
}

test "the standard frame: a prefix's positions whatever images follow, rows past the last image at slot plus delta" {
    const a = std.testing.allocator;
    // turn 1: 3 text tokens and a 2x3 image (6 tokens, shift 3); turn 2 adds a 2x2 image (4 tokens, shift 2) at slot 12
    const one = [_]Span{.{ .at = 3, .n = 6, .mh = 2, .mw = 3, .row = 0 }};
    const two = [_]Span{ one[0], .{ .at = 12, .n = 4, .mh = 2, .mw = 2, .row = 6 } };
    const p1 = try standardFrame(a, &one);
    defer a.free(p1);
    const p2 = try standardFrame(a, &two);
    defer a.free(p2);
    // every row turn 1 holds sits where turn 2 puts it: its state resumes (the shifted frame moves them all by 2)
    try std.testing.expectEqualSlices(i32, p1, p2[0..p1.len]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 1, 2, 2, 2 }, p1[0..9]);
    const s1 = try frame(a, &one);
    defer a.free(s1);
    const s2 = try frame(a, &two);
    defer a.free(s2);
    try std.testing.expect(!std.mem.eql(i32, s1, s2[0..s1.len]));
    // past turn 1's image: slot 9 at 9 - 3 = 6 (transformers' next text position), in both frames' turn-2 tables
    const v1: VisPrompt = .{ .rows = undefined, .spans = &one, .pos3 = p1, .frame_end = 9, .delta = @intCast(-shiftOf(&one)), .prep = undefined };
    const v2: VisPrompt = .{ .rows = undefined, .spans = &two, .pos3 = p2, .frame_end = 16, .delta = @intCast(-shiftOf(&two)), .prep = undefined };
    for (9..12) |slot| try std.testing.expectEqual(v1.at(slot), v2.at(slot));
    try std.testing.expectEqual([3]i32{ 6, 6, 6 }, v1.at(9));
    try std.testing.expectEqual([3]i32{ 11, 11, 11 }, v2.at(16)); // 16 - 5
    // block positions: a block's first row's (rows 0, 4, 8, 12, 16: text, image 1's tokens 1 and 5, image 2's first, past)
    var bp: [5 * 3]i32 = undefined;
    v2.blockPositions(&bp, 0, 5);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 3, 3, 4, 3, 4, 5, 9, 9, 9, 11, 11, 11 }, &bp);
}

test "the frame key: the shift, so one image set's states fit only its own frame" {
    const px: []const f32 = &.{};
    try std.testing.expectEqual(@as(u32, 0), frameKey(&.{}));
    // a 4 x 6 patch grid (2 x 3 merged, 6 tokens) moves the frame by 3; a 2 x 4 one (1 x 2, 2 tokens) by none
    try std.testing.expectEqual(@as(u32, 3), frameKey(&.{.{ .pixels = px, .gh = 4, .gw = 6, .at = 3, .tokens = 6 }}));
    try std.testing.expectEqual(@as(u32, 3), frameKey(&.{ .{ .pixels = px, .gh = 4, .gw = 6, .at = 3, .tokens = 6 }, .{ .pixels = px, .gh = 2, .gw = 4, .at = 12, .tokens = 2 } }));
}

test "the rows of a span a chunk holds" {
    const s: Span = .{ .at = 10, .n = 20, .mh = 4, .mw = 5, .row = 7 };
    try std.testing.expectEqual(@as(?[3]usize, null), VisPrompt.overlap(s, 0, 10));
    try std.testing.expectEqual(@as(?[3]usize, .{ 5, 7, 5 }), VisPrompt.overlap(s, 5, 10));
    try std.testing.expectEqual(@as(?[3]usize, .{ 0, 12, 15 }), VisPrompt.overlap(s, 15, 100));
}
