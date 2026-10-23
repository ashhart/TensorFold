//! An image prompt in the replay engine: its tower rows, where they sit, and its frame (shifted, or the standard one).
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

    /// Blocks [from, to): each block's first row's three positions at `out[b * 3 ..]`, as fz_idx_pool_pos reads them.
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

/// get_rope_index's own positions (the standard frame): a prefix keeps them as images follow; later rows take `delta`.
pub fn standardFrame(gpa: std.mem.Allocator, spans: []const Span) ![]i32 {
    return frameBy(gpa, spans, 0);
}

fn frameBy(gpa: std.mem.Allocator, spans: []const Span, shift: i64) ![]i32 {
    if (spans.len == 0) return &.{};
    var end: usize = 0;
    for (spans) |s| {
        if (s.mh == 0 or s.mw == 0 or s.n == 0 or @as(u64, s.n) != @as(u64, s.mh) * s.mw or s.at < end)
            return error.BadImageSpans;
        end = @as(usize, s.at) + s.n;
        if (end > std.math.maxInt(i32)) return error.BadImageSpans;
    }
    const out = try gpa.alloc(i32, end * 3);
    errdefer gpa.free(out);
    var cur: i64 = 0; // transformers' current_pos
    var slot: usize = 0;
    for (spans) |s| {
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
    // block positions, each its first row's: text, image 1's tokens 1 and 5, image 2's first, past the frame
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

/// Decode rows past the final image use the standard frame's offset; cache and sampling slots stay unchanged.
pub fn decodePositions(out: []u32, delta: i32) void {
    if (delta != 0) for (out) |*p| {
        p.* = @intCast(@as(i64, p.*) + delta);
    };
}

/// A larger session retains every old block position; extra decode blocks follow the last image's offset.
pub fn moveBlocks(out: []i32, old: []const i32, delta: i32) void {
    @memcpy(out[0..old.len], old);
    for (old.len / 3..out.len / 3) |b| @memcpy(out[b * 3 ..][0..3], &@as([3]i32, @splat(@as(i32, @intCast(4 * b)) + delta)));
}

test "Flash Next standard frame: malformed spans refuse before allocation or position writes" {
    const a = std.testing.allocator;
    for ([_]Span{
        .{ .at = 0, .n = 0, .mh = 0, .mw = 0, .row = 0 },
        .{ .at = 0, .n = 1, .mh = 0, .mw = 1, .row = 0 },
        .{ .at = 0, .n = 1, .mh = 65536, .mw = 65536, .row = 0 },
        .{ .at = std.math.maxInt(u32), .n = 1, .mh = 1, .mw = 1, .row = 0 },
    }) |s| {
        try std.testing.expectError(error.BadImageSpans, standardFrame(a, &.{s}));
        try std.testing.expectError(error.BadImageSpans, frame(a, &.{s}));
    }
    const out_of_order = [_]Span{
        .{ .at = 4, .n = 4, .mh = 2, .mw = 2, .row = 0 },
        .{ .at = 1, .n = 1, .mh = 1, .mw = 1, .row = 4 },
    };
    try std.testing.expectError(error.BadImageSpans, standardFrame(a, &out_of_order));
    try std.testing.expectEqual(@as(usize, 0), (try standardFrame(a, &.{})).len);
}

test "Flash Next standard frame: chunked and resumed positions equal a fresh image append" {
    const a = std.testing.allocator;
    const spans = [_]Span{
        .{ .at = 3, .n = 6, .mh = 2, .mw = 3, .row = 0 },
        .{ .at = 12, .n = 4, .mh = 2, .mw = 2, .row = 6 },
    };
    const first = try standardFrame(a, spans[0..1]);
    defer a.free(first);
    const both = try standardFrame(a, &spans);
    defer a.free(both);
    const old: VisPrompt = .{ .rows = undefined, .spans = spans[0..1], .pos3 = first, .frame_end = 9, .delta = -3, .prep = undefined };
    const fresh: VisPrompt = .{ .rows = undefined, .spans = &spans, .pos3 = both, .frame_end = 16, .delta = -5, .prep = undefined };
    var whole: [24]i32 = undefined;
    var whole3: [72]i32 = undefined;
    fresh.positions(0, 24, &whole, &whole3);
    for ([_]usize{ 1, 2, 4, 7, 16 }) |chunk| {
        var resumed: [24]i32 = undefined;
        var resumed3: [72]i32 = undefined;
        old.positions(0, 12, resumed[0..12], resumed3[0..36]);
        var at: usize = 12;
        while (at < 24) {
            const n = @min(chunk, 24 - at);
            fresh.positions(at, n, resumed[at..][0..n], resumed3[at * 3 ..][0 .. n * 3]);
            at += n;
        }
        try std.testing.expectEqualSlices(i32, &whole, &resumed);
        try std.testing.expectEqualSlices(i32, &whole3, &resumed3);
    }
    try std.testing.expectEqual([3]i32{ 6, 6, 6 }, fresh.at(9));
    try std.testing.expectEqual([3]i32{ 9, 9, 9 }, fresh.at(12));
    try std.testing.expectEqual([3]i32{ 19, 19, 19 }, fresh.at(24));
}

test "Flash Next standard frame: mixed decode windows keep independent offsets and sampling slots" {
    const meta = @import("batch_meta.zig");
    const windows = [_]meta.Window{ .{ .slot = 0, .pos = 16, .rows = 3 }, .{ .slot = 1, .pos = 27, .rows = 2 } };
    const plan = try meta.Plan.init(&windows, 64);
    var positions = plan.positions;
    decodePositions(positions[plan.starts[0]..][0..windows[0].rows], -5);
    decodePositions(positions[plan.starts[1]..][0..windows[1].rows], 0);
    try std.testing.expectEqualSlices(u32, &.{ 11, 12, 13, 27, 28 }, positions[0..plan.total]);
    try std.testing.expectEqualSlices(u32, &.{ 17, 18, 19, 28, 29 }, plan.keys[0..plan.total]);
    var solo = [_]u32{ 16, 17, 18 };
    decodePositions(&solo, -5);
    try std.testing.expectEqualSlices(u32, &solo, positions[0..3]);
}

test "Flash Next standard frame: session growth preserves image axes and extends only decode blocks" {
    const a = std.testing.allocator;
    const spans = [_]Span{.{ .at = 3, .n = 6, .mh = 2, .mw = 3, .row = 0 }};
    const pos = try standardFrame(a, &spans);
    defer a.free(pos);
    const vis: VisPrompt = .{ .rows = undefined, .spans = &spans, .pos3 = pos, .frame_end = 9, .delta = -3, .prep = undefined };
    var old: [12]i32 = undefined;
    vis.blockPositions(&old, 0, 4);
    var moved: [24]i32 = undefined;
    moveBlocks(&moved, &old, -3);
    var fresh: [24]i32 = undefined;
    vis.blockPositions(&fresh, 0, 8);
    try std.testing.expectEqualSlices(i32, &fresh, &moved);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 3, 3, 4, 3, 4, 5, 9, 9, 9 }, moved[0..12]);
    try std.testing.expectEqualSlices(i32, &.{ 13, 13, 13, 17, 17, 17, 21, 21, 21, 25, 25, 25 }, moved[12..24]);
}
