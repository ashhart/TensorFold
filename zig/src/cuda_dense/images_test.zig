//! A lane's image rows on the host: spans checked, encoded in order, cut by chunk, spliced over the embedding rows.
const std = @import("std");
const cuda = @import("cuda");
const pass = @import("pass.zig");
const mdl = @import("model.zig");
const recipe = @import("recipe.zig");
const Held = @import("images.zig").Held;
const fixtures = @import("tiny_test.zig");
const t = std.testing;

const Image = recipe.Image;
const mark: u32 = 500;
const hidden = 256;
const row = hidden * 2;

/// A prompt of `len` text tokens with each image's rows set to the image mark.
fn promptOf(buf: []u32, images: []const Image) []u32 {
    @memset(buf, 7);
    for (images) |im| @memset(buf[im.at..][0..im.tokens], mark);
    return buf;
}

fn image(at: u32, tokens: u32) Image {
    return .{ .pixels = &.{}, .gh = 4, .gw = 4, .at = at, .tokens = tokens };
}

test "image spans: ascending, disjoint, inside the prompt, on image marks, within the budget, else refused" {
    var buf: [32]u32 = undefined;
    const two = [_]Image{ image(3, 8), image(20, 4) };
    const p = promptOf(&buf, &two);
    try Held.check(&two, p, mark, 12); // the budget exactly
    try Held.check(&.{}, p, mark, 0);
    try t.expectError(error.DenseImageSpan, Held.check(&two, p, mark, 11));
    try t.expectError(error.DenseImageSpan, Held.check(&.{ image(20, 4), image(3, 8) }, p, mark, 64)); // descending
    try t.expectError(error.DenseImageSpan, Held.check(&.{ image(3, 8), image(10, 2) }, p, mark, 64)); // overlapping
    try t.expectError(error.DenseImageSpan, Held.check(&.{image(30, 4)}, p, mark, 64)); // past the prompt's end
    try t.expectError(error.DenseImageSpan, Held.check(&.{image(32, 1)}, p, mark, 64));
    try t.expectError(error.DenseImageSpan, Held.check(&.{image(3, 0)}, p, mark, 64));
    try t.expectError(error.DenseImageSpan, Held.check(&.{image(2, 8)}, p, mark, 64)); // a text row inside the span
    try t.expectError(error.DenseImageSpan, Held.check(&.{image(3, 8)}, p, mark + 1, 64)); // another mark
    try Held.check(&.{ image(3, 8), image(11, 1) }, promptOf(&buf, &.{ image(3, 8), image(11, 1) }), mark, 64); // adjacent
}

test "a chunk's injects: the image rows inside it, cut inside, at the first and at the last row of an image" {
    var h = try Held.init(t.allocator, &.{ image(3, 8), image(20, 4) }, 0x10000, hidden);
    defer h.deinit(t.allocator);
    try t.expect(h.spans[0].at == 0 and h.spans[1].at == 8);
    const In = pass.Inject;
    try t.expectEqualSlices(In, &.{.{ .row = 3, .rows = 8, .src = 0x10000 }}, h.intersect(0, 16)); // whole
    try t.expectEqualSlices(In, &.{.{ .row = 3, .rows = 2, .src = 0x10000 }}, h.intersect(0, 5)); // cut inside
    try t.expectEqualSlices(In, &.{ .{ .row = 0, .rows = 6, .src = 0x10000 + 2 * row }, .{ .row = 15, .rows = 1, .src = 0x10000 + 8 * row } }, h.intersect(5, 21));
    try t.expectEqualSlices(In, &.{.{ .row = 0, .rows = 1, .src = 0x10000 + 7 * row }}, h.intersect(10, 12)); // its last row
    try t.expectEqualSlices(In, &.{.{ .row = 2, .rows = 1, .src = 0x10000 }}, h.intersect(1, 4)); // its first row
    try t.expectEqual(@as(usize, 0), h.intersect(11, 20).len); // between the images
    try t.expectEqual(@as(usize, 0), h.intersect(0, 3).len);
    try t.expectEqualSlices(In, &.{.{ .row = 0, .rows = 4, .src = 0x10000 + 8 * row }}, h.intersect(20, 1024));
    var none: Held = .{};
    try t.expectEqual(@as(usize, 0), none.intersect(0, 1024).len);
}

const Src = struct {
    images: []const Image,
    after: usize, // cancelled from this check on
    checks: usize = 0,
    pub fn isCancelled(s: *Src) bool {
        s.checks += 1;
        return s.checks > s.after;
    }
};

test "encode: each image in order at its offset in the lane's rows, a cancel check before each" {
    const images = [_]Image{ image(3, 8), image(20, 4) };
    const h = try Held.init(t.allocator, &images, 0x10000, hidden);
    defer {
        var x = h;
        x.deinit(t.allocator);
    }
    const s: cuda.Stream = .{ .d = undefined, .handle = null };
    var tower: fixtures.Fake = .{};
    var all: Src = .{ .images = &images, .after = 2 };
    try h.encode(fixtures.Fake, &tower, s, &all);
    try t.expectEqual(@as(usize, 2), tower.n);
    try t.expectEqual(fixtures.Fake.Call{ .at = 3, .tokens = 8, .out = 0x10000 }, tower.calls[0]);
    try t.expectEqual(fixtures.Fake.Call{ .at = 20, .tokens = 4, .out = 0x10000 + 8 * row }, tower.calls[1]);
    tower.n = 0;
    var cut: Src = .{ .images = &images, .after = 1 };
    try t.expectError(error.Cancelled, h.encode(fixtures.Fake, &tower, s, &cut));
    try t.expectEqual(@as(usize, 1), tower.n); // the second image never reached the tower
    var other: Src = .{ .images = images[0..1], .after = 9 };
    try t.expectError(error.DenseImageSpan, h.encode(fixtures.Fake, &tower, s, &other));
}

fn model(caps: []usize) mdl.Model {
    return .{ .gpa = t.allocator, .d = undefined, .g = fixtures.tiny, .context = 4096, .slots = caps.len, .caps = caps, .acc_width = 512, .part_bytes = 0 };
}

test "the splice's bounds: image rows inside their chain and only in a prompt chunk" {
    var caps = [_]usize{ 4096, 4096 };
    const m = model(&caps);
    const ids: [16]u32 = @splat(mark);
    const In = pass.Inject;
    const inside = [_]In{ .{ .row = 0, .rows = 4, .src = 1 }, .{ .row = 12, .rows = 4, .src = 2 } };
    try t.expectEqual(@as(usize, 16), try pass.rowsOf(&m, &.{.{ .slot = 0, .p0 = 0, .tokens = &ids, .images = &inside }}, true, 15, 1));
    for ([_]In{ .{ .row = 13, .rows = 4, .src = 1 }, .{ .row = 16, .rows = 1, .src = 1 }, .{ .row = 0, .rows = 0, .src = 1 }, .{ .row = std.math.maxInt(usize), .rows = 2, .src = 1 } }) |bad|
        try t.expectError(error.BadChain, pass.rowsOf(&m, &.{.{ .slot = 0, .p0 = 0, .tokens = &ids, .images = &.{bad} }}, true, 15, 1));
    try t.expectEqual(@as(usize, 1), try pass.rowsOf(&m, &.{.{ .slot = 1, .p0 = 7, .tokens = ids[0..1] }}, false, 0, 1)); // control
    try t.expectError(error.BadWindow, pass.rowsOf(&m, &.{.{ .slot = 1, .p0 = 7, .tokens = ids[0..1], .images = inside[0..1] }}, false, 0, 1));
}

const Copies = struct {
    var got: [4][3]u64 = undefined;
    var n: usize = 0;
    var stream: cuda.abi.Stream = null;
    fn copy(dst: u64, src: u64, len: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        got[n] = .{ dst, src, len };
        n += 1;
        stream = s;
        return 0;
    }
};

test "the splice: one device copy an inject over the embedding rows, each chain's rows after the ones before" {
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyDtoDAsync_v2 = Copies.copy;
    Copies.n = 0;
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x1800) };
    const ids: [8]u32 = @splat(mark);
    const a = [_]pass.Inject{ .{ .row = 1, .rows = 2, .src = 0x9000 }, .{ .row = 6, .rows = 2, .src = 0xa000 } };
    const b = [_]pass.Inject{.{ .row = 0, .rows = 3, .src = 0xb000 }};
    try pass.splice(&d, s, 0x100000, hidden, &.{ .{ .slot = 0, .p0 = 0, .tokens = &ids, .images = &a }, .{ .slot = 1, .p0 = 0, .tokens = ids[0..4] }, .{ .slot = 2, .p0 = 0, .tokens = ids[0..3], .images = &b } });
    try t.expectEqual(@as(usize, 3), Copies.n);
    try t.expectEqual([3]u64{ 0x100000 + row, 0x9000, 2 * row }, Copies.got[0]);
    try t.expectEqual([3]u64{ 0x100000 + 6 * row, 0xa000, 2 * row }, Copies.got[1]);
    try t.expectEqual([3]u64{ 0x100000 + 12 * row, 0xb000, 3 * row }, Copies.got[2]);
    try t.expect(Copies.stream == s.handle);
    Copies.n = 0;
    try pass.splice(&d, s, 0x100000, hidden, &.{.{ .slot = 0, .p0 = 0, .tokens = &ids }});
    try t.expectEqual(@as(usize, 0), Copies.n); // text rows alone: no copy
}
