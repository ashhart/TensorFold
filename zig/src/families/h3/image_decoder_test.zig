//! MiniMax-H3-Image's image decoder host rules (image_decoder.zig): ids, angles, the suffix, unpatchify, tensors.
const std = @import("std");
const st = @import("safetensors");
const rounding = @import("bf16_round");
const h = @import("image_decoder.zig");
const vae = @import("picture_vae.zig");
const testing = std.testing;

fn arena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(testing.allocator);
}

test "picture decoder ids: each step rounds to bf16, so axes of 6, 10 and 12 cells lose their symmetry" {
    // worked by hand in bf16: the exact values are -5/6, -1/2, -1/6, 1/6, 1/2, 5/6
    try testing.expectEqualSlices(f32, &.{ -0.83203125, -0.5, -0.16796875, 0.1640625, 0.5, 0.8359375 }, &ids(6));
    try testing.expectEqualSlices(f32, &.{ -0.8984375, -0.69921875, -0.5, -0.30078125, -0.1015625, 0.1015625, 0.296875, 0.5, 0.703125, 0.8984375 }, &ids(10));
    try testing.expectEqualSlices(f32, &.{ -0.91796875, -0.75, -0.5859375, -0.41796875, -0.25, -0.08203125, 0.0859375, 0.25, 0.4140625, 0.5859375, 0.75, 0.9140625 }, &ids(12));
    // a full tile's 16 cells are exact: (2i + 1) / 16 - 1
    for (0..16) |i| try testing.expectEqual(@as(f32, @floatFromInt(2 * i + 1)) / 16 - 1, h.axisId(i, 16));
}

fn ids(comptime n: usize) [n]f32 {
    var out: [n]f32 = undefined;
    for (&out, 0..) |*v, i| v.* = h.axisId(i, n);
    return out;
}

test "picture decoder angles: 8 frequencies an axis, 2 pi id rounded to bf16, laid [h 8 | w 8] twice" {
    const inv = h.inverseFrequencies(100);
    try testing.expectEqual(@as(f32, 1), inv[0]);
    try testing.expectEqual(@as(f32, 0.1), inv[4]);
    try testing.expectEqual(@as(f32, 1.0 / @as(f32, @floatCast(std.math.pow(f64, 100, 0.375)))), inv[3]);
    const a = h.angles(0.5, -0.25, inv);
    // 2 pi x 0.5 is 3.1415927 in fp32 and 3.140625 in bf16; -pi / 2 becomes -1.5703125
    try testing.expectEqual(@as(f32, 3.140625), a[0]);
    try testing.expectEqual(@as(f32, -1.5703125), a[8]);
    for (0..8) |j| {
        try testing.expectEqual(@as(f32, 3.140625) * inv[j], a[j]);
        try testing.expectEqual(@as(f32, -1.5703125) * inv[j], a[8 + j]);
        try testing.expectEqual(a[j], a[16 + j]);
        try testing.expectEqual(a[8 + j], a[24 + j]);
    }
}

test "picture decoder rotary table: cells row-major at their ids, the suffix at (0, 0), cos and sin in bf16" {
    var cos: [(6 + h.suffix) * 32]u16 = undefined;
    var sin: [cos.len]u16 = undefined;
    try h.rotary(2, 3, 100, &cos, &sin);
    // token 4 is cell (1, 1): ids 0.5 on the 2-cell axis and 0 on the 3-cell one
    const inv = h.inverseFrequencies(100);
    const angle: f32 = 3.140625 * inv[3];
    try testing.expectEqual(rounding.bits(@floatCast(@cos(@as(f64, angle)))), cos[4 * 32 + 3]);
    try testing.expectEqual(rounding.bits(@floatCast(@sin(@as(f64, angle)))), sin[4 * 32 + 19]);
    try testing.expectEqual(@as(u16, 0x3f80), cos[4 * 32 + 8]);
    // cell 3 is (1, 0): its w id on 3 cells is -2/3, which bf16's steps make -0.6640625
    try testing.expectEqual(@as(f32, -0.6640625), h.axisId(0, 3));
    const turned: f32 = rounding.round32(h.two_pi * -0.6640625) * inv[5];
    try testing.expectEqual(rounding.bits(@floatCast(@cos(@as(f64, turned)))), cos[3 * 32 + 13]);
    for (6..6 + h.suffix) |t| for (0..32) |i| {
        try testing.expectEqual(@as(u16, 0x3f80), cos[t * 32 + i]);
        try testing.expectEqual(@as(u16, 0), sin[t * 32 + i]);
    };
    try testing.expectError(error.H3BadRotary, h.rotary(2, 3, 100, cos[0..32], sin[0..32]));
}

test "picture decoder suffix: the cells, then 4 registers, then one zero token; 1 to 16 cells a side" {
    for (0..6) |t| try testing.expectEqual(h.Token{ .cell = t }, try h.token(t, 6));
    for (0..4) |r| try testing.expectEqual(h.Token{ .register = r }, try h.token(6 + r, 6));
    try testing.expectEqual(h.Token.zero, try h.token(10, 6));
    try testing.expectError(error.H3BadTokenIndex, h.token(11, 6));
    try testing.expectEqual(@as(usize, 261), try h.tokens(16, 16));
    try testing.expectEqual(@as(usize, 7), try h.tokens(1, 2));
    try testing.expectError(error.H3UnsupportedVae, h.tokens(17, 16));
    try testing.expectError(error.H3UnsupportedVae, h.tokens(4, 0));
}

test "picture decoder unpatchify: a token's 768 values are colour, then row, then column of its 16 x 16 pixels" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    // a 2 x 3 tile: 6 tokens, distinct bf16 bit patterns everywhere
    const rows = try x.alloc(u16, 6 * h.patch);
    for (rows, 0..) |*v, i| v.* = @intCast(i + 1);
    const out = try x.alloc(f32, rows.len);
    try h.unpatchify(rows, 2, 3, out);
    // token (0, 1), red, pixel row 0, column 5 (its value 773, holding 774) lands at row 0, column 16 + 5
    try testing.expectEqual(rounding.value(774), out[21]);
    // token (1, 0), green, pixel row 2, column 3 (its value 2,595) lands at plane 1, row 16 + 2, column 3 of 48
    try testing.expectEqual(rounding.value(2596), out[1536 + 18 * 48 + 3]);
    // token (1, 2), blue, pixel row 15, column 15: the last value
    try testing.expectEqual(rounding.value(@intCast(rows.len)), out[out.len - 1]);
    try testing.expectError(error.H3BadLatentLength, h.unpatchify(rows[1..], 2, 3, out[1..]));
}

/// A header of `items` as F32: item `skip` left out, `bend` a row longer, `retype` BF16; the mask, an encoder tensor.
fn header(a: std.mem.Allocator, items: []const h.Item, skip: ?usize, bend: ?usize, retype: ?usize, extra: []const u8) !st.Header {
    var json: std.ArrayList(u8) = .empty;
    try json.appendSlice(a, "{\"decoder.mask_token\": {\"dtype\": \"F32\", \"shape\": [1, 1, 1024], \"data_offsets\": [0, 4096]}");
    try json.appendSlice(a, ", \"encoder.conv_in.bias\": {\"dtype\": \"F32\", \"shape\": [128], \"data_offsets\": [4096, 4608]}");
    var at: usize = 4608;
    if (extra.len > 0) {
        try json.print(a, ", \"{s}\": {{\"dtype\": \"F32\", \"shape\": [1024], \"data_offsets\": [{d}, {d}]}}", .{ extra, at, at + 4096 });
        at += 4096;
    }
    for (items, 0..) |it, i| {
        if (skip == i) continue;
        var n: usize = if (retype == i) 2 else 4;
        try json.print(a, ", \"{s}\": {{\"dtype\": \"{s}\", \"shape\": [", .{ it.name, if (retype == i) "BF16" else "F32" });
        for (it.part.shape, 0..) |d, k| {
            const dim = if (bend == i and k == 0) d + 1 else d;
            n *= dim;
            try json.print(a, "{s}{d}", .{ if (k == 0) "" else ", ", dim });
        }
        try json.print(a, "], \"data_offsets\": [{d}, {d}]}}", .{ at, at + n });
        at += n;
    }
    try json.append(a, '}');
    return st.parseHeader(a, json.items, at);
}

test "picture decoder tensors: the released names and shapes, read in full; others refused, the mask ignored" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    const plan = try h.Plan.init(vae.layers);
    const items = try plan.items(x);
    // the released file's decoder side: 297 tensors read, decoder.mask_token not
    try testing.expectEqual(@as(usize, 297), items.len);
    try testing.expectEqualStrings("post_quant_conv.weight", items[0].name);
    try testing.expectEqualSlices(usize, &.{ 24, 24, 1, 1 }, items[0].part.shape);
    const qkv = items[5 + 23 * 12 + 1];
    try testing.expectEqualStrings("decoder.transformer_blocks.23.attn.to_qkv.weight", qkv.name);
    try testing.expectEqualSlices(usize, &.{ 3072, 1024 }, qkv.part.shape);
    try testing.expectEqualStrings("decoder.transformer_blocks.0.ff.w2.weight", items[5 + 9].name);
    try testing.expectEqualSlices(usize, &.{ 1024, 4096 }, items[5 + 9].part.shape);
    try testing.expectEqualStrings("decoder.proj_out.weight", items[items.len - 2].name);
    try testing.expectEqualSlices(usize, &.{ 768, 1024 }, items[items.len - 2].part.shape);
    // places: aligned to 128 values, in order, none overlapping, the last ending at the total
    for (items[0 .. items.len - 1], items[1..]) |p, q| {
        try testing.expect(p.offset % 128 == 0 and p.offset + h.count(p.part) <= q.offset);
    }
    try testing.expectEqual(plan.total(), std.mem.alignForward(usize, items[items.len - 1].offset + 768, 128));
    var culprit: []const u8 = "";
    try h.check(x, &try header(x, items, null, null, null, ""), items, &culprit);
    try testing.expectError(error.H3MissingVaeTensor, h.check(x, &try header(x, items, 40, null, null, ""), items, &culprit));
    try testing.expectEqualStrings(items[40].name, culprit);
    try testing.expectError(error.H3BadVaeTensor, h.check(x, &try header(x, items, null, 41, null, ""), items, &culprit));
    try testing.expectEqualStrings(items[41].name, culprit);
    try testing.expectError(error.H3BadVaeTensor, h.check(x, &try header(x, items, null, null, 42, ""), items, &culprit));
    try testing.expectEqualStrings(items[42].name, culprit);
    const extra = "decoder.transformer_blocks.24.scale1";
    try testing.expectError(error.H3UnexpectedVaeTensor, h.check(x, &try header(x, items, null, null, null, extra), items, &culprit));
    try testing.expectEqualStrings(extra, culprit);
}

test "picture decoder geometry: tiles of at most 16 cells, latent lengths, requested sizes refused when malformed" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    var c: vae.Config = .{ .mean = @splat(0), .std = @splat(1), .tile = 256, .overlap = 64, .rope_theta = 100 };
    // a 2048 x 2048 canvas: 11 x 11 tiles of 16 x 16 cells; a small one is a single tile of its own size
    const big = try h.tiles(x, &c, 128, 128);
    try testing.expectEqual(@as(usize, 121), big.total());
    try testing.expectEqual(@as(usize, 16), big.h);
    const last = big.at(120);
    try testing.expectEqual(@as(usize, 112), last.row);
    try testing.expectEqual(@as(usize, 10), last.c);
    const small = try h.tiles(x, &c, 6, 10);
    try testing.expectEqual(@as(usize, 1), small.total());
    try testing.expectEqual(@as(usize, 6), small.h);
    try testing.expectEqual(@as(usize, 10), small.w);
    c.tile = 272;
    try testing.expectError(error.H3UnsupportedVae, h.tiles(x, &c, 128, 128));
    try testing.expectError(error.H3BadPictureGrid, h.tiles(x, &c, 0, 4));
    try testing.expectEqual(@as(usize, 24 * 6 * 10), try h.latentLength(6, 10));
    try testing.expectError(error.H3BufferSizeOverflow, h.latentLength(std.math.maxInt(usize) / 2, 3));
    try testing.expectEqual(h.Size{ .width = 1280, .height = 720 }, try h.size("1280x720"));
    for ([_][]const u8{ "1280", "0x720", "x720", "1280x", "1280x-1", "65537x8", "12x8x4" }) |bad| {
        try testing.expectError(error.H3BadPictureSize, h.size(bad));
    }
    try testing.expectError(error.H3UnsupportedVae, h.Plan.init(0));
}
