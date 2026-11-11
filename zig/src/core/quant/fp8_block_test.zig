//! fp8 block: the shared block-FP8 reader (fp8.read) and its exact scale widening (fp8.widen), on the host.
const std = @import("std");
const fp8 = @import("fp8.zig");
const types = @import("types.zig");
const Why = @import("../modelopt.zig").Why;

const Tensor = types.Tensor;
const testing = std.testing;

fn tensor(dtype: types.DType, shape: []const usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

/// A store of named tensors, as safetensors maps, Drama 3's shards and the Flash-Next weight store answer `get`.
const Store = struct {
    names: []const []const u8,
    tensors: []const Tensor,
    gets: usize = 0,

    pub fn get(s: *Store, name: []const u8) !Tensor {
        s.gets += 1;
        for (s.names, s.tensors) |n, t| if (std.mem.eql(u8, n, name)) return t;
        return error.MissingTensor;
    }
};

test "fp8 block: every bf16 bit pattern widens to its f32 value exactly, NaN and infinity kept without `finite`" {
    var bits: [1 << 16]u16 = undefined;
    for (&bits, 0..) |*b, i| b.* = @intCast(i);
    var out: [1 << 16]f32 = undefined;
    try fp8.widen(tensor(.bf16, &.{ 256, 256 }, std.mem.sliceAsBytes(&bits)), &out, false);
    for (bits, out) |b, v| try testing.expectEqual(@as(u32, b) << 16, @as(u32, @bitCast(v)));
}

test "fp8 block: f32 scales copy bit for bit, NaN payloads included" {
    var prng = std.Random.DefaultPrng.init(0xf8b1);
    var words: [4096]u32 = undefined;
    prng.random().bytes(std.mem.sliceAsBytes(&words));
    words[0] = 0x7f800001; // a signalling NaN with a payload
    words[1] = 0xffc12345;
    var out: [4096]f32 = undefined;
    try fp8.widen(tensor(.f32, &.{ 64, 64 }, std.mem.sliceAsBytes(&words)), &out, false);
    try testing.expectEqualSlices(u32, &words, @ptrCast(&out));
}

test "fp8 block: every finite f16 bit pattern widens to its value exactly" {
    var bits: [1 << 16]u16 = undefined;
    for (&bits, 0..) |*b, i| b.* = @intCast(i);
    var out: [1 << 16]f32 = undefined;
    try fp8.widen(tensor(.f16, &.{1 << 16}, std.mem.sliceAsBytes(&bits)), &out, false);
    for (bits, out) |b, v| {
        const h: f16 = @bitCast(b);
        if (std.math.isFinite(h)) try testing.expectEqual(@as(f64, @floatCast(h)), @as(f64, @floatCast(v)));
    }
}

test "fp8 block: `finite` refuses a NaN or infinite scale, and only then" {
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f81 }) |bad| {
        var bits = [_]u16{ 0x3f80, bad, 0x4000 };
        var out: [3]f32 = undefined;
        try testing.expectError(error.UnsupportedQuantization, fp8.widen(tensor(.bf16, &.{ 1, 3 }, std.mem.sliceAsBytes(&bits)), &out, true));
        try fp8.widen(tensor(.bf16, &.{ 1, 3 }, std.mem.sliceAsBytes(&bits)), &out, false);
    }
    const ok = [_]f32{ 0.5, 1e-30, -2 };
    var out: [3]f32 = undefined;
    try fp8.widen(tensor(.f32, &.{ 1, 3 }, std.mem.sliceAsBytes(&ok)), &out, true);
    try testing.expectEqualSlices(f32, &ok, &out);
}

test "fp8 block: widen refuses a wrong output length and a dtype the kernels do not read" {
    const two = [_]f32{ 1, 2 };
    var out: [3]f32 = undefined;
    try testing.expectError(error.UnexpectedTensor, fp8.widen(tensor(.f32, &.{ 1, 2 }, std.mem.sliceAsBytes(&two)), &out, false));
    const wide = [_]f64{1};
    try testing.expectError(error.UnexpectedTensor, fp8.widen(tensor(.f64, &.{ 1, 1 }, std.mem.sliceAsBytes(&wide)), out[0..1], false));
    try testing.expectError(error.UnexpectedTensor, fp8.widen(tensor(.f32, &.{ 1, 2 }, std.mem.sliceAsBytes(&two)[0..4]), out[0..2], false));
}

const codes: [130 * 260]u8 = @splat(0x38);
const inv: [2 * 3]f32 = .{ 1, 2, 3, 4, 5, 6 };
const inv16: [2 * 3]u16 = .{ 1, 2, 3, 4, 5, 6 };

test "fp8 block: read returns the stored views of weight and weight_scale_inv, a partial last block counted" {
    var s: Store = .{
        .names = &.{ "m.q.weight", "m.q.weight_scale_inv" },
        .tensors = &.{ tensor(.f8_e4m3, &.{ 130, 260 }, &codes), tensor(.f32, &.{ 2, 3 }, std.mem.sliceAsBytes(&inv)) },
    };
    var why: Why = .{};
    const h = try fp8.read(&s, "m.q", 130, 260, .f32, &why);
    try testing.expectEqual(@intFromPtr(&codes), @intFromPtr(h.weight.bytes.ptr));
    try testing.expectEqual(@intFromPtr(&inv), @intFromPtr(h.scale_inv.bytes.ptr));
    try testing.expectEqual(@as(usize, 2), s.gets);
    try testing.expectEqual(@as(usize, 0), why.len);
    try testing.expectEqual(@as(usize, 2), fp8.blocks(130));
    try testing.expectEqual(@as(usize, 1), fp8.blocks(128));
}

test "fp8 block: read refuses by name a wrong dtype, a wrong shape and short bytes; a missing tensor is the store's" {
    const Case = struct { w: Tensor, s: Tensor, n: usize, k: usize, dtype: types.DType, text: []const u8 };
    const w = tensor(.f8_e4m3, &.{ 130, 260 }, &codes);
    const s = tensor(.f32, &.{ 2, 3 }, std.mem.sliceAsBytes(&inv));
    const cases = [_]Case{
        .{ .w = tensor(.u8, &.{ 130, 260 }, &codes), .s = s, .n = 130, .k = 260, .dtype = .f32, .text = "m.q.weight: u8 { 130, 260 }, expected f8_e4m3 { 130, 260 }" },
        .{ .w = w, .s = s, .n = 128, .k = 260, .dtype = .f32, .text = "m.q.weight: f8_e4m3 { 130, 260 }, expected f8_e4m3 { 128, 260 }" },
        .{ .w = w, .s = s, .n = 130, .k = 260, .dtype = .bf16, .text = "m.q.weight_scale_inv: f32 { 2, 3 }, expected bf16 { 2, 3 }" },
        .{ .w = w, .s = tensor(.f32, &.{ 3, 2 }, std.mem.sliceAsBytes(&inv)), .n = 130, .k = 260, .dtype = .f32, .text = "m.q.weight_scale_inv: f32 { 3, 2 }, expected f32 { 2, 3 }" },
        .{ .w = w, .s = tensor(.f32, &.{ 2, 3 }, std.mem.sliceAsBytes(&inv)[0..20]), .n = 130, .k = 260, .dtype = .f32, .text = "m.q.weight_scale_inv: f32 { 2, 3 }, expected f32 { 2, 3 }" },
        .{ .w = tensor(.f8_e4m3, &.{ 130, 260 }, codes[1..]), .s = s, .n = 130, .k = 260, .dtype = .f32, .text = "m.q.weight: f8_e4m3 { 130, 260 }, expected f8_e4m3 { 130, 260 }" },
    };
    for (cases) |c| {
        var store: Store = .{ .names = &.{ "m.q.weight", "m.q.weight_scale_inv" }, .tensors = &.{ c.w, c.s } };
        var why: Why = .{};
        try testing.expectError(error.UnexpectedTensor, fp8.read(&store, "m.q", c.n, c.k, c.dtype, &why));
        try testing.expectEqualStrings(c.text, why.text());
    }
    var none: Store = .{ .names = &.{"m.q.weight"}, .tensors = &.{w} };
    var why: Why = .{};
    try testing.expectError(error.MissingTensor, fp8.read(&none, "m.q", 130, 260, .f32, &why));
    // bf16 scales of the same blocks read as stored
    var b: Store = .{ .names = &.{ "m.q.weight", "m.q.weight_scale_inv" }, .tensors = &.{ w, tensor(.bf16, &.{ 2, 3 }, std.mem.sliceAsBytes(&inv16)) } };
    _ = try fp8.read(&b, "m.q", 130, 260, .bf16, &why);
}

test "fp8 block: read then widen gives reference()'s scales, block by block" {
    var cs: [130 * 260]u8 = @splat(0x38); // 1.0
    cs[129 * 260 + 259] = 0x40; // 2.0 in the last, partial block
    var s: Store = .{ .names = &.{ "e.weight", "e.weight_scale_inv" }, .tensors = &.{ tensor(.f8_e4m3, &.{ 130, 260 }, &cs), tensor(.f32, &.{ 2, 3 }, std.mem.sliceAsBytes(&inv)) } };
    var why: Why = .{};
    const h = try fp8.read(&s, "e", 130, 260, .f32, &why);
    var widened: [6]f32 = undefined;
    try fp8.widen(h.scale_inv, &widened, true);
    var ref: [130 * 260]f64 = undefined;
    try fp8.reference(h, &ref);
    for (0..130) |r| for (0..260) |c| {
        const want = fp8.e4m3(cs[r * 260 + c]) * widened[(r / fp8.block) * 3 + c / fp8.block];
        try testing.expectEqual(want, ref[r * 260 + c]);
    };
}
