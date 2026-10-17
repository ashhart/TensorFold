//! Gemma Q/K norm and interleaved RoPE on synthetic operands, against FlashInfer and SGLang BF16 goldens.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const rows = 7;
const qheads = 16;
const kheads = 2;
const dim = 256;
const qw = qheads * dim;
const kw = kheads * dim;
const qkv = 2 * qw + 2 * kw;
const start = 181;
const context = 192;
const guard = 32;
const sentinel: u16 = 0x7fc1;
const Expected = struct { index: usize, bits: u16 };
const expected_q = [_]Expected{
    .{ .index = 0, .bits = 0xbcc7 },
    .{ .index = 1, .bits = 0x3b2c },
    .{ .index = 63, .bits = 0xbc8a },
    .{ .index = 64, .bits = 0xbc0b },
    .{ .index = 255, .bits = 0x3d4b },
    .{ .index = 256, .bits = 0xbedf },
    .{ .index = 4095, .bits = 0x3b19 },
    .{ .index = 24576, .bits = 0xbce2 },
    .{ .index = 24577, .bits = 0xbc52 },
    .{ .index = 24639, .bits = 0xbcab },
    .{ .index = 24640, .bits = 0xbc41 },
    .{ .index = 24831, .bits = 0x3cfa },
    .{ .index = 24832, .bits = 0x3f05 },
    .{ .index = 28671, .bits = 0x0000 },
};
// Indices 1065 and 2055 separate unrounded arithmetic from the reference's rounding.
const expected_k = [_]Expected{
    .{ .index = 0, .bits = 0x3c99 },
    .{ .index = 1, .bits = 0x3c67 },
    .{ .index = 63, .bits = 0x3a7e },
    .{ .index = 64, .bits = 0x3d41 },
    .{ .index = 255, .bits = 0xbe93 },
    .{ .index = 256, .bits = 0xbec1 },
    .{ .index = 511, .bits = 0xbc4d },
    .{ .index = 1065, .bits = 0x3fde },
    .{ .index = 2055, .bits = 0x3a3b },
    .{ .index = 3072, .bits = 0xbbb4 },
    .{ .index = 3073, .bits = 0x3c87 },
    .{ .index = 3135, .bits = 0xbc35 },
    .{ .index = 3136, .bits = 0x3c89 },
    .{ .index = 3327, .bits = 0xbeb9 },
    .{ .index = 3328, .bits = 0xbf3d },
    .{ .index = 3583, .bits = 0x3c2b },
};

fn bf16Exact(value: f32) u16 {
    return @truncate(@as(u32, @bitCast(value)) >> 16);
}

fn runCase(gpu: check.Gpu, stream: cuda.Stream, kernel: cuda.Function, chunk: usize) !void {
    var x: [rows * qkv]u16 = undefined;
    var weights: [2][dim]u16 = undefined;
    const scale = [_]f32{ 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16 };
    // Integer mantissas and powers of two make all operands exactly BF16-representable, without a host rounding oracle.
    for (&x, 0..) |*value, i| {
        const mantissa: i32 = @intCast((37 * i + 11) % 257);
        value.* = bf16Exact(@as(f32, @floatFromInt(mantissa - 128)) / 16 * scale[i % scale.len]);
    }
    for (0..dim) |i| {
        const q: i32 = @intCast((5 * i + 3) % 17);
        const k: i32 = @intCast((7 * i + 1) % 19);
        weights[0][i] = bf16Exact(@as(f32, @floatFromInt(q - 8)) / 16);
        weights[1][i] = bf16Exact(@as(f32, @floatFromInt(k - 9)) / 32);
    }
    var queries: [guard + rows * qw + guard]u16 = @splat(sentinel);
    var keys: [guard + context * kw + guard]u16 = @splat(sentinel);
    var values: @TypeOf(keys) = @splat(sentinel);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&x));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&weights));
    defer dw.free();
    var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&queries));
    defer dq.free();
    var dk = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&keys));
    defer dk.free();
    var dv = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&values));
    defer dv.free();
    try gpu.ctx.synchronize();
    var at: usize = 0;
    while (at < rows) {
        const count = @min(chunk, rows - at);
        var args: cuda.Args = .{};
        args.add(dx.ptr + at * qkv * 2);
        args.add(dw.ptr);
        args.add(dw.ptr + dim * 2);
        args.add(dq.ptr + (guard + at * qw) * 2);
        args.add(dk.ptr + guard * 2);
        args.add(dv.ptr + guard * 2);
        args.add(@as(i32, @intCast(start + at)));
        args.add(@as(i32, qheads));
        args.add(@as(i32, kheads));
        args.add(@as(i32, dim));
        args.add(@as(i32, 64));
        args.add(@as(f32, 10000000));
        args.add(@as(f32, 1e-6));
        try cuda.launch.launch(kernel, .{ .grid = .{ .x = @intCast(count * (qheads + kheads)) }, .block = .{ .x = 256 } }, stream, &args);
        at += count;
    }
    try stream.synchronize();
    try dq.download(0, std.mem.asBytes(&queries));
    try dk.download(0, std.mem.asBytes(&keys));
    try dv.download(0, std.mem.asBytes(&values));
    for (expected_q) |e| try check.expect(queries[guard + e.index] == e.bits, "Drama query index {d}, chunk {d}: {x} != {x}", .{ e.index, chunk, queries[guard + e.index], e.bits });
    for (expected_k) |e| try check.expect(keys[guard + start * kw + e.index] == e.bits, "Drama key index {d}, chunk {d}: {x} != {x}", .{ e.index, chunk, keys[guard + start * kw + e.index], e.bits });
    for (0..guard) |i| try check.expect(queries[i] == sentinel and queries[guard + rows * qw + i] == sentinel, "Drama query guard {d}, chunk {d}", .{ i, chunk });
    for (keys, values, 0..) |key, value, i| {
        if (i < guard + start * kw or i >= guard + (start + rows) * kw)
            try check.expect(key == sentinel and value == sentinel, "Drama KV cache outside append range at {d}, chunk {d}", .{ i, chunk });
    }
    var x_after: @TypeOf(x) = undefined;
    var weights_after: @TypeOf(weights) = undefined;
    try dx.download(0, std.mem.asBytes(&x_after));
    try dw.download(0, std.mem.asBytes(&weights_after));
    try check.sameBytes("Drama QKV input unchanged", std.mem.asBytes(&x_after), std.mem.asBytes(&x));
    try check.sameBytes("Drama Q/K gamma unchanged", std.mem.asBytes(&weights_after), std.mem.asBytes(&weights));
}

pub fn run(gpu: check.Gpu) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.drama_tower);
    defer module.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const kernel = try module.function("drama_rope_cache");
    for ([_]usize{ 1, 3, 7 }) |chunk| try runCase(gpu, stream, kernel, chunk);
    check.pass("Drama Q/K: reference-rounded norm and RoPE, positions 181..187, chunks 1/3/7, cache bounds", .{});
}
