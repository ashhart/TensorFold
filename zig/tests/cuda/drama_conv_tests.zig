//! Causal convolution's phase-specific precision, raw history, stride and chunk boundaries; no checkpoint required.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const channels = 257; // Exercise a partial second CUDA block.
const stride = channels + 3;
const rows = 7;
const guard = 32;
const sentinel: u16 = 0x7fc1;
const input = [rows][3]u16{
    .{ 0xc06e, 0xbfee, 0x0000 }, .{ 0xbfa5, 0x3f12, 0x401b },
    .{ 0x3f92, 0x4040, 0xc02e }, .{ 0x4065, 0xc009, 0xbe92 },
    .{ 0xbfc9, 0x3e92, 0x4009 }, .{ 0x3f5b, 0x402e, 0xc040 },
    .{ 0x4052, 0xc01b, 0xbf12 },
};
const weights = [3][4]u16{
    .{ 0xbf9c, 0x0000, 0x3f9c, 0xbde4 },
    .{ 0xbf2b, 0x3f0e, 0xbf47, 0x3ee4 },
    .{ 0xbde4, 0x3f8e, 0xbe64, 0x3f80 },
};
const initial = [3][3]u16{
    .{ 0xbfa3, 0xbf23, 0x0000 }, .{ 0x3ee9, 0x3f8c, 0xbf69 }, .{ 0xbee9, 0x3e3a, 0x3f51 },
};
// Evaluated from the stored BF16 operands; row zero tells one-row prefill from decode.
const prefill = [rows][3]u16{
    .{ 0x3f92, 0x3d02, 0xbe8e }, .{ 0xbd0f, 0x3f4c, 0x4055 },
    .{ 0xbe8d, 0xbde8, 0xbde9 }, .{ 0x40b0, 0xbe86, 0x4037 },
    .{ 0x40c3, 0x403c, 0xbe8c }, .{ 0xbde1, 0xbe61, 0xbdd4 },
    .{ 0xbdb7, 0xbe8a, 0x4014 },
};
const decode = [rows][3]u16{
    .{ 0x3f91, 0x3d05, 0xbe8e }, .{ 0xbd0f, 0x3f4c, 0x4055 },
    .{ 0xbe8d, 0xbde8, 0xbde9 }, .{ 0x40b0, 0xbe85, 0x4037 },
    .{ 0x40c3, 0x403c, 0xbe8c }, .{ 0xbde1, 0xbe60, 0xbdd4 },
    .{ 0xbdb9, 0xbe8a, 0x4014 },
};

fn runCase(gpu: check.Gpu, stream: cuda.Stream, kernel: cuda.Function, chunk: usize, expected: *const [rows][3]u16) !void {
    var x: [rows * stride]u16 = @splat(sentinel);
    var w: [channels * 4]u16 = undefined;
    var history: [guard + 3 * channels + guard]u16 = @splat(sentinel);
    var output: [guard + rows * channels + guard]u16 = @splat(sentinel);
    for (0..channels) |c| {
        for (0..rows) |r| x[r * stride + c] = input[r][c % 3];
        for (0..4) |k| w[4 * c + k] = weights[c % 3][k];
        for (0..3) |r| history[guard + r * channels + c] = initial[r][c % 3];
    }
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&x));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&w));
    defer dw.free();
    var dh = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&history));
    defer dh.free();
    var dy = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&output));
    defer dy.free();
    try gpu.ctx.synchronize();
    var start: usize = 0;
    while (start < rows) {
        const count = @min(chunk, rows - start);
        var args: cuda.Args = .{};
        args.add(dx.ptr + start * stride * 2);
        args.add(@as(i32, stride));
        args.add(dw.ptr);
        args.add(dh.ptr + guard * 2);
        args.add(dy.ptr + (guard + start * channels) * 2);
        args.add(@as(i32, @intCast(count)));
        args.add(@as(i32, channels));
        try cuda.launch.launch(kernel, .{ .grid = .{ .x = (channels + 255) / 256 }, .block = .{ .x = 256 } }, stream, &args);
        try stream.synchronize();
        try dh.download(0, std.mem.asBytes(&history));
        // History contains raw inputs, including the initial state until three rows have arrived.
        const seen = start + count;
        for (0..3) |r| for (0..channels) |c| {
            const index = seen + r;
            const want = if (index < 3) initial[index][c % 3] else input[index - 3][c % 3];
            try check.expect(history[guard + r * channels + c] == want, "conv history at row {d}, channel {d}, chunk {d}", .{ seen, c, chunk });
        };
        start = seen;
    }
    try dy.download(0, std.mem.asBytes(&output));
    for (0..rows) |r| for (0..channels) |c| {
        try check.expect(output[guard + r * channels + c] == expected[r][c % 3], "conv output row {d}, channel {d}, chunk {d}: {x} != {x}", .{ r, c, chunk, output[guard + r * channels + c], expected[r][c % 3] });
    };
    for (0..guard) |i| {
        try check.expect(output[i] == sentinel and output[guard + rows * channels + i] == sentinel, "conv output guard {d}", .{i});
        try check.expect(history[i] == sentinel and history[guard + 3 * channels + i] == sentinel, "conv history guard {d}", .{i});
    }
    var x_after: @TypeOf(x) = undefined;
    var w_after: @TypeOf(w) = undefined;
    try dx.download(0, std.mem.asBytes(&x_after));
    try dw.download(0, std.mem.asBytes(&w_after));
    try check.sameBytes("conv input unchanged", std.mem.asBytes(&x_after), std.mem.asBytes(&x));
    try check.sameBytes("conv weights unchanged", std.mem.asBytes(&w_after), std.mem.asBytes(&w));
}

pub fn run(gpu: check.Gpu) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.drama_backbone);
    defer module.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const prefill_fn = try module.function("drama_gdn_conv_prefill");
    const decode_fn = try module.function("drama_gdn_conv");
    for ([_]usize{ 1, 3, 7 }) |chunk| {
        try runCase(gpu, stream, prefill_fn, chunk, &prefill);
        try runCase(gpu, stream, decode_fn, chunk, &decode);
    }
    check.pass("Drama convolution: prefill/decode precision, raw history, chunks 1/3/7, stride and partial block", .{});
}
