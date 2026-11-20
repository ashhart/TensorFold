const std = @import("std");
const gpu = @import("tensor_metal");
const abi = @import("kernel_abi.zig");
const part = @import("runtime.zig").part;
const gridRow = @import("spatial.zig").gridRow;
const Tensor = gpu.Tensor;
const Size = gpu.mtl.Size;
const group = Size.of(256, 1, 1);
const unit = std.math.floatEps(f32);

pub fn run(g: *gpu.Gpu) !void {
    try checkRotaryTable(g);
    try checkRotate(g);
    try checkSplitRotate(g);
    try checkTiles(g);
    try checkDeconvolution(g);
    try checkConvolution(g);
    try checkDepthwise(g);
    try checkGroupNorm(g);
    try checkSquash(g);
    try checkChannelFill(g);
    try checkFirstMaskConv(g);
}

fn filled(g: *gpu.Gpu, count: usize, rate: f32, amplitude: f32) !Tensor {
    const tensor = try g.alloc(count, .f32);
    for (tensor.slice(f32, count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * rate) * amplitude;
    return tensor;
}

fn word(tensor: Tensor, index: usize) u32 {
    return tensor.slice(u32, tensor.count)[index];
}

fn generatedTable(g: *gpu.Gpu, side: usize, head_width: usize, scale: f32) !Tensor {
    const table = try g.allocDevice(side * side * head_width, .f32);
    errdefer table.deinit();
    const params = abi.Rotary{ .rows = 0, .heads = 0, .head_width = @intCast(head_width), .rotated_rows = 0, .positions = @intCast(side * side), .side = @intCast(side), .scale = scale };
    try g.dispatch(abi.rotary_table, &.{table}, params, Size.of(table.count / 2, 1, 1), group);
    return table;
}

fn rotaryAngle(position: usize, pair: usize, side: usize, head_width: usize, scale: f32) f64 {
    const quarter = head_width / 4;
    const coordinate: f64 = @floatFromInt(if (pair < quarter) position % side else position / side);
    const exponent = -@as(f64, @floatFromInt(pair % quarter)) / @as(f64, @floatFromInt(quarter));
    return coordinate * scale * std.math.pow(f64, 10000, exponent);
}

fn checkRotaryTable(g: *gpu.Gpu) !void {
    var worst: f64 = 0;
    for ([_]usize{ 32, 64 }) |head_width| for ([_]usize{ 24, 72 }) |side| for ([_]f32{ 1, @as(f32, 24) / @as(f32, 72) }) |scale| {
        const table = try generatedTable(g, side, head_width, scale);
        defer table.deinit();
        const pairs = head_width / 2;
        for (0..side * side) |position| for (0..pairs) |pair| {
            const theta = rotaryAngle(position, pair, side, head_width, scale);
            const at = (position * pairs + pair) * 2;
            worst = @max(worst, @abs(@cos(theta) - table.get(at)), @abs(@sin(theta) - table.get(at + 1)));
        };
    };
    std.debug.print("SAM rotary tables: maximum float64 cos/sin error {e}\n", .{worst});
    if (!(worst <= 5e-4)) return error.SamRotaryTableParity;
}

fn checkRotate(g: *gpu.Gpu) !void {
    const rows = 40;
    const heads = 2;
    const head_width = 8;
    const positions = 16;
    const rotated = 37;
    const table = try generatedTable(g, 4, head_width, 1);
    defer table.deinit();
    const x = try filled(g, rows * heads * head_width, 0.37, 1.5);
    defer x.deinit();
    const y = try g.allocDevice(x.count, .f32);
    defer y.deinit();
    const params = abi.Rotary{ .rows = rows, .heads = heads, .head_width = head_width, .rotated_rows = rotated, .positions = positions, .side = 4, .scale = 1 };
    try g.dispatch(abi.rotate, &.{ x, table, y }, params, Size.of(x.count, 1, 1), group);
    var worst: f64 = 0;
    for (0..x.count) |i| {
        const row = i / (heads * head_width);
        const lane = i % head_width;
        if (row >= rotated) {
            if (word(y, i) != word(x, i)) return error.SamRotateCopy;
            continue;
        }
        const at = ((row % positions) * (head_width / 2) + lane / 2) * 2;
        const sine: f64 = table.get(at + 1);
        const own = @as(f64, x.get(i)) * table.get(at);
        const other = @as(f64, x.get(i ^ 1)) * (if (lane % 2 == 1) sine else -sine);
        const bound = 2 * unit * (@abs(own) + @abs(other));
        const err = @abs(own + other - y.get(i));
        if (err > bound) return error.SamRotateParity;
        if (bound > 0) worst = @max(worst, err / bound);
    }
    std.debug.print("SAM rotation: worst float64 error {d:.3} of its bound; rows past the rotated count are copies\n", .{worst});
}

fn checkSplitRotate(g: *gpu.Gpu) !void {
    const rows = 40;
    const width = 16;
    const head_width = 8;
    const positions = 16;
    const table = try generatedTable(g, 4, head_width, 1);
    defer table.deinit();
    const count = rows * width;
    const packed_qkv = try filled(g, 3 * count, 0.23, 2);
    defer packed_qkv.deinit();
    const wide = try g.allocDevice(3 * count, .f32);
    defer wide.deinit();
    const narrow = try g.allocDevice(3 * count, .f16);
    defer narrow.deinit();
    const thirds = try g.alloc(3 * count, .f32);
    defer thirds.deinit();
    const rotated = try g.allocDevice(2 * count, .f32);
    defer rotated.deinit();
    for (0..3) |index| for (0..rows) |row| for (0..width) |column| {
        thirds.slice(f32, thirds.count)[index * count + row * width + column] = packed_qkv.get((row * 3 + index) * width + column);
    };
    const split = abi.RotarySplit{ .rows = rows, .width = width, .head_width = head_width, .positions = positions };
    const grid = Size.of(count, 1, 1);
    const outputs = [3]Tensor{ part(wide, 0, count), part(wide, count, count), part(wide, 2 * count, count) };
    const halves = [3]Tensor{ part(narrow, 0, count), part(narrow, count, count), part(narrow, 2 * count, count) };
    try g.dispatch(abi.rotate_split_f32, &.{ packed_qkv, table, outputs[0], outputs[1], outputs[2] }, split, grid, group);
    try g.dispatch(abi.rotate_split_f16, &.{ packed_qkv, table, halves[0], halves[1], halves[2] }, split, grid, group);
    const rotation = abi.Rotary{ .rows = rows, .heads = width / head_width, .head_width = head_width, .rotated_rows = rows, .positions = positions, .side = 4, .scale = 1 };
    for (0..2) |index| try g.dispatch(abi.rotate, &.{ part(thirds, index * count, count), table, part(rotated, index * count, count) }, rotation, grid, group);
    for (0..3 * count) |i| {
        const expected = if (i < 2 * count) word(rotated, i) else word(thirds, i);
        if (word(wide, i) != expected) return error.SamSplitRotateParity;
        const rounded: f16 = @floatCast(wide.get(i));
        if (narrow.slice(u16, narrow.count)[i] != @as(u16, @bitCast(rounded))) return error.SamSplitRotateHalf;
    }
    std.debug.print("SAM split rotation: queries and keys equal the rotation kernel, values copy, FP16 rounds FP32\n", .{});
}

fn checkTiles(g: *gpu.Gpu) !void {
    const channels = 3;
    const x = try filled(g, 5184 * channels, 0.071, 3);
    defer x.deinit();
    const tiled = try g.allocDevice(x.count, .f32);
    defer tiled.deinit();
    const back = try g.allocDevice(x.count, .f32);
    defer back.deinit();
    const into = abi.Tiles{ .channels = channels, .grid = 72, .tile = 24, .direction = 0 };
    var from = into;
    from.direction = 1;
    try g.dispatch(abi.tiles, &.{ x, tiled }, into, Size.of(x.count, 1, 1), group);
    try g.dispatch(abi.tiles, &.{ tiled, back }, from, Size.of(x.count, 1, 1), group);
    for (0..x.count) |i| {
        if (word(back, i) != word(x, i)) return error.SamTilesRoundTrip;
        if (word(tiled, i) != word(x, gridRow(72, 24, i / channels) * channels + i % channels)) return error.SamTilesGather;
    }
    std.debug.print("SAM tile order: gathers by the grid-row map and returns bit for bit\n", .{});
}

fn checkDeconvolution(g: *gpu.Gpu) !void {
    const input = try filled(g, 3 * 5 * 32, 0.03, 1);
    defer input.deinit();
    const raw = try g.alloc(32 * 16 * 4, .f32);
    defer raw.deinit();
    for (raw.slice(f32, raw.count), 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.07) * 0.03;
    const bias = try g.alloc(16, .f32);
    defer bias.deinit();
    for (bias.slice(f32, bias.count), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) / 256;
    const matrix = try g.allocDevice(raw.count, .f32);
    defer matrix.deinit();
    const product = try g.allocDevice(15 * 64, .f32);
    defer product.deinit();
    const scattered = try g.allocDevice(6 * 10 * 16, .f32);
    defer scattered.deinit();
    const reference = try g.allocDevice(scattered.count, .f32);
    defer reference.deinit();
    const shape = abi.ConvShape{ .in_h = 3, .in_w = 5, .in_c = 32, .out_c = 16, .kernel = 2, .out_h = 6, .out_w = 10, .stride = 2, .pad = 0, .first_row = 0, .row_count = 0 };
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    try g.dispatch(abi.deconv_pack, &.{ raw, matrix }, [3]u32{ 16, 32, 4 }, Size.of(raw.count, 1, 1), group);
    try g.matmulInto(input, matrix, product, 15, 64, 32);
    try g.dispatch(abi.deconv_scatter, &.{ product, bias, scattered }, shape, Size.of(scattered.count, 1, 1), group);
    try g.dispatch("im_sam_deconv", &.{ input, raw, bias, reference }, shape, Size.of(reference.count / 4, 1, 1), group);
    try g.endBatch();
    var maximum: f32 = 0;
    for (scattered.slice(f32, scattered.count), reference.slice(f32, reference.count)) |x, y| maximum = @max(maximum, @abs(x - y));
    std.debug.print("SAM transposed convolution: packed product maximum error {e} against the direct kernel\n", .{maximum});
    if (maximum > 1e-4) return error.SamDeconvolutionParity;
}

const ConvCase = struct { h: usize, w: usize, in: usize, out: usize, k: usize, stride: usize, pad: usize };

fn checkConvolution(g: *gpu.Gpu) !void {
    var worst: f64 = 0;
    for ([_]ConvCase{ .{ .h = 9, .w = 7, .in = 5, .out = 6, .k = 3, .stride = 2, .pad = 1 }, .{ .h = 28, .w = 42, .in = 3, .out = 4, .k = 14, .stride = 14, .pad = 0 } }) |case| {
        const oh = (case.h + 2 * case.pad - case.k) / case.stride + 1;
        const ow = (case.w + 2 * case.pad - case.k) / case.stride + 1;
        const width = case.k * case.k * case.in;
        const pixels = oh * ow;
        const x = try filled(g, case.h * case.w * case.in, 0.13, 1);
        defer x.deinit();
        const raw = try filled(g, case.out * width, 0.29, 0.25);
        defer raw.deinit();
        const bias = try filled(g, case.out, 0.9, 0.5);
        defer bias.deinit();
        const weight = try g.allocDevice(raw.count, .f32);
        defer weight.deinit();
        const output = try g.allocDevice(pixels * case.out, .f32);
        defer output.deinit();
        try g.dispatch(abi.conv_pack, &.{ raw, weight }, [3]u32{ @intCast(case.out), @intCast(case.in), @intCast(case.k * case.k) }, Size.of(raw.count, 1, 1), group);
        for ([_][2]usize{ .{ 0, pixels / 2 }, .{ pixels / 2, pixels - pixels / 2 } }) |chunk| {
            const first, const rows = chunk;
            const columns = try g.allocDevice(rows * width, .f32);
            defer columns.deinit();
            const shape = abi.ConvShape{ .in_h = @intCast(case.h), .in_w = @intCast(case.w), .in_c = @intCast(case.in), .out_c = @intCast(case.out), .kernel = @intCast(case.k), .out_h = @intCast(oh), .out_w = @intCast(ow), .stride = @intCast(case.stride), .pad = @intCast(case.pad), .first_row = @intCast(first), .row_count = @intCast(rows) };
            try g.dispatch(abi.unfold, &.{ x, columns }, shape, Size.of(columns.count, 1, 1), group);
            const dst = part(output, first * case.out, rows * case.out);
            try g.matmulInto(columns, weight, dst, rows, case.out, width);
            try g.dispatch(abi.bias_rows, &.{ dst, bias }, [2]u32{ @intCast(dst.count), @intCast(case.out) }, Size.of(dst.count, 1, 1), group);
        }
        for (0..pixels) |pixel| for (0..case.out) |o| {
            var expected: f64 = bias.get(o);
            var magnitude: f64 = @abs(expected);
            for (0..case.k) |ky| for (0..case.k) |kx| for (0..case.in) |c| {
                const y = @as(isize, @intCast(pixel / ow * case.stride + ky)) - @as(isize, @intCast(case.pad));
                const z = @as(isize, @intCast(pixel % ow * case.stride + kx)) - @as(isize, @intCast(case.pad));
                if (y < 0 or z < 0 or y >= @as(isize, @intCast(case.h)) or z >= @as(isize, @intCast(case.w))) continue;
                const term = @as(f64, x.get((@as(usize, @intCast(y)) * case.w + @as(usize, @intCast(z))) * case.in + c)) * raw.get(((o * case.in + c) * case.k + ky) * case.k + kx);
                expected += term;
                magnitude += @abs(term);
            };
            const err = @abs(expected - output.get(pixel * case.out + o));
            if (err > 8 * unit * magnitude) return error.SamConvolutionParity;
            if (magnitude > 0) worst = @max(worst, err / (8 * unit * magnitude));
        };
    }
    std.debug.print("SAM convolution (unfold, product, bias rows): worst float64 error {d:.3} of its bound\n", .{worst});
}

fn checkDepthwise(g: *gpu.Gpu) !void {
    const h = 10;
    const w = 9;
    const channels = 6;
    const k = 7;
    const pad = 3;
    const x = try filled(g, h * w * channels, 0.19, 1.25);
    defer x.deinit();
    const weight = try filled(g, channels * k * k, 0.41, 0.5);
    defer weight.deinit();
    const bias = try filled(g, channels, 1.3, 0.75);
    defer bias.deinit();
    const out = try g.allocDevice(x.count, .f32);
    defer out.deinit();
    const shape = abi.ConvShape{ .in_h = h, .in_w = w, .in_c = channels, .out_c = channels, .kernel = k, .out_h = h, .out_w = w, .stride = 1, .pad = pad, .first_row = 0, .row_count = 0 };
    try g.dispatch(abi.depthwise, &.{ x, weight, bias, out }, shape, Size.of(out.count, 1, 1), group);
    var worst: f64 = 0;
    for (0..h) |y| for (0..w) |z| for (0..channels) |c| {
        var expected: f64 = bias.get(c);
        var magnitude: f64 = @abs(expected);
        for (0..k) |ky| for (0..k) |kx| {
            const sy = @as(isize, @intCast(y + ky)) - pad;
            const sz = @as(isize, @intCast(z + kx)) - pad;
            if (sy < 0 or sz < 0 or sy >= h or sz >= w) continue;
            const term = @as(f64, x.get((@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sz))) * channels + c)) * weight.get((c * k + ky) * k + kx);
            expected += term;
            magnitude += @abs(term);
        };
        const err = @abs(expected - out.get((y * w + z) * channels + c));
        if (err > 8 * unit * magnitude) return error.SamDepthwiseParity;
        if (magnitude > 0) worst = @max(worst, err / (8 * unit * magnitude));
    };
    std.debug.print("SAM depthwise 7x7: worst float64 error {d:.3} of its bound\n", .{worst});
}

fn checkGroupNorm(g: *gpu.Gpu) !void {
    const pixels = 37;
    const channels = 32;
    const groups = 8;
    const x = try filled(g, pixels * channels, 0.053, 2);
    defer x.deinit();
    for (x.slice(f32, x.count)) |*value| value.* += 0.375;
    const weight = try filled(g, channels, 0.7, 1);
    defer weight.deinit();
    const bias = try filled(g, channels, 1.1, 0.5);
    defer bias.deinit();
    const out = try g.allocDevice(x.count, .f32);
    defer out.deinit();
    try g.dispatch(abi.group_norm, &.{ x, weight, bias, out }, [3]u32{ pixels, channels, groups }, Size.of(groups * 256, 1, 1), group);
    const width = channels / groups;
    var maximum: f64 = 0;
    for (0..groups) |index| {
        var mean: f64 = 0;
        for (0..pixels) |p| for (0..width) |c| {
            mean += x.get(p * channels + index * width + c);
        };
        mean /= pixels * width;
        var variance: f64 = 0;
        for (0..pixels) |p| for (0..width) |c| {
            const centred = x.get(p * channels + index * width + c) - mean;
            variance += centred * centred;
        };
        variance /= pixels * width;
        for (0..pixels) |p| for (0..width) |c| {
            const at = p * channels + index * width + c;
            const expected = (x.get(at) - mean) / @sqrt(variance + 1e-5) * weight.get(index * width + c) + bias.get(index * width + c);
            maximum = @max(maximum, @abs(expected - out.get(at)));
        };
    }
    std.debug.print("SAM group normalization: maximum float64 error {e}\n", .{maximum});
    if (!(maximum <= 2e-5)) return error.SamGroupNormParity;
}

fn squashOracle(logit: f32, active: bool, threshold: bool, gain: f32, offset: f32) f64 {
    if (!active) return 0;
    const value: f64 = logit;
    const squashed: f64 = if (threshold) @floatFromInt(@intFromBool(value > 0)) else 1 / (1 + @exp(-value));
    return squashed * gain + offset;
}

fn channelFillOracle(channel: usize, first: usize, count: usize, bits: u32, previous: f32) f32 {
    if (channel < first or channel >= first + count) return previous;
    return @floatFromInt((bits >> @intCast(channel - first)) & 1);
}

fn checkSquash(g: *gpu.Gpu) !void {
    const pixels = 7;
    const sentinel: f32 = -12345.5;
    var maximum: f64 = 0;
    for ([_][3]u32{ .{ 1, 1, 1 }, .{ 16, 16, 32 } }) |layout| {
        const sources, const slots, const stride = layout;
        const logits = try g.alloc(pixels * sources, .f32);
        defer logits.deinit();
        for (logits.slice(f32, logits.count), 0..) |*value, i| value.* = (@as(f32, @floatFromInt(i % 29)) - 14) / 3;
        const out = try g.alloc(pixels * stride, .f32);
        defer out.deinit();
        for ([_]u32{ 1, 0x8002, 0xffff }) |bits| for (0..2) |threshold| for (0..2) |accurate| for ([_][2]f32{ .{ 12.5, -6.25 }, .{ -0.75, 0.5 }, .{ 1, 0 } }) |affine| {
            @memset(out.slice(f32, out.count), sentinel);
            const params = abi.Squash{ .pixels = pixels, .stride = stride, .slots = slots, .sources = sources, .active = bits, .threshold = @intCast(threshold), .accurate = @intCast(accurate), .gain = affine[0], .offset = affine[1] };
            try g.dispatch(abi.mask_squash, &.{ logits, out }, params, Size.of(pixels * slots, 1, 1), group);
            for (0..pixels) |p| for (0..stride) |c| {
                const got = out.get(p * stride + c);
                if (c >= slots) {
                    if (got != sentinel) return error.SamSquashWroteChannel;
                    continue;
                }
                const active = (bits >> @intCast(c)) & 1 != 0;
                const expected = squashOracle(logits.get(p * sources + c), active, threshold == 1, affine[0], affine[1]);
                const err = @abs(expected - got);
                if (threshold == 1 and err != 0) return error.SamSquashThreshold;
                if (!(err <= 4e-6)) return error.SamSquashParity;
                maximum = @max(maximum, err);
            };
        };
    }
    std.debug.print("SAM mask squash: threshold mode exact, sigmoid modes maximum float64 error {e}\n", .{maximum});
}

fn checkChannelFill(g: *gpu.Gpu) !void {
    const pixels = 7;
    const channels = 32;
    const out = try g.alloc(pixels * channels, .f32);
    defer out.deinit();
    for ([_][3]u32{ .{ 16, 16, 0x8001 }, .{ 0, 1, 1 }, .{ 3, 5, 0b10110 }, .{ 16, 16, 0xffff } }) |fill| {
        const first, const count, const bits = fill;
        for (out.slice(f32, out.count), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) + 0.5;
        const params = abi.ChannelFill{ .pixels = pixels, .channels = channels, .first = first, .count = count, .bits = bits };
        try g.dispatch(abi.channel_fill, &.{out}, params, Size.of(pixels * count, 1, 1), group);
        for (0..pixels) |p| for (0..channels) |c| {
            const previous = @as(f32, @floatFromInt(p * channels + c)) + 0.5;
            const expected: f32 = channelFillOracle(c, first, count, bits, previous);
            if (@as(u32, @bitCast(expected)) != word(out, p * channels + c)) return error.SamChannelFillParity;
        };
    }
    std.debug.print("SAM channel fill: indicator channels exact, other channels untouched\n", .{});
}

fn checkFirstMaskConv(g: *gpu.Gpu) !void {
    const x = try g.alloc(1152 * 1152, .f32);
    defer x.deinit();
    const w = try g.alloc(16 * 32 * 9, .f32);
    defer w.deinit();
    const bias = try g.alloc(16, .f32);
    defer bias.deinit();
    const out = try g.allocDevice(576 * 576 * 16, .f32);
    defer out.deinit();
    for (x.slice(f32, x.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.017);
    for (w.slice(f32, w.count), 0..) |*value, i| value.* = @as(f32, @floatCast(@as(f16, @floatCast(@cos(@as(f32, @floatFromInt(i)) * 0.037) * 0.1))));
    for (bias.slice(f32, bias.count), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) / 256;
    var maximum: f64 = 0;
    for (0..2) |prompted| {
        try g.dispatch(abi.mask_first_conv, &.{ x, w, bias, out }, @as(u32, @intCast(prompted)), Size.of(out.count, 1, 1), group);
        for ([_]usize{ 0, 1, 575, 576, 577, 1151, 576 * 288 + 288, 576 * 576 - 1 }) |pixel| for (0..16) |c| {
            var expected: f64 = bias.get(c);
            for (0..3) |ky| for (0..3) |kx| {
                const y = @as(isize, @intCast(pixel / 576 * 2 + ky)) - 1;
                const z = @as(isize, @intCast(pixel % 576 * 2 + kx)) - 1;
                if (y < 0 or z < 0 or y >= 1152 or z >= 1152) continue;
                expected += @as(f64, x.get(@intCast(y * 1152 + z))) * w.get(c * 32 * 9 + ky * 3 + kx);
                if (prompted == 1) expected += w.get((c * 32 + 16) * 9 + ky * 3 + kx);
            };
            maximum = @max(maximum, @abs(expected - out.get(pixel * 16 + c)));
        };
    }
    std.debug.print("SAM first mask convolution: maximum float64 oracle error {e}\n", .{maximum});
    if (maximum > 1e-6) return error.SamConvolutionParity;
}

test "SAM squash oracle maps hand values without model constants" {
    try std.testing.expectEqual(@as(f64, 0), squashOracle(0, true, false, 2, -1));
    try std.testing.expectEqual(@as(f64, -1), squashOracle(0, true, true, 2, -1));
    try std.testing.expectEqual(@as(f64, 1), squashOracle(0.5, true, true, 2, -1));
    try std.testing.expectEqual(@as(f64, 0), squashOracle(3, false, true, 2, -1));
    try std.testing.expectEqual(@as(f64, 0.25), squashOracle(std.math.nan(f32), true, true, -4, 0.25));
    try std.testing.expect(std.math.isNan(squashOracle(std.math.nan(f32), true, false, 2, -1)));
}

test "SAM channel-fill oracle writes bits inside its range and keeps other channels" {
    try std.testing.expectEqual(@as(f32, 1), channelFillOracle(16, 16, 16, 0x8001, 7));
    try std.testing.expectEqual(@as(f32, 0), channelFillOracle(17, 16, 16, 0x8001, 7));
    try std.testing.expectEqual(@as(f32, 1), channelFillOracle(31, 16, 16, 0x8001, 7));
    try std.testing.expectEqual(@as(f32, 7), channelFillOracle(15, 16, 16, 0xffff, 7));
    try std.testing.expectEqual(@as(f32, 7), channelFillOracle(32, 16, 16, 0xffff, 7));
}
