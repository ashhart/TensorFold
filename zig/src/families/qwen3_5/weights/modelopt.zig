//! ModelOpt's weight-only formats read as BF16 at load: NVFP4 e2m1 codes and FP8 e4m3, each with its scales.

const std = @import("std");
const core = @import("core");
const quant = core.quant;
const bf16_round = core.bf16_round;
const table = @import("table.zig");
const host = @import("host.zig");

const convert = quant.convert;
const Tensor = table.Tensor;
const Allocator = std.mem.Allocator;

/// Inputs sharing one NVFP4 scale, as ModelOpt writes it.
pub const group = quant.nvfp4.group;

/// The ModelOpt algorithms this reader takes.
pub const Algo = enum { nvfp4, fp8 };

/// `key`'s stored algorithm, by the tensors beside it; null when they are not ModelOpt's.
pub fn algo(t: *const table.Table, key: []const u8) ?Algo {
    var w: [320]u8 = undefined;
    var s2: [320]u8 = undefined;
    var s: [320]u8 = undefined;
    if (!t.has(name(&w, key, ".weight") orelse return null)) return null;
    if (t.has(name(&s2, key, ".weight_scale_2") orelse return null)) return .nvfp4;
    if (t.has(name(&s, key, ".weight_scale") orelse return null)) return .fp8;
    return null;
}

fn name(buf: []u8, key: []const u8, suffix: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ key, suffix }) catch null;
}

/// `key`'s projection with its weights dequantized to BF16: the engine's BF16 path runs it.
pub fn read(a: Allocator, t: *const table.Table, key: []const u8, what: Algo) !host.Projection {
    return .{ .bf16 = .{ .weight = switch (what) {
        .nvfp4 => try nvfp4(a, t, key),
        .fp8 => try fp8(a, t, key),
    } } };
}

fn nvfp4(a: Allocator, t: *const table.Table, key: []const u8) !Tensor {
    var b: [3][320]u8 = undefined;
    const w = try t.get(name(&b[0], key, ".weight") orelse return error.MissingTensor);
    const s = try t.get(name(&b[1], key, ".weight_scale") orelse return error.MissingTensor);
    const g = try t.get(name(&b[2], key, ".weight_scale_2") orelse return error.MissingTensor);
    if (w.dtype != .u8 or w.rank != 2) return refuse(b[0][0..], "an NVFP4 weight is U8 [N, K/2]");
    const n = w.shape[0];
    const k = 2 * w.shape[1];
    if (s.rank != 2 or s.shape[0] != n or s.shape[1] * group != k) return refuse(b[1][0..], "NVFP4 scales are [N, K/16]");
    if (g.numel() != 1) return refuse(b[2][0..], "NVFP4's second scale is one number");
    if (s.dtype != .f8_e4m3 and s.dtype != .f32) return refuse(b[1][0..], "NVFP4 scales are E4M3 or F32");
    const global = convert.load(.f32, g.bytes, 0);
    const out = try a.alloc(u16, n * k);
    errdefer a.free(out);
    for (0..n) |row| {
        const per_row = k / group;
        const codes = w.bytes[row * (k / 2) ..][0..(k / 2)];
        nvfp4Row(out[row * k ..][0..k], codes, s, s.dtype == .f32, row * per_row, per_row, global);
    }
    return .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 0, 0, 0 }, .bytes = std.mem.sliceAsBytes(out) };
}

/// One row's `k` bf16 weights from its `k/2` codes, group scales starting at `first`.
fn nvfp4Row(out: []u16, codes: []const u8, scales: Tensor, f32_scales: bool, first: usize, per_row: usize, global: f32) void {
    for (0..out.len) |i| {
        const byte = codes[i / 2];
        const code = if (i % 2 == 0) byte & 0xf else byte >> 4;
        const at = first + i / group;
        std.debug.assert(at < first + per_row);
        const scale: f64 = if (f32_scales) @floatCast(convert.load(.f32, scales.bytes, at)) else quant.fp8.e4m3(scales.bytes[at]);
        out[i] = bf16_round.bits(@floatCast(quant.nvfp4.e2m1(code) * scale * @as(f64, global)));
    }
}

fn fp8(a: Allocator, t: *const table.Table, key: []const u8) !Tensor {
    var b: [2][320]u8 = undefined;
    const w = try t.get(name(&b[0], key, ".weight") orelse return error.MissingTensor);
    const s = try t.get(name(&b[1], key, ".weight_scale") orelse return error.MissingTensor);
    if (w.dtype != .f8_e4m3 or w.rank != 2) return refuse(b[0][0..], "an FP8 weight is E4M3 [N, K]");
    const n = w.shape[0];
    const k = w.shape[1];
    if (s.dtype != .f32) return refuse(b[1][0..], "FP8's weight scale is F32");
    if (s.numel() != 1 and s.numel() != n) return refuse(b[1][0..], "FP8's weight scale is one number or one a row");
    const out = try a.alloc(u16, n * k);
    errdefer a.free(out);
    for (0..n) |row| {
        const scale = convert.load(.f32, s.bytes, if (s.numel() == 1) 0 else row);
        for (0..k) |i| out[row * k + i] = bf16_round.bits(@floatCast(quant.fp8.e4m3(w.bytes[row * k + i]) * @as(f64, scale)));
    }
    return .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 0, 0, 0 }, .bytes = std.mem.sliceAsBytes(out) };
}

fn refuse(key: []const u8, what: []const u8) error{UnexpectedTensor} {
    return quant.types.refuse(key, what);
}

test "an NVFP4 row dequantizes through e2m1, its group scales and the global one" {
    var codes: [8]u8 = @splat(0);
    codes[0] = 0x21; // input 0 is code 1 (+0.5), input 1 is code 2 (+1.0)
    const e4m3 = [_]u8{ 0x38, 0x38 }; // 1.0 each
    const scales: Tensor = .{ .dtype = .f8_e4m3, .rank = 1, .shape = .{ e4m3.len, 0, 0, 0, 0 }, .bytes = &e4m3 };
    var out: [16]u16 = @splat(0);
    nvfp4Row(&out, &codes, scales, false, 0, 2, 0.25);
    try std.testing.expectEqual(@as(f32, 0.125), fromBf16(out[0]));
    try std.testing.expectEqual(@as(f32, 0.25), fromBf16(out[1]));
    try std.testing.expectEqual(@as(f32, 0), fromBf16(out[2]));
    try std.testing.expectEqual(@as(f32, 0), fromBf16(out[15]));
}

fn fromBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
