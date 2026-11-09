//! ModelOpt's weight-only formats: NVFP4 kept as stored, the product kernel decoding it in flight; FP8 widened to BF16.

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

/// `key`'s projection: ModelOpt's NVFP4 stays as stored (the kernel decodes e2m1 in flight); FP8 widens to BF16.
pub fn read(a: Allocator, t: *const table.Table, key: []const u8, what: Algo) !host.Projection {
    return switch (what) {
        .nvfp4 => .{ .nvfp4 = try quant.nvfp4.read(a, t, key) },
        .fp8 => .{ .bf16 = .{ .weight = try fp8(a, t, key) } },
    };
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
    for (0..s.numel()) |i| {
        const scale = convert.load(.f32, s.bytes, i);
        if (!std.math.isFinite(scale)) return refuse(b[1][0..], "FP8's weight scale is infinite or nan");
    }
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
