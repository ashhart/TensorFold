//! ModelOpt's weight-only formats, NVFP4 and FP8, kept as stored with a product kernel decoding each in flight.

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

/// Why the reader refused a tensor, kept for the caller's log line (tests read it instead).
pub const Why = @import("core").modelopt.Why;

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

/// `key`'s projection: ModelOpt's NVFP4 and FP8 stay as stored, their kernels decoding in flight.
pub fn read(a: Allocator, t: *const table.Table, key: []const u8, what: Algo) !host.Projection {
    var why: Why = .{};
    return readWhy(a, t, key, what, &why) catch |e| {
        if (why.len > 0) std.log.err("{s}", .{why.text()});
        return e;
    };
}

/// `read`, the refusal kept in `why` instead of the log (tests read it).
pub fn readWhy(a: Allocator, t: *const table.Table, key: []const u8, what: Algo, why: *Why) !host.Projection {
    return switch (what) {
        .nvfp4 => .{ .nvfp4 = try quant.nvfp4.read(a, t, key) },
        .fp8 => .{ .fp8 = try fp8(a, t, key, why) },
    };
}

fn fp8(a: Allocator, t: *const table.Table, key: []const u8, why: *Why) !host.StoredFp8 {
    var b: [2][320]u8 = undefined;
    const wname = name(&b[0], key, ".weight") orelse return error.MissingTensor;
    const sname = name(&b[1], key, ".weight_scale") orelse return error.MissingTensor;
    const w = try t.get(wname);
    const s = try t.get(sname);
    if (w.dtype != .f8_e4m3 or w.rank != 2) return refuse(why, wname, "an FP8 weight is E4M3 [N, K]");
    if (s.dtype != .f32) return refuse(why, sname, "FP8's weight scale is F32");
    if (s.numel() != 1) return refuse(why, sname, "FP8's weight scale is one number a projection, as ModelOpt writes it");
    const one = convert.load(.f32, s.bytes, 0);
    if (!std.math.isFinite(one) or one == 0) return refuse(why, sname, "FP8's weight scale is zero, infinite or nan");
    for (w.bytes) |byte| if (byte & 0x7f == 0x7f) return refuse(why, wname, "an FP8 weight holds an E4M3 nan code");
    // the codes stay as stored; the product kernel decodes them and applies this scale once to its f32 sum
    const scale = try a.alloc(f32, 1);
    scale[0] = one;
    return .{ .weight = w, .scale = scale };
}

fn refuse(why: *Why, key: []const u8, what: []const u8) error{UnexpectedTensor} {
    why.set("{s}: {s}", .{ key, what });
    return error.UnexpectedTensor;
}
