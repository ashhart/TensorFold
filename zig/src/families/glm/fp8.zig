//! The FP8 checkpoint's routed experts as stored (E4M3 codes, f32 128 x 128 block scales): loading and kernels.
const std = @import("std");
const affine = @import("affine.zig");
const quant = @import("quant.zig");
const prefix = @import("weights.zig").prefix;

/// glm_rows.metal's FP8 routed-expert kernels, in `names` order.
pub const Kernel = enum { moe, moe_glu, moe_u, moe_glu_u, sorted, glu_sorted, ids, glu_ids };
pub const names = [_][:0]const u8{
    "glm_moe_rows_f8_s8", "glm_moe_rows_glu_f8_s8", "glm_moe_rows_u_f8_s8", "glm_moe_rows_glu_u_f8_s8",
    "glm_rows_sorted_f8", "glm_rows_glu_sorted_f8", "glm_rows_ids_f8",      "glm_rows_glu_ids_f8",
};

/// Whether layer i's expert e's `proj` is stored as the FP8 checkpoint's codes and block scales.
pub fn isExpert(l: anytype, i: usize, e: usize, proj: []const u8) bool {
    return l.has(prefix ++ "layers.{d}.mlp.experts.{d}.{s}.weight_scale_inv", .{ i, e, proj });
}

/// Experts [lo, lo + e) of layer i's `proj` [n, k]: codes stacked [e, n, k], then scales [e, n / 128, k / 128].
pub fn experts(l: anytype, i: usize, proj: []const u8, n: usize, k: usize, whole: bool, lo: usize, e: usize) !affine.Affine {
    var first: [200]u8 = undefined;
    const name0 = try std.fmt.bufPrint(&first, prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, lo, proj });
    if (!whole) {
        quant.say("glm: {s} is FP8, which loads whole on one Mac", .{name0});
        return error.UnsupportedWidth;
    }
    const sb = affine.fp8Scales(n, k) orelse {
        quant.say("glm: {s} is [{d}, {d}], not whole 128 x 128 scale blocks", .{ name0, n, k });
        return error.UnexpectedTensor;
    };
    const w = l.take(e * n * k);
    const out: affine.Affine = .{ .w = w, .s = l.take(e * sb), .b = l.take(0), .n = @intCast(n), .k = @intCast(k), .bits = 8, .group = affine.fp8_block, .fp8 = true };
    for (0..e) |x| {
        var buf: [200]u8 = undefined;
        const module = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, lo + x, proj });
        if (!isExpert(l, i, lo + x, proj)) {
            quant.say("glm: {s} is FP8 and {s} is not; one stack holds both", .{ name0, module });
            return error.MixedWidths;
        }
        var a: [240]u8 = undefined;
        var b: [240]u8 = undefined;
        const wn = try std.fmt.bufPrint(&a, "{s}.weight", .{module});
        const sn = try std.fmt.bufPrint(&b, "{s}.weight_scale_inv", .{module});
        const ws = l.names.get(wn) orelse return error.MissingTensor;
        const ss = l.names.get(sn).?;
        try @TypeOf(l.*).expect(ws, wn, .f8_e4m3, &.{ n, k });
        try @TypeOf(l.*).expect(ss, sn, .f32, &.{ n / affine.fp8_block, k / affine.fp8_block });
        try l.copyTo(ws, out.w.at(x * n * k));
        try l.copyTo(ss, out.s.at(x * sb));
    }
    return out;
}
