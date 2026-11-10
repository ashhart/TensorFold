//! Host-only source checks use the same effective parameter values that the primary CUDA loader uploads.
const std = @import("std");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const plans = @import("cuda_nvfp4_source_plan.zig");

pub fn sameNorm(mlx: []const u8, hf: []const u8) bool {
    if (mlx.len != hf.len or mlx.len % 2 != 0) return false;
    var at: usize = 0;
    while (at < mlx.len) : (at += 2) {
        const a = std.mem.readInt(u16, mlx[at..][0..2], .little);
        const b = std.mem.readInt(u16, hf[at..][0..2], .little);
        const x: f32 = @bitCast(@as(u32, a) << 16);
        const y: f32 = @bitCast(@as(u32, b) << 16);
        if (!std.math.isFinite(x) or !std.math.isFinite(y) or a != core.bf16_round.bits(y + 1)) return false;
    }
    return true;
}

fn word(t: anytype, index: usize, narrow: bool) !u32 {
    const u: u32 = if (t.dtype == .bf16) @as(u32, std.mem.readInt(u16, t.bytes[index * 2 ..][0..2], .little)) << 16 else if (t.dtype == .f32)
        std.mem.readInt(u32, t.bytes[index * 4 ..][0..4], .little)
    else
        return error.Nvfp4MlpBaseMismatch;
    if (!std.math.isFinite(@as(f32, @bitCast(u))) or (narrow and u & 0xffff != 0)) return error.Nvfp4MlpBaseMismatch;
    return u;
}

fn sameVector(a: anytype, b: anytype, n: usize, narrow: bool) !void {
    if (!a.is(a.dtype, &.{n}) or !b.is(b.dtype, &.{n})) return error.Nvfp4MlpBaseMismatch;
    for (0..n) |i| if (try word(a, i, narrow) != try word(b, i, narrow)) return error.Nvfp4MlpBaseMismatch;
}

fn raw(a: anytype, b: anytype) !void {
    if (!std.mem.eql(u8, a.bytes, b.bytes)) return error.Nvfp4MlpBaseMismatch;
    for (0..a.bytes.len / 2) |i| _ = try word(a, i, false);
}

fn scalar(ck: anytype, name: []const u8) !void {
    const t = try ck.get(name);
    if (t.dtype != .f32 or t.numel() != 1 or t.bytes.len != 4) return error.UnexpectedTensor;
    const v: f32 = @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
    const inverse = 1.0 / v;
    if (!(v > 0) or !std.math.isFinite(v) or !std.math.isFinite(inverse) or !std.math.isFinite(1.0 / inverse))
        return error.BadNativeNvfp4Scale;
}

pub fn validate(original: anytype, source: anytype, g: geometry.Geometry, plan: plans.Plan) !void {
    var left: [192]u8 = undefined;
    var right: [192]u8 = undefined;
    for (0..g.layers) |i| {
        for ([_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |norm| {
            const a = try original.expect(try std.fmt.bufPrint(&left, "language_model.model.layers.{d}.{s}.weight", .{ i, norm }), .bf16, &.{g.hidden});
            const b = try source.expect(try std.fmt.bufPrint(&right, "model.language_model.layers.{d}.{s}.weight", .{ i, norm }), .bf16, &.{g.hidden});
            if (!sameNorm(a.bytes, b.bytes)) return error.Nvfp4MlpBaseMismatch;
        }
        if (plan.scope == .layers) {
            if (g.linear[i]) {
                const old = try std.fmt.bufPrint(&left, "language_model.model.layers.{d}.linear_attn.", .{i});
                const new = try std.fmt.bufPrint(&right, "model.language_model.layers.{d}.linear_attn.", .{i});
                try delta(original, source, g, old, new);
            } else for ([_][]const u8{ "q_norm", "k_norm" }) |norm| {
                const a = try original.expect(try std.fmt.bufPrint(&left, "language_model.model.layers.{d}.self_attn.{s}.weight", .{ i, norm }), .bf16, &.{geometry.head_dim});
                const b = try source.expect(try std.fmt.bufPrint(&right, "model.language_model.layers.{d}.self_attn.{s}.weight", .{ i, norm }), .bf16, &.{geometry.head_dim});
                if (!sameNorm(a.bytes, b.bytes)) return error.Nvfp4MlpBaseMismatch;
            }
        }
        for (weights.slots(g, i)) |slot| if (plan.kind(i, slot)) |kind_| {
            const nk = weights.shape(g, slot);
            const prefix = try std.fmt.bufPrint(&right, "model.language_model.layers.{d}.{s}", .{ i, geometry.slot_modules.get(slot) });
            try projection(source, prefix, nk[0], nk[1], kind_);
        };
    }
    const a = try original.expect("language_model.model.norm.weight", .bf16, &.{g.hidden});
    const b = try source.expect("model.language_model.norm.weight", .bf16, &.{g.hidden});
    if (!sameNorm(a.bytes, b.bytes)) return error.Nvfp4MlpBaseMismatch;
}

fn delta(original: anytype, source: anytype, g: geometry.Geometry, old: []const u8, new: []const u8) !void {
    var left: [192]u8 = undefined;
    var right: [192]u8 = undefined;
    const a = try original.expect(try std.fmt.bufPrint(&left, "{s}conv1d.weight", .{old}), .bf16, &.{ g.convWidth(), geometry.taps, 1 });
    const b = try source.expect(try std.fmt.bufPrint(&right, "{s}conv1d.weight", .{new}), .bf16, &.{ g.convWidth(), 1, geometry.taps });
    try raw(a, b);
    try raw(try original.expect(try std.fmt.bufPrint(&left, "{s}norm.weight", .{old}), .bf16, &.{geometry.dk}), try source.expect(try std.fmt.bufPrint(&right, "{s}norm.weight", .{new}), .bf16, &.{geometry.dk}));
    for ([_][]const u8{ "A_log", "dt_bias" }) |name| {
        const av = try original.get(try std.fmt.bufPrint(&left, "{s}{s}", .{ old, name }));
        const bv = try source.get(try std.fmt.bufPrint(&right, "{s}{s}", .{ new, name }));
        try sameVector(av, bv, g.v_heads, std.mem.eql(u8, name, "dt_bias"));
    }
}

pub fn projection(ck: anytype, prefix: []const u8, n: usize, k: usize, kind: geometry.Kind) !void {
    var buf: [224]u8 = undefined;
    if (kind == .bf16) {
        _ = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}), .bf16, &.{ n, k });
        return;
    }
    if (kind != .nvfp4) return error.UnsupportedNvfp4MlpSource;
    const ct = ck.has(try std.fmt.bufPrint(&buf, "{s}.weight_packed", .{prefix}));
    _ = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, if (ct) "weight_packed" else "weight" }), .u8, &.{ n, k / 2 });
    _ = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{prefix}), .f8_e4m3, &.{ n, k / 16 });
    try scalar(ck, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, if (ct) "weight_global_scale" else "weight_scale_2" }));
    try scalar(ck, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, if (ct) "input_global_scale" else "input_scale" }));
}
