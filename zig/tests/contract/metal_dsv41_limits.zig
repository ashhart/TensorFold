//! DeepSeek-V4.1's gate-5 launchOf branches, tried in the order metal_limits held them.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const limits = @import("contract").limits;
const Launch = @import("metal_limits.zig").Launch;

/// The dsv41 launch's sources, or null when the name belongs to another family.
pub fn launchOf(a: std.mem.Allocator, device: mtl.Device, d: limits.MetalLaunch, fun: []const u8, comptime assemble: bool) !?Launch {
    _ = device;
    if (std.mem.startsWith(u8, d.name, "dsv41 xmma ")) {
        if (!assemble) return @as(Launch, undefined);
        const b = try tf.dsv41_metal.xmma.source(a, "", &.{});
        return .{ .decl = d, .function = fun, .here = b, .pre_m5 = b };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 fp8 mma ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.fp8_mma, .pre_m5 = sources.dsv41.fp8_mma };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 attn ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.csa2_attn, .pre_m5 = sources.dsv41.csa2_attn };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 topk512 ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.topk512, .pre_m5 = sources.dsv41.topk512 };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 ix ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.index, .pre_m5 = sources.dsv41.index };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 experts ")) {
        if (!assemble) return @as(Launch, undefined);
        const body = try tf.dsv41_metal.experts.source(a, "", &.{});
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 route ")) {
        if (!assemble) return @as(Launch, undefined);
        const e: usize = if (std.mem.indexOf(u8, d.name, "backbone") != null) 384 else 128;
        const k: usize = if (e == 384) 6 else 3;
        const body = try std.fmt.allocPrint(a, "#define TF_DSV41_E {d}\n#define TF_DSV41_TOPK {d}\n{s}", .{ e, k, sources.dsv41.route });
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 engram ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.engram, .pre_m5 = sources.dsv41.engram };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 glue_tmp ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.glue_tmp, .pre_m5 = sources.dsv41.glue_tmp };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 fused ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.fused, .pre_m5 = sources.dsv41.fused };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 glue ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.glue, .pre_m5 = sources.dsv41.glue };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 cand ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.cand, .pre_m5 = sources.dsv41.cand };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 compress ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.compress, .pre_m5 = sources.dsv41.compress };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 mv ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.mv, .pre_m5 = sources.dsv41.mv };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 vision ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.vision, .pre_m5 = sources.dsv41.vision };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 fold ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.fold, .pre_m5 = sources.dsv41.fold };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 hc ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.hc, .pre_m5 = sources.dsv41.hc };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 fp8 rows ")) {
        if (!assemble) return @as(Launch, undefined);
        return .{ .decl = d, .function = fun, .here = sources.dsv41.fp8_rows, .pre_m5 = sources.dsv41.fp8_rows };
    }
    return null;
}
