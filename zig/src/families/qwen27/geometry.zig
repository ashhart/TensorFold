//! Dense geometries use the existing runtime projections and fixed attention and DeltaNet head arithmetic.
const std = @import("std");
const cfg = @import("config.zig");

pub const Shape = struct { n: usize, k: usize };

/// The fused serving matrices, including the vocabulary head; no captured per-model projection table is needed.
pub fn projections(c: cfg.Config) [9]Shape {
    return .{
        .{ .n = 2 * c.intermediate, .k = c.hidden },
        .{ .n = c.hidden, .k = c.intermediate },
        .{ .n = c.gdnQkvDim(), .k = c.hidden },
        .{ .n = c.zbaDim(), .k = c.hidden },
        .{ .n = c.hidden, .k = c.gdnValueDim() },
        .{ .n = c.qDim(), .k = c.hidden },
        .{ .n = 2 * c.kvDim(), .k = c.hidden },
        .{ .n = c.hidden, .k = c.heads * c.head_dim },
        .{ .n = c.vocab, .k = c.hidden },
    };
}

/// Sizes are read from config; retain the kernel's head layouts, 64-value input groups and 32-row fused tiles.
pub fn supported(c: cfg.Config) !void {
    if (c.layers == 0 or c.layers > cfg.max_layers or c.head_dim != 256 or c.dk != 128 or c.dv != 128 or
        c.conv_kernel != 4 or c.rope_dims != 64 or c.k_heads == 0 or c.kv_heads == 0 or
        c.heads == 0 or c.heads % c.kv_heads != 0 or c.v_heads == 0 or c.v_heads % c.k_heads != 0)
        return error.UnqualifiedQwenShape;
    if (c.hidden == 0 or c.intermediate == 0 or c.vocab == 0 or c.hidden % 64 != 0 or c.intermediate % 64 != 0)
        return error.UnqualifiedQwenShape;
    for (projections(c)) |p| {
        if (p.n == 0 or p.k == 0 or p.n > std.math.maxInt(u32) or p.k > std.math.maxInt(u32) or
            p.n % 32 != 0 or p.k % 64 != 0) return error.UnqualifiedQwenShape;
    }
    for (c.kinds[0..c.layers], 0..) |kind, i| {
        if (kind != (if ((i + 1) % 4 == 0) cfg.Kind.attention else cfg.Kind.linear))
            return error.UnqualifiedLayerSchedule;
    }
}
