//! Output projections a lesson folded into the shards as bf16 (before living weights), loaded as 4-bit codes and rest.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const host4 = @import("../../core/affine4_host.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const learn_shape = @import("learn_shape.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Each layer's bf16 [out, in] residual, its weight minus its codes' values: the forward adds it after the projection.
pub const Residuals = [cfg.max_layers]?mtl.Buffer;

pub const moduleName = learn_shape.moduleName;
pub const inWidth = learn_shape.inWidth;

/// Every layer whose output projection is bf16 gets codes, scales and biases in the checkpoint, and its residual.
pub fn split(gpa: std.mem.Allocator, device: mtl.Device, ck: *ckpt.Checkpoint, c: cfg.Config) !Residuals {
    var out: Residuals = @splat(null);
    errdefer free(out);
    for (0..c.layers) |i| {
        var name: [160]u8 = undefined;
        const base = try moduleName(&name, c, i);
        var full: [192]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&full, "{s}.weight", .{base}));
        if (w.dtype != .bf16) continue;
        if (w.rank != 2 or w.shape[0] != c.hidden or w.shape[1] != inWidth(c, i)) return error.UnexpectedTensor;
        out[i] = try splitOne(gpa, device, ck, base, w);
    }
    return out;
}

fn splitOne(gpa: std.mem.Allocator, device: mtl.Device, ck: *ckpt.Checkpoint, base: []const u8, w: ckpt.Tensor) !mtl.Buffer {
    const n = w.count();
    const values = try gpa.alloc(f32, n);
    defer gpa.free(values);
    for (values, w.host(u16)) |*v, b| v.* = host4.f32of(b);
    const groups = n / host4.group;
    const codes = try device.buffer(n / 2 + 4 * groups, opts);
    ck.own(codes, n / 2 + 4 * groups) catch |e| {
        codes.deinit();
        return e;
    };
    const words = codes.slice(u32, n / 8);
    const sb = @as([*]u16, @ptrCast(@alignCast(codes.contents() + n / 2)))[0 .. 2 * groups];
    host4.quantize(values, words, sb[0..groups], sb[groups..]);
    const rest = try device.buffer(n * 2, opts);
    errdefer rest.deinit();
    const back = try gpa.alloc(f32, n);
    defer gpa.free(back);
    host4.dequantize(words, sb[0..groups], sb[groups..], back);
    for (rest.slice(u16, n), values, back) |*r, v, q| r.* = host4.bf16of(v - q);
    const d = w.shape[0];
    const k = w.shape[1];
    var full: [192]u8 = undefined;
    try ck.set(try std.fmt.bufPrint(&full, "{s}.weight", .{base}), .{ .buffer = codes, .offset = 0, .bytes = n / 2, .dtype = .u32, .shape = .{ d, k / 8, 1, 1, 1 }, .rank = 2 });
    try ck.set(try std.fmt.bufPrint(&full, "{s}.scales", .{base}), .{ .buffer = codes, .offset = n / 2, .bytes = 2 * groups, .dtype = .bf16, .shape = .{ d, k / 64, 1, 1, 1 }, .rank = 2 });
    try ck.set(try std.fmt.bufPrint(&full, "{s}.biases", .{base}), .{ .buffer = codes, .offset = n / 2 + 2 * groups, .bytes = 2 * groups, .dtype = .bf16, .shape = .{ d, k / 64, 1, 1, 1 }, .rank = 2 });
    return rest;
}

/// The residuals into their layers, the weights owning them from here (freed here when that fails).
pub fn attach(w: *wts.Weights, residuals: Residuals) !void {
    var held: usize = 0;
    for (residuals) |r| held += @intFromBool(r != null);
    w.owned.ensureUnusedCapacity(w.allocator, held) catch |e| {
        free(residuals);
        return e;
    };
    for (residuals, 0..) |r, i| if (r) |b| {
        w.owned.appendAssumeCapacity(b);
        switch (w.layers[i]) {
            .moe => |*m| m.slide = b,
            .mamba => |*m| m.slide = b,
            .attention => |*a| a.slide = b,
        }
    };
}

/// Layer i's residual, if its projection was loaded from bf16.
pub fn residual(w: *const wts.Weights, i: usize) ?mtl.Buffer {
    return switch (w.layers[i]) {
        .moe => |m| m.slide,
        .mamba => |m| m.slide,
        .attention => |a| a.slide,
    };
}

pub fn free(residuals: Residuals) void {
    for (residuals) |r| if (r) |b| b.deinit();
}
