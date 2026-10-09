//! A shared-expert down projection baked to bf16, loaded as the kernels' 4-bit codes plus what they miss.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const affine4 = @import("../../core/affine4.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Each layer's bf16 [D, W] residual, W minus its codes' values: the forward adds it as it adds a learned change.
pub const Residuals = [cfg.max_layers]?mtl.Buffer;

/// Layer i's shared-expert down projection, as the checkpoint names its module.
pub fn downName(buf: []u8, i: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "backbone.layers.{d}.mixer.shared_experts.down_proj", .{i});
}

/// Every MoE layer whose down projection is bf16 gets codes, scales and biases in the checkpoint, and its residual.
pub fn split(gpa: std.mem.Allocator, device: mtl.Device, ck: *ckpt.Checkpoint, c: cfg.Config) !Residuals {
    var out: Residuals = @splat(null);
    errdefer free(out);
    for (0..c.layers) |i| {
        if (c.kinds[i] != .moe) continue;
        var name: [160]u8 = undefined;
        const base = try downName(&name, i);
        var full: [192]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&full, "{s}.weight", .{base}));
        if (w.dtype != .bf16) continue;
        if (w.rank != 2 or w.shape[0] != c.hidden or w.shape[1] != c.shared_width) return error.UnexpectedTensor;
        out[i] = try splitOne(gpa, device, ck, base, w);
    }
    return out;
}

fn splitOne(gpa: std.mem.Allocator, device: mtl.Device, ck: *ckpt.Checkpoint, base: []const u8, w: ckpt.Tensor) !mtl.Buffer {
    const n = w.count();
    const values = try gpa.alloc(f32, n);
    defer gpa.free(values);
    for (values, w.host(u16)) |*v, b| v.* = affine4.f32of(b);
    const groups = n / affine4.group;
    const codes = try device.buffer(n / 2 + 4 * groups, opts);
    ck.own(codes, n / 2 + 4 * groups) catch |e| {
        codes.deinit();
        return e;
    };
    const words = codes.slice(u32, n / 8);
    const sb = @as([*]u16, @ptrCast(@alignCast(codes.contents() + n / 2)))[0 .. 2 * groups];
    affine4.quantize(values, words, sb[0..groups], sb[groups..]);
    const residual = try device.buffer(n * 2, opts);
    errdefer residual.deinit();
    const back = try gpa.alloc(f32, n);
    defer gpa.free(back);
    affine4.dequantize(words, sb[0..groups], sb[groups..], back);
    for (residual.slice(u16, n), values, back) |*r, v, q| r.* = affine4.bf16of(v - q);
    const d = w.shape[0];
    const k = w.shape[1];
    var full: [192]u8 = undefined;
    try ck.set(try std.fmt.bufPrint(&full, "{s}.weight", .{base}), .{ .buffer = codes, .offset = 0, .bytes = n / 2, .dtype = .u32, .shape = .{ d, k / 8, 1, 1 }, .rank = 2 });
    try ck.set(try std.fmt.bufPrint(&full, "{s}.scales", .{base}), .{ .buffer = codes, .offset = n / 2, .bytes = 2 * groups, .dtype = .bf16, .shape = .{ d, k / 64, 1, 1 }, .rank = 2 });
    try ck.set(try std.fmt.bufPrint(&full, "{s}.biases", .{base}), .{ .buffer = codes, .offset = n / 2 + 2 * groups, .bytes = 2 * groups, .dtype = .bf16, .shape = .{ d, k / 64, 1, 1 }, .rank = 2 });
    return residual;
}

/// The residuals into their layers' Moe, the weights owning them from here (freed here when that fails).
pub fn attach(w: *wts.Weights, residuals: Residuals) !void {
    var held: usize = 0;
    for (residuals) |r| held += @intFromBool(r != null);
    w.owned.ensureUnusedCapacity(w.allocator, held) catch |e| {
        free(residuals);
        return e;
    };
    for (residuals, 0..) |r, i| if (r) |b| {
        w.owned.appendAssumeCapacity(b);
        w.layers[i].moe.slide = b;
    };
}

pub fn free(residuals: Residuals) void {
    for (residuals) |r| if (r) |b| b.deinit();
}
