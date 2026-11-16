//! The optional int8 sidecar: one pre-rotated fp8 weight and its scales per projection, read from local files.
const std = @import("std");
const mtl = @import("metal");
const w = @import("weights.zig");
const Matrix = w.Matrix;
const Model = w.Model;

pub fn attachInt8(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, model: *Model, dir: []const u8) !void {
    const direct = std.c.getenv("TF_DSV41_I8_ROT") != null;
    // An instrument: attach one projection at a time to localise a numerics fault (TF_DSV41_I8_ONLY=wo_a).
    const only: ?[]const u8 = if (std.c.getenv("TF_DSV41_I8_ONLY")) |v| std.mem.span(v) else null;
    for (&model.layers, 0..) |*layer, l| {
        try attachInt8One(gpa, io, device, &layer.attn.wq_b, dir, l, "attn.wq_b", direct, only);
        try attachInt8One(gpa, io, device, &layer.attn.wo_b, dir, l, "attn.wo_b", direct, only);
        for (&layer.attn.wo_a, 0..) |*m, g| {
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "attn.wo_a.slice.{d}", .{g});
            try attachInt8One(gpa, io, device, m, dir, l, name, direct, only);
        }
    }
    model.int8_direct = direct;
    std.log.info("exl3: int8 sidecar attached from {s}: {d} layers, {d} x {d} + {d} x {d} + 8 x {d} x {d}, direct={}", .{
        dir,
        model.layers.len,
        model.layers[0].attn.wq_b.k,
        model.layers[0].attn.wq_b.n,
        model.layers[0].attn.wo_b.k,
        model.layers[0].attn.wo_b.n,
        model.layers[0].attn.wo_a[0].k,
        model.layers[0].attn.wo_a[0].n,
        direct,
    });
}

pub fn attachInt8One(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, m: *Matrix, dir: []const u8, layer: usize, projection: []const u8, direct: bool, only: ?[]const u8) !void {
    if (m.int8 != null) return;
    if (only) |o| if (std.mem.indexOf(u8, o, projection) == null) return;
    const weight_bytes = try std.math.mul(usize, m.k, m.n);
    var path: [512]u8 = undefined;
    const wpath = try std.fmt.bufPrint(&path, "{s}/layers.{d}.{s}.fp8", .{ dir, layer, projection });
    const wb = try std.Io.Dir.cwd().readFileAlloc(io, wpath, gpa, .limited(weight_bytes + 16));
    defer gpa.free(wb);
    if (wb.len != weight_bytes) return error.BadInt8Sidecar;
    var spath_buf: [512]u8 = undefined;
    const spath = try std.fmt.bufPrint(&spath_buf, "{s}/layers.{d}.{s}.scale", .{ dir, layer, projection });
    const scale_bytes = try std.math.mul(usize, m.n, 4);
    const sb = try std.Io.Dir.cwd().readFileAlloc(io, spath, gpa, .limited(scale_bytes + 16));
    defer gpa.free(sb);
    if (sb.len != scale_bytes) return error.BadInt8Sidecar;
    const weight = try device.buffer(weight_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer weight.deinit();
    @memcpy(weight.contents()[0..weight_bytes], wb);
    const scales = try device.buffer(scale_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer scales.deinit();
    @memcpy(scales.contents()[0..scale_bytes], sb);
    m.int8 = .{ .weight = .{ .buffer = weight, .offset = 0 }, .scales = .{ .buffer = scales, .offset = 0 }, .k = m.k, .n = m.n };
    m.int8_direct = direct;
}
