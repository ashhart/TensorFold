//! Gate 5's sources and precision tags for all SAM3 Metal launches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

const Owner = struct { prefix: []const u8, tensor: bool, file: type };

const owners = [_]Owner{
    .{ .prefix = "sam3 layout ", .tensor = false, .file = @import("metal_sam3_layout.zig") },
    .{ .prefix = "sam3 arithmetic ", .tensor = false, .file = @import("metal_sam3_arithmetic.zig") },
    .{ .prefix = "sam3 rotary ", .tensor = false, .file = @import("metal_sam3_rotary.zig") },
    .{ .prefix = "sam3 rows attention ", .tensor = false, .file = @import("metal_sam3_rows.zig") },
    .{ .prefix = "sam3 flash ", .tensor = true, .file = @import("metal_sam3_flash.zig") },
    .{ .prefix = "sam3 trunk tile ", .tensor = true, .file = @import("metal_sam3_trunk.zig") },
    .{ .prefix = "sam3 tensor ops ", .tensor = true, .file = @import("metal_sam3_tensor_ops.zig") },
    .{ .prefix = "sam3 core attention ", .tensor = true, .file = @import("metal_sam3_core_attention.zig") },
};

const extra_prefixes = [_][]const u8{
    "sam3 fp32 gemm ",
    "sam3 device projection ",
    "sam3 activation ",
    "sam3 resize ",
    "sam3 sparse mask ",
    "sam3 compensated normalization ",
    "sam3 small compensated projection ",
    "sam3 deconvolution ",
    "sam3 attention ",
    "sam3 fp32 mma ",
};

pub fn owns(name: []const u8) bool {
    inline for (owners) |o| if (std.mem.startsWith(u8, name, o.prefix)) return true;
    inline for (extra_prefixes) |p| if (std.mem.startsWith(u8, name, p)) return true;
    return false;
}

/// Tensor-op launches build only where the engine's SAM library does (the M5's tensor units).
pub fn tensorOnly(name: []const u8) bool {
    inline for (owners) |o| if (o.tensor and std.mem.startsWith(u8, name, o.prefix)) return true;
    return std.mem.startsWith(u8, name, "sam3 fp32 mma ") or
        std.mem.startsWith(u8, name, "sam3 fp32 gemm ") or
        std.mem.startsWith(u8, name, "sam3 attention ") or
        std.mem.startsWith(u8, name, "sam3 device projection ") or
        std.mem.startsWith(u8, name, "sam3 deconvolution matrix ");
}

pub fn printPrecision(name: []const u8) bool {
    if (std.mem.indexOf(u8, name, "exact expansion ") != null or std.mem.indexOf(u8, name, " planes ") != null) {
        std.debug.print("PRECISION limits {s}: exact FP16 components, FP32 accumulation\n", .{name});
        return true;
    }
    if (std.mem.indexOf(u8, name, "exact half weight ") != null) {
        std.debug.print("PRECISION limits {s}: FP32 activations and accumulation, exact FP16 weights\n", .{name});
        return true;
    }
    if (std.mem.startsWith(u8, name, "sam3 attention ") or std.mem.startsWith(u8, name, "sam3 device projection ")) {
        std.debug.print("PRECISION limits {s}: strict FP32 tensor library\n", .{name});
        return true;
    }
    return false;
}

/// The source a record's pipeline compiles here, and before the M5 (empty: tensor-op source, no M1/M2 build).
pub fn text(a: std.mem.Allocator, device: mtl.Device, name: []const u8, fun: []const u8) ![2][]const u8 {
    const control = std.mem.indexOf(u8, name, " control ") != null;
    inline for (owners) |o| if (std.mem.startsWith(u8, name, o.prefix)) {
        const here = try o.file.limitText(a, name, control);
        return .{ here, if (o.tensor) "" else here };
    };
    if (std.mem.startsWith(u8, name, "sam3 fp32 gemm "))
        return .{ try @import("metal_sam3_gemm.zig").text(a, device, std.mem.indexOf(u8, fun, "_exact_") != null, std.mem.indexOf(u8, fun, "_compact_") != null), "" };
    if (std.mem.startsWith(u8, name, "sam3 device projection ")) {
        if (std.mem.startsWith(u8, name, "sam3 device projection pair "))
            return .{ try @import("metal_sam3_pair.zig").text(a, control), "" };
        if (std.mem.startsWith(u8, name, "sam3 device projection planes "))
            return .{ try @import("metal_sam3_planes.zig").text(a, control), "" };
        if (std.mem.startsWith(u8, name, "sam3 device projection exact half weight "))
            return .{ try @import("metal_sam3_dense_small.zig").halfText(a, control), "" };
        return .{ try @import("metal_sam3_dense_small.zig").deviceText(a, control), "" };
    }
    if (std.mem.startsWith(u8, name, "sam3 activation ")) {
        const source = try std.mem.concat(a, u8, &.{ "#define TF_UNROLL _Pragma(\"clang loop unroll(full)\")\n", sources.sam3_dense_small, sources.sam3_activation });
        return .{ source, source };
    }
    if (std.mem.startsWith(u8, name, "sam3 resize ")) {
        const source = if (std.mem.indexOf(u8, name, " nonfinite-control ") != null) try @import("metal_sam3_resize.zig").nonfiniteText(a) else try @import("metal_sam3_resize.zig").text(a, control);
        return .{ source, source };
    }
    if (std.mem.startsWith(u8, name, "sam3 sparse mask ")) {
        const source = try @import("metal_sam3_sparse_mask.zig").text(a, control);
        return .{ source, source };
    }
    if (std.mem.startsWith(u8, name, "sam3 compensated normalization ")) {
        const source = try @import("metal_sam3_norm.zig").text(a, control);
        return .{ source, source };
    }
    if (std.mem.startsWith(u8, name, "sam3 small compensated projection ")) {
        const source = try a.dupe(u8, if (std.mem.indexOf(u8, name, " baseline ") != null) sources.sam3_dense_small_baseline else sources.sam3_dense_small);
        return .{ source, source };
    }
    if (std.mem.startsWith(u8, name, "sam3 deconvolution ")) {
        if (std.mem.indexOf(u8, name, " matrix ") != null)
            return .{ try @import("metal_sam3_deconv.zig").matrixText(a), "" };
        const source = try a.dupe(u8, sources.sam3_deconv ++ sources.sam3_deconv_baseline);
        return .{ source, source };
    }
    if (std.mem.startsWith(u8, name, "sam3 attention ")) {
        if (std.mem.startsWith(u8, name, "sam3 attention D32 split ") or std.mem.startsWith(u8, name, "sam3 attention D64 split "))
            return .{ try @import("metal_sam3_attention.zig").split32Text(a, control), "" };
        if (std.mem.indexOf(u8, name, " device ") != null)
            return .{ try @import("metal_sam3_attention.zig").text(a, false), "" };
        if (std.mem.indexOf(u8, name, " split ") != null)
            return .{ try @import("metal_sam3_attention.zig").splitText(a, control), "" };
        if (std.mem.indexOf(u8, name, " two-pass ") != null)
            return .{ try @import("metal_sam3_attention.zig").twoPassText(a, control), "" };
        return .{ try @import("metal_sam3_attention.zig").text(a, control), "" };
    }
    if (std.mem.startsWith(u8, name, "sam3 fp32 mma "))
        return .{ try @import("metal_sam3_mma.zig").text(a, device), "" };
    return error.UnknownLaunch;
}
