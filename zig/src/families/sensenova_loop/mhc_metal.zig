//! Metal launch wrappers and ABI checks for SenseNova-Loop two-stream mHC kernels.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const tg: u32 = 256;
pub const max_hidden: u32 = 4096;
pub const coeff_width: u32 = 6;

pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,
};

pub const Error = error{
    BadMhcShape,
    BadMhcParams,
};

pub const EnterArgs = extern struct {
    rows: u32,
    hidden: u32,
    has_pending: u32,
    write_h: u32,
};

pub const PreNormArgs = extern struct {
    rows: u32,
    hidden: u32,
    iters: u32,
    shift_norm: u32,
    xs_group: u32,
    xs_stride: u32,
    pre_scale: f32,
    eps: f32,
};

pub const PostMergeArgs = extern struct {
    rows: u32,
    hidden: u32,
    exit: u32,
    pad0: u32 = 0,
};

comptime {
    if (@sizeOf(EnterArgs) != 16 or @offsetOf(EnterArgs, "write_h") != 12) @compileError("EnterArgs ABI mismatch");
    if (@sizeOf(PreNormArgs) != 32 or @offsetOf(PreNormArgs, "eps") != 28) @compileError("PreNormArgs ABI mismatch");
    if (@sizeOf(PostMergeArgs) != 16 or @offsetOf(PostMergeArgs, "exit") != 8) @compileError("PostMergeArgs ABI mismatch");
}

pub const PreNormCall = struct {
    iters: u32 = 20,
    shift_norm: bool = true,
    xs_group: u32 = 0,
    xs_stride: u32 = 0,
    pre_scale: f32 = 1.0,
    eps: f32 = 1e-6,
};

pub const Pipelines = struct {
    enter: mtl.Pipeline,
    pre_norm: mtl.Pipeline,
    post_merge: mtl.Pipeline,
    post_and_pre_norm: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.core_sensenova_mhc);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.enter = try mtl.Pipeline.init(device, lib, "sn_mhc_enter", false);
        errdefer p.enter.deinit();
        p.pre_norm = try mtl.Pipeline.init(device, lib, "sn_mhc_pre_norm", false);
        errdefer p.pre_norm.deinit();
        p.post_merge = try mtl.Pipeline.init(device, lib, "sn_mhc_post_merge", false);
        errdefer p.post_merge.deinit();
        p.post_and_pre_norm = try mtl.Pipeline.init(device, lib, "sn_mhc_post_and_pre_norm", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.enter.deinit();
        p.pre_norm.deinit();
        p.post_merge.deinit();
        p.post_and_pre_norm.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

pub fn checkShape(rows: u32, hidden: u32) Error!void {
    if (rows == 0 or hidden == 0 or hidden > max_hidden) return error.BadMhcShape;
}

pub fn checkPreNorm(rows: u32, hidden: u32, call: PreNormCall, has_xs: bool) Error!void {
    try checkShape(rows, hidden);
    if (call.iters == 0) return error.BadMhcParams;
    if (!std.math.isFinite(call.pre_scale) or call.pre_scale <= 0) return error.BadMhcParams;
    if (!std.math.isFinite(call.eps) or call.eps < 0) return error.BadMhcParams;
    if (call.xs_group != 0 and call.xs_group != 64 and call.xs_group != 128) return error.BadMhcParams;
    if (call.xs_group != 0) {
        if (!has_xs or hidden % call.xs_group != 0 or call.xs_stride < rows) return error.BadMhcParams;
    }
}

/// Expand `[rows, H]` (optionally `bf16(h + pending)`) into `[rows, 2, H]` residual streams.
pub fn enter(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    h_in: Ref,
    pending: ?Ref,
    r_out: Ref,
    h_out: ?Ref,
    rows: u32,
    hidden: u32,
) Error!void {
    try checkShape(rows, hidden);
    const elems = if (hidden % 8 == 0) hidden / 8 else hidden;
    const grid_x = (elems + tg - 1) / tg;
    e.setPipeline(p.enter);
    bind(e, &.{ h_in, pending orelse h_in, r_out, h_out orelse r_out });
    e.setValue(EnterArgs{
        .rows = rows,
        .hidden = hidden,
        .has_pending = @intFromBool(pending != null),
        .write_h = @intFromBool(h_out != null),
    }, 4);
    e.dispatchGroups(mtl.Size.of(grid_x, rows, 1), mtl.Size.of(tg, 1, 1));
}

/// Compute mHC routing coefficients, collapsed branch RMSNorm and optional group sums in one pass.
pub fn preNorm(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    r_in: Ref,
    w: Ref,
    b: Ref,
    sub_w: Ref,
    coeff_out: Ref,
    norm_out: Ref,
    xs_out: ?Ref,
    rows: u32,
    hidden: u32,
    call: PreNormCall,
) Error!void {
    try checkPreNorm(rows, hidden, call, xs_out != null);
    e.setPipeline(p.pre_norm);
    bind(e, &.{ r_in, w, b, sub_w, coeff_out, norm_out, xs_out orelse norm_out });
    e.setValue(PreNormArgs{
        .rows = rows,
        .hidden = hidden,
        .iters = call.iters,
        .shift_norm = @intFromBool(call.shift_norm),
        .xs_group = call.xs_group,
        .xs_stride = call.xs_stride,
        .pre_scale = call.pre_scale,
        .eps = call.eps,
    }, 7);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(tg, 1, 1));
}

/// Mix two residual streams with `coeff`, add branch update and optionally mean-collapse on exit.
pub fn postMerge(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    r_in: Ref,
    y_in: Ref,
    coeff_in: Ref,
    out: Ref,
    rows: u32,
    hidden: u32,
    exit: bool,
) Error!void {
    try checkShape(rows, hidden);
    const elems = if (hidden % 8 == 0) hidden / 8 else hidden;
    const grid_x = (elems + tg - 1) / tg;
    e.setPipeline(p.post_merge);
    bind(e, &.{ r_in, y_in, coeff_in, out });
    e.setValue(PostMergeArgs{ .rows = rows, .hidden = hidden, .exit = @intFromBool(exit) }, 4);
    e.dispatchGroups(mtl.Size.of(grid_x, rows, 1), mtl.Size.of(tg, 1, 1));
}

/// Fused post_merge + next pre_norm keeping `R_next` in threadgroup memory, bit-identical to sequential.
pub fn postAndPreNorm(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    r_in: Ref,
    y_in: Ref,
    coeff_in: Ref,
    w_next: Ref,
    b_next: Ref,
    sub_w_next: Ref,
    r_out: Ref,
    coeff_out: Ref,
    norm_out: Ref,
    xs_out: ?Ref,
    rows: u32,
    hidden: u32,
    call: PreNormCall,
) Error!void {
    try checkPreNorm(rows, hidden, call, xs_out != null);
    e.setPipeline(p.post_and_pre_norm);
    bind(e, &.{ r_in, y_in, coeff_in, w_next, b_next, sub_w_next, r_out, coeff_out, norm_out, xs_out orelse norm_out });
    e.setValue(PreNormArgs{
        .rows = rows,
        .hidden = hidden,
        .iters = call.iters,
        .shift_norm = @intFromBool(call.shift_norm),
        .xs_group = call.xs_group,
        .xs_stride = call.xs_stride,
        .pre_scale = call.pre_scale,
        .eps = call.eps,
    }, 10);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(tg, 1, 1));
}

test "mhc_metal: validates shapes, pre_scale, xs_group and struct layout" {
    try checkShape(1, 4096);
    try checkPreNorm(4, 4096, .{ .xs_group = 64, .xs_stride = 4 }, true);
    try std.testing.expectError(error.BadMhcShape, checkShape(0, 4096));
    try std.testing.expectError(error.BadMhcShape, checkShape(1, 0));
    try std.testing.expectError(error.BadMhcShape, checkShape(1, 4097));
    try std.testing.expectError(error.BadMhcParams, checkPreNorm(4, 4096, .{ .iters = 0 }, false));
    try std.testing.expectError(error.BadMhcParams, checkPreNorm(4, 4096, .{ .pre_scale = 0 }, false));
    try std.testing.expectError(error.BadMhcParams, checkPreNorm(4, 4096, .{ .xs_group = 32, .xs_stride = 4 }, true));
    try std.testing.expectError(error.BadMhcParams, checkPreNorm(4, 4096, .{ .xs_group = 64, .xs_stride = 3 }, true));
    try std.testing.expectError(error.BadMhcParams, checkPreNorm(4, 4096, .{ .xs_group = 64, .xs_stride = 4 }, false));
}
