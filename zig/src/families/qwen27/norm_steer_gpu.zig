//! The steered norm on the GPU: the residual norm with a control vector's direction added inside it.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("qwen_runtime_sources");
const profile = @import("core").gpu_profile;
const quant = @import("quant_gpu.zig");
const steer = @import("steer.zig");
const Ref = quant.Ref;

pub const symbol = "custom_kernel_q27_norm_steer_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float_float_float";

/// The steered norm's source for a `width`-wide row, specialised as the plain norms are.
pub fn source(a: std.mem.Allocator, width: u32) ![]u8 {
    const template = sources.norm_steer;
    const mark = std.mem.indexOf(u8, template, "Q27_SPECIALIZE") orelse return error.BadKernelSource;
    const definitions = try std.fmt.allocPrint(a, "constexpr int K={d};", .{width});
    defer a.free(definitions);
    return std.mem.concat(a, u8, &.{ "#define Q27_XS 64\n", template[0..mark], definitions, template[mark + "Q27_SPECIALIZE".len ..] });
}

/// The plan's pipeline, compiled at its first dispatch: the plan's rows are all `width` wide.
fn pipeline(k: *quant.Kernels, plan: *steer.Plan, width: u32) !mtl.Pipeline {
    if (plan.pipeline) |p| return p;
    const text = try source(k.allocator, width);
    defer k.allocator.free(text);
    const library = try mtl.Library.fromSource(k.device, text, mtl.CompileOptions.mlx());
    defer library.deinit();
    plan.pipeline = try mtl.Pipeline.init(k.device, library, symbol, false);
    return plan.pipeline.?;
}

/// The norm with a direction inside it: alpha times the direction is added before the normalisation reads the row.
pub fn encode(k: *quant.Kernels, plan: *steer.Plan, input_encoder: mtl.ComputeEncoder, h: Ref, residual: Ref, weight: Ref, eps: Ref, dims: Ref, combined: Ref, out: Ref, xs: Ref, dir: mtl.Buffer, width: u32, rows: u32) !void {
    var span = profile.begin(k.profiler, input_encoder, .norm);
    errdefer span.cancel();
    const e = span.encoder;

    e.setPipeline(try pipeline(k, plan, width));
    inline for (.{ h, residual, weight, eps, dims, combined, out, xs }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    e.setBuffer(dir, 0, 8);
    e.setValue(plan.alpha, 9);
    e.dispatchGroups(mtl.Size.of(1, quant.padded(rows), 1), mtl.Size.of(width / 16, 1, 1));
    // The steered output has no tf_sq_fits words: the last plain norm's must not outlive it.
    k.fits = null;
    try span.end();
}
