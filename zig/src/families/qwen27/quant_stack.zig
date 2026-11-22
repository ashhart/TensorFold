//! Mixed-stack prompt-row staging and placement at the fused output stride.
const mtl = @import("metal");
const quant = @import("quant_gpu.zig");
const Kernels = quant.Kernels;
const Ref = quant.Ref;
const profile = @import("core").gpu_profile;

pub const Stack = extern struct { rows: u32, width: u32, stride: u32, offset: u32 };

/// The stage buffer, grown to at least `bytes`.
pub fn staging(k: *Kernels, bytes: usize) !Ref {
    if (bytes > k.stage_bytes) {
        const b = try k.device.buffer(bytes, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        if (k.stage) |old| old.deinit();
        k.stage = b;
        k.stage_bytes = bytes;
    }
    return .{ .buffer = k.stage.?, .offset = 0 };
}

/// Rows of `p.width` values at `x` into `y` at a stride of `p.stride` (bf16 bits kept), between barriers.
pub fn place(k: *Kernels, input_encoder: mtl.ComputeEncoder, x: Ref, y: Ref, p: Stack) !void {
    if (k.stack_pipeline == null) {
        const library = try mtl.Library.fromSource(k.device, stack_source, mtl.CompileOptions.mlx());
        defer library.deinit();
        k.stack_pipeline = try mtl.Pipeline.init(k.device, library, "q27_stack", false);
    }
    var span = profile.begin(k.profiler, input_encoder, .copy);
    errdefer span.cancel();
    const e = span.encoder;
    e.barrier(); // the lane product's rows are written
    e.setPipeline(k.stack_pipeline.?);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setValue(p, 1);
    e.setBuffer(y.buffer, y.offset, 2);
    e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(256, 1, 1));
    e.barrier(); // the stage is free for the next member
    try span.end();
}

const stack_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\struct Stack { uint rows, width, stride, offset; };
    \\[[max_total_threads_per_threadgroup(256)]]
    \\kernel void q27_stack(device const ushort* X [[buffer(0)]], constant Stack& p [[buffer(1)]],
    \\                      device ushort* Y [[buffer(2)]], uint2 t [[thread_position_in_grid]]) {
    \\  if (t.x < p.width && t.y < p.rows) Y[t.y * p.stride + p.offset + t.x] = X[t.y * p.width + t.x];
    \\}
;
