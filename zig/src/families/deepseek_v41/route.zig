//! Exhaustive 384-expert sqrt-softplus top-six routing. Bias affects selection, never returned weights.
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const View = @import("view.zig").View;
pub const Dtype = @import("ops.zig").Dtype;
pub const Pipelines = struct {
    select: mtl.Pipeline,
    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_route, options);
        defer lib.deinit();
        return .{ .select = try mtl.Pipeline.init(device, lib, "dsv41_route", false) };
    }
    pub fn deinit(p: *Pipelines) void {
        p.select.deinit();
    }
};
pub const Shape = struct {
    rows: u32,
    logit_stride: u32 = 384,
    output_stride: u32 = 6,
    temperature: f32 = 1,
    bias_dtype: Dtype = .f32,
    /// Nonzero: also write the six IDs at this stride into `merged`, slot 6 = merged_extra.
    merged_stride: u32 = 0,
    merged_extra: u32 = 0,
};
/// FP32 logits[rows,384], bias F16/BF16/FP32 -> u32 IDs and FP32 weights[rows,6]; lower ID wins biased-score ties.

// Nonzero rows, logit/output pitches >=384/6, finite positive temperature, complete storage are caller invariants.

// All six execute; weights use unbiased scores / (sum+1e-20) * 1.5. Invalid comparisons emit invalid IDs/NaNs.
pub fn encode(e: mtl.ComputeEncoder, p: *const Pipelines, logits: View, bias: View, ids: View, weights: View, d: Shape) void {
    encodeMerged(e, p, logits, bias, ids, weights, null, d);
}

/// `encode`, plus the IDs at `d.merged_stride` (seven: six routed slots and `d.merged_extra`) into `merged`.
pub fn encodeMerged(e: mtl.ComputeEncoder, p: *const Pipelines, logits: View, bias: View, ids: View, weights: View, merged: ?View, d: Shape) void {
    e.setPipeline(p.select);
    inline for (.{ logits, bias, ids, weights }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    const out = merged orelse ids;
    e.setBuffer(out.buffer, out.offset, 5);
    const D = extern struct { rows: u32, logit_stride: u32, output_stride: u32, temperature: f32, bias_dtype: u32, merged_stride: u32, merged_extra: u32 };
    e.setValue(D{ .rows = d.rows, .logit_stride = d.logit_stride, .output_stride = d.output_stride, .temperature = d.temperature, .bias_dtype = @backingInt(d.bias_dtype), .merged_stride = if (merged != null) d.merged_stride else 0, .merged_extra = d.merged_extra }, 4);
    e.dispatchGroups(mtl.Size.of(d.rows, 1, 1), mtl.Size.of(128, 1, 1));
}
