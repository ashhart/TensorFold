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
/// The fused gate+up dispatch's tables: the 2E-descriptor pick list and the input row each pick reads.
pub const Fused = struct { ids: View, sources: View };
/// The kernel's `Shape`: the launch record route_exl3.metal reads.
pub const Args = extern struct { rows: u32, logit_stride: u32, output_stride: u32, temperature: f32, bias_dtype: u32, fused: u32, logit_dtype: u32, experts: u32, topk: u32, moe_up: u32 };
pub const Shape = struct {
    rows: u32,
    logit_stride: u32 = 384,
    output_stride: u32 = 6,
    temperature: f32 = 1,
    bias_dtype: Dtype = .f32,
    /// Nonzero: also write the fused gate+up dispatch's pick table (a token's six gate picks then its six up
    /// picks one table entry further, `fused` = the block's expert count). Zero keeps every shipped caller's
    /// writes and buffers exactly as they were.
    fused: u32 = 0,
    /// 0: FP32 logits (every shipped caller); 1: BF16 logits (TF_DSV41_ROUTE_GEMM's gate GEMM writes them).
    logit_dtype: u32 = 0,
    /// The block's own router geometry: the backbone's 384 experts and six picks, the drafter's 128 experts and
    /// three picks (the pack's dspark_n_routed_experts / dspark_num_experts_per_tok).
    experts: u32 = 384,
    topk: u32 = 6,
    /// Nonzero writes the shared expert's two pick blocks into the fused tables after the routed twelve a token, so
    /// the routed gate/up and the shared gate/up are ONE indexed dispatch (the comparator's moe_up form). The
    /// kernel writes them, not the host, so no CPU/GPU buffer range is shared.
    moe_up: u32 = 0,
};
/// FP32 logits[rows,384], bias F16/BF16/FP32 -> u32 IDs and FP32 weights[rows,6]; lower ID wins biased-score ties.

// Nonzero rows, logit/output pitches >=384/6, finite positive temperature, complete storage are caller invariants.

// Invalid comparisons emit invalid IDs/NaNs.
pub fn encode(e: mtl.ComputeEncoder, p: *const Pipelines, logits: View, bias: View, ids: View, weights: View, fused: Fused, d: Shape) void {
    e.setPipeline(p.select);
    inline for (.{ logits, bias, ids, weights }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    e.setBuffer(fused.ids.buffer, fused.ids.offset, 5);
    e.setBuffer(fused.sources.buffer, fused.sources.offset, 6);
    e.setValue(Args{ .rows = d.rows, .logit_stride = d.logit_stride, .output_stride = d.output_stride, .temperature = d.temperature, .bias_dtype = @backingInt(d.bias_dtype), .fused = d.fused, .logit_dtype = d.logit_dtype, .experts = d.experts, .topk = d.topk, .moe_up = d.moe_up }, 4);
    e.dispatchGroups(mtl.Size.of(d.rows, 1, 1), mtl.Size.of(128, 1, 1));
}
