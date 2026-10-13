//! Encode exact BF16 per-row top-k into caller-owned output buffers.
const std = @import("std");
const mtl = @import("metal");
pub const contract = @import("bf16_topk");
const source = @import("bf16_topk_sources").text;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
/// Spans a wide row is split into (TF_TOPK_SPANS in the kernel) and the narrowest row split.
const spans = 16;
const wide_vocab = 4096;
/// Threads per threadgroup, in every dispatch (TF_TOPK_LANES in the kernel).
const lanes = 256;
comptime {
    @setEvalBranchQuota(100_000);
    // the kernel's arrays, strides, tournament and entry counts are all written with these three
    for (.{ .{ "TF_TOPK_LANES", lanes }, .{ "TF_TOPK_MAX_K", contract.max_k }, .{ "TF_TOPK_SPANS", spans } }) |named| {
        const define = std.fmt.comptimePrint("#define {s} {d}\n", .{ named[0], named[1] });
        if (std.mem.indexOf(u8, source, define) == null) @compileError("bf16_topk.metal's " ++ named[0] ++ " differs from the Zig constant");
    }
    // the merge kernel gives each span's max_k entries one lane, and the tournament halves the lanes
    if (spans * contract.max_k != lanes or !std.math.isPowerOfTwo(lanes)) @compileError("bf16 top-k merge needs spans * max_k == lanes, a power of two");
}
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const Ops = struct {
    pipeline: mtl.Pipeline,
    max_pipeline: ?mtl.Pipeline = null,
    span_pipeline: mtl.Pipeline,
    merge_pipeline: mtl.Pipeline,
    /// Every span's top k, between a wide row's two dispatches: one wide top-k may be in flight per Ops.
    part: mtl.Buffer,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk", false);
        errdefer pipeline.deinit();
        const max_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_max", false);
        errdefer max_pipeline.deinit();
        const span_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_span", false);
        errdefer span_pipeline.deinit();
        const merge_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_merge", false);
        errdefer merge_pipeline.deinit();
        const part = try device.buffer(contract.max_rows * spans * @sizeOf(contract.Row), options);
        return .{ .pipeline = pipeline, .max_pipeline = max_pipeline, .span_pipeline = span_pipeline, .merge_pipeline = merge_pipeline, .part = part };
    }
    pub fn deinit(ops: Ops) void {
        ops.pipeline.deinit();
        if (ops.max_pipeline) |pipeline| pipeline.deinit();
        ops.span_pipeline.deinit();
        ops.merge_pipeline.deinit();
        ops.part.deinit();
    }
    pub fn encode(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, output: Ref, p: contract.Params) !void {
        try contract.check(p);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const input_bytes = try std.math.mul(usize, cells, 2);
        const output_bytes = @as(usize, p.rows) * @sizeOf(contract.Row);
        try check(logits, input_bytes, 2);
        try check(output, output_bytes, 4);
        if (logits.buf.id == output.buf.id and logits.off < output.off + output_bytes and output.off < logits.off + input_bytes) return error.TopKAlias;
        if (p.k > 1 and p.vocab >= wide_vocab) {
            e.setPipeline(ops.span_pipeline);
            e.setBuffer(logits.buf, logits.off, 0);
            e.setBuffer(ops.part, 0, 1);
            e.setValue(p, 2);
            e.dispatchGroups(mtl.Size.of(spans, p.rows, 1), mtl.Size.of(lanes, 1, 1));
            e.barrier();
            e.setPipeline(ops.merge_pipeline);
            e.setBuffer(ops.part, 0, 0);
            e.setBuffer(output.buf, output.off, 1);
            e.setValue(p, 2);
            e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(lanes, 1, 1));
            return;
        }
        e.setPipeline(if (p.k == 1) ops.max_pipeline orelse ops.pipeline else ops.pipeline);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(output.buf, output.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(lanes, 1, 1));
    }
};
fn check(ref: Ref, bytes: usize, alignment: usize) !void {
    if (ref.off % alignment != 0 or ref.off > ref.buf.length() or bytes > ref.buf.length() - ref.off) return error.BadTopKBuffer;
}
