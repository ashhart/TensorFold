//! Encode exact BF16 per-row top-k into caller-owned output buffers.
const std = @import("std");
const mtl = @import("metal");
pub const contract = @import("bf16_topk");
const source = @import("bf16_topk_sources").text;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Ops = struct {
    pipeline: mtl.Pipeline,
    max_pipeline: ?mtl.Pipeline = null,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk", false);
        errdefer pipeline.deinit();
        return .{ .pipeline = pipeline, .max_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_max", false) };
    }
    pub fn deinit(ops: Ops) void {
        ops.pipeline.deinit();
        if (ops.max_pipeline) |pipeline| pipeline.deinit();
    }
    pub fn encode(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, output: Ref, p: contract.Params) !void {
        try contract.check(p);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const input_bytes = try std.math.mul(usize, cells, 2);
        const output_bytes = @as(usize, p.rows) * @sizeOf(contract.Row);
        try check(logits, input_bytes, 2);
        try check(output, output_bytes, 4);
        if (logits.buf.id == output.buf.id and logits.off < output.off + output_bytes and output.off < logits.off + input_bytes) return error.TopKAlias;
        e.setPipeline(if (p.k == 1) ops.max_pipeline orelse ops.pipeline else ops.pipeline);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(output.buf, output.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }
};
fn check(ref: Ref, bytes: usize, alignment: usize) !void {
    if (ref.off % alignment != 0 or ref.off > ref.buf.length() or bytes > ref.buf.length() - ref.off) return error.BadTopKBuffer;
}
