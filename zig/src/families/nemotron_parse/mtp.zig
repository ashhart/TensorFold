//! The checkpoint's trained one-step proposal: unscaled token embedding, biased projection, residual and head.
const hip = @import("hip");
const c = @import("config.zig");
const Compute = @import("compute.zig").Compute;

/// Pack one state row and one pending token per chain; the target alone verifies the returned proposal logits.
pub fn logits(compute: Compute, hidden: hip.ops.Tensor, ids: u64, rows: usize) !hip.ops.Tensor {
    if (rows == 0) return error.EmptyProposal;
    const embedding = try compute.alloc(rows * c.hidden, .bf16);
    try compute.kernels.embed(compute.stream, try compute.weights.get("decoder.embed_tokens.weight"), ids, embedding, rows, c.hidden, 1);
    const projected = try compute.linear(embedding, "decoder.extra_proj.0", rows, c.hidden, c.hidden, true);
    try compute.residual(projected, hidden, rows * c.hidden);
    const out = try compute.linear(projected, "decoder.extra_heads.0", rows, c.hidden, c.hidden, true);
    return @import("decoder.zig").logits(compute, out, rows);
}
