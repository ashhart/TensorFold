//! The four T622 preprocessing launches; the reference pipelines are frozen beside this file.
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const glue = q.glue;
const Ref = q.projection.Ref;
fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}
pub fn encode(e: mtl.ComputeEncoder, norm: mtl.Pipeline, rope: mtl.Pipeline, p: q.attention.Preprocess) void {
    const g = p.geometry;
    e.setPipeline(norm);
    bind(e, .{ p.qg, p.q_gain });
    e.setValue(glue.Head{ .width = g.dim, .rows = p.rows * g.heads, .stride = 2 * g.dim, .activation = .bf16, .gain = .bf16, .eps = p.eps }, 2);
    e.setBuffer(p.norm_q.buffer, p.norm_q.offset, 3);
    e.dispatchGroups(mtl.Size.of(p.rows * g.heads, 1, 1), mtl.Size.of(64, 1, 1));
    e.setPipeline(norm);
    bind(e, .{ p.mixed, p.k_gain });
    e.setValue(glue.Head{ .width = g.dim, .rows = p.rows * g.kv_heads, .stride = g.dim, .heads = g.kv_heads, .row_stride = (if (p.key_row_stride == 0) 2 * g.kv_heads * g.dim else p.key_row_stride), .activation = .bf16, .gain = .bf16, .eps = p.eps }, 2);
    e.setBuffer(p.norm_k.buffer, p.norm_k.offset, 3);
    e.dispatchGroups(mtl.Size.of(p.rows * g.kv_heads, 1, 1), mtl.Size.of(64, 1, 1));
    e.barrier();
    for ([_]struct { x: Ref, y: Ref, heads: u32 }{ .{ .x = p.norm_q, .y = p.query, .heads = g.heads }, .{ .x = p.norm_k, .y = p.rotated_key, .heads = g.kv_heads } }) |r| {
        e.setPipeline(rope);
        bind(e, .{ r.x, p.positions });
        e.setValue(glue.Rope{ .width = g.dim, .heads = r.heads, .rows = p.rows, .rotary = g.rotary, .activation = .bf16, .theta = @floatCast(p.theta), .h_end = p.h_end, .w_end = p.w_end, .plane = p.plane }, 2);
        e.setBuffer(r.y.buffer, r.y.offset, 3);
        e.dispatchThreads(mtl.Size.of(g.dim, r.heads, p.rows), mtl.Size.of(32, 1, 1));
    }
    e.barrier();
}
