//! One FastH3 block in bf16: products on tfp::gemm_nax (MLX's GEMM, fp32 sums), FastH3's routed attention in bf16.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3.zig");

const kernels = @import("h3_kernels");

const Model = h3.Model;
const Ref = h3.Ref;

/// d (rows, n) = x (rows, k) w^T for w (n, k) in bf16.
fn gemm(m: *const Model, enc: mtl.ComputeEncoder, x: mtl.Buffer, w: Ref, d: mtl.Buffer, n: usize, k: usize) void {
    const tn = n / 128;
    const tm = (m.rows + 63) / 64;
    enc.setPipeline(if (m.rows % 64 == 0) m.pipeline(kernels.gemm_whole) else m.pipeline(kernels.gemm_ragged));
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(w.buffer, w.offset, 1);
    var p: [16]i32 = @splat(0);
    const values = [_]usize{ m.rows, n, k, k, k, n, tn, tm, 2, k / 256 };
    for (values, 0..) |v, i| p[i] = @intCast(v);
    enc.setBytes(std.mem.asBytes(&p), 2);
    enc.setBuffer(d, 0, 3);
    enc.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}

/// The stream's modulated norm to bf16 rows in xn; with `gates` it first adds the branch in `y` gated by that part.
fn norm(m: *const Model, enc: mtl.ComputeEncoder, w: Ref, gates: ?Ref, gate: i32, tables: Ref, scale: i32, shift: i32) void {
    const g = gates orelse tables;
    enc.setPipeline(m.pipeline("h3_norm"));
    enc.setBuffer(m.x, 0, 0);
    enc.setBuffer(m.y, 0, 1);
    enc.setBuffer(w.buffer, w.offset, 2);
    enc.setBuffer(g.buffer, g.offset, 3);
    enc.setBuffer(tables.buffer, tables.offset, 4);
    enc.setBuffer(m.line, 0, 5);
    enc.setValue([6]i32{ @intCast(m.rows), @intCast(m.hidden), @intCast(m.lines), if (gates == null) -1 else gate, scale, shift }, 6);
    enc.setValue([1]f32{h3.eps}, 7);
    enc.setBuffer(m.xn, 0, 8);
    enc.dispatchGroups(mtl.Size.of(m.rows, 1, 1), mtl.Size.of(32, 1, 1));
}

fn attend(m: *const Model, enc: mtl.ComputeEncoder, list: mtl.Buffer, queries: usize, keys: usize, first: usize, per_query: bool) void {
    enc.setPipeline(m.pipeline("h3_attention"));
    enc.setBuffer(m.tq, 0, 0);
    enc.setBuffer(m.tk, 0, 1);
    enc.setBuffer(m.tv, 0, 2);
    enc.setBuffer(list, 0, 3);
    enc.setBuffer(m.sizes.buffer, m.sizes.offset, 4);
    enc.setBuffer(m.row_of, 0, 5);
    enc.setValue([4]i32{ @intCast(queries), @intCast(keys), @intCast(first), @intFromBool(per_query) }, 6);
    enc.setValue([1]f32{1.0 / @sqrt(@as(f32, h3.head_dim))}, 7);
    enc.setBuffer(m.qr, 0, 8);
    enc.dispatchGroups(mtl.Size.of(queries, m.heads, 1), mtl.Size.of(256, 1, 1));
}

/// Block `index` of a forward; the block before left its MLP's rows in y, added here with this block's first norm.
pub fn run(m: *Model, pass: *h3.Pass, index: usize) void {
    const b = &m.blocks[index];
    const enc = &pass.enc;
    const heads = m.heads;
    const gates: ?Ref = if (index == 0) null else m.blocks[index - 1].tables;
    norm(m, enc.*, b.norm1, gates, h3.gate_m, b.tables, h3.scale_a, h3.shift_a);
    m.lap(pass, "norms");
    gemm(m, enc.*, m.xn, b.q, m.qr, m.inner, m.hidden);
    gemm(m, enc.*, m.xn, b.k, m.kr, m.inner, m.hidden);
    gemm(m, enc.*, m.xn, b.v, m.vr, m.inner, m.hidden);
    if (!m.dense) gemm(m, enc.*, m.xn, b.compress, m.cr, m.inner, m.hidden);
    m.lap(pass, "q k v gate");
    enc.setPipeline(m.pipeline("h3_heads"));
    enc.setBuffer(m.qr, 0, 0);
    enc.setBuffer(m.kr, 0, 1);
    enc.setBuffer(m.vr, 0, 2);
    enc.setBuffer(b.norm_q.buffer, b.norm_q.offset, 3);
    enc.setBuffer(b.norm_k.buffer, b.norm_k.offset, 4);
    enc.setBuffer(m.cos.buffer, m.cos.offset, 5);
    enc.setBuffer(m.sin.buffer, m.sin.offset, 6);
    enc.setBuffer(m.slot.buffer, m.slot.offset, 7);
    enc.setValue([4]i32{ @intCast(m.rows), @intCast(heads), @intCast(m.rot), 0 }, 8);
    enc.setValue([1]f32{h3.eps}, 9);
    enc.setBuffer(m.tq, 0, 10);
    enc.setBuffer(m.tk, 0, 11);
    enc.setBuffer(m.tv, 0, 12);
    enc.dispatchGroups(mtl.Size.of(m.rows, heads, 1), mtl.Size.of(32, 1, 1));
    m.lap(pass, "heads");
    if (m.dense) {
        attend(m, enc.*, m.all_tiles, m.tiles, m.tiles, 0, false);
        m.lap(pass, "attention");
    } else {
        enc.setPipeline(m.pipeline("h3_pool"));
        enc.setBuffer(m.tq, 0, 0);
        enc.setBuffer(m.tk, 0, 1);
        enc.setBuffer(m.tv, 0, 2);
        enc.setBuffer(m.sizes.buffer, m.sizes.offset, 3);
        enc.setBuffer(m.qp, 0, 4);
        enc.setBuffer(m.kp, 0, 5);
        enc.setBuffer(m.vp, 0, 6);
        enc.dispatchGroups(mtl.Size.of(m.tiles, heads, 1), mtl.Size.of(32, 1, 1));
        m.route(enc.*);
        m.lap(pass, "routing");
        // the prefix's query tiles see every tile; a video tile sees the prefix and its chosen tiles
        attend(m, enc.*, m.all_tiles, m.prefix_tiles, m.tiles, 0, false);
        attend(m, enc.*, m.chosen, m.tiles - m.prefix_tiles, m.prefix_tiles + m.keep, m.prefix_tiles, true);
        m.lap(pass, "attention");
        // the pooled branch gated into the attention rows in place
        enc.setPipeline(m.pipeline("h3_gate_mix"));
        enc.setBuffer(m.qr, 0, 0);
        enc.setBuffer(m.cr, 0, 1);
        enc.setBuffer(m.coarse, 0, 2);
        enc.setBuffer(m.slot.buffer, m.slot.offset, 3);
        enc.setValue([3]i32{ @intCast(m.rows), @intCast(heads), @intCast(m.tiles) }, 4);
        enc.dispatchThreads(mtl.Size.of(m.rows, heads, 1), mtl.Size.of(64, 4, 1));
        m.lap(pass, "gate mix");
    }
    gemm(m, enc.*, m.qr, b.out, m.y, m.hidden, m.inner);
    m.lap(pass, "attention out");
    norm(m, enc.*, b.norm2, b.tables, h3.gate_a, b.tables, h3.scale_m, h3.shift_m);
    m.lap(pass, "norms");
    // the first projection's rows are [value | gate], one product of twice the MLP's width
    gemm(m, enc.*, m.xn, b.fc1, m.wide, 2 * m.mlp, m.hidden);
    enc.barrier();
    enc.setPipeline(m.pipeline("h3_swiglu"));
    enc.setBuffer(m.wide, 0, 0);
    enc.setValue([2]i32{ @intCast(m.rows), @intCast(m.mlp) }, 1);
    enc.setBuffer(m.act, 0, 2);
    enc.dispatchThreads(mtl.Size.of(m.mlp, m.rows, 1), mtl.Size.of(256, 1, 1));
    m.lap(pass, "mlp in");
    gemm(m, enc.*, m.act, b.fc2, m.y, m.hidden, m.mlp);
    m.lap(pass, "mlp out");
}
