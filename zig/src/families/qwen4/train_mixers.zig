//! Native mixer reverse passes consume recorded causal state and fixed QSA selections, never recomputed rankings.
const core = @import("core");
pub fn gdn(t: anytype, l: usize, w: anytype) !void {
    const c = &t.e.c;
    const o = t.ops();
    const raw = try t.tape.indexed(l, "linear_attn.projections", t.rows);
    const state = try t.tape.indexed(l, "linear_attn.recurrent", t.rows);
    try o.matrix(w.out, t.buf(.branch), t.buf(.mixer), t.rows, 0, 0, false);
    try o.zero(t.buf(.projection));
    var p: core.living_gdn.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(c.key_heads);
    p.u[2] = @intCast(c.value_heads);
    p.u[3] = @intCast(c.convWidth() + c.valueWidth() + 2 * c.value_heads);
    p.u[4] = @intFromBool(w.conv_bf16);
    p.u[5] = @intFromBool(c.gate == .sigmoid);
    p.u[6] = @intFromBool(w.norm.mid_bf16);
    p.f[0] = c.eps;
    try o.gdn(.front, p, &.{ .{ 0, raw }, .{ 1, o.raw(w.conv) }, .{ 8, t.buf(.gdn_act) } });
    try o.gdn(.norm, p, &.{ .{ 0, t.buf(.gdn_act) }, .{ 8, t.buf(.gdn_qk) } });
    try o.gdn(.readout, p, &.{ .{ 0, raw }, .{ 1, t.buf(.gdn_qk) }, .{ 2, state }, .{ 3, o.raw(w.norm.buffer) }, .{ 4, t.buf(.mixer) }, .{ 8, t.buf(.gdn_gy) }, .{ 9, t.buf(.projection) } });
    try o.gdn(.scan, p, &.{ .{ 0, t.buf(.gdn_act) }, .{ 1, t.buf(.gdn_qk) }, .{ 2, state }, .{ 3, t.buf(.gdn_gy) }, .{ 4, raw }, .{ 5, o.raw(w.alog) }, .{ 6, o.raw(w.dt) }, .{ 8, t.buf(.gdn_parts) }, .{ 9, t.buf(.gdn_work) } });
    try o.gdn(.qk, p, &.{ .{ 0, t.buf(.gdn_act) }, .{ 2, t.buf(.gdn_parts) }, .{ 8, t.buf(.gdn_input) } });
    try o.gdn(.gates, p, &.{ .{ 0, t.buf(.gdn_parts) }, .{ 1, raw }, .{ 2, o.raw(w.alog) }, .{ 3, o.raw(w.dt) }, .{ 8, t.buf(.projection) } });
    try o.gdn(.conv, p, &.{ .{ 0, raw }, .{ 1, o.raw(w.conv) }, .{ 2, t.buf(.gdn_input) }, .{ 8, t.buf(.projection) } });
    const stride = p.u[3];
    try o.matrix(w.qkv, t.buf(.projection), t.buf(.mixed), t.rows, stride, 0, false);
    try o.matrix(w.z, t.buf(.projection), t.buf(.mixed), t.rows, stride, c.convWidth(), true);
    try o.matrix(w.b, t.buf(.projection), t.buf(.mixed), t.rows, stride, c.convWidth() + c.valueWidth(), true);
    try o.matrix(w.a, t.buf(.projection), t.buf(.mixed), t.rows, stride, c.convWidth() + c.valueWidth() + c.value_heads, true);
}
pub fn qsa(t: anytype, l: usize, w: anytype) !void {
    const c = &t.e.c;
    const o = t.ops();
    var problem: @import("config.zig").Problem = .{};
    const g = try @import("cuda_attention.zig").Geometry.of(c, &problem);
    const raw = try t.tape.indexed(l, "self_attn.projections", t.rows);
    const q = try t.tape.indexed(l, "self_attn.query", t.rows);
    const k = try t.tape.indexed(l, "self_attn.key", t.rows);
    const v = try t.tape.indexed(l, "self_attn.value", t.rows);
    const ids = try t.tape.raw(l, "self_attn.selected_ids", t.rows);
    const counts = try t.tape.indexed(l, "self_attn.selection_count", t.rows);
    const sparse = try t.tape.indexed(l, "self_attn.selection_sparse", t.rows);
    try o.matrix(w.out, t.buf(.branch), t.buf(.mixer), t.rows, 0, 0, false);
    try o.zero(t.buf(.projection));
    var p: core.living_rows.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(c.heads * c.head_dim);
    p.u[2] = @intCast(c.head_dim);
    p.u[3] = @intCast(g.projectionWidth());
    try o.row(.gate, p, &.{ .{ 0, try t.tape.indexed(l, "self_attn.attention", t.rows) }, .{ 1, raw }, .{ 2, t.buf(.mixer) }, .{ 8, t.buf(.qsa_context) }, .{ 9, t.buf(.projection) } });
    var ap: core.living_qsa.Params = .{};
    ap.u[0] = @intCast(t.rows);
    ap.u[1] = @intCast(c.heads);
    ap.u[2] = @intCast(c.kv_heads);
    ap.u[3] = @intCast(c.head_dim);
    ap.u[4] = @intCast(try g.ids());
    ap.u[5] = @intFromBool(t.runtime.kernels.probabilityCut(&t.e.gpu));
    ap.f[0] = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    try selected(t, ap, q, k, v, ids, counts, sparse);
    const frequency = try t.runtime.kernels.frequency(&t.e.gpu, c.rotary / 2 * 4);
    p.u[1] = @intCast(c.heads);
    p.u[3] = @intCast(c.rotary);
    p.u[4] = @intCast(g.projectionWidth());
    p.u[5] = 0;
    p.u[6] = @intCast(2 * c.head_dim);
    p.f[0] = c.eps;
    try o.row(.rope_rms, p, &.{ .{ 0, raw }, .{ 1, o.raw(w.qn.buffer) }, .{ 2, t.buf(.qsa_q) }, .{ 3, frequency }, .{ 8, t.buf(.projection) } });
    p.u[1] = @intCast(c.kv_heads);
    p.u[5] = @intCast(2 * c.heads * c.head_dim);
    p.u[6] = @intCast(c.head_dim);
    try o.row(.rope_rms, p, &.{ .{ 0, raw }, .{ 1, o.raw(w.kn.buffer) }, .{ 2, t.buf(.qsa_k) }, .{ 3, frequency }, .{ 8, t.buf(.projection) } });
    const kv = c.kv_heads * c.head_dim;
    for (0..t.rows) |row| try o.copy(try t.buf(.qsa_v).plus(row * kv * 4, kv * 4), try t.buf(.projection).plus((row * g.projectionWidth() + 2 * c.heads * c.head_dim + kv) * 4, kv * 4));
    try o.matrix(w.q, t.buf(.projection), t.buf(.mixed), t.rows, g.projectionWidth(), 0, false);
    try o.matrix(w.k, t.buf(.projection), t.buf(.mixed), t.rows, g.projectionWidth(), 2 * c.heads * c.head_dim, true);
    try o.matrix(w.v, t.buf(.projection), t.buf(.mixed), t.rows, g.projectionWidth(), 2 * c.heads * c.head_dim + kv, true);
}

/// The native selected-attention chain is shared with its independent prepared-node regression.
pub fn selected(t: anytype, ap: core.living_qsa.Params, q: anytype, k: @TypeOf(q), v: @TypeOf(q), ids: @TypeOf(q), counts: @TypeOf(q), sparse: @TypeOf(q)) !void {
    const o = t.ops();
    try o.qsa(.scores, ap, &.{ .{ 0, q }, .{ 1, k }, .{ 2, v }, .{ 3, t.buf(.qsa_context) }, .{ 4, ids }, .{ 5, counts }, .{ 6, sparse }, .{ 8, t.buf(.qsa_scores) }, .{ 9, t.buf(.qsa_dp) } });
    try o.qsa(.softmax, ap, &.{ .{ 0, t.buf(.qsa_scores) }, .{ 1, t.buf(.qsa_dp) }, .{ 2, ids }, .{ 3, counts }, .{ 4, sparse }, .{ 8, t.buf(.qsa_p) }, .{ 9, t.buf(.qsa_ds) } });
    try o.qsa(.query, ap, &.{ .{ 0, t.buf(.qsa_ds) }, .{ 1, k }, .{ 2, ids }, .{ 3, counts }, .{ 4, sparse }, .{ 8, t.buf(.qsa_q) } });
    try o.qsa(.key_value, ap, &.{ .{ 0, t.buf(.qsa_ds) }, .{ 1, t.buf(.qsa_p) }, .{ 2, q }, .{ 3, t.buf(.qsa_context) }, .{ 4, ids }, .{ 5, counts }, .{ 6, sparse }, .{ 8, t.buf(.qsa_k) }, .{ 9, t.buf(.qsa_v) } });
}
