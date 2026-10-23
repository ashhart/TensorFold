//! nemotron_h_puzzle (Bolt) MoE in the moe_latent_size latent: mlx_lm NemotronHMoE's order, cuda_latent.zig run()'s.
const mtl = @import("metal");
const wts = @import("weights.zig");
const fwd = @import("forward.zig");
const layers = @import("layers.zig");

const Buffer = mtl.Buffer;
const Enc = fwd.Enc;
const Forward = fwd.Forward;

/// y [rows, n] = x W^T a row at a time in MLX's gemv at its tiles for (k, n) (tools/zig/bolt_op_cases.py: bit for bit).
pub fn gemv(f: Forward, e: *Enc, w: wts.Dense, x: Buffer, x_off: usize, y: Buffer, rows: usize) void {
    const wide = w.k >= 16 * w.n; // matmul.cpp: bn 8 along K, else bm 8 from N 4096 or bm 4, bn 1
    const bm: usize = if (wide) 1 else if (w.n >= 4096) 8 else 4;
    e.pipe(f.p.?.get(if (wide) "tf_gemv_bf16_tm4" else if (bm == 8) "tf_gemv_bf16_bm8_tm4" else "tf_gemv_bf16_bm4_tm4"));
    e.buf(w.w, w.off, 0);
    e.buf(x, x_off, 1);
    e.bytes([3]i32{ @intCast(w.k), @intCast(w.n), @intCast(w.k) }, 2);
    e.buf(y, 0, 3);
    const groups = (w.n + 4 * bm - 1) / (4 * bm);
    if (wide) e.run(.{ 32 * groups, 8, rows }, .{ 32, 8, 1 }) else e.run(.{ 32 * groups, 1, bm * rows }, .{ 32, 1, bm });
}

/// Gate, top-k, the shared expert in hidden, into, the routed experts in the latent, their combine, out, add_norm.
pub fn moe(f: Forward, e: *Enc, m: wts.Moe, next: anytype, rows: usize, eps: f32) void {
    const c = f.c;
    const s = f.s;
    const pairs = rows * c.top_k;
    const E = m.gate.shape[0]; // this layer's own routed-expert count (per-layer in the Puzzle)

    // router logits = x . gate^T in bf16, then mlx_lm's group_expert_select (sigmoid + bias top-k, normalised, scaled)
    gemv(f, e, .{ .w = m.gate.buffer, .off = m.gate.offset, .n = E, .k = c.hidden }, s.x, 0, s.logits_e, rows);
    e.pipe(f.p.?.get("tf_route_topk"));
    e.buf(s.logits_e, 0, 0);
    e.buf(m.gate_bias.buffer, m.gate_bias.offset, 1);
    e.bytes([2]f32{ 1e-20, c.routed_scaling }, 2);
    e.bytes([2]i32{ @intCast(E), @intCast(c.top_k) }, 3);
    e.buf(s.idx, 0, 4);
    e.buf(s.wt, 0, 5);
    e.run(.{ E, rows, 1 }, .{ E, 1, 1 });

    // the shared expert runs in hidden, as in the Lightning path
    layers.upRelu2(f, e, m.shared_up, rows);
    f.coop(e, "down", m.shared_down, s.sh_act, 0, s.sh_xs, s.sh, rows);

    // into: hidden -> latent, then the routed experts up/down in the latent (runtime-dim row kernels)
    gemv(f, e, m.into.?, s.x, 0, s.x_lat, rows);
    f.indexed(e, m.routed.?[0], c.top_k, pairs, true, s.x_lat, s.idx, s.act);
    f.indexed(e, m.routed.?[1], 1, pairs, false, s.act, s.idx, s.ey_lat);

    // the picks combined by weight in MLX's order ((y * scores).sum(-2) in fp32), into x_lat now the up has read it
    e.pipe(f.p.?.get("tf_moe_combine_rows"));
    e.buf(s.ey_lat, 0, 0);
    e.buf(s.wt, 0, 1);
    e.bytes([2]i32{ @intCast(c.top_k), @intCast(c.latent) }, 2);
    e.buf(s.x_lat, 0, 3);
    e.run(.{ c.latent, rows, 1 }, .{ 256, 1, 1 });

    // out: the combined latent back to hidden, once a row (delta is free in an MoE layer)
    gemv(f, e, m.out.?, s.x_lat, 0, s.delta, rows);

    // add_norm at one pick of weight 1: bf16(y + shared), bf16(h + y), the next input norm and its sums
    e.pipe(f.k.get("add_norm_moe_xs_bolt"));
    e.buf(s.h, 0, 0);
    e.buf(s.delta, 0, 1);
    e.buf(s.ones, 0, 2);
    e.buf(s.sh, 0, 3);
    e.buf(next.buffer, next.offset, 4);
    e.bytes(eps, 5);
    e.bytes(Forward.mdims(rows), 6);
    e.buf(s.h, 0, 7);
    e.buf(s.x, 0, 8);
    e.buf(s.xs, 0, 9);
    e.run(.{ 512 * Forward.mp(rows), 1, 1 }, .{ 512, 1, 1 });
}
