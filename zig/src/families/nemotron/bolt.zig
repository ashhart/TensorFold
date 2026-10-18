//! nemotron_h_puzzle (Bolt) MoE in the moe_latent_size latent: mlx_lm NemotronHMoE's order, cuda_latent.zig run()'s.
const wts = @import("weights.zig");
const fwd = @import("forward.zig");
const layers = @import("layers.zig");

const Enc = fwd.Enc;
const Forward = fwd.Forward;

/// Gate, top-k, the shared expert in hidden, into, the routed experts in the latent, their combine, out, add_norm.
pub fn moe(f: Forward, e: *Enc, m: wts.Moe, next: anytype, rows: usize, eps: f32) void {
    const c = f.c;
    const s = f.s;
    const pairs = rows * c.top_k;
    const E: u32 = @intCast(m.gate.shape[0]); // this layer's own routed-expert count (per-layer in the Puzzle)
    const K: u32 = @intCast(c.top_k);
    const hidden: u32 = @intCast(c.hidden);
    const latent: u32 = @intCast(c.latent);

    // router logits = x . gate^T (gate bf16 [E, hidden], full precision)
    e.pipe(f.k.get("dense_bf16"));
    e.buf(s.x, 0, 0);
    e.buf(m.gate.buffer, m.gate.offset, 1);
    e.buf(s.logits_e, 0, 2);
    e.bytes([3]u32{ @intCast(rows), E, hidden }, 3);
    e.run(.{ 32 * E, (rows + 7) / 8, 1 }, .{ 256, 1, 1 });

    // sigmoid + e_score_correction_bias top-k (uniform K), weights normalised and scaled
    e.pipe(f.k.get("route_bolt"));
    e.buf(s.logits_e, 0, 0);
    e.buf(m.gate_bias.buffer, m.gate_bias.offset, 1);
    e.bytes(c.routed_scaling, 2);
    e.buf(s.idx, 0, 3);
    e.buf(s.wt, 0, 4);
    e.bytes(E, 5);
    e.bytes(K, 6);
    e.run(.{ 32 * rows, 1, 1 }, .{ 32, 1, 1 });

    // the shared expert runs in hidden, as in the Lightning path
    layers.upRelu2(f, e, m.shared_up, rows);
    f.coop(e, "down", m.shared_down, s.sh_act, 0, s.sh_xs, s.sh, rows);

    // into: hidden -> latent, then the routed experts up/down in the latent (runtime-dim row kernels)
    e.pipe(f.k.get("dense_bf16"));
    e.buf(s.x, 0, 0);
    e.buf(m.into.?.w, m.into.?.off, 1);
    e.buf(s.x_lat, 0, 2);
    e.bytes([3]u32{ @intCast(rows), latent, hidden }, 3);
    e.run(.{ 32 * latent, (rows + 7) / 8, 1 }, .{ 256, 1, 1 });
    f.indexed(e, m.routed.?[0], c.top_k, pairs, true, s.x_lat, s.idx, s.act);
    f.indexed(e, m.routed.?[1], 1, pairs, false, s.act, s.idx, s.ey_lat);

    // the picks combined by weight in the latent ((y * scores).sum(-2) in fp32), into x_lat now the up has read it
    e.pipe(f.k.get("combine_latent"));
    e.buf(s.ey_lat, 0, 0);
    e.buf(s.wt, 0, 1);
    e.buf(s.x_lat, 0, 2);
    e.bytes([3]u32{ @intCast(rows), K, latent }, 3);
    e.run(.{ latent, rows, 1 }, .{ 256, 1, 1 });

    // out: the combined latent back to hidden, once a row (delta is free in an MoE layer)
    e.pipe(f.k.get("dense_bf16"));
    e.buf(s.x_lat, 0, 0);
    e.buf(m.out.?.w, m.out.?.off, 1);
    e.buf(s.delta, 0, 2);
    e.bytes([3]u32{ @intCast(rows), hidden, latent }, 3);
    e.run(.{ 32 * hidden, (rows + 7) / 8, 1 }, .{ 256, 1, 1 });

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
