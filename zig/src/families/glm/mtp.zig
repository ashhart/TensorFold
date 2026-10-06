//! The MTP draft head (layer 45): enorm(embed(next)) | hnorm(h) -> eh_proj, then a plain MLA + MoE block, then
//! shared_head.norm and the LM head. Rows enter its own latent cache at the backbone positions they follow.
const std = @import("std");
const mtl = @import("metal");
const fwd = @import("forward.zig");
const wts = @import("weights.zig");
const Ref = wts.Ref;

fn add(x: *const fwd.Ctx, e: mtl.ComputeEncoder, a: Ref, b: Ref, out: Ref, n: u32) void {
    e.setPipeline(x.k.add);
    e.setBuffer(a.buf, a.off, 0);
    e.setBuffer(b.buf, b.off, 1);
    e.setBuffer(out.buf, out.off, 2);
    e.setValue(n, 3);
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

/// `rows` rows of h (final-normed backbone rows, or the head's own output rows when chaining) with the tokens after
/// them (`next`, u32) at head positions pos..: the head's output rows into m_x (pre-norm), drafts into `picks`.
pub fn run(x: *fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, next: Ref, rows: u32, pos: u32, picks: Ref) void {
    const c = x.c;
    const sc = x.sc;
    const D = c.hidden;
    const L = &x.w.layers[c.layers];
    const m = x.w.mtp.?;
    fwd.embedRows(x, e, next, sc.m_emb, rows);
    fwd.rms(x, e, sc.m_emb, m.enorm, sc.m_eh, rows, D, D, 2 * D, c.eps);
    fwd.rms(x, e, h, m.hnorm, sc.m_eh.at(@as(usize, D) * 2), rows, D, D, 2 * D, c.eps);
    fwd.qmv(x, e, x.k.qmv_kda_out, sc.m_eh, m.eh_proj, sc.m_x, rows);
    fwd.rms(x, e, sc.m_x, L.in_norm, sc.m_xn, rows, D, D, D, c.eps);
    fwd.mla(x, e, c.countKind(.mla), &L.attn.mla, sc.m_xn, rows, pos);
    add(x, e, sc.m_x, sc.branch, sc.m_out, rows * D);
    fwd.rms(x, e, sc.m_out, L.post_norm, sc.m_xn, rows, D, D, D, c.eps);
    fwd.moe(x, e, &L.mlp.moe, sc.m_xn, rows);
    add(x, e, sc.m_out, sc.branch, sc.m_x, rows * D);
    fwd.rms(x, e, sc.m_x, m.norm, sc.m_hn, rows, D, D, D, c.eps);
    fwd.head(x, e, sc.m_hn.at(@as(usize, rows - 1) * D * 2), sc.m_logits, picks, 1);
}

/// One chained draft from `h` (a row of the head's previous output, m_x) and the draft before it (`token`, u32).
pub fn chain(x: *fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, token: Ref, pos: u32, picks: Ref) void {
    run(x, e, h, token, 1, pos, picks);
}
