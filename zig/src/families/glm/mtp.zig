//! The MTP draft head (layer 45): enorm(embed(next)) | hnorm(h), eh_proj, a plain MLA + MoE block, its norm and the LM head.
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

/// `rows` rows of h with their next tokens at head positions pos..: output rows into m_x (pre-norm), drafts into `picks`.
pub fn run(x: *fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, next: Ref, rows: u32, pos: u32, picks: ?Ref) void {
    layer(x, e, h, next, rows, pos);
    const out = picks orelse return; // the rows absorbed into the head's cache only, no draft
    drafts(x, e, x.sc.m_x.at(@as(usize, rows - 1) * x.c.hidden * 2), 1, out);
}

/// The head's block over `rows` rows of h and their next tokens (positions pos.., or each segment's): m_x, pre-norm.
pub fn layer(x: *fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, next: Ref, rows: u32, pos: u32) void {
    absorb(x, e, h, next, rows, pos);
    finish(x, e, x.sc.m_x, x.sc.m_xn, rows, pos, false);
}

/// The rows' entries in the head's cache: embed, norms and eh_proj into m_x, its norm into m_xn, then the key writes.
pub fn absorb(x: *fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, next: Ref, rows: u32, pos: u32) void {
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
    fwd.mlaKeys(x, e, c.countKind(.mla), &L.attn.mla, sc.m_xn, rows, pos);
}

/// The rest of the block for absorbed rows (`res`: m_x's, `normed`: m_xn's; `gathered`: their x_proj again): m_x.
pub fn finish(x: *fwd.Ctx, e: mtl.ComputeEncoder, res: Ref, normed: Ref, rows: u32, pos: u32, gathered: bool) void {
    const c = x.c;
    const sc = x.sc;
    const D = c.hidden;
    const L = &x.w.layers[c.layers];
    const mi = c.countKind(.mla);
    if (gathered) { // the rows' queries and indexer weights, as the absorb computed them where the rows were
        fwd.qmv(x, e, x.k.qmv_x, normed, L.attn.mla.x_proj, sc.xp, rows);
        fwd.indexWeights(x, e, sc.xp, c.xProj(), sc.iw, rows);
    }
    fwd.mlaAttend(x, e, mi, &L.attn.mla, rows, pos);
    add(x, e, res, sc.branch, sc.m_out, rows * D);
    fwd.rms(x, e, sc.m_out, L.post_norm, sc.m_xn, rows, D, D, D, c.eps);
    var local = x.*; // every Mac holds the MTP layer's experts whole: its MoE is one Mac's, with no exchange
    local.ep = null;
    fwd.moe(&local, e, &L.mlp.moe, sc.m_xn, rows);
    add(x, e, sc.m_out, sc.branch, sc.m_x, rows * D);
}

/// Drafts for `rows` pre-norm head outputs at `out`: the head's norm, then the LM head over the draft vocabulary.
pub fn drafts(x: *const fwd.Ctx, e: mtl.ComputeEncoder, out: Ref, rows: u32, picks: Ref) void {
    const c = x.c;
    const D = c.hidden;
    fwd.rms(x, e, out, x.w.mtp.?.norm, x.sc.m_hn, rows, D, D, D, c.eps);
    fwd.headOver(x, e, x.sc.m_hn, x.sc.m_logits.at(@as(usize, x.m_row) * c.vocab * 2), picks, rows, x.draft_vocab);
}

/// One chained draft from `h` (a row of the head's previous output, m_x) and the draft before it (`token`, u32).
pub fn chain(x: *fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, token: Ref, pos: u32, picks: Ref) void {
    run(x, e, h, token, 1, pos, picks);
}
