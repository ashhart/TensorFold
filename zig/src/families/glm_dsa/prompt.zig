//! A prompt window on the pair: every layer's projections and experts for all of the window's rows at once.
const mtl = @import("metal");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const ly = @import("layer.zig");
const lp = @import("layer_prompt.zig");
const Model = @import("model.zig").Model;
const Ref = kn.Ref;
const R = gpu.max_rows;

fn at(b: mtl.Buffer, row: u32, width: u32) Ref {
    return .{ .buf = b, .off = @as(usize, row) * width * 4 };
}

/// One decoder layer for the window's rows (consecutive positions from pos0): the exchanges cover them all.
fn layer(m: *Model, L: *const ly.Layer, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, rows: u32, pos0: u32, table: mtl.Buffer, next: Ref) !void {
    const p = m.pair;
    const hid = m.c.hidden;
    try lp.attention(&m.K, L, s, q, e, &m.c, rows, pos0, table, .{ .buf = q.x }, p.part());
    e.barrier();
    p.addRms(e, .{ .buf = q.h }, L.k.at(L.w.post_norm), .{ .buf = q.x }, rows, hid, m.c.eps);
    e.barrier();
    const o = p.part();
    if (L.w.mlp == .moe) {
        try lp.route(&m.K, L, s, q, e, &m.c, rows, .{ .buf = q.x });
        try lp.experts(&m.K, L, q, e, &m.c, rows, .{ .buf = q.x }, o);
    } else try lp.denseMlp(&m.K, L, q, e, &m.c, rows, .{ .buf = q.x }, o);
    e.barrier();
    p.addRms(e, .{ .buf = q.h }, next, .{ .buf = q.x }, rows, hid, m.c.eps);
    e.barrier();
}

/// The window's rows (q.ids at q.rows' positions from pos0) through every layer: q.h the hidden states, q.x normed.
pub fn backbone(m: *Model, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, rows: u32, pos0: u32) !void {
    const hid = m.c.hidden;
    const v = &m.vocab;
    kn.embed(&m.K.k, e, .{ .buf = q.ids }, v.k.at(v.embed_fp8), v.k.at(v.embed_scale), m.glue(hid, hid), .{ .buf = q.h }, rows);
    e.barrier();
    const L0 = &m.layers[0];
    kn.addRms(&m.K.k, e, .{ .buf = q.h }, .{ .buf = q.h }, L0.k.at(L0.w.in_norm), m.glue(hid, hid), .{ .buf = q.x }, rows);
    e.barrier();
    for (m.layers, 0..) |*L, i| {
        const next = if (i + 1 < m.layers.len) m.layers[i + 1].k.at(m.layers[i + 1].w.in_norm) else v.k.at(v.norm);
        try layer(m, L, s, q, e, rows, pos0, q.rows, next);
    }
}

/// The MTP layer over the window's rows: tokens q.ids_mtp, hidden rows m.head.prev, positions pos0 on in q.rows_mtp.
pub fn draft(m: *Model, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, rows: u32, pos0: u32) !void {
    const L = &(m.mtp orelse return error.NoMtpHead);
    const t = L.w.mtp.?;
    const hid = m.c.hidden;
    const v = &m.vocab;
    kn.embed(&m.K.k, e, .{ .buf = q.ids_mtp }, v.k.at(v.embed_fp8), v.k.at(v.embed_scale), m.glue(hid, hid), .{ .buf = q.h }, rows);
    e.barrier();
    kn.addRms(&m.K.k, e, .{ .buf = q.h }, .{ .buf = q.h }, L.k.at(t.enorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat }, rows);
    kn.addRms(&m.K.k, e, .{ .buf = m.head.prev }, .{ .buf = q.h }, L.k.at(t.hnorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat, .off = hid * 4 }, rows);
    e.barrier();
    try lp.linear(&m.K, L, e, t.eh, L.groups[ly.Layer.eh_g], .{ .buf = m.head.cat }, 2 * hid, .{ .buf = q.rot }, .{ .buf = q.h }, hid, rows);
    e.barrier();
    kn.addRms(&m.K.k, e, .{ .buf = q.h }, .{ .buf = q.h }, L.k.at(L.w.in_norm), m.glue(hid, hid), .{ .buf = q.x }, rows);
    e.barrier();
    try layer(m, L, s, q, e, rows, pos0, q.rows_mtp, L.k.at(t.head_norm));
}
