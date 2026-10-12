//! A prompt window on the pair: attention 16 rows at a time, then every expert once for all of the window's rows.
const mtl = @import("metal");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const ly = @import("layer.zig");
const Model = @import("model.zig").Model;
const Ref = kn.Ref;
const R = gpu.max_rows;

/// The window's 16-row parts: part k's rows table entries and its rows' offsets.
fn part(s: *const gpu.Scratch, table: mtl.Buffer, k: u32) gpu.Scratch {
    var sk = s.*;
    sk.rows = table;
    sk.rows_off = @as(usize, k) * @sizeOf(kn.Row);
    return sk;
}

fn at(b: mtl.Buffer, row: u32, width: u32) Ref {
    return .{ .buf = b, .off = @as(usize, row) * width * 4 };
}

/// One decoder layer for the window's rows (consecutive positions from pos0): the exchanges cover them all.
fn layer(m: *Model, L: *const ly.Layer, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, rows: u32, pos0: u32, table: mtl.Buffer, next: Ref) !void {
    const p = m.pair;
    const hid = m.c.hidden;
    const a = p.part();
    var k: u32 = 0;
    while (k < rows) : (k += R) {
        const n = @min(R, rows - k);
        const sk = part(s, table, k);
        try ly.attention(&m.K, L, &sk, e, &m.c, n, pos0 + k + n - 1, null, at(q.x, k, hid), .{ .buf = a.buf, .off = a.off + @as(usize, k) * hid * 4 });
        e.barrier();
    }
    p.addRms(e, .{ .buf = q.h }, L.k.at(L.w.post_norm), .{ .buf = q.x }, rows, hid, m.c.eps);
    e.barrier();
    const o = p.part();
    k = 0;
    while (k < rows) : (k += R) {
        const n = @min(R, rows - k);
        const out: Ref = .{ .buf = o.buf, .off = o.off + @as(usize, k) * hid * 4 };
        if (L.w.mlp == .moe) {
            try ly.route(&m.K, L, s, e, &m.c, n, at(q.x, k, hid), at(q.pick, k, m.c.topk), at(q.wts8, k, m.c.topk));
        } else try ly.denseMlp(&m.K, L, s, e, &m.c, n, at(q.x, k, hid), out);
        e.barrier();
    }
    if (L.w.mlp == .moe) try ly.expertsPrompt(&m.K, L, q, e, &m.c, rows, .{ .buf = q.x }, o);
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
    var k: u32 = 0;
    while (k < rows) : (k += R) {
        const n = @min(R, rows - k);
        try ly.linear(&m.K, L, e, t.eh, L.groups[ly.Layer.eh_g], at(m.head.cat, k, 2 * hid), 2 * hid, .{ .buf = s.rot }, at(q.h, k, hid), hid, n);
        e.barrier();
    }
    kn.addRms(&m.K.k, e, .{ .buf = q.h }, .{ .buf = q.h }, L.k.at(L.w.in_norm), m.glue(hid, hid), .{ .buf = q.x }, rows);
    e.barrier();
    try layer(m, L, s, q, e, rows, pos0, q.rows_mtp, L.k.at(t.head_norm));
}
