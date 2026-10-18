//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const mtl = @import("metal");
const dense = fz.dense;
const lane_w = fz.lane_w;
const rows_load = fz.rows_load;
const variants = fz.variants;
const D = fz.D;
const WIDE = fz.WIDE;
const MAXR = fz.MAXR;
const Buf = fz.Buf;
const POOL_BLOCKS = fz.POOL_BLOCKS;
const Hc = fz.Hc;
const Lane = fz.Lane;
const Slot = fz.Slot;
const Model = fz.Model;
const finish = Model.finish;

pub fn mtpRun(m: *Model, nexts: []const u32, streams: Buf) !u32 {
    const r = m.r;
    const h = &m.mtp;
    const rows = nexts.len;
    h.pos -= h.drafted;
    h.drafted = 0;
    h.pooled_n = @min(h.pooled_n, h.pos / 4); // blocks dropped rows completed are pooled again
    const ids = h.slots[0].ids8.b.slice(u32, MAXR);
    for (0..MAXR) |i| ids[i] = if (i < rows) nexts[i] else 0;
    const cb = r.queue.commandBuffer();
    r.enc = cb.compute(if (r.serial) .serial else .concurrent);
    try m.mtpEncode(0, rows, h.slots[0].ids8, streams, h.pick);
    try m.finish(cb);
    h.pos += rows;
    return h.pick.b.slice(u32, 1)[0];
}

pub fn mtpEncode(m: *Model, slot: usize, rows: usize, ids: Buf, streams: Buf, out: Buf) !void {
    const r = m.r;
    const h = &m.mtp;
    const sl = &h.slots[slot];
    if (!r.gpu_round) try m.mtpMeta(sl, rows);
    try m.mtpLayer(sl, rows, ids, streams, out);
}

pub fn mtpMeta(m: *Model, sl: *Slot, rows: usize) !void {
    const h = &m.mtp;
    sl.rows.b.slice(i32, 1)[0] = @intCast(rows);
    sl.md.b.slice(i32, 2)[0] = @intCast(rows);
    sl.md4.b.slice(i32, 2)[0] = @intCast(4 * rows);
    sl.md4.b.slice(i32, 2)[1] = @intCast(16 * ((4 * rows + 15) / 16)); // rows padded to whole 16-row tiles
    const pos8 = sl.pos8.b.slice(i32, MAXR);
    const nk8 = sl.nk8.b.slice(i32, MAXR);
    for (0..MAXR) |i| {
        pos8[i] = if (i < rows) @intCast(h.pos + i) else 0;
        nk8[i] = if (i < rows) @intCast(h.pos + i + 1) else 0;
    }
    const kvm = sl.kvmeta.b.slice(u32, 3);
    kvm[0], kvm[1], kvm[2] = .{ @intCast(h.pos), @intCast(m.cap), @intCast(rows) };
    sl.n_add.b.slice(u32, 1)[0] = @intCast(rows * WIDE);
}

pub fn mtpLayer(m: *Model, sl: *Slot, rows: usize, ids: Buf, streams: Buf, out: Buf) !void {
    const r = m.r;
    const t = &m.t;
    const h = &m.mtp;
    r.rows = rows;
    var ce: [2]variants.Const = undefined;
    try variants.callAs(r, "mtp:qa_embed_rows@embed", rows, &.{ ids, m.embed[0], m.embed[1], m.embed[2] }, &.{h.emb}, variants.width(&ce, "BITS", "GS", m.embed_w));
    try r.call("mtp:q4_rms_rows@mtp.enorm", &.{ h.emb, h.enorm, t.eps }, &.{h.en});
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ h.en, sl.md }, &.{t.xs});
    try m.mtpLane(h.fce, "mtp:lane_qmm_bytes_grouped@mtp.fce", h.en, sl.md, rows, D, h.e);
    try r.call("mtp:q4_rms_rows@mtp.hnorm", &.{ streams, h.hnorm, t.eps }, &.{h.hn});
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[4R, 2560]", &.{ h.hn, sl.md4 }, &.{t.xs});
    try m.mtpLane(h.fch, "mtp:lane_qmm_bytes_grouped@mtp.fch", h.hn, sl.md4, 4 * rows, D, h.hs);
    r.enc.setPipeline(r.add_pipe);
    for ([_]Buf{ h.e, h.hs, h.h[0], sl.n_add }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    r.enc.dispatchThreads(mtl.Size.of(rows * WIDE, 1, 1), mtl.Size.of(256, 1, 1));
    if (!r.serial) r.enc.barrier();
    try r.call("mtp:q4_hc_norm_none#[10240]", &.{h.h[0]}, &.{ h.h[1], t.ssp });
    const down = [_][]const u8{ "mtp:qa_hc_down@mtp.ahc", "mtp:qa_hc_down@mtp.mhc", "mtp:qa_hc_down@mtp.mix" };
    const up = [_][]const u8{ "mtp:qa_hc_up@mtp.ahc", "mtp:qa_hc_up@mtp.mhc", "mtp:qa_hc_up@mtp.mix" };
    try m.mtpProject(h.h[1], h.ahc, down[0], up[0], t.inj_a, sl.rows);
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ t.mixed, sl.md }, &.{t.xs});
    try m.mtpLane(h.proj, "mtp:lane_qmm_bytes_grouped@mtp.att.proj", t.mixed, sl.md, rows, D, t.p);
    try r.call("mtp:q4_attn_prep@mtp.att", &.{ t.p, sl.pos8, h.qn, h.kn, h.iqn, t.eps, t.log2base }, &.{ t.q, t.kout, t.iq });
    r.enc.setPipeline(r.kv_pipe);
    for ([_]Buf{ t.kout, t.p, h.keys, h.vals, h.raw, sl.kvmeta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    r.enc.dispatchThreads(mtl.Size.of(512 * rows, 1, 1), mtl.Size.of(256, 1, 1));
    if (!r.serial) r.enc.barrier();
    if (r.gpu_round and h.gsel != null) { // the call's selection on the GPU, from its own first position
        const g = &h.gsel.?;
        const ss = &r.sel.?;
        r.enc.setPipeline(r.sel_meta_pipe);
        r.enc.setBuffer(sl.pos8.b, sl.pos8.off - 4, 0); // fz_sel_meta reads the first position at [1]
        const wu: u32 = @intCast(rows);
        r.enc.setBytes(std.mem.asBytes(&wu), 1);
        r.enc.setBuffer(g.sel.b, 0, 2);
        r.enc.setBuffer(g.pooled_shape, 0, 3);
        r.enc.setBuffer(g.sc_shape, 0, 4);
        r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        if (!r.serial) r.enc.barrier();
        r.enc.setPipeline(r.pool_abs_pipe);
        for ([_]Buf{ h.raw, g.sel, h.pool, t.eps, t.log2base, h.pooled }, 0..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
        r.enc.dispatchThreads(mtl.Size.of(128, POOL_BLOCKS, 1), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
        try r.shapes.put(r.arena, "POOLED_shape", g.pooled_shape);
        try r.shapes.put(r.arena, "Q_shape", g.q_shape[rows]);
        try r.shapes.put(r.arena, "SC_shape", g.sc_shape);
        try r.bindV(ss.scores, &.{ t.iq, h.pooled, g.view(0) }, &.{g.sc});
        r.enc.dispatchThreads(mtl.Size.of(((g.nb_ub + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
        try r.bindV(ss.select, &.{ g.sc, g.view(0), g.view(MAXR) }, &.{g.keys});
        r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
        const dense_ids = r.shapes.get("IDS_shape").?;
        try r.shapes.put(r.arena, "IDS_shape", g.ids_shape[rows]);
        try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, g.keys, g.view(2 * MAXR), g.view(3 * MAXR), t.scale }, &.{ t.po, t.pm });
        try r.shapes.put(r.arena, "IDS_shape", dense_ids);
    } else if (r.sel != null and !r.gpu_round and r.sel.?.meta(h.pos, rows)) { // one call a command buffer
        var ss = &r.sel.?;
        try ss.encode(r, h, t.iq, t.eps, t.log2base, h.pos, rows);
        const dense_ids = r.shapes.get("IDS_shape").?;
        try r.shapes.put(r.arena, "IDS_shape", ss.ids_shape);
        try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, ss.keys, ss.counts, ss.sparse, t.scale }, &.{ t.po, t.pm });
        try r.shapes.put(r.arena, "IDS_shape", dense_ids);
    } else try r.call("mtp:q4_attn_parts#[24, 256]", &.{ t.q, h.keys, h.vals, t.ids81, sl.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
    try r.call("mtp:q4_attn_merge_gate#[24, 16, 256]", &.{ t.po, t.pm, t.p }, &.{t.aout});
    if (rows > 1) { // only the kept row goes on: the host's last row, or the row the GPU's verdict kept, into row 0
        const keep: Buf = if (r.gpu_round) r.ar else sl.rows;
        r.enc.setPipeline(r.take_pipe);
        for ([_]Buf{ t.aout, h.h[1], t.inj_a, keep }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchThreads(mtl.Size.of(WIDE, 1, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
    }
    r.rows = 1;
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[6144]", &.{ t.aout, h.md1 }, &.{t.xs});
    try m.mtpLane(h.out, "mtp:lane_qmm_bytes_grouped@mtp.att.o", t.aout, h.md1, 1, 6144, t.branch);
    try r.call("mtp:q4_hc_norm_plain#[10240]", &.{ h.h[1], t.inj_a, t.branch }, &.{ h.h[0], t.ssp });
    try m.mtpProject(h.h[0], h.mhc, down[1], up[1], t.inj_m, h.md1);
    try r.router("mtp:q4_router_float@mtp.moe", t.mixed, h.router, h.md1, t.lg, 1);
    if (h.xw) |w| try variants.experts(r, "mtp:qa_expert_gateup@mtp.moe.gate", "mtp:qa_expert_down_y@mtp.moe.down", t.mixed, t.lg, h.ex, t.act, t.pick, t.wts, h.md1, t.ydown, w) else try r.experts("mtp:qa_expert_gateup@mtp.moe.gate", "mtp:qa_expert_down_y@mtp.moe.down", t.mixed, t.lg, h.ex, t.act, t.pick, t.wts, h.md1, t.ydown);
    try r.call("mtp:q4_hc_norm_grouped#[10240]", &.{ h.h[0], t.inj_m, t.ydown, t.wts, t.lg }, &.{ h.h[1], t.ssp });
    h.last = h.h[1];
    try m.mtpProject(h.h[1], h.mix, down[2], up[2], t.inj_a, h.md1);
    const x = t.mixed;
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ x, h.md1 }, &.{t.xs});
    // TP: this Mac's half of the draft vocabulary's tiles, then one exact argmax swap
    if (r.tp != null and r.dense and h.draft.w == null) {
        const tp = r.tp.?;
        const six = try h.draft.recorded();
        const n = six.wq.b.length() * 4 / (3 * D);
        const tiles = n / 32;
        const half = (tiles + 1) / 2;
        const t0: usize = if (tp.rank == 0) 0 else half;
        const tn: usize = if (tp.rank == 0) half else tiles - half;
        r.denseTiles(x, D, six, h.logits, t0, tn);
        const lo = @min(t0 * 32, h.ids_n);
        const hi = @min((t0 + tn) * 32, h.ids_n);
        tp.argmax(r.enc, h.logits, n, lo, hi - lo, Buf{ .b = tp.pick_tmp }, Buf{ .b = tp.one }, 1);
        tp.mapIds(r.enc, h.ids, out);
        return;
    }
    try m.mtpLane(h.draft, "mtp:lane_qmm_bytes_grouped@mtp.draft", x, h.md1, 1, D, h.logits);
    r.enc.setPipeline(r.argids_pipe);
    for ([_]Buf{ h.logits, out, h.n_ids, h.ids }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
    if (!r.serial) r.enc.barrier();
}

pub fn mtpLane(m: *Model, l: Lane, role: []const u8, x: Buf, md: Buf, rows: usize, k: usize, y: Buf) !void {
    if (l.rows) |d| if (m.r.tp == null) return rows_load.launch(m.r, d, role, x, rows, y);
    if (l.w) |w| return lane_w.launch(m.r, w, role, x, md, rows, y);
    const six = try l.recorded();
    if (m.r.dense) return m.r.denseRows(x, k, six, rows, y);
    try m.r.call(role, &.{ x, m.t.xs, six.wq, six.sbt, md }, &.{y});
}

pub fn mtpProject(m: *Model, hn: Buf, hc: Hc, down: []const u8, up: []const u8, inj: Buf, rows: Buf) !void {
    const t = &m.t;
    if (hc.rows) |sh| if (m.r.tp == null) return rows_load.hc(
        m.r,
        sh,
        hc,
        down,
        hn,
        t.ssp,
        t.eps,
        rows,
        m.r.rows,
        t.part,
        t.mixed,
        inj,
    );
    var c: [2][2]variants.Const = undefined;
    const k = [2][]const variants.Const{ variants.width(&c[0], "BITS", "GS", if (hc.w) |w| w[0] else null), variants.width(&c[1], "BITS", "GS", if (hc.w) |w| w[1] else null) };
    try variants.callAs(m.r, down, m.r.rows, &.{ hn, t.ssp, hc.scale, hc.dw, hc.ds, hc.db, t.eps, rows }, &.{t.part}, k[0]);
    try variants.callAs(m.r, up, m.r.rows, &.{ hn, t.ssp, hc.scale, t.part, hc.uw, hc.us, hc.ub, t.eps, rows }, &.{ t.mixed, inj }, k[1]);
}

pub fn mtpAbsorb(m: *Model, nexts: []const u32, streams: Buf) !u32 {
    var at: usize = 0;
    var d: u32 = 0;
    while (at < nexts.len) {
        const n = @min(MAXR, nexts.len - at);
        d = try m.mtpRun(nexts[at .. at + n], .{ .b = streams.b, .off = streams.off + at * WIDE * 2 });
        at += n;
    }
    return d;
}

pub fn mtpChain(m: *Model, draft: u32) !u32 {
    const prev = m.mtp.last;
    const rows_before = m.mtp.drafted;
    m.mtp.drafted = 0;
    const d = try m.mtpRun(&.{draft}, prev);
    m.mtp.drafted = rows_before + 1;
    return d;
}
