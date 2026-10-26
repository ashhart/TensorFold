//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const segments = @import("../../core/segments.zig");
const tpm = @import("tp.zig");
const mark_slots = @import("marks.zig");
const variants = fz.variants;
const Tp2 = fz.Tp2;
const vis_prompt = fz.vis_prompt;
const D = fz.D;
const WIDE = fz.WIDE;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const Buf = fz.Buf;
const Lane = fz.Lane;
const MARKS = fz.MARKS;
const linearIndex = mark_slots.linearIndex;
const Model = fz.Model;
const PMAX = fz.PMAX;
const Prompt = fz.Prompt;
const barrier = Prompt.barrier;
const partials = Prompt.partials;
const bind = Prompt.bind;
const qmm = Prompt.qmm;
const hc = Prompt.hc;
const moe = Prompt.moe;
const copyWords = Prompt.copyWords;

pub const Seg = struct {
    p: *Prompt,
    rows: usize,
    pos: usize,
    ra: usize,
    rr: usize,
    wa: usize,
    cur: usize,
    pending: bool,
    mk: [MARKS]i32 = @splat(0), // segment rows after which the DeltaNet layers write their states to m.marks (0: none)
};

pub fn bindMarks(p: *Prompt, m: *const Model, mk: [MARKS]i32, li: usize, rows_at: usize, buf_at: usize, scan: bool) void {
    const r = p.r;
    r.enc.setBytes(std.mem.asBytes(&mk), rows_at);
    const fallback = m.layers[0].cs[0];
    const b = if (m.marks) |k| (if (scan) k.so_at(li, 0) else k.cs_at(li, 0)) else fallback;
    r.enc.setBuffer(b.b, b.off, buf_at);
}

pub fn prep(p: *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32, pos: usize, hist: [2]i64) !void {
    const b = &p.b;
    const rows = tokens.len;
    @memcpy(b.ids.b.slice(u32, rows), tokens);
    for (0..rows) |i| {
        b.pos.b.slice(i32, PMAX)[i] = @intCast(pos + i);
        b.nk.b.slice(i32, PMAX)[i] = @intCast(pos + i + 1);
    }
    if (m.vis) |v| v.positions(pos, rows, b.pos.b.slice(i32, PMAX), b.pos3.b.slice(i32, 3 * PMAX)); // an image prompt
    const kvm = b.kvmeta.b.slice(u32, 3);
    kvm[0], kvm[1], kvm[2] = .{ @intCast(pos), @intCast(m.cap), @intCast(rows) };
    b.rows.b.slice(i32, 1)[0] = @intCast(rows);
    const pp = &m.ple;
    const seq = try gpa.alloc(i64, 2 + rows);
    defer gpa.free(seq);
    seq[0], seq[1] = .{ hist[0], hist[1] };
    for (tokens, 0..) |tok, i| seq[2 + i] = tok;
    const out = b.pids.b.slice(u32, 16 * PMAX);
    var last_eos: i64 = -1;
    if (seq[0] == pp.eos) last_eos = 0;
    if (seq[1] == pp.eos) last_eos = 1;
    for (0..rows) |row| {
        const at = 2 + row;
        const in_seg = @as(i64, @intCast(at)) - (last_eos + 1);
        var sh: [3]i64 = undefined;
        for (0..3) |s2| sh[s2] = if (in_seg >= @as(i64, @intCast(s2))) seq[at - s2] else pp.eos;
        for (2..4) |ng| {
            var mixed: i64 = sh[0] *% pp.mult[0];
            for (1..ng) |q2| mixed ^= sh[q2] *% pp.mult[q2];
            for (0..8) |k| {
                const hh = (ng - 2) * 8 + k;
                out[row * 16 + hh] = @intCast(@mod(mixed, pp.sizes[hh]) + pp.offsets[hh]);
            }
        }
        if (seq[at] == pp.eos) last_eos = @intCast(at);
    }
}

pub fn segPre(s: *Seg, m: *Model, i: usize) !void {
    const p = s.p;
    const r = p.r;
    const b = &p.b;
    const t = &m.t;
    const rows = s.rows;
    const L = &m.layers[i];
    if (i == 1) {
        if (s.pending) try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[s.cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - s.cur], b.ssp }, null);
        if (s.pending) s.cur = 1 - s.cur;
        s.pending = false;
        const pp = &m.ple;
        var tabs: [2 + 3 * GROUPS]Buf = undefined;
        tabs[0] = b.pids;
        tabs[1] = pp.starts;
        for (0..3 * GROUPS) |j| tabs[2 + j] = pp.tables[j];
        if (r.fsr) |stream| if (try stream.tables(r, b.pids, rows)) |staged| {
            tabs = staged;
        };
        var cp: [2]variants.Const = undefined;
        try r.callRows("qa_ple_lookup@ple", rows, &tabs, &.{b.emb}, try variants.oneRow(r, "qa_ple_lookup@ple", variants.width(&cp, "BITS", "GS", pp.w)));
        if (pp.scale) |sc| sc.apply(r.enc, b.emb, rows * D, r.serial);
        p.qmm(b.emb, p.ple_kv, D, 12800, rows, b.kvp, 0);
        try r.callRows("q4_ple_gate@ple", rows, &.{ b.kvp, b.h[s.cur], pp.ks, pp.qs, pp.cs, t.eps }, &.{ b.gated, .{ .b = b.cin.b, .off = PLE_TAIL * WIDE * 2 } }, null);
        try r.callRows("q4_ple_conv@ple", rows, &.{ b.cin, pp.conv, b.gated, b.h[s.cur] }, &.{b.hout}, null);
        try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.hout}, &.{ b.h[1 - s.cur], b.ssp }, null);
    } else if (!s.pending) {
        try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.h[s.cur]}, &.{ b.h[1 - s.cur], b.ssp }, null);
    } else {
        try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[s.cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - s.cur], b.ssp }, null);
    }
    s.cur = 1 - s.cur;
    p.hc(m, b.h[s.cur], L.ahc, rows, b.inj_a);
}

pub fn segMixer(s: *Seg, m: *Model, i: usize) !void {
    const p = s.p;
    const r = p.r;
    const b = &p.b;
    const t = &m.t;
    const rows = s.rows;
    const L = &m.layers[i];
    if (L.linear) {
        p.qmm(b.mixed, p.proj[i], D, 16480, rows, b.p, 0);
        const cs_in: Buf = .{ .b = L.cs[s.ra].b, .off = L.cs[s.ra].off + s.rr * CS_ROW };
        const so_in: Buf = .{ .b = L.so[s.ra].b, .off = L.so[s.ra].off + s.rr * SO_ROW };
        if (p.skip & 2 == 0) {
            const ri: i32 = @intCast(rows);
            const li = linearIndex(m, i);
            p.bind(p.pl[11], &.{ b.p, cs_in, L.conv, L.alog, L.dt });
            r.enc.setBytes(std.mem.asBytes(&ri), 5);
            for ([_]Buf{ b.qn, b.kn, b.v, b.gg, b.beta, L.cs[s.wa] }, 6..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
            p.bindMarks(m, s.mk, li, 12, 13, false);
            r.enc.dispatchThreads(mtl.Size.of(80 * 128, rows, 1), mtl.Size.of(128, 1, 1));
            p.barrier();
            const scan: usize = if (p.scan4 and p.scan8) 15 else if (p.scan4) 14 else 12;
            p.bind(p.pl[scan], &.{ b.qn, b.kn, b.v, b.gg, b.beta, so_in });
            r.enc.setBytes(std.mem.asBytes(&ri), 6);
            r.enc.setBuffer(b.ys.b, b.ys.off, 7);
            r.enc.setBuffer(L.so[s.wa].b, L.so[s.wa].off, 8);
            p.bindMarks(m, s.mk, li, 9, 10, true);
            switch (scan) {
                15 => r.enc.dispatchThreads(mtl.Size.of(48 * 2 * 256, 1, 1), mtl.Size.of(256, 1, 1)),
                14 => r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 256, 1, 1), mtl.Size.of(256, 1, 1)),
                else => r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 1024, 1, 1), mtl.Size.of(1024, 1, 1)),
            }
            p.barrier();
            p.bind(p.pl[13], &.{ b.ys, b.p, L.norm, t.eps, b.gout });
            r.enc.dispatchThreads(mtl.Size.of(48 * 128, rows, 1), mtl.Size.of(128, 1, 1));
            p.barrier();
        }
        p.qmm(b.gout, p.out[i], 6144, D, rows, b.branch, 0);
    } else {
        p.qmm(b.mixed, p.proj[i], D, 13952, rows, b.p, 0);
        if (m.vis != null and s.pos < m.vis.?.frame_end) { // rows up to the last image's end: three positions a row
            r.enc.setPipeline(m.vis.?.prep);
            for ([_]Buf{ b.p, b.pos3, L.qn, L.kn, L.iqn, t.eps, t.log2base, b.q, b.kout, b.iq }, 0..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
            r.enc.dispatchThreads(mtl.Size.of(256, 30, rows), mtl.Size.of(256, 1, 1));
            if (!r.serial) r.enc.barrier();
        } else try r.callRows("q4_attn_prep@att", rows, &.{ b.p, b.pos, L.qn, L.kn, L.iqn, t.eps, t.log2base }, &.{ b.q, b.kout, b.iq }, null);
        p.bind(r.kv_pipe, &.{ b.kout, b.p, L.keys, L.vals, L.raw, b.kvmeta });
        r.enc.dispatchThreads(mtl.Size.of(512 * rows, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        const sparse = p.sel != null and p.sel.?.meta(s.pos, rows);
        if (sparse and p.skip & 1024 != 0) {
            // timing knock-out: no attention for chunks with rows past the dense range
        } else if (p.fast_attn and p.sel != null and p.skip & 4 == 0) {
            var sl = &p.sel.?;
            if (sparse) {
                sl.nax_scores = p.scores_nax;
                defer sl.nax_scores = null;
                try sl.encodeAt(r, L, b.iq, t.eps, t.log2base, s.pos, rows, m.bpos);
            }
            p.bind(p.sattn, &.{ b.q, L.keys, L.vals, sl.keys, sl.counts, sl.sparse, b.p, t.scale });
            const cap: i32 = @intCast(m.cap);
            r.enc.setBytes(std.mem.asBytes(&cap), 8);
            r.enc.setBuffer(b.aout.b, b.aout.off, 9);
            r.enc.dispatchThreads(mtl.Size.of(rows * 128, 2, 1), mtl.Size.of(128, 1, 1));
            p.barrier();
        } else if (sparse) {
            var sl = &p.sel.?;
            try sl.encodeAt(r, L, b.iq, t.eps, t.log2base, s.pos, rows, m.bpos);
            try p.partials();
            const dense_ids = r.shapes.get("IDS_shape").?;
            try r.shapes.put(r.arena, "IDS_shape", sl.ids_shape);
            try r.callRows("q4_attn_parts#[24, 256]", rows, &.{ b.q, L.keys, L.vals, sl.keys, sl.counts, sl.sparse, t.scale }, &.{ b.po, b.pm }, null);
            try r.shapes.put(r.arena, "IDS_shape", dense_ids);
            try r.callRows("q4_attn_merge_gate#[24, 16, 256]", rows, &.{ b.po, b.pm, b.p }, &.{b.aout}, null);
        } else if (p.skip & 4 == 0) {
            p.bind(p.attn256, &.{ b.q, L.keys, L.vals, b.p });
            const ap = [4]i32{ @intCast(rows), @intCast(s.pos + rows), @intCast(s.pos), @intCast(m.cap) };
            r.enc.setBytes(std.mem.asBytes(&ap), 4);
            r.enc.setBuffer(t.scale.b, t.scale.off, 5);
            r.enc.setBuffer(b.aout.b, b.aout.off, 6);
            r.enc.dispatchThreads(mtl.Size.of(((rows + 63) / 64) * 128, 24, 1), mtl.Size.of(128, 1, 1));
            p.barrier();
        }
        p.qmm(b.aout, p.out[i], 6144, D, rows, b.branch, 0);
    }
}

pub fn segPost(s: *Seg, m: *Model, i: usize) !void {
    const p = s.p;
    const b = &p.b;
    const L = &m.layers[i];
    try p.r.callRows("q4_hc_norm_plain#[10240]", s.rows, &.{ b.h[s.cur], b.inj_a, b.branch }, &.{ b.h[1 - s.cur], b.ssp }, null);
    s.cur = 1 - s.cur;
    p.hc(m, b.h[s.cur], L.mhc, s.rows, b.inj_m);
    try p.moe(L, s.rows, i);
    s.pending = true;
}

pub const Pair = struct {
    tp: *Tp2,
    call: u32,
    k: usize, // this Mac's segment: 0 hands each layer to the peer, 1 waits for it
    pos0: usize, // segment 0's first position and rows: what segment 1 receives
    rows0: usize,
    a: usize, // the DeltaNet slot the chunk starts from; segment 0 writes 1 - a, segment 1 writes a
};

pub fn bytesOf(b: Buf) [*]const u8 {
    return b.b.contents() + b.off;
}

pub fn pairSend(pr: *const Pair, m: *Model, s: *Seg, i: usize) void {
    const L = &m.layers[i];
    const slot = tpm.layerSlot(i);
    var ws: [8]tpm.Write = undefined;
    var n: usize = 0;
    if (L.linear) {
        ws[0] = .{ .src = bytesOf(L.cs[1 - pr.a]), .len = CS_ROW, .dst = slot };
        ws[1] = .{ .src = bytesOf(L.so[1 - pr.a]), .len = SO_ROW, .dst = slot + CS_ROW };
        n = 2;
    } else {
        for (0..2) |hd| {
            ws[n] = .{ .src = bytesOf(L.keys) + (hd * m.cap + pr.pos0) * 512, .len = pr.rows0 * 512, .dst = slot + hd * tpm.KV_HEAD };
            ws[n + 1] = .{ .src = bytesOf(L.vals) + (hd * m.cap + pr.pos0) * 512, .len = pr.rows0 * 512, .dst = slot + (2 + hd) * tpm.KV_HEAD };
            n += 2;
        }
        ws[n] = .{ .src = bytesOf(L.raw) + pr.pos0 * 256, .len = pr.rows0 * 256, .dst = slot + 4 * tpm.KV_HEAD };
        n += 1;
    }
    if (i == 1) {
        ws[n] = .{ .src = bytesOf(s.p.b.cin) + s.rows * WIDE * 2, .len = PLE_TAIL * WIDE * 2, .dst = tpm.TAIL };
        n += 1;
    }
    pr.tp.send(s.p.r.enc, ws[0..n], tpm.LAYER_FLAG + 8 * i, pr.call);
}

pub fn pairTake(pr: *const Pair, m: *Model, s: *Seg, i: usize) void {
    const L = &m.layers[i];
    const w = pr.tp.window();
    const slot = tpm.layerSlot(i);
    const p = s.p;
    pr.tp.waitWord(p.r.enc, tpm.LAYER_FLAG + 8 * i, pr.call);
    if (L.linear) {
        p.copyWords(.{ .b = w, .off = slot }, L.cs[1 - pr.a], CS_ROW / 4);
        p.copyWords(.{ .b = w, .off = slot + CS_ROW }, L.so[1 - pr.a], SO_ROW / 4);
    } else {
        for (0..2) |hd| {
            p.copyWords(.{ .b = w, .off = slot + hd * tpm.KV_HEAD }, .{ .b = L.keys.b, .off = L.keys.off + (hd * m.cap + pr.pos0) * 512 }, pr.rows0 * 128);
            p.copyWords(.{ .b = w, .off = slot + (2 + hd) * tpm.KV_HEAD }, .{ .b = L.vals.b, .off = L.vals.off + (hd * m.cap + pr.pos0) * 512 }, pr.rows0 * 128);
        }
        p.copyWords(.{ .b = w, .off = slot + 4 * tpm.KV_HEAD }, .{ .b = L.raw.b, .off = L.raw.off + pr.pos0 * 256 }, pr.rows0 * 64);
    }
    if (i == 1) p.copyWords(.{ .b = w, .off = tpm.TAIL }, p.b.cin, PLE_TAIL * WIDE * 2 / 4);
}

pub const Hooks = struct {
    m: *Model,
    segs: []Seg,
    pair: ?*const Pair = null,

    fn at(h: *Hooks, l: *const segments.Lane) *Seg {
        const s = &h.segs[l.k];
        s.p.r.enc = l.enc;
        return s;
    }
    pub fn begin(h: *Hooks, l: *segments.Lane) !void {
        const s = h.at(l);
        const b = &s.p.b;
        const m = h.m;
        var ce: [2]variants.Const = undefined;
        const ev = try variants.oneRow(s.p.r, "qa_embed_rows@embed", variants.width(&ce, "BITS", "GS", m.embed_w));
        try s.p.r.callRows("qa_embed_rows@embed", s.rows, &.{ b.ids, m.embed[0], m.embed[1], m.embed[2] }, &.{b.h[0]}, ev);
        if (m.vis) |v| for (v.spans) |sp| if (vis_prompt.VisPrompt.overlap(sp, s.pos, s.rows)) |o| { // the image rows
            const row = WIDE * 2; // the embedding's row: a copy for each of the four streams
            s.p.copyWords(.{ .b = v.rows, .off = o[1] * row }, b.h[0].at(o[0] * row), o[2] * row / 4);
        };
    }
    pub fn wait(_: *Hooks, i: usize) segments.Wait {
        return if (i == 1) .pre else .mixer;
    }
    pub fn handoff(h: *Hooks, l: *segments.Lane, i: usize) !void {
        if (i != 1) return;
        const s = h.at(l);
        const prev = &h.segs[l.k - 1];
        s.p.copyWords(.{ .b = prev.p.b.cin.b, .off = prev.p.b.cin.off + prev.rows * WIDE * 2 }, s.p.b.cin, PLE_TAIL * WIDE * 2 / 4);
    }
    pub fn pre(h: *Hooks, l: *segments.Lane, i: usize) !void {
        const s = h.at(l);
        if (h.pair) |pr| if (pr.k == 1 and i == 1) pairTake(pr, h.m, s, i);
        try segPre(s, h.m, i);
    }
    pub fn mixer(h: *Hooks, l: *segments.Lane, i: usize) !void {
        const s = h.at(l);
        if (h.pair) |pr| if (pr.k == 1 and i != 1) pairTake(pr, h.m, s, i);
        try segMixer(s, h.m, i);
        if (h.pair) |pr| if (pr.k == 0) pairSend(pr, h.m, s, i);
    }
    pub fn post(h: *Hooks, l: *segments.Lane, i: usize) !void {
        try segPost(h.at(l), h.m, i);
    }
    /// Every segment's final streams (the MTP head's prompt keys read them); the head on the last one's last row.
    pub fn finish(h: *Hooks, l: *segments.Lane) !void {
        const s = h.at(l);
        const p = s.p;
        const r = p.r;
        const b = &p.b;
        const m = h.m;
        const t = &m.t;
        try r.callRows("q4_hc_norm_grouped#[10240]", s.rows, &.{ b.h[s.cur], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1 - s.cur], b.ssp }, null);
        s.cur = 1 - s.cur;
        p.last = b.h[s.cur];
        if (l.k + 1 < h.segs.len) return;
        if (h.pair) |pr| if (pr.k == 0) return; // the peer's segment ends the chunk: it runs the head
        m.last = p.last;
        p.hc(m, b.h[s.cur], m.mix, s.rows, b.inj_a);
        r.rows = 1;
        t.mdims.b.slice(i32, 2)[0] = 1;
        try m.lane(.{ .b = b.mixed.b, .off = b.mixed.off + (s.rows - 1) * D * 2 }, D, m.head, "lane_qmm_bytes_grouped@head", t.logits);
        r.enc.setPipeline(r.argmax_pipe);
        r.enc.setBuffer(t.logits.b, 0, 0);
        r.enc.setBuffer(t.picks.b, 0, 1);
        r.enc.setBuffer(t.vocab.b, 0, 2);
        r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
    }
};
