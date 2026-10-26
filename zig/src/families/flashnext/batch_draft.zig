//! A shared round's MTP drafts packed across streams: row-wise steps once, attention per stream, one tail a step.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const sample = @import("sample.zig");
const fz = @import("replay.zig");
const variants = @import("variants.zig");
const Session = @import("session.zig").Session;
const Buf = fz.Buf;
const MAXR = fz.MAXR;
const WIDE = fz.WIDE;
const D = fz.D;
const CALLS = 2 * MAXR; // one a step: an absorb in at most MAXR groups, then at most MAXR - 1 chained steps

/// Host values every packed step reads at run time, one slot a step (the command buffer reads them all at once).
pub const Scratch = struct {
    md: mtl.Buffer, // [rows, 16, 0 ...] 8 words, two a step: its rows, then its streams (the tail)
    md4: mtl.Buffer, // [4 rows, padded to 16, 0 ...] 8 words a step
    one: mtl.Buffer, // [rows] 4 words a step
    n_add: mtl.Buffer, // [rows * WIDE] 4 words a step
    pos8: mtl.Buffer, // the packed rows' positions, MAXR a step
    ids: mtl.Buffer, // an absorb's packed follow ids, MAXR a step
    keep: mtl.Buffer, // stream q's last row less q, plus one (fz_mtp_take's keep), MAXR a step
    outs: mtl.Buffer, // step j's draft for stream p at j * MAXR + p
    chain: Buf, // the tail's output rows, stream p at row p: the next step's input streams
    logits: Buf, // the draft head's rows
};

const Row = struct { s: *Session, follow: []const u32, hidden: Buf, steps: usize, sampling: ?lanes.Sampling = null, position: u64 = 0 };

/// FZ_BATCH_MTP_ROWS=0: each stream's drafts encoded alone, as before.
pub const State = struct {
    off: ?bool = null,
    x: ?Scratch = null,
    rows: [MAXR]Row = undefined,
    n: usize = 0,
    calls: usize = 0,
};

/// FZ_KEYED_DRAFTS=0: a sampled stream drafts by argmax, as before.
fn keyed() bool {
    const v = std.c.getenv("FZ_KEYED_DRAFTS") orelse return true;
    return v[0] != '0';
}

pub fn on(st: *State) bool {
    if (st.off == null) st.off = if (std.c.getenv("FZ_BATCH_MTP_ROWS")) |v| v[0] == '0' else false;
    return !st.off.?;
}

fn scratch(st: *State, r: *fz.Run, ids_n: usize) !*Scratch {
    if (st.x == null) st.x = .{
        .md = try r.buffer(2 * CALLS * 32),
        .md4 = try r.buffer(CALLS * 32),
        .one = try r.buffer(CALLS * 16),
        .n_add = try r.buffer(CALLS * 16),
        .pos8 = try r.buffer(CALLS * MAXR * 4),
        .ids = try r.buffer(CALLS * MAXR * 4),
        .keep = try r.buffer(CALLS * MAXR * 4),
        .outs = try r.buffer(MAXR * MAXR * 4),
        .chain = .{ .b = try r.buffer(MAXR * WIDE * 2) },
        .logits = .{ .b = try r.buffer(MAXR * ids_n * 2) },
    };
    return &st.x.?;
}

fn deeper(_: void, a: Row, b: Row) bool {
    return a.steps > b.steps;
}

/// Encodes every stream's absorb and chained drafts into the bound encoder; false: off, nothing encoded.
pub fn encode(b: anytype, sessions: []const *Session, follows: []const []const u32, hiddens: []const Buf, depths: []const usize, samplings: []const ?lanes.Sampling, positions: []const u64) !bool {
    const st = &b.packed_drafts;
    const r = b.e.r;
    if (!on(st) or r.tp != null or sessions.len == 0) return false;
    const x = try scratch(st, r, b.primary.mtp.ids_n);
    for (sessions, follows, hiddens, depths, 0..) |s, f, hd, d, k| st.rows[k] = .{ .s = s, .follow = f, .hidden = hd, .steps = @max(d, 1), .sampling = samplings[k], .position = positions[k] };
    st.n = sessions.len;
    st.calls = 0;
    const list = st.rows[0..st.n];
    std.sort.insertion(Row, list, {}, deeper); // the streams still drafting at a step are always a prefix
    const old = try b.bind(list[0].s); // host mode; every per-stream view is set below
    defer old.restore(b.e);
    for (list) |w| {
        const h = &w.s.m.mtp;
        h.pos -= h.drafted;
        h.drafted = 0;
        h.pooled_n = @min(h.pooled_n, h.pos / 4);
        @memcpy(h.slots[0].ids8.b.slice(u32, w.follow.len), w.follow);
    }
    var at: usize = 0;
    while (at < list.len) { // the absorb, in groups of at most MAXR rows
        var end = at;
        var n: usize = 0;
        while (end < list.len and n + list[end].follow.len <= MAXR) : (end += 1) n += list[end].follow.len;
        try step(st, b.primary, x, list[at..end], at, 0);
        at = end;
    }
    for (1..list[0].steps) |j| {
        var n: usize = 0;
        while (n < list.len and list[n].steps > j) n += 1;
        try step(st, b.primary, x, list[0..n], 0, j);
    }
    return true;
}

/// After the command buffer: each stream's drafts into its own draft_ids, as its own loop leaves them.
pub fn read(st: *State) void {
    const x = &st.x.?;
    const outs = x.outs.slice(u32, MAXR * MAXR);
    for (st.rows[0..st.n], 0..) |w, p| {
        const ids = w.s.draft_ids.b.slice(u32, MAXR);
        for (0..w.steps) |j| ids[j] = outs[j * MAXR + p];
    }
}

fn setMd(x: *Scratch, at: usize, rows: usize) Buf {
    const md = x.md.slice(i32, 2 * CALLS * 8)[at * 8 ..][0..8];
    md.* = .{ @intCast(rows), 16, 0, 0, 0, 0, 0, 0 };
    return .{ .b = x.md, .off = at * 32 };
}

/// One draft step (`j`) over the streams `g`, the first at packed stream row `p0`.
fn step(st: *State, m: *fz.Model, x: *Scratch, g: []const Row, p0: usize, j: usize) !void {
    const r = m.r;
    const t = &m.t;
    const h = &m.mtp;
    const c = st.calls;
    if (c >= CALLS) return error.DraftOutOfStep;
    st.calls += 1;
    const pos8 = x.pos8.slice(i32, CALLS * MAXR)[c * MAXR ..][0..MAXR];
    const ids = x.ids.slice(u32, CALLS * MAXR)[c * MAXR ..][0..MAXR];
    const keep = x.keep.slice(i32, CALLS * MAXR)[c * MAXR ..][0..MAXR];
    var starts: [MAXR]usize = undefined;
    var n: usize = 0;
    for (g, 0..) |w, q| {
        const hs = &w.s.m.mtp;
        const rows = if (j == 0) w.follow.len else 1;
        try w.s.m.mtpMeta(&hs.slots[j], rows); // its own slot's values: the attention and the next round read them
        starts[q] = n;
        for (0..rows) |i| pos8[n + i] = @intCast(hs.pos + i);
        if (j == 0) @memcpy(ids[n..][0..rows], w.follow);
        n += rows;
        keep[q] = @intCast(n - q);
    }
    @memset(pos8[n..], 0);
    const md = setMd(x, 2 * c, n);
    const md1 = setMd(x, 2 * c + 1, g.len);
    const md4 = x.md4.slice(i32, CALLS * 8)[c * 8 ..][0..8];
    md4.* = .{ @intCast(4 * n), @intCast(16 * ((4 * n + 15) / 16)), 0, 0, 0, 0, 0, 0 };
    x.one.slice(i32, CALLS * 4)[c * 4] = @intCast(n);
    x.n_add.slice(u32, CALLS * 4)[c * 4] = @intCast(n * WIDE);
    const pos: Buf = .{ .b = x.pos8, .off = c * MAXR * 4 };
    const in: Buf = if (j == 0) .{ .b = x.ids, .off = c * MAXR * 4 } else .{ .b = x.outs, .off = (j - 1) * MAXR * 4 };
    r.rows = n;
    try @import("rotary_cache.zig").append(r, t.log2base, pos, n);
    var ce: [2]variants.Const = undefined;
    try variants.callAs(r, "mtp:qa_embed_rows@embed", n, &.{ in, m.embed[0], m.embed[1], m.embed[2] }, &.{h.emb}, variants.width(&ce, "BITS", "GS", m.embed_w));
    try r.call("mtp:q4_rms_rows@mtp.enorm", &.{ h.emb, h.enorm, t.eps }, &.{h.en});
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ h.en, md }, &.{t.xs});
    try m.mtpLane(h.fce, "mtp:lane_qmm_bytes_grouped@mtp.fce", h.en, md, n, D, h.e);
    if (j == 0) { // each stream's hidden rows from where verify left them, into its packed rows
        for (g, starts[0..g.len]) |w, s0| {
            r.rows = w.follow.len;
            try r.call("mtp:q4_rms_rows@mtp.hnorm", &.{ w.hidden, h.hnorm, t.eps }, &.{h.hn.at(s0 * WIDE * 2)});
        }
        r.rows = n;
    } else try r.call("mtp:q4_rms_rows@mtp.hnorm", &.{ x.chain, h.hnorm, t.eps }, &.{h.hn});
    const md4b: Buf = .{ .b = x.md4, .off = c * 32 };
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[4R, 2560]", &.{ h.hn, md4b }, &.{t.xs});
    try m.mtpLane(h.fch, "mtp:lane_qmm_bytes_grouped@mtp.fch", h.hn, md4b, 4 * n, D, h.hs);
    r.enc.setPipeline(r.add_pipe);
    for ([_]Buf{ h.e, h.hs, h.h[0], .{ .b = x.n_add, .off = c * 16 } }, 0..) |bb, i| r.enc.setBuffer(bb.b, bb.off, i);
    r.enc.dispatchThreads(mtl.Size.of(n * WIDE, 1, 1), mtl.Size.of(256, 1, 1));
    if (!r.serial) r.enc.barrier();
    try r.call("mtp:q4_hc_norm_none#[10240]", &.{h.h[0]}, &.{ h.h[1], t.ssp });
    try m.mtpProject(h.h[1], h.ahc, "mtp:qa_hc_down@mtp.ahc", "mtp:qa_hc_up@mtp.ahc", t.inj_a, .{ .b = x.one, .off = c * 16 });
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ t.mixed, md }, &.{t.xs});
    try m.mtpLane(h.proj, "mtp:lane_qmm_bytes_grouped@mtp.att.proj", t.mixed, md, n, D, t.p);
    try r.call("mtp:q4_attn_prep@mtp.att", &.{ t.p, pos, h.qn, h.kn, h.iqn, t.eps, t.log2base }, &.{ t.q, t.kout, t.iq });
    const kc = r.shapes.get("Kc_shape").?;
    for (g, starts[0..g.len]) |w, s0| try attend(r, t, w.s, j, s0, if (j == 0) w.follow.len else 1);
    try r.shapes.put(r.arena, "Kc_shape", kc);
    r.rows = n;
    try r.call("mtp:q4_attn_merge_gate#[24, 16, 256]", &.{ t.po, t.pm, t.p }, &.{t.aout});
    var took = false;
    for (0..g.len) |q| if (keep[q] > 1) { // stream q's last row into row q (sources lie past every row written)
        r.enc.setPipeline(r.take_pipe);
        const kb: Buf = .{ .b = x.keep, .off = (c * MAXR + q) * 4 };
        for ([_]Buf{ t.aout.at(q * 6144 * 2), h.h[1].at(q * WIDE * 2), t.inj_a.at(q * 4 * 2), kb }, 0..) |bb, i| r.enc.setBuffer(bb.b, bb.off, i);
        r.enc.dispatchThreads(mtl.Size.of(WIDE, 1, 1), mtl.Size.of(256, 1, 1));
        took = true;
    };
    if (took and !r.serial) r.enc.barrier();
    try tail(m, x, g, md1, p0, j);
    for (g) |w| {
        const hs = &w.s.m.mtp;
        hs.pos += if (j == 0) w.follow.len else 1;
        hs.last = hs.h[1];
    }
}

/// A stream's MTP cache write, block selection and attention parts on its rows of the packed temps.
fn attend(r: *fz.Run, t: *fz.Tmp, s: *Session, j: usize, row: usize, rows: usize) !void {
    const h = &s.m.mtp;
    const sl = &h.slots[j];
    const q = t.q.at(row * 24 * 256 * 2);
    const iq = t.iq.at(row * 4 * 128 * 2);
    const po = t.po.at(row * 24 * 16 * 256 * 4);
    const pm = t.pm.at(row * 24 * 16 * 2 * 4);
    r.rows = rows;
    r.enc.setPipeline(r.kv_pipe);
    for ([_]Buf{ t.kout.at(row * 2 * 256 * 2), t.p.at(row * 13952 * 2), h.keys, h.vals, h.raw, sl.kvmeta }, 0..) |bb, i| r.enc.setBuffer(bb.b, bb.off, i);
    r.enc.dispatchThreads(mtl.Size.of(512 * rows, 1, 1), mtl.Size.of(256, 1, 1));
    if (!r.serial) r.enc.barrier();
    try r.shapes.put(r.arena, "Kc_shape", s.shape);
    r.sel = s.draft_select[j];
    if (r.sel != null and r.sel.?.meta(h.pos, rows)) {
        var ss = &r.sel.?;
        try ss.encode(r, h, iq, t.eps, t.log2base, h.pos, rows);
        const dense_ids = r.shapes.get("IDS_shape").?;
        try r.shapes.put(r.arena, "IDS_shape", ss.ids_shape);
        try r.call("mtp:q4_attn_parts#[24, 256]", &.{ q, h.keys, h.vals, ss.keys, ss.counts, ss.sparse, t.scale }, &.{ po, pm });
        try r.shapes.put(r.arena, "IDS_shape", dense_ids);
    } else try r.call("mtp:q4_attn_parts#[24, 256]", &.{ q, h.keys, h.vals, t.ids81, sl.nk8, t.zero8, t.scale }, &.{ po, pm });
}

/// The head's tail over one kept row a stream: its output rows to `chain` from row p0, its drafts to step j's outs.
fn tail(m: *fz.Model, x: *Scratch, g: []const Row, md1: Buf, p0: usize, j: usize) !void {
    const n = g.len;
    const r = m.r;
    const t = &m.t;
    const h = &m.mtp;
    r.rows = n;
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[6144]", &.{ t.aout, md1 }, &.{t.xs});
    try m.mtpLane(h.out, "mtp:lane_qmm_bytes_grouped@mtp.att.o", t.aout, md1, n, 6144, t.branch);
    try r.call("mtp:q4_hc_norm_plain#[10240]", &.{ h.h[1], t.inj_a, t.branch }, &.{ h.h[0], t.ssp });
    try m.mtpProject(h.h[0], h.mhc, "mtp:qa_hc_down@mtp.mhc", "mtp:qa_hc_up@mtp.mhc", t.inj_m, md1);
    try r.router("mtp:q4_router_float@mtp.moe", t.mixed, h.router, md1, t.lg, n);
    if (h.xw) |w| try variants.experts(r, "mtp:qa_expert_gateup@mtp.moe.gate", "mtp:qa_expert_down_y@mtp.moe.down", t.mixed, t.lg, h.ex, t.act, t.pick, t.wts, md1, t.ydown, w) else try r.experts("mtp:qa_expert_gateup@mtp.moe.gate", "mtp:qa_expert_down_y@mtp.moe.down", t.mixed, t.lg, h.ex, t.act, t.pick, t.wts, md1, t.ydown);
    const out = x.chain.at(p0 * WIDE * 2);
    try r.call("mtp:q4_hc_norm_grouped#[10240]", &.{ h.h[0], t.inj_m, t.ydown, t.wts, t.lg }, &.{ out, t.ssp });
    try m.mtpProject(out, h.mix, "mtp:qa_hc_down@mtp.mix", "mtp:qa_hc_up@mtp.mix", t.inj_a, md1);
    if (!r.fused_xsum) try r.call("mtp:lane_qmm_xsum#[2560]", &.{ t.mixed, md1 }, &.{t.xs});
    try m.mtpLane(h.draft, "mtp:lane_qmm_bytes_grouped@mtp.draft", t.mixed, md1, n, D, x.logits);
    var draws: [MAXR]?lanes.Sampling = @splat(null);
    var at: [MAXR]u64 = @splat(0);
    var any = false;
    for (g, 0..) |w, q| {
        at[q] = w.position + j;
        if (w.sampling) |s| if (s.temperature > 0 and keyed()) {
            draws[q] = s;
            any = true;
        };
    }
    if (any) { // a sampled stream's draft under its keyed rule, every stream's row in one dispatch
        sample.drawIds(r.enc, r.draw_ids_pipe, x.logits, h.ids, h.ids_n, draws[0..n], at[0..n], .{ .b = x.outs, .off = (j * MAXR + p0) * 4 });
        if (!r.serial) r.enc.barrier();
        return;
    }
    for (0..n) |q| { // fz_argmax_ids takes one row: one dispatch a stream
        const o: Buf = .{ .b = x.outs, .off = (j * MAXR + p0 + q) * 4 };
        r.enc.setPipeline(r.argids_pipe);
        for ([_]Buf{ x.logits.at(q * h.ids_n * 2), o, h.n_ids, h.ids }, 0..) |bb, i| r.enc.setBuffer(bb.b, bb.off, i);
        r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
    }
    if (!r.serial) r.enc.barrier();
}
