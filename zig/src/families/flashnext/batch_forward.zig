//! One target forward over packed session windows, with segmented recurrence, convolution and attention.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const Session = @import("session.zig").Session;
const Engine = @import("engine.zig").Engine;
const Buf = fz.Buf;
pub const Seg = struct { s: *Session, row0: usize, rows: usize };

pub fn encode(e: *Engine, segs: []const Seg, rows: usize, ids: Buf) !void {
    const r = e.r;
    const m = e.m;
    const t = &m.t;
    if (r.tp != null or r.gpu_round) return error.BatchModeUnsupported;
    r.rows = rows;
    var cur: usize = 0;
    try r.call("qa_embed_rows@embed", &.{ ids, m.embed[0], m.embed[1], m.embed[2] }, &.{t.h[0]});
    for (0..fz.LAYERS) |i| {
        const l = &m.layers[i];
        if (i > 0) try m.grouped(t.h[cur], t.h[1 - cur]) else try r.call("q4_hc_norm_none#[10240]", &.{t.h[cur]}, &.{ t.h[1 - cur], t.ssp });
        cur = 1 - cur;
        if (i == 1) {
            const p = &m.ple;
            var tabs: [2 + 3 * fz.GROUPS]Buf = undefined;
            tabs[0] = t.ple_ids;
            tabs[1] = p.starts;
            for (p.tables, 0..) |b, j| tabs[2 + j] = b;
            try r.call("qa_ple_lookup@ple", &tabs, &.{t.emb});
            try m.lane(t.emb, fz.D, p.kv, "lane_qmm_bytes_grouped@ple.kv", t.kvp);
            for (segs) |g| {
                const v = view(t.*, g.row0);
                r.rows = g.rows;
                const pp = &g.s.m.ple;
                try r.call("q4_ple_gate@ple", &.{ v.kvp, v.h[cur], pp.ks, pp.qs, pp.cs, t.eps }, &.{ v.gated, pp.cin.at(fz.PLE_TAIL * fz.WIDE * 2) });
                try r.call("q4_ple_conv@ple", &.{ pp.cin, pp.conv, v.gated, v.h[cur] }, &.{v.hout});
            }
            r.rows = rows;
            try r.call("q4_hc_norm_none#[10240]", &.{t.hout}, &.{ t.h[1 - cur], t.ssp });
            cur = 1 - cur;
        }
        try m.hcProject(t.h[cur], l.ahc, "qa_hc_down@ahc", "qa_hc_up@ahc", t.inj_a);
        try m.lane(t.mixed, fz.D, l.proj, if (l.linear) "lane_qmm_bytes_grouped@gdn.in" else "lane_qmm_bytes_grouped@att.proj", t.p);
        r.touch(l.out.wq);
        r.touch(l.out.sbt);
        if (l.linear) {
            for (segs) |g| {
                const v = view(t.*, g.row0);
                const sl = &g.s.m.layers[i];
                r.rows = g.rows;
                const ins = [_]Buf{ v.p, sl.cs[0], sl.so[0], l.conv, l.alog, l.dt, l.norm, t.eps, g.s.m.t.rows };
                const outs = [_]Buf{ v.gout, sl.cs[1], sl.so[1] };
                if (r.gdn_pipe) |pipe| fz.gdn_step.step(r, pipe, &ins, &outs, 0, 48) else try r.callAs("q4_gdn@gdn", if (r.gdn_step and g.rows > 1) 8 else g.rows, &ins, &outs);
            }
        } else {
            try r.call("q4_attn_prep@att", &.{ t.p, t.pos8, l.qn, l.kn, l.iqn, t.eps, t.log2base }, &.{ t.q, t.kout, t.iq });
            for (segs) |g| try attention(e, g, i);
            r.rows = rows;
            try r.call("q4_attn_merge_gate#[24, 16, 256]", &.{ t.po, t.pm, t.p }, &.{t.aout});
        }
        r.rows = rows;
        try m.lane(if (l.linear) t.gout else t.aout, 6144, l.out, if (l.linear) "lane_qmm_bytes_grouped@gdn.out" else "lane_qmm_bytes_grouped@att.o", t.branch);
        try r.call("q4_hc_norm_plain#[10240]", &.{ t.h[cur], t.inj_a, t.branch }, &.{ t.h[1 - cur], t.ssp });
        cur = 1 - cur;
        r.touch(l.router);
        try m.hcProject(t.h[cur], l.mhc, "qa_hc_down@mhc", "qa_hc_up@mhc", t.inj_m);
        try r.router("q4_router_float@moe", t.mixed, l.router, t.rows, t.lg, rows);
        try r.experts("qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down", t.mixed, t.lg, l.ex, t.act, t.pick, t.wts, t.rows, t.ydown);
    }
    try m.grouped(t.h[cur], t.h[1 - cur]);
    cur = 1 - cur;
    m.last = t.h[cur];
    try m.hcProject(t.h[cur], m.mix, "qa_hc_down@mix", "qa_hc_up@mix", t.inj_a);
    try m.lane(t.mixed, fz.D, m.head, "lane_qmm_bytes_grouped@head", t.logits);
    r.enc.setPipeline(r.argmax_pipe);
    for ([_]Buf{ t.logits, t.picks, t.vocab }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
    if (!r.serial) r.enc.barrier();
}

fn attention(e: *Engine, g: Seg, i: usize) !void {
    const r = e.r;
    var t = view(e.m.t, g.row0);
    t.p = e.m.t.p.at(g.row0 * 13952 * 2);
    const sl = &g.s.m.layers[i];
    const st = &g.s.m.t;
    r.rows = g.rows;
    r.enc.setPipeline(r.kv_pipe);
    for ([_]Buf{ t.kout, t.p, sl.keys, sl.vals, sl.raw, st.kvmeta }, 0..) |b, j|
        r.enc.setBuffer(b.b, b.off, j);
    r.enc.dispatchThreads(mtl.Size.of(512 * g.rows, 1, 1), mtl.Size.of(256, 1, 1));
    if (!r.serial) r.enc.barrier();
    const old_shape = r.shapes.get("Kc_shape").?;
    const old_ids = r.shapes.get("IDS_shape").?;
    try r.shapes.put(r.arena, "Kc_shape", g.s.shape);
    defer r.shapes.put(r.arena, "Kc_shape", old_shape) catch unreachable;
    defer r.shapes.put(r.arena, "IDS_shape", old_ids) catch unreachable;
    if (g.s.select[i / 4]) |*sel| {
        if (sel.meta(g.s.m.pos, g.rows)) {
            try sel.encode(r, sl, t.iq, t.eps, t.log2base, g.s.m.pos, g.rows);
            try r.shapes.put(r.arena, "IDS_shape", sel.ids_shape);
            try r.call("q4_attn_parts#[24, 256]", &.{ t.q, sl.keys, sl.vals, sel.keys, sel.counts, sel.sparse, t.scale }, &.{ t.po, t.pm });
            return;
        }
    }
    try r.call("q4_attn_parts#[24, 256]", &.{ t.q, sl.keys, sl.vals, t.ids81, st.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
}

fn view(t: fz.Tmp, row: usize) fz.Tmp {
    var v = t;
    v.h = .{ t.h[0].at(row * fz.WIDE * 2), t.h[1].at(row * fz.WIDE * 2) };
    v.kvp = t.kvp.at(row * 12800 * 2);
    v.gated = t.gated.at(row * fz.WIDE * 2);
    v.hout = t.hout.at(row * fz.WIDE * 2);
    v.p = t.p.at(row * 16480 * 2);
    v.gout = t.gout.at(row * 6144 * 2);
    v.q = t.q.at(row * 24 * 256 * 2);
    v.kout = t.kout.at(row * 2 * 256 * 2);
    v.iq = t.iq.at(row * 4 * 128 * 2);
    v.po = t.po.at(row * 24 * 16 * 256 * 4);
    v.pm = t.pm.at(row * 24 * 16 * 2 * 4);
    return v;
}
