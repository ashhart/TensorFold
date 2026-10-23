//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const split = fz.split;
const dense = fz.dense;
const lane_w = fz.lane_w;
const rows_load = fz.rows_load;
const variants = fz.variants;
const gx_mod = @import("gx.zig");
const gdn_step = fz.gdn_step;
const vis_prompt = fz.vis_prompt;
const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const TRAINED = fz.TRAINED;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const HEAD_TILES = VOCAB / 32;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const Buf = fz.Buf;
const Run = fz.Run;
const POOL_BLOCKS = fz.POOL_BLOCKS;
const Hc = fz.Hc;
const Lane = fz.Lane;
const Layer = fz.Layer;
const Ple = fz.Ple;
const Tmp = fz.Tmp;
const Mtp = fz.Mtp;
const Marks = fz.Marks;
const TP_CLASS = fz.TP_CLASS;

pub const Model = struct {
    r: *Run,
    layers: [LAYERS]Layer,
    mix: Hc,
    head: Lane,
    embed: [3]Buf,
    embed_w: ?[2]usize = null, // the embedding's width when not 6/32
    ple: Ple,
    t: Tmp,
    pos: usize = 0,
    cap: usize = TRAINED, // the model's capacity, set at load
    state: usize = 0, // the DeltaNet buffer pair holding the state
    state_row: usize = 0, // its row
    gpu_seconds: f64 = 0,
    mtp: Mtp = undefined,
    last: Buf = undefined, // the last window's streams before the final mixer
    recs: ?[2]Buf = null, // GPU-side rounds with r.gdn_kept: each window's DeltaNet replay records, by round parity
    rec_slot: usize = 0, // the record this window writes (the other holds the previous window's)
    marks: ?Marks = null, // DeltaNet states at marks inside a prompt chunk, as a chunk ending there would leave them
    vis: ?*const vis_prompt.VisPrompt = null, // an image prompt's rows and positions while its prompt pass runs
    bpos: ?Buf = null, // an image stream's indexer block positions [cap / 4 + 1, 3] (int32), the standard frame
    delta: i32 = 0, // and its rows past the last image rotate at their slot plus this

    pub fn reset(m: *Model) void {
        m.pos = 0;
        m.bpos = null;
        m.delta = 0;
        m.state = 0;
        m.state_row = 0;
        for (&m.layers) |*L| if (L.linear) {
            @memset(L.cs[0].b.contents()[L.cs[0].off .. L.cs[0].off + CS_ROW], 0);
            @memset(L.so[0].b.contents()[L.so[0].off .. L.so[0].off + SO_ROW], 0);
        } else {
            L.pooled_n = 0;
        };
        m.ple.hist = .{ m.ple.eos, m.ple.eos };
        @memset(m.ple.cin.b.contents()[m.ple.cin.off .. m.ple.cin.off + (PLE_TAIL + MAXR) * WIDE * 2], 0);
        m.mtp.pooled_n = 0;
    }

    /// The replay records when this window keeps one DeltaNet state a layer (GPU-side rounds with r.gdn_kept).
    pub fn keptState(m: *const Model) ?[2]Buf {
        return if (m.r.gpu_round and m.r.gdn_kept != null) m.recs else null;
    }

    pub fn lane(m: *Model, x: Buf, k: usize, l: Lane, role: []const u8, y: Buf) !void {
        if (l.rows) |d| if (m.r.tp == null) return rows_load.launch(m.r, d, role, x, m.r.rows, y);
        if (l.w) |w| return lane_w.launch(m.r, w, role, x, m.t.mdims, m.r.rows, y);
        const six = try l.recorded();
        if (m.r.dense_target) return m.r.denseRows(x, k, six, m.r.rows, y);
        if (m.r.lane_new) return dense.lane(m.r, role, x, six, m.t.mdims, y);
        if (!m.r.fused_xsum) try m.r.call(if (k == D) "lane_qmm_xsum#[2560]" else "lane_qmm_xsum#[6144]", &.{ x, m.t.mdims }, &.{m.t.xs});
        try m.r.call(role, &.{ x, m.t.xs, six.wq, six.sbt, m.t.mdims }, &.{y});
    }

    pub fn hcProject(m: *Model, hn: Buf, hc: Hc, down: []const u8, up: []const u8, inj: Buf) !void {
        const t = &m.t;
        if (hc.rows) |sh| if (m.r.tp == null) return rows_load.hc(
            m.r,
            sh,
            hc,
            down,
            hn,
            t.ssp,
            t.eps,
            t.rows,
            m.r.rows,
            t.part,
            t.mixed,
            inj,
        );
        const as_rows = if (m.r.hc_mma and m.r.rows > 1) 8 else m.r.rows;
        try m.r.hcProject(down, up, as_rows, hn, t.ssp, hc, t.eps, t.rows, t.part, t.mixed, inj);
    }

    pub fn grouped(m: *Model, h: Buf, out: Buf) !void {
        const t = &m.t;
        if (m.r.tp) |tp| return if (m.r.skip & TP_CLASS == 0) tp.combine(m.r.enc, h, t.inj_m, out, t.ssp, t.rows, m.r.rows); // TP: each rank's rows' branches
        try m.r.call("q4_hc_norm_grouped#[10240]", &.{ h, t.inj_m, t.ydown, t.wts, t.lg }, &.{ out, t.ssp });
    }

    /// The window's n-gram row ids [rows, 16] after the history (NGramEmbedding.ids on [history, tokens]).
    pub fn pleIds(m: *Model, tokens: []const u32) void {
        const p = &m.ple;
        var seq: [2 + MAXR]i64 = undefined;
        seq[0], seq[1] = .{ p.hist[0], p.hist[1] };
        for (tokens, 0..) |tok, i| seq[2 + i] = tok;
        const out = m.t.ple_ids.b.slice(u32, 16 * MAXR);
        for (0..tokens.len) |row| {
            const at = 2 + row;
            var before: i64 = -1;
            for (0..at) |q| if (seq[q] == p.eos) {
                before = @intCast(q);
            };
            const in_seg = @as(i64, @intCast(at)) - (before + 1);
            var sh: [3]i64 = undefined;
            for (0..3) |s| sh[s] = if (in_seg >= @as(i64, @intCast(s))) seq[at - s] else p.eos;
            for (2..4) |ng| {
                var mixed: i64 = sh[0] *% p.mult[0];
                for (1..ng) |q| mixed ^= sh[q] *% p.mult[q];
                for (0..8) |k| {
                    const hh = (ng - 2) * 8 + k;
                    out[row * 16 + hh] = @intCast(@mod(mixed, p.sizes[hh]) + p.offsets[hh]);
                }
            }
        }
    }

    /// A window's per-row values from the cache position (rows, matmul dims, positions, key counts, kv rows).
    pub fn windowMeta(m: *Model, rows: usize) void {
        if (m.r.gpu_round) return; // fz_accept wrote them
        const t = &m.t;
        t.rows.b.slice(i32, 1)[0] = @intCast(rows);
        t.mdims.b.slice(i32, 2)[0] = @intCast(rows);
        const pos8 = t.pos8.b.slice(i32, MAXR);
        const nk8 = t.nk8.b.slice(i32, MAXR);
        for (0..MAXR) |i| {
            pos8[i] = if (i < rows) @intCast(m.pos + i) else 0;
            nk8[i] = if (i < rows) @intCast(m.pos + i + 1) else 0;
        }
        const kvm = t.kvmeta.b.slice(u32, 3);
        kvm[0], kvm[2] = .{ @intCast(m.pos), @intCast(rows) };
    }

    /// One forward over `tokens` (a window of up to MAXR rows from the cache's position): each row's argmax.
    pub fn window(m: *Model, tokens: []const u32, picks: []u32) !void {
        const r = m.r;
        const t = &m.t;
        const rows = tokens.len;
        m.windowMeta(rows);
        const ids = t.ids8.b.slice(u32, MAXR);
        for (0..MAXR) |i| ids[i] = if (i < rows) tokens[i] else 0;
        m.pleIds(tokens);
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        try m.windowEncode(rows, t.ids8);
        try m.finish(cb);
        @memcpy(picks[0..rows], t.picks.b.slice(u32, rows));
    }

    pub fn finish(m: *Model, cb: mtl.CommandBuffer) !void {
        m.r.enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        if (m.r.tp) |tp| if (tp.failed.load(.acquire)) return error.TpLinkFailed;
        m.gpu_seconds += cb.gpuSeconds();
    }

    /// The n-gram ids of the window in `ids` hashed on the GPU (the window's drafts never reach the host).
    pub fn pleIdsGpu(m: *Model, rows: usize, ids: Buf) void {
        const r = m.r;
        const p = &m.ple;
        if (!r.gpu_round) { // in GPU-side rounds fz_accept keeps the history and the rest is set once
            const pm = m.t.ple_meta.b.slice(i64, 39);
            pm[0], pm[1], pm[2], pm[3] = .{ p.hist[0], p.hist[1], p.eos, @intCast(rows) };
            for (0..3) |k| pm[4 + k] = p.mult[k];
            for (0..16) |k| {
                pm[7 + k] = p.sizes[k];
                pm[23 + k] = p.offsets[k];
            }
        }
        r.enc.setPipeline(r.pleids_pipe);
        for ([_]Buf{ ids, m.t.ple_meta, m.t.ple_ids }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        const n: u32 = @intCast(rows);
        r.enc.setBytes(std.mem.asBytes(&n), 3);
        r.enc.dispatchThreads(mtl.Size.of(16 * rows, 1, 1), mtl.Size.of(16 * rows, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// Encode the window's forward (tokens read from `ids`, n-gram ids already in t.ple_ids) and its argmax.
    pub fn windowEncode(m: *Model, rows: usize, ids: Buf) !void {
        const r = m.r;
        const t = &m.t;
        r.rows = rows;
        if (r.gpu_round and r.gsel != null) {
            const g = &r.gsel.?;
            r.enc.setPipeline(r.sel_meta_pipe);
            r.enc.setBuffer(r.ar.b, r.ar.off, 0);
            const wu: u32 = @intCast(rows);
            r.enc.setBytes(std.mem.asBytes(&wu), 1);
            r.enc.setBuffer(g.sel.b, 0, 2);
            r.enc.setBuffer(g.pooled_shape, 0, 3);
            r.enc.setBuffer(g.sc_shape, 0, 4);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            if (!r.serial) r.enc.barrier();
        }
        try @import("rotary_cache.zig").prepare(r, t.log2base, t.pos8, rows);
        var cur: usize = 0;
        var ce: [2]variants.Const = undefined;
        try variants.callAs(r, "qa_embed_rows@embed", rows, &.{ ids, m.embed[0], m.embed[1], m.embed[2] }, &.{t.h[0]}, variants.width(&ce, "BITS", "GS", m.embed_w));
        var pending: enum { none, grouped } = .none;
        for (0..LAYERS) |i| {
            const L = &m.layers[i];
            if (i == 1) { // the PLE layer: write the pending MoE back, then the n-gram gate and conv
                try m.grouped(t.h[cur], t.h[1 - cur]);
                cur = 1 - cur;
                pending = .none;
                const p = &m.ple;
                var tabs: [2 + 3 * GROUPS]Buf = undefined;
                tabs[0] = t.ple_ids;
                tabs[1] = p.starts;
                for (0..3 * GROUPS) |j| tabs[2 + j] = p.tables[j];
                var cp: [2]variants.Const = undefined;
                try variants.callAs(r, "qa_ple_lookup@ple", rows, &tabs, &.{t.emb}, variants.width(&cp, "BITS", "GS", p.w));
                if (p.scale) |sc| sc.apply(r.enc, t.emb, rows * D, r.serial);
                try m.lane(t.emb, D, p.kv, "lane_qmm_bytes_grouped@ple.kv", t.kvp);
                try r.call("q4_ple_gate@ple", &.{ t.kvp, t.h[cur], p.ks, p.qs, p.cs, t.eps }, &.{ t.gated, .{ .b = p.cin.b, .off = PLE_TAIL * WIDE * 2 } });
                if (r.tree) |tr| try tr.pleConv(r, &.{ p.cin, p.conv, t.gated, t.h[cur] }, &.{t.hout}) else try r.call("q4_ple_conv@ple", &.{ p.cin, p.conv, t.gated, t.h[cur] }, &.{t.hout});
                try r.call("q4_hc_norm_none#[10240]", &.{t.hout}, &.{ t.h[1 - cur], t.ssp });
            } else if (pending == .none) {
                try r.call("q4_hc_norm_none#[10240]", &.{t.h[cur]}, &.{ t.h[1 - cur], t.ssp });
            } else {
                try m.grouped(t.h[cur], t.h[1 - cur]);
            }
            cur = 1 - cur;
            try m.hcProject(t.h[cur], L.ahc, "qa_hc_down@ahc", "qa_hc_up@ahc", t.inj_a);
            var plain = false; // speed-up mode's DeltaNet exchange made the stream update
            if (L.linear and r.tp != null) { // TP: this Mac's 8 key heads and 24 value heads, then one partial-sum exchange
                const tp = r.tp.?;
                const k0: usize = tp.rank;
                if (!r.fused_xsum) try r.call("lane_qmm_xsum#[2560]", &.{ t.mixed, t.mdims }, &.{t.xs});
                // [q 64 tiles | k 64 | v 192 | z 192 | b, a 3]: this Mac's q, k, v and z heads, b and a whole, 8 K...
                const cols = [_][2]usize{ .{ 32 * k0, 32 }, .{ 64 + 32 * k0, 32 }, .{ 128 + 96 * k0, 96 }, .{ 320 + 96 * k0, 96 }, .{ 512, 3 } };
                const pj = try L.proj.recorded();
                try split.laneTiles(r, "lane_qmm_bytes_grouped@gdn.in", &.{ t.mixed, t.xs, pj.wq, pj.sbt, t.mdims }, t.p, &cols, 8, null);
                const a = m.state;
                const cs_in: Buf = .{ .b = L.cs[a].b, .off = L.cs[a].off + m.state_row * CS_ROW };
                const so_in: Buf = .{ .b = L.so[a].b, .off = L.so[a].off + m.state_row * SO_ROW };
                if (m.keptState()) |recs| {
                    const li = i - i / 4;
                    gdn_step.stepKept(r, r.gdn_kept.?, &.{ t.p, cs_in, so_in, L.conv, L.alog, L.dt, L.norm, t.eps, t.rows }, &.{ t.gout, L.cs[1 - a] }, recs[1 - m.rec_slot].at(li * gdn_step.RECORD), recs[m.rec_slot].at(li * gdn_step.RECORD), r.ar, 24 * tp.rank, 24);
                } else try split.gdnHeads(r, "q4_gdn@gdn", if (r.gdn_step and rows > 1) 8 else rows, &.{ t.p, cs_in, so_in, L.conv, L.alog, L.dt, L.norm, t.eps, t.rows }, &.{ t.gout, L.cs[1 - a], L.so[1 - a] }, tp.rank);
                const pn = tp.partNext();
                const part: Buf = .{ .b = pn.b, .off = pn.off };
                const po = try L.out.recorded();
                try split.laneTiles(r, "lane_qmm_bytes_grouped@gdn.out", &.{ t.gout, t.xs, po.wq, po.sbt, t.mdims }, part, &.{.{ 0, 80 }}, 8, .{ 96 * k0, 96 }); // one kernel at every width: drafted == plain
                if (r.skip & TP_CLASS == 0) { // the partials' sum and the stream update in the exchange's launch
                    tp.plain(r.enc, t.h[cur], t.inj_a, t.h[1 - cur], t.ssp, t.rows, rows);
                    plain = true;
                }
            } else if (L.linear) {
                try m.lane(t.mixed, D, L.proj, "lane_qmm_bytes_grouped@gdn.in", t.p);
                const a = m.state;
                const cs_in: Buf = .{ .b = L.cs[a].b, .off = L.cs[a].off + m.state_row * CS_ROW };
                const so_in: Buf = .{ .b = L.so[a].b, .off = L.so[a].off + m.state_row * SO_ROW };
                lane_w.touch(r, L.out);
                const gins = [_]Buf{ t.p, cs_in, so_in, L.conv, L.alog, L.dt, L.norm, t.eps, t.rows };
                const gouts = [_]Buf{ t.gout, L.cs[1 - a], L.so[1 - a] };
                if (r.tree) |tr| {
                    gdn_step.stepTree(r, tr.gdn, &gins, &gouts, tr.table, 0, 48);
                } else if (m.keptState()) |recs| {
                    const li = i - i / 4;
                    gdn_step.stepKept(r, r.gdn_kept.?, &gins, gouts[0..2], recs[1 - m.rec_slot].at(li * gdn_step.RECORD), recs[m.rec_slot].at(li * gdn_step.RECORD), r.ar, 0, 48);
                } else if (r.gdn_pipe) |pipe| gdn_step.step(r, pipe, &gins, &gouts, 0, 48) else try r.callAs("q4_gdn@gdn", if (r.gdn_step and rows > 1) 8 else rows, &gins, &gouts);
                try m.lane(t.gout, 6144, L.out, "lane_qmm_bytes_grouped@gdn.out", t.branch);
            } else {
                try m.lane(t.mixed, D, L.proj, "lane_qmm_bytes_grouped@att.proj", t.p);
                lane_w.touch(r, L.out);
                try r.call("q4_attn_prep@att", &.{ t.p, t.pos8, L.qn, L.kn, L.iqn, t.eps, t.log2base }, &.{ t.q, t.kout, t.iq });
                if (r.skip & (1 << 5) == 0) {
                    r.enc.setPipeline(r.kv_pipe);
                    for ([_]Buf{ t.kout, t.p, L.keys, L.vals, L.raw, t.kvmeta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
                    r.enc.dispatchThreads(mtl.Size.of(512 * rows, 1, 1), mtl.Size.of(256, 1, 1));
                }
                if (!r.serial) r.enc.barrier();
                if (r.gpu_round and r.gsel != null) { // the round's selection, all on the GPU
                    const g = &r.gsel.?;
                    const sl = &r.sel.?;
                    r.enc.setPipeline(r.pool_abs_pipe);
                    for ([_]Buf{ L.raw, g.sel, L.pool, t.eps, t.log2base, L.pooled }, 0..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
                    r.enc.dispatchThreads(mtl.Size.of(128, POOL_BLOCKS, 1), mtl.Size.of(128, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    try r.shapes.put(r.arena, "POOLED_shape", g.pooled_shape);
                    try r.shapes.put(r.arena, "Q_shape", g.q_shape[rows]);
                    try r.shapes.put(r.arena, "SC_shape", g.sc_shape);
                    try r.bindV(sl.scores, &.{ t.iq, L.pooled, g.view(0) }, &.{g.sc});
                    r.enc.dispatchThreads(mtl.Size.of(((g.nb_ub + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    try r.bindV(sl.select, &.{ g.sc, g.view(0), g.view(MAXR) }, &.{g.keys});
                    r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
                    if (!r.serial) r.enc.barrier();
                    const dense_ids = r.shapes.get("IDS_shape").?;
                    try r.shapes.put(r.arena, "IDS_shape", g.ids_shape[rows]);
                    try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, g.keys, g.view(2 * MAXR), g.view(3 * MAXR), t.scale }, &.{ t.po, t.pm });
                    try r.shapes.put(r.arena, "IDS_shape", dense_ids);
                } else if (r.sel != null and !r.gpu_round and r.sel.?.meta(m.pos, rows)) {
                    var sl = &r.sel.?;
                    try sl.encode(r, L, t.iq, t.eps, t.log2base, m.pos, rows);
                    const dense_ids = r.shapes.get("IDS_shape").?;
                    try r.shapes.put(r.arena, "IDS_shape", sl.ids_shape);
                    try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, sl.keys, sl.counts, sl.sparse, t.scale }, &.{ t.po, t.pm });
                    try r.shapes.put(r.arena, "IDS_shape", dense_ids);
                } else if (r.tree) |tr| {
                    try tr.attnParts(r, &.{ t.q, L.keys, L.vals, t.ids81, t.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
                } else try r.call("q4_attn_parts#[24, 256]", &.{ t.q, L.keys, L.vals, t.ids81, t.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
                try r.call("q4_attn_merge_gate#[24, 16, 256]", &.{ t.po, t.pm, t.p }, &.{t.aout});
                try m.lane(t.aout, 6144, L.out, "lane_qmm_bytes_grouped@att.o", t.branch);
            }
            if (!plain) try r.call("q4_hc_norm_plain#[10240]", &.{ t.h[cur], t.inj_a, t.branch }, &.{ t.h[1 - cur], t.ssp });
            cur = 1 - cur;
            r.touch(L.router);
            try m.hcProject(t.h[cur], L.mhc, "qa_hc_down@mhc", "qa_hc_up@mhc", t.inj_m);
            try r.router("q4_router_float@moe", t.mixed, L.router, t.rows, t.lg, rows);
            r.tp_layer = true;
            const by_expert = if (r.gx) |g| rows >= g.min_rows and r.tp == null and L.xw != null and gx_mod.applies(L.xw.?) else false;
            if (by_expert) r.gx.?.encode(r, t.mixed, t.lg, L.ex, t.act, t.pick, t.wts, t.rows, t.ydown, rows) else if (L.xw) |w| try variants.experts(r, "qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down", t.mixed, t.lg, L.ex, t.act, t.pick, t.wts, t.rows, t.ydown, w) else try r.experts("qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down", t.mixed, t.lg, L.ex, t.act, t.pick, t.wts, t.rows, t.ydown);
            r.tp_layer = false;
            if (r.tp) |tp| if (r.skip & TP_CLASS == 0) tp.exchange(r.enc, t.ydown, t.wts, t.lg, rows);
            if (r.probe) |pb| r.copyKept(t.pick, .{ .b = pb.b, .off = pb.off + i * MAXR * 10 * 4 }, rows * 10, 0, 0, 0, 1, -1);
            if (r.lg_probe) |pb| r.copyKept(t.lg, pb.at(i * MAXR * 513 * 4), rows * 513, 0, 0, 0, 1, 0);
            pending = .grouped;
        }
        try m.grouped(t.h[cur], t.h[1 - cur]);
        cur = 1 - cur;
        m.last = t.h[cur];
        try m.hcProject(t.h[cur], m.mix, "qa_hc_down@mix", "qa_hc_up@mix", t.inj_a);
        if (r.tp) |tp| { // TP: this Mac's half of the vocabulary (the same kernel, half its tiles), then one swap
            const half = HEAD_TILES / 2;
            if (!r.fused_xsum) try r.call("lane_qmm_xsum#[2560]", &.{ t.mixed, t.mdims }, &.{t.xs});
            const hd = try m.head.recorded();
            try split.laneTiles(r, "lane_qmm_bytes_grouped@head", &.{ t.mixed, t.xs, hd.wq, hd.sbt, t.mdims }, t.logits, &.{.{ half * tp.rank, half }}, 1, null);
            if (r.skip & TP_CLASS == 0) tp.argmax(r.enc, t.logits, HEAD_TILES * 32, half * 32 * tp.rank, half * 32, t.picks, t.rows, rows);
            return;
        }
        try m.lane(t.mixed, D, m.head, "lane_qmm_bytes_grouped@head", t.logits);
        r.argmax(.{ .b = t.logits.b }, .{ .b = t.vocab.b }, .{ .b = t.picks.b }, null, rows);
    }

    /// The MTP head on `rows` rows: each row's next token and the streams it follows (target or head output, row by...
    pub const mtpRun = @import("replay_mtp.zig").mtpRun;

    /// Encode the MTP head at its position on `rows` rows (tokens from `ids`), its draft written to `out`; meta in...
    pub const mtpEncode = @import("replay_mtp.zig").mtpEncode;

    pub const mtpMeta = @import("replay_mtp.zig").mtpMeta;

    pub const mtpLayer = @import("replay_mtp.zig").mtpLayer;

    /// An MTP head lane: at its widths' runs, as dense rows (r.dense), or the recorded launch.
    pub const mtpLane = @import("replay_mtp.zig").mtpLane;

    pub const mtpProject = @import("replay_mtp.zig").mtpProject;

    /// Absorb `nexts.len` rows (chunks of up to MAXR) from `streams`; returns the draft after the last row.
    pub const mtpAbsorb = @import("replay_mtp.zig").mtpAbsorb;

    /// One chained draft after `draft`, from the head's last output row; its cache entry is trimmed next absorb.
    pub const mtpChain = @import("replay_mtp.zig").mtpChain;

    /// Keep the window's first `keep` rows: the DeltaNet state of row keep-1, the n-gram history and conv tail.
    pub fn keepRows(m: *Model, tokens: []const u32, keep: usize) void {
        m.state = 1 - m.state;
        m.state_row = keep - 1;
        m.pos += keep;
        for (&m.layers) |*L| if (!L.linear) {
            L.pooled_n = @min(L.pooled_n, m.pos / 4); // a block a rejected row completed is pooled again
        };
        const cin = m.ple.cin.b.contents();
        std.mem.copyForwards(u8, cin[0 .. PLE_TAIL * WIDE * 2], cin[keep * WIDE * 2 .. (keep + PLE_TAIL) * WIDE * 2]);
        for (tokens[0..keep]) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
    }
};
