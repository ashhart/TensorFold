//! GLM-5.3-Flash's shared-round MTP: segmented head forward across active streams and one batched draw per level.

const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const q4 = @import("cuda_q4.zig");
const engine = @import("cuda_engine.zig");
const types = @import("cuda_engine_types.zig");
const pick = @import("cuda_engine_pick.zig");
const sampler = @import("cuda_sampler.zig");
const prompt_hold = @import("cuda_prompt_hold.zig");
const Engine = engine.Engine;

pub const DraftPart = @import("cuda_mtp_lanes.zig").DraftPart;

/// Segmented MTP: shared projections and MoE, with per-stream KV writes, attention and proposing heads.
pub fn mtpMode(f: *const fw.Forward, ids: []const i32, hidden: []const u64, propose: bool) !void {
    const n = ids.len;
    if (f.ranks.len == 0 or hidden.len != f.ranks.len) return error.BadWindow;
    const seg0 = f.ranks[0].segs;
    if (seg0.len == 0) return error.BadWindow;
    var at0: usize = 0;
    var k_prop: usize = 0;
    var prop_rows: usize = 0;
    for (seg0) |g| {
        if (g.row0 != at0 or g.rows == 0 or g.rows > fw.window_rows) return error.BadWindow;
        at0 += g.rows;
        if (propose and g.propose) {
            k_prop += 1;
            prop_rows = g.row0 + g.rows;
        }
    }
    if (at0 != n) return error.BadWindow;
    if (!f.ranks[0].b.prefill and k_prop > 16) return error.WindowTooWide;
    const c = &f.ranks[0].w.config;
    const D: usize = c.hidden;
    for (f.ranks, hidden) |r, h| {
        if (r.w.mtp == null) return error.NoMtpHead;
        if (n == 0 or n > r.b.rows or r.b.prefill or r.segs.len != seg0.len) return error.BadWindow;
        for (r.segs, seg0) |g, g0| {
            if (g.row0 != g0.row0 or g.rows != g0.rows or g.propose != g0.propose) return error.BadWindow;
            try g.st.readyMtp(f.stream, g.rows);
        }
        try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
        if (h != r.b.hin) try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(r.b.hin, h, n * D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
        const m = r.w.mtp.?;
        const b = r.b;
        try glue.embedB16(f.set(), f.stream, b.ids, r.w.embed, b.me, n, D, 1);
        for (r.segs) |g| {
            if (f.images) |im| try im.v.splice(f.stream, g.image_slot, g.st.mtp_len + 1, g.st.mtp_len + 1 + g.rows, b.me + g.row0 * D * 2, 1);
            if (g.st.mtp_len == 0) try f.d.check(f.d.api.cuMemsetD8Async(b.me + g.row0 * D * 2, 0, D * 2, f.stream.handle), "cuMemsetD8Async");
        }
        try glue.rmsnorm(f.set(), f.stream, b.me, D, m.enorm, b.mcat, 2 * D, null, c.eps, n, D);
        try glue.rmsnorm(f.set(), f.stream, b.hin, D, m.hnorm, b.mcat + D * 2, 2 * D, null, c.eps, n, D);
        try b16.matmul(f.set(), f.stream, b.mcat, 2 * D, m.eh, b.mx, b.kda.mm_part, n, D, 2 * D, false);
        try glue.rmsnormBy(f.set(), f.stream, b.mx, D, m.layer.in_norm, b.normed, D, b.xs, c.eps, n, D);
        const a = m.layer.mixer.mla;
        var sparse_any = false;
        for (r.segs) |g| if (propose and g.propose) {
            sparse_any = sparse_any or dsa_block.sparseRows(a, g.st.mtp_len, g.rows);
        };
        try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, D, n, sparse_any);
        const dk: dsa_block.Kernels = .{ .set = f.set(), .ops = f.m.ops() };
        for (r.segs) |g| {
            if (propose and g.propose) {
                const nch = @min(latent.chunksFor(g.st.mtp_len + g.rows), b.nch);
                try dsa_block.attend(dk, f.stream, a, g.st.mtp_caches, b.dsa, g.row0, g.rows, g.st.mtp_pos_dev, g.st.mtp_len, nch, b.part_rows);
            } else {
                try dsa_block.writeKeysAt(dk, f.stream, a, g.st.mtp_caches, b.dsa, g.row0, g.rows, g.st.mtp_pos_dev);
                if (g.row0 < prop_rows) try f.d.check(f.d.api.cuMemsetD8Async(b.dsa.ol + g.row0 * a.heads * a.lw * 2, 0, g.rows * a.heads * a.lw * 2, f.stream.handle), "cuMemsetD8Async");
            }
        }
        if (k_prop > 0) {
            try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, prop_rows, a.heads, a.dv, a.lw);
            try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, prop_rows, fw.Forward.slot(r, prop_rows));
        }
    }
    if (k_prop == 0) return;
    try f.resolve(prop_rows);
    for (f.ranks) |r| {
        const m = r.w.mtp.?;
        const b = r.b;
        try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, prop_rows, D, r.w.share.world);
        try glue.rmsnormBy(f.set(), f.stream, b.mx, D, m.layer.post_norm, b.normed, D, b.xs, c.eps, prop_rows, D);
        try f.ffn(r, m.layer, prop_rows, fw.Forward.slot(r, prop_rows));
    }
    try f.resolve(prop_rows);
    for (f.ranks) |r| {
        const m = r.w.mtp.?;
        const b = r.b;
        try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, prop_rows, D, r.w.share.world);
        var src = b.mx;
        if (prop_rows != k_prop) {
            var p: usize = 0;
            for (r.segs) |g| if (g.propose) {
                try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(b.normed + p * D * 2, b.mx + (g.row0 + g.rows - 1) * D * 2, D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
                p += 1;
            };
            src = b.normed;
        }
        try glue.rmsnorm(f.set(), f.stream, src, D, m.norm, b.fnormed, D, b.fxs, c.eps, k_prop, D);
        try q4.matmul(&f.m.q4, f.stream, b.fnormed, D, b.fxs, m.draft, b.logits, k_prop);
    }
}

/// Segmented MTP absorb and chained level draws across `parts` on this TP rank.
pub fn draftShared(e: *Engine, parts: []const DraftPart) !void {
    if (!e.mtp) return error.NoDraftHead;
    if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
    for (parts, 0..) |p, i| {
        if (p.depth > engine.max_drafts) return error.TooDeep;
        if (p.follow.len == 0 or p.follow.len > fw.window_rows) return error.WindowTooWide;
        if (p.from_prompt and p.follow.len != 1) return error.WindowTooWide;
        if (p.out.len < p.depth) return error.WindowTooWide;
        if (p.chances) |ch| if (ch.len < p.depth) return error.WindowTooWide;
        if (p.slot >= e.seqs.items.len or e.seqs.items[p.slot] == null) return error.UnknownSlot;
        for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
    }
    if (parts.len == 1) {
        const p = parts[0];
        try e.bind(p.slot);
        if (p.kept > 0) try e.keep(p.kept);
        return e.draft(p.from_prompt, p.follow, p.depth, p.out, p.chances);
    }
    var seqs: [engine.max_shared]*types.Seq = undefined;
    var hiddens: [engine.max_shared]u64 = undefined;
    for (parts, 0..) |p, i| {
        try e.bind(p.slot);
        if (p.kept > 0) try e.keep(p.kept);
        const seq = e.cur;
        seqs[i] = seq;
        hiddens[i] = if (p.from_prompt) prompt_hold.input(e) else seq.rows_at;
        try seq.st.setMtpLen(e.stream, seq.st.mtp_len - seq.in_head);
        seq.in_head = 0;
    }
    var order: [engine.max_shared]usize = undefined; // deepest first so each level's live streams form a prefix
    for (parts, 0..) |p, i| {
        var at = i;
        while (at > 0 and parts[order[at - 1]].depth < p.depth) : (at -= 1) order[at] = order[at - 1];
        order[at] = i;
    }
    const max_depth = parts[order[0]].depth;
    var d0: usize = 0;
    while (d0 < parts.len and parts[order[d0]].depth > 0) : (d0 += 1) {}
    const D = e.c.hidden;
    const vocab_bytes = e.share.vocab.len() * 2;
    var one: [1]fw.Rank = undefined;
    var f = e.view(&e.bm, &one);
    var segs: [engine.max_shared]fw.Seg = undefined;
    var ids_buf: [fw.window_rows]u32 = undefined;
    var start: usize = 0;
    var prop_saved: usize = 0;
    var multi_group = false;
    while (start < parts.len) {
        var end = start;
        var rows: usize = 0;
        var gk: usize = 0;
        while (end < parts.len) {
            const idx = order[end];
            const n = parts[idx].follow.len;
            if (rows > 0 and rows + n > fw.window_rows) break;
            try e.copy(e.bm.hin + rows * D * 2, hiddens[idx], n * D * 2);
            @memcpy(ids_buf[rows..][0..n], parts[idx].follow);
            const prop = parts[idx].depth > 0;
            segs[end - start] = .{ .st = &seqs[idx].st, .row0 = rows, .rows = n, .propose = prop, .image_slot = parts[idx].slot };
            rows += n;
            gk += @intFromBool(prop);
            end += 1;
        }
        if (start == 0 and end < parts.len) multi_group = true;
        one[0] = .{ .w = &e.w, .st = segs[0].st, .b = &e.bm, .segs = segs[0 .. end - start] };
        try f.mtpMode(@ptrCast(ids_buf[0..rows]), &.{e.bm.hin}, gk > 0);
        for (segs[0 .. end - start]) |g| try g.st.setMtpLen(e.stream, g.st.mtp_len + g.rows);
        if (multi_group and gk > 0) {
            const bs = &(e.bs orelse return error.NoSharedWindows);
            try e.copy(bs.mx + prop_saved * D * 2, e.bm.fnormed, gk * D * 2);
            try e.copy(bs.logits + prop_saved * vocab_bytes, e.bm.logits, gk * vocab_bytes);
            prop_saved += gk;
        }
        start = end;
    }
    if (max_depth == 0) {
        if (e.images) |v| try v.activate(e.slot);
        return;
    }
    var level0_logits = e.bm.logits;
    if (multi_group) {
        const bs = &e.bs.?;
        level0_logits = bs.logits;
        if (max_depth > 1) {
            var d1: usize = 0;
            while (d1 < d0 and parts[order[d1]].depth > 1) : (d1 += 1) {}
            if (d1 > 0) try e.copy(e.bm.fnormed, bs.mx, d1 * D * 2);
        }
    }
    var live = d0;
    for (0..max_depth) |j| {
        while (live > 0 and parts[order[live - 1]].depth <= j) : (live -= 1) {}
        try drawLevel(e, if (j == 0) level0_logits else e.bm.logits, order[0..live], parts, &seqs, j);
        var next_live = live;
        while (next_live > 0 and parts[order[next_live - 1]].depth <= j + 1) : (next_live -= 1) {}
        if (next_live > 0) {
            for (order[0..next_live], 0..) |idx, r| {
                ids_buf[r] = parts[idx].out[j];
                segs[r] = .{ .st = &seqs[idx].st, .row0 = r, .rows = 1, .propose = true, .image_slot = parts[idx].slot };
            }
            one[0] = .{ .w = &e.w, .st = segs[0].st, .b = &e.bm, .segs = segs[0..next_live] };
            try f.mtp(@ptrCast(ids_buf[0..next_live]), &.{e.bm.fnormed});
            for (order[0..next_live]) |idx| {
                try seqs[idx].st.setMtpLen(e.stream, seqs[idx].st.mtp_len + 1);
                seqs[idx].in_head += 1;
            }
        }
    }
    if (e.images) |v| try v.activate(e.slot);
}

/// Draws one level across `order` with one synchronization for greedy, confidence and sampled streams.
fn drawLevel(e: *Engine, logits: u64, order: []const usize, parts: []const DraftPart, seqs: *const [engine.max_shared]*types.Seq, level: usize) !void {
    if (e.structured) {
        for (order, 0..) |idx, row| {
            const p = parts[idx];
            try e.bind(p.slot);
            try e.drawWhole(logits + row * e.share.vocab.len() * 2, 1, seqs[idx].st.pos + 1 + level, 0, p.out[level..][0..1]);
            if (p.chances) |c| c[level] = 1;
        }
        return;
    }
    var g_rows: [engine.max_shared]usize = undefined;
    var g_idx: [engine.max_shared]usize = undefined;
    var g_sums: [engine.max_shared]bool = undefined;
    var s_rows: [engine.max_shared]usize = undefined;
    var s_idx: [engine.max_shared]usize = undefined;
    var s_rules: [engine.max_shared]sampler.Rule = undefined;
    var s_firsts: [engine.max_shared]u64 = undefined;
    var ng: usize = 0;
    var ns: usize = 0;
    for (order, 0..) |idx, row| {
        const p = parts[idx];
        if (seqs[idx].sampling) |s| {
            s_rows[ns] = row;
            s_idx[ns] = idx;
            s_rules[ns] = sampler.rule(s);
            s_firsts[ns] = seqs[idx].st.pos + 1 + level;
            ns += 1;
        } else {
            g_rows[ng] = row;
            g_idx[ng] = idx;
            g_sums[ng] = p.chances != null;
            ng += 1;
        }
    }
    if (ng > 0) try pick.launchGreedyIndexed(e, logits, g_rows[0..ng], g_sums[0..ng]);
    if (ns > 0) {
        try e.smp.gatherIndexed(e.stream, e.exchange(), e.spans, e.share.rank, logits, s_rows[0..ns], e.smp.full.ptr);
        try e.smp.launchMany(e.stream, s_rules[0..ns], s_firsts[0..ns]);
    }
    try e.stream.synchronize();
    try e.checkGathers();
    if (ng > 0) {
        var g_toks: [engine.max_shared]u32 = undefined;
        var g_ch: [engine.max_shared]f64 = undefined;
        try pick.finishGreedyRows(e, ng, g_sums[0..ng], g_toks[0..ng], g_ch[0..ng]);
        for (g_idx[0..ng], g_toks[0..ng], g_ch[0..ng]) |idx, tok, ch| {
            parts[idx].out[level] = tok;
            if (parts[idx].chances) |c| c[level] = ch;
        }
    }
    if (ns > 0) {
        var s_toks: [engine.max_shared]u32 = undefined;
        try e.smp.tok.download(0, std.mem.sliceAsBytes(s_toks[0..ns]));
        for (s_idx[0..ns], s_toks[0..ns]) |idx, tok| {
            parts[idx].out[level] = tok;
            if (parts[idx].chances) |c| c[level] = 1;
        }
    }
}
