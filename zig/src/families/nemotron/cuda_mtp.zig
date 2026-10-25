//! Nemotron-H's MTP head on CUDA (mtp.py): drafts the tokens after the pending one; the target verifies every one.

const std = @import("std");
const cuda = @import("cuda");
const torch_ops = @import("cuda_torch_ops.zig");
const sampler = @import("cuda_sampler.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const streams = @import("cuda_engine.zig").max_streams;

pub const max_chain = state.max_rows - 1; // drafts a window holds beside the pending token
/// What a sequence keeps of the head between rounds: its caches and its drafts' confidences.
pub const seq_fields = [_][]const u8{ "k_cache", "v_cache", "probs" };
const meta_words = 1 + state.max_rows; // a meta row: the head's position, then each window row's
const host_drafts = meta_words; // pinned layout: meta, drafts, confidences, then a prompt chunk's tokens (int32 words)
const host_probs = host_drafts + max_chain;
const host_tokens = host_probs + max_chain;
const host_metas = host_tokens + state.pass_rows; // a meta row a stream for `chains`

/// The output rows a draft level reads: the draft vocabulary's (the draft head), or the target's whole head.
pub const Vocab = @import("cuda_mtp_projection.zig").Vocab;

/// One stream of a batched draft (`chains`): its head and target buffers, kept rows, depth and head.
pub const Part = struct {
    k_cache: u64,
    v_cache: u64,
    table: u64 = 0, // its sequence's page table (0: contiguous rows)
    probs: u64,
    ids: u64,
    hidden: u64,
    sampled: u64,
    rule: ?u64, // null: greedy drafts
    pos: usize,
    keep: usize,
    depth: usize,
    vocab: Vocab = .subset,
};

/// A stream's rows in a head forward: where they start, how many, its meta row, its head caches and its head.
pub const Span = struct { row0: usize, rows: usize, meta: u64, k_cache: u64, v_cache: u64, vocab: Vocab, table: u64 = 0 };

pub const Head = struct {
    // Host fixtures replace only the network transform; projection, chain scheduling and sample dispatch stay real.
    pub var test_network: if (@import("builtin").is_test) ?*const fn (*Head, []const Span, bool) anyerror!void else void = if (@import("builtin").is_test) null else {};
    e: *Engine,
    buf: cuda.DeviceBuffer,
    pinned: cuda.HostBuffer,
    copied: cuda.Event,
    ready: [max_chain]cuda.Event,
    absorb_graphs: [state.max_rows + 1]?cuda.graph.Exec = @splat(null),
    first_graphs: [4][state.max_rows + 1]?cuda.graph.Exec = @splat(@splat(null)), // by graphSet: draw mode, then head
    chain_graphs: [4][max_chain + 1]?cuda.graph.Exec = @splat(@splat(null)),
    k_cache: u64,
    v_cache: u64,
    meta: u64,
    hin: u64,
    tok: u64,
    probs: u64,
    out: u64,
    emb: u64,
    cat: u64,
    cxs: u64,
    x: u64,
    h1: u64,
    hn: u64,
    row: u64,
    rxs: u64,
    logits: u64,
    flog: u64,
    vals: u64,
    cols: u64,
    cand: u64,
    fp: u64, // zeros: the greedy _keyed's SEED and FP
    outs: u64, // a batched level's output rows, one a stream
    metas: u64, // a batched draft's meta rows, one a stream
    atok: u64,
    topk: u64,
    invalid: u64, // the id lookup's error word: topk's columns are the table's, so it stays zero
    pos: usize = 0,
    keep: usize = 0,
    levels: usize = 0,
    has_buf: bool = false,
    has_pinned: bool = false,
    has_copied: bool = false,
    ready_made: usize = 0,

    /// The head a stream drafts its next round with: the subset while the draft ids cover its context's tail.
    pub fn vocab(h: *const Head, context: []const u32) Vocab {
        const c = if (h.e.w.draft_coverage) |*c| c else return .subset;
        return if (c.covers(context)) .subset else .full;
    }

    /// Logits a row of head `v` holds.
    pub fn width(h: *const Head, v: Vocab) usize {
        return switch (v) {
            .subset => h.e.w.draft_count,
            .full => h.e.c.vocab,
        };
    }

    fn output(h: *const Head, v: Vocab) weights.QLinear {
        return switch (v) {
            .subset => h.e.w.draft_head.?,
            .full => h.e.w.head,
        };
    }

    /// first_graphs' and chain_graphs' set: the bound draw mode (greedy, sampled), then head `v` (subset, full).
    fn graphSet(h: *const Head, v: Vocab) usize {
        return 2 * @as(usize, @intFromBool(h.e.sampling != null)) + @backingInt(v);
    }

    pub fn init(e: *Engine) !*Head {
        if (e.c.bolt or e.w.mtp == null) return error.NoMtpHead;
        const c = e.c;
        const D: usize = c.hidden;
        const R: usize = state.pass_rows;
        const kv: usize = @as(usize, e.max_len) * c.kv_heads * c.head_dim * 2;
        const n = if (e.w.draft_coverage != null) c.vocab else e.w.draft_count;
        const S: usize = streams;
        const sizes = [_]usize{ kv, kv, meta_words * 4, state.max_rows * D * 2, state.max_rows * 4, max_chain * 4, D * 2, R * D * 2, R * 2 * D * 2, R * 2 * D / 64 * 4, R * D * 2, S * D * 2, S * D * 2, S * D * 2, S * D / 64 * 4, S * n * 2, n * 4, sampler.max_candidates * 4, sampler.max_candidates * 8, sampler.max_candidates * 8, 32, S * D * 2, S * meta_words * 4, R * 4, torch_ops.topkScratchBytes(1, n), 4 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        const h = try e.gpa.create(Head);
        errdefer e.gpa.destroy(h);
        h.* = .{ .e = e, .buf = undefined, .pinned = undefined, .copied = undefined, .ready = undefined, .k_cache = 0, .v_cache = 0, .meta = 0, .hin = 0, .tok = 0, .probs = 0, .out = 0, .emb = 0, .cat = 0, .cxs = 0, .x = 0, .h1 = 0, .hn = 0, .row = 0, .rxs = 0, .logits = 0, .flog = 0, .vals = 0, .cols = 0, .cand = 0, .fp = 0, .outs = 0, .metas = 0, .atok = 0, .topk = 0, .invalid = 0 };
        errdefer if (!h.cleanup()) @panic("MTP construction completion failed; keeping all parents until process exit");
        h.buf = try cuda.DeviceBuffer.alloc(e.ctx.d, total);
        h.has_buf = true;
        const fields = [_]*u64{ &h.k_cache, &h.v_cache, &h.meta, &h.hin, &h.tok, &h.probs, &h.out, &h.emb, &h.cat, &h.cxs, &h.x, &h.h1, &h.hn, &h.row, &h.rxs, &h.logits, &h.flog, &h.vals, &h.cols, &h.cand, &h.fp, &h.outs, &h.metas, &h.atok, &h.topk, &h.invalid };
        var at: usize = 0;
        for (fields, sizes) |f, s| {
            f.* = h.buf.ptr + at;
            at += std.mem.alignForward(usize, s, 256);
        }
        try h.buf.fill8(0, e.stream.handle); // on the engine's stream: the legacy one would race its first uploads
        h.pinned = try cuda.HostBuffer.alloc(e.ctx.d, (host_metas + S * meta_words) * 4);
        h.has_pinned = true;
        h.copied = try cuda.Event.init(e.ctx.d, false);
        h.has_copied = true;
        try h.copied.record(e.stream);
        for (&h.ready) |*r| {
            r.* = try cuda.Event.init(e.ctx.d, false);
            h.ready_made += 1;
        }
        e.head = h;
        inline for (seq_fields, &e.own.head) |name, *ptr| ptr.* = @field(h, name);
        return h;
    }

    pub fn deinit(h: *Head) void {
        if (!h.tryDeinit()) @panic("MTP completion failed; keeping all parents until process exit");
    }
    /// False leaves this head and every engine parent alive for the caller to retry.
    pub fn tryDeinit(h: *Head) bool {
        if (!h.cleanup()) return false;
        if (h.e.head == h) h.e.head = null;
        h.e.gpa.destroy(h);
        return true;
    }
    fn cleanup(h: *Head) bool {
        if (h.has_buf) h.e.stream.synchronize() catch return false;
        for (&h.absorb_graphs) |*g| if (g.*) |*x| x.deinit();
        for (&h.first_graphs) |*set| for (set) |*g| if (g.*) |*x| x.deinit();
        for (&h.chain_graphs) |*set| for (set) |*g| if (g.*) |*x| x.deinit();
        for (h.ready[0..h.ready_made]) |*r| r.deinit();
        h.ready_made = 0;
        if (h.has_copied) h.copied.deinit();
        if (h.has_pinned) h.pinned.free();
        if (h.has_buf) h.buf.free();
        h.has_copied = false;
        h.has_pinned = false;
        h.has_buf = false;
        return true;
    }

    /// Bytes a snapshot of the head's KV cache takes.
    pub fn snapshotBytes(h: *const Head) usize {
        return h.v_cache - h.k_cache + h.kvBytes();
    }

    /// MTPHead.snapshot: the head's KV cache (its position is the caller's to keep).
    pub fn snapshot(h: *Head) !cuda.DeviceBuffer {
        var copy = try cuda.DeviceBuffer.alloc(h.e.ctx.d, h.snapshotBytes());
        errdefer copy.free();
        try h.e.ops().copy(copy.ptr, h.k_cache, copy.len);
        return copy;
    }

    pub fn restore(h: *Head, saved: cuda.DeviceBuffer, pos: usize) !void {
        try h.e.ops().copy(h.k_cache, saved.ptr, saved.len);
        h.pos = pos;
    }

    /// Each seq_fields buffer's bytes.
    pub fn seqSizes(h: *const Head) [seq_fields.len]usize {
        return .{ h.kvBytes(), h.kvBytes(), max_chain * 4 };
    }

    /// The engine bound sequence `s`: its head buffers and position.
    pub fn bindSeq(h: *Head, s: *const state.Seq) void {
        inline for (seq_fields, s.head) |name, ptr| @field(h, name) = ptr;
        h.pos = s.head_pos;
    }

    fn kvBytes(h: *const Head) usize {
        return @as(usize, h.e.max_len) * h.e.c.kv_heads * h.e.c.head_dim * 2;
    }

    /// The head's caches zeroed (a paged sequence's pages are written before any read: not those) and position 0.
    pub fn reset(h: *Head) !void {
        if (h.e.b.table == 0) {
            try h.e.ops().fill32(h.k_cache, 0, h.kvBytes() / 4);
            try h.e.ops().fill32(h.v_cache, 0, h.kvBytes() / 4);
        }
        h.pos = 0;
    }

    /// MTPHead._forward: rows at the head's positions; with `tail`, the last row's output and head `v`'s logits.
    fn forward(h: *Head, rows: usize, tail: bool, meta: u64, v: Vocab) !void {
        _ = try h.forwardSpans(&.{.{ .row0 = 0, .rows = rows, .meta = meta, .k_cache = h.k_cache, .v_cache = h.v_cache, .vocab = v, .table = h.e.b.table }}, tail);
    }

    /// Each span's rows at its head positions; with `tail`, logits row k is span k's by its own head (stride returned).
    fn forwardSpans(h: *Head, spans: []const Span, tail: bool) !usize {
        try h.networkSpans(spans, tail);
        if (!tail) return 0;
        const c = h.e.c;
        var o = h.e.ops();
        o.ck8 = h.e.b.ck8; // the FP8 rows a checkpoint head quantizes into, as the forward's ops carry them
        return @import("cuda_mtp_projection.zig").project(o, spans, [2]weights.QLinear{ h.e.w.draft_head.?, h.e.w.head }, .{ h.e.w.draft_count, c.vocab }, h.row, h.rxs, h.logits, c.hidden);
    }
    fn networkSpans(h: *Head, spans: []const Span, tail: bool) !void {
        if (comptime @import("builtin").is_test) if (test_network) |run| return run(h, spans, tail);
        const e = h.e;
        const c = e.c;
        const m = e.w.mtp.?;
        const b = &e.b;
        const f = e.forward(null);
        const o = f.ops;
        const t = f.glue;
        const D: u64 = c.hidden;
        const qd: u64 = c.heads * c.head_dim;
        var rows: usize = 0;
        for (spans) |s| rows += s.rows;
        try f.embed(h.tok, h.emb, rows);
        try t.concatNorms(h.emb, h.hin, m.enorm, m.hnorm, h.cat, h.cxs, rows, c.hidden, c.eps);
        try o.dense(h.cat, h.cxs, m.eh_proj, h.x, rows);
        try t.addRmsnorm(h.x, null, m.attn_norm, h.x, b.y, b.xs, rows, c.hidden, c.eps);
        try o.dense(b.y, b.xs, m.attn.qkv, b.qkv, rows);
        for (spans) |s| {
            const r0: u64 = s.row0;
            const qkv = b.qkv + r0 * c.qkvDim() * 2;
            var at = f.ashape();
            at.pt = s.table;
            try t.kvWrite(qkv, s.k_cache, s.v_cache, s.meta, s.rows, at);
            try t.attention(qkv, s.k_cache, s.v_cache, s.meta, b.po + r0 * f.nch * qd * 4, b.pm + r0 * f.nch * c.heads * 4, b.pl + r0 * f.nch * c.heads * 4, b.att + r0 * qd * 2, b.axs + r0 * (qd / 64) * 4, s.rows, at);
        }
        try o.dense(b.att, b.axs, m.attn.o, b.delta, rows);
        if (!tail) return;
        if (spans.len == 1) {
            const last = spans[0].row0 + spans[0].rows - 1;
            try t.addRmsnorm(h.x + last * D * 2, b.delta + last * D * 2, m.moe_norm, h.h1, b.y, b.xs, 1, c.hidden, c.eps);
        } else {
            // span k's last row moves to row k: one norm launch on aligned rows, which a captured set needs
            for (spans, 0..) |s, k| {
                const last = s.row0 + s.rows - 1;
                if (last == k) continue;
                try o.copy(h.x + k * D * 2, h.x + last * D * 2, D * 2);
                try o.copy(b.delta + k * D * 2, b.delta + last * D * 2, D * 2);
            }
            try t.addRmsnorm(h.x, b.delta, m.moe_norm, h.h1, b.y, b.xs, spans.len, c.hidden, c.eps);
        }
        if (m.moe.latent != null) {
            try @import("cuda_latent.zig").run(&f, m.moe, spans.len);
            try t.addRmsnorm(h.h1, b.delta, m.final_norm, h.hn, h.row, h.rxs, spans.len, c.hidden, c.eps);
        } else {
            try f.experts(m.moe, spans.len, false);
            try t.addMoeNorm(h.h1, b.ymoe, true, b.wts, m.final_norm, h.hn, h.row, h.rxs, b.delta, spans.len, c.hidden, c.eps, c.top_k, c.slots());
        }
    }

    /// MTPHead._level: 0 absorbs the window's kept rows, 1 absorbs and drafts after them, j > 1 extends; `v` drafts.
    pub fn level(h: *Head, rows: usize, j: usize, v: Vocab) !void {
        const e = h.e;
        const o = e.ops();
        const D: u64 = e.c.hidden;
        if (j <= 1) {
            try o.copy(h.hin, e.b.hidden, rows * D * 2);
            try o.copy(h.tok, e.b.sampled, rows * 4);
        } else {
            try o.copy(h.hin, h.out, D * 2);
            try o.copy(h.tok, e.b.ids + (j - 1) * 4, 4);
        }
        if (j == 0) return h.forward(rows, false, h.meta, v);
        try h.forward(rows, true, if (j == 1) h.meta else h.meta + (j - 1) * 4, v);
        try o.copy(h.out, h.row, D * 2);
        try h.sample(h.logits, if (e.sampling != null) e.b.rule else null, h.meta + 4, e.b.ids + j * 4, h.probs + (j - 1) * 4, j, v);
        try o.download(std.mem.sliceAsBytes(h.pinned.slice(u32)[host_drafts + j - 1 ..][0..1]), e.b.ids + j * 4);
        try o.download(std.mem.sliceAsBytes(h.pinned.slice(u32)[host_probs + j - 1 ..][0..1]), h.probs + (j - 1) * 4);
    }

    /// Level j's draft from head `v`'s logits: by the stream's rule (sample.cu), else greedy _keyed over the top 28.
    fn sample(h: *Head, logits: u64, rule: ?u64, meta: u64, out: u64, prob: u64, j: usize, v: Vocab) !void {
        const e = h.e;
        const n = h.width(v);
        const ids: ?u64 = if (v == .subset) e.w.draft_ids else null; // the whole head's columns are the ids
        if (rule) |r| return e.ops().draw(logits, n, r, meta, j - 1, out, 1, ids, prob);
        const k = sampler.greedy_draft;
        const count = sampler.count(k.k, n);
        const t = e.ops().torch();
        try t.toF32(logits, h.flog, n);
        try t.topk(h.flog, n, 1, count, h.vals, h.cols, h.topk);
        if (ids) |x| try t.lookup(x, n, h.cols, h.cand, count, h.invalid);
        // GREEDY loads SEED and FP but reads neither: the head's zeroed fp serves as both
        try e.forward(null).glue.keyed(h.vals, if (ids != null) h.cand else h.cols, meta, out, h.fp, h.fp, prob, j - 1, 1, count, k);
    }

    /// MTPHead.capture: levels 0 and 1 at every kept-row count, later levels at one row, in the draw mode, per head.
    pub fn capture(h: *Head) !void {
        const rows = state.windowRows(h.e.c);
        for ([_]Vocab{ .subset, .full }) |v| {
            if (v == .full and h.e.w.draft_coverage == null) continue;
            const m = h.graphSet(v);
            for (1..rows + 1) |k| {
                if (h.absorb_graphs[k] == null) h.absorb_graphs[k] = try h.record(k, 0, v);
                h.first_graphs[m][k] = try h.record(k, 1, v);
            }
            for (2..rows) |j| h.chain_graphs[m][j] = try h.record(1, j, v);
        }
    }

    fn record(h: *Head, rows: usize, j: usize, v: Vocab) !cuda.graph.Exec {
        return h.e.record(.{ .head = .{ .h = h, .rows = rows, .j = j, .vocab = v } });
    }

    /// MTPHead.begin: the head absorbs the last window's first `keep` rows, then drafts the positions after them.
    pub fn begin(h: *Head, keep: usize) !void {
        try h.copied.synchronize();
        const m = h.pinned.slice(u32)[0..meta_words];
        m[0] = @intCast(h.pos);
        for (0..max_chain + 1) |j| m[1 + j] = @intCast(h.pos + keep + j);
        try h.e.ops().upload(h.meta, std.mem.sliceAsBytes(m));
        try h.copied.record(h.e.stream);
        h.pos += keep;
        h.keep = keep;
        h.levels = 0;
    }

    /// begin again for the `keep` rows the head last began (another driver resumes the stream): its meta rewritten.
    pub fn again(h: *Head, keep: usize) !void {
        if (keep > h.pos) return error.BadDraftRequest;
        h.pos -= keep;
        try h.begin(keep);
    }

    /// MTPHead.level: queue level j of this round (0: absorb only) into the engine's ids[j], drafted by head `v`.
    pub fn launch(h: *Head, j: usize, v: Vocab) !void {
        const m = h.graphSet(v);
        const rows = if (j <= 1) h.keep else 1;
        const g = if (j == 0) h.absorb_graphs[h.keep] else if (j == 1) h.first_graphs[m][h.keep] else h.chain_graphs[m][j];
        const w = state.max_rows + 1;
        const id = if (j == 0) 1000 + h.keep else if (j == 1) 2000 + m * w + h.keep else 3000 + m * (max_chain + 1) + j;
        if (g) |x| try h.e.launchGraph(x, @intCast(id), .{ .head = .{ .h = h, .rows = rows, .j = j, .vocab = v } }) else try h.level(rows, j, v);
        if (j > 0) {
            try h.ready[j - 1].record(h.e.stream);
            h.levels = j;
        }
    }

    /// The lane core's draft request: absorb `follow.len` kept rows (follow: the token after each), then `depth` levels.
    pub fn chain(h: *Head, follow: []const u32, depth: usize, v: Vocab) !void {
        if (follow.len < 1 or follow.len > state.windowRows(h.e.c) or depth >= state.windowRows(h.e.c)) return error.BadDraftRequest;
        try h.e.ops().upload(h.e.b.sampled, std.mem.sliceAsBytes(follow));
        try h.begin(follow.len);
        if (depth == 0) return h.launch(0, v);
        for (1..depth + 1) |j| try h.launch(j, v);
    }

    /// Several streams' draft requests level by level, a row a stream: drafts to their `ids`, confidences to `probs`.
    pub fn chains(h: *Head, parts: []Part) !void {
        const e = h.e;
        const o = e.ops();
        const D: u64 = e.c.hidden;
        if (parts.len < 1 or parts.len > streams) return error.BadDraftRequest;
        var total: usize = 0;
        for (parts) |p| {
            if (p.keep < 1 or p.depth < 1 or p.depth >= state.windowRows(e.c)) return error.BadDraftRequest;
            total += p.keep;
        }
        if (total > state.windowRows(e.c)) return error.BadDraftRequest;
        // the deepest first: the streams still drafting at a level are the leading rows of the last level's output
        std.mem.sort(Part, parts, {}, struct {
            fn deeper(_: void, x: Part, y: Part) bool {
                return x.depth > y.depth;
            }
        }.deeper);
        try h.copied.synchronize();
        const host = h.pinned.slice(u32)[host_metas..][0 .. parts.len * meta_words];
        for (parts, 0..) |p, k| {
            const m = host[k * meta_words ..][0..meta_words];
            m[0] = @intCast(p.pos);
            for (0..max_chain + 1) |j| m[1 + j] = @intCast(p.pos + p.keep + j);
        }
        try o.upload(h.metas, std.mem.sliceAsBytes(host));
        try h.copied.record(e.stream);
        var spans: [streams]Span = undefined;
        var row0: usize = 0;
        for (parts, spans[0..parts.len]) |p, *s| {
            try o.copy(h.hin + row0 * D * 2, p.hidden, p.keep * D * 2);
            try o.copy(h.tok + row0 * 4, p.sampled, p.keep * 4);
            s.* = .{ .row0 = row0, .rows = p.keep, .meta = 0, .k_cache = p.k_cache, .v_cache = p.v_cache, .table = p.table, .vocab = p.vocab };
            row0 += p.keep;
        }
        var live = parts.len;
        for (1..parts[0].depth + 1) |j| {
            while (parts[live - 1].depth < j) live -= 1;
            if (j > 1) try o.copy(h.hin, h.outs, live * D * 2);
            for (parts[0..live], spans[0..live], 0..) |p, *s, k| {
                s.meta = h.metas + k * meta_words * 4 + (j - 1) * 4;
                if (j == 1) continue;
                try o.copy(h.tok + k * 4, p.ids + (j - 1) * 4, 4);
                s.row0 = k;
                s.rows = 1;
            }
            const n = try h.forwardSpans(spans[0..live], true);
            try o.copy(h.outs, h.row, live * D * 2);
            for (parts[0..live], 0..) |p, k| {
                const meta = h.metas + k * meta_words * 4;
                try h.sample(h.logits + k * n * 2, p.rule, meta + 4, p.ids + j * 4, p.probs + (j - 1) * 4, j, p.vocab);
            }
        }
    }

    /// Draft j's confidence: its share of the head's top-k, at temperature 1 or the request's (waits for level j).
    pub fn confidence(h: *Head, j: usize) !f32 {
        try h.ready[j - 1].synchronize();
        const p: *volatile f32 = &h.pinned.slice(f32)[host_probs + j - 1];
        return p.*;
    }

    /// The drafts of this round's levels (valid once their levels have run).
    pub fn drafts(h: *Head) []const u32 {
        return h.pinned.slice(u32)[host_drafts..][0..h.levels];
    }

    /// One draw from host draft-head logits (bf16 bytes), as level `offset + 1` makes it: the draft and its confidence.
    pub fn draw(h: *Head, logits: []const u8, offset: u32) !struct { token: u32, prob: f32 } {
        const e = h.e;
        try e.ops().upload(h.logits, logits);
        const j = offset + 1;
        try h.sample(h.logits, if (e.sampling != null) e.b.rule else null, h.meta + 4, e.b.ids + j * 4, h.probs + (j - 1) * 4, j, .subset);
        var tok: [1]u32 = undefined;
        var prob: [1]f32 = undefined;
        try e.ops().download(std.mem.sliceAsBytes(&tok), e.b.ids + j * 4);
        try e.ops().download(std.mem.sliceAsBytes(&prob), h.probs + (j - 1) * 4);
        try e.stream.synchronize();
        return .{ .token = tok[0], .prob = prob[0] };
    }

    /// MTPHead.absorb_rows: a prompt chunk's rows into the head's cache, keys and values only.
    pub fn absorb(h: *Head, hidden: u64, tokens: []const u32) !void {
        const e = h.e;
        const c = e.c;
        const m = e.w.mtp.?;
        const b = &e.b;
        const o = e.ops();
        const f = e.forward(null);
        const rows = tokens.len;
        if (rows == 0) return;
        try h.copied.synchronize();
        const host = h.pinned.slice(u32)[host_tokens..][0..tokens.len];
        @memcpy(host, tokens);
        try o.upload(h.atok, std.mem.sliceAsBytes(host));
        try f.embed(h.atok, h.emb, rows);
        try f.glue.concatNorms(h.emb, hidden, m.enorm, m.hnorm, h.cat, h.cxs, rows, c.hidden, c.eps);
        try o.prefillDense(h.cat, m.eh_proj, h.x, rows);
        try f.glue.addRmsnorm(h.x, null, m.attn_norm, h.x, b.y, b.xs, rows, c.hidden, c.eps);
        try o.prefillDense(b.y, m.attn.qkv, b.qkv, rows);
        try o.fill32(b.p_meta, @intCast(h.pos), 4);
        try f.glue.kvWrite(b.qkv, h.k_cache, h.v_cache, b.p_meta, rows, f.ashape());
        try h.copied.record(e.stream);
        h.pos += rows;
    }
};
