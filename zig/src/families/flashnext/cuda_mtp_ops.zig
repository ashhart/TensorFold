//! The head's work over a sequence: chunked prompt passes, its forward and chains, verify windows and rollbacks.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");
const mtp = @import("cuda_mtp.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const vocab = wt.vocab;
const eps = wt.eps;
const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const state_n = wt.state_n;
const out_sk = wt.out_sk;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_dim = wt.index_dim;
const ple_tail_n = wt.ple_tail_n;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const max_chain = mtp.max_chain;
const max_rows = mtp.max_rows;
const conv_bytes = mtp.conv_bytes;
const proj_bytes = mtp.proj_bytes;
const fc_sk = mtp.fc_sk;
const piece_rows = mtp.piece_rows;
const pss_n = mtp.pss_n;
/// Streams one batched head pass drafts for at most.
pub const max_lanes: usize = 16;
const Head = mtp.Head;
const Draft = mtp.Draft;
const Chain = mtp.Chain;
const Pick = mtp.Pick;
const Sampling = mtp.Sampling;
const drawRow = mtp.drawRow;
const maskRow = mtp.maskRow;
const promote = mtp.promote;

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn copyAsync(stream: cuda.Stream, dst: u64, src: u64, n: usize) !void {
    if (n == 0) return;
    const d = stream.d;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream.handle), "cuMemcpyDtoDAsync");
}

fn download(stream: cuda.Stream, dst: []u8, src: u64) !void {
    const d = stream.d;
    try stream.synchronize();
    try d.check(d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
}

/// An owner can mirror main-model mutations to a peer without exposing device pointers on the control wire.
pub const Main = struct {
    ptr: *anyopaque,
    forward: *const fn (*anyopaque, *prompt.Seq, []const u32, batch.Hooks) anyerror!void,
    verify: *const fn (*anyopaque, *prompt.Seq, *Draft, []const u32) anyerror!void,
    rollback: *const fn (*anyopaque, *prompt.Seq, *Draft, usize) anyerror!void,
};

/// The head's work over a sequence: prompt pass, verify windows, rollback, forward and chains; `Tri` launches kernels.
pub fn Ops(comptime Tri: type) type {
    return struct {
        e: *prompt.Engine,
        tri: Tri,
        h: *Head,
        main: ?Main = null,

        const Self = @This();

        fn forwardMain(o: Self, seq: *prompt.Seq, ids: []const u32, hooks: batch.Hooks) !void {
            if (o.main) |m| return m.forward(m.ptr, seq, ids, hooks);
            return o.e.forward(Tri, o.tri, seq, ids, hooks);
        }

        fn stream(o: Self) cuda.Stream {
            return o.e.stream.*;
        }

        /// Prompt rows [at, stop): the pass with its streams kept, then the head over each row with a known next token.
        pub fn prefillChunk(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32, at: usize, stop: usize) !void {
            if (stop > d.cap or stop <= at or stop - at > d.rows_cap) return error.PromptTooLong;
            if (seq.fed != at or d.len != at or d.drafted != 0) return error.PositionMismatch;
            d.media = if (seq.media) |*m| m.view() else null;
            const last = stop == ids.len;
            try o.forwardMain(seq, ids[at..stop], .{ .logits = if (last) .last else .none, .streams = d.streams.ptr });
            const head_to = @min(stop, ids.len - 1);
            if (head_to > at) try o.forwardHead(d, ids[at + 1 .. head_to + 1], d.streams.ptr, false);
            if (last) d.prompt_last = d.streams.ptr + (stop - 1 - at) * wide * 2;
        }

        /// The whole prompt a chunk of the engine's rows at a time, from an empty head cache.
        pub fn prefill(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32) !void {
            d.reset();
            var at: usize = 0;
            while (at < ids.len) {
                const stop = @min(ids.len, at + @min(o.e.chunk, d.rows_cap));
                try o.prefillChunk(seq, d, ids, at, stop);
                at = stop;
            }
        }

        /// The head over `next.len` rows of streams `rows`: cache at d.len, last row in d.out, scores in d.dlogits.
        pub fn forwardHead(o: Self, d: *Draft, next: []const u32, rows: u64, scores: bool) !void {
            const e = o.e;
            const tri = o.tri;
            const h = o.h;
            if (next.len == 0) return;
            if (d.len + next.len > d.cap) return error.PromptTooLong;
            const want = @max(@as(usize, 32), 2 * @min(next.len, piece_rows));
            const s = try e.reserve(want);
            const piece = @max(@as(usize, 1), @min(piece_rows, s.cap / 2));
            // the engine's block scores (the passes share one stream), for rows past 2,048 keys of the head's cache
            const score_rows = try e.reserveScores(d.cap, piece);
            var p: batch.Pass(Tri) = .{
                .gpa = e.gpa,
                .io = e.io,
                .stream = e.stream.*,
                .tri = tri,
                .k = &e.batch_k,
                .s = s,
                .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = (&h.layer)[0..1], .mixer = h.mixer, .head = h.draft, .ngram = &e.ngram },
                .seq = .{ .media = d.media, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows },
                .hooks = .{ .logits = .none },
            };
            var at: usize = 0;
            while (at < next.len) {
                const m = @min(piece, next.len - at);
                p.rows = m;
                p.pos0 = d.len;
                p.pending = null;
                try p.stage(next[at..][0..m]);
                // the embedding branch: the token's row (stream 0 of the four copies), normed, fc_embedding
                if (e.table.dense) {
                    try tri.embedDense(s.ids, e.table.w.ptr, s.ple_keys, dims, streams, m);
                } else {
                    try tri.embed(s.ids, e.table.w.ptr, e.table.s.ptr, e.table.b.ptr, s.ple_keys, dims, streams, m);
                }
                try @import("cuda_media.zig").injectNext(d.media, o.stream(), s.ple_keys, d.len, m);
                try tri.rmsnorm(s.ple_keys, h.norm_e.ptr, s.mixed, s.xsm, eps, wide, m, dims, dims);
                const esk = try p.lane(s.mixed, s.xsm, h.fc_e, s.dummy, s.part, m, fc_sk);
                try p.sumSlices(s.part, s.branch, true, m * dims, esk);
                // the hidden branch: the four streams normed as one row, fc_hidden on each stream
                try tri.rmsnorm(rows + at * wide * 2, h.norm_h.ptr, s.norm, s.gq, eps, wide, m, wide, wide);
                const hsk = try p.lane(s.norm, s.gq, h.fc_h, s.dummy, s.part, streams * m, fc_sk);
                try p.sumSlices(s.part, s.ple_gated, true, streams * m * dims, hsk);
                try tri.addStreams(s.branch, s.ple_gated, s.h, m, dims, streams);
                // the layer: attention over the head's own cache, its experts
                try p.writeback(0, m);
                try p.readout(h.layer.attn_hc, 0, m);
                try p.attnBlock(&h.layer, 0);
                try tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, p.out_slices);
                try p.readout(h.layer.mlp_hc, 0, m);
                try p.moeBlock(&h.layer);
                p.pending = 0;
                try p.writeback(0, m);
                try p.copy(d.out.ptr, s.h + (m - 1) * wide * 2, wide * 2);
                d.len += m;
                at += m;
                if (at == next.len and scores) {
                    // the last row's streams and sums at row 0, where the captured kernels' pointers stay aligned
                    if (m > 1) {
                        try p.copy(s.h, s.h + (m - 1) * wide * 2, wide * 2);
                        try p.copy(s.pss, s.pss + (m - 1) * pss_n * 4, pss_n * 4);
                    }
                    try p.readout(h.mixer, 0, 1);
                    try h.project(&p, s.mixed, s.xsm, d.dlogits.ptr, 1);
                }
            }
        }

        /// Python draft.absorb: the chain's entries trimmed, then the kept rows (main streams `rows`, next tokens).
        pub fn absorb(o: Self, d: *Draft, next: []const u32, rows: u64, scores: bool) !void {
            if (d.drafted > 0) {
                d.len -= d.drafted;
                d.drafted = 0;
            }
            try o.forwardHead(d, next, rows, scores);
        }

        /// The draft at `position` and its temperature-1 probability (Python Engine.sample_draft).
        pub fn pick(o: Self, d: *Draft, position: u64, sm: ?Sampling) !Pick {
            const h = o.h;
            const view_ = h.view();
            const n = view_.logical;
            try h.best(o.stream(), d.dlogits.ptr, n, n, 1, d.best.ptr);
            try download(o.stream(), std.mem.sliceAsBytes(d.best_host[0..4]), d.best.ptr);
            const top = d.best_host[0];
            const lse = d.best_host[1];
            const greedy = if (sm) |s| s.temperature <= 0 else true;
            if (greedy) {
                const col: usize = @intCast(@as(i32, @bitCast(d.best_host[2])));
                return .{ .token = try view_.global(@intCast(col)), .prob = @exp(@as(f64, top) - @as(f64, lse)) };
            }
            const row = d.host[0..n];
            try download(o.stream(), std.mem.sliceAsBytes(row), d.dlogits.ptr);
            const tok = try drawRow(o.e.gpa, row, h.ids, position, sm);
            const col = std.sort.binarySearch(u32, h.ids, tok, struct {
                fn order(t: u32, x: u32) std.math.Order {
                    return std.math.order(t, x);
                }
            }.order) orelse return .{ .token = tok, .prob = 0 };
            return .{ .token = tok, .prob = @exp(@as(f64, promote(row[col])) - @as(f64, lse)) };
        }

        /// Python decode.draft: absorb the kept rows, then chain up to `count` drafts, the first always kept.
        pub fn draft(o: Self, d: *Draft, next: []const u32, rows: u64, position: u64, count: usize, sm: ?Sampling, confidence: f64, out: *Chain) !void {
            out.n = 0;
            try o.absorb(d, next, rows, count > 0);
            var j: usize = 0;
            while (j < @min(count, max_chain)) : (j += 1) {
                const got = try o.pick(d, position + j, sm);
                const low = confidence > 0 and got.prob < confidence;
                if (low and j > 0) break;
                out.tokens[out.n] = got.token;
                out.probs[out.n] = got.prob;
                out.n += 1;
                if (low) break;
                if (j + 1 < count) {
                    try o.forwardHead(d, &.{got.token}, d.out.ptr, true);
                    d.drafted += 1;
                }
            }
        }

        /// forwardHead over several drafts in one pass, a lane each: every row's ops and sums as a lone forwardHead's.
        pub fn forwardHeadMany(o: Self, ds: []const *Draft, nexts: []const []const u32, rows: []const u64, scores: bool) !void {
            const e = o.e;
            const tri = o.tri;
            const h = o.h;
            const n_l = ds.len;
            if (n_l == 0) return;
            var total: usize = 0;
            var cap: usize = 0;
            for (ds, nexts) |d, nx| {
                if (nx.len == 0 or nx.len > piece_rows or d.media != null) return error.UnexpectedTensor;
                if (d.len + nx.len > d.cap) return error.PromptTooLong;
                total += nx.len;
                cap = @max(cap, d.cap);
            }
            const s = try e.reserve(@max(@as(usize, 32), 2 * total));
            if (total > s.cap or n_l > max_lanes) return error.UnexpectedTensor;
            const score_rows = try e.reserveScores(cap, piece_rows);
            var lane_buf: [max_lanes]batch.Lane = undefined;
            var ids_buf: [max_lanes * piece_rows]u32 = undefined;
            var at: usize = 0;
            for (ds, nexts, 0..) |d, nx, i| {
                lane_buf[i] = .{ .seq = .{ .media = null, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows }, .pos = d.len, .rows = nx.len };
                @memcpy(ids_buf[at..][0..nx.len], nx);
                at += nx.len;
            }
            var p: batch.Pass(Tri) = .{
                .gpa = e.gpa,
                .io = e.io,
                .stream = e.stream.*,
                .tri = tri,
                .k = &e.batch_k,
                .s = s,
                .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = (&h.layer)[0..1], .mixer = h.mixer, .head = h.draft, .ngram = &e.ngram },
                .seq = lane_buf[0].seq,
                .hooks = .{ .logits = .none },
                .lanes = lane_buf[0..n_l],
            };
            p.rows = total;
            p.pos0 = 0;
            p.pending = null;
            try p.stage(ids_buf[0..total]);
            if (e.table.dense) {
                try tri.embedDense(s.ids, e.table.w.ptr, s.ple_keys, dims, streams, total);
            } else {
                try tri.embed(s.ids, e.table.w.ptr, e.table.s.ptr, e.table.b.ptr, s.ple_keys, dims, streams, total);
            }
            try tri.rmsnorm(s.ple_keys, h.norm_e.ptr, s.mixed, s.xsm, eps, wide, total, dims, dims);
            const esk = try p.lane(s.mixed, s.xsm, h.fc_e, s.dummy, s.part, total, fc_sk);
            try p.sumSlices(s.part, s.branch, true, total * dims, esk);
            // each lane's rows of main-model streams, side by side (the head reads them as one block of rows)
            at = 0;
            for (rows, nexts) |r, nx| {
                try p.copy(s.ple_nrow + at * wide * 2, r, nx.len * wide * 2);
                at += nx.len;
            }
            try tri.rmsnorm(s.ple_nrow, h.norm_h.ptr, s.norm, s.gq, eps, wide, total, wide, wide);
            const hsk = try p.lane(s.norm, s.gq, h.fc_h, s.dummy, s.part, streams * total, fc_sk);
            try p.sumSlices(s.part, s.ple_gated, true, streams * total * dims, hsk);
            try tri.addStreams(s.branch, s.ple_gated, s.h, total, dims, streams);
            try p.writeback(0, total);
            try p.readout(h.layer.attn_hc, 0, total);
            try p.attnBlock(&h.layer, 0);
            try tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", total * dims, total, dims, streams, 4, 1, 1, p.out_slices);
            try p.readout(h.layer.mlp_hc, 0, total);
            try p.moeBlock(&h.layer);
            p.pending = 0;
            try p.writeback(0, total);
            // each lane's last row: its d.out, and gathered to row i for the draft head (row i <= its last row)
            at = 0;
            for (ds, nexts, 0..) |d, nx, i| {
                const last = at + nx.len - 1;
                try p.copy(d.out.ptr, s.h + last * wide * 2, wide * 2);
                if (scores and last != i) {
                    try p.copy(s.h + i * wide * 2, s.h + last * wide * 2, wide * 2);
                    try p.copy(s.pss + i * pss_n * 4, s.pss + last * pss_n * 4, pss_n * 4);
                }
                d.len += nx.len;
                at += nx.len;
            }
            if (!scores) return;
            try p.readout(h.mixer, 0, n_l);
            const n_ids = h.ids.len;
            _ = try p.lane(s.mixed, s.xsm, h.draft, s.head_local, s.dummy, n_l, 1);
            for (ds, 0..) |d, i| try p.copy(d.dlogits.ptr, s.head_local + i * n_ids * 2, n_ids * 2);
        }

        /// draft over several greedy drafts: kept rows in one pass, then a pass a step, each chaining as it would alone
        pub fn draftMany(o: Self, ds: []const *Draft, nexts: []const []const u32, rows: []const u64, counts: []const usize, confidence: f64, outs: []const *Chain) !void {
            const h = o.h;
            const n_l = ds.len;
            for (ds, outs) |d, out| {
                out.n = 0;
                if (d.drafted > 0) {
                    d.len -= d.drafted;
                    d.drafted = 0;
                }
            }
            try o.forwardHeadMany(ds, nexts, rows, true);
            var act: [max_lanes]usize = undefined;
            var n_act: usize = 0;
            for (counts, 0..) |c, i| if (c > 0) {
                act[n_act] = i;
                n_act += 1;
            };
            var j: usize = 0;
            var best_host: [max_lanes * 4]f32 = undefined;
            var step_ds: [max_lanes]*Draft = undefined;
            var step_next: [max_lanes][1]u32 = undefined;
            var step_nexts: [max_lanes][]const u32 = undefined;
            var step_rows: [max_lanes]u64 = undefined;
            while (n_act > 0 and j < max_chain) : (j += 1) {
                // every active lane's best draft (fn_row_best on its own scores), then one download of all of them
                for (act[0..n_act], 0..) |i, r| try h.best(o.stream(), ds[i].dlogits.ptr, h.ids.len, h.ids.len, 1, ds[0].best.ptr + r * 16);
                try download(o.stream(), std.mem.sliceAsBytes(best_host[0 .. n_act * 4]), ds[0].best.ptr);
                var next_act: usize = 0;
                var n_step: usize = 0;
                for (act[0..n_act], 0..) |i, r| {
                    const top = best_host[r * 4];
                    const lse = best_host[r * 4 + 1];
                    const col: usize = @intCast(@as(i32, @bitCast(best_host[r * 4 + 2])));
                    const got: Pick = .{ .token = h.ids[col], .prob = @exp(@as(f64, top) - @as(f64, lse)) };
                    const out = outs[i];
                    const low = confidence > 0 and got.prob < confidence;
                    if (low and j > 0) continue;
                    out.tokens[out.n] = got.token;
                    out.probs[out.n] = got.prob;
                    out.n += 1;
                    if (low) continue;
                    if (j + 1 < @min(counts[i], max_chain)) {
                        step_ds[n_step] = ds[i];
                        step_next[n_step] = .{got.token};
                        step_nexts[n_step] = &step_next[n_step];
                        step_rows[n_step] = ds[i].out.ptr;
                        n_step += 1;
                        act[next_act] = i;
                        next_act += 1;
                    }
                }
                if (n_step > 0) {
                    try o.forwardHeadMany(step_ds[0..n_step], step_nexts[0..n_step], step_rows[0..n_step], true);
                    for (step_ds[0..n_step]) |d| d.drafted += 1;
                }
                n_act = next_act;
            }
            _ = n_l;
        }

        /// One forward over the window `ids` from seq.fed: each row's scores in d.logits, its streams in d.streams.
        pub fn verify(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32) !void {
            if (o.main) |m| return m.verify(m.ptr, seq, d, ids);
            o.e.round_graph.clear();
            const n = ids.len;
            if (n == 0 or n > max_rows) return error.WindowTooWide;
            if (seq.fed + n > seq.cap) return error.PromptTooLong;
            d.settle(seq);
            d.fed0 = seq.fed;
            d.rows = n;
            d.history = seq.history;
            @memcpy(d.ids[0..n], ids);
            const keeps = n > 1;
            if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.tail_at, seq.tail.ptr, ple_tail_n * wide * 2);
            try o.e.forward(Tri, o.tri, seq, ids, .{ .logits = .all, .logits_out = d.logits.ptr, .keep = if (keeps) &d.keep else null, .streams = d.streams.ptr });
            if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.nrow_at, o.e.batch_s.?.ple_nrow, n * wide * 2);
            d.open = keeps;
        }

        /// Several streams' windows in one forward, each left as its `verify` leaves it; `out` takes all rows' scores.
        pub fn verifyMany(o: Self, seqs: []const *prompt.Seq, ds: []const *Draft, idss: []const []const u32, out: u64) !void {
            if (o.main != null) return error.UnsupportedRankOperation;
            // every window refused before any stream's state moves
            for (seqs, idss) |seq, ids| {
                if (ids.len == 0 or ids.len > max_rows) return error.WindowTooWide;
                if (seq.fed + ids.len > seq.cap) return error.PromptTooLong;
            }
            o.e.round_graph.clear();
            const wins = try o.e.gpa.alloc(prompt.Engine.Window, seqs.len);
            defer o.e.gpa.free(wins);
            for (seqs, ds, idss, wins) |seq, d, ids, *w| {
                const n = ids.len;
                d.settle(seq);
                d.fed0 = seq.fed;
                d.rows = n;
                d.history = seq.history;
                @memcpy(d.ids[0..n], ids);
                const keeps = n > 1;
                if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.tail_at, seq.tail.ptr, ple_tail_n * wide * 2);
                w.* = .{ .seq = seq, .ids = ids, .keep = if (keeps) &d.keep else null, .streams = d.streams.ptr, .logits = d.logits.ptr };
            }
            try o.e.windows(Tri, o.tri, wins, out);
            var row0: usize = 0;
            for (ds, idss) |d, ids| {
                const keeps = ids.len > 1;
                if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.nrow_at, o.e.batch_s.?.ple_nrow + row0 * wide * 2, ids.len * wide * 2);
                d.open = keeps;
                row0 += ids.len;
            }
        }

        /// The window's rows drawn at their keyed positions (fed0 + 1 + r), each within its `masks` row if any.
        pub fn draws(o: Self, d: *Draft, sm: ?Sampling, masks: []const u32, out: []u32) !void {
            const n = d.rows;
            const greedy = if (sm) |s| s.temperature <= 0 else true;
            if (greedy and masks.len == 0) {
                try o.h.best(o.stream(), d.logits.ptr, vocab, vocab, n, d.best.ptr);
                try download(o.stream(), std.mem.sliceAsBytes(d.best_host[0 .. 4 * n]), d.best.ptr);
                for (out[0..n], 0..) |*t, r| t.* = @intCast(@as(i32, @bitCast(d.best_host[4 * r + 2])));
                return;
            }
            const rows = d.host[0 .. n * vocab];
            try download(o.stream(), std.mem.sliceAsBytes(rows), d.logits.ptr);
            const words = masks.len / n;
            for (out[0..n], 0..) |*t, r| {
                const row = rows[r * vocab ..][0..vocab];
                if (words > 0) maskRow(row, masks[r * words ..][0..words]);
                t.* = try drawRow(o.e.gpa, row, null, d.fed0 + 1 + r, sm);
            }
        }

        /// Keep the window's first `keep` rows (Python forward.commit) from its saved rows; a repeat is a no-op.
        pub fn rollback(o: Self, seq: *prompt.Seq, d: *Draft, keep: usize) !void {
            if (o.main) |m| return m.rollback(m.ptr, seq, d, keep);
            o.e.round_graph.clear();
            if (keep == 0 or keep > d.rows) return error.CannotRewind;
            if (keep == d.rows) return d.settle(seq);
            if (!d.open) return error.CannotRewind;
            const e = o.e;
            const st = o.stream();
            const s = if (e.batch_s) |*bs| bs else return error.CannotRewind;
            const k = &e.batch_k;
            seq.fed = d.fed0 + keep;
            var hist = d.history;
            for (d.ids[0..keep]) |t| hist = .{ hist[1], t };
            seq.history = hist;
            try copyAsync(st, seq.tail.ptr, d.saved.ptr + Draft.tail_at, ple_tail_n * wide * 2);
            try shift(k, st, seq.tail.ptr, d.saved.ptr + Draft.nrow_at, wide, keep, ple_tail_n, wide);
            var li: usize = 0;
            for (e.layers[0..e.n_layers], 0..) |*lw, layer| {
                if (!lw.linear) continue;
                defer li += 1;
                // the sequence's state is still the window's start (the pass wrote its end state to the spare)
                const lk = d.keep.layers[layer];
                try copyAsync(st, seq.conv[li].ptr, lk.hist, conv_bytes);
                var f: cuda.Args = .{};
                f.add(lk.proj);
                f.add(seq.conv[li].ptr);
                f.add(lw.cw.ptr);
                f.add(lw.a_log.ptr);
                f.add(lw.dt.ptr);
                f.add(s.gq);
                f.add(s.gk);
                f.add(s.gv);
                f.add(s.ggb);
                try cuda.launch.launch(k.front, .{ .grid = .{ .x = @intCast(keep), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, st, &f);
                var a: cuda.Args = .{};
                a.add(s.gq);
                a.add(s.gk);
                a.add(s.gv);
                a.add(s.ggb);
                a.add(seq.state[li].ptr);
                a.add(cint(keep));
                a.add(s.ys);
                try cuda.launch.launch(k.chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, st, &a);
                try shift(k, st, seq.conv[li].ptr, lk.proj, gdn.proj_width, keep, gdn.taps - 1, gdn.conv_dim);
            }
            d.rows = keep;
            d.open = false;
        }

        fn shift(k: *const batch.Kernels, st: cuda.Stream, win: u64, rows_ptr: u64, stride: usize, n: usize, taps: usize, c: usize) !void {
            var a: cuda.Args = .{};
            a.add(win);
            a.add(rows_ptr);
            a.add(@as(c_longlong, @intCast(stride)));
            a.add(cint(n));
            a.add(cint(taps));
            a.add(cint(c));
            try cuda.launch.launch(k.shift, .{ .grid = .{ .x = @intCast((c + 255) / 256) }, .block = .{ .x = 256 } }, st, &a);
        }
    };
}
