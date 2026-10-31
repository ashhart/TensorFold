//! The head's work over a sequence: chunked prompt passes, its forward and chains, verify windows and rollbacks.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const parts = @import("cuda_batch_state.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");
const mtp = @import("cuda_mtp.zig");
const plan = @import("cuda_mtp_plan.zig");
const roll = @import("cuda_mtp_roll.zig");
const pieces = @import("cuda_mtp_pieces.zig");
const draw = @import("cuda_mtp_draw.zig");

/// One group of lanes in the batched head's pass: the lanes' drafts, their rows and whether the head scores.
const Group = struct {
    ds: []const *Draft,
    nexts: []const []const u32,
    rows: []const u64,
    scores: bool,
};

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

        /// Prompt rows [at, stop): the pass (keeping `marks` on its way) and its streams, then the head over the rows.
        pub fn prefillChunk(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32, at: usize, stop: usize, marks: [parts.max_marks]parts.Mark) !void {
            if (stop > d.cap or stop <= at or stop - at > d.rows_cap) return error.PromptTooLong;
            if (seq.fed != at or d.len != at or d.drafted != 0 or d.pending != 0) return error.PositionMismatch;
            d.media = if (seq.media) |*m| m.view() else null;
            const last = stop == ids.len;
            try o.forwardMain(seq, ids[at..stop], .{ .logits = if (last) .last else .none, .streams = d.streams.ptr, .prompt = true, .marks = marks });
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
                try o.prefillChunk(seq, d, ids, at, stop, .{ .{}, .{} });
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
                .seq = .{ .media = d.media, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .pg = d.pg, .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows },
                .hooks = .{ .logits = .none },
            };
            var at: usize = 0;
            while (at < next.len) {
                const m = @min(piece, next.len - at);
                p.rows = m;
                p.pos0 = d.len;
                try p.stage(next[at..][0..m]);
                const one: pieces.Piece(Self, batch.Pass(Tri)) = .{ .o = o, .p = &p, .d = d, .input = rows + at * wide * 2, .m = m, .score = at + m == next.len and scores };
                try pieces.runPiece(one, score_rows);
                d.len += m;
                at += m;
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
            if (sm) |x| if (draw.usable(x)) return draw.pickHead(h, o.stream(), o.e.gpa, d, lse, position, x);
            const row = d.host[0..n];
            try download(o.stream(), std.mem.sliceAsBytes(row), d.dlogits.ptr);
            return mtp.drawPick(o.e.gpa, row, h.ids, lse, position, sm);
        }

        /// Python decode.draft: absorb the kept rows, then chain up to `count` drafts under mtp.Stop, the first kept.
        pub fn draft(o: Self, d: *Draft, next: []const u32, rows: u64, position: u64, count: usize, sm: ?Sampling, confidence: f64, out: *Chain) !void {
            out.n = 0;
            try o.absorb(d, next, rows, count > 0);
            var stop: mtp.Stop = .{ .c = confidence };
            var j: usize = 0;
            while (j < @min(count, max_chain)) : (j += 1) {
                const got = try o.pick(d, position + j, sm);
                const t = stop.take(j, got.prob);
                if (!t.propose) break;
                out.tokens[out.n] = got.token;
                out.probs[out.n] = got.prob;
                out.n += 1;
                if (!t.more) break;
                if (j + 1 < count) {
                    try o.forwardHead(d, &.{got.token}, d.out.ptr, true);
                    d.drafted += 1;
                }
            }
        }

        /// forwardHead over several drafts in one pass, a lane each: every row's ops and sums as a lone forwardHead's.
        pub fn forwardHeadMany(o: Self, ds: []const *Draft, nexts: []const []const u32, rows: []const u64, scores: bool) !void {
            const e = o.e;
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
            if (n_l > max_lanes) return error.UnexpectedTensor;
            // the head's plan: the admitted capacity bounds the pass's rows, so the head never grows the shared scratch
            const pl = plan.plan(e.admittedRows(), total);
            const s = try e.reserve(pl.pass_rows);
            // the engine's block scores, priced to the piece (the single-stream head's rule)
            const score_rows = try e.reserveScores(cap, pl.score_rows);
            // the lanes in groups whose rows fit the piece
            var lane_rows: [max_lanes]usize = undefined;
            for (nexts, 0..) |nx, i| lane_rows[i] = nx.len;
            var group_sizes: [max_lanes]usize = undefined;
            const n_groups = plan.groups(lane_rows[0..n_l], pl.piece, group_sizes[0..]);
            var at: usize = 0;
            for (group_sizes[0..n_groups]) |g| {
                try o.headGroup(.{ .ds = ds[at..][0..g], .nexts = nexts[at..][0..g], .rows = rows[at..][0..g], .scores = scores }, s, score_rows);
                at += g;
            }
        }

        /// One group of lanes' rows in one pass: the head's rows beside the others', each lane's last row its own.
        fn headGroup(o: Self, group: Group, s: *batch.Scratch, score_rows: u64) !void {
            const e = o.e;
            const tri = o.tri;
            const h = o.h;
            const ds = group.ds;
            const nexts = group.nexts;
            const n_l = ds.len;
            var total: usize = 0;
            var lane_buf: [max_lanes]batch.Lane = undefined;
            var ids_buf: [max_lanes * piece_rows]u32 = undefined;
            var at: usize = 0;
            for (ds, nexts, 0..) |d, nx, i| {
                total += nx.len;
                lane_buf[i] = .{ .seq = .{ .media = null, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows }, .pos = d.len, .rows = nx.len };
                @memcpy(ids_buf[at..][0..nx.len], nx);
                at += nx.len;
            }
            if (total > s.cap) return error.UnexpectedTensor;
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
            try pieces.runGroup(pieces.Group(Self, batch.Pass(Tri), Group){ .o = o, .p = &p, .group = group, .total = total }, score_rows);
            for (ds, nexts) |d, nx| d.len += nx.len;
        }

        /// draft over several drafts, greedy or sampled: kept rows in one pass, then one a step, each as it would alone
        pub fn draftMany(o: Self, ds: []const *Draft, nexts: []const []const u32, rows: []const u64, counts: []const usize, positions: []const u64, sms: []const ?Sampling, confidence: f64, outs: []const *Chain) !void {
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
            var stops: [max_lanes]mtp.Stop = undefined;
            for (stops[0..ds.len]) |*st| st.* = .{ .c = confidence };
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
                    // a sampled lane draws from its own row, as its lone pick does; a greedy one takes its best column
                    const got: Pick = if (mtp.isSampled(sms[i])) sampled: {
                        if (draw.usable(sms[i].?)) break :sampled try draw.pickHead(h, o.stream(), o.e.gpa, ds[i], lse, positions[i] + j, sms[i].?);
                        const row = ds[i].host[0..h.view().logical];
                        try download(o.stream(), std.mem.sliceAsBytes(row), ds[i].dlogits.ptr);
                        break :sampled try mtp.drawPick(o.e.gpa, row, h.ids, lse, positions[i] + j, sms[i]);
                    } else .{ .token = h.ids[col], .prob = @exp(@as(f64, top) - @as(f64, lse)) };
                    const out = outs[i];
                    const t = stops[i].take(j, got.prob);
                    if (!t.propose) continue;
                    out.tokens[out.n] = got.token;
                    out.probs[out.n] = got.prob;
                    out.n += 1;
                    if (!t.more) continue;
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
            const keeps = try o.lateBefore(seq, d, n);
            d.fed0 = seq.fed;
            d.rows = n;
            d.history = seq.history;
            @memcpy(d.ids[0..n], ids);
            if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.tail_at, seq.tail.ptr, ple_tail_n * wide * 2);
            try o.e.forward(Tri, o.tri, seq, ids, .{ .logits = .all, .logits_out = d.logits.ptr, .keep = if (keeps) &d.keep else null, .streams = d.streams.ptr, .host_logits = @import("cuda_batch_state.zig").verifyHostLogits() });
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
                const keeps = try o.lateBefore(seq, d, n);
                d.fed0 = seq.fed;
                d.rows = n;
                d.history = seq.history;
                @memcpy(d.ids[0..n], ids);
                if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.tail_at, seq.tail.ptr, ple_tail_n * wide * 2);
                w.* = .{ .seq = seq, .ids = ids, .keep = if (keeps) &d.keep else null, .streams = d.streams.ptr, .logits = d.logits.ptr };
            }
            try o.e.windows(Tri, o.tri, wins, out);
            var row0: usize = 0;
            for (ds, idss, wins) |d, ids, w| {
                const keeps = w.keep != null;
                if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.nrow_at, o.e.batch_s.?.ple_nrow + row0 * wide * 2, ids.len * wide * 2);
                d.open = keeps;
                row0 += ids.len;
            }
        }

        /// Whether an n-row window keeps a rollback; with a log its control word goes to the device (full: flushed).
        fn lateBefore(o: Self, seq: *prompt.Seq, d: *Draft, n: usize) !bool {
            if (d.log == null) return n > 1;
            if (d.pending + n > batch.late_rows) try o.flushLate(seq, d);
            if (n == 1 and d.pending == 0) return false;
            try o.lateCtl(d);
            return true;
        }

        /// The log's first pending row | its pending rows << 16 on the device (fn_late's ctl), in stream order.
        fn lateCtl(o: Self, d: *const Draft) !void {
            const st = o.stream();
            try st.d.check(st.d.api.cuMemsetD32Async(d.log.?.ptr, d.log_first | d.pending << 16, 1, st.handle), "cuMemsetD32Async");
        }

        /// The pending rows applied to the DeltaNet states in place: they stand at seq.fed as without a log.
        pub fn flushLate(o: Self, seq: *prompt.Seq, d: *Draft) !void {
            if (d.log == null or d.pending == 0) return;
            if (d.open) return error.CannotRewind;
            const st = o.stream();
            try o.lateCtl(d);
            var li: usize = 0;
            for (o.e.layers[0..o.e.n_layers], 0..) |*lw, layer| {
                if (!lw.linear) continue;
                defer li += 1;
                const lk = d.keep.layers[layer];
                var a: cuda.Args = .{};
                a.add(@as(u64, 0));
                a.add(lk.log);
                a.add(lk.log_ctl);
                a.add(cint(batch.late_rows));
                a.add(cint(batch.late_rows));
                a.add(seq.state[li].ptr);
                a.add(seq.state[li].ptr);
                a.add(cint(0));
                a.add(@as(u64, 0));
                try cuda.launch.launch(o.e.batch_k.chain_late, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, st, &a);
            }
            d.log_first = 0;
            d.pending = 0;
        }

        /// The window's rows drawn at their keyed positions (fed0 + 1 + r), each within its `masks` row if any.
        pub fn draws(o: Self, d: *Draft, sm: ?Sampling, masks: []const u32, out: []u32) !void {
            return draw.draws(o.h, o.stream(), o.e.gpa, d, sm, masks, out);
        }

        /// Keep the window's first `keep` rows (Python forward.commit) from its saved rows; a repeat is a no-op.
        pub fn rollback(o: Self, seq: *prompt.Seq, d: *Draft, keep: usize) !void {
            if (o.main) |m| return m.rollback(m.ptr, seq, d, keep);
            o.e.round_graph.clear();
            if (keep == 0 or keep > d.rows) return error.CannotRewind;
            if (keep == d.rows) return d.settle(seq);
            if (!d.open) return error.CannotRewind;
            const e = o.e;
            const s = if (e.batch_s) |*bs| bs else return error.CannotRewind;
            seq.fed = d.fed0 + keep;
            var hist = d.history;
            for (d.ids[0..keep]) |t| hist = .{ hist[1], t };
            seq.history = hist;
            if (d.log != null) {
                // pending rows: the kept rows stay in the log; the conv windows and n-gram tail move on in one launch
                try roll.lateRoll(o, seq, d, keep);
                d.keptLate(keep);
                d.rows = keep;
                d.open = false;
                return;
            }
            try roll.replay(e, s, seq, d, keep);
            d.rows = keep;
            d.open = false;
        }
    };
}
