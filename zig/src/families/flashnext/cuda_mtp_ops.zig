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
const Head = mtp.Head;
const Draft = mtp.Draft;
const Chain = mtp.Chain;
const Pick = mtp.Pick;
const Sampling = mtp.Sampling;
const drawRow = mtp.drawRow;
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

/// The head's work over a sequence: prompt pass, verify windows, rollback, forward and chains; `Tri` launches kernels.
pub fn Ops(comptime Tri: type) type {
    return struct {
        e: *prompt.Engine,
        tri: Tri,
        h: *Head,

        const Self = @This();

        fn stream(o: Self) cuda.Stream {
            return o.e.stream.*;
        }

        /// Prompt rows [at, stop): the pass with its streams kept, then the head over each row with a known next token.
        pub fn prefillChunk(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32, at: usize, stop: usize) !void {
            if (stop > d.cap or stop <= at or stop - at > d.rows_cap) return error.PromptTooLong;
            if (seq.fed != at or d.len != at or d.drafted != 0) return error.PositionMismatch;
            const last = stop == ids.len;
            try o.e.forward(Tri, o.tri, seq, ids[at..stop], .{ .logits = if (last) .last else .none, .streams = d.streams.ptr });
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
                .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .layers = (&h.layer)[0..1], .mixer = h.mixer, .head = h.draft, .ngram = &e.ngram },
                .seq = .{ .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows },
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
                try tri.embed(s.ids, e.table.w.ptr, e.table.s.ptr, e.table.b.ptr, s.ple_keys, dims, streams, m);
                try tri.rmsnorm(s.ple_keys, h.norm_e.ptr, s.mixed, s.xsm, eps, wide, m, dims, dims);
                try p.lane(s.mixed, s.xsm, h.fc_e, s.dummy, s.part, m, fc_sk);
                try p.sumSlices(s.part, s.branch, true, m * dims, fc_sk);
                // the hidden branch: the four streams normed as one row, fc_hidden on each stream
                try tri.rmsnorm(rows + at * wide * 2, h.norm_h.ptr, s.norm, s.gq, eps, wide, m, wide, wide);
                try p.lane(s.norm, s.gq, h.fc_h, s.dummy, s.part, streams * m, fc_sk);
                try p.sumSlices(s.part, s.ple_gated, true, streams * m * dims, fc_sk);
                try tri.addStreams(s.branch, s.ple_gated, s.h, m, dims, streams);
                // the layer: attention over the head's own cache, its experts
                try p.writeback(0, m);
                try p.readout(h.layer.attn_hc, 0, m);
                try p.attnBlock(&h.layer, 0);
                try tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, out_sk);
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
                    try p.lane(s.mixed, s.xsm, h.draft, d.dlogits.ptr, s.dummy, 1, 1);
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
            const n = h.ids.len;
            try h.best(o.stream(), d.dlogits.ptr, n, n, 1, d.best.ptr);
            try download(o.stream(), std.mem.sliceAsBytes(d.best_host[0..4]), d.best.ptr);
            const top = d.best_host[0];
            const lse = d.best_host[1];
            const greedy = if (sm) |s| s.temperature <= 0 else true;
            if (greedy) {
                const col: usize = @intCast(@as(i32, @bitCast(d.best_host[2])));
                return .{ .token = h.ids[col], .prob = @exp(@as(f64, top) - @as(f64, lse)) };
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

        /// One forward over the window `ids` from seq.fed: each row's scores in d.logits, its streams in d.streams.
        pub fn verify(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32) !void {
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

        /// The window's rows drawn at their keyed positions (seq position fed0 + 1 + r), as serial decoding draws.
        pub fn draws(o: Self, d: *Draft, sm: ?Sampling, out: []u32) !void {
            const n = d.rows;
            const greedy = if (sm) |s| s.temperature <= 0 else true;
            if (greedy) {
                try o.h.best(o.stream(), d.logits.ptr, vocab, vocab, n, d.best.ptr);
                try download(o.stream(), std.mem.sliceAsBytes(d.best_host[0 .. 4 * n]), d.best.ptr);
                for (out[0..n], 0..) |*t, r| t.* = @intCast(@as(i32, @bitCast(d.best_host[4 * r + 2])));
                return;
            }
            const rows = d.host[0 .. n * vocab];
            try download(o.stream(), std.mem.sliceAsBytes(rows), d.logits.ptr);
            for (out[0..n], 0..) |*t, r| t.* = try drawRow(o.e.gpa, rows[r * vocab ..][0..vocab], null, d.fed0 + 1 + r, sm);
        }

        /// Keep the window's first `keep` rows (Python forward.commit) from its saved rows; a repeat is a no-op.
        pub fn rollback(o: Self, seq: *prompt.Seq, d: *Draft, keep: usize) !void {
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
