//! GLM-5.3-Flash on CUDA: prompts, verify steps, commits and MTP drafts (decode.py), for any set of streams. A prompt
//! runs alone in chunks of 2048 rows; verify steps and draft steps take every stream's rows at once.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kern = @import("kern.zig");
const sq = @import("seq.zig");
const forward = @import("forward.zig");
const blocks = @import("blocks.zig");
const dflash = @import("dflash.zig");
const draw_mod = @import("draw.zig");

const Engine = forward.Engine;
const Buffers = forward.Buffers;
const Seq = sq.Seq;
const Seg = forward.Seg;
const D = kern.D;
const HEAD_N = forward.HEAD_N;

pub const Pick = struct { tok: u32, p: f64 };

/// The gathered candidates -> each row's token (the larger logit; the lower id at a tie: rank 0's) and its
/// probability under the full softmax (both ranks' log-sum-exps).
fn readPicks(e: *Engine, b: *Buffers, rows: i64, out: []Pick) !void {
    const d = e.ctx.d;
    const n: usize = @intCast(2 * rows * 4);
    const w: []i32 = @alignCast(std.mem.bytesAsSlice(i32, e.host[0 .. n * 4]));
    try d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(w.ptr), b.cgath, n * 4, e.k.stream.handle), "candidates");
    try e.k.stream.synchronize();
    if (e.ctx.roce) |g| try g.check();
    const R: usize = @intCast(rows);
    for (0..R) |r| {
        const a = w[4 * r ..][0..4];
        const c = w[4 * (R + r) ..][0..4];
        const v0: f32 = @bitCast(a[1]);
        const v1: f32 = @bitCast(c[1]);
        const second = v1 > v0 or (v1 == v0 and c[0] < a[0]);
        const l0: f64 = @as(f32, @bitCast(a[2]));
        const l1: f64 = @as(f32, @bitCast(c[2]));
        const top = @max(l0, l1);
        const total = top + @log(@exp(l0 - top) + @exp(l1 - top));
        const v: f64 = if (second) v1 else v0;
        out[r] = .{ .tok = @intCast(if (second) c[0] else a[0]), .p = @exp(v - total) };
    }
}

/// A row's draw rule and the position its draw is keyed at (null rule: greedy, read from the candidates).
pub const Draw = struct { rule: ?lanes.Sampling, position: u64 };

/// The sampled rows' tokens: each rank's 64 candidates of every row and its log-sum-exps at the row's temperature
/// (fn_cand_topk), gathered, drawn on the host (draw.zig); a rule past the candidates draws over the row's gathered
/// logits. Both ranks run it alike, so their gathers pair up.
/// TF_GLM_TRACE: sampled rows' draw time (the candidates' kernel, gather and copy; the host's draws) and full rows.
pub var draw_stats: struct { gpu_ms: f64 = 0, host_ms: f64 = 0, rows: u64 = 0, full: u64 = 0, calls: u64 = 0, mismatch: u64 = 0 } = .{};

fn msFrom(e: *Engine, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Timestamp.now(e.io, .awake)).toNanoseconds())) / 1e6;
}

fn drawRows(e: *Engine, b: *Buffers, rows: i64, draws: []const Draw, out: []Pick) !void {
    const k = &e.k;
    const t_gpu = std.Io.Timestamp.now(e.io, .awake);
    var inv: [forward.STEP_ROWS]f32 = undefined;
    for (draws, 0..) |dr, r| inv[r] = if (dr.rule) |rule| @floatCast(1.0 / @max(rule.temperature, 1e-6)) else 1.0;
    try k.upload(b.invt, std.mem.sliceAsBytes(inv[0..@intCast(rows)]));
    const n: usize = @intCast(rows * draw_mod.W);
    // the candidates by radix selection (fn_cand_topk_sel); TF_GLM_TOPK_CHECK=1: the per-thread lists' kernel
    // (fn_cand_topk) first, its bytes compared with the selection's (dev check: the same 64, the same sums)
    const check = topk_check orelse blk: {
        topk_check = std.c.getenv("TF_GLM_TOPK_CHECK") != null;
        break :blk topk_check.?;
    };
    if (check) {
        try candTopk(e, b, rows, "fn_cand_topk");
        try k.stream.synchronize();
        try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(e.cand_host.ptr), b.tk, n * 4), "candidates (check)");
        @memcpy(e.cand_host[n .. 2 * n], e.cand_host[0..n]);
    }
    try candTopk(e, b, rows, "fn_cand_topk_sel");
    if (check) {
        try k.stream.synchronize();
        try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(e.cand_host.ptr), b.tk, n * 4), "candidates (check)");
        if (!std.mem.eql(i32, e.cand_host[0..n], e.cand_host[n .. 2 * n])) {
            draw_stats.mismatch += 1;
            std.log.warn("glm draws: fn_cand_topk_sel differs from fn_cand_topk ({d} rows)", .{rows});
        }
    }
    try e.ctx.allGather(b.tk, b.gtk, n, .i32, "candidates");
    try k.stream.synchronize();
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(e.cand_host.ptr), b.gtk, 2 * n * 4), "candidates");
    draw_stats.gpu_ms += msFrom(e, t_gpu);
    const t_host = std.Io.Timestamp.now(e.io, .awake);
    for (draws, 0..) |dr, r| {
        const rule = dr.rule orelse continue;
        const blocks_ = [2][]const i32{ e.cand_host[r * draw_mod.W ..][0..draw_mod.W], e.cand_host[n + r * draw_mod.W ..][0..draw_mod.W] };
        draw_stats.rows += 1;
        out[r] = .{ .tok = switch (try draw_mod.draw(e.gpa, blocks_, rule, dr.position)) {
            .token => |t| t,
            .full => blk: {
                draw_stats.full += 1;
                break :blk try drawFull(e, b.logits + @as(u64, @intCast(@as(i64, @intCast(r)) * HEAD_N * 2)), rule, dr.position);
            },
        }, .p = 0 };
    }
    draw_stats.host_ms += msFrom(e, t_host);
    draw_stats.calls += 1;
}

var topk_check: ?bool = null;

fn candTopk(e: *Engine, b: *Buffers, rows: i64, name: [:0]const u8) !void {
    const f = try e.k.ext(.sample, name);
    var a: cuda.Args = .{};
    a.add(b.logits);
    a.add(@as(i32, @intCast(HEAD_N)));
    a.add(@as(i32, @intCast(HEAD_N)));
    a.add(@as(i32, @intCast(e.w.vocab_offset)));
    a.add(b.invt);
    a.add(b.tk);
    try e.k.go(f, .{ .x = @intCast(rows) }, 256, 0, &a);
}

fn drawFull(e: *Engine, row: u64, rule: lanes.Sampling, position: u64) !u32 {
    try e.ctx.nccl.check(e.ctx.nccl.api.ncclAllGather(row, e.full.ptr, @intCast(HEAD_N), .bf16, e.ctx.comm, e.k.stream.handle), "full row");
    try e.k.stream.synchronize();
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(e.full_host.ptr), e.full.ptr, e.full_host.len * 2), "full row");
    return draw_mod.drawFullBf16(e.gpa, e.full_host, rule, position);
}

/// The MTP head's rows of one prompt chunk (or a resume's pending row): ids and their hidden rows at mtp_len; the
/// caller advances mtp_len.
fn mtpForward(e: *Engine, b: *Buffers, s: *Seq, next: []const u32, hidden: u64) !void {
    const n: i64 = @intCast(next.len);
    try forward.stageIds(e, b, next);
    if (hidden != b.hin) try e.k.copy(b.hin, hidden, @intCast(n * D * 2));
    const segs = [_]Seg{.{ .s = s, .rows = n }};
    try forward.prepare(e, b, &segs, .mtp);
    try blocks.mtpCompute(e, b);
}

/// The prompt in chunks of 2048 rows from ``start`` (0, or the kept state's point: resumed), each chunk's rows
/// absorbed into the MTP cache, and the first token drawn. ``keep_at`` (0: none): the point the prompt's state is kept
/// at (the stream's kept state), cut inside its chunk when it falls inside one.
pub fn prefill(e: *Engine, s: *Seq, prompt: []const u32, start: usize, keep_at: usize) !u32 {
    defer s.media = null; // attach's images serve this prefill alone
    if (prompt.len == 0 or start >= prompt.len) return error.EmptyPrompt;
    if (keep_at > prompt.len or (keep_at > 0 and keep_at < @max(start, 1))) return error.BadKeepPoint;
    const b = &e.pbuf;
    const n_kda: u64 = @intCast(e.w.nkda);
    const rec_all = n_kda * @as(u64, @intCast(sq.REC * 4));
    const conv_all = n_kda * @as(u64, @intCast(3 * sq.CONV_C * 2));
    e.k.prompt = true;
    defer e.k.prompt = false;
    if (start == 0) {
        try sq.reset(&e.k, s);
    } else { // restore the kept state, then the pending row absorbed with the prompt's token at the point
        if (s.snap_at != start) return error.NoKeptState;
        try e.k.copy(s.recAt(s.cur, 0), s.snap_rec, rec_all);
        try e.k.copy(s.conv, s.snap_conv, conv_all);
        s.pos = @intCast(start);
        s.mtp_len = s.snap_mtp_len;
        s.mtp_drafted = 0;
        try sq.ringSnap(&e.k, s, 0, @intCast(start), true);
        try sq.ringSnap(&e.k, s, 1, s.snap_mtp_len, true);
        if (e.w.mtp != null) {
            try mtpForward(e, b, s, prompt[start .. start + 1], s.snap_pending);
            s.mtp_len += 1;
        }
    }
    // the live caches change from here: a state kept below is the only one valid after (a resume kept at its own
    // point keeps the state it resumed)
    if (!(start > 0 and keep_at == start)) s.snap_at = 0;
    s.d_from = @intCast(start); // DFlash2's ring holds this prompt's rows from here (a kept state's were overwritten since)
    var first: [1]Pick = undefined;
    var at = start;
    while (at < prompt.len) {
        const R: usize = @min(prompt.len - at, @as(usize, @intCast(e.pbuf.rows)));
        try forward.stageIds(e, b, prompt[at .. at + R]);
        const point: usize = if (keep_at > at and keep_at <= at + R) keep_at - at else 0;
        e.cut_at = if (point > 0 and point < R) @intCast(point) else 0;
        if (e.cut_at > 0) try e.k.copy(e.cut_conv, s.conv, conv_all); // the cut's windows start from the chunk's
        const segs = [_]Seg{.{ .s = s, .rows = @intCast(R) }};
        try forward.prepare(e, b, &segs, .verify);
        try blocks.compute(e, b);
        e.cut_at = 0;
        if (e.drafter) |dr| { // the chunk's rows within the first draft's window into the drafter's context
            const lo = @max(at, prompt.len -| @as(usize, @intCast(dflash.WIN - 1)));
            if (lo < at + R) try dflash.ingest(dr, e, &.{.{ .s = s, .taps = dr.taps_prefill + @as(u64, @intCast((lo - at) * 5 * D * 2)), .n = at + R - lo, .pos0 = @intCast(lo) }});
            if (start == 0 and at == 0 and R == prompt.len and e.ctx.rank == 0) try dflash.dumpPrompt(dr, e, R);
        }
        try e.k.copy(s.last_hidden, b.fnormed + @as(u64, @intCast((R - 1) * D * 2)), @intCast(D * 2));
        if (point > 0) { // the snapshot at the kept point (before the MTP head writes its rows)
            if (point < R) {
                try e.k.copy(s.snap_rec, e.cut_rec, rec_all);
                try e.k.copy(s.snap_conv, e.cut_conv, conv_all);
            } else {
                try e.k.copy(s.snap_rec, s.recAt(1 - s.cur, 0), rec_all); // the chunk's states (the parity flips below)
                try e.k.copy(s.snap_conv, s.conv, conv_all); // the windows after the chunk (shifted in every KDA layer)
            }
            try e.k.copy(s.snap_pending, b.fnormed + @as(u64, @intCast((point - 1) * D * 2)), @intCast(D * 2));
            s.snap_mtp_len = @intCast(keep_at - 1);
        }
        // the first token now: the MTP absorb below writes the same logits buffer
        if (at + R == prompt.len) {
            try forward.candidates(e, b, 1);
            if (s.sampled()) |rule| try drawRows(e, b, 1, &.{.{ .rule = rule, .position = prompt.len }}, &first);
        }
        if (e.w.mtp != null) {
            const nxt = prompt[at + 1 .. @min(at + R + 1, prompt.len)];
            if (nxt.len > 0) {
                try mtpForward(e, b, s, nxt, b.fnormed);
                s.mtp_len += @intCast(nxt.len);
            }
        }
        if (point > 0) { // the kept points' incomplete pool rows (the MTP layer's written by the absorb above)
            try sq.ringSnap(&e.k, s, 0, @intCast(keep_at), false);
            try sq.ringSnap(&e.k, s, 1, s.snap_mtp_len, false);
        }
        // a prompt chunk commits whole: its KDA layers already wrote their states (the parity flips) and windows
        s.cur = 1 - s.cur;
        s.pos += @intCast(R);
        if (point > 0) s.snap_at = keep_at;
        at += R;
    }
    if (s.sampled() != null) return first[0].tok;
    try readPicks(e, b, 1, &first);
    return first[0].tok;
}

/// A stream's verify window: the pending token, then its drafts.
pub const Win = struct { s: *Seq, ids: []const u32 };

/// Every window through the model in one step (each at its stream's position); each row's token into ``out`` (rows
/// in window order). The windows stay uncommitted until ``keep``.
pub fn verify(e: *Engine, wins: []const Win, out: []Pick) !void {
    var segs: [forward.MAX_STREAMS]Seg = undefined;
    var ids: [forward.STEP_ROWS]u32 = undefined;
    var draws: [forward.STEP_ROWS]Draw = undefined;
    var R: usize = 0;
    var sampled = false;
    for (wins, 0..) |w, i| {
        if (w.ids.len == 0 or w.ids.len > sq.MAX_ROWS) return error.WindowTooWide;
        segs[i] = .{ .s = w.s, .rows = @intCast(w.ids.len) };
        w.s.hidden = e.buf.fnormed + R * @as(usize, @intCast(D)) * 2;
        for (w.ids, 0..) |t, r| {
            ids[R + r] = t;
            const rule = w.s.sampled();
            draws[R + r] = .{ .rule = rule, .position = @intCast(w.s.pos + 1 + @as(i64, @intCast(r))) };
            if (rule != null) sampled = true;
        }
        R += w.ids.len;
    }
    const b = &e.buf;
    try forward.stageIds(e, b, ids[0..R]);
    try forward.prepare(e, b, segs[0..wins.len], .verify);
    try forward.run(e);
    try readPicks(e, b, @intCast(R), out[0..R]);
    if (sampled) try drawRows(e, b, @intCast(R), draws[0..R], out[0..R]);
}

/// Keep the stream's last verify window's first ``kept`` of ``R`` rows (the KDA states replayed when fewer than all).
pub fn keep(e: *Engine, s: *Seq, R: i64, kept: i64) !void {
    if (kept < 1 or kept > R) return error.BadKeep;
    const n: i64 = @intCast(e.w.nkda);
    if (kept < R) try kern.kdaReplayLayers(&e.k, s.recAt(s.cur, 0), s.ks.k, s.ks.v, s.ks.g, s.ks.b, n, sq.MAX_ROWS, kept, s.recAt(1 - s.cur, 0));
    s.cur = 1 - s.cur;
    try kern.convShift(&e.k, s.conv, s.proj, kept, 3 * sq.CONV_C, sq.MAX_ROWS * sq.PROJ_W, sq.PROJ_W, n, sq.CONV_C);
    s.pos += kept;
}

/// A stream's draft request: its kept rows' final-normed rows (``hidden``, D bf16 apart) and the tokens after them,
/// then up to ``count`` drafts while their probabilities' product stays at or above ``confidence`` (the first always
/// kept). ``got`` and ``out``/``probs`` receive the drafts.
pub const DraftReq = struct { s: *Seq, hidden: u64, next: []const u32, count: usize, confidence: f64, out: []u32, probs: []f64, got: usize = 0 };

/// Every request's rows absorbed into its stream's MTP cache in one step, then the chains, one row a stream a step,
/// each stream leaving at its own confidence cut.
pub fn draft(e: *Engine, reqs: []DraftReq) !void {
    if (e.w.mtp == null) return;
    const b = &e.mbuf;
    var segs: [forward.MAX_STREAMS]Seg = undefined;
    var ids: [forward.STEP_ROWS]u32 = undefined;
    var active: [forward.MAX_STREAMS]usize = undefined;
    var picks: [forward.MAX_STREAMS]Pick = undefined;
    var chain: [forward.MAX_STREAMS]f64 = @splat(1.0);
    var na: usize = 0;
    var R: usize = 0;
    for (reqs, 0..) |*q, i| {
        q.got = 0;
        if (q.count == 0) continue;
        const s = q.s;
        if (s.mtp_drafted > 0) {
            s.mtp_len -= s.mtp_drafted;
            s.mtp_drafted = 0;
        }
        segs[na] = .{ .s = s, .rows = @intCast(q.next.len) };
        @memcpy(ids[R..][0..q.next.len], q.next);
        try e.k.copy(b.hin + R * @as(usize, @intCast(D)) * 2, q.hidden, q.next.len * @as(usize, @intCast(D)) * 2);
        R += q.next.len;
        active[na] = i;
        na += 1;
    }
    if (na == 0) return;
    try forward.stageIds(e, b, ids[0..R]);
    try forward.prepare(e, b, segs[0..na], .mtp);
    try forward.run(e);
    for (segs[0..na]) |g| g.s.mtp_len += g.rows;
    var j: usize = 0;
    while (na > 0) : (j += 1) {
        try readPicks(e, b, @intCast(na), picks[0..na]);
        var keep_n: usize = 0;
        for (0..na) |a| {
            const q = &reqs[active[a]];
            const p = picks[a];
            if (j > 0 and chain[a] * p.p < q.confidence) continue;
            q.out[q.got] = p.tok;
            q.probs[q.got] = p.p;
            q.got += 1;
            chain[a] *= p.p;
            if (chain[a] < q.confidence or j + 1 >= q.count) continue;
            // the stream drafts on: its token and the head's row for it (row a of this step's head rows)
            ids[keep_n] = p.tok;
            try e.k.copy(b.hin + keep_n * @as(usize, @intCast(D)) * 2, b.fnormed + a * @as(usize, @intCast(D)) * 2, @intCast(D * 2));
            segs[keep_n] = .{ .s = q.s, .rows = 1 };
            chain[keep_n] = chain[a];
            active[keep_n] = active[a];
            keep_n += 1;
        }
        na = keep_n;
        if (na == 0) break;
        try forward.stageIds(e, b, ids[0..na]);
        try forward.prepare(e, b, segs[0..na], .mtp);
        try forward.run(e);
        for (segs[0..na]) |g| {
            g.s.mtp_len += 1;
            g.s.mtp_drafted += 1;
        }
    }
}
