//! Flash Next engine prompt passes and served decode rounds.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const fz = @import("replay.zig");
const segments = @import("../../core/segments.zig");
const CallLog = @import("call_log.zig").CallLog;
const follow_mod = @import("follow.zig");
const marks_mod = @import("marks.zig");
const think = @import("think.zig");
const sample = @import("sample.zig");
const Think = @import("engine.zig").Think;
const WIDE = fz.WIDE;
const VOCAB = fz.VOCAB;
const PLE_TAIL = fz.PLE_TAIL;
const MAXR = fz.MAXR;
const TOP = fz.TOP;
const Buf = fz.Buf;
const Prompt = fz.Prompt;
const DepthRule = fz.DepthRule;
const CostDepth = @import("drafts_cost.zig").CostDepth;
const RING = @import("engine.zig").RING;
const tree_mod = @import("tree.zig");
const gate_mod = @import("gate.zig");
const MARGIN = @import("engine.zig").MARGIN;
const Result = @import("engine.zig").Result;
const Out = @import("engine.zig").Out;
const Engine = @import("engine.zig").Engine;
const load = Engine.load;
const rounds = Engine.rounds;
const hostMode = Engine.hostMode;
const gpuMode = Engine.gpuMode;
const statesCopy = Engine.statesCopy;
const chunkCall = Engine.chunkCall;
const isEos = Engine.isEos;

pub fn generate(e: *Engine, prompt: []const u32, max_tokens: usize, eos: []const u32, depth: ?usize, out: Out) !Result {
    return e.generateFrom(prompt, 0, &.{}, max_tokens, eos, depth, .{}, out);
}

pub fn generateFrom(e: *Engine, prompt: []const u32, from: usize, marks: []const u32, max_tokens: usize, eos: []const u32, depth: ?usize, budget: Think, out: Out) !Result {
    const r = e.r;
    const m = e.m;
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len + max_tokens + MARGIN > e.m.cap) return error.ContextFull;
    if (from >= prompt.len or (from > 0 and m.pos != from)) return error.NotResumed;
    for (marks, 0..) |mk, i| if (mk <= from or mk >= prompt.len or (i > 0 and mk <= marks[i - 1])) return error.Marks;
    if (out.sampling != null and r.tp != null) return error.SpeedUpGreedyOnly; // each Mac holds half the logits
    var watch = try think.Watch.init(budget);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (r.tp) |tp| if (tp.rank == 0) { // served speed-up mode: rank 1 runs the same request (follow)
        const t0 = platform.c.mach_absolute_time();
        defer e.handoff_ns = @intCast((platform.c.mach_absolute_time() - t0) * 125 / 3); // mach ticks are 125/3 ns on Apple silicon
        try follow_mod.send(e, tp, prompt, from, marks, max_tokens, eos, depth, budget);
        if (!try tp.waitAck()) return error.PeerNotResumed;
    };
    e.hostMode();
    defer e.hostMode();
    if (from == 0) m.reset();
    m.mtp.pos = 0;
    m.mtp.drafted = 0;
    var pick: u32 = 0;
    var at: usize = from;
    var last_n: usize = 1;
    var dr: marks_mod.Driver = .{ .marks = marks };
    const ps = [2]*Prompt{ e.pr, e.pr2 };
    var calls: CallLog = .{};
    while (at < prompt.len) {
        if (try e.agree(out.cancelled(out.ctx))) return .{ .reason = .cancelled };
        const t_call = platform.c.mach_absolute_time();
        const g_call = m.gpu_seconds;
        if (r.tp) |tp| { // speed-up mode: the call's rows split across the two Macs, running on past up to MARKS marks
            const pair_taps = e.mark_taps and m.marks != null;
            var call = segments.next(if (pair_taps) prompt.len - at else dr.toNext(at, prompt.len), e.pr.step, e.pair_min);
            var passed: [fz.MARKS]u32 = undefined;
            var n_passed: usize = 0;
            if (call.parts == 2 and pair_taps) {
                const in_ = dr.inside(at, call.rows, &passed);
                n_passed = in_.n;
                if (in_.cut) |cut| call = segments.next(cut, e.pr.step, e.pair_min);
            }
            if (call.parts == 2) {
                const hist0 = m.ple.hist;
                pick = try Prompt.chunkPair(e.pr, m, e.gpa, prompt[at .. at + call.rows], tp, passed[0..n_passed]);
                var span: [2][2]usize = undefined; // each Mac's segment: first position, rows with an MTP key
                for (0..2) |k| {
                    const s = at + segments.start(call.rows, 2, k);
                    const n = segments.rows(call.rows, 2, k);
                    span[k] = .{ s, if (s + n < prompt.len) n else n - 1 };
                }
                const mine = span[tp.rank];
                try e.pr.mtpKeys(m, mine[0], prompt[mine[0] + 1 .. mine[0] + 1 + mine[1]], e.pr.last);
                try e.pr.pairMtp(m, tp, mine, span[1 - tp.rank]);
                const last_call = at + call.rows == prompt.len;
                try dr.cross(e, passed[0..n_passed], segments.rows(call.rows, 2, 0), segments.start(call.rows, 2, tp.rank), last_call);
                last_n = 1; // m.last points at the last row
                for (passed[0..n_passed], 0..) |row, j| { // both Macs hold each passed mark's states once it crosses
                    dr.passed(e, out, .{ .at = at + row, .slot = j, .tail = m.marks.?.tail_at(j), .hist = marks_mod.histAt(hist0, prompt, at, row) }, last_call);
                }
                calls.note(call.rows, 2, t_call, g_call, m.gpu_seconds);
                at += call.rows;
                dr.after(out, at, n_passed); // a pair chunk can end on a mark too
                continue;
            }
        }
        // a call runs on past marks and writes the DeltaNet states there (up to MARKS); more marks end it at the...
        const taps = e.mark_taps and m.marks != null;
        var c: segments.Call = e.chunkCall(if (taps) prompt.len - at else dr.toNext(at, prompt.len));
        var inside: [fz.MARKS]u32 = undefined;
        var n_in: usize = 0;
        if (taps) {
            const in_ = dr.inside(at, c.rows, &inside);
            n_in = in_.n;
            if (in_.cut) |cut| c = e.chunkCall(cut);
        }
        const hist0 = m.ple.hist; // the two tokens before the call: the n-gram history at a passed mark follows from them
        pick = try Prompt.chunkMarked(ps[0..c.parts], m, e.gpa, prompt[at .. at + c.rows], inside[0..n_in]);
        if (at + c.rows == prompt.len) if (out.sampling) |p| { // the first token, from the last row
            e.draw_at.b.slice(u32, MAXR)[0] = @intCast(prompt.len);
            const cb = r.queue.commandBuffer();
            r.begin(cb, .serial);
            errdefer if (r.fsr != null) r.abort();
            sample.draw(r.enc, r.draw_pipe, m.t.logits, e.draw_at, m.t.picks, 1, VOCAB, p);
            try m.finish(cb);
            pick = m.t.picks.b.slice(u32, 1)[0];
        };
        for (ps[0..c.parts], 0..) |p, k| { // the MTP head's keys for each segment's rows, from its streams
            const s = at + segments.start(c.rows, c.parts, k);
            const n = segments.rows(c.rows, c.parts, k);
            const nexts = if (s + n < prompt.len) n else n - 1;
            try p.mtpKeys(m, s, prompt[s + 1 .. s + 1 + nexts], p.last);
            last_n = n;
        }
        for (inside[0..n_in], 0..) |row, j| { // the passed marks: the cache keeps each from where the call left it
            const seg = marks_mod.markSeg(ps[0..c.parts], c.rows, row);
            dr.passed(e, out, .{ .at = at + row, .slot = j, .tail = .{ .b = seg.p.b.cin.b, .off = seg.p.b.cin.off + seg.row * WIDE * 2 }, .hist = marks_mod.histAt(hist0, prompt, at, row) }, at + c.rows == prompt.len);
        }
        calls.note(c.rows, c.parts, t_call, g_call, m.gpu_seconds);
        at += c.rows;
        dr.after(out, at, n_in);
    }
    out.prefilled(out.ctx);
    pick = watch.first(pick);
    const first_stop = out.tokens(out.ctx, &.{pick}) or isEos(eos, pick);
    try dr.flush(e, out); // after the first token: the slots, rows and tail stay put till the rounds
    if (e.call_log) calls.log(if (r.tp) |tp| tp.rank else 0);
    if (try e.agree(first_stop)) return .{ .reason = .stop };
    if (max_tokens <= 1) return .{ .reason = .length };
    return e.decode(prompt, pick, .{ .b = m.last.b, .off = m.last.off + (last_n - 1) * WIDE * 2 }, max_tokens, eos, depth, &watch, out, null);
}

pub const Handoff = struct { yielded: bool = false, cache: usize = 0 };

pub fn decode(e: *Engine, prompt: []const u32, pick: u32, streams: Buf, max_tokens: usize, eos: []const u32, depth: ?usize, watch: *think.Watch, out: Out, handoff: ?*Handoff) !Result {
    const r = e.r;
    const m = e.m;
    defer e.hostMode();
    var res: Result = .{ .reason = .length };
    var emitted: usize = 1;
    const ar = r.ar.b.slice(i32, fz.AR_WORDS);
    if (m.state == 1) { // the state into the rounds' state buffers
        ar[0] = 1;
        const cb = r.queue.commandBuffer();
        r.begin(cb, .serial);
        errdefer if (r.fsr != null) r.abort();
        e.statesCopy(true);
        try m.finish(cb);
        m.state = 0;
        m.state_row = 0;
    }
    var rule: DepthRule = .{ .pair = r.tp != null };
    var by_cost: ?CostDepth = if (r.tp == null and !r.device.tensorUnits()) .{} else null; // 0.6.5's rule before M5
    var last_end: f64 = 0; // the previous read round's GPU end (s): a round's time is the gap to its own
    const busy_log = platform.c.getenv("FZ_BUSY") != null; // dev: the GPU's busy share of a reply's rounds
    var busy_s: f64 = 0;
    var gap_s: f64 = 0;
    var first_start: f64 = 0;
    defer if (busy_log and last_end > first_start) std.log.info("rounds: GPU busy {d:.1} ms, gaps {d:.2} ms, span {d:.1} ms", .{ busy_s * 1e3, gap_s * 1e3, (last_end - first_start) * 1e3 });
    var copy_rate: f64 = 0.6; // drafts landed over offered: copied rounds, the head's rounds
    var head_rate: f64 = 0.6;
    const long = prompt.len + max_tokens + 8 > 4 * TOP;
    // tree rounds (FZ_TREE) for drafted greedy replies without a thinking budget, below the indexer's selection
    const drafted = if (depth) |d| d > 0 else true;
    const tr: ?*tree_mod.Tree = if (e.tree) |t| (if (drafted and out.sampling == null and !watch.armed and r.tp == null and !long) t else null) else null;
    r.tree = tr;
    defer r.tree = null;
    if (tr) |t| t.select(prompt);
    // gated lanes (FZ_GATE) for the same replies: the head's confidence sets each round's width on the GPU
    const gt: ?*gate_mod.Gate = if (e.gate) |g| (if (tr == null and drafted and out.sampling == null and !watch.armed and r.tp == null and !long) g else null) else null;
    const gfix: ?usize = if (platform.c.getenv("FZ_GATE_FIX")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch null else null; // dev: a fixed host window
    const fixed = if (tr) |t| t.a + t.b else if (gt != null) (gfix orelse gate_mod.WMAX_FIRST) - 1 else depth;
    const depth0 = fixed orelse if (by_cost) |*c| c.pick() else rule.pick();
    var next_depth: usize = depth0; // the rule's drafts for the next round to encode (rank 0's in speed-up mode)
    var gw: usize = gfix orelse gate_mod.WMAX_FIRST; // a gated reply's window, rows (up to the gate's width)
    const w = e.wids.b.slice(u32, 16);
    w[0] = pick;
    if (depth0 > 0) { // the head: the last prompt row with the first token, then its chain
        m.mtp.pos = prompt.len - 1;
        w[1] = try m.mtpRun(&.{pick}, streams);
        for (2..(if (tr) |t| t.a else depth0) + 1) |j| w[j] = try m.mtpChain(w[j - 1]);
        if (tr) |t| t.firstDrafts(w);
    }
    const W0 = depth0 + 1;
    const T: i32 = @intCast(m.pos);
    if (long) { // selection on the GPU: every complete block the target and the head hold, pooled
        const cb = r.queue.commandBuffer();
        r.begin(cb, .serial);
        errdefer if (r.fsr != null) r.abort();
        try r.sel.?.catchUp(r, m, m.pos / 4);
        try m.finish(cb);
        e.tsel.sel.b.slice(i32, 128)[fz.SEL_POOLED] = @intCast(m.pos / 4);
        e.msel.sel.b.slice(i32, 128)[fz.SEL_POOLED] = @intCast(m.pos / 4);
    }
    @memset(ar, 0);
    ar[1] = T;
    const hs = e.hist.slice(u32, e.m.cap + 64); // the history: the prompt and the first token
    @memcpy(hs[0..prompt.len], prompt);
    hs[prompt.len] = pick;
    ar[fz.AR_HLEN] = @intCast(prompt.len + 1);
    for (0..MAXR) |i| {
        const d: i32 = @intCast(if (tr != null and i < W0) tr.?.depth(i) else i); // a tree row's position is its depth
        ar[fz.AR_POS + i] = if (i < W0) T + d else 0;
        ar[fz.AR_NK + i] = if (i < W0) T + d + 1 else 0;
    }
    ar[fz.AR_KV], ar[fz.AR_KV + 1], ar[fz.AR_KV + 2] = .{ T, @intCast(e.m.cap), @intCast(W0) };
    watch.arm(ar, @intCast(T));
    const pm = m.t.ple_meta.b.slice(i64, 39);
    pm[0], pm[1], pm[2], pm[3] = .{ m.ple.hist[0], m.ple.hist[1], m.ple.eos, 0 };
    for (0..3) |k| pm[4 + k] = m.ple.mult[k];
    for (0..16) |k| {
        pm[7 + k] = m.ple.sizes[k];
        pm[23 + k] = m.ple.offsets[k];
    }
    e.gpuMode();
    if (gt) |g| { // each gated chain step reads its row counts from the gate
        g.open(W0);
        for (1..MAXR - 1) |j| {
            m.mtp.slots[j].rows = g.run(j);
            m.mtp.slots[j].md = g.md(j);
            m.mtp.slots[j].md4 = g.md4(j);
        }
    }
    if (long) {
        r.gsel = e.tsel;
        m.mtp.gsel = e.msel;
    }
    const base = r.event_value;
    var cbs: [4]?mtl.CommandBuffer = .{ null, null, null, null };
    var widths: [4]usize = undefined;
    widths[0] = W0;
    var round: usize = 0;
    var done: usize = 0;
    var w_sum: usize = 0;
    res.min_rows = @intCast(W0);
    const rg = e.ring.slice(u32, fz.RING_WORDS * RING);
    var failed: ?anyerror = null;
    var kv_len: usize = m.pos; // the cache's rows after the rounds landed so far
    while (true) {
        const wr = widths[round % 4];
        const cb = r.queue.commandBuffer();
        if (round > 0) cb.waitForEvent(r.event, base + round);
        r.begin(cb, .serial);
        errdefer if (r.fsr != null) r.abort();
        m.t.rows = if (gt) |g| g.rows() else e.rows_w[wr];
        m.t.mdims = e.mdims_w[wr];
        if (round > 0) {
            const wp = widths[(round - 1) % 4];
            if (tr != null) { // the kept path's last row: the verdict leaves path[keep - 1] + 1 past the arena's own words
                const ar0 = r.ar;
                r.ar = .{ .b = ar0.b, .off = ar0.off + tree_mod.AR_LAST * 4 };
                e.statesCopy(true);
                r.ar = ar0;
            } else e.statesCopy(r.gdn_kept == null); // kept states replay the kept rows themselves
            r.copyKept(e.cins[(round - 1) % 2], e.cins[round % 2], PLE_TAIL * WIDE / 2, WIDE / 2, 0, 0, 1, 0);
            if (wr > 1) {
                try m.mtpEncode(MAXR - 1 + wp, wp, m.t.picks, m.last, .{ .b = e.wids.b, .off = 4 });
                if (gt) |g| g.encode(r, m.mtp.logits, 0, wr);
                if (tr) |t| if (t.sib > 0) t.siblingsEncode(r, m.mtp.logits, m.mtp.n_ids, m.mtp.ids, e.wids, 1);
                for (1..if (tr) |t| t.a else wr - 1) |j| {
                    r.head_md1 = if (gt) |g| g.md1(j) else null; // a gated chain step runs only when the gate opened it
                    defer r.head_md1 = null;
                    try m.mtpEncode(j, 1, .{ .b = e.wids.b, .off = 4 * j }, m.mtp.h[1], .{ .b = e.wids.b, .off = 4 * (j + 1) });
                    if (gt) |g| g.encode(r, m.mtp.logits, @intCast(j), wr);
                    if (tr) |t| if (t.sib > 0) t.siblingsEncode(r, m.mtp.logits, m.mtp.n_ids, m.mtp.ids, e.wids, j + 1);
                }
                if (tr) |t| {
                    if (t.sib == 0) t.feedEncode(r, e.wids);
                } else if (e.copy) { // copied drafts replace the head's when the history repeats its last tokens
                    r.enc.setPipeline(r.lookup_pipe);
                    r.enc.setBuffer(e.hist, 0, 0);
                    r.enc.setBuffer(r.ar.b, r.ar.off, 1);
                    r.enc.setBuffer(e.wids.b, e.wids.off, 2);
                    const lenient = copy_rate + 0.05 >= head_rate;
                    const lc = [4]u32{ @intCast(wr), if (lenient) e.copy_min else 4, if (lenient) e.copy_long else 9, if (lenient) 1 else 2 };
                    r.enc.setBytes(std.mem.asBytes(&lc), 3);
                    const opens = if (gt) |g| g.rows() else e.rows_w[MAXR]; // a copy opens a gated window to its drafts
                    r.enc.setBuffer(opens.b, opens.off, 4);
                    const gated: u32 = @intFromBool(gt != null);
                    r.enc.setBytes(std.mem.asBytes(&gated), 5);
                    r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
                }
            }
        }
        if (watch.armed) think.force(r, e.wids, e.hist, wr); // the close's tokens as the window's drafts
        m.ple.cin = e.cins[round % 2];
        m.rec_slot = round % 2;
        if (tr) |t| t.pleIds(r, wr, e.wids, m.t.ple_meta, m.t.ple_ids) else m.pleIdsGpu(wr, e.wids);
        w_sum += wr;
        const ub = (@as(usize, @intCast(T)) + w_sum) / 4 + 2;
        if (r.gsel) |*g| g.nb_ub = ub;
        if (m.mtp.gsel) |*g| g.nb_ub = ub;
        r.limit = if (gt) |g| g.rows() else null;
        try m.windowEncode(wr, e.wids);
        r.limit = null;
        if (out.sampling) |p| { // the window's picks redrawn from its logits, in place of the head's argmax
            sample.positions(r.enc, r.positions_pipe, r.ar, e.draw_at, wr);
            if (!r.serial) r.enc.barrier();
            sample.draw(r.enc, r.draw_pipe, m.t.logits, e.draw_at, m.t.picks, wr, VOCAB, p);
            if (!r.serial) r.enc.barrier();
        }
        // the widest windows while copies land whole (a gated reply's too); otherwise the gate's or the head's rule
        const copying = e.copy and copy_rate + 0.05 >= head_rate and copy_rate > 0.9;
        const copy_d = if (by_cost) |*c| c.copyDrafts(copy_rate) else MAXR - 1;
        const wn = (if (tr != null) fixed.? else if (gt != null) (if (copying) copy_d else gw - 1) else if (fixed) |d| d else if (copying) copy_d else next_depth) + 1;
        widths[(round + 1) % 4] = wn;
        if (watch.armed) think.span(r, m.t.picks, wr); // and in place of the target's picks
        if (tr) |t| {
            t.acceptEncode(r, .{ e.wids, m.t.picks, r.ar, .{ .b = e.ring }, m.t.ple_meta }, e.hist, .{ @intCast(wr), @intCast(wn), @intCast(e.m.cap), @intCast(t.a) });
            t.compact(r, &m.layers, e.m.cap, m.last, m.t.picks, e.cins[round % 2]);
        } else {
            const cfg = [4]u32{ @intCast(wr), @intCast(wn), @intCast(e.m.cap), 0 };
            r.enc.setPipeline(r.accept_pipe);
            for ([_]Buf{ e.wids, m.t.picks, r.ar, .{ .b = e.ring }, m.t.ple_meta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBytes(std.mem.asBytes(&cfg), 5);
            r.enc.setBuffer(e.hist, 0, 6);
            const verified = if (gt) |g| g.rows() else e.rows_w[MAXR]; // the rows the verdict reads
            r.enc.setBuffer(verified.b, verified.off, 7);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        }
        const final = try r.finishEncoding(cb);
        final.signalEvent(r.event, base + round + 1);
        final.commit();
        cbs[round % 4] = final;
        round += 1;
        if (round < 2) continue;
        const prev = cbs[done % 4].?;
        prev.wait();
        cbs[done % 4] = null;
        if (prev.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            failed = error.GpuFailed;
            break;
        }
        if (r.tp) |tp| if (tp.failed.load(.acquire)) { // the link failed: the host drained the round, its tokens are wrong
            failed = error.TpLinkFailed;
            break;
        };
        const slot = (done % RING) * fz.RING_WORDS;
        const keep = rg[slot];
        const wd: usize = if (gt != null) rg[slot + 19] else widths[done % 4]; // a gated round's width is the GPU's
        if (by_cost) |*c| if (last_end > 0) c.time(wd - 1, (prev.gpuEnd() - last_end) * 1e3);
        if (busy_log) { // dev: GPU time inside rounds against gaps between them
            if (first_start == 0) first_start = prev.gpuStart();
            const from = if (last_end > 0) @max(prev.gpuStart(), last_end) else prev.gpuStart();
            busy_s += prev.gpuEnd() - from;
            if (last_end > 0 and prev.gpuStart() > last_end) gap_s += prev.gpuStart() - last_end;
        }
        last_end = prev.gpuEnd();
        const got = rg[slot + 3 .. slot + 3 + keep];
        const learn = !watch.read(got); // a round that landed the close's tokens teaches no rate, rule or counter
        if (learn and wd > 1) {
            const frac = @as(f64, @floatFromInt(keep - 1)) / @as(f64, @floatFromInt(wd - 1));
            if (rg[slot + 1] != 0) copy_rate = 0.8 * copy_rate + 0.2 * frac else head_rate = 0.8 * head_rate + 0.2 * frac;
            if (rg[slot + 1] == 0 and wd == MAXR) copy_rate *= 0.8; // a full window that found no copy: copies stopped landing
        }
        if (learn and rg[slot + 1] != 0) {
            res.copy_rounds += 1;
            res.copy_accepted += keep - 1;
            const n = @min(rg[slot + 2], 8);
            res.copy_by_len[n][0] += 1;
            res.copy_by_len[n][1] += keep - 1;
        }
        if (gt) |g| if (learn and gfix == null and rg[slot + 1] == 0) { // a row wider after the gate kept the window whole, one narrower after two short
            const wh = widths[done % 4];
            if (wd == wh) gw = @min(gw + 1, g.width) else if (wd + 2 <= wh) gw = @max(gw - 1, gate_mod.WMAX_FIRST);
        };
        if (learn and fixed == null and wd > 1 and rg[slot + 1] == 0) rule.update(wd - 1, keep - 1);
        if (learn and wd > 1 and rg[slot + 1] == 0) if (by_cost) |*c| c.observe(wd - 1, keep - 1);
        res.rounds += 1;
        if (learn) res.drafted += wd - 1;
        if (learn) res.accepted += keep - 1;
        res.min_rows = @min(res.min_rows, @as(u32, @intCast(wd)));
        done += 1;
        kv_len += keep;
        var take: usize = 0;
        var stop = false;
        while (take < got.len and emitted + take < max_tokens) {
            take += 1;
            if (isEos(eos, got[take - 1])) {
                stop = true;
                break;
            }
        }
        if (take > 0 and out.tokens(out.ctx, got[0..take])) stop = true;
        emitted += take;
        const cancel = !stop and emitted < max_tokens and out.cancelled(out.ctx);
        const ag = e.agreeRound(
            stop or emitted >= max_tokens or cancel,
            if (by_cost) |*c| c.pick() else rule.pick(),
        ) catch |err| {
            failed = err;
            break;
        };
        next_depth = ag.depth;
        const quit = ag.quit;
        if (quit) {
            res.reason = if (stop) .stop else if (emitted >= max_tokens) .length else .cancelled;
            break;
        }
    }
    for (&cbs) |*c| if (c.*) |cb| {
        cb.wait();
        c.* = null;
    };
    r.event_value = base + round + 1;
    if (failed) |err| return err;
    if (handoff) |h| if (res.reason == .cancelled) try e.handOver(h, round, kv_len);
    return res;
}

pub fn handOver(e: *Engine, h: *Handoff, round: usize, cache: usize) !void {
    if (e.r.gdn_kept != null) return error.HandoffUnsupported; // kept states replay their rows in the next window
    if ((round - 1) % 2 == 1) {
        const n = PLE_TAIL * WIDE * 2;
        @memcpy(e.cins[0].b.contents()[e.cins[0].off..][0..n], e.cins[1].b.contents()[e.cins[1].off..][0..n]);
    }
    h.* = .{ .yielded = true, .cache = cache };
}

pub fn agree(e: *Engine, quit: bool) !bool {
    return if (e.r.tp) |tp| (try tp.agree(quit, 0)).quit else quit;
}

pub fn agreeRound(e: *Engine, quit: bool, depth: usize) !struct { quit: bool, depth: usize } {
    const tp = e.r.tp orelse return .{ .quit = quit, .depth = depth };
    const a = try tp.agree(quit, @intCast(depth));
    return .{ .quit = a.quit, .depth = a.depth };
}
