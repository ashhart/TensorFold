//! The GLM backend's draft round: the MTP head, DFlash2 or a copy, chosen per stream by measured yield.
const std = @import("std");
const decode = @import("decode.zig");
const copy = @import("copy.zig");
const dflash = @import("dflash.zig");
const be = lanes.backend;
const Writer = proto.Writer;
const lanes = @import("lanes");
const proto = @import("proto.zig");
const native = @import("native.zig");
const max_rows = native.max_rows;
const max_depth = native.max_depth;
const max_streams = native.max_streams;
const Lane = native.Lane;
const Owned = native.Owned;
const settle = native.settle;
const of = native.of;
const laneOf = native.laneOf;
const readFn = native.readFn;
const msSince = native.msSince;

pub fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const self = of(p);
    if (requests.len > max_streams) return error.TooManyStreams;
    var ls: [max_streams]*Lane = undefined;
    var kept: [max_streams]u32 = undefined;
    for (requests, 0..) |r, i| {
        if (r.lanes != null) return error.TreesNotBuilt;
        ls[i] = try laneOf(self, r.stream);
        // the kept rows (a lone round has kept already and settles nothing)
        kept[i] = if (r.rows) |rows| @intCast(rows.len) else ls[i].rows;
    }
    try settle(self, ls[0..requests.len], kept[0..requests.len]);
    var follow: [max_streams][max_rows]u32 = undefined;
    var reqs: [max_streams]decode.DraftReq = undefined;
    var depths: [max_streams]u32 = undefined;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(requests.len);
    for (requests, 0..) |r, i| {
        const l = ls[i];
        var n: usize = 1;
        var first = false;
        if (r.rows) |rows| {
            for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
            n = rows.len;
            @memcpy(follow[i][0..n], r.follow[0..n]);
        } else {
            follow[i][0] = switch (r.first orelse return error.NoFirstToken) {
                .handle => |h| try readFn(p, h),
                .value => |v| v,
            };
            first = true;
        }
        const depth: u32 = @min(r.depth, max_depth);
        depths[i] = depth;
        if (self.e.drafter != null) chooseDrafter(self, l, if (r.rows != null) n else 0);
        const df = self.e.drafter != null and l.use_df;
        // a stream DFlash2 drafts for still feeds the MTP head its rows (one step, its draft unused): it stays current
        reqs[i] = .{ .s = l.s, .hidden = if (first) l.s.last_hidden else l.s.hidden, .next = follow[i][0..n], .count = if (df) 1 else depth, .confidence = self.confidence, .out = l.held[0..depth], .probs = l.probs[0..depth] };
        try w.int(l.s.uid);
        try w.tokens(follow[i][0..n]);
        try w.int(depth);
        try w.int(@intFromBool(first)); // 1: the first draft after the prompt (its hidden row: the prompt's last)
        try w.int(@as(u64, @bitCast(self.confidence))); // the confidence cut (rank 1 must stop where rank 0 does: the head's steps gather)
        try w.int(@intFromBool(df)); // 1: DFlash2 drafts this stream's round
    }
    const t0 = std.Io.Timestamp.now(self.io, .awake);
    try self.send(.draft, &w);
    // the MTP head: every stream's rows, drafts for the streams it drafts for (none if DFlash2 alone: TF_GLM_MIX=0)
    if (feedsMtp(self)) try decode.draft(self.e, reqs[0..requests.len]);
    if (self.e.drafter) |dr| {
        var dq: [max_streams]dflash.Req = undefined;
        for (requests, 0..) |r, i| {
            const nf = reqs[i].next.len;
            dq[i] = .{ .s = ls[i].s, .n = if (r.rows != null) nf else 0, .anchor = follow[i][nf - 1], .out = ls[i].held[0..max_depth], .probs = ls[i].probs[0..max_depth], .drafts = ls[i].use_df };
        }
        try dflash.draft(dr, self.e, dq[0..requests.len]);
        for (reqs[0..requests.len], ls[0..requests.len], depths[0..requests.len]) |*q, l, depth| { // the confidence cut, as the head's chains stop
            if (!l.use_df) continue;
            var chain: f64 = 1;
            q.got = 0;
            while (q.got < @min(depth, dflash.BLOCK - 1)) : (q.got += 1) { // a block's 7 (a copy may run longer)
                chain *= l.probs[q.got];
                if (q.got > 0 and (if (self.marginal > 0) l.probs[q.got] < self.marginal else chain < q.confidence)) break;
            }
        }
    }
    if (self.trace.on) {
        self.trace.draft_ms += msSince(self.io, t0);
        self.trace.calls += 1;
        for (reqs[0..requests.len]) |q| self.trace.drafts += q.got;
        self.trace.drafts_calls_total += requests.len;
    }
    for (requests, 0..) |r, i| {
        const l = ls[i];
        const depth: u32 = @min(r.depth, max_depth);
        const got = reqs[i].got;
        for (got..depth) |j| { // past the confidence cut: held with chance 0 (the allocator leaves them out)
            l.held[j] = if (got > 0) l.held[got - 1] else follow[i][reqs[i].next.len - 1];
            l.probs[j] = 0;
        }
        l.nheld = depth;
        if (self.trace.on and r.rows != null and l.trace_held > 0) self.trace.landed += @min(l.trace_held, reqs[i].next.len - 1);
        l.trace_held = 0;
        if (l.copy) |*h| try copyDrafts(self, l, h, follow[i][0..reqs[i].next.len], r.rows != null, depth, got);
    }
}

/// Whether a draft round runs the MTP head: always without DFlash2, else while the mix may pick it (TF_GLM_MIX).
pub fn feedsMtp(self: *const Owned) bool {
    return self.e.drafter == null or (self.mix_on and self.e.w.mtp != null);
}

/// This stream's draft source: the higher tokens per round time, the other every 8th round; ``kept``: last commit.
pub fn chooseDrafter(self: *Owned, l: *Lane, kept: usize) void {
    const now = std.Io.Timestamp.now(self.io, .awake);
    if (!self.mix_on or self.e.w.mtp == null) {
        l.use_df = true;
        return;
    }
    if (kept > 0) if (l.mix_t) |t| {
        const d: usize = @intFromBool(l.use_df);
        const ms = @as(f64, @floatFromInt(t.durationTo(now).toNanoseconds())) / 1e6;
        l.mix_tok[d] = 0.85 * l.mix_tok[d] + 0.15 * @as(f64, @floatFromInt(kept));
        l.mix_ms[d] = 0.85 * l.mix_ms[d] + 0.15 * ms;
    };
    l.mix_t = now;
    l.mix_round += 1;
    const df_better = l.mix_tok[1] / l.mix_ms[1] >= l.mix_tok[0] / l.mix_ms[0];
    l.use_df = if (l.mix_round % 8 == 0) !df_better else df_better;
    if (self.trace.on) self.trace.df_rounds += @intFromBool(l.use_df);
}

/// A copy over the head's drafts when it expects more landed tokens; ``follow``: just committed; ``kept_rows``.
pub fn copyDrafts(self: *Owned, l: *Lane, h: *copy.Copy, follow: []const u32, kept_rows: bool, depth: u32, got: usize) !void {
    const landed: u32 = @intCast(follow.len - 1);
    if (kept_rows and l.copy_verified > 0) self.copy_rate.record(l.copy_verified, landed);
    if (kept_rows and l.draft_verified > 0) self.draft_rate.record(l.draft_verified, landed);
    l.copy_verified = 0;
    l.draft_verified = 0;
    l.copied = 0;
    try h.append(self.gpa, follow);
    if (depth == 0) return;
    var buf: [max_depth]u32 = undefined;
    const n = h.propose(buf[0..depth]);
    if (n == 0) return;
    const q = self.copy_rate.chance();
    // the copy agrees with every draft and runs on: the drafts stay, the copy's tail follows them
    var same: usize = 0;
    while (same < @min(n, got) and buf[same] == l.held[same]) same += 1;
    const extends = same == got and n > got;
    if (!extends) { // else the source whose drafts are expected to land more, by their measured rates
        const mine = expected(q, n);
        const theirs = expected(self.draft_rate.chance(), got);
        if (mine <= theirs) return;
    }
    const keep = if (extends) got else 0;
    @memcpy(l.held[keep..n], buf[keep..n]);
    for (l.probs[keep..n]) |*pr| pr.* = q;
    for (n..depth) |j| {
        l.held[j] = buf[n - 1];
        l.probs[j] = 0;
    }
    l.copied = @intCast(n);
    if (self.trace.on) {
        self.trace.copies += 1;
        self.trace.copy_tokens += n;
    }
}

/// Tokens a chain of ``n`` drafts lands on average when each lands with chance ``p`` given the ones before.
pub fn expected(p: f64, n: usize) f64 {
    var sum: f64 = 0;
    var chain: f64 = 1;
    for (0..n) |_| {
        chain *= p;
        sum += chain;
    }
    return sum;
}
