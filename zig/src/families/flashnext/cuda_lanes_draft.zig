//! A shared round's streams drafted in one head pass a step (cuda_lanes.zig's draftFn), and the depth log.
const std = @import("std");
const lanes = @import("lanes");
const mtp = @import("cuda_mtp.zig");
const mtp_ops = @import("cuda_mtp_ops.zig");
const be = lanes.backend;

/// TF_FLASHNEXT_BATCH_HEADS=0 drafts a shared round's streams one at a time (the batched head gives the same drafts).
pub fn batchHeads() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_BATCH_HEADS", true) catch true;
}

/// TF_FLASHNEXT_SAMPLED_HEADS=0 keeps sampled streams (temperature above zero) out of a shared round's batched head.
pub fn sampledHeads() bool {
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_SAMPLED_HEADS", true) catch true;
}

/// TF_FLASHNEXT_DEPTH_LOG=1: every 200 draft calls, the drafts a stream got by round width (streams drafting together).
var depth_hist: [5][max_hist]u64 = @splat(@splat(0));
var depth_calls: u64 = 0;
const max_hist = 17;

/// A round's greedy chains: depth (main's 6 past `wideStreams()` drafting) and stop (0.6 beside another open stream).
pub const Rule = struct { greedy: usize, greedy_stop: f64 };
pub const shared_stop: f64 = 0.6;

pub fn rule(self: anytype, width: usize) Rule {
    const greedy = if (width <= wideStreams()) self.depth_greedy else @min(self.depth_greedy, mtp.default_depth);
    const stop = if (sharedRule() and busy(&self.lanes)) shared_stop else self.confidence;
    return .{ .greedy = greedy, .greedy_stop = stop };
}

/// A chain's stop rule: a greedy stream's from the round, a sampled stream's always the engine's.
pub fn stopFor(self: anytype, r: Rule, sm: ?mtp.Sampling) f64 {
    return if (mtp.isSampled(sm)) self.confidence else r.greedy_stop;
}

/// TF_FLASHNEXT_GREEDY_WIDE: the round width (streams drafting together) up to which greedy chains keep their depth.
pub const default_wide: usize = 4;
fn wideStreams() usize {
    return @import("sys").environment.read(usize, "TF_FLASHNEXT_GREEDY_WIDE", default_wide) catch default_wide;
}

/// TF_FLASHNEXT_SHARED_DRAFTS=0: greedy chains keep the engine's stop rule beside other streams too.
fn sharedRule() bool {
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_SHARED_DRAFTS", true) catch true;
}

/// More than one stream in `lanes` (keyed by stream) is neither finished nor paused: decoding, filling or resuming.
pub fn busy(lanes_map: anytype) bool {
    var open: usize = 0;
    var it = lanes_map.keyIterator();
    while (it.next()) |st| {
        if (st.*.finished or st.*.paused) continue;
        open += 1;
        if (open > 1) return true;
    }
    return false;
}

/// Whether a plain stream skips the head (EXL3 with offloaded experts: its pass costs misses); see cuda_mtp_stash.zig.
fn deferring(self: anytype) bool {
    return @import("exl3/model/cuda_exl3_pass.zig").offloaded(self.engine);
}

/// The window's kept rows: an open window is rolled back to them; none open (a plain round) leaves nothing to do.
pub fn rewind(self: anytype, seq: anytype, d: *mtp.Draft, keep: usize) !void {
    if (d.open) try self.ops().rollback(seq, d, keep);
}

/// Whether a drafting lane's one-row window runs as a plain decode round (the stream's guard left its hidden rows out).
pub fn plainRound(self: anytype, l: anytype, w: be.Window) !bool {
    const d = if (l.draft) |*x| x else return false;
    if (w.hidden or w.rows() != 1 or w.stream.logprobs != null or !deferring(self) or d.media != null) {
        d.fresh = true;
        return false;
    }
    try preparePlain(self.ops(), &l.seq, d);
    return true;
}

fn preparePlain(ops: anytype, seq: anytype, d: *mtp.Draft) !void {
    try ops.flushLate(seq, d);
    d.fresh = false;
    d.verify_cache.invalidate();
}

/// A request for no drafts that the head skips: the window's kept rows are stashed, not run. False: draft as usual.
pub fn deferred(self: anytype, r: be.DraftRequest) !bool {
    const path = r.rows orelse return false;
    if (!deferring(self) or r.lanes != null) return false;
    const l = try self.lane(r.stream);
    const d = if (l.draft) |*x| x else return false;
    if (d.media != null) return false;
    if (!d.fresh) {
        // the round left the hidden rows out: nothing to absorb or draft from; the next window taps them
        l.chain.n = 0;
        return true;
    }
    for (path, 0..) |row, i| if (row != i) return false;
    if (r.follow.len < path.len) return error.PositionMismatch;
    if (@min(@min(r.depth, self.depth), l.seq.cap - l.seq.fed - 1) != 0) return false;
    try rewind(self, &l.seq, d, path.len);
    if (r.position != l.seq.fed + 1) return error.PositionMismatch;
    if (!try self.ops().stash(d, r.follow[0..path.len], d.streams.ptr)) return false;
    l.chain.n = 0;
    return true;
}

/// Requests without trees or media drafted together: draftFn's rollbacks, one head pass a step; `done` marks them.
pub fn batched(self: anytype, requests: []const be.DraftRequest, done: []bool) !void {
    const rl = rule(self, requests.len);
    var ds: [mtp_ops.max_lanes]*mtp.Draft = undefined;
    var nexts: [mtp_ops.max_lanes][]const u32 = undefined;
    var firsts: [mtp_ops.max_lanes][1]u32 = undefined;
    var rows: [mtp_ops.max_lanes]u64 = undefined;
    var counts: [mtp_ops.max_lanes]usize = undefined;
    var positions: [mtp_ops.max_lanes]u64 = undefined;
    var sms: [mtp_ops.max_lanes]?mtp.Sampling = undefined;
    var stops: [mtp_ops.max_lanes]f64 = undefined;
    var outs: [mtp_ops.max_lanes]*mtp.Chain = undefined;
    var which: [mtp_ops.max_lanes]usize = undefined;
    var n: usize = 0;
    for (requests, 0..) |r, ri| {
        if (r.lanes != null) continue;
        if (mtp.isSampled(r.stream.sampling) and !sampledHeads()) continue;
        if (try deferred(self, r)) {
            done[ri] = true;
            continue;
        }
        const l = try self.lane(r.stream);
        const d = if (l.draft) |*x| x else continue;
        if (d.media != null) continue;
        if (r.rows) |path| {
            for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            if (r.follow.len < path.len) return error.PositionMismatch;
        }
        if (r.rows) |path| try rewind(self, &l.seq, d, path.len) else {
            if (d.open) try self.ops().rollback(&l.seq, d, d.rows);
        }
        if (r.position != l.seq.fed + 1) return error.PositionMismatch;
        const room = l.seq.cap - l.seq.fed - 1;
        counts[n] = @min(@min(r.depth, mtp.depthFor(r.stream.sampling, self.depth, rl.greedy)), room);
        if (r.rows) |path| {
            nexts[n] = r.follow[0..path.len];
            rows[n] = d.streams.ptr;
        } else {
            firsts[n] = .{self.value(r.first orelse return error.NoFirstToken)};
            nexts[n] = &firsts[n];
            rows[n] = d.prompt_last;
        }
        ds[n] = d;
        positions[n] = r.position;
        sms[n] = r.stream.sampling;
        stops[n] = stopFor(self, rl, r.stream.sampling);
        outs[n] = &l.chain;
        which[n] = ri;
        n += 1;
    }
    if (n < 2) {
        // one stream: it drafts alone below (its rollback is idempotent)
        return;
    }
    try self.ops().draftMany(ds[0..n], nexts[0..n], rows[0..n], counts[0..n], positions[0..n], sms[0..n], stops[0..n], outs[0..n]);
    for (which[0..n]) |ri| done[ri] = true;
}

/// TF_FLASHNEXT_DEPTH_LOG=1: each stream's chain length by how many streams drafted together.
pub fn depthLog(self: anytype, requests: []const be.DraftRequest) void {
    const v = std.c.getenv("TF_FLASHNEXT_DEPTH_LOG") orelse return;
    if (!std.mem.eql(u8, std.mem.span(v), "1")) return;
    const w = @min(requests.len, 5) - 1;
    for (requests) |r| {
        const l = self.lane(r.stream) catch continue;
        depth_hist[w][@min(l.chain.n, max_hist - 1)] += 1;
    }
    depth_calls += 1;
    if (depth_calls % 200 != 0) return;
    for (depth_hist, 1..) |row, width| {
        var any: u64 = 0;
        for (row) |c| any += c;
        if (any == 0) continue;
        std.debug.print("DEPTH width={d}{s}", .{ width, if (width == 5) "+" else "" });
        for (row, 0..) |c, d| if (c > 0) std.debug.print(" {d}:{d}", .{ d, c });
        std.debug.print("\n", .{});
    }
}

test {
    _ = @import("cuda_lanes_draft_test.zig");
}

test "exl3: plain guard flushes pending rows before changing draft state" {
    const Fake = struct {
        fail: bool = false,
        fn flushLate(self: *@This(), _: *u8, d: *mtp.Draft) !void {
            if (self.fail) return error.FlushFailed;
            d.pending = 0;
        }
    };
    var fake: Fake = .{ .fail = true };
    var seq: u8 = 0;
    var draft: mtp.Draft = undefined;
    draft.pending = 5;
    draft.fresh = true;
    draft.verify_cache = .{};
    draft.verify_cache.publish(draft.verify_cache.generation, 1, 0);
    try std.testing.expectError(error.FlushFailed, preparePlain(&fake, &seq, &draft));
    try std.testing.expect(draft.fresh and draft.verify_cache.ready != null);
    fake.fail = false;
    try preparePlain(&fake, &seq, &draft);
    try std.testing.expect(!draft.fresh and draft.verify_cache.ready == null);
}
