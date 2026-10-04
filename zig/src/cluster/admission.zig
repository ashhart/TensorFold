//! Admission for converged rounds: the streams and context that fit, rows a round, and prompt chunks riding decode rounds.
const std = @import("std");
const model = @import("model.zig");
const plan_mod = @import("plan.zig");
const budget = @import("budget.zig");
const cost = @import("cost.zig");
const traffic = @import("traffic.zig");
const round = @import("round.zig");

const Plan = plan_mod.Plan;

/// Every rank's need with caches for `streams` x `context` in place of the plan's, against its GPU limit.
pub fn fitsAt(p: *const Plan, ns: []const budget.Need, s: *const model.Shape, streams: u32, context: u64) bool {
    var q = p.*;
    q.opts.streams = streams;
    q.opts.context = context;
    for (ns, 0..) |n, r| {
        const held = budget.caches(&q, s, @intCast(r));
        const need = n.weightBytes() + n.drafter + budget.buffers(&q, s, @intCast(r)) + held.kv + held.state + p.opts.margin;
        if (need > p.nodes[r].gpu_limit) return false;
    }
    return true;
}

/// The most streams whose caches fit every rank at `context` (0 when even one does not).
pub fn maxStreams(p: *const Plan, ns: []const budget.Need, s: *const model.Shape, context: u64) u32 {
    var lo: u32 = 0;
    var hi: u32 = 1 << 16;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (fitsAt(p, ns, s, mid, context)) lo = mid else hi = mid - 1;
    }
    return lo;
}

/// The longest context every one of `streams` streams can hold.
pub fn maxContext(p: *const Plan, ns: []const budget.Need, s: *const model.Shape, streams: u32) u64 {
    var lo: u64 = 0;
    var hi: u64 = @max(s.max_context, 1);
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (fitsAt(p, ns, s, streams, mid)) lo = mid else hi = mid - 1;
    }
    return lo;
}

/// A plan to choose among: its placements and every rank's need.
pub const Candidate = struct { p: *const Plan, ns: []const budget.Need };

/// The candidate that fits its planned streams and context with the fastest round at that many rows (else the first).
pub fn best(cands: []const Candidate, s: *const model.Shape, link: traffic.Link) usize {
    var pick: ?usize = null;
    var pick_ms: f64 = std.math.inf(f64);
    for (cands, 0..) |c, i| {
        const o = c.p.opts;
        if (!fitsAt(c.p, c.ns, s, o.streams, o.context)) continue;
        const ms = cost.round(c.p, c.ns, s, .{ .rows = o.streams, .streams = o.streams, .context = o.context }, link, .{}).total_ms;
        if (ms >= pick_ms) continue;
        pick = i;
        pick_ms = ms;
    }
    return pick orelse 0;
}

pub const Limits = struct {
    /// Rows one round may carry: decode windows plus prompt rows.
    rows: u32,
    /// Prompt rows a round takes while streams decode.
    chunk: u32,
};

/// Prompt rows that keep a round within (1 + share) of its decode-only time, in steps of 16, at least 16.
pub fn chunkRows(p: *const Plan, ns: []const budget.Need, s: *const model.Shape, decode: u32, context: u64, share: f64, link: traffic.Link) u32 {
    const base = cost.round(p, ns, s, .{ .rows = decode, .streams = decode, .context = context }, link, .{}).total_ms;
    var chunk: u32 = 16;
    while (chunk < 4096) : (chunk += 16) {
        const x: cost.Round = .{ .rows = decode, .streams = decode + 1, .context = context, .prompt = chunk + 16 };
        if (cost.round(p, ns, s, x, link, .{}).total_ms > base * (1 + share)) break;
    }
    return chunk;
}

/// A prompt being filled into its slot's caches a chunk at a time.
pub const Pending = struct {
    slot: u32,
    tokens: []const u32,
    done: u32 = 0,

    pub fn left(p: Pending) u32 {
        return @intCast(p.tokens.len - p.done);
    }
};

/// One round's rows: every decode row first, then prompt rows oldest prompt first, within the limits.
pub fn compose(a: std.mem.Allocator, limits: Limits, decode: []const round.Row, pending: []Pending) ![]round.Row {
    var out: std.ArrayList(round.Row) = .empty;
    try out.appendSlice(a, decode);
    var room = @min(limits.chunk, limits.rows -| @as(u32, @intCast(decode.len)));
    for (pending) |*p| {
        while (room > 0 and p.left() > 0) : (room -= 1) {
            const at = p.done;
            try out.append(a, .{ .slot = p.slot, .token = p.tokens[at], .index = at, .sample = at + 1 == p.tokens.len });
            p.done += 1;
        }
        if (room == 0) break;
    }
    return out.items;
}

test {
    _ = @import("admission_test.zig");
}
