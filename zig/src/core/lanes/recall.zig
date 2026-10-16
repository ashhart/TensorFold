//! Recall: drafts copied from a record of what a server read and wrote (not the prompt cache's learned states).
const std = @import("std");
const Allocator = std.mem.Allocator;
const fill = @import("fill.zig");
pub const record = @import("recall_record.zig");
pub const Record = record.Record;

/// The record and what its copies landed, shared by every stream; `lock` guards both.
pub const Shared = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    record: Record,
    rates: Rates = .{},

    pub fn lock(s: *Shared) void {
        s.mutex.lockUncancelable(s.io);
    }

    pub fn unlock(s: *Shared) void {
        s.mutex.unlock(s.io);
    }
};

/// Match lengths that open each band: a band's copies land at their own learned rate.
pub const bands = [_]u32{ 16, 32, 64 };
/// The longest wait between probes of a band whose copies lose.
const probe_most: u32 = 1024;

/// Every stream's Recall copies by band: what landed, so a cold or noisy record stops proposing by itself.
pub const Rates = struct {
    landed: [bands.len]f64 = .{ 0.75, 2, 4.5 }, // priors: 0.6, 0.8 and 0.9 a copied id, worth half a miss each
    missed: [bands.len]f64 = .{ 0.5, 0.5, 0.5 },
    probe_every: u32 = 32, // unpriced lookups between probes
    since: u32 = 0,
    proposals: u64 = 0,
    proposed: u64 = 0,
    accepted: u64 = 0,

    pub fn band(len: u32) usize {
        var b: usize = 0;
        while (b + 1 < bands.len and len >= bands[b + 1]) b += 1;
        return b;
    }

    /// A band's chance that a copied id lands when the ids before it did.
    pub fn rate(r: *const Rates, b: usize) f64 {
        return r.landed[b] / (r.landed[b] + r.missed[b]);
    }

    /// A copy of `proposed` ids landed `accepted`; a probe that landed half or more ends the backoff, else doubles it.
    pub fn observe(r: *Rates, b: usize, proposed: u64, accepted: u64, probing: bool) void {
        r.landed[b] = 0.95 * r.landed[b] + @as(f64, @floatFromInt(accepted));
        r.missed[b] = 0.95 * r.missed[b] + @as(f64, @floatFromInt(@intFromBool(accepted < proposed)));
        r.proposals += 1;
        r.proposed += proposed;
        r.accepted += accepted;
        if (probing) r.probe_every = if (2 * accepted >= proposed) 32 else @min(probe_most, 2 * r.probe_every);
    }

    /// Whether some band's copy of `n` ids is expected to land more than `head` tokens a round.
    pub fn pays(r: *const Rates, n: usize, head: f64) bool {
        for (0..bands.len) |b| if (expected(r.rate(b), n) > head) return true;
        return false;
    }

    /// Whether this ask, priced out, is the next probe.
    fn probe(r: *Rates) bool {
        r.since += 1;
        if (r.since < r.probe_every) return false;
        r.since = 0;
        return true;
    }
};

/// Tokens a round with a copy of `n` ids is expected to land at `q` an id, its own token included.
pub fn expected(q: f64, n: usize) f64 {
    var total: f64 = 1;
    var reach: f64 = 1;
    for (0..n) |_| {
        reach *= q;
        total += reach;
    }
    return total;
}

pub const Params = struct {
    min_match: u32 = bands[0], // matched ids a Recall copy or graft needs
    quiet: u64 = 4, // rounds a stream leaves Recall after a copy that landed nothing
};

/// A stream's reads of the record: copies priced against its own rounds, and candidates for its grafts.
pub const Reader = struct {
    gpa: Allocator,
    shared: *Shared,
    params: Params = .{},
    out: std.ArrayList(u32) = .empty,
    head: f64 = 2, // tokens the stream's own rounds landed lately: the bar a copy's expected tokens must pass
    band: usize = 0, // the last copy's band
    probing: bool = false, // the last copy was a probe
    quiet_until: u64 = 0,
    memo: ?u64 = null, // the last ask's context and width, hashed: the same ask again in its round gets the same answer
    lookups: u64 = 0,
    proposals: u64 = 0,
    proposed: u64 = 0,
    accepted: u64 = 0,

    pub fn deinit(x: *Reader) void {
        x.out.deinit(x.gpa);
    }

    /// The record's continuation of `ctx` (owned until the next call), or empty when it is not expected to pay.
    pub fn propose(x: *Reader, ctx: []const u32, most: usize, round: u64) []const u32 {
        if (ctx.len == 0 or most < 2 or round < x.quiet_until) return &.{};
        const key = memoKey(ctx, most);
        if (x.memo) |m| if (m == key) return x.out.items;
        x.out.clearRetainingCapacity();
        x.out.ensureTotalCapacity(x.gpa, most) catch return &.{};
        x.memo = key;
        x.shared.lock();
        defer x.shared.unlock();
        const rates = &x.shared.rates;
        var probing = false; // the price comes first: the record is walked only when a copy can pay, or for a probe
        if (!rates.pays(most, x.head)) {
            probing = rates.probe();
            if (!probing) return &.{};
        }
        x.lookups += 1;
        const m = x.shared.record.longest(ctx);
        if (m.len < x.params.min_match) return &.{};
        const got = x.shared.record.copy(m.at, x.out.allocatedSlice()[0..most]);
        if (got.len < 2) return &.{};
        const b = Rates.band(m.len);
        if (!probing and expected(rates.rate(b), got.len) <= x.head) {
            probing = rates.probe();
            if (!probing) return &.{};
        }
        x.out.items.len = got.len;
        x.band = b;
        x.probing = probing;
        x.proposals += 1;
        x.proposed += got.len;
        return x.out.items;
    }

    /// What a lookup reads of `ctx` (its last max_match ids) and the width asked, hashed.
    fn memoKey(ctx: []const u32, most: usize) u64 {
        return std.hash.Wyhash.hash(most, std.mem.sliceAsBytes(ctx[ctx.len -| record.max_match..]));
    }

    /// The last copy landed `accepted` of its `proposed` ids; none landed keeps the stream off a few rounds.
    pub fn observe(x: *Reader, proposed: i64, accepted: i64, round: u64) void {
        x.memo = null;
        if (proposed <= 0) return;
        const p: u64 = @intCast(proposed);
        const a: u64 = @intCast(std.math.clamp(accepted, 0, proposed));
        x.accepted += a;
        if (a == 0) x.quiet_until = round + 1 + x.params.quiet;
        x.shared.lock();
        defer x.shared.unlock();
        x.shared.rates.observe(x.band, p, a, x.probing);
    }

    /// A round of the stream's own drafts (a head, a tree or none) landed `tokens`.
    pub fn noteHead(x: *Reader, tokens: usize) void {
        x.memo = null;
        x.head += 0.25 * (@as(f64, @floatFromInt(tokens)) - x.head);
    }

    /// Up to `most` distinct continuations of `ctx` as graft proposals of `span` ids, weighed by match up to `longest`.
    pub fn candidates(
        x: *Reader,
        a: Allocator,
        ctx: []const u32,
        span: usize,
        most: usize,
        longest: u32,
    ) ![]fill.Proposal {
        if (span == 0 or most == 0) return &.{};
        var found: [8]record.Found = undefined;
        const scratch = try a.alloc(u32, span);
        x.shared.lock();
        defer x.shared.unlock();
        x.lookups += 1;
        const room = found[0..@min(most, found.len)];
        const n = x.shared.record.continuations(ctx, x.params.min_match, span, scratch, room);
        const out = try a.alloc(fill.Proposal, n);
        for (out, found[0..n]) |*o, f| o.* = .{
            .tokens = x.shared.record.copy(f.at, try a.alloc(u32, span)),
            .weight = @floatFromInt(@min(f.len, longest)),
            .recalled = true,
        };
        return out;
    }

    /// The stream's counters as the reply's telemetry: `{"recall": {...}}`.
    pub fn telemetry(x: *const Reader, buf: []u8) []const u8 {
        const fmt = "{{\"recall\": {{\"lookups\": {d}, \"proposals\": {d}, \"proposed\": {d}, \"accepted\": {d}}}}}";
        return std.fmt.bufPrint(buf, fmt, .{ x.lookups, x.proposals, x.proposed, x.accepted }) catch "";
    }
};

test {
    _ = @import("recall_test.zig");
    _ = @import("recall_engine_test.zig");
}
