//! Priced per-request lookup drafting (TF_DSV41_LOOKUP_PLAN=1): history suffix matches drafted only when they pay.

const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const joint = @import("joint.zig");

/// (shortest match of the band, prior p)
pub const bands = [_]struct { least: i64, prior: f64 }{ .{ .least = 0, .prior = 0.45 }, .{ .least = 8, .prior = 0.7 }, .{ .least = 16, .prior = 0.8 }, .{ .least = 32, .prior = 0.9 } };
pub const prior_weight: f64 = 2.0;
pub const decay: f64 = 0.85;
pub const probe_every = 8;
pub const probe_match: i64 = 16;
pub const alt_window = 8;
pub const candidates = 64;
pub const extend_to = 64;
/// An image row's virtual id base: never a token to draft
pub const vbase: u32 = 1 << 24;
pub const max_rows = depth.slot_rows; // 16

/// TF_DSV41_LOOKUP_PLAN, TF_DSV41_LOOKUP, TF_DSV41_LOOKUP_MIN, TF_DSV41_MAX_ROWS.
pub const Settings = struct {
    plan: bool = false,
    lookup: bool = true,
    min_match: i64 = 4,
    max_rows: i64 = max_rows,

    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        s.plan = try flag(std.c.getenv("TF_DSV41_LOOKUP_PLAN"), false);
        s.lookup = try flag(std.c.getenv("TF_DSV41_LOOKUP"), true);
        if (std.c.getenv("TF_DSV41_LOOKUP_MIN")) |v| if (std.mem.span(v).len > 0) {
            s.min_match = std.fmt.parseInt(i64, std.mem.trim(u8, std.mem.span(v), " "), 10) catch return error.BadLookupMin;
        };
        if (std.c.getenv("TF_DSV41_MAX_ROWS")) |v| if (std.mem.span(v).len > 0) {
            s.max_rows = std.fmt.parseInt(i64, std.mem.trim(u8, std.mem.span(v), " "), 10) catch return error.BadMaxRows;
        };
        if (s.min_match < 1 or s.min_match > 64) return error.BadLookupMin;
        if (s.max_rows < 2 or s.max_rows > max_rows) return error.BadMaxRows;
        return s;
    }

    /// The lookup flag: anything but 0 / false / off / no / "" is on; unset is `default`.
    fn flag(v: ?[*:0]const u8, default: bool) !bool {
        const raw = std.mem.trim(u8, std.mem.span(v orelse return default), " \t");
        var low: [8]u8 = undefined;
        if (raw.len > low.len) return true;
        const l = std.ascii.lowerString(&low, raw);
        for ([_][]const u8{ "0", "false", "off", "no", "" }) |n| if (std.mem.eql(u8, l, n)) return false;
        return true;
    }
};

/// Where every n-gram of a growing history ends.
pub const SuffixIndex = struct {
    gpa: Allocator,
    n: usize,
    tokens: std.ArrayList(u32) = .empty,
    ends: std.AutoHashMapUnmanaged([3]u32, std.ArrayList(u32)) = .empty,
    indexed: usize = 0,

    pub fn init(gpa: Allocator, n: usize) SuffixIndex {
        std.debug.assert(n >= 1 and n <= 3);
        return .{ .gpa = gpa, .n = n };
    }

    pub fn deinit(x: *SuffixIndex) void {
        var it = x.ends.valueIterator();
        while (it.next()) |l| l.deinit(x.gpa);
        x.ends.deinit(x.gpa);
        x.tokens.deinit(x.gpa);
    }

    pub fn extend(x: *SuffixIndex, tokens: []const u32) !void {
        try x.tokens.appendSlice(x.gpa, tokens);
    }

    fn key(x: *const SuffixIndex, end: usize) [3]u32 {
        var k: [3]u32 = @splat(0);
        @memcpy(k[0..x.n], x.tokens.items[end - x.n .. end]);
        return k;
    }

    /// The n-grams ending at indexed .. len - 1 (the one ending at the history's end joins once a token follows it).
    fn index(x: *SuffixIndex) !void {
        const last = x.tokens.items.len;
        var end = @max(x.indexed, x.n);
        while (end < last) : (end += 1) {
            const g = try x.ends.getOrPut(x.gpa, x.key(end));
            if (!g.found_existing) g.value_ptr.* = .empty;
            try g.value_ptr.append(x.gpa, @intCast(end));
        }
        x.indexed = @max(x.indexed, last);
    }

    pub const Match = struct { len: i64, end: i64 };

    /// The longest match of the history's suffix ending before its end (ties to the most recent), (0, -1) for none.
    pub fn match(x: *SuffixIndex) !Match {
        const t = x.tokens.items;
        const L = t.len;
        if (L <= x.n) return .{ .len = 0, .end = -1 };
        try x.index();
        const ends = x.ends.get(x.key(L)) orelse return .{ .len = 0, .end = -1 };
        const reach = @max(extend_to, x.n);
        var best: Match = .{ .len = 0, .end = -1 };
        const from = ends.items.len -| candidates;
        var i = ends.items.len;
        while (i > from) {
            i -= 1;
            const end: usize = ends.items[i];
            var length: usize = x.n;
            const limit = @min(reach, end);
            while (length < limit and t[end - 1 - length] == t[L - 1 - length]) length += 1;
            if (length > best.len) {
                best = .{ .len = @intCast(length), .end = @intCast(end) };
                if (length >= reach) break;
            }
        }
        return best;
    }

    /// Up to `count` drafts (into `out`) from the longest match of at least `min_match` tokens, and its length.
    pub fn propose(x: *SuffixIndex, count: usize, min_match: i64, out: *std.ArrayList(u32)) !i64 {
        out.clearRetainingCapacity();
        const m = try x.match();
        if (m.end < 0 or m.len < min_match or count == 0) return m.len;
        const t = x.tokens.items;
        const end: usize = @intCast(m.end);
        for (0..count) |i| {
            const j = end + i;
            const d = if (j < t.len) t[j] else out.items[j - t.len]; // past the end the copy reads its own output
            if (d >= vbase) break;
            try out.append(x.gpa, d);
        }
        return m.len;
    }
};

pub const Kind = enum { serial, dspark, lookup };

/// One request's drafting state (the lookup part).
pub const Request = struct {
    index: SuffixIndex,
    acc: [bands.len]f64 = @splat(0.0),
    tri: [bands.len]f64 = @splat(0.0),
    skipped: [bands.len]u32 = @splat(0),
    alt: [alt_window][2]f64 = undefined, // (tokens, ms), oldest first from `alt_head`
    alt_len: usize = 0,
    alt_head: usize = 0,
    kind: Kind = .serial,
    band: usize = 0,
    out: std.ArrayList(u32) = .empty,

    pub fn deinit(r: *Request) void {
        r.index.deinit();
        r.out.deinit(r.index.gpa);
    }

    fn pushAlt(r: *Request, tokens: f64, ms: f64) void {
        r.alt[(r.alt_head + r.alt_len) % alt_window] = .{ tokens, ms };
        if (r.alt_len < alt_window) r.alt_len += 1 else r.alt_head = (r.alt_head + 1) % alt_window;
    }
};

/// The batcher's lookup decisions: `start`, `window`, `observe`.
pub const Planner = struct {
    set: Settings,
    costs: *const costs_mod.Costs,
    dspark: bool,
    n: usize,

    pub fn init(set: Settings, costs: *const costs_mod.Costs, dspark: bool) Planner {
        return .{ .set = set, .costs = costs, .dspark = dspark, .n = @intCast(@max(1, @min(3, set.min_match))) };
    }

    pub fn start(p: *const Planner, gpa: Allocator, prompt: []const u32) !Request {
        var r: Request = .{ .index = SuffixIndex.init(gpa, p.n) };
        if (p.set.lookup) try r.index.extend(prompt);
        return r;
    }

    fn band(length: i64) usize {
        var b: usize = 0;
        for (bands, 0..) |x, i| if (length >= x.least) {
            b = i;
        };
        return b;
    }

    pub fn pOf(r: *const Request, b: usize) f64 {
        return (r.acc[b] + bands[b].prior * prior_weight) / (r.tri[b] + prior_weight);
    }

    /// Tokens a modelled ms of recent non-lookup rounds (before any: a one-draft DSpark round, or a serial step).
    pub fn altRate(p: *const Planner, r: *const Request) f64 {
        if (r.alt_len > 0) {
            var tokens: f64 = 0.0;
            var ms: joint.PySum = .{};
            for (0..r.alt_len) |i| {
                const x = r.alt[(r.alt_head + i) % alt_window];
                tokens += x[0];
                ms.add(x[1]);
            }
            return tokens / ms.total();
        }
        const c = p.costs;
        return if (p.dspark) 2.0 / (c.windowMs(2) + c.draft) else 1.0 / c.windowMs(1);
    }

    /// This round's lookup drafts (empty: none); `left`: tokens still allowed; `eos`: cut at the first end token.
    pub fn window(p: *const Planner, r: *Request, left: i64, drafts: bool, eos: []const u32) ![]const u32 {
        const room: i64 = @max(0, left - 1);
        r.kind = .serial;
        r.out.clearRetainingCapacity();
        if (!drafts or room <= 0) return &.{};
        if (p.set.lookup) {
            const count: usize = @intCast(@min(p.set.max_rows - 1, room));
            const length = try r.index.propose(count, p.set.min_match, &r.out);
            var ds = r.out.items;
            for (ds, 0..) |d, i| if (std.mem.indexOfScalar(u32, eos, d) != null) {
                ds = ds[0..i];
                break;
            };
            if (ds.len > 0) {
                const b = band(length);
                var qs: [max_rows]f64 = undefined;
                const q = @min(pOf(r, b), 0.995);
                for (qs[0..ds.len]) |*x| x.* = q;
                const best = depth.bestK(qs[0..ds.len], p.costs.verify, p.altRate(r), 1);
                var go = best.k > 0 and best.surplus > 0.0;
                if (!go) {
                    r.skipped[b] += 1;
                    go = best.k > 0 and bands[b].least >= probe_match and r.skipped[b] >= probe_every;
                }
                if (go) {
                    r.skipped[b] = 0;
                    r.kind = .lookup;
                    r.band = b;
                    return ds[0..best.k];
                }
            }
        }
        if (p.dspark) r.kind = .dspark;
        return &.{};
    }

    /// After the round: `rows` the window's rows, `kept` its drafts kept, `emitted` the tokens it added.
    pub fn observe(p: *const Planner, r: *Request, rows: usize, kept: usize, emitted: []const u32) !void {
        const drafts = rows -| 1;
        const k: f64 = @floatFromInt(kept);
        if (r.kind == .lookup) {
            const b = r.band;
            r.acc[b] = r.acc[b] * decay + k;
            r.tri[b] = r.tri[b] * decay + k + @as(f64, if (kept < drafts) 1.0 else 0.0);
        } else {
            const ms = p.costs.windowMs(@intCast(rows)) + (if (r.kind == .dspark) p.costs.draft else 0.0);
            r.pushAlt(k + 1.0, ms);
        }
        if (p.set.lookup) try r.index.extend(emitted);
    }
};

/// The planner as a stream's lanes proposer (`priced`): one decision per history length, `round` feeds outcomes back.
pub const Proposer = struct {
    planner: *const Planner,
    req: Request,
    eos: []const u32,
    /// the history length the planner's index holds (grows by committed tokens)
    synced: usize,
    /// the round's ask (history length, room) and its drafts: a second ask reads them back
    asked: ?struct { len: usize, max_draft: i64 } = null,
    kept_k: usize = 0,

    pub fn init(planner: *const Planner, gpa: Allocator, prompt: []const u32, eos: []const u32) !Proposer {
        return .{ .planner = planner, .req = try planner.start(gpa, prompt), .eos = eos, .synced = prompt.len };
    }

    pub fn deinit(x: *Proposer) void {
        x.req.deinit();
    }

    pub fn proposer(x: *Proposer) lanes.proposer.Proposer {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: lanes.proposer.Proposer.VTable = .{ .propose = propose, .last_match = lastMatch, .priced = true, .round = round };

    fn self(p: *anyopaque) *Proposer {
        return @ptrCast(@alignCast(p));
    }

    fn propose(p: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const x = self(p);
        if (x.asked) |a| if (a.len == context.len and a.max_draft == max_draft) return x.req.out.items[0..x.kept_k];
        // the history's new tokens (the reply so far) into the index
        if (context.len > x.synced and x.planner.set.lookup) try x.req.index.extend(context[x.synced..]);
        x.synced = @max(x.synced, context.len);
        const ds = try x.planner.window(&x.req, max_draft + 1, true, x.eos);
        x.kept_k = ds.len; // the drafts are a prefix of req.out
        x.asked = .{ .len = context.len, .max_draft = max_draft };
        return ds;
    }

    fn lastMatch(_: *anyopaque) i64 {
        return 0;
    }

    fn round(p: *anyopaque, rows: u32, kept: u32) void {
        const x = self(p);
        if (x.asked == null) x.req.kind = .serial; // not asked this round: a stream that does not draft
        x.asked = null;
        // no tokens: the index takes the committed ones from the context at the next propose (the same tokens)
        x.planner.observe(&x.req, rows, kept, &.{}) catch unreachable;
    }
};

test "lookup plan: a repeated span is drafted when its band pays, priced against the request's own rate" {
    const gpa = std.testing.allocator;
    var verify = [_]f64{ 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117 };
    const c: costs_mod.Costs = .{ .verify = &verify, .draft = 5.6 };
    const pl = Planner.init(.{ .plan = true }, &c, true);
    const span = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    try prompt.appendSlice(gpa, &span);
    try prompt.appendSlice(gpa, &.{ 1, 2 });
    try prompt.appendSlice(gpa, span[0..12]); // a 12-token match: band 1 (p 0.7)
    var r = try pl.start(gpa, prompt.items);
    defer r.deinit();
    const ds = try pl.window(&r, 100, true, &.{});
    try std.testing.expect(r.kind == .lookup and ds.len > 0 and ds[0] == 22);
    // no match: DSpark
    var r2 = try pl.start(gpa, &.{ 1, 2, 3, 4, 5 });
    defer r2.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try pl.window(&r2, 100, true, &.{})).len);
    try std.testing.expect(r2.kind == .dspark);
}
