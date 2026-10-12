//! Cost-derived DSpark depth: verify the k drafts maximizing E(k) - lambda verify[k]; clock-free, so ranks agree.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Costs = @import("costs.zig").Costs;
const calib = @import("calib.zig");
const joint = @import("joint.zig");
const tree = @import("tree.zig");

pub const rate_window = 16; // rounds in a slot's running rate
pub const slot_rows = 16; // a slot's window at most

pub const Mode = enum { cost, static };

/// The knobs (TF_DSV41_DEPTH, _DRAFT_DEPTH, _DRAFT_SKIP, _DEPTH_JOINT, _DEPTH_FAIR, _DEPTH_HOST_MS, _GRAPH_ROWS_MAX).
pub const Settings = struct {
    mode: Mode = .cost,
    cap: usize = 5,
    skip: bool = true,
    joint: u2 = 1,
    fair: f64 = joint.fair,
    host_ms: f64 = joint.host_ms,
    max_rows: u64 = 64,
    code_accept: bool = false, // TF_DSV41_CODE_ACCEPT: expand high-acceptance short-context chains
    context_bucket: u64 = 2048, // TF_DSV41_GRAPH_BUCKET; short contexts only use its linear buckets
};

/// [E(0) .. E(len)]: tokens a round verifying k drafts commits on average, into `out` (len + 1).
pub fn expectedTokens(qs: []const f64, out: []f64) void {
    var run: f64 = 1.0;
    var e: f64 = 1.0;
    out[0] = 1.0;
    for (qs, out[1..]) |q, *o| {
        run *= q;
        e += run;
        o.* = e;
    }
}

pub const Best = struct { k: usize, surplus: f64 };

/// The k from `least` to len(qs) (bounded by the table) maximizing E(k) - rate verify[k]; ties to the smaller k.
pub fn bestK(qs: []const f64, verify: []const f64, rate: f64, least: usize) Best {
    var es: [slot_rows + 1]f64 = undefined;
    const n = @min(qs.len, slot_rows);
    expectedTokens(qs[0..n], es[0 .. n + 1]);
    const top = @min(n, verify.len - 1);
    var best: Best = .{ .k = 0, .surplus = -std.math.inf(f64) };
    var k = least;
    while (k <= top) : (k += 1) {
        const s = es[k] - rate * verify[k];
        if (s > best.surplus + 1e-12) best = .{ .k = k, .surplus = s };
    }
    return best;
}

const Round = struct { tokens: f64, ms: f64 };

/// The last `rate_window` rounds: (tokens, modelled ms), oldest first.
const Ring = struct {
    items: [rate_window]Round = undefined,
    len: usize = 0,
    head: usize = 0,

    fn push(r: *Ring, x: Round) void {
        r.items[(r.head + r.len) % rate_window] = x;
        if (r.len < rate_window) r.len += 1 else r.head = (r.head + 1) % rate_window;
    }

    fn sums(r: Ring) Round {
        var tokens: f64 = 0.0; // whole tokens: exact
        var ms: joint.PySum = .{};
        for (0..r.len) |i| {
            const x = r.items[(r.head + i) % rate_window];
            tokens += x.tokens;
            ms.add(x.ms);
        }
        return .{ .tokens = tokens, .ms = ms.total() };
    }
};

const First = struct { token: u32, p: f64 };

pub const Slot = struct {
    rounds: Ring = .{},
    used: [slot_rows]f64 = undefined, // this window's confidences
    used_len: usize = 0,
    drafted: bool = false,
    first: ?First = null, // a zero-depth round's first draft, until its bonus
    q1: ?f64 = null,
    rows: usize = 0,
    keep: usize = 0,
    ms: ?f64 = null, // a tree round's modelled ms

    fn use(st: *Slot, conf: []const f64) void {
        st.used_len = @min(conf.len, slot_rows);
        @memcpy(st.used[0..st.used_len], conf[0..st.used_len]);
    }
};

const Acc = struct { slot: u32, tokens: f64, rows: u64, drafted: bool };

pub const Depth = struct {
    gpa: Allocator,
    set: Settings,
    costs: Costs,
    cal: calib.Calibration,
    stats: calib.Stats,
    slots: std.ArrayList(?Slot) = .empty,
    acc: std.ArrayList(Acc) = .empty, // the open round's commits, in commit order
    agg: [costs_rows + 1]Ring = @splat(.{}), // by windows in a round: recent (tokens, modelled ms)
    expanded_rounds: u64 = 0,
    expanded_rows: u64 = 0,
    forward_rows: u64 = 16, // Lanes.model supplies the real forward cap, not the allocator's graph limit

    const costs_rows = 64;

    /// `costs` is copied; `block` is the DSpark block (the calibration's positions).
    pub fn init(gpa: Allocator, costs: Costs, block: usize, set: Settings) !Depth {
        var cal = try calib.Calibration.init(gpa, block);
        errdefer cal.deinit(gpa);
        return .{ .gpa = gpa, .set = set, .costs = try costs.clone(gpa), .cal = cal, .stats = .{ .positions = block } };
    }

    pub fn deinit(d: *Depth) void {
        d.costs.deinit(d.gpa);
        d.cal.deinit(d.gpa);
        d.slots.deinit(d.gpa);
        d.acc.deinit(d.gpa);
    }

    pub fn reset(d: *Depth, slot: u32) void {
        if (slot < d.slots.items.len) d.slots.items[slot] = null;
    }

    pub fn state(d: *Depth, slot: u32) !*Slot {
        while (d.slots.items.len <= slot) try d.slots.append(d.gpa, null);
        const s = &d.slots.items[slot];
        if (s.* == null) s.* = .{};
        return &s.*.?;
    }

    /// The slot's committed tokens per modelled ms over its recent rounds (two a drafting 2-row round before any).
    pub fn rate(d: *Depth, slot: u32) !f64 {
        const st = try d.state(slot);
        if (st.rounds.len > 0) {
            const s = st.rounds.sums();
            return s.tokens / s.ms;
        }
        return 2.0 / (d.costs.windowMs(2) + d.costs.draft);
    }

    fn least(d: *const Depth, alone: bool, first: ?u32) usize {
        return if (d.set.skip and alone and first != null) 0 else 1;
    }

    /// Drafts to verify of `conf` (1 .. min(most, len, cap); 0 .. when alone with zero depth on and `first` known).
    pub fn choose(d: *Depth, slot: u32, conf: []const f64, most: usize, alone: bool, first: ?u32) !usize {
        const top = @min(@min(most, conf.len), d.set.cap);
        const st = try d.state(slot);
        st.first = null;
        st.q1 = null;
        st.ms = null;
        if (top == 0) {
            st.used_len = 0;
            st.drafted = false;
            return 0;
        }
        var k = top;
        if (d.set.mode == .cost) {
            var qs: [slot_rows]f64 = undefined;
            for (conf[0..top], qs[0..top], 0..) |p, *q, j| q.* = d.cal.q(j, p);
            const lo = d.least(alone, first);
            const r = try d.rate(slot);
            k = @max(lo, @min(bestK(qs[0..top], d.costs.verify, r, lo).k, top));
            st.q1 = qs[0];
            if (k == 0) st.first = .{ .token = first.?, .p = conf[0] };
        }
        st.use(conf[0..k]);
        st.drafted = true;
        return k;
    }

    /// Modelled ms of a round: its window of `rows` total rows, each further window's overhead, the pass.
    pub fn roundMs(d: *const Depth, rows: u64, windows: u64, drafted: bool) f64 {
        return d.costs.rowsMs(@intCast(rows)) + d.costs.slot * @as(f64, @floatFromInt(windows -| 1)) +
            (if (drafted) d.costs.draft else 0.0) + (if (d.set.joint == 2) d.set.host_ms else 0.0);
    }

    /// The open round's commits -> one (tokens, modelled ms) of the aggregate rate by its windows.
    fn close(d: *Depth) void {
        if (d.acc.items.len == 0) return;
        var tokens: f64 = 0.0;
        var rows: u64 = 0;
        var drafted = false;
        for (d.acc.items) |a| {
            tokens += a.tokens;
            rows += a.rows;
            drafted = drafted or a.drafted;
        }
        const n = d.acc.items.len;
        d.agg[@min(n, costs_rows)].push(.{ .tokens = tokens, .ms = d.roundMs(rows, n, drafted) });
        d.acc.clearRetainingCapacity();
    }

    /// The aggregate rate of a round of `windows` windows, else two tokens a drafting window.
    pub fn jointRate(d: *const Depth, windows: usize) f64 {
        const r = d.agg[@min(windows, costs_rows)];
        if (r.len > 0) {
            const s = r.sums();
            return s.tokens / s.ms;
        }
        const n: u64 = @max(1, windows);
        return 2.0 * @as(f64, @floatFromInt(n)) / d.roundMs(2 * n, n, true);
    }

    /// The slot's recent tokens a round, null before its first commit.
    pub fn tpr(d: *Depth, slot: u32) !?f64 {
        const st = try d.state(slot);
        if (st.rounds.len == 0) return null;
        return st.rounds.sums().tokens / @as(f64, @floatFromInt(st.rounds.len));
    }

    pub const Ask = struct { slot: u32, conf: []const f64, most: usize, first: ?u32 = null, position: u64 = std.math.maxInt(u64) };

    fn eligible(d: *Depth, a: Ask) !bool {
        const st = try d.state(a.slot);
        return a.position <= 4096 and st.rounds.len >= 4 and (try d.tpr(a.slot)).? >= 3.0;
    }

    fn expandable(d: *Depth, a: Ask, k: usize) !bool {
        const top = @min(@min(@min(a.most, a.conf.len), d.set.cap), slot_rows - 1);
        // Keep context graph keys stable. Past the linear buckets leave the old policy in charge.
        return top > k and d.set.context_bucket > 0 and a.position + top < 16 * d.set.context_bucket and
            (a.position + k) / d.set.context_bucket == (a.position + top) / d.set.context_bucket and try d.eligible(a);
    }

    /// Keep every old depth as a floor. Prose/cold slots are frozen; any long-context ask preserves the whole round.
    fn expand(d: *Depth, asks: []const Ask, others: u64, other_windows: usize, ks: []usize) !void {
        if (!d.set.code_accept or d.set.mode != .cost or d.set.joint == 2 or others != 0 or asks.len < 2) return;
        for (asks) |a| if (a.position > 4096) return;
        var baseline_rows: u64 = 0;
        for (ks) |k| baseline_rows += 1 + k;
        const bucket: u64 = blk: {
            // These are rowtab.BUCKETS. Never enlarge the existing graph or turn one forward into two.
            for ([_]u64{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64 }) |r| {
                if (r >= baseline_rows and r <= d.forward_rows) break :blk r;
            }
            return;
        };
        if (baseline_rows >= @min(d.set.max_rows, bucket)) return;
        var any = false;
        for (asks, ks) |a, k| any = any or try d.expandable(a, k);
        if (!any) return; // cold/prose/full windows allocate no policy scratch
        const slots = try d.gpa.alloc(joint.Slot, asks.len);
        defer d.gpa.free(slots);
        const qs = try d.gpa.alloc([slot_rows]f64, asks.len);
        defer d.gpa.free(qs);
        for (asks, ks, slots, qs) |a, k, *s, *q| {
            const top_available = @min(@min(@min(a.most, a.conf.len), d.set.cap), slot_rows - 1);
            const active = try d.expandable(a, k);
            const top = if (active) top_available else k;
            for (a.conf[0..top], q[0..top], 0..) |p, *out, j| out.* = d.cal.q(j, p);
            s.* = .{ .qs = q[0..top], .least = k, .top = top };
        }
        const n = asks.len + other_windows;
        const fixed = d.costs.draft + d.costs.slot * @as(f64, @floatFromInt(n -| 1));
        const added = try joint.expand(d.gpa, slots, others, d.costs, fixed, @min(d.set.max_rows, bucket), bucket, ks);
        if (added == 0) return;
        d.expanded_rounds += 1;
        d.expanded_rows += added;
        for (asks, ks) |a, k| (try d.state(a.slot)).use(a.conf[0..k]);
    }

    /// Every DSpark slot's k of one round, together; `others`: rows of the round's other windows. Writes `ks`.
    pub fn chooseJoint(d: *Depth, asks: []const Ask, others: u64, other_windows: usize, alone_in: ?bool, ks: []usize) !void {
        d.close();
        const alone = (alone_in orelse (asks.len == 1 and other_windows == 0)) and asks.len == 1 and other_windows == 0;
        if (d.set.joint == 0 or d.set.mode == .static or alone or asks.len + other_windows <= 1) {
            for (asks, ks) |a, *k| k.* = try d.choose(a.slot, a.conf, a.most, alone, a.first);
            // Exact-row single windows have no paid padding to spend.
            return;
        }
        const two = d.set.joint == 2;
        const qbuf = try d.gpa.alloc([slot_rows]f64, asks.len);
        defer d.gpa.free(qbuf);
        const slots = try d.gpa.alloc(joint.Slot, asks.len);
        defer d.gpa.free(slots);
        const tops = try d.gpa.alloc(usize, asks.len);
        defer d.gpa.free(tops);
        for (asks, qbuf, slots, tops) |a, *qb, *s, *t| {
            t.* = @min(@min(@min(a.most, a.conf.len), d.set.cap), slot_rows - 1);
            const st = try d.state(a.slot);
            st.first = null;
            st.q1 = null;
            st.ms = null;
            for (a.conf[0..t.*], qb[0..t.*], 0..) |p, *q, j| q.* = d.cal.q(j, p);
            st.q1 = if (t.* > 0) qb[0] else null;
            s.* = .{ .qs = qb[0..t.*], .least = if (t.* == 0 or (two and d.set.skip and a.first != null)) 0 else 1, .top = t.* };
        }
        const r = d.jointRate(asks.len + other_windows);
        if (two) {
            const tp = try d.gpa.alloc(?f64, asks.len);
            defer d.gpa.free(tp);
            const w = try d.gpa.alloc(f64, asks.len);
            defer d.gpa.free(w);
            for (asks, tp) |a, *x| x.* = try d.tpr(a.slot);
            joint.weights(tp, d.set.fair, w);
            for (slots, w) |*s, x| s.weight = x;
        }
        _ = try joint.allocate(d.gpa, slots, others, d.costs, r, d.set.max_rows, ks);
        try d.expand(asks, others, other_windows, ks);
        var rows: u64 = others;
        var drafts: u64 = 0;
        var zero: u64 = 0;
        for (asks, ks, tops) |a, k, t| {
            const st = try d.state(a.slot);
            st.use(a.conf[0..k]);
            st.drafted = t > 0;
            if (k == 0 and t > 0) {
                st.first = .{ .token = a.first.?, .p = a.conf[0] };
                zero += 1;
            }
            rows += 1 + k;
            drafts += k;
        }
        d.stats.joint[0] += 1;
        d.stats.joint[1] += rows;
        d.stats.joint[2] += drafts;
        d.stats.joint[3] += zero;
    }

    pub const TreeChoice = struct { k: usize, lens: [3]usize = @splat(0) };

    /// choose with first-position siblings (`sibs`: their calibrated chances); `dup`: tree.plan's duplicate pricing.
    pub fn chooseTree(d: *Depth, slot: u32, conf: []const f64, most: usize, sibs: []const f64, alone: bool, first: ?u32, dup: tree.Dup, sib_rows: usize) !TreeChoice {
        const top = @min(@min(most, conf.len), d.set.cap);
        if (top == 0 or d.set.mode == .static or sibs.len == 0) return .{ .k = try d.choose(slot, conf, most, alone, first) };
        const st = try d.state(slot);
        st.first = null;
        st.q1 = null;
        st.ms = null;
        var qs: [slot_rows]f64 = undefined;
        for (conf[0..top], qs[0..top], 0..) |p, *q, j| q.* = d.cal.q(j, p);
        const lo = d.least(alone, first);
        const p = tree.plan(qs[0..top], sibs, qs[1..top], d.costs.verify, try d.rate(slot), lo, dup, @min(sib_rows, top), slot_rows);
        var out: TreeChoice = .{ .k = @max(lo, @min(p.k, top)), .lens = p.lens };
        st.q1 = qs[0];
        if (out.k == 0) {
            st.first = .{ .token = first.?, .p = conf[0] };
            out.lens = @splat(0);
        }
        if (out.lens[0] + out.lens[1] + out.lens[2] > 0) st.ms = p.ms + d.costs.draft;
        st.use(conf[0..out.k]);
        st.drafted = true;
        return out;
    }

    /// The slot's last window had `rows` rows (its main chain's) and committed `keep`; `tokens`: a tree round's.
    pub fn record(d: *Depth, slot: u32, rows: usize, keep: usize, tokens: ?usize) !void {
        for (d.acc.items) |a| if (a.slot == slot) {
            d.close();
            break;
        };
        const st = try d.state(slot);
        const got: f64 = @floatFromInt(tokens orelse keep);
        try d.acc.append(d.gpa, .{ .slot = slot, .tokens = got, .rows = rows, .drafted = st.drafted });
        if (st.drafted) {
            const used = st.used[0..@min(st.used_len, rows -| 1)];
            d.cal.record(used, keep);
            d.stats.rounds += 1;
            if (st.first == null) d.stats.add(st.q1, used.len, keep, if (used.len > 0) keep > 1 else null);
        }
        var ms = d.costs.windowMs(@intCast(rows)) + (if (st.drafted) d.costs.draft else 0.0);
        if (st.ms) |m| ms = m;
        st.rounds.push(.{ .tokens = got, .ms = ms });
        st.used_len = 0;
        st.drafted = false;
        st.ms = null;
        st.rows = rows;
        st.keep = keep;
    }

    /// After record: the token the window chose after its last kept row teaches a zero-depth round's first draft.
    pub fn bonus(d: *Depth, slot: u32, token: u32) void {
        if (slot >= d.slots.items.len) return;
        const st = &(d.slots.items[slot] orelse return);
        const f = st.first orelse return;
        st.first = null;
        if (st.rows != 1 or st.keep != 1) return;
        const kept = token == f.token;
        d.cal.record(&.{f.p}, if (kept) 2 else 1);
        d.stats.zero += 1;
        d.stats.add(st.q1, 0, 1, kept);
    }
};

test "a lone window skips a weak first draft and keeps a strong chain" {
    const gpa = std.testing.allocator;
    var verify: [16]f64 = undefined;
    for (&verify, 0..) |*v, r| v.* = 21.4 + 4.6 * @as(f64, @floatFromInt(r));
    var d = try Depth.init(gpa, .{ .verify = &verify, .draft = 3.52, .slot = 0.65 }, 5, .{});
    defer d.deinit();
    try std.testing.expectEqual(@as(usize, 0), try d.choose(0, &.{ 0.05, 0.05, 0.05, 0.05, 0.05 }, 5, true, 7));
    try d.record(0, 1, 1, null);
    d.bonus(0, 7); // the skipped draft would have been kept
    try std.testing.expect(d.cal.kept[0] == 1.0 and d.stats.zero == 1);
    try std.testing.expectEqual(@as(usize, 1), try d.choose(0, &.{ 0.05, 0.05, 0.05, 0.05, 0.05 }, 5, false, 7));
    try std.testing.expectEqual(@as(usize, 5), try d.choose(1, &.{ 0.99, 0.99, 0.99, 0.99, 0.99 }, 5, true, 7));
}

test "high acceptance fills paid padding and preserves cold, prose, context and forward limits" {
    const gpa = std.testing.allocator;
    var verify: [64]f64 = undefined;
    for (&verify, 0..) |*v, i| v.* = 20 + 3 * @as(f64, @floatFromInt(i));
    const asks = [_]Depth.Ask{
        .{ .slot = 0, .conf = &.{ 0.95, 0.95, 0.95, 0.95, 0.95 }, .most = 5, .position = 100 },
        .{ .slot = 1, .conf = &.{ 0.95, 0.95, 0.95, 0.95, 0.95 }, .most = 5, .position = 100 },
        .{ .slot = 2, .conf = &.{ 0.95, 0.95, 0.95, 0.95, 0.95 }, .most = 5, .position = 100 },
        .{ .slot = 3, .conf = &.{ 0.95, 0.95, 0.95, 0.95, 0.95 }, .most = 5, .position = 100 },
    };
    for (0..7) |case| {
        var d = try Depth.init(gpa, .{ .verify = &verify, .draft = 4 }, 5, .{ .code_accept = true });
        defer d.deinit();
        d.forward_rows = if (case == 1) 16 else 64;
        if (case == 2) d.set.max_rows = 20;
        var input = asks;
        if (case == 3) input[0].position = 4097;
        if (case == 4) for (&input) |*a| {
            a.position = 2043;
        };
        for (0..4) |slot| {
            const st = try d.state(@intCast(slot));
            for (0..(if (case == 5) @as(usize, 3) else 4)) |_| st.rounds.push(.{ .tokens = if (case == 6) 2 else 4, .ms = 40 });
        }
        var ks = [_]usize{ 4, 3, 3, 3 }; // 17 real rows, already paying for a 24-row graph
        try d.expand(&input, 0, 0, &ks);
        var rows: usize = 4;
        for (ks) |k| rows += k;
        if (case == 0) {
            try std.testing.expectEqualSlices(usize, &.{ 5, 5, 5, 5 }, &ks);
            try std.testing.expectEqual(@as(u64, 7), d.expanded_rows);
        } else if (case == 2) {
            try std.testing.expectEqual(@as(usize, 20), rows);
        } else {
            try std.testing.expectEqualSlices(usize, &.{ 4, 3, 3, 3 }, &ks);
            try std.testing.expectEqual(@as(u64, 0), d.expanded_rounds);
        }
    }
}
