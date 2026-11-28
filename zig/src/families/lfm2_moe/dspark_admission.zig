//! Full context debt and bounded probes for request-stable admission; the same-head, zero-tolerance full gate rules.
const std = @import("std");
const lanes = @import("lanes");
const hp = lanes.hold_price;
const Table = lanes.table.Table;

pub const max_depth = 16;
pub const prior: [max_depth]f64 = @splat(0.84);
const margin = (lanes.depth.Constants{}).plain_margin;

pub const Prices = struct {
    target: ?*const Table,
    holds: []const hp.HoldCost,
    contexts: []const hp.ContextCost,
    streams: usize,
    heads: usize,
    target_rows: u32,
    pass_rows: u32,
    block_rows: u32,
    depth: u32,
    /// Largest complete prompt or outstanding peer debt, not a fictitious zero-debt first hold.
    context_rows: usize,
    context_chunk: u32,
};

pub const Shape = struct {
    heads: usize,
    depth: u32,
    rows: u32,
    passes: usize,
    gain: f64,
    ms: f64,
    tokens: f64,
};

pub const Quote = struct {
    plain_ms: f64,
    streams: usize,
    best: ?Shape = null,

    pub fn pays(q: Quote) bool {
        const b = q.best orelse return false;
        return b.tokens * q.plain_ms > margin * @as(f64, @floatFromInt(q.streams)) * b.ms;
    }
};

fn positive(x: f64) bool {
    return x > 0 and std.math.isFinite(x);
}

/// Only calibrated retained prefixes fund admission; smaller prefixes do not shrink their query blocks.
pub fn quote(p: Prices, rates: []const f64) ?Quote {
    const target = p.target orelse return null;
    if (p.streams == 0 or p.streams > p.target_rows or p.block_rows == 0 or p.pass_rows < p.block_rows or p.context_chunk == 0) return null;
    // Missing grids remain unknown, never the hold-price helper's intentionally free empty convention.
    if (p.holds.len == 0 or p.contexts.len == 0) return null;
    for (p.holds) |h| if (h.streams == 0 or h.depth == 0 or !positive(h.ms)) return null;
    for (p.contexts) |c| if (c.streams == 0 or c.rows == 0 or !positive(c.ms)) return null;
    const plain = target.get(@intCast(p.streams)) orelse return null;
    if (!positive(plain)) return null;
    var result: Quote = .{ .plain_ms = plain, .streams = p.streams };
    const room = p.target_rows - @as(u32, @intCast(p.streams));
    const depth = @min(p.depth, @min(p.block_rows, rates.len));
    var survival: f64 = 1;
    var gain: f64 = 0;
    for (rates[0..depth], 1..) |rate, n| {
        if (!(rate >= 0 and rate <= 1) or !std.math.isFinite(rate)) return null;
        survival *= rate;
        gain += survival;
        const heads = @min(p.heads, room / n);
        for (1..heads + 1) |count| {
            const rows = p.streams + count * n;
            const verify = target.get(@intCast(rows)) orelse continue;
            if (!positive(verify)) return null;
            // Pay the whole prompt/context debt now, never amortized over an unknown reply length.
            const debt = @max(p.context_rows, n + 1);
            // Overflow may flush streams separately, so full chunks pay the individual price.
            const overflow = (debt - 1) / p.context_chunk;
            const remainder: u32 = @intCast(debt - overflow * p.context_chunk);
            const one = hp.contextMs(p.contexts, 1, remainder);
            const flushed = if (overflow == 0) 0 else hp.contextMs(p.contexts, 1, p.context_chunk) * @as(f64, @floatFromInt(overflow));
            if (!positive(one) or !(flushed >= 0) or !std.math.isFinite(flushed)) return null;
            const View = struct {
                count: usize,
                depth: i64,
                debt: u32,
                pub fn len(v: @This()) usize {
                    return v.count;
                }
                pub fn at(v: @This(), _: usize) i64 {
                    return v.depth;
                }
                pub fn contextRows(v: @This(), _: usize) u32 {
                    return v.debt;
                }
            };
            const view: View = .{ .count = count, .depth = @intCast(n), .debt = remainder };
            const clean = hp.rowsMs(p.holds, &.{}, p.pass_rows, p.block_rows, view);
            const grouped = hp.rowsMs(p.holds, p.contexts, p.pass_rows, p.block_rows, view);
            if (!positive(clean) or !positive(grouped)) return null;
            const count_f: f64 = @floatFromInt(count);
            // Remaining debt pays at least both the grouped and individual quotes.
            const ms = verify + @max(grouped, clean + one * count_f) + flushed * count_f;
            if (!positive(ms)) return null;
            const tokens = @as(f64, @floatFromInt(p.streams)) + gain * count_f;
            if (result.best == null or tokens * result.best.?.ms > result.best.?.tokens * ms) {
                const per_pass = p.pass_rows / p.block_rows;
                result.best = .{ .heads = count, .depth = @intCast(n), .rows = @intCast(rows), .passes = (count + per_pass - 1) / per_pass, .gain = gain * count_f, .ms = ms, .tokens = tokens };
            }
        }
    }
    // An entirely unpriced candidate set remains unknown, not a free enrollment.
    if (result.best == null and room > 0 and depth > 0 and p.heads > 0) return null;
    return result;
}

/// Provider/model/precision-local history, separate for greedy and sampled requests.
pub const Recent = struct {
    rates: [max_depth]f64 = prior,
    // Four prior trials remain recoverable as the effective observation count saturates.
    weight: [max_depth]f64 = @splat(4),
    observed: bool = false,
    remaining: u16 = 0,
    interval: u16 = 64,

    pub fn observe(r: *Recent, proposed: u32, accepted: u32) void {
        if (proposed == 0 or accepted > proposed) return;
        // Accepted conditionals and the first rejection were attempted; deeper conditionals did not fail.
        const tried = @min(max_depth, @min(@as(u64, proposed), @as(u64, accepted) + 1));
        for (0..tried) |i| {
            r.weight[i] = @min(32, r.weight[i] + 1);
            const hit: f64 = if (i < accepted) 1 else 0;
            r.rates[i] += (hit - r.rates[i]) / r.weight[i];
        }
        r.observed = true;
    }

    /// Only a complete, paying quote funds enrollment; cold paid trials retain their bounded request cadence.
    pub fn enroll(r: *Recent, p: Prices, conservative_rates: []const f64) bool {
        const priced = quote(p, conservative_rates) orelse return false;
        if (!priced.pays()) return false;
        if (r.observed) {
            r.remaining = 64;
            r.interval = 64;
            return true;
        }
        if (r.remaining != 0) {
            r.remaining -= 1;
            return false;
        }
        // Enrollment alone changes no acceptance data, even if the probe runs zero holds.
        r.remaining = r.interval;
        r.interval = @min(1024, r.interval * 2);
        return true;
    }
};

pub const History = struct {
    modes: [2]Recent = .{ .{}, .{} },

    pub fn recent(h: *History, sampled: bool) *Recent {
        return &h.modes[@intFromBool(sampled)];
    }
};

test "lfm2 dspark: admission sparse native target calibration does not turn unpriced depths into free enrollment" {
    const gpa = std.testing.allocator;
    // Native lone calibration measures these widths, not every retained-prefix length.
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .window_costs = &.{
        .{ .width = 1, .ms = 10 },
        .{ .width = 2, .ms = 40 },
        .{ .width = 4, .ms = 80 },
        .{ .width = 8, .ms = 160 },
        .{ .width = 16, .ms = 320 },
    } }, 16, 15);
    defer cfg.deinit(gpa);
    const holds = [_]hp.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1 }, .{ .streams = 1, .depth = 9, .ms = 1 } };
    const contexts = [_]hp.ContextCost{.{ .streams = 1, .rows = 2048, .ms = 0.01 }};
    const prices: Prices = .{
        .target = &cfg.family_costs,
        .holds = &holds,
        .contexts = &contexts,
        .streams = 1,
        .heads = 1,
        .target_rows = 16,
        .pass_rows = 32,
        .block_rows = 9,
        .depth = 9,
        .context_rows = 14,
        .context_chunk = 2048,
    };
    var recent: Recent = .{};
    try std.testing.expect(!recent.enroll(prices, &recent.rates));
    try cfg.family_costs.put(gpa, 4, 12);
    try std.testing.expect(recent.enroll(prices, &recent.rates));
    try std.testing.expectEqual(@as(u32, 3), quote(prices, &recent.rates).?.best.?.depth);
}
