//! Exact-count, exact-mode shared prices: a plain intercept and a drafted row excess.
const std = @import("std");
const Allocator = std.mem.Allocator;
pub const every: u64 = 64;
pub const stale: u64 = 256;
const Point = struct { residual: f64, rows: f64, at: u64 };
const Bucket = struct {
    streams: usize,
    sampled: bool,
    rounds: u64 = 0,
    last_probe: u64 = 0,
    plain: ?Point = null,
    draft: ?Point = null,
};
pub const Price = struct { plain: f64, fixed: f64, row: f64 };

pub const Prices = struct {
    buckets: std.ArrayList(Bucket) = .empty,

    pub fn deinit(p: *Prices, a: Allocator) void {
        p.buckets.deinit(a);
    }

    fn find(p: *const Prices, streams: usize, sampled: bool) ?usize {
        for (p.buckets.items, 0..) |b, i| if (b.streams == streams and b.sampled == sampled) return i;
        return null;
    }

    fn get(p: *Prices, a: Allocator, streams: usize, sampled: bool) !*Bucket {
        if (p.find(streams, sampled)) |i| return &p.buckets.items[i];
        try p.buckets.append(a, .{ .streams = streams, .sampled = sampled });
        return &p.buckets.items[p.buckets.items.len - 1];
    }

    /// Missing/stale plain prices earn a trial on round 64, then at least 64 completed rounds apart.
    pub fn due(p: *Prices, a: Allocator, streams: usize, sampled: bool) !bool {
        const b = try p.get(a, streams, sampled);
        const next = b.rounds +| 1;
        const old = if (b.plain) |v| next -| v.at >= stale else true;
        return old and next -| b.last_probe >= every;
    }

    /// The plain intercept is never updated by a round that verifies or prepares any draft.
    pub fn observe(p: *Prices, a: Allocator, streams: usize, rows: u64, residual: f64, sampled: bool, plain: bool, calibration: bool) !void {
        const b = try p.get(a, streams, sampled);
        b.rounds +|= 1;
        if (calibration) b.last_probe = b.rounds;
        if (!std.math.isFinite(residual)) return;
        const point: Point = .{ .residual = residual, .rows = @floatFromInt(rows), .at = b.rounds };
        if (plain and rows == streams) {
            blend(&b.plain, point);
        } else if (rows > streams) blend(&b.draft, point);
    }

    /// Before a plain observation, its independent prior is 8 ms, never the drafted residual.
    pub fn price(p: *const Prices, streams: usize, sampled: bool, fallback: f64, row: f64) ?Price {
        const b = p.buckets.items[p.find(streams, sampled) orelse return null];
        var out: Price = .{ .plain = 8, .fixed = fallback, .row = row };
        const plain = b.plain orelse return out;
        out.plain = plain.residual;
        out.fixed = plain.residual;
        if (b.draft) |draft| if (draft.rows > plain.rows) {
            out.row = @max(0, (draft.residual - plain.residual) / (draft.rows - plain.rows));
        };
        return out;
    }
};

fn blend(to: *?Point, v: Point) void {
    if (to.*) |*old| {
        old.residual += 0.2 * (v.residual - old.residual);
        old.rows += 0.2 * (v.rows - old.rows);
        old.at = v.at;
    } else to.* = v;
}

/// Explicit confidence, custom family trim and forced tokens retain their own policies.
pub fn calibrate(e: anytype, plans: anytype) !bool {
    if (!e.cfg.timed or !e.cfg.family_mtp or !e.cfg.plain_guard or e.cfg.draft_confidence != null or
        e.trim != null or plans.len < 2 or e.cfg.shared_costs.empty()) return false;
    for (plans) |p| if (p.kind == .forced or p.stream.rounds == 0) return false;
    return e.rule.shared_price.due(e.gpa, plans.len, @import("windows.zig").anySampled(plans));
}
