//! A shared round's DSpark depths chosen together: the m best marginal gains, scanned against cost by total rows.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Costs = @import("costs.zig").Costs;

extern "c" fn pow(x: f64, y: f64) f64;

pub const eps = 1e-12;
pub const fair = 0.5; // TF_DSV41_DEPTH_FAIR (mode 2's alpha)
pub const host_ms = 3.0; // TF_DSV41_DEPTH_HOST_MS (mode 2)
pub const tpr0 = 2.0; // a slot's tokens a round before its first commit

/// CPython's sum() of floats (3.12+): Neumaier's compensated sum, the compensation added at the end.
pub const PySum = struct {
    sum: f64 = 0.0,
    c: f64 = 0.0,

    pub fn add(p: *PySum, x: f64) void {
        const t = p.sum + x;
        p.c += if (@abs(p.sum) >= @abs(x)) (p.sum - t) + x else (x - t) + p.sum;
        p.sum = t;
    }

    pub fn total(p: PySum) f64 {
        return if (p.c != 0.0 and std.math.isFinite(p.c)) p.sum + p.c else p.sum;
    }
};

/// Each slot's weight (mode 2): (mean tpr / tpr_s) ^ alpha, scaled so sum w_s tpr_s = sum tpr_s. Writes `out`.
pub fn weights(tpr: []const ?f64, alpha: f64, out: []f64) void {
    if (tpr.len == 0 or alpha <= 0.0) {
        @memset(out, 1.0);
        return;
    }
    var total: PySum = .{};
    for (tpr) |x| total.add(@max(x orelse tpr0, eps));
    const mean = total.total() / @as(f64, @floatFromInt(tpr.len));
    var dot: PySum = .{};
    for (tpr, out) |x, *w| {
        const t = @max(x orelse tpr0, eps);
        w.* = pow(mean / t, alpha);
        dot.add(w.* * t);
    }
    const z = total.total() / dot.total();
    for (out) |*w| w.* *= z;
}

/// Marginal expected tokens of drafts least+1 .. top: q_1 ... q_{j+1} for j = least .. top - 1, into `out`.
pub fn gains(qs: []const f64, least: usize, top: usize, out: *std.ArrayList(f64), gpa: Allocator) !void {
    var run: f64 = 1.0;
    for (0..top) |j| {
        run *= qs[j];
        if (j >= least) try out.append(gpa, run);
    }
}

pub const Slot = struct { qs: []const f64, least: usize, top: usize, weight: f64 = 1.0 };

const Cand = struct { neg: f64, slot: u32, pos: u32 };

fn before(_: void, a: Cand, b: Cand) bool {
    if (a.neg != b.neg) return a.neg < b.neg;
    if (a.slot != b.slot) return a.slot < b.slot;
    return a.pos < b.pos;
}

/// Each slot's k into `ks` and the objective sum w E - rate C(R) without constants. `shared`: the round's other rows.
pub fn allocate(gpa: Allocator, slots: []const Slot, shared: u64, costs: Costs, rate: f64, max_rows: u64, ks: []usize) !f64 {
    var base: u64 = shared;
    for (slots, ks) |s, *k| {
        k.* = @min(s.least, s.top);
        base += 1 + k.*;
    }
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    var g: std.ArrayList(f64) = .empty;
    defer g.deinit(gpa);
    for (slots, ks, 0..) |s, lo, i| {
        g.clearRetainingCapacity();
        try gains(s.qs, lo, @max(lo, s.top), &g, gpa);
        for (g.items, 0..) |x, j| try cands.append(gpa, .{ .neg = -x * s.weight, .slot = @intCast(i), .pos = @intCast(lo + j) });
    }
    std.mem.sort(Cand, cands.items, {}, before);
    const room = if (max_rows > base) max_rows - base else 0;
    var best_m: usize = 0;
    var best = -rate * costs.rowsMs(@intCast(base));
    var acc: f64 = 0.0;
    for (0..@min(room, cands.items.len)) |i| {
        acc += -cands.items[i].neg;
        const v = acc - rate * costs.rowsMs(@intCast(base + i + 1));
        if (v > best + eps) {
            best_m = i + 1;
            best = v;
        }
    }
    for (cands.items[0..best_m]) |c| ks[c.slot] += 1;
    return best;
}

/// Expand an allocation only when expected tokens / full round ms improves 1%; scans all row counts (nonconvex).
pub fn expand(gpa: Allocator, slots: []const Slot, shared: u64, costs: Costs, fixed_ms: f64, max_rows: u64, priced_rows: ?u64, ks: []usize) !usize {
    var rows: u64 = shared;
    var tokens: f64 = 0;
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    for (slots, ks, 0..) |s, k, i| {
        rows += 1 + k;
        var survival: f64 = 1;
        tokens += 1;
        for (s.qs[0..s.top], 0..) |q, j| {
            survival *= q;
            if (j < k) tokens += survival else try cands.append(gpa, .{ .neg = -survival, .slot = @intCast(i), .pos = @intCast(j) });
        }
    }
    if (rows >= max_rows or cands.items.len == 0) return 0;
    std.mem.sort(Cand, cands.items, {}, before);
    const baseline = tokens / (costs.rowsMs(@intCast(priced_rows orelse rows)) + fixed_ms);
    var best = baseline;
    var count: usize = 0;
    for (cands.items[0..@min(cands.items.len, max_rows - rows)], 0..) |c, i| {
        tokens -= c.neg;
        const ratio = tokens / (costs.rowsMs(@intCast(priced_rows orelse (rows + i + 1))) + fixed_ms);
        if (ratio > best + eps) {
            best = ratio;
            count = i + 1;
        }
    }
    if (best < baseline * 1.01) return 0;
    for (cands.items[0..count]) |c| ks[c.slot] += 1;
    return count;
}

test "expansion maximizes full round throughput under every depth floor and row cap" {
    const gpa = std.testing.allocator;
    // Nonconvex row prices: scanning past an expensive row can still win.
    var verify = [_]f64{ 20, 22, 27, 27, 28, 33, 33, 34, 36, 40, 41, 42 };
    const c: Costs = .{ .verify = &verify, .draft = 4 };
    const qs = [_][5]f64{ .{ 0.95, 0.9, 0.85, 0.8, 0.75 }, .{ 0.9, 0.8, 0.7, 0.6, 0.5 } };
    const slots = [_]Slot{ .{ .qs = &qs[0], .least = 1, .top = 5 }, .{ .qs = &qs[1], .least = 1, .top = 5 } };
    for (1..6) |a| for (1..6) |b| for (4..13) |cap| {
        var ks = [_]usize{ a, b };
        var es: [2][6]f64 = undefined;
        for (qs, &es) |q, *e| {
            e[0] = 1;
            var survival: f64 = 1;
            for (q, e[1..], 0..) |v, *out, j| {
                survival *= v;
                out.* = e[j] + survival;
            }
        }
        const baseline = (es[0][a] + es[1][b]) / (c.rowsMs(@intCast(2 + a + b)) + 8);
        var best = baseline;
        for (a..6) |x| for (b..6) |y| {
            if (2 + x + y > cap) continue;
            best = @max(best, (es[0][x] + es[1][y]) / (c.rowsMs(@intCast(2 + x + y)) + 8));
        };
        const added = try expand(gpa, &slots, 0, c, 8, cap, null, &ks);
        const got = (es[0][ks[0]] + es[1][ks[1]]) / (c.rowsMs(@intCast(2 + ks[0] + ks[1])) + 8);
        try std.testing.expect(ks[0] >= a and ks[1] >= b);
        if (best < baseline * 1.01) {
            try std.testing.expectEqual(@as(usize, 0), added);
            try std.testing.expectEqual(baseline, got);
        } else try std.testing.expectApproxEqAbs(best, got, 1e-12);
    };
}

/// allocate's objective of given depths (the brute-force check).
pub fn objective(gpa: Allocator, slots: []const Slot, ks: []const usize, shared: u64, costs: Costs, rate: f64) !f64 {
    var total: f64 = 0.0;
    var rows: u64 = shared;
    var g: std.ArrayList(f64) = .empty;
    defer g.deinit(gpa);
    for (slots, ks) |s, k| {
        g.clearRetainingCapacity();
        try gains(s.qs, s.least, k, &g, gpa);
        var sum: f64 = 0.0;
        for (g.items) |x| sum += x;
        total += s.weight * sum;
        rows += 1 + k;
    }
    return total - rate * costs.rowsMs(@intCast(rows));
}

test "allocate equals brute force over every allocation" {
    const gpa = std.testing.allocator;
    var verify: [64]f64 = undefined;
    for (&verify, 0..) |*v, r| v.* = 21.0 + 4.0 * @as(f64, @floatFromInt(r)) + (if (r >= 8) @as(f64, 3.0) else 0.0);
    const costs: Costs = .{ .verify = &verify, .draft = 3.5 };
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (0..200) |_| {
        var qbuf: [3][5]f64 = undefined;
        var slots: [3]Slot = undefined;
        for (&slots, &qbuf) |*s, *q| {
            for (q) |*x| x.* = rnd.float(f64);
            s.* = .{ .qs = q, .least = rnd.uintLessThan(usize, 2), .top = 1 + rnd.uintLessThan(usize, 5), .weight = 0.5 + rnd.float(f64) };
        }
        const rate = 0.02 + 0.08 * rnd.float(f64);
        var ks: [3]usize = undefined;
        const got = try allocate(gpa, &slots, 1, costs, rate, 64, &ks);
        var best = -std.math.inf(f64);
        var try_k: [3]usize = undefined;
        for (slots[0].least..slots[0].top + 1) |a| for (slots[1].least..slots[1].top + 1) |b| for (slots[2].least..slots[2].top + 1) |c| {
            try_k = .{ a, b, c };
            best = @max(best, try objective(gpa, &slots, &try_k, 1, costs, rate));
        };
        try std.testing.expectApproxEqAbs(best, try objective(gpa, &slots, &ks, 1, costs, rate), 1e-9);
        try std.testing.expectApproxEqAbs(got, best, 1e-9);
    }
}
