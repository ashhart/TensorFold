//! Original FamilyRounds depth policy: per-depth acceptance, measured costs, and
//! one deeper probe every eight choices. All costs are milliseconds.
const std = @import("std");
pub const Adaptive = struct {
    rates: [15]f64 = @splat(0),
    rate_count: usize,
    budget: usize,
    forward_ms: [17]f64 = @splat(0),
    round_ms: [16]f64 = @splat(0),
    mtp_ms: f64 = 0,
    choices: usize = 0,
    pub fn init(budget: usize, prior: []const f64) !Adaptive {
        if (budget > 15 or prior.len == 0 or prior.len > 15) return error.InvalidDraftPolicy;
        for (prior) |rate| if (!std.math.isFinite(rate) or rate < 0 or rate > 1) return error.InvalidDraftPolicy;
        var out = Adaptive{ .budget = budget, .rate_count = @min(prior.len, @max(1, budget)) };
        @memcpy(out.rates[0..out.rate_count], prior[0..out.rate_count]);
        return out;
    }
    pub fn choose(p: *Adaptive, room: usize) usize {
        const most = @min(p.budget, @max(1, room -| 1));
        if (most == 0) return 0;
        var measured = false;
        for (p.forward_ms) |cost| if (cost > 0) {
            measured = true;
            break;
        };
        if (!measured) return @max(1, @min(most, if (p.rates[0] < 0.8) @as(usize, 1) else if (p.rates[0] < 0.9) @as(usize, 2) else 3));
        var best: usize = 1;
        var best_rate: f64 = -1;
        var expected: f64 = 1;
        var run: f64 = 1;
        for (1..most + 1) |depth| {
            const cost = if (p.round_ms[depth] > 0) p.round_ms[depth] else if (p.forward_ms[depth + 1] > 0) p.forward_ms[depth + 1] + p.mtp_ms * @as(f64, @floatFromInt(depth)) else break;
            run *= p.rates[@min(depth - 1, p.rate_count - 1)];
            expected += run;
            if (expected / cost > best_rate) {
                best = depth;
                best_rate = expected / cost;
            }
        }
        p.choices += 1;
        if (best < most and p.choices % 8 == 0) best += 1;
        return best;
    }
    pub fn observe(p: *Adaptive, proposed: usize, accepted: usize, milliseconds: f64) !void {
        if (proposed > p.budget or accepted > proposed or !std.math.isFinite(milliseconds) or milliseconds <= 0) return error.InvalidDraftObservation;
        for (0..@min(proposed, p.rate_count)) |j| {
            if (accepted < j) break;
            p.rates[j] += 0.15 * ((if (accepted > j) @as(f64, 1) else 0) - p.rates[j]);
        }
        if (proposed == 0) return;
        const before = p.round_ms[proposed];
        p.round_ms[proposed] = if (before == 0) milliseconds else before + 0.2 * (milliseconds - before);
    }
};
pub const flash_prior = [_]f64{ 0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
pub const nemotron_prior = [_]f64{ 0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5 };

pub fn check(io: std.Io, file: []const u8) !void {
    const Step = struct { room: usize, expected: usize, proposed: usize, accepted: usize, ms: f64, rates: []const f64 };
    const Case = struct { budget: usize, prior: []const f64, costs: [17]f64, mtp_ms: f64, steps: []const Step };
    const bytes = try @import("weights.zig").readFile(io, file);
    defer @import("mlx.zig").allocator.free(bytes);
    const fixtures = try std.json.parseFromSlice([]const Case, std.heap.c_allocator, bytes, .{});
    defer fixtures.deinit();
    var count: usize = 0;
    for (fixtures.value) |case| {
        var policy = try Adaptive.init(case.budget, case.prior);
        policy.forward_ms = case.costs;
        policy.mtp_ms = case.mtp_ms;
        for (case.steps) |step| {
            try std.testing.expectEqual(step.expected, policy.choose(step.room));
            try policy.observe(step.proposed, step.accepted, step.ms);
            try std.testing.expectEqual(step.rates.len, policy.rate_count);
            for (step.rates, policy.rates[0..policy.rate_count]) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
            count += 1;
        }
    }
    std.debug.print("PASS: {d} adaptive depth decisions and acceptance updates match original Python\n", .{count});
}

test "adaptive policy rejects invalid budgets, priors and observations" {
    try std.testing.expectError(error.InvalidDraftPolicy, Adaptive.init(16, &flash_prior));
    try std.testing.expectError(error.InvalidDraftPolicy, Adaptive.init(3, &.{}));
    try std.testing.expectError(error.InvalidDraftPolicy, Adaptive.init(3, &.{std.math.nan(f64)}));
    var p = try Adaptive.init(3, &flash_prior);
    try std.testing.expectError(error.InvalidDraftObservation, p.observe(4, 0, 1));
    try std.testing.expectError(error.InvalidDraftObservation, p.observe(1, 2, 1));
    try std.testing.expectError(error.InvalidDraftObservation, p.observe(1, 1, 0));
}
