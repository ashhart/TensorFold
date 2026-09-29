const std = @import("std");

pub const Cost = struct { rows: usize, ms: f64 };
const Candidate = struct { stream: usize, probability: f64 };

fn order(_: void, left: Candidate, right: Candidate) std.math.Order {
    const probability = std.math.order(right.probability, left.probability);
    return if (probability == .eq) std.math.order(left.stream, right.stream) else probability;
}

fn rate(rows: usize, expected: f64, costs: []const Cost, overhead: f64) f64 {
    if (costs.len == 0) return expected / (1.0 + overhead);
    for (costs) |cost| if (cost.rows == rows) return expected / (cost.ms + overhead);
    return -1.0;
}

/// Prefixes preserve each stream's parent order; equal probabilities favor the earlier stream.
pub fn allocate(a: std.mem.Allocator, fixed: []const usize, probabilities: []const []const f64, costs: []const Cost, overhead: f64, max_rows: usize) ![]usize {
    if (fixed.len != probabilities.len or !std.math.isFinite(overhead) or overhead < 0) return error.InvalidAllocation;
    for (costs, 0..) |cost, i| {
        if (!std.math.isFinite(cost.ms) or cost.ms < 0 or cost.ms + overhead <= 0) return error.InvalidAllocation;
        for (costs[0..i]) |before| if (cost.rows == before.rows) return error.InvalidAllocation;
    }
    const counts = try a.alloc(usize, fixed.len);
    defer a.free(counts);
    @memset(counts, 0);
    const best_counts = try a.alloc(usize, fixed.len);
    errdefer a.free(best_counts);
    @memset(best_counts, 0);
    var heap = std.PriorityQueue(Candidate, void, order).initContext({});
    defer heap.deinit(a);
    var rows: usize = 0;
    for (fixed, probabilities, 0..) |mandatory, chances, stream| {
        rows = try std.math.add(usize, rows, mandatory);
        for (chances) |chance| if (!std.math.isFinite(chance) or chance < 0 or chance > 1) return error.InvalidAllocation;
        if (chances.len != 0) try heap.push(a, .{ .stream = stream, .probability = chances[0] });
    }
    var expected: f64 = @floatFromInt(rows);
    var best = rate(rows, expected, costs, overhead);
    while (rows < max_rows) {
        const candidate = heap.pop() orelse break;
        const stream = candidate.stream;
        counts[stream] += 1;
        rows += 1;
        expected += candidate.probability;
        if (counts[stream] < probabilities[stream].len) try heap.push(a, .{ .stream = stream, .probability = probabilities[stream][counts[stream]] });
        const current = rate(rows, expected, costs, overhead);
        if (current > best) {
            best = current;
            @memcpy(best_counts, counts);
        }
    }
    return best_counts;
}

pub fn chainProbabilities(rates: []const f64, output: []f64) !void {
    if (output.len != 0 and rates.len == 0) return error.InvalidAllocation;
    for (rates) |p| if (!std.math.isFinite(p) or p < 0 or p > 1) return error.InvalidAllocation;
    var reach: f64 = 1;
    for (output, 0..) |*p, i| {
        reach *= rates[@min(i, rates.len - 1)];
        p.* = reach;
    }
}

pub fn check(io: std.Io, path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { fixed: []const usize, probabilities: []const []const f64, costs: []const Cost, overhead: f64, max_rows: usize, expected: []const usize };
    const Chain = struct { rates: []const f64, expected: []const f64 };
    const Fixture = struct { cases: []const Case, chains: []const Chain };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    const fixture = (try std.json.parseFromSlice(Fixture, a, bytes, .{})).value;
    for (fixture.cases) |case| {
        const actual = try allocate(a, case.fixed, case.probabilities, case.costs, case.overhead, case.max_rows);
        try std.testing.expectEqualSlices(usize, case.expected, actual);
    }
    for (fixture.chains) |chain| {
        const actual = try a.alloc(f64, chain.expected.len);
        try chainProbabilities(chain.rates, actual);
        try std.testing.expectEqualSlices(f64, chain.expected, actual);
    }
    std.debug.print("PASS: {d} upstream shared row allocations and {d} acceptance chains match exactly\n", .{ fixture.cases.len, fixture.chains.len });
}

test "shared allocation preserves mandatory rows, ties and zero-probability stopping" {
    const a = std.testing.allocator;
    const tied = try allocate(a, &.{ 1, 1 }, &.{ &.{ 0.5, 0.5 }, &.{0.5} }, &.{}, 0, 3);
    defer a.free(tied);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, tied);
    const full = try allocate(a, &.{ 4, 4 }, &.{ &.{1}, &.{1} }, &.{}, 0, 2);
    defer a.free(full);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, full);
    const zero = try allocate(a, &.{1}, &.{&.{ 0, 0 }}, &.{}, 0, 8);
    defer a.free(zero);
    try std.testing.expectEqualSlices(usize, &.{0}, zero);
    try std.testing.expectError(error.InvalidAllocation, allocate(a, &.{1}, &.{&.{std.math.nan(f64)}}, &.{}, 0, 8));
    try std.testing.expectError(error.InvalidAllocation, allocate(a, &.{1}, &.{&.{1}}, &.{.{ .rows = 1, .ms = 0 }}, 0, 8));
    var invalid_chain = [_]f64{1};
    try std.testing.expectError(error.InvalidAllocation, chainProbabilities(&.{}, &invalid_chain));
}
