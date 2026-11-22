const std = @import("std");

pub const heuristic = "Greedy marginal histogram coverage per line byte, not real miss-cost optimization. Equal ratios prefer fewer lines, then the lower decoder ordinal. Exact line completion uses bounded search, potentially exponential with many distinct costs. Candidates must also preserve exact device picks bytes; otherwise sizing stays uniform.";
pub const experts: usize = 512;

pub const Request = struct {
    counts: []const []const u32,
    rank: usize,
    world: usize,
    line_bytes: []const usize,
    uniform_lines: usize,
    min_lines: usize,
    max_rows: usize,
    slots: usize,
};

pub const Budget = struct {
    baseline_bytes: u64,
    allocated_bytes: u64,
    uniform_fallback: bool = false,
};

pub const Error = std.mem.Allocator.Error || error{ InvalidDimensions, InvalidRank, InvalidLineCost, InvalidBounds, Overflow, UnfillableBudget };

fn poolBytes(line_bytes: usize, lines: usize, max_rows: usize, slots: usize, min_lines: usize) Error!u64 {
    if (lines == 0) return error.InvalidBounds;
    if (min_lines == 0) return error.InvalidBounds;
    if (max_rows == 0 or slots == 0) return error.InvalidDimensions;
    const picks = try std.math.mul(usize, max_rows, slots);
    const max_passes = 2 + experts / min_lines + @intFromBool(experts % min_lines != 0);
    const passes = @min(max_passes, @min(picks, experts) + 1);
    const picks_bytes = try std.math.mul(usize, try std.math.mul(usize, passes, picks), @sizeOf(i32));
    return std.math.add(u64, try std.math.mul(u64, lines, line_bytes), picks_bytes);
}

const Group = struct {
    bytes: u64,
    available: usize,

    fn descending(_: void, left: Group, right: Group) bool {
        return left.bytes > right.bytes;
    }
};

pub fn allocate(gpa: std.mem.Allocator, request: Request, output: []usize) Error!Budget {
    if ((request.world != 1 and request.world != 2) or request.rank >= request.world) return error.InvalidRank;
    const local_experts = experts / request.world;
    if (request.min_lines == 0 or request.min_lines > request.uniform_lines or request.uniform_lines > local_experts) return error.InvalidBounds;
    if (request.counts.len == 0 or request.counts.len != request.line_bytes.len or output.len != request.counts.len) return error.InvalidDimensions;
    var baseline_bytes: u64 = 0;
    var line_budget_bytes: u64 = 0;
    var floor_bytes: u64 = 0;
    for (request.counts, request.line_bytes) |histogram, bytes| {
        if (histogram.len != experts) return error.InvalidDimensions;
        if (bytes == 0) return error.InvalidLineCost;
        baseline_bytes = try std.math.add(u64, baseline_bytes, try poolBytes(bytes, request.uniform_lines, request.max_rows, request.slots, request.min_lines));
        line_budget_bytes = try std.math.add(u64, line_budget_bytes, try std.math.mul(u64, request.uniform_lines, bytes));
        floor_bytes = try std.math.add(u64, floor_bytes, try std.math.mul(u64, request.min_lines, bytes));
    }
    const ranked = try gpa.alloc([experts]u32, request.counts.len);
    defer gpa.free(ranked);
    const lines = try gpa.alloc(usize, request.counts.len);
    defer gpa.free(lines);
    const storage = try gpa.alloc(Group, request.counts.len);
    defer gpa.free(storage);
    const layer_groups = try gpa.alloc(usize, request.counts.len);
    defer gpa.free(layer_groups);
    const attempted = try gpa.alloc(bool, request.counts.len);
    defer gpa.free(attempted);
    const suffix_length = try std.math.add(usize, request.counts.len, 1);
    const suffix_bytes = try gpa.alloc(u64, suffix_length);
    defer gpa.free(suffix_bytes);
    const suffix_gcd = try gpa.alloc(u64, suffix_length);
    defer gpa.free(suffix_gcd);
    const first = request.rank * local_experts;
    for (request.counts, ranked) |histogram, *ordered| {
        @memcpy(ordered[0..local_experts], histogram[first..][0..local_experts]);
        std.mem.sort(u32, ordered[0..local_experts], {}, std.sort.desc(u32));
    }
    @memset(lines, request.min_lines);
    var group_count: usize = 0;
    for (request.line_bytes) |bytes| {
        var found: ?usize = null;
        for (storage[0..group_count], 0..) |group, index| {
            if (group.bytes == bytes) {
                found = index;
                break;
            }
        }
        const group_index = found orelse new: {
            storage[group_count] = .{ .bytes = bytes, .available = 0 };
            group_count += 1;
            break :new group_count - 1;
        };
        storage[group_index].available = try std.math.add(usize, storage[group_index].available, local_experts - request.min_lines);
    }
    const groups = storage[0..group_count];
    std.mem.sort(Group, groups, {}, Group.descending);
    for (request.line_bytes, layer_groups) |bytes, *group_index| {
        for (groups, 0..) |group, index| {
            if (group.bytes == bytes) {
                group_index.* = index;
                break;
            }
        }
    }
    var remaining = line_budget_bytes - floor_bytes;
    while (remaining != 0) {
        @memset(attempted, false);
        var selected: ?usize = null;
        for (0..lines.len) |_| {
            var best: ?usize = null;
            for (lines, request.line_bytes, 0..) |count, bytes, layer| {
                if (attempted[layer] or count == local_experts or bytes > remaining) continue;
                if (best == null or better(layer, best.?, ranked, lines, request.line_bytes)) best = layer;
            }
            const candidate = best orelse break;
            attempted[candidate] = true;
            const group_index = layer_groups[candidate];
            groups[group_index].available -= 1;
            const possible = canFill(groups, remaining - request.line_bytes[candidate], suffix_bytes, suffix_gcd);
            groups[group_index].available += 1;
            if (possible) {
                selected = candidate;
                break;
            }
        }
        const layer = selected orelse return error.UnfillableBudget;
        lines[layer] += 1;
        groups[layer_groups[layer]].available -= 1;
        remaining -= request.line_bytes[layer];
    }
    var allocated_bytes: u64 = 0;
    for (lines, request.line_bytes) |count, bytes| allocated_bytes = try std.math.add(u64, allocated_bytes, try poolBytes(bytes, count, request.max_rows, request.slots, request.min_lines));
    const uniform_fallback = allocated_bytes != baseline_bytes;
    if (uniform_fallback) {
        @memset(lines, request.uniform_lines);
        allocated_bytes = baseline_bytes;
    }
    @memcpy(output, lines);
    return .{ .baseline_bytes = baseline_bytes, .allocated_bytes = allocated_bytes, .uniform_fallback = uniform_fallback };
}

fn better(left: usize, right: usize, ranked: []const [experts]u32, lines: []const usize, bytes: []const usize) bool {
    const left_score = @as(u128, ranked[left][lines[left]]) * bytes[right];
    const right_score = @as(u128, ranked[right][lines[right]]) * bytes[left];
    if (left_score != right_score) return left_score > right_score;
    if (lines[left] != lines[right]) return lines[left] < lines[right];
    return left < right;
}

fn gcd(left: u64, right: u64) u64 {
    var numerator = left;
    var denominator = right;
    while (denominator != 0) {
        const remainder = numerator % denominator;
        numerator = denominator;
        denominator = remainder;
    }
    return numerator;
}

fn canFill(groups: []const Group, target: u64, suffix_bytes: []u64, suffix_gcd: []u64) bool {
    suffix_bytes[groups.len] = 0;
    suffix_gcd[groups.len] = 0;
    var index = groups.len;
    while (index != 0) {
        index -= 1;
        const group = groups[index];
        const capacity = @as(u128, group.bytes) * group.available + suffix_bytes[index + 1];
        suffix_bytes[index] = @intCast(@min(capacity, target));
        suffix_gcd[index] = if (group.available == 0) suffix_gcd[index + 1] else gcd(group.bytes, suffix_gcd[index + 1]);
    }
    return fill(groups, 0, target, suffix_bytes, suffix_gcd);
}

fn fill(groups: []const Group, index: usize, target: u64, suffix_bytes: []const u64, suffix_gcd: []const u64) bool {
    if (target == 0) return true;
    if (index == groups.len or target > suffix_bytes[index] or suffix_gcd[index] == 0 or target % suffix_gcd[index] != 0) return false;
    const group = groups[index];
    if (group.available == 0) return fill(groups, index + 1, target, suffix_bytes, suffix_gcd);
    const max_take = @min(group.available, target / group.bytes);
    const required = target - @min(target, suffix_bytes[index + 1]);
    const min_take = required / group.bytes + @intFromBool(required % group.bytes != 0);
    if (min_take > max_take) return false;
    var take = max_take;
    while (true) {
        if (fill(groups, index + 1, target - take * group.bytes, suffix_bytes, suffix_gcd)) return true;
        if (take == min_take) break;
        take -= 1;
    }
    return false;
}

fn testRequest(counts: []const []const u32, bytes: []const usize, uniform: usize, minimum: usize) Request {
    return .{ .counts = counts, .rank = 0, .world = 2, .line_bytes = bytes, .uniform_lines = uniform, .min_lines = minimum, .max_rows = 1, .slots = 1 };
}

test "profile budget: sorted marginal counts favor hot layers with exact bytes" {
    var hot: [experts]u32 = @splat(0);
    const cold: [experts]u32 = @splat(1);
    @memset(hot[0..6], 100);
    var output: [2]usize = undefined;
    const plan = try allocate(std.testing.allocator, testRequest(&.{ &hot, &cold }, &.{ 4, 6 }, 3, 1), &output);
    try std.testing.expectEqualSlices(usize, &.{ 6, 1 }, &output);
    try std.testing.expectEqual(@as(u64, 30 + 2 * 8), plan.baseline_bytes);
    try std.testing.expectEqual(plan.baseline_bytes, plan.allocated_bytes);
    hot[0] = 0;
    hot[255] = 100;
    _ = try allocate(std.testing.allocator, testRequest(&.{ &hot, &cold }, &.{ 4, 6 }, 3, 1), &output);
    try std.testing.expectEqualSlices(usize, &.{ 6, 1 }, &output);
}

test "profile budget: rank filtering handles asymmetric counts and TP1" {
    var first: [experts]u32 = @splat(0);
    var second: [experts]u32 = @splat(0);
    @memset(first[0..4], 10);
    @memset(second[257..260], 20);
    second[511] = 20;
    var output: [2]usize = undefined;
    var request = testRequest(&.{ &first, &second }, &.{ 8, 8 }, 2, 1);
    _ = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, &output);
    request.rank = 1;
    _ = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, &output);
    request.rank = 0;
    request.world = 1;
    _ = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, &output);
}

test "profile budget: zero and flat histograms retain uniform sizing deterministically" {
    for ([_]u32{ 0, 7, std.math.maxInt(u32) }) |value| {
        const histogram: [experts]u32 = @splat(value);
        var output: [3]usize = undefined;
        for (0..3) |_| {
            _ = try allocate(std.testing.allocator, testRequest(&.{ &histogram, &histogram, &histogram }, &.{ 16, 16, 16 }, 17, 10), &output);
            try std.testing.expectEqualSlices(usize, &.{ 17, 17, 17 }, &output);
        }
    }
}

test "profile budget: tied ratios prefer fewer lines then lower decoder ordinal" {
    var first: [experts]u32 = @splat(0);
    var second: [experts]u32 = @splat(0);
    const third: [experts]u32 = @splat(0);
    @memset(first[0..3], 10);
    @memset(second[0..3], 10);
    var output: [3]usize = undefined;
    _ = try allocate(std.testing.allocator, testRequest(&.{ &first, &second, &third }, &.{ 8, 8, 8 }, 2, 1), &output);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1 }, &output);
}

test "profile budget: ratios use wide products and avoid greedy dead ends" {
    const hot: [experts]u32 = @splat(std.math.maxInt(u32) - 1);
    const cold: [experts]u32 = @splat(std.math.maxInt(u32));
    const unit: usize = @as(usize, 1) << 32;
    var output: [2]usize = undefined;
    const plan = try allocate(std.testing.allocator, testRequest(&.{ &hot, &cold }, &.{ 2 * unit, 3 * unit }, 3, 1), &output);
    try std.testing.expectEqualSlices(usize, &.{ 6, 1 }, &output);
    try std.testing.expectEqual(@as(u64, 15) * unit + 2 * 8, plan.allocated_bytes);
    const low: [experts]u32 = @splat(1);
    _ = try allocate(std.testing.allocator, testRequest(&.{ &hot, &low }, &.{ 4, 6 }, 2, 1), &output);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, &output);
}

test "profile budget: floors caps and single layer budgets" {
    const hot: [experts]u32 = @splat(100);
    const cold: [experts]u32 = @splat(0);
    var output: [2]usize = undefined;
    _ = try allocate(std.testing.allocator, testRequest(&.{ &hot, &cold }, &.{ 8, 8 }, 200, 10), &output);
    try std.testing.expectEqualSlices(usize, &.{ 256, 144 }, &output);
    var request = testRequest(&.{ &hot, &cold }, &.{ 8, 8 }, 400, 10);
    request.world = 1;
    _ = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 512, 288 }, &output);
    _ = try allocate(std.testing.allocator, testRequest(&.{ &hot, &cold }, &.{ 7, 11 }, 10, 10), &output);
    try std.testing.expectEqualSlices(usize, &.{ 10, 10 }, &output);
    _ = try allocate(std.testing.allocator, testRequest(&.{&hot}, &.{13}, 47, 10), output[0..1]);
    try std.testing.expectEqual(@as(usize, 47), output[0]);
}

test "profile budget: malformed inputs and overflow leave output unchanged" {
    const histogram: [experts]u32 = @splat(0);
    const shared: [513]u32 = @splat(0);
    var output = [_]usize{ 999, 999 };
    const valid = testRequest(&.{&histogram}, &.{8}, 20, 10);
    for ([_][]const []const u32{ &.{}, &.{histogram[0..511]}, &.{&shared} }) |counts| {
        var request = valid;
        request.counts = counts;
        try std.testing.expectError(error.InvalidDimensions, allocate(std.testing.allocator, request, output[0..1]));
    }
    var request = valid;
    request.line_bytes = &.{};
    try std.testing.expectError(error.InvalidDimensions, allocate(std.testing.allocator, request, output[0..1]));
    try std.testing.expectError(error.InvalidDimensions, allocate(std.testing.allocator, valid, &output));
    request = valid;
    request.line_bytes = &.{0};
    try std.testing.expectError(error.InvalidLineCost, allocate(std.testing.allocator, request, output[0..1]));
    for ([_][2]usize{ .{ 0, 0 }, .{ 3, 0 }, .{ 2, 2 }, .{ 1, 1 }, .{ 2, std.math.maxInt(usize) } }) |identity| {
        request = valid;
        request.world = identity[0];
        request.rank = identity[1];
        try std.testing.expectError(error.InvalidRank, allocate(std.testing.allocator, request, output[0..1]));
    }
    for ([_][2]usize{ .{ 20, 0 }, .{ 9, 10 }, .{ 257, 10 }, .{ 0, 0 } }) |bounds| {
        request = valid;
        request.uniform_lines = bounds[0];
        request.min_lines = bounds[1];
        try std.testing.expectError(error.InvalidBounds, allocate(std.testing.allocator, request, output[0..1]));
    }
    try std.testing.expectError(error.Overflow, allocate(std.testing.allocator, testRequest(&.{&histogram}, &.{std.math.maxInt(usize)}, 2, 1), output[0..1]));
    const bytes = std.math.maxInt(usize) / 2 + 1;
    try std.testing.expectError(error.Overflow, allocate(std.testing.allocator, testRequest(&.{ &histogram, &histogram }, &.{ bytes, bytes }, 1, 1), &output));
    try std.testing.expectEqualSlices(usize, &.{ 999, 999 }, &output);
}

fn allocationFailureProbe(gpa: std.mem.Allocator) !void {
    const histogram: [experts]u32 = @splat(7);
    var output = [_]usize{ 999, 999 };
    const result = allocate(gpa, testRequest(&.{ &histogram, &histogram }, &.{ 4, 6 }, 3, 1), &output) catch |err| {
        try std.testing.expectEqualSlices(usize, &.{ 999, 999 }, &output);
        return err;
    };
    try std.testing.expectEqual(result.baseline_bytes, result.allocated_bytes);
}

test "profile budget: allocation failures release scratch and preserve output" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}

test "profile budget: feasibility matches exhaustive bounded enumeration" {
    const groups = [_]Group{ .{ .bytes = 11, .available = 1 }, .{ .bytes = 7, .available = 2 }, .{ .bytes = 5, .available = 3 }, .{ .bytes = 3, .available = 4 } };
    var suffix_bytes: [5]u64 = undefined;
    var suffix_gcd: [5]u64 = undefined;
    for (0..65) |target| {
        var expected = false;
        for (0..2) |first| {
            for (0..3) |second| {
                for (0..4) |third| {
                    for (0..5) |fourth| {
                        if (11 * first + 7 * second + 5 * third + 3 * fourth == target) expected = true;
                    }
                }
            }
        }
        try std.testing.expectEqual(expected, canFill(&groups, target, &suffix_bytes, &suffix_gcd));
    }
    try std.testing.expect(!canFill(&.{ .{ .bytes = 6, .available = 10 }, .{ .bytes = 4, .available = 10 } }, 2, &suffix_bytes, &suffix_gcd));
}

test "profile budget: heterogeneous allocations preserve the feasible uniform witness" {
    const first: [experts]u32 = @splat(17);
    const second: [experts]u32 = @splat(11);
    const third: [experts]u32 = @splat(5);
    var output: [3]usize = undefined;
    for (1..6) |first_cost| {
        for (1..6) |second_cost| {
            for (1..6) |third_cost| {
                const bytes = [_]usize{ first_cost, second_cost, third_cost };
                const plan = try allocate(std.testing.allocator, testRequest(&.{ &first, &second, &third }, &bytes, 2, 1), &output);
                try std.testing.expectEqual(@as(u64, 2 * (first_cost + second_cost + third_cost) + 3 * 8), plan.allocated_bytes);
                try std.testing.expectEqual(plan.baseline_bytes, plan.allocated_bytes);
                var observed: u64 = 0;
                for (output, bytes) |count, cost| {
                    try std.testing.expect(count >= 1 and count <= 256);
                    observed += try poolBytes(cost, count, 1, 1, 1);
                }
                try std.testing.expectEqual(plan.allocated_bytes, observed);
            }
        }
    }
}

test "profile budget: device picks capacity stays uniform across pool sizes" {
    const line_bytes = (1 << 20) + 8;
    for ([_]usize{ 10, 11, 31, 32, 33, 64, 256, 512 }) |lines| {
        const actual = try poolBytes(line_bytes, lines, 64, 11, 10);
        try std.testing.expectEqual(@as(u64, lines * line_bytes + 54 * 64 * 11 * 4), actual);
        try std.testing.expectEqual(@as(usize, 54), @divExact(@as(usize, @intCast(actual - lines * line_bytes)), 64 * 11 * 4));
    }
    try std.testing.expectEqual(@as(u64, 10 * line_bytes + 12 * 11 * 4), try poolBytes(line_bytes, 10, 1, 11, 10));
    try std.testing.expectEqual(@as(u64, 10 * line_bytes + 2 * 4), try poolBytes(line_bytes, 10, 1, 1, 10));
    try std.testing.expectEqual(@as(u64, 256 * line_bytes + 12 * 11 * 4), try poolBytes(line_bytes, 256, 1, 11, 10));
}

test "profile budget: redistribution preserves expert planner and picks bytes exactly" {
    var hot: [experts]u32 = @splat(0);
    @memset(hot[0..55], 100);
    const cold: [experts]u32 = @splat(1);
    const line_bytes = (1 << 20) + 8;
    var request = testRequest(&.{ &hot, &cold }, &.{ line_bytes, line_bytes }, 54, 53);
    request.max_rows = 64;
    request.slots = 11;
    var output: [2]usize = undefined;
    const plan = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 55, 53 }, &output);
    try std.testing.expect(!plan.uniform_fallback);
    try std.testing.expectEqual(@as(u64, 108 * line_bytes + 2 * 12 * 64 * 11 * 4), plan.baseline_bytes);
    var observed: u64 = 0;
    for (output) |lines| {
        try std.testing.expect(lines >= request.min_lines);
        observed += try poolBytes(line_bytes, lines, request.max_rows, request.slots, request.min_lines);
    }
    try std.testing.expectEqual(plan.baseline_bytes, observed);
    try std.testing.expectEqual(observed, plan.allocated_bytes);
}

test "profile budget: fixed picks capacity preserves exact bytes across pass thresholds" {
    var hot: [experts]u32 = @splat(0);
    @memset(hot[0..33], 100);
    const cold: [experts]u32 = @splat(1);
    const line_bytes = (1 << 20) + 8;
    var request = testRequest(&.{ &hot, &cold }, &.{ line_bytes, line_bytes }, 32, 31);
    var output: [2]usize = undefined;
    _ = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 33, 31 }, &output);
    request.max_rows = 64;
    request.slots = 11;
    const plan = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 33, 31 }, &output);
    try std.testing.expect(!plan.uniform_fallback);
    try std.testing.expectEqual(@as(u64, 64 * line_bytes + 2 * 19 * 64 * 11 * 4), plan.baseline_bytes);
    try std.testing.expectEqual(plan.baseline_bytes, plan.allocated_bytes);
}

test "profile budget: fewer picks bytes also retain the exact uniform budget" {
    const hot: [experts]u32 = @splat(100);
    const cold: [experts]u32 = @splat(1);
    var request = testRequest(&.{ &hot, &cold }, &.{ 4, 6 }, 3, 1);
    request.max_rows = 16;
    request.slots = 11;
    var output: [2]usize = undefined;
    const plan = try allocate(std.testing.allocator, request, &output);
    try std.testing.expectEqualSlices(usize, &.{ 6, 1 }, &output);
    try std.testing.expect(!plan.uniform_fallback);
    try std.testing.expectEqual(@as(u64, 30 + 2 * 177 * 16 * 11 * 4), plan.baseline_bytes);
    try std.testing.expectEqual(plan.baseline_bytes, plan.allocated_bytes);
}

test "profile budget: TP2 line count above the local expert bank is rejected for caller fallback" {
    const histogram: [experts]u32 = @splat(1);
    var request = testRequest(&.{&histogram}, &.{8}, 257, 10);
    var output: [1]usize = .{257};
    try std.testing.expectError(error.InvalidBounds, allocate(std.testing.allocator, request, &output));
    try std.testing.expectEqual(@as(usize, 257), output[0]);
    request.uniform_lines = 256;
    _ = try allocate(std.testing.allocator, request, &output);
}

test "profile budget: picks dimensions and byte overflow preserve output" {
    const histogram: [experts]u32 = @splat(0);
    var request = testRequest(&.{&histogram}, &.{8}, 20, 10);
    var output = [_]usize{999};
    request.max_rows = 0;
    try std.testing.expectError(error.InvalidDimensions, allocate(std.testing.allocator, request, &output));
    request.max_rows = 16;
    request.slots = 0;
    try std.testing.expectError(error.InvalidDimensions, allocate(std.testing.allocator, request, &output));
    request.slots = 11;
    request.max_rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, allocate(std.testing.allocator, request, &output));
    request.slots = 1;
    request.max_rows = std.math.maxInt(usize) / 4;
    try std.testing.expectError(error.Overflow, allocate(std.testing.allocator, request, &output));
    try std.testing.expectEqual(@as(usize, 999), output[0]);
}
