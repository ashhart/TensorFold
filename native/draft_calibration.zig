const std = @import("std");
const depth_edges = [_]f64{ 0, 1, 2, 3, 5, 8 };
const score_edges = [_]f64{ -6, -4.5, -3.5, -2.8, -2.2, -1.7, -1.3, -1, -0.75, -0.5, -0.3, -0.15, -0.05 };

pub const Table = struct {
    depth_edges: []const f64,
    score_edges: []const f64,
    table: []const []const f64,

    pub fn validate(t: Table) !void {
        if (t.depth_edges.len == 0 or t.table.len != t.depth_edges.len) return error.InvalidCalibration;
        for ([_][]const f64{ t.depth_edges, t.score_edges }) |edges| for (edges, 0..) |edge, i| {
            if (!std.math.isFinite(edge) or (i > 0 and edge <= edges[i - 1])) return error.InvalidCalibration;
        };
        for (t.table) |row| {
            if (row.len != t.score_edges.len + 1) return error.InvalidCalibration;
            for (row) |p| if (!std.math.isFinite(p) or p < 0 or p > 1) return error.InvalidCalibration;
        }
    }

    pub fn probability(t: Table, depth: i32, score: f64) f64 {
        return t.table[depthBin(t.depth_edges, depth)][scoreBin(t.score_edges, score)];
    }
};

pub const File = struct {
    source: std.json.Value = .null,
    tables: std.json.ArrayHashMap(Table),
};

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(File) {
    const parsed = try std.json.parseFromSlice(File, a, bytes, .{ .allocate = .alloc_always });
    errdefer parsed.deinit();
    for (parsed.value.tables.map.values()) |table| try table.validate();
    return parsed;
}

pub fn load(a: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(File) {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 * 1024 * 1024));
    defer a.free(bytes);
    return parse(a, bytes);
}

fn depthBin(edges: []const f64, depth: i32) usize {
    var row: usize = 0;
    for (edges, 0..) |edge, i| {
        if (@as(f64, @floatFromInt(depth)) < edge) break;
        row = i;
    }
    return row;
}

fn scoreBin(edges: []const f64, score: f64) usize {
    for (edges, 0..) |edge, i| if (!(edge < score)) return i;
    return edges.len;
}

pub const Sample = struct { depth: i32, score: f64, landed: bool };

// Returned table storage belongs to the caller's arena.
pub fn fit(a: std.mem.Allocator, samples: []const Sample, depths: []const f64, scores: []const f64) !Table {
    const rows = try a.alloc([]const f64, depths.len);
    const cols = try std.math.add(usize, scores.len, 1);
    for (rows) |*row| {
        const values = try a.alloc(f64, cols);
        @memset(values, 0);
        row.* = values;
    }
    const result = Table{ .depth_edges = try a.dupe(f64, depths), .score_edges = try a.dupe(f64, scores), .table = rows };
    try result.validate();
    const cells = try std.math.mul(usize, depths.len, cols);
    const hits = try a.alloc(f64, cells);
    const counts = try a.alloc(f64, cells);
    @memset(hits, 0);
    @memset(counts, 0);
    for (samples) |sample| {
        if (!std.math.isFinite(sample.score)) return error.InvalidCalibrationSample;
        const index = depthBin(depths, sample.depth) * cols + scoreBin(scores, sample.score);
        hits[index] += @floatFromInt(@intFromBool(sample.landed));
        counts[index] += 1;
    }
    const Block = struct { value: f64, weight: f64, count: usize };
    const blocks = try a.alloc(Block, cols);
    for (rows, 0..) |row, r| {
        var len: usize = 0;
        for (0..cols) |c| {
            const index = r * cols + c;
            const weight = counts[index] + 1;
            blocks[len] = .{ .value = (hits[index] + 0.5) / weight, .weight = weight, .count = 1 };
            len += 1;
            while (len > 1 and blocks[len - 2].value > blocks[len - 1].value) {
                const right = blocks[len - 1];
                const left = &blocks[len - 2];
                left.value = (left.value * left.weight + right.value * right.weight) / (left.weight + right.weight);
                left.weight += right.weight;
                left.count += right.count;
                len -= 1;
            }
        }
        var col: usize = 0;
        for (blocks[0..len]) |block| {
            @memset(@constCast(row[col..][0..block.count]), block.value);
            col += block.count;
        }
    }
    return result;
}

pub const Ranked = struct { tokens: []i32, parents: []i32, scores: []f64, probabilities: []f64 };

// Returned arrays belong to the caller's arena. Equal chances retain upstream's original-index tie break.
pub fn rank(a: std.mem.Allocator, tokens: []const i32, parents: []const i32, scores: []const f64, table: ?Table) !Ranked {
    const n = tokens.len;
    if (parents.len != n or scores.len != n) return error.InvalidDraftTree;
    const depths = try a.alloc(i32, n);
    const chances = try a.alloc(f64, n);
    const place = try a.alloc(i32, n);
    @memset(place, -1);
    for (parents, scores, 0..) |parent, score, i| {
        if (parent < -1 or parent >= @as(i64, @intCast(i)) or !std.math.isFinite(score) or score > 0) return error.InvalidDraftTree;
        depths[i] = if (parent < 0) 0 else depths[@intCast(parent)] + 1;
        const chance = if (table) |t| t.probability(depths[i], score) else @exp(score);
        chances[i] = if (parent < 0) chance else @min(chance, chances[@intCast(parent)]);
    }
    const result = Ranked{ .tokens = try a.alloc(i32, n), .parents = try a.alloc(i32, n), .scores = try a.alloc(f64, n), .probabilities = try a.alloc(f64, n) };
    for (0..n) |out| {
        var best: ?usize = null;
        for (parents, 0..) |parent, i| {
            if (place[i] >= 0 or (parent >= 0 and place[@intCast(parent)] < 0)) continue;
            if (best == null or chances[i] > chances[best.?]) best = i;
        }
        const i = best orelse return error.InvalidDraftTree;
        place[i] = @intCast(out);
        result.tokens[out] = tokens[i];
        result.parents[out] = if (parents[i] < 0) -1 else place[@intCast(parents[i])];
        result.scores[out] = scores[i];
        result.probabilities[out] = chances[i];
    }
    return result;
}

pub fn encode(a: std.mem.Allocator, data: File) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    var rounded = File{ .source = data.source, .tables = .{} };
    for (data.tables.map.keys(), data.tables.map.values()) |name, table| {
        try table.validate();
        const rows = try scratch.alloc([]const f64, table.table.len);
        for (table.table, rows) |row, *out| {
            const values = try scratch.alloc(f64, row.len);
            for (row, values) |value, *p| p.* = roundProbability(value);
            out.* = values;
        }
        try rounded.tables.map.put(scratch, name, .{ .depth_edges = table.depth_edges, .score_edges = table.score_edges, .table = rows });
    }
    return std.json.Stringify.valueAlloc(a, rounded, .{ .whitespace = .indent_2 });
}

fn roundProbability(value: f64) f64 {
    // Python round(p, 4): round the exact binary value to decimal with ties to even.
    // Scaling in f64 first would turn values adjacent to a decimal midpoint into ties.
    const bits: u64 = @bitCast(value);
    const exponent: i32 = @intCast((bits >> 52) & 0x7ff);
    const fraction = bits & ((@as(u64, 1) << 52) - 1);
    const mantissa: u128 = fraction | (if (exponent == 0) @as(u64, 0) else @as(u64, 1) << 52);
    const shift: i32 = if (exponent == 0) 1074 else 1075 - exponent;
    if (shift >= 128) return 0;
    const width: u7 = @intCast(shift);
    const scaled = mantissa * 10000;
    const denominator = @as(u128, 1) << width;
    var rounded = scaled >> width;
    const remainder = scaled & (denominator - 1);
    const half = denominator >> 1;
    if (remainder > half or (remainder == half and rounded & 1 != 0)) rounded += 1;
    return @as(f64, @floatFromInt(rounded)) / 10000;
}

pub fn fitFile(io: std.Io, input: []const u8, output: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Input = struct {
        source: std.json.Value = .null,
        depth_edges: []const f64 = &depth_edges,
        score_edges: []const f64 = &score_edges,
        samples: std.json.ArrayHashMap([]const Sample),
    };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, input, a, .limited(256 * 1024 * 1024));
    const parsed = try std.json.parseFromSlice(Input, a, bytes, .{});
    var data = File{ .source = parsed.value.source, .tables = .{} };
    for (parsed.value.samples.map.keys(), parsed.value.samples.map.values()) |name, samples|
        try data.tables.map.put(a, name, try fit(a, samples, parsed.value.depth_edges, parsed.value.score_edges));
    const encoded = try encode(a, data);
    const file = try std.Io.Dir.cwd().createFile(io, output, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, encoded);
    try file.writeStreamingAll(io, "\n");
}

pub fn check(io: std.Io, path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Query = struct { depth: i32, score: f64, expected: f64 };
    const Case = struct { samples: []const Sample, expected: Table, serialized: Table, queries: []const Query };
    const Tree = struct { tokens: []const i32, parents: []const i32, scores: []const f64, table: ?Table, expected: struct { tokens: []const i32, parents: []const i32, probabilities: []const f64 } };
    const Fixture = struct { fits: []const Case, trees: []const Tree, shipped: File, rounding: []const struct { value: f64, expected: f64 } };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    const fixture = (try std.json.parseFromSlice(Fixture, a, bytes, .{})).value;
    for (fixture.rounding) |case| try std.testing.expectEqual(case.expected, roundProbability(case.value));
    var queries: usize = 0;
    for (fixture.fits) |case| {
        const fitted = try fit(a, case.samples, case.expected.depth_edges, case.expected.score_edges);
        for (fitted.table, case.expected.table) |actual, expected| try std.testing.expectEqualSlices(f64, expected, actual);
        for (case.queries) |query| try std.testing.expectEqual(query.expected, fitted.probability(query.depth, query.score));
        queries += case.queries.len;
        var data = File{ .tables = .{} };
        try data.tables.map.put(a, "sampled", fitted);
        const saved = try parse(a, try encode(a, data));
        defer saved.deinit();
        for (saved.value.tables.map.get("sampled").?.table, case.serialized.table) |actual, expected| try std.testing.expectEqualSlices(f64, expected, actual);
    }
    for (fixture.trees) |tree| {
        if (tree.table) |table| try table.validate();
        const ranked = try rank(a, tree.tokens, tree.parents, tree.scores, tree.table);
        try std.testing.expectEqualSlices(i32, tree.expected.tokens, ranked.tokens);
        try std.testing.expectEqualSlices(i32, tree.expected.parents, ranked.parents);
        for (ranked.probabilities, tree.expected.probabilities) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 2e-16);
    }
    const shipped = try parse(a, @import("native_runtime").dflash_calibration);
    defer shipped.deinit();
    for (fixture.shipped.tables.map.keys(), fixture.shipped.tables.map.values()) |name, table| {
        const embedded = shipped.value.tables.map.get(name) orelse return error.MissingCalibrationRegime;
        try std.testing.expectEqualSlices(f64, table.depth_edges, embedded.depth_edges);
        try std.testing.expectEqualSlices(f64, table.score_edges, embedded.score_edges);
        for (table.table, embedded.table) |expected, actual| try std.testing.expectEqualSlices(f64, expected, actual);
    }
    std.debug.print("PASS: {d} upstream fitted calibration tables, {d} boundary queries, {d} ranked trees, {d} decimal rounding boundaries and shipped tables\n", .{ fixture.fits.len, queries, fixture.trees.len, fixture.rounding.len });
}

test "calibration rejects malformed tables and non-topological draft trees" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "{\"tables\":{\"x\":{\"depth_edges\":[],\"score_edges\":[],\"table\":[]}}}",
        "{\"tables\":{\"x\":{\"depth_edges\":[0],\"score_edges\":[-1],\"table\":[[0.5]]}}}",
        "{\"tables\":{\"x\":{\"depth_edges\":[0],\"score_edges\":[],\"table\":[[1.1]]}}}",
        "{\"tables\":{\"x\":{\"depth_edges\":[0,0],\"score_edges\":[],\"table\":[[0.5],[0.5]]}}}",
    }) |bytes| try std.testing.expectError(error.InvalidCalibration, parse(a, bytes));
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    for ([_]i32{ -2, 0, 1 }) |parent| try std.testing.expectError(error.InvalidDraftTree, rank(arena.allocator(), &.{42}, &.{parent}, &.{-1}, null));
    try std.testing.expectError(error.InvalidDraftTree, rank(arena.allocator(), &.{42}, &.{-1}, &.{std.math.nan(f64)}, null));
}
