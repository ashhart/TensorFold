//! One prompt prefix, then round-major lane blocks; addresses never move, and holes are hidden by byte masks.
const std = @import("std");
pub const max_lanes = 16;

pub const Shape = struct {
    lanes: usize,
    prompt_capacity: usize,
    round_capacity: usize,
    kv_heads: usize,
    head_dim: usize,

    pub fn validate(s: Shape) !void {
        if (s.lanes == 0 or s.lanes > max_lanes or s.kv_heads == 0 or s.head_dim == 0 or s.round_capacity == 0) return error.InvalidKvShape;
        const n = try std.math.add(usize, s.prompt_capacity, try std.math.mul(usize, s.lanes, s.round_capacity));
        if (n > std.math.maxInt(u32)) return error.InvalidKvShape;
        _ = try std.math.mul(usize, n, try std.math.mul(usize, s.kv_heads, s.head_dim));
    }
    pub fn capacity(s: Shape) usize {
        return s.prompt_capacity + s.lanes * s.round_capacity;
    }
    pub fn rowElements(s: Shape) usize {
        return s.kv_heads * s.head_dim;
    }
    pub fn activeMask(s: Shape) u16 {
        return @intCast((@as(u32, 1) << @intCast(s.lanes)) - 1);
    }
};

pub const Round = struct {
    generation: u64,
    base: usize,
    active: u16,
    positions: [max_lanes]u32,
};

pub const View = struct { keys: usize, current_base: ?usize = null, generation: ?u64 = null };

pub const Cache = struct {
    gpa: std.mem.Allocator,
    shape: Shape,
    k: []u16, // bf16 [capacity,kv_heads,head_dim], already rotated keys
    v: []u16,
    tokens: []u32,
    positions: []u32,
    valid: []u8,
    prompt_len: usize = 0,
    rounds: usize = 0,
    counts: [max_lanes]u32 = @splat(0),
    generation: u64 = 0,
    pending: ?Round = null,
    filled: u16 = 0,

    pub fn init(gpa: std.mem.Allocator, shape: Shape) !Cache {
        try shape.validate();
        const count = shape.capacity();
        const elements = count * shape.rowElements();
        const k = try gpa.alloc(u16, elements);
        errdefer gpa.free(k);
        const v = try gpa.alloc(u16, elements);
        errdefer gpa.free(v);
        const tokens = try gpa.alloc(u32, count);
        errdefer gpa.free(tokens);
        const positions = try gpa.alloc(u32, count);
        errdefer gpa.free(positions);
        const valid = try gpa.alloc(u8, count);
        errdefer gpa.free(valid);
        @memset(k, 0);
        @memset(v, 0);
        @memset(tokens, 0);
        @memset(positions, 0);
        @memset(valid, 0);
        return .{ .gpa = gpa, .shape = shape, .k = k, .v = v, .tokens = tokens, .positions = positions, .valid = valid };
    }
    pub fn deinit(c: *Cache) void {
        c.gpa.free(c.k);
        c.gpa.free(c.v);
        c.gpa.free(c.tokens);
        c.gpa.free(c.positions);
        c.gpa.free(c.valid);
        c.* = undefined;
    }

    /// Prefill chunks append to one prefix. K already carries the model's RoPE; V is unchanged.
    pub fn appendPrompt(c: *Cache, tokens: []const u32, k: []const u16, v: []const u16) !void {
        if (c.pending != null or c.rounds != 0) return error.PromptAlreadySealed;
        if (tokens.len > c.shape.prompt_capacity - c.prompt_len) return error.PromptCapacity;
        const elems = try std.math.mul(usize, tokens.len, c.shape.rowElements());
        if (k.len != elems or v.len != elems) return error.InvalidKvStorage;
        try c.disjoint(std.mem.sliceAsBytes(tokens));
        try c.disjoint(std.mem.sliceAsBytes(k));
        try c.disjoint(std.mem.sliceAsBytes(v));
        const start = c.prompt_len * c.shape.rowElements();
        @memcpy(c.k[start..][0..elems], k);
        @memcpy(c.v[start..][0..elems], v);
        @memcpy(c.tokens[c.prompt_len..][0..tokens.len], tokens);
        for (c.prompt_len..c.prompt_len + tokens.len) |i| {
            c.positions[i] = @intCast(i);
            c.valid[i] = 1;
        }
        c.prompt_len += tokens.len;
    }

    pub fn position(c: *const Cache, lane: usize) !u32 {
        if (lane >= c.shape.lanes) return error.InvalidLane;
        return @intCast(c.prompt_len + c.counts[lane]);
    }
    pub fn committed(c: *const Cache) View {
        return .{ .keys = c.prompt_len + c.rounds * c.shape.lanes };
    }

    /// Reserve exactly N slots. Inactive lanes leave holes masked out of every query.
    pub fn begin(c: *Cache, active: u16) !Round {
        if (c.pending != null) return error.RoundInFlight;
        if (c.rounds == c.shape.round_capacity) return error.RoundCapacity;
        if (active == 0 or active & ~c.shape.activeMask() != 0) return error.InvalidActiveLanes;
        if (c.generation == std.math.maxInt(u64)) return error.GenerationExhausted;
        var r = Round{ .generation = c.generation + 1, .base = c.committed().keys, .active = active, .positions = @splat(0) };
        for (0..c.shape.lanes) |i| r.positions[i] = try c.position(i);
        @memset(c.valid[r.base..][0..c.shape.lanes], 0);
        c.generation = r.generation;
        c.pending = r;
        c.filled = 0;
        return r;
    }
    fn check(c: *const Cache, r: Round) !void {
        const p = c.pending orelse return error.StaleRound;
        if (!std.meta.eql(p, r)) return error.StaleRound;
    }
    pub fn slot(c: *const Cache, r: Round, lane: usize) !usize {
        try c.check(r);
        if (lane >= c.shape.lanes or r.active & (@as(u16, 1) << @intCast(lane)) == 0) return error.InvalidLane;
        return r.base + lane;
    }
    pub fn put(c: *Cache, r: Round, lane: usize, token: u32, k: []const u16, v: []const u16) !void {
        const index = try c.slot(r, lane);
        const bit = @as(u16, 1) << @intCast(lane);
        if (c.filled & bit != 0) return error.LaneAlreadyWritten;
        if (k.len != c.shape.rowElements() or v.len != k.len) return error.InvalidKvStorage;
        try c.disjoint(std.mem.sliceAsBytes(k));
        try c.disjoint(std.mem.sliceAsBytes(v));
        const start = index * k.len;
        @memcpy(c.k[start..][0..k.len], k);
        @memcpy(c.v[start..][0..v.len], v);
        c.tokens[index] = token;
        c.positions[index] = r.positions[lane];
        c.valid[index] = 1;
        c.filled |= bit;
    }
    pub fn current(c: *const Cache, r: Round) !View {
        try c.check(r);
        if (c.filled != r.active) return error.RoundNotReady;
        return .{ .keys = r.base + c.shape.lanes, .current_base = r.base, .generation = r.generation };
    }
    pub fn commit(c: *Cache, r: Round) !void {
        _ = try c.current(r);
        for (0..c.shape.lanes) |i| if (r.active & (@as(u16, 1) << @intCast(i)) != 0) {
            c.counts[i] += 1;
        };
        c.rounds += 1;
        c.pending = null;
        c.filled = 0;
    }
    pub fn abort(c: *Cache, r: Round) !void {
        try c.check(r);
        @memset(c.valid[r.base..][0..c.shape.lanes], 0);
        c.pending = null;
        c.filled = 0;
    }

    fn disjoint(c: *const Cache, data: []const u8) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(c.k), std.mem.sliceAsBytes(c.v), std.mem.sliceAsBytes(c.tokens), std.mem.sliceAsBytes(c.positions), c.valid }) |own| {
            const aa = @intFromPtr(own.ptr);
            const bb = @intFromPtr(data.ptr);
            const overlaps = if (aa <= bb) bb - aa < own.len else aa - bb < data.len;
            if (data.len != 0 and overlaps) return error.AliasedKvInput;
        }
    }

    /// Byte masks [query_row,view.keys], filled from ready slots. The caller can then hide any key per row.
    pub fn visibility(c: *const Cache, view: View, rows: usize, out: []u8) !void {
        try c.disjoint(out);
        if (rows == 0 or rows > max_lanes or out.len != try std.math.mul(usize, rows, view.keys)) return error.InvalidVisibility;
        if (view.current_base) |base| {
            const p = c.pending orelse return error.StaleRound;
            if (view.generation != p.generation or p.base != base or view.keys != base + c.shape.lanes or c.filled != p.active) return error.StaleRound;
        } else if (view.keys > c.committed().keys) return error.StaleRound;
        for (0..rows) |r| @memcpy(out[r * view.keys ..][0..view.keys], c.valid[0..view.keys]);
    }
};

test "one prompt, independent positions, earlier and current lane visibility, no prompt copies" {
    var c = try Cache.init(std.testing.allocator, .{ .lanes = 3, .prompt_capacity = 4, .round_capacity = 4, .kv_heads = 1, .head_dim = 2 });
    defer c.deinit();
    try c.appendPrompt(&.{ 10, 11 }, &.{ 1, 2, 3, 4 }, &.{ 5, 6, 7, 8 });
    try c.appendPrompt(&.{12}, &.{ 9, 10 }, &.{ 11, 12 });
    const kp = c.k.ptr;
    const vp = c.v.ptr;
    const prompt_k = try std.testing.allocator.dupe(u16, c.k[0..6]);
    defer std.testing.allocator.free(prompt_k);
    const prompt_v = try std.testing.allocator.dupe(u16, c.v[0..6]);
    defer std.testing.allocator.free(prompt_v);
    const first = try c.begin(0b111);
    try std.testing.expectEqual(@as(usize, 3), first.base);
    try std.testing.expectEqual(@as(u32, 3), first.positions[2]);
    try c.put(first, 0, 20, &.{ 13, 14 }, &.{ 15, 16 });
    try std.testing.expectError(error.RoundNotReady, c.current(first));
    try c.put(first, 1, 21, &.{ 17, 18 }, &.{ 19, 20 });
    try c.put(first, 2, 22, &.{ 21, 22 }, &.{ 23, 24 });
    var mask: [18]u8 = undefined;
    try c.visibility(try c.current(first), 3, &mask);
    for (mask) |x| try std.testing.expectEqual(@as(u8, 1), x);
    try c.commit(first);
    const second = try c.begin(0b101);
    try c.put(second, 0, 30, &.{ 25, 26 }, &.{ 27, 28 });
    try c.put(second, 2, 32, &.{ 29, 30 }, &.{ 31, 32 });
    try c.commit(second);
    const third = try c.begin(0b111);
    try std.testing.expectEqualSlices(u32, &.{ 5, 4, 5 }, third.positions[0..3]);
    for (0..3) |i| try c.put(third, i, @intCast(40 + i), &.{ 33, 34 }, &.{ 35, 36 });
    var later: [36]u8 = undefined;
    try c.visibility(try c.current(third), 3, &later);
    for (0..3) |r| {
        try std.testing.expectEqual(@as(u8, 0), later[r * 12 + 7]);
        try std.testing.expectEqual(@as(u8, 1), later[r * 12 + 11]);
    }
    try c.commit(third);
    try std.testing.expectEqualSlices(u16, prompt_k, c.k[0..6]);
    try std.testing.expectEqualSlices(u16, prompt_v, c.v[0..6]);
    try std.testing.expect(kp == c.k.ptr and vp == c.v.ptr);
    try std.testing.expectError(error.StaleRound, c.put(first, 0, 99, &.{ 0, 0 }, &.{ 0, 0 }));
    try std.testing.expectError(error.PromptAlreadySealed, c.appendPrompt(&.{99}, &.{ 0, 0 }, &.{ 0, 0 }));
}

test "an incomplete or aborted round cannot publish; a reused base has a new generation" {
    var c = try Cache.init(std.testing.allocator, .{ .lanes = 2, .prompt_capacity = 0, .round_capacity = 1, .kv_heads = 1, .head_dim = 1 });
    defer c.deinit();
    try std.testing.expectError(error.InvalidActiveLanes, c.begin(4));
    const old = try c.begin(3);
    try c.put(old, 0, 1, &.{1}, &.{2});
    try std.testing.expectError(error.RoundNotReady, c.commit(old));
    try std.testing.expectError(error.LaneAlreadyWritten, c.put(old, 0, 1, &.{1}, &.{2}));
    try c.abort(old);
    const next = try c.begin(1);
    try std.testing.expectEqual(old.base, next.base);
    try std.testing.expectError(error.StaleRound, c.current(old));
    try c.put(next, 0, 3, &.{3}, &.{4});
    try c.commit(next);
    try std.testing.expectError(error.RoundCapacity, c.begin(1));
}

test "visibility leases reject an aborted round even when its base is reused" {
    var c = try Cache.init(std.testing.allocator, .{ .lanes = 1, .prompt_capacity = 1, .round_capacity = 2, .kv_heads = 1, .head_dim = 1 });
    defer c.deinit();
    try c.appendPrompt(&.{1}, &.{2}, &.{3});
    const prior = try c.begin(1);
    try c.put(prior, 0, 2, &.{4}, &.{5});
    const stale = try c.current(prior);
    try c.abort(prior);
    const next = try c.begin(1);
    try c.put(next, 0, 3, &.{6}, &.{7});
    var mask: [2]u8 = undefined;
    try std.testing.expectError(error.StaleRound, c.visibility(stale, 1, &mask));
    try c.visibility(try c.current(next), 1, &mask);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1 }, &mask);
    try std.testing.expectError(error.AliasedKvInput, c.visibility(try c.current(next), 1, c.valid[0..2]));
    try c.commit(next);
    const third = try c.begin(1);
    try std.testing.expectError(error.AliasedKvInput, c.put(third, 0, 4, c.k[0..1], &.{8}));
    try std.testing.expectEqual(@as(u16, 0), c.filled);
}
