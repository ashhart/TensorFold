//! Independent sort-based QSA membership, block-order emission and dense-row oracle.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec(padded: bool) c.Spec {
    return .{
        .name = if (padded) "cuda fn_qsa_select padded scores" else "cuda fn_qsa_select packed scores",
        .accum = .none,
        .depth = depth,
        .out = .raw,
        .nan = .{ .stated = "float total order includes signed NaN payloads and signed zero; ties choose earlier blocks" },
        // rows = query rows, n = first row's complete blocks, k = its tail length plus one.
        .tiles = .{ .{ .size = 32 }, .{ .size = 1024, .min = 0 }, .{ .size = 4 } },
        .served = &.{
            .{ .rows = 8, .n = 1025, .k = 1 },
            .{ .rows = 1, .n = 0, .k = 2 },
            .{ .rows = 4, .n = 0, .k = 4 },
            .{ .rows = 8, .n = 512, .k = 1 },
            .{ .rows = 1, .n = 511, .k = 4 },
            .{ .rows = 1, .n = 512, .k = 4 },
            .{ .rows = 1, .n = 513, .k = 1 },
            .{ .rows = 256, .n = 513, .k = 4 },
            .{ .rows = 3, .n = 1023, .k = 2 },
            .{ .rows = 5, .n = 1024, .k = 3 },
            .{ .rows = 7, .n = 1025, .k = 4 },
            .{ .rows = 5, .n = 4097, .k = 2 },
            .{ .rows = 1, .n = 8192, .k = 4 },
        },
    };
}

pub const Layout = struct {
    rows: usize,
    nb: usize,
    pos0: i32,
    top: usize = 512,
    ratio: usize = 4,
    idw: usize = 2052,

    pub fn valid(l: Layout) bool {
        if (l.rows == 0 or l.rows > 256 or l.nb == 0 or l.nb > 8263 or l.pos0 < 0) return false;
        if (l.top == 0 or l.top > 512 or l.ratio == 0 or l.ratio > 32) return false;
        if (l.idw < l.top * l.ratio + l.ratio - 1 or l.idw > 16384) return false;
        const last = @as(usize, @intCast(l.pos0)) + l.rows;
        return last <= std.math.maxInt(i32) and last / l.ratio <= l.nb;
    }
};

/// Fixture bounds only: the production launcher is unchanged.
pub fn layout(s: c.Shape, padded: bool) ?Layout {
    if (s.rows == 0 or s.rows > 256 or s.n > 8192 or s.k == 0 or s.k > 4) return null;
    const end = s.n * 4 + s.k - 1;
    if (end == 0) return null;
    const l: Layout = .{
        .rows = s.rows,
        .nb = @max(1, (end + s.rows - 1) / 4) + (if (padded) @as(usize, 7) else 0),
        .pos0 = @intCast(end - 1),
    };
    return if (l.valid()) l else null;
}

const Entry = struct { score: f32, index: usize };

/// Numeric comparisons plus explicit IEEE sign/NaN payload policy; no radix key transform or scan is reused.
fn before(_: void, a: Entry, b: Entry) bool {
    const an = std.math.isNan(a.score);
    const bn = std.math.isNan(b.score);
    const ab: u32 = @bitCast(a.score);
    const bb: u32 = @bitCast(b.score);
    const negative_a = std.math.signbit(a.score);
    const negative_b = std.math.signbit(b.score);
    if (an or bn) {
        if (an and bn) {
            if (negative_a != negative_b) return !negative_a;
            const ap = ab & 0x7fffff;
            const bp = bb & 0x7fffff;
            if (ap != bp) return if (negative_a) ap < bp else ap > bp;
        } else return if (an) !negative_a else negative_b;
    } else {
        if (a.score != b.score) return a.score > b.score;
        if (negative_a != negative_b) return !negative_a;
    }
    return a.index < b.index;
}

pub const Result = struct {
    ids: []i32,
    nk: []i32,
    sparse: []i32,

    pub fn deinit(r: Result, a: std.mem.Allocator) void {
        a.free(r.ids);
        a.free(r.nk);
        a.free(r.sparse);
    }

    pub fn bytes(r: Result, a: std.mem.Allocator) ![]u8 {
        const out = try a.alloc(u8, (r.ids.len + r.nk.len + r.sparse.len) * 4);
        var i: usize = 0;
        for ([_][]const i32{ r.ids, r.nk, r.sparse }) |part| for (part) |v| {
            std.mem.writeInt(i32, out[i..][0..4], v, .little);
            i += 4;
        };
        return out;
    }
};

/// Choose membership by a full sort, then enumerate selected blocks in original order and append live tail tokens.
pub fn select(a: std.mem.Allocator, scores: []const f32, l: Layout) !Result {
    if (!l.valid() or scores.len != l.rows * l.nb) return error.InvalidFixture;
    const ids = try a.alloc(i32, l.rows * l.idw);
    errdefer a.free(ids);
    const nk = try a.alloc(i32, l.rows);
    errdefer a.free(nk);
    const sparse = try a.alloc(i32, l.rows);
    errdefer a.free(sparse);
    @memset(ids, -1);
    const entries = try a.alloc(Entry, l.nb);
    defer a.free(entries);
    const chosen = try a.alloc(bool, l.nb);
    defer a.free(chosen);
    for (0..l.rows) |r| {
        const end = @as(usize, @intCast(l.pos0)) + r + 1;
        const complete = end / l.ratio;
        if (complete <= l.top) {
            nk[r] = @intCast(end);
            sparse[r] = 0;
            continue;
        }
        for (entries[0..complete], scores[r * l.nb ..][0..complete], 0..) |*entry, value, b|
            entry.* = .{ .score = value, .index = b };
        std.mem.sort(Entry, entries[0..complete], {}, before);
        @memset(chosen, false);
        for (entries[0..l.top]) |entry| chosen[entry.index] = true;
        var at = r * l.idw;
        for (chosen[0..complete], 0..) |yes, b| if (yes) {
            for (0..l.ratio) |j| {
                ids[at] = @intCast(b * l.ratio + j);
                at += 1;
            }
        };
        for (complete * l.ratio..end) |token| {
            ids[at] = @intCast(token);
            at += 1;
        }
        nk[r] = @intCast(at - r * l.idw);
        sparse[r] = 1;
    }
    return .{ .ids = ids, .nk = nk, .sparse = sparse };
}

pub const special_bits = [_]u32{
    0xffc00007, 0xff800001, 0xff800000, 0x80000000, 0x00000000,
    0x7f800000, 0x7f800001, 0x7fc00001, 0x7fc00007,
};

pub fn fill(scores: []f32, l: Layout, kind: g.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    // Discrete random scores deliberately create ties on both sides of the selection boundary.
    for (0..l.rows) |r| {
        const complete = (@as(usize, @intCast(l.pos0)) + r + 1) / l.ratio;
        for (scores[r * l.nb ..][0..l.nb], 0..) |*v, b| {
            v.* = if (b < complete) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 33))) - 16) else @bitCast(@as(u32, 0x7fffffff));
            if (kind == .nonfinite and b < complete) v.* = switch (r % 4) {
                0 => 0,
                1 => @bitCast(@as(u32, 0x7fc00000) + @as(u32, @intCast(b % 1024 + 1))),
                2 => @bitCast(@as(u32, 0xffc00000) + @as(u32, @intCast(b % 1024 + 1))),
                else => if (b % 2 == 0) 0.0 else -0.0,
            };
        }
        if (kind == .nonfinite) for (special_bits, 0..) |bits, i| {
            if (i < complete) scores[r * l.nb + i] = @bitCast(bits);
        };
    }
}

pub fn compare(want: []const u8, got: []const u8) ?[]const u8 {
    return if (std.mem.eql(u8, want, got)) null else "selected ids, row counts, sparse flags or untouched padding differ";
}

test "independent score order specifies infinities, signed zeros, signed NaNs and payloads" {
    var entries: [special_bits.len]Entry = undefined;
    for (&entries, special_bits, 0..) |*entry, bits, i| entry.* = .{ .score = @bitCast(bits), .index = i };
    std.mem.sort(Entry, &entries, {}, before);
    for (entries, 0..) |entry, i| try std.testing.expectEqual(special_bits.len - 1 - i, entry.index);
}

test "membership ties choose earliest blocks and output follows block order rather than rank order" {
    const a = std.testing.allocator;
    const l: Layout = .{ .rows = 1, .nb = 5, .pos0 = 6, .top = 2, .ratio = 2, .idw = 6 };
    const result = try select(a, &.{ 1, 9, 9, 999, 999 }, l);
    defer result.deinit(a);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4, 5, 6, -1 }, result.ids);
    try std.testing.expectEqualSlices(i32, &.{5}, result.nk);
    try std.testing.expectEqualSlices(i32, &.{1}, result.sparse);
    const ranked = try select(a, &.{ 8, 0, 9, 999, 999 }, l);
    defer ranked.deinit(a);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 4, 5, 6, -1 }, ranked.ids);
}

test "dense-to-sparse transition leaves dense ids untouched and excludes future poisoned scores" {
    const a = std.testing.allocator;
    const l: Layout = .{ .rows = 3, .nb = 6, .pos0 = 4, .top = 2, .ratio = 2, .idw = 6 };
    const scores = [_]f32{ 9, 8, 999, 999, 999, 999, 7, 8, 9, 999, 999, 999, 9, 8, 7, 999, 999, 999 };
    const result = try select(a, &scores, l);
    defer result.deinit(a);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, -1, -1, -1, -1 }, result.ids[0..6]);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4, 5, -1, -1 }, result.ids[6..12]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 6, -1 }, result.ids[12..]);
    try std.testing.expectEqualSlices(i32, &.{ 5, 4, 5 }, result.nk);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 1 }, result.sparse);
}

test "raw comparison catches omitted tails, wrong tie ids, unused padding and flag corruption" {
    const a = std.testing.allocator;
    const l: Layout = .{ .rows = 1, .nb = 3, .pos0 = 6, .top = 2, .ratio = 2, .idw = 6 };
    const result = try select(a, &.{ 1, 1, 1 }, l);
    defer result.deinit(a);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 6, -1 }, result.ids);
    const want = try result.bytes(a);
    defer a.free(want);
    const broken = try a.dupe(u8, want);
    defer a.free(broken);
    for ([_]usize{ 0, 16, 20, 24, 28 }) |byte| {
        @memcpy(broken, want);
        broken[byte] ^= 1;
        try std.testing.expect(compare(want, broken) != null);
    }
    try std.testing.expect(compare(want, want) == null);
}

test "nonfinite membership places signed NaNs and signed zero on the correct side of a cutoff" {
    const a = std.testing.allocator;
    const l: Layout = .{ .rows = 1, .nb = 9, .pos0 = 8, .top = 2, .ratio = 1, .idw = 4 };
    var scores: [9]f32 = undefined;
    for (&scores, special_bits) |*value, bits| value.* = @bitCast(bits);
    const result = try select(a, &scores, l);
    defer result.deinit(a);
    try std.testing.expectEqualSlices(i32, &.{ 7, 8, -1, -1 }, result.ids);
    const zeros = try select(a, &.{ -0.0, 0.0 }, .{ .rows = 1, .nb = 2, .pos0 = 1, .top = 1, .ratio = 1, .idw = 2 });
    defer zeros.deinit(a);
    try std.testing.expectEqualSlices(i32, &.{ 1, -1 }, zeros.ids);
}

test "finite random sort membership agrees with an independent quadratic outranker count" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 3, 4, 17, 33, 65 }) |blocks| {
        const l: Layout = .{ .rows = 3, .nb = blocks + 8, .pos0 = @intCast(blocks * 2 - 1), .top = 3, .ratio = 2, .idw = 9 };
        const scores = try a.alloc(f32, l.rows * l.nb);
        defer a.free(scores);
        fill(scores, l, .random, blocks);
        const result = try select(a, scores, l);
        defer result.deinit(a);
        for (0..l.rows) |r| {
            const end = @as(usize, @intCast(l.pos0)) + r + 1;
            const complete = end / l.ratio;
            const ids = result.ids[r * l.idw ..][0..l.idw];
            if (complete <= l.top) {
                try std.testing.expectEqual(@as(i32, 0), result.sparse[r]);
                for (ids) |id| try std.testing.expectEqual(@as(i32, -1), id);
                continue;
            }
            var at: usize = 0;
            const row = scores[r * l.nb ..][0..complete];
            for (row, 0..) |value, b| {
                var better: usize = 0;
                for (row, 0..) |other, j| if (other > value or (other == value and j < b)) {
                    better += 1;
                };
                if (better < l.top) for (0..l.ratio) |j| {
                    try std.testing.expectEqual(@as(i32, @intCast(b * l.ratio + j)), ids[at]);
                    at += 1;
                };
            }
            for (complete * l.ratio..end) |token| {
                try std.testing.expectEqual(@as(i32, @intCast(token)), ids[at]);
                at += 1;
            }
            try std.testing.expectEqual(@as(i32, @intCast(at)), result.nk[r]);
            for (ids[at..]) |id| try std.testing.expectEqual(@as(i32, -1), id);
        }
    }
}

test "fixture domains retain all served and valid tile edges and explicitly refuse unsafe layouts" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |padded| {
        const s = spec(padded);
        for (s.served) |shape| try std.testing.expect(layout(shape, padded) != null);
        const edges = try c.shapes(a, s);
        defer a.free(edges);
        for (edges) |shape| if (shape.k <= 4 and shape.n * 4 + shape.k > 1)
            try std.testing.expect(layout(shape, padded) != null);
    }
    try std.testing.expect(layout(.{ .rows = 1, .n = 1, .k = 5 }, false) == null);
    try std.testing.expect(!Layout.valid(.{ .rows = 1, .nb = 3, .pos0 = -1 }));
    try std.testing.expect(!Layout.valid(.{ .rows = 1, .nb = 3, .pos0 = 16 }));
    try std.testing.expect(!Layout.valid(.{ .rows = 1, .nb = 3, .pos0 = 6, .ratio = 0 }));
    try std.testing.expect(!Layout.valid(.{ .rows = 1, .nb = 3, .pos0 = 6, .idw = 2048 }));
}

test "directed positive NaN payload cutoff spans two blocks and seeded inputs repeat exactly" {
    const a = std.testing.allocator;
    const l = layout(.{ .rows = 8, .n = 1025, .k = 1 }, true).?;
    const scores = try a.alloc(f32, l.rows * l.nb);
    defer a.free(scores);
    const repeated = try a.alloc(f32, scores.len);
    defer a.free(repeated);
    fill(scores, l, .nonfinite, 91);
    for (0..5) |_| {
        fill(repeated, l, .nonfinite, 91);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(scores), std.mem.sliceAsBytes(repeated));
    }
    const result = try select(a, scores, l);
    defer result.deinit(a);
    // Row 1's 512 largest positive NaN payloads are blocks 512 through 1023.
    for (0..512) |b| for (0..4) |j|
        try std.testing.expectEqual(@as(i32, @intCast((512 + b) * 4 + j)), result.ids[l.idw + b * 4 + j]);
    try std.testing.expectEqual(@as(i32, 4100), result.ids[l.idw + 2048]);
    try std.testing.expectEqual(@as(i32, 2049), result.nk[1]);
    for (result.ids[l.idw + 2049 .. 2 * l.idw]) |id| try std.testing.expectEqual(@as(i32, -1), id);
}
