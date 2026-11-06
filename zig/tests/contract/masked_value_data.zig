//! Isolated exceptional V: finite zero queries/keys, ordinary values, and one later key per fixture.
const std = @import("std");
const ct = @import("contract");
const c = ct.spec;

pub const Poison = enum { inf, nan };
pub const Layout = enum { token, head };
pub const Geometry = struct {
    heads: usize,
    kv: usize,
    dim: usize,
    cap: usize = 2048,
    first: usize = 16,
    window: usize = 0,
    ring: usize = 0,
    cache: Layout = .token,
    query: Layout = .token,
    output: Layout = .token,
    gated: bool = false,
    latent: bool = false,
    poison_key: ?usize = null,
    skip_zero_first: bool = false,
    selected_prefix: ?usize = null,
    selected_shift_first: usize = 0,
    sink: bool = false,
};

pub const Data = struct {
    geo: Geometry,
    rows: usize,
    poisoned: bool,
    poison: Poison,
    q: []u16,
    k: []u16,
    v: []u16,

    pub fn init(a: std.mem.Allocator, geo: Geometry, rows: usize, poisoned: bool, poison: Poison) !Data {
        const d: Data = .{ .geo = geo, .rows = rows, .poisoned = poisoned, .poison = poison, .q = try a.alloc(u16, @max(64, rows) * geo.heads * geo.dim), .k = try a.alloc(u16, geo.cap * geo.kv * geo.dim), .v = try a.alloc(u16, geo.cap * geo.kv * geo.dim) };
        @memset(d.q, 0);
        @memset(d.k, 0);
        @memset(d.v, c.toBf16(1));
        if (poisoned) {
            const at = d.cacheAt(geo.poison_key orelse geo.first + 1, 0, 7);
            d.v[at] = if (poison == .inf) 0x7f80 else 0x7fc0;
            if (geo.latent) d.k[at] = d.v[at];
        }
        return d;
    }

    pub fn cacheAt(d: Data, key: usize, head: usize, col: usize) usize {
        const g = d.geo;
        const slot = if (g.ring == 0) key else key % g.ring;
        return if (g.cache == .token) (slot * g.kv + head) * g.dim + col else (head * g.cap + slot) * g.dim + col;
    }

    pub fn outputAt(d: Data, row: usize, head: usize, col: usize) usize {
        const g = d.geo;
        return if (g.output == .token) (row * g.heads + head) * g.dim + col else (head * d.rows + row) * g.dim + col;
    }

    pub fn count(d: Data) usize {
        return d.rows * d.geo.heads * d.geo.dim;
    }

    /// Binary64 softmax at the actual zero scores, excluding unseen keys before either K or V is read.
    pub fn reference(d: Data, want: []f64, mag: []f64) void {
        const g = d.geo;
        for (0..d.rows) |r| {
            const hi = g.first + r + 1;
            const skip = if (r == 0) @max(if (g.skip_zero_first) @as(usize, 1) else 0, g.selected_shift_first) else 0;
            const lo = @max(if (g.window > 0) hi -| g.window else 0, skip);
            for (0..g.heads) |h| {
                var ordinary: f64 = 0;
                var exceptional: f64 = 0;
                var total: f64 = 0;
                for (lo..hi) |key| {
                    if (g.selected_prefix) |prefix| if (key >= prefix + (if (r == 0) g.selected_shift_first else 0) and key < (hi - 1) / 8 * 8) continue;
                    const kh = h / (g.heads / g.kv);
                    const score: f64 = if (g.latent and kh == 0 and d.poisoned and key == (g.poison_key orelse g.first + 1))
                        0 * @as(f64, c.fromBf16(d.k[d.cacheAt(key, kh, 7)]))
                    else
                        0;
                    const p = @exp(score);
                    ordinary += p;
                    exceptional += p * @as(f64, c.fromBf16(d.v[d.cacheAt(key, kh, 7)]));
                    total += p;
                }
                if (g.sink) total += @exp(@as(f64, 0));
                for (0..g.dim) |col| {
                    const i = d.outputAt(r, h, col);
                    const x = (if (col == 7) exceptional else ordinary) / total;
                    want[i] = if (g.gated) @as(f64, c.fromBf16(c.toBf16(@floatCast(x)))) * 0.5 else x;
                    mag[i] = @abs(want[i]);
                }
            }
        }
    }
};

test "masked value: hidden Inf and NaN do not enter the binary64 formula" {
    for (std.enums.values(Poison)) |poison| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const d = try Data.init(a, .{ .heads = 4, .kv = 2, .dim = 32, .cap = 64 }, 5, true, poison);
        const want = try a.alloc(f64, d.count());
        const mag = try a.alloc(f64, want.len);
        d.reference(want, mag);
        try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(0, 0, 7)]);
        try std.testing.expectEqual(if (poison == .inf) c.Class.pos_inf else c.Class.nan, c.classOf(want[d.outputAt(1, 0, 7)]));
        try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(1, 2, 7)]);
    }
}

test "masked value: aliased latent keys retain the visible score exception" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 1, .dim = 32, .cap = 64, .latent = true }, 5, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[0]);
    try std.testing.expect(std.math.isNan(want[d.outputAt(1, 0, 0)]));
}

test "masked value: ring wrap uses absolute visibility and physical slots" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 2, .dim = 32, .cap = 32, .first = 31, .window = 8, .ring = 32 }, 5, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(0, 0, 7)]);
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[d.outputAt(1, 0, 7)]));
    try std.testing.expectEqual(d.cacheAt(0, 0, 7), d.cacheAt(32, 0, 7));
}

test "masked value: sparse padding omits key zero only where the real list does" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 1, .dim = 32, .cap = 64, .latent = true, .poison_key = 0, .skip_zero_first = true }, 5, true, .nan);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[0]);
    try std.testing.expect(std.math.isNan(want[d.outputAt(1, 0, 0)]));
}

test "masked value: sparse own region stops exposing an unselected prior region" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 2, .dim = 32, .cap = 8192, .first = 4104, .selected_prefix = 4096 }, 16, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(0, 0, 7)]);
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[d.outputAt(1, 0, 7)]));
    try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(8, 0, 7)]);
}

test "masked value: a zero-score sink adds to the denominator but has no value" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 1, .dim = 32, .cap = 64, .sink = true }, 5, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 17.0 / 18.0), want[0]);
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[d.outputAt(1, 0, 7)]));
}
