//! GatedNorm's host side: the launch checks, the bf16 rounding and the reference's steps.
const std = @import("std");
const gn = @import("gated_norm.zig");
const config = @import("config.zig");
const norms = @import("norms.zig");
const ct = @import("config_test.zig");

test "the kernel covers every GatedNorm width of both releases, and refuses what it does not" {
    for ([_]u32{ 512, 1536, 2560, 7168 }) |w| try gn.check(64, w, w, w);
    try std.testing.expectError(error.Width, gn.check(1, 0, 0, 0));
    try std.testing.expectError(error.Width, gn.check(1, gn.max_width + 1, gn.max_width + 1, gn.max_width + 1));
    try std.testing.expectError(error.Stride, gn.check(2, 2560, 2559, 2560));
    try std.testing.expectError(error.Stride, gn.check(2, 2560, 2560, 100));
    try gn.check(1, 2560, 0, 0); // one row reads no stride
    try std.testing.expectError(error.Rows, gn.check(@as(usize, std.math.maxInt(i32)) + 1, 2560, 2560, 2560));
    // the rows' shared bytes and the kernel's 96 static bytes fit the 48 KiB every SM grants without opting in
    try std.testing.expect(gn.sharedBytes(gn.max_width) + 96 <= 48 * 1024);
}

test "bf16: nearest, ties to even, NaN kept" {
    try std.testing.expectEqual(@as(f64, 1), gn.bf16(1 + 0x1p-9)); // a tie: the even neighbour
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-6), gn.bf16(1 + 0x1p-7 + 0x1p-8)); // a tie: up to even
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-7), gn.bf16(1 + 0x1p-7 + 0x1p-9)); // below the tie
    try std.testing.expect(std.math.isNan(gn.bf16(std.math.nan(f64))));
    try std.testing.expectEqual(std.math.inf(f64), gn.bf16(0x1.ffp127));
}

const n = 512;

const Row = struct {
    x: [n]f64,
    w: [n]f64,
    up: [gn.rank * n]f64,
    down: [n * gn.rank]f64,

    fn init(seed: u64) Row {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var r: Row = undefined;
        for (&r.x) |*v| v.* = gn.bf16(rng.float(f64) * 4 - 2);
        for (&r.w) |*v| v.* = gn.bf16(0.5 + rng.float(f64));
        for (&r.up) |*v| v.* = gn.bf16((rng.float(f64) * 2 - 1) * 0.1);
        for (&r.down) |*v| v.* = gn.bf16(rng.float(f64) * 2 - 1);
        return r;
    }
};

test "a gate of zero halves 2y: the output is the normed row, every point kept" {
    var r = Row.init(1);
    @memset(&r.down, 0);
    var h: [gn.rank]f64 = undefined;
    var out: [n]f64 = undefined;
    gn.row(&r.x, &r.w, &r.up, &r.down, 1e-6, 2, gn.Point.all, &h, &out);
    var ss: f64 = 0;
    for (r.x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / n + 1e-6);
    for (out, r.x, r.w) |o, v, g| try std.testing.expectEqual(gn.bf16(v * inv * g), o);
}

test "every output is a bf16 value, and each dropped point changes some of them" {
    const r = Row.init(2);
    var h: [gn.rank]f64 = undefined;
    var all: [n]f64 = undefined;
    gn.row(&r.x, &r.w, &r.up, &r.down, 1e-6, 2, gn.Point.all, &h, &all);
    for (all) |o| try std.testing.expectEqual(gn.bf16(o), o);
    for ([_]u32{ gn.Point.normed, gn.Point.up, gn.Point.silu, gn.Point.down, gn.Point.sigmoid }) |p| {
        var out: [n]f64 = undefined;
        gn.row(&r.x, &r.w, &r.up, &r.down, 1e-6, 2, gn.Point.all & ~p, &h, &out);
        var moved: usize = 0;
        for (out, all) |a, b| moved += @intFromBool(a != b);
        try std.testing.expect(moved > 0);
    }
}

test "a NaN in the row makes the whole row NaN, as torch's mean of squares does" {
    var r = Row.init(3);
    r.x[17] = std.math.nan(f64);
    var h: [gn.rank]f64 = undefined;
    var out: [n]f64 = undefined;
    gn.row(&r.x, &r.w, &r.up, &r.down, 1e-6, 2, gn.Point.all, &h, &out);
    for (out) |o| try std.testing.expect(std.math.isNan(o));
}

test "the norm sites the kernel serves are the gated ones, at both releases' widths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: config.Problem = .{};
    const big = try config.parse(std.testing.allocator, ct.json_708b, &p);
    const small = try config.parse(std.testing.allocator, try ct.edited(a, ct.json_708b, &ct.small), &p);
    for ([_]config.Config{ big, small }) |c| for ([_]norms.Site{ .input, .post_self_attn, .post_attention, .post_feedforward, .q_a, .kv_a, .final }) |s| {
        try std.testing.expectEqual(norms.Kind.gated, norms.kind(s));
        const w = norms.width(&c, s);
        try gn.check(1, w, w, w);
    };
    try std.testing.expectEqual(@as(u32, 2560), norms.width(&small, .input));
}
