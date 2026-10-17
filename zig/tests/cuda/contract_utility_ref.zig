//! Independent host data and oracles for ordered slice sums and in-place history copies.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;

pub fn depth(s: c.Shape) usize {
    return s.k - 1;
}

pub fn sumSpec(bf16: bool) c.Spec {
    return .{
        .name = if (bf16) "cuda fn_sum_slices bf16" else "cuda fn_sum_slices f32",
        .accum = .f32,
        .depth = depth,
        .out = if (bf16) .bf16 else .f32,
        .nan = .{ .stated = "propagates NaN and infinities; finite outputs use ordered f32 and one final RNE" },
        .tiles = .{ .{ .size = 16 }, .{ .size = 256 }, .{ .size = 4 } },
        .served = &.{ .{ .rows = 17, .n = 10240, .k = 4 }, .{ .rows = 4, .n = 2560, .k = 8 } },
    };
}

pub fn shiftSpec(padded: bool) c.Spec {
    return .{
        .name = if (padded) "cuda fn_shift_window strided" else "cuda fn_shift_window contiguous",
        .accum = .none,
        .depth = depth,
        .out = .bf16,
        .in_place = true,
        .nan = .{ .stated = "copies all bf16 bits, including NaN payloads, infinities and signed zero" },
        // rows = appended rows, n = channels, k = retained taps; stride is independently padded.
        .tiles = .{ .{ .size = 16 }, .{ .size = 256 }, .{ .size = 9 } },
        .served = &.{
            .{ .rows = 5, .n = 10240, .k = 9 },
            .{ .rows = 1, .n = 10240, .k = 3 },
            .{ .rows = 4, .n = 10240, .k = 9 },
            .{ .rows = 9, .n = 257, .k = 9 },
            .{ .rows = 16, .n = 255, .k = 16 },
            .{ .rows = 32, .n = 10240, .k = 9 },
        },
    };
}

/// Bounded fixture domains, not validation added to the production launchers.
pub fn sumShape(s: c.Shape) bool {
    return s.rows > 0 and s.rows <= 64 and s.n > 0 and s.n <= 16384 and s.k > 0 and s.k <= 8;
}

pub fn shiftShape(s: c.Shape) bool {
    return s.rows > 0 and s.rows <= 64 and s.n > 0 and s.n <= 16384 and s.k > 0 and s.k <= 16;
}

pub fn sumFill(t: []f32, s: c.Shape, fill: g.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const e = s.rows * s.n;
    for (t) |*v| v.* = (rng.float(f32) * 2 - 1) *
        std.math.ldexp(@as(f32, 1), @as(i32, @intCast(rng.uintLessThan(u32, 17))) - 8);
    switch (fill) {
        .random => {},
        .nonfinite => {
            t[0] = std.math.nan(f32);
            t[1] = std.math.inf(f32);
            t[2] = -std.math.inf(f32);
            t[3] = std.math.inf(f32);
            if (s.k > 1) t[e + 3] = -std.math.inf(f32);
            // The initial term is loaded, not added to a synthetic positive zero.
            for (0..s.k) |j| t[j * e + 4] = -0.0;
        },
        .stress => for (0..s.k) |j| {
            const pattern = [_]f32{ 32768, 1, -32768, 1 };
            for (0..e) |i| t[j * e + i] = pattern[j % 4];
        },
    }
}

pub fn sumReference(t: []const f32, s: c.Shape, want: []f64, mag: []f64) void {
    const e = s.rows * s.n;
    for (want, mag, 0..) |*w, *m, i| {
        var total: f64 = 0;
        var absolute: f64 = 0;
        for (0..s.k) |j| {
            const term: f64 = t[j * e + i];
            total += term;
            absolute += @abs(term);
        }
        w.* = total;
        m.* = absolute;
    }
}

pub fn sumSimulate(t: []const f32, s: c.Shape, acc: c.Accum, order: g.Order, out: []f64) void {
    var terms: [8]f64 = undefined;
    for (out, 0..) |*v, i| {
        for (terms[0..s.k], 0..) |*term, j| term.* = t[j * out.len + i];
        v.* = core.contract.rows.sumIn(acc, order, terms[0..s.k]);
    }
}

/// A separate serial f32 oracle checks the declared slice order and single final conversion.
pub fn ordered(t: []const f32, e: usize, k: usize, i: usize) f32 {
    var value = t[i];
    for (1..k) |j| value += t[j * e + i];
    return value;
}

/// Both the independent real-arithmetic bound and the documented serial f32 order must hold.
pub fn sumCheck(a: std.mem.Allocator, t: []const f32, s: c.Shape, bf16: bool, out: []const u8, nonfinite: bool) !?[]const u8 {
    const e = s.rows * s.n;
    const want = try a.alloc(f64, e);
    defer a.free(want);
    const mag = try a.alloc(f64, e);
    defer a.free(mag);
    sumReference(t, s, want, mag);
    const spec = sumSpec(bf16);
    if (out.len != e * c.size(spec.out)) return "output byte count differs from the declared shape";
    var planted: usize = 0;
    for (want, mag, 0..) |w, m, i| {
        const value = ordered(t, e, s.k, i);
        const rounded: f32 = if (bf16) c.fromBf16(c.toBf16(value)) else value;
        const got = c.read(spec.out, out, i);
        if (c.classOf(w) != .finite) planted += 1;
        if (c.classOf(got) != c.classOf(w)) return "an output's class differs from the float64 oracle";
        if (std.math.isNan(rounded)) continue;
        if (@as(u32, @bitCast(@as(f32, @floatCast(got)))) != @as(u32, @bitCast(rounded)))
            return "an output differs from ordered f32 with one final conversion";
        if (std.math.isFinite(w) and !(@abs(got - w) <= c.bound(spec, s, w, m)))
            return "an output exceeds the independent float64 error bound";
    }
    if (nonfinite and (planted == 0 or planted == e))
        return "nonfinite planting did not retain both finite and nonfinite outputs";
    return null;
}

pub fn copyFill(win: []u16, rows: []u16, s: c.Shape, stride: usize, fill: g.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (win) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 32);
    // Padding has unique nonfinite poison; it must never become an output channel.
    @memset(rows, 0x7fff);
    for (0..s.rows) |r| for (rows[r * stride ..][0..s.n]) |*v| {
        v.* = c.toBf16((rng.float(f32) * 2 - 1) * 32);
    };
    if (fill == .nonfinite) {
        const bits = [_]u16{ 0x7fc1, 0x7f80, 0xff80, 0x8000, 0x0000, 0xffc7 };
        // Plant both the retained old history and appended rows. At least one finite channel remains.
        for (win, 0..) |*v, i| if (i % 13 < bits.len) {
            v.* = bits[i % 13];
        };
        for (0..s.rows) |r| for (rows[r * stride ..][0..s.n], 0..) |*v, ch| {
            if (ch % 13 < bits.len) v.* = bits[ch % 13];
        };
    }
}

/// Append channel rows to a host sequence, then retain its tail; no in-place update algorithm is reused.
pub fn copyReference(a: std.mem.Allocator, win: []const u16, rows: []const u16, s: c.Shape, stride: usize) ![]u16 {
    const sequence = try a.alloc(u16, (s.k + s.rows) * s.n);
    defer a.free(sequence);
    @memcpy(sequence[0..win.len], win);
    for (0..s.rows) |r| @memcpy(sequence[win.len + r * s.n ..][0..s.n], rows[r * stride ..][0..s.n]);
    return a.dupe(u16, sequence[sequence.len - win.len ..]);
}

test "ordered sum stress proves fp32 and rejects reduced accumulators in all three orders" {
    const a = std.testing.allocator;
    inline for (.{ false, true }) |bf16| for ([_]usize{ 4, 8 }) |k| {
        const s: c.Shape = .{ .rows = 2, .n = 3, .k = k };
        var terms: [48]f32 = undefined;
        const t = terms[0 .. s.rows * s.n * k];
        sumFill(t, s, .stress, 42);
        var want: [6]f64 = undefined;
        var mag: [6]f64 = undefined;
        sumReference(t, s, &want, &mag);
        const spec = sumSpec(bf16);
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            for ([_]c.Accum{ .f32, .f16, .bf16 }) |acc| {
                var got: [6]f64 = undefined;
                sumSimulate(t, s, acc, order, &got);
                for (want, got) |w, v| {
                    const ratio = @abs(v - w) / c.boundExact(spec, w);
                    if (acc == .f32) try std.testing.expect(ratio <= 1) else try std.testing.expect(ratio >= 100);
                }
            }
        }
        const shapes = try c.shapes(a, spec);
        defer a.free(shapes);
        for (spec.served) |served| try std.testing.expect(sumShape(served));
    };
}

test "sum oracle preserves initial negative zero and propagates nonfinite classes" {
    const s: c.Shape = .{ .rows = 1, .n = 8, .k = 4 };
    var terms: [32]f32 = undefined;
    sumFill(&terms, s, .nonfinite, 71);
    try std.testing.expectEqual(c.Class.nan, c.classOf(ordered(&terms, 8, 4, 0)));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(ordered(&terms, 8, 4, 1)));
    try std.testing.expectEqual(c.Class.neg_inf, c.classOf(ordered(&terms, 8, 4, 2)));
    try std.testing.expectEqual(c.Class.nan, c.classOf(ordered(&terms, 8, 4, 3)));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(ordered(&terms, 8, 4, 4))));
}

test "history oracle covers partial replacement, exact replacement and overflow with padding" {
    const a = std.testing.allocator;
    const win = [_]u16{ 1, 2, 3, 4, 5, 6 };
    const rows = [_]u16{ 7, 8, 99, 9, 10, 99, 11, 12, 99, 13, 14, 99 };
    for ([_]usize{ 1, 3, 4 }) |n| {
        const want = try copyReference(a, &win, &rows, .{ .rows = n, .n = 2, .k = 3 }, 3);
        defer a.free(want);
        const expected: []const u16 = switch (n) {
            1 => &.{ 3, 4, 5, 6, 7, 8 },
            3 => &.{ 7, 8, 9, 10, 11, 12 },
            else => &.{ 9, 10, 11, 12, 13, 14 },
        };
        try std.testing.expectEqualSlices(u16, expected, want);
    }
}

test "fixture geometry refuses zero, oversized slice counts and unsafe local history lengths" {
    try std.testing.expect(!sumShape(.{ .rows = 1, .n = 1, .k = 0 }));
    try std.testing.expect(!sumShape(.{ .rows = 1, .n = 1, .k = 9 }));
    try std.testing.expect(!shiftShape(.{ .rows = 0, .n = 1, .k = 1 }));
    try std.testing.expect(!shiftShape(.{ .rows = 1, .n = 1, .k = 17 }));
    for (shiftSpec(true).served) |s| try std.testing.expect(shiftShape(s));
}

test "sum checker rejects wrong order, missing terms and repeated bf16 conversion" {
    const a = std.testing.allocator;
    const s: c.Shape = .{ .rows = 1, .n = 1, .k = 4 };
    const t = [_]f32{ 16777216, 1, -16777216, 1 };
    const correct = [_]f32{1};
    const reordered = [_]f32{2};
    const lost_tail = [_]f32{0};
    try std.testing.expect(try sumCheck(a, &t, s, false, std.mem.sliceAsBytes(&correct), false) == null);
    try std.testing.expect(try sumCheck(a, &t, s, false, std.mem.sliceAsBytes(&reordered), false) != null);
    try std.testing.expect(try sumCheck(a, &t, s, false, std.mem.sliceAsBytes(&lost_tail), false) != null);
    const small = [_]f32{ 1, 0.00390625, 0.00390625, 0 };
    const once = [_]u16{0x3f81};
    const repeated = [_]u16{0x3f80};
    try std.testing.expect(try sumCheck(a, &small, s, true, std.mem.sliceAsBytes(&once), false) == null);
    try std.testing.expect(try sumCheck(a, &small, s, true, std.mem.sliceAsBytes(&repeated), false) != null);
}

test "history copy retains raw nonfinite payloads and distinguishes padding and repeated mutation" {
    const a = std.testing.allocator;
    const s: c.Shape = .{ .rows = 1, .n = 3, .k = 3 };
    const win = [_]u16{ 1, 2, 3, 0x7fc1, 0xffc7, 0x8000, 0x7f80, 0xff80, 0 };
    const source = [_]u16{ 4, 5, 6, 0x7fff };
    const once = try copyReference(a, &win, &source, s, 4);
    defer a.free(once);
    try std.testing.expectEqualSlices(u16, &.{ 0x7fc1, 0xffc7, 0x8000, 0x7f80, 0xff80, 0, 4, 5, 6 }, once);
    const twice = try copyReference(a, once, &source, s, 4);
    defer a.free(twice);
    try std.testing.expect(!std.mem.eql(u16, once, twice));
    try std.testing.expect(std.mem.indexOfScalar(u16, once, 0x7fff) == null);
    for ([_]bool{ false, true }) |padded| {
        const spec = shiftSpec(padded);
        const shapes = try c.shapes(a, spec);
        defer a.free(shapes);
        for (shapes) |shape| try std.testing.expect(shiftShape(shape));
    }
}

test "seeded finite and nonfinite slice outputs satisfy both oracles and reject untouched output" {
    const a = std.testing.allocator;
    const s: c.Shape = .{ .rows = 5, .n = 17, .k = 8 };
    var t: [680]f32 = undefined;
    var f: [85]f32 = undefined;
    var b: [85]u16 = undefined;
    for ([_]g.Fill{ .random, .nonfinite }) |fill| {
        sumFill(&t, s, fill, 0x51ce);
        for (&f, &b, 0..) |*fv, *bv, i| {
            fv.* = ordered(&t, f.len, s.k, i);
            bv.* = c.toBf16(fv.*);
        }
        try std.testing.expect(try sumCheck(a, &t, s, false, std.mem.sliceAsBytes(&f), fill == .nonfinite) == null);
        try std.testing.expect(try sumCheck(a, &t, s, true, std.mem.sliceAsBytes(&b), fill == .nonfinite) == null);
        f[f.len - 1] = std.math.nan(f32);
        try std.testing.expect(try sumCheck(a, &t, s, false, std.mem.sliceAsBytes(&f), fill == .nonfinite) != null);
    }
}

test "history oracle agrees with independent row-at-a-time queue including all sixteen taps" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 3, 9, 16 }) |taps| for ([_]usize{ 1, 5, 16, 32 }) |n| {
        const s: c.Shape = .{ .rows = n, .n = 17, .k = taps };
        const stride = s.n + 7;
        const win = try a.alloc(u16, taps * s.n);
        defer a.free(win);
        const source = try a.alloc(u16, n * stride);
        defer a.free(source);
        copyFill(win, source, s, stride, .nonfinite, 0x7157);
        const want = try copyReference(a, win, source, s, stride);
        defer a.free(want);
        const queue = try a.dupe(u16, win);
        defer a.free(queue);
        for (0..n) |r| {
            std.mem.copyForwards(u16, queue[0 .. queue.len - s.n], queue[s.n..]);
            @memcpy(queue[queue.len - s.n ..], source[r * stride ..][0..s.n]);
        }
        try std.testing.expectEqualSlices(u16, queue, want);
    };
}
