//! Independent float64 dots over scalar-decoded EXL3 tiles, with bounded reference work at full served dimensions.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;

pub fn For(comptime format: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        shape: c.Shape,
        width: format.Width,
        book: format.Codebook,
        fill: g.Fill,
        words: []i16,
        x: []f16,
        decoded: []f16,
        special: [3]usize,

        pub fn init(a: std.mem.Allocator, shape: c.Shape, halves: u16, book: format.Codebook, fill: g.Fill, seed: u64) !Self {
            if (shape.k < 128 or shape.k % 128 != 0 or shape.n < 128 or shape.n % 128 != 0 or shape.rows == 0)
                return error.InvalidFixtureShape;
            const width = format.Width{ .halves = halves };
            const words = try a.alloc(i16, shape.k / 16 * (shape.n / 16) * width.tileWords());
            errdefer a.free(words);
            const x = try a.alloc(f16, shape.rows * shape.k);
            errdefer a.free(x);
            const decoded = try a.alloc(f16, 3 * (shape.n / 16) * 256);
            errdefer a.free(decoded);
            var f = Self{ .a = a, .shape = shape, .width = width, .book = book, .fill = fill, .words = words, .x = x, .decoded = decoded, .special = .{ 0, shape.k / 32, shape.k / 16 - 1 } };
            f.weights(seed);
            f.inputs(seed ^ 0x3c881407);
            return f;
        }

        pub fn deinit(f: Self) void {
            f.a.free(f.words);
            f.a.free(f.x);
            f.a.free(f.decoded);
        }

        fn weights(f: Self, seed: u64) void {
            @memset(f.words, 0);
            const tw = f.width.tileWords();
            const nb = f.shape.n / 16;
            var rng = std.Random.DefaultPrng.init(seed);
            if (f.fill == .stress) {
                var tile: [128]i16 = undefined;
                for (tile[0..tw]) |*v| v.* = rng.random().int(i16);
                for (0..f.shape.k / 16) |kb| for (0..nb) |j| {
                    @memcpy(f.words[(kb * nb + j) * tw ..][0..tw], tile[0..tw]);
                };
                for (0..3 * nb) |j| format.unpack(f.width, f.book, tile[0..tw], 1, 1, f.decoded[j * 256 ..][0..256]);
            } else {
                for (f.special, 0..) |kb, s| for (0..nb) |j| {
                    const tile = f.words[(kb * nb + j) * tw ..][0..tw];
                    for (tile) |*v| v.* = rng.random().int(i16);
                    format.unpack(f.width, f.book, tile, 1, 1, f.decoded[(s * nb + j) * 256 ..][0..256]);
                };
            }
        }

        fn inputs(f: Self, seed: u64) void {
            const k = f.shape.k;
            const unique = @min(f.shape.rows, 8);
            var rng = std.Random.DefaultPrng.init(seed);
            for (0..unique) |r| {
                const row = f.x[r * k ..][0..k];
                if (f.fill == .stress) {
                    @memset(row, 0);
                    for (0..k / 32) |b| {
                        const q: f32 = @floatFromInt(1 + rng.random().uintLessThan(u32, 2));
                        row[b * 16] = @floatCast(q * 4096);
                        row[k / 2 + b * 16] = @floatCast(-(q + @as(f32, if (b < 8) 1 else 0)) * 4096);
                    }
                } else {
                    for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2 - 1);
                    row[k - 1] = @floatCast(@as(f32, @floatFromInt(r + 1)) / 8);
                }
            }
            if (f.fill == .nonfinite) {
                const positions = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const values = [_]f16{ std.math.nan(f16), std.math.inf(f16), -std.math.inf(f16), std.math.inf(f16), -std.math.inf(f16) };
                for (positions, values) |p, v| if (p[0] < unique) {
                    f.x[p[0] * k + p[1]] = v;
                };
            }
            for (unique..f.shape.rows) |r| @memcpy(f.x[r * k ..][0..k], f.x[(r % unique) * k ..][0..k]);
        }

        pub fn weight(f: Self, i: usize, j: usize) f64 {
            const nb = f.shape.n / 16;
            const cell = (i % 16) * 16 + j % 16;
            if (f.fill == .stress) return f.decoded[(j / 16) * 256 + cell];
            for (f.special, 0..) |kb, s| if (i / 16 == kb) return f.decoded[(s * nb + j / 16) * 256 + cell];
            return format.codebookValue(f.book, 0);
        }

        pub fn reference(f: Self, want: []f64, mag: []f64) !void {
            const k = f.shape.k;
            const n = f.shape.n;
            const unique = @min(f.shape.rows, 8);
            const base: f64 = format.codebookValue(f.book, 0);
            for (0..unique) |r| {
                const x = f.x[r * k ..][0..k];
                var sum: f64 = 0;
                var abs: f64 = 0;
                var bad: [5]usize = undefined;
                var bad_n: usize = 0;
                for (x, 0..) |v, i| {
                    if (!std.math.isFinite(v)) {
                        if (bad_n == bad.len) return error.TooManyNonfinitePlants;
                        bad[bad_n] = i;
                        bad_n += 1;
                    } else {
                        sum += @as(f64, v);
                        abs += @abs(@as(f64, v));
                    }
                }
                for (0..n) |j| {
                    var value: f64 = if (f.fill == .stress) sum * f.weight(0, j) else sum * base;
                    var magnitude: f64 = if (f.fill == .stress) abs * @abs(f.weight(0, j)) else abs * @abs(base);
                    if (f.fill != .stress) for (f.special) |kb| {
                        for (0..16) |t| {
                            const i = kb * 16 + t;
                            const xv: f64 = x[i];
                            if (!std.math.isFinite(xv)) continue;
                            const w = f.weight(i, j);
                            value += xv * (w - base);
                            magnitude += @abs(xv) * (@abs(w) - @abs(base));
                        }
                    };
                    for (bad[0..bad_n]) |i| {
                        const term = @as(f64, x[i]) * f.weight(i, j);
                        value += term;
                        magnitude += @abs(term);
                    }
                    want[r * n + j] = value;
                    mag[r * n + j] = @max(0, magnitude);
                }
            }
            for (unique..f.shape.rows) |r| {
                @memcpy(want[r * n ..][0..n], want[(r % unique) * n ..][0..n]);
                @memcpy(mag[r * n ..][0..n], mag[(r % unique) * n ..][0..n]);
            }
        }

        pub fn simulate(f: Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            if (f.fill != .stress) return error.NotStressFixture;
            const k = f.shape.k;
            const n = f.shape.n;
            const terms = try f.a.alloc(f64, k);
            defer f.a.free(terms);
            const unique = @min(f.shape.rows, 8);
            for (0..unique) |r| {
                var values: [16]f64 = undefined;
                for (0..16) |j| {
                    for (terms, 0..) |*v, i| v.* = @as(f64, f.x[r * k + i]) * f.weight(i, j);
                    values[j] = contract.rows.sumIn(acc, order, terms);
                }
                for (out[r * n ..][0..n], 0..) |*v, j| v.* = values[j % 16];
            }
            for (unique..f.shape.rows) |r| @memcpy(out[r * n ..][0..n], out[(r % unique) * n ..][0..n]);
        }
    };
}

test "compressed EXL3 full-shape reference agrees with every scalar-unpacked column including nonfinite plants" {
    const format = @import("exl3_format");
    const Fixture = For(format);
    const a = std.testing.allocator;
    for ([_]g.Fill{ .random, .nonfinite, .stress }) |fill| for ([_]format.Codebook{ .inst3, .mcg, .mul1 }) |book| {
        const f = try Fixture.init(a, .{ .rows = 9, .n = 128, .k = 128 }, if (book == .mul1) 3 else 6, book, fill, 73);
        defer f.deinit();
        const dense = try a.alloc(f16, 128 * 128);
        defer a.free(dense);
        format.unpack(f.width, book, f.words, 8, 8, dense);
        var want: [9 * 128]f64 = undefined;
        var mag: [9 * 128]f64 = undefined;
        try f.reference(&want, &mag);
        for (0..9) |r| for (0..128) |j| {
            var sum: f64 = 0;
            var absolute: f64 = 0;
            for (0..128) |i| {
                const term = @as(f64, f.x[r * 128 + i]) * @as(f64, dense[i * 128 + j]);
                sum += term;
                absolute += @abs(term);
            }
            try std.testing.expectEqual(c.classOf(sum), c.classOf(want[r * 128 + j]));
            if (std.math.isFinite(sum)) {
                try std.testing.expectApproxEqAbs(sum, want[r * 128 + j], 0x1p-40 * @max(1, absolute));
                try std.testing.expectApproxEqAbs(absolute, mag[r * 128 + j], 0x1p-40 * @max(1, absolute));
            }
        };
    };
}

test "EXL3 precision plants remain exact in float32 with every replay order" {
    const format = @import("exl3_format");
    const Fixture = For(format);
    const a = std.testing.allocator;
    for ([_]format.Codebook{ .inst3, .mcg, .mul1 }) |book| {
        const f = try Fixture.init(a, .{ .rows = 5, .n = 128, .k = 8192 }, 6, book, .stress, 41);
        defer f.deinit();
        var want: [5 * 128]f64 = undefined;
        var mag: [5 * 128]f64 = undefined;
        try f.reference(&want, &mag);
        var actual: [5 * 128]f64 = undefined;
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            try f.simulate(.f32, order, &actual);
            try std.testing.expectEqualSlices(f64, &want, &actual);
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try f.simulate(acc, order, &actual);
                var margin: f64 = 0;
                for (actual, want) |got, expected| {
                    const bound = 0x1p-24 * @max(0x1p-126, @abs(expected));
                    margin = @max(margin, if (!std.math.isFinite(got)) std.math.inf(f64) else @abs(got - expected) / bound);
                }
                try std.testing.expect(margin >= 100);
            }
        }
    }
}
