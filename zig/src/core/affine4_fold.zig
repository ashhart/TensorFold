//! A change folded into MLX 4-bit codes on each group's own grid, each column's rounding error carried on as in GPTQ.
const std = @import("std");
const host4 = @import("affine4_host.zig");

const group = host4.group;
const lanes = 8;
const V = @Vector(lanes, f32);

/// `h` [n, n] (PSD) becomes upper U, U^T U = (h + add I)^-1, via h = Z Z^T, U = Z^-1 in f64 for ill-conditioned h.
pub fn inverseFactor(gpa: std.mem.Allocator, h: []f32, n: usize, add: f32) !void {
    const z = try gpa.alloc(f64, n * n);
    defer gpa.free(z);
    for (z, h[0 .. n * n]) |*o, x| o.* = x;
    for (0..n) |i| z[i * n + i] += add;
    // Z, upper, column by column from the last: H[i][j] = Z[i][j] Z[j][j] + sum over k > j of Z[i][k] Z[j][k]
    var j = n;
    while (j > 0) {
        j -= 1;
        const zj = z[j * n ..][j + 1 .. n];
        const d = z[j * n + j] - dot(zj, zj);
        if (!(d > 0)) return error.NotPositiveDefinite;
        const zz = @sqrt(d);
        z[j * n + j] = zz;
        try parallel(j, struct {
            fn run(m: []f64, nn: usize, jj: usize, piv: f64, from: usize, to: usize) void {
                const zjj = m[jj * nn ..][jj + 1 .. nn];
                for (from..to) |i| m[i * nn + jj] = (m[i * nn + jj] - dot(m[i * nn ..][jj + 1 .. nn], zjj)) / piv;
            }
        }.run, .{ z, n, j, zz });
    }
    for (0..n) |r| @memset(z[r * n ..][0..r], 0);
    // U = Z^-1, upper, row by row from the last: row i reads its own Z row and the rows below, already U
    const row = try gpa.alloc(f64, n);
    defer gpa.free(row);
    var i = n;
    while (i > 0) {
        i -= 1;
        const zi = z[i * n ..][0..n];
        const inv = 1 / zi[i];
        @memset(row, 0);
        row[i] = inv;
        for (i + 1..n) |k| {
            const zik = zi[k];
            if (zik == 0) continue;
            axpy(row[k..n], -zik, z[k * n ..][k..n]);
        }
        for (i + 1..n) |c| row[c] *= inv;
        @memcpy(zi[i..n], row[i..n]);
    }
    for (h[0 .. n * n], z) |*o, x| o.* = @floatCast(x);
}

/// `delta` added to `words`, rounded onto their grid, errors carried along `h` (both scratch); how many codes changed.
pub fn addRounded(gpa: std.mem.Allocator, words: []u32, scales: []const u16, biases: []const u16, rows: usize, cols: usize, delta: []f32, h: []f32, add: f32) !usize {
    const now = try gpa.alloc(f32, rows * cols);
    defer gpa.free(now);
    host4.dequantize(words[0 .. rows * cols / 8], scales, biases, now);
    for (delta[0 .. rows * cols], now) |*t, w| t.* += w;
    try inverseFactor(gpa, h, cols, add);
    const fresh = try gpa.alloc(u32, rows * cols / 8);
    defer gpa.free(fresh);
    try round(delta, rows, cols, scales, biases, h, fresh);
    var changed: usize = 0;
    for (words[0 .. rows * cols / 8], fresh) |*o, x| {
        var diff = o.* ^ x;
        while (diff != 0) : (diff >>= 4) changed += @intFromBool(diff & 15 != 0);
        o.* = x;
    }
    return changed;
}

/// New codes for `target` ([rows, cols] f32, used as scratch) on each group's grid, with `u` from inverseFactor.
pub fn round(target: []f32, rows: usize, cols: usize, scales: []const u16, biases: []const u16, u: []const f32, words: []u32) !void {
    @memset(words, 0);
    try parallel(rows, struct {
        fn run(t: []f32, c: usize, s: []const u16, b: []const u16, uu: []const f32, ww: []u32, from: usize, to: usize) void {
            for (from..to) |r| roundRow(t[r * c ..][0..c], c, s[r * (c / group) ..][0 .. c / group], b[r * (c / group) ..][0 .. c / group], uu, ww[r * c / 8 ..][0 .. c / 8]);
        }
    }.run, .{ target, cols, scales, biases, u, words });
}

/// One row: each column rounded in turn, its error carried into the columns after it along u's row.
fn roundRow(w: []f32, cols: usize, scales: []const u16, biases: []const u16, u: []const f32, words: []u32) void {
    for (0..cols) |j| {
        const s = host4.f32of(scales[j / group]);
        const b = host4.f32of(biases[j / group]);
        const q: u32 = if (s == 0) 0 else @intFromFloat(std.math.clamp(@round((w[j] - b) / s), 0, 15));
        words[j / 8] |= q << @intCast(4 * (j % 8));
        const e = (w[j] - (s * @as(f32, @floatFromInt(q)) + b)) / u[j * cols + j];
        if (e == 0) continue;
        const uj = u[j * cols ..][0..cols];
        var k = j + 1;
        while (k + lanes <= cols) : (k += lanes) {
            const wv: V = w[k..][0..lanes].*;
            const uv: V = uj[k..][0..lanes].*;
            w[k..][0..lanes].* = wv - @as(V, @splat(e)) * uv;
        }
        while (k < cols) : (k += 1) w[k] -= e * uj[k];
    }
}

const V64 = @Vector(4, f64);

fn dot(a: []const f64, b: []const f64) f64 {
    var acc: V64 = @splat(0);
    var i: usize = 0;
    while (i + 4 <= a.len) : (i += 4) acc += @as(V64, a[i..][0..4].*) * @as(V64, b[i..][0..4].*);
    var s: f64 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

/// y += alpha x.
fn axpy(y: []f64, alpha: f64, x: []const f64) void {
    var i: usize = 0;
    const av: V64 = @splat(alpha);
    while (i + 4 <= y.len) : (i += 4) y[i..][0..4].* = @as(V64, y[i..][0..4].*) + av * @as(V64, x[i..][0..4].*);
    while (i < y.len) : (i += 1) y[i] += alpha * x[i];
}

/// f(args..., from, to) over 0..n split across the machine's threads (inline when n is small).
fn parallel(n: usize, comptime f: anytype, args: anytype) !void {
    const cpus = std.Thread.getCpuCount() catch 1;
    const k = @min(@max(cpus, 1), 32, @max(n / 16, 1));
    if (k <= 1) return @call(.auto, f, args ++ .{ @as(usize, 0), n });
    var threads: [32]?std.Thread = @splat(null);
    for (0..k) |t| {
        const from = n * t / k;
        const to = n * (t + 1) / k;
        threads[t] = std.Thread.spawn(.{}, f, args ++ .{ from, to }) catch blk: {
            @call(.auto, f, args ++ .{ from, to });
            break :blk null;
        };
    }
    for (threads[0..k]) |t| if (t) |th| th.join();
}

test "the factor's square is the damped second moment's inverse" {
    const gpa = std.testing.allocator;
    const n = 48;
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    const x = try gpa.alloc(f32, 200 * n);
    defer gpa.free(x);
    for (x) |*v| v.* = r.floatNorm(f32);
    const h = try gpa.alloc(f32, n * n);
    defer gpa.free(h);
    const keep = try gpa.alloc(f32, n * n);
    defer gpa.free(keep);
    for (0..n) |a| for (0..n) |b| {
        var s: f32 = 0;
        for (0..200) |t| s += x[t * n + a] * x[t * n + b];
        h[a * n + b] = s;
    };
    try inverseFactor(gpa, h, n, 0);
    for (0..n) |a| for (0..n) |b| {
        var s: f32 = 0;
        for (0..200) |t| s += x[t * n + a] * x[t * n + b];
        keep[a * n + b] = s;
    };
    // H (U^T U) = I
    for (0..n) |a| for (0..n) |b| {
        var s: f64 = 0;
        for (0..n) |m| {
            var hinv: f64 = 0;
            for (0..n) |q| hinv += @as(f64, h[q * n + m]) * h[q * n + b];
            s += @as(f64, keep[a * n + m]) * hinv;
        }
        try std.testing.expectApproxEqAbs(@as(f64, if (a == b) 1 else 0), s, 1e-3);
    };
}

test "rounding with carried errors moves outputs on the inputs it saw less than rounding each weight alone" {
    const gpa = std.testing.allocator;
    const rows = 8;
    const cols = 2 * group;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    // a grid from random weights, then a small change on top
    const base = try gpa.alloc(f32, rows * cols);
    defer gpa.free(base);
    for (base) |*v| v.* = r.floatNorm(f32) * 0.05;
    const words = try gpa.alloc(u32, rows * cols / 8);
    defer gpa.free(words);
    const scales = try gpa.alloc(u16, rows * cols / group);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, rows * cols / group);
    defer gpa.free(biases);
    host4.quantize(base, words, scales, biases);
    host4.dequantize(words, scales, biases, base);
    const target = try gpa.alloc(f32, rows * cols);
    defer gpa.free(target);
    for (target, base) |*t, b| t.* = b + r.floatNorm(f32) * 0.004;
    // inputs that mostly lie in a few directions, as activations do
    const n_in = 300;
    const x = try gpa.alloc(f32, n_in * cols);
    defer gpa.free(x);
    for (0..n_in) |t| {
        const c0 = r.floatNorm(f32);
        const c1 = r.floatNorm(f32);
        for (0..cols) |c| x[t * cols + c] = c0 * @sin(@as(f32, @floatFromInt(c)) * 0.1) + c1 * @cos(@as(f32, @floatFromInt(c)) * 0.05) + 0.05 * r.floatNorm(f32);
    }
    const h = try gpa.alloc(f32, cols * cols);
    defer gpa.free(h);
    for (0..cols) |a| for (0..cols) |b| {
        var s: f32 = 0;
        for (0..n_in) |t| s += x[t * cols + a] * x[t * cols + b];
        h[a * cols + b] = s;
    };
    var mean: f32 = 0;
    for (0..cols) |i| mean += h[i * cols + i];
    try inverseFactor(gpa, h, cols, 0.01 * mean / cols);
    const scratch = try gpa.dupe(f32, target);
    defer gpa.free(scratch);
    const carried = try gpa.alloc(u32, rows * cols / 8);
    defer gpa.free(carried);
    try round(scratch, rows, cols, scales, biases, h, carried);
    const alone = try gpa.alloc(u32, rows * cols / 8);
    defer gpa.free(alone);
    @memset(alone, 0);
    for (0..rows * cols) |i| {
        const s = host4.f32of(scales[i / group]);
        const b = host4.f32of(biases[i / group]);
        const q: u32 = @intFromFloat(std.math.clamp(@round((target[i] - b) / s), 0, 15));
        alone[i / 8] |= q << @intCast(4 * (i % 8));
    }
    const got = try gpa.alloc(f32, rows * cols);
    defer gpa.free(got);
    var err_carried: f64 = 0;
    var err_alone: f64 = 0;
    for ([_][]const u32{ carried, alone }, [_]*f64{ &err_carried, &err_alone }) |codes, err| {
        host4.dequantize(codes, scales, biases, got);
        for (0..n_in) |t| for (0..rows) |o| {
            var s: f64 = 0;
            for (0..cols) |c| s += @as(f64, got[o * cols + c] - target[o * cols + c]) * x[t * cols + c];
            err.* += s * s;
        };
    }
    try std.testing.expect(err_carried < err_alone);
}

test "a rank-one change rounded into the codes comes out at its key, where rounding each weight alone loses it" {
    const gpa = std.testing.allocator;
    const rows = 16;
    const cols = 4 * group;
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    const base = try gpa.alloc(f32, rows * cols);
    defer gpa.free(base);
    for (base) |*v| v.* = r.floatNorm(f32) * 0.02;
    const words = try gpa.alloc(u32, rows * cols / 8);
    defer gpa.free(words);
    const scales = try gpa.alloc(u16, rows * cols / group);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, rows * cols / group);
    defer gpa.free(biases);
    host4.quantize(base, words, scales, biases);
    const before = try gpa.dupe(u32, words);
    defer gpa.free(before);
    // a key, and kept inputs that share none of its own direction
    const key = try gpa.alloc(f32, cols);
    defer gpa.free(key);
    for (key) |*v| v.* = @max(r.floatNorm(f32), 0);
    const n_in = 400;
    const x = try gpa.alloc(f32, n_in * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = @max(r.floatNorm(f32), 0);
    // the change: outputs b at the key, a = key / |key|^2, each weight far below a grid step
    const out = try gpa.alloc(f32, rows);
    defer gpa.free(out);
    for (out) |*v| v.* = r.floatNorm(f32) * 0.01;
    var kk: f32 = 0;
    for (key) |v| kk += v * v;
    const delta = try gpa.alloc(f32, rows * cols);
    defer gpa.free(delta);
    for (0..rows) |o| for (0..cols) |i| {
        delta[o * cols + i] = out[o] * key[i] / kk;
    };
    const h = try gpa.alloc(f32, cols * cols);
    defer gpa.free(h);
    for (0..cols) |a| for (0..cols) |b| {
        var s: f32 = 0;
        for (0..n_in) |t| s += x[t * cols + a] * x[t * cols + b];
        h[a * cols + b] = s / n_in + 1e3 * key[a] * key[b];
    };
    var mean: f32 = 0;
    for (0..cols) |i| mean += h[i * cols + i];
    const changed = try addRounded(gpa, words, scales, biases, rows, cols, delta, h, 1e-3 * mean / cols);
    try std.testing.expect(changed > 0);
    const was = try gpa.alloc(f32, rows * cols);
    defer gpa.free(was);
    const now = try gpa.alloc(f32, rows * cols);
    defer gpa.free(now);
    host4.dequantize(before, scales, biases, was);
    host4.dequantize(words, scales, biases, now);
    // at the key: the change the codes make against the one asked; rounding alone moves no code at all here
    var miss: f64 = 0;
    var want: f64 = 0;
    for (0..rows) |o| {
        var got: f64 = 0;
        for (0..cols) |i| got += @as(f64, now[o * cols + i] - was[o * cols + i]) * key[i];
        miss += (got - out[o]) * (got - out[o]);
        want += @as(f64, out[o]) * out[o];
    }
    try std.testing.expect(miss < 0.05 * want);
}
