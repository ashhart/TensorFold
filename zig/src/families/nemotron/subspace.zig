//! Small dense linear algebra for a lesson's directions: orthonormal rows, and where one set of rows outweighs another.
const std = @import("std");

const V = @Vector(8, f32);

pub fn dot(x: []const f32, y: []const f32) f32 {
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) acc += @as(V, x[i..][0..8].*) * @as(V, y[i..][0..8].*);
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) s += x[i] * y[i];
    return s;
}

/// y += c x.
pub fn axpy(y: []f32, c: f32, x: []const f32) void {
    const cv: V = @splat(c);
    var i: usize = 0;
    while (i + 8 <= y.len) : (i += 8) y[i..][0..8].* = @as(V, y[i..][0..8].*) + cv * @as(V, x[i..][0..8].*);
    while (i < y.len) : (i += 1) y[i] += c * x[i];
}

/// Rows of `m` made orthonormal in place, each against those before it twice over; a row that vanishes becomes zero.
pub fn orthonormal(m: []f32, rows: usize, len: usize) void {
    for (0..rows) |j| {
        const row = m[j * len ..][0..len];
        const size = @sqrt(dot(row, row));
        for (0..2) |_| for (0..j) |i| {
            const prev = m[i * len ..][0..len];
            axpy(row, -dot(row, prev), prev);
        };
        const left = @sqrt(dot(row, row));
        if (!(left > 1e-4 * size)) {
            @memset(row, 0);
            continue;
        }
        for (row) |*v| v.* /= left;
    }
}

/// Each row of `m` (rows x len) with its parts along the orthonormal rows of `basis` taken out, twice over.
pub fn remove(m: []f32, rows: usize, basis: []const f32, count: usize, len: usize) void {
    for (0..rows) |q| {
        const row = m[q * len ..][0..len];
        for (0..2) |_| for (0..count) |u| {
            const dir = basis[u * len ..][0..len];
            axpy(row, -dot(row, dir), dir);
        };
    }
}

/// The k orthonormal directions (rows of out) where `more` (n x n) outweighs `less`: more v = mu (less + lambda) v.
pub fn outweigh(gpa: std.mem.Allocator, more: []const f64, less: []const f64, n: usize, k: usize, out: []f32) !void {
    const l = try gpa.dupe(f64, less);
    defer gpa.free(l);
    var trace: f64 = 0;
    for (0..n) |i| trace += l[i * n + i];
    const lambda = 1e-3 * trace / @as(f64, @floatFromInt(n)) + 1e-12;
    for (0..n) |i| l[i * n + i] += lambda;
    try cholesky(l, n);
    const s = try gpa.alloc(f64, n * n);
    defer gpa.free(s);
    const t = try gpa.alloc(f64, n * n);
    defer gpa.free(t);
    @memcpy(t, more);
    for (0..n) |col| lower(l, n, t, col);
    for (0..n) |i| for (0..n) |j| {
        s[i * n + j] = t[j * n + i];
    };
    for (0..n) |col| lower(l, n, s, col);
    const vecs = try gpa.alloc(f64, n * n);
    defer gpa.free(vecs);
    jacobi(s, n, vecs);
    const vals = try gpa.alloc(f64, n);
    defer gpa.free(vals);
    for (vals, 0..) |*v, i| v.* = s[i * n + i];
    const order = try gpa.alloc(usize, n);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, vals, struct {
        fn bigger(v: []const f64, a: usize, b: usize) bool {
            return v[a] > v[b];
        }
    }.bigger);
    const w = try gpa.alloc(f64, n);
    defer gpa.free(w);
    for (0..k) |q| {
        for (w, 0..) |*x, i| x.* = vecs[i * n + order[q]];
        upper(l, n, w);
        for (out[q * n ..][0..n], w) |*o, x| o.* = @floatCast(x);
    }
    orthonormal(out, k, n);
}

/// x = a^-1 b for a symmetric positive definite a (n x n, overwritten by its Cholesky factor).
pub fn solve(a: []f64, n: usize, b: []f64) !void {
    try cholesky(a, n);
    for (0..n) |i| {
        var v = b[i];
        for (0..i) |k| v -= a[i * n + k] * b[k];
        b[i] = v / a[i * n + i];
    }
    upper(a, n, b);
}

/// a = L L^T in place (L in the lower triangle, the upper zeroed); a must be symmetric positive definite.
fn cholesky(a: []f64, n: usize) !void {
    for (0..n) |j| {
        var d = a[j * n + j];
        for (0..j) |k| d -= a[j * n + k] * a[j * n + k];
        if (!(d > 0)) return error.NotPositiveDefinite;
        const root = @sqrt(d);
        a[j * n + j] = root;
        for (j + 1..n) |i| {
            var v = a[i * n + j];
            for (0..j) |k| v -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = v / root;
        }
        for (j + 1..n) |k| a[j * n + k] = 0;
    }
}

/// Column `col` of m (n x n) replaced by L^-1 times it.
fn lower(l: []const f64, n: usize, m: []f64, col: usize) void {
    for (0..n) |i| {
        var v = m[i * n + col];
        for (0..i) |k| v -= l[i * n + k] * m[k * n + col];
        m[i * n + col] = v / l[i * n + i];
    }
}

/// w replaced by L^-T w.
fn upper(l: []const f64, n: usize, w: []f64) void {
    var i = n;
    while (i > 0) {
        i -= 1;
        var v = w[i];
        for (i + 1..n) |k| v -= l[k * n + i] * w[k];
        w[i] = v / l[i * n + i];
    }
}

/// A symmetric a (n x n) brought to its eigenvalues on the diagonal by Jacobi rotations; vecs' columns: eigenvectors.
fn jacobi(a: []f64, n: usize, vecs: []f64) void {
    @memset(vecs, 0);
    for (0..n) |i| vecs[i * n + i] = 1;
    for (0..60) |_| {
        var off: f64 = 0;
        var all: f64 = 0;
        for (0..n) |i| for (0..n) |j| {
            all += a[i * n + j] * a[i * n + j];
            if (i != j) off += a[i * n + j] * a[i * n + j];
        };
        if (off <= 1e-24 * all) return;
        for (0..n) |p| for (p + 1..n) |q| {
            const apq = a[p * n + q];
            if (apq == 0) continue;
            const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
            const t = std.math.sign(theta) / (@abs(theta) + @sqrt(theta * theta + 1)) + @as(f64, if (theta == 0) 1 else 0);
            const c = 1 / @sqrt(t * t + 1);
            const s = t * c;
            for (0..n) |k| {
                const akp = a[k * n + p];
                const akq = a[k * n + q];
                a[k * n + p] = c * akp - s * akq;
                a[k * n + q] = s * akp + c * akq;
            }
            for (0..n) |k| {
                const apk = a[p * n + k];
                const aqk = a[q * n + k];
                a[p * n + k] = c * apk - s * aqk;
                a[q * n + k] = s * apk + c * aqk;
            }
            for (0..n) |k| {
                const vkp = vecs[k * n + p];
                const vkq = vecs[k * n + q];
                vecs[k * n + p] = c * vkp - s * vkq;
                vecs[k * n + q] = s * vkp + c * vkq;
            }
        };
    }
}

test "orthonormal rows, and the direction one set of rows outweighs another along" {
    const len = 64;
    var avoid: [3 * len]f32 = undefined;
    var seek: [2 * len]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&avoid) |*v| v.* = prng.random().float(f32) - 0.5;
    for (&seek) |*v| v.* = prng.random().float(f32) - 0.5;
    orthonormal(&avoid, 3, len);
    remove(&seek, 2, &avoid, 3, len);
    orthonormal(&seek, 2, len);
    for (0..2) |q| {
        const row = seek[q * len ..][0..len];
        try std.testing.expectApproxEqAbs(@as(f32, 1), dot(row, row), 1e-4);
        for (0..3) |u| try std.testing.expectApproxEqAbs(@as(f32, 0), dot(row, avoid[u * len ..][0..len]), 1e-4);
    }
    // more is large on axis 0 and 1, less is large on axis 1: the best single direction is axis 0
    var more = [_]f64{ 4, 0, 0, 0, 4, 0, 0, 0, 1 };
    var less = [_]f64{ 1, 0, 0, 0, 9, 0, 0, 0, 1 };
    var out: [3]f32 = undefined;
    try outweigh(std.testing.allocator, &more, &less, 3, 1, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1), @abs(out[0]), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0), out[1], 1e-4);
}
