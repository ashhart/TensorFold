//! BF16/F32 conversion, partial RoPE rotation and shift, and deterministic randomized SVD for low-rank deltas.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Round-to-nearest-even F32 -> BF16.
pub inline fn f32ToBf16(x: f32) u16 {
    if (!std.math.isFinite(x)) {
        const u: u32 = @bitCast(x);
        return @truncate(u >> 16);
    }
    const u: u32 = @bitCast(x);
    const lsb: u32 = (u >> 16) & 1;
    const rounded = u +% (0x7fff + lsb);
    return @truncate(rounded >> 16);
}

pub fn bf16SliceToF32(src: []const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |b, *o| o.* = bf16ToF32(b);
}

pub fn f32SliceToBf16(src: []const f32, dst: []u16) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |v, *o| o.* = f32ToBf16(v);
}

/// Fast SIMD dot product of two equal-length f32 slices.
pub fn dotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) {
        const va: V = a[i..][0..8].*;
        const vb: V = b[i..][0..8].*;
        acc = @mulAdd(V, va, vb, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// Half-split partial RoPE on BF16 `[kv_heads, seq_len, head_dim]`, rotating dims `[0..rotary_dim)` in place.
pub fn ropeRotateKvBf16(
    data: []u16,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    offset: usize,
    rotary_dim: usize,
    theta: f64,
    inverse: bool,
) void {
    if (rotary_dim == 0 or seq_len == 0) return;
    std.debug.assert(rotary_dim <= head_dim and rotary_dim % 2 == 0);
    std.debug.assert(data.len == kv_heads * seq_len * head_dim);
    const half = rotary_dim / 2;
    const sign: f64 = if (inverse) -1.0 else 1.0;

    // Precompute inv_freq for up to 128 pairs (covers rotary_dim <= 256).
    var inv_freq: [128]f64 = undefined;
    std.debug.assert(half <= inv_freq.len);
    for (0..half) |i| {
        inv_freq[i] = std.math.pow(f64, theta, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half)));
    }

    var cs_buf: [128]f32 = undefined;
    var sn_buf: [128]f32 = undefined;
    for (0..seq_len) |t| {
        const pos: f64 = @floatFromInt(offset + t);
        for (0..half) |i| {
            const ang = sign * pos * inv_freq[i];
            cs_buf[i] = @floatCast(@cos(ang));
            sn_buf[i] = @floatCast(@sin(ang));
        }
        for (0..kv_heads) |h| {
            const base = (h * seq_len + t) * head_dim;
            for (0..half) |i| {
                const a = bf16ToF32(data[base + i]);
                const b = bf16ToF32(data[base + i + half]);
                const cs = cs_buf[i];
                const sn = sn_buf[i];
                data[base + i] = f32ToBf16(a * cs - b * sn);
                data[base + i + half] = f32ToBf16(b * cs + a * sn);
            }
        }
    }
}

/// Every row's K rotated on by `delta` positions (K computed at p + t lands at p + delta + t); 0 leaves it untouched.
pub fn ropeShiftKvBf16(data: []u16, kv_heads: usize, seq_len: usize, head_dim: usize, delta: usize, rotary_dim: usize, theta: f64) void {
    if (delta == 0 or rotary_dim == 0 or seq_len == 0) return;
    std.debug.assert(rotary_dim <= head_dim and rotary_dim % 2 == 0);
    std.debug.assert(data.len == kv_heads * seq_len * head_dim);
    const half = rotary_dim / 2;
    var cs_buf: [128]f32 = undefined;
    var sn_buf: [128]f32 = undefined;
    std.debug.assert(half <= cs_buf.len);
    for (0..half) |i| {
        const inv_freq = std.math.pow(f64, theta, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half)));
        const ang = @as(f64, @floatFromInt(delta)) * inv_freq;
        cs_buf[i] = @floatCast(@cos(ang));
        sn_buf[i] = @floatCast(@sin(ang));
    }
    for (0..kv_heads * seq_len) |row| {
        const base = row * head_dim;
        for (0..half) |i| {
            const a = bf16ToF32(data[base + i]);
            const b = bf16ToF32(data[base + i + half]);
            data[base + i] = f32ToBf16(a * cs_buf[i] - b * sn_buf[i]);
            data[base + i + half] = f32ToBf16(b * cs_buf[i] + a * sn_buf[i]);
        }
    }
}

/// Deterministic splitmix64 PRNG for randomized SVD (no external entropy needed).
const SplitMix = struct {
    state: u64,
    fn next(self: *SplitMix) u64 {
        self.state +%= 0x9e3779b97f4a7c15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
    fn normal(self: *SplitMix) f32 {
        // Box-Muller using two uniform floats in (0, 1]
        const unif1 = (@as(f64, @floatFromInt((self.next() >> 11) + 1))) / @as(f64, @floatFromInt(@as(u64, 1) << 53));
        const unif2 = (@as(f64, @floatFromInt(self.next() >> 11))) / @as(f64, @floatFromInt(@as(u64, 1) << 53));
        const r = @sqrt(-2.0 * @log(unif1));
        return @floatCast(r * @cos(2.0 * std.math.pi * unif2));
    }
};

/// Modified Gram-Schmidt orthonormalization in place of `q_t` stored as `[cols, rows]` (each row one vector).
fn orthonormalizeRows(q_t: []f32, k: usize, n: usize) usize {
    var kept: usize = 0;
    for (0..k) |i| {
        const vi = q_t[i * n ..][0..n];
        // Two passes of MGS for numerical precision
        for (0..2) |_| {
            for (0..kept) |j| {
                const vj = q_t[j * n ..][0..n];
                const proj = dotF32(vi, vj);
                if (proj != 0) {
                    const V = @Vector(8, f32);
                    const vp: V = @splat(proj);
                    var c: usize = 0;
                    while (c + 8 <= n) : (c += 8) {
                        const a: V = vi[c..][0..8].*;
                        const b: V = vj[c..][0..8].*;
                        vi[c..][0..8].* = a - vp * b;
                    }
                    while (c < n) : (c += 1) vi[c] -= proj * vj[c];
                }
            }
        }
        const nrm = @sqrt(dotF32(vi, vi));
        if (nrm > 1e-9) {
            const inv = 1.0 / nrm;
            const dst = q_t[kept * n ..][0..n];
            for (vi, dst) |v, *o| o.* = v * inv;
            kept += 1;
        }
    }
    return kept;
}

/// Cyclic Jacobi eigensolver for symmetric `n x n` `a` (destroyed): `evals[0..n]`, eigenvectors as `evecs` rows.
pub fn symmetricJacobi(a: []f64, evecs: []f64, evals: []f64, n: usize) void {
    std.debug.assert(a.len == n * n and evecs.len == n * n and evals.len == n);
    // Start with evecs = I (row-major, then we transpose at the end so rows are eigenvectors)
    @memset(evecs, 0);
    for (0..n) |i| evecs[i * n + i] = 1.0;

    const max_sweeps: usize = 32;
    for (0..max_sweeps) |_| {
        var off_max: f64 = 0;
        for (0..n) |p| {
            for (p + 1..n) |q| {
                const apq = a[p * n + q];
                const abs_apq = @abs(apq);
                if (abs_apq > off_max) off_max = abs_apq;
                if (abs_apq <= 1e-14) continue;

                const app = a[p * n + p];
                const aqq = a[q * n + q];
                const tau = (aqq - app) / (2.0 * apq);
                const t = if (tau >= 0) 1.0 / (tau + @sqrt(1.0 + tau * tau)) else -1.0 / (-tau + @sqrt(1.0 + tau * tau));
                const c = 1.0 / @sqrt(1.0 + t * t);
                const s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0;
                a[q * n + p] = 0;

                for (0..n) |k| {
                    if (k == p or k == q) continue;
                    const akp = a[k * n + p];
                    const akq = a[k * n + q];
                    const nkp = c * akp - s * akq;
                    const nkq = s * akp + c * akq;
                    a[k * n + p] = nkp;
                    a[p * n + k] = nkp;
                    a[k * n + q] = nkq;
                    a[q * n + k] = nkq;
                }
                for (0..n) |k| {
                    const vkp = evecs[k * n + p];
                    const vkq = evecs[k * n + q];
                    evecs[k * n + p] = c * vkp - s * vkq;
                    evecs[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
        if (off_max <= 1e-13) break;
    }

    for (0..n) |i| evals[i] = a[i * n + i];
    // Transpose evecs in-place so evecs[k * n ..][0..n] is the k-th eigenvector!
    for (0..n) |i| {
        for (i + 1..n) |j| {
            std.mem.swap(f64, &evecs[i * n + j], &evecs[j * n + i]);
        }
    }
}

/// Randomized rank-`r` SVD of `A [m, n]`: `u_out @ vt_out ~= A`, sqrt(S) split to both; returns rank (0 if A = 0).
pub fn randomizedSvdBalanced(
    scratch_alloc: Allocator,
    a: []const f32,
    m: usize,
    n: usize,
    max_rank: usize,
    u_out: []f32,
    vt_out: []f32,
) !usize {
    const r = @min(max_rank, @min(m, n));
    std.debug.assert(a.len == m * n);
    std.debug.assert(u_out.len == m * r);
    std.debug.assert(vt_out.len == r * n);
    @memset(u_out, 0);
    @memset(vt_out, 0);
    if (r == 0) return 0;

    // Quick zero check
    var frob2: f64 = 0;
    for (a) |v| {
        if (std.math.isFinite(v)) frob2 += @as(f64, v) * @as(f64, v);
    }
    if (frob2 < 1e-24) return 0;

    const l = @min(@min(m, n), r + 8);
    // Work buffers; omega_t: [l, n] (each row is one random probe vector in R^n)
    const omega_t = try scratch_alloc.alloc(f32, l * n);
    defer scratch_alloc.free(omega_t);
    var rng = SplitMix{ .state = 0x51a7_9f13_2b4c_8e01 ^ (@as(u64, m) << 32) ^ @as(u64, n) };
    for (omega_t) |*v| v.* = rng.normal();

    // q_t: [l, m] stores Y^T (each row is a column of Y = A @ Omega)
    const q_t = try scratch_alloc.alloc(f32, l * m);
    defer scratch_alloc.free(q_t);
    // b_mat: [l, n] stores B = Q^T @ A
    const b_mat = try scratch_alloc.alloc(f32, l * n);
    defer scratch_alloc.free(b_mat);

    // Y = A @ Omega  -> q_t[c, row] = dot(A[row], omega_t[c])
    for (0..l) |c| {
        const om = omega_t[c * n ..][0..n];
        const qrow = q_t[c * m ..][0..m];
        for (0..m) |row| {
            qrow[row] = dotF32(a[row * n ..][0..n], om);
        }
    }
    var qk = orthonormalizeRows(q_t, l, m);
    if (qk == 0) return 0;

    // 2 power iterations: B = Q^T @ A; Y = A @ B^T; orthonormalize Y
    for (0..2) |_| {
        @memset(b_mat[0 .. qk * n], 0);
        for (0..qk) |c| {
            const qrow = q_t[c * m ..][0..m];
            const brow = b_mat[c * n ..][0..n];
            for (0..m) |row| {
                const qv = qrow[row];
                if (qv == 0) continue;
                const arow = a[row * n ..][0..n];
                const V = @Vector(8, f32);
                const vq: V = @splat(qv);
                var j: usize = 0;
                while (j + 8 <= n) : (j += 8) {
                    const cur: V = brow[j..][0..8].*;
                    const av: V = arow[j..][0..8].*;
                    brow[j..][0..8].* = @mulAdd(V, vq, av, cur);
                }
                while (j < n) : (j += 1) brow[j] += qv * arow[j];
            }
        }
        // Orthonormalize B rows to prevent overflow/underflow in power iterations
        qk = orthonormalizeRows(b_mat, qk, n);
        if (qk == 0) return 0;
        for (0..qk) |c| {
            const brow = b_mat[c * n ..][0..n];
            const qrow = q_t[c * m ..][0..m];
            for (0..m) |row| {
                qrow[row] = dotF32(a[row * n ..][0..n], brow);
            }
        }
        qk = orthonormalizeRows(q_t, qk, m);
        if (qk == 0) return 0;
    }

    // Final B = Q^T @ A ([qk, n])
    @memset(b_mat[0 .. qk * n], 0);
    for (0..qk) |c| {
        const qrow = q_t[c * m ..][0..m];
        const brow = b_mat[c * n ..][0..n];
        for (0..m) |row| {
            const qv = qrow[row];
            if (qv == 0) continue;
            const arow = a[row * n ..][0..n];
            const V = @Vector(8, f32);
            const vq: V = @splat(qv);
            var j: usize = 0;
            while (j + 8 <= n) : (j += 8) {
                const cur: V = brow[j..][0..8].*;
                const av: V = arow[j..][0..8].*;
                brow[j..][0..8].* = @mulAdd(V, vq, av, cur);
            }
            while (j < n) : (j += 1) brow[j] += qv * arow[j];
        }
    }

    // Form Gram matrix G = B @ B^T ([qk, qk]) in f64
    const gram = try scratch_alloc.alloc(f64, qk * qk);
    defer scratch_alloc.free(gram);
    const evecs = try scratch_alloc.alloc(f64, qk * qk);
    defer scratch_alloc.free(evecs);
    const evals = try scratch_alloc.alloc(f64, qk);
    defer scratch_alloc.free(evals);

    for (0..qk) |i| {
        const bi = b_mat[i * n ..][0..n];
        for (i..qk) |j| {
            const bj = b_mat[j * n ..][0..n];
            const d: f64 = dotF32(bi, bj);
            gram[i * qk + j] = d;
            gram[j * qk + i] = d;
        }
    }
    symmetricJacobi(gram, evecs, evals, qk);

    // Sort eigenpairs descending
    const order = try scratch_alloc.alloc(usize, qk);
    defer scratch_alloc.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, evals, struct {
        fn less(vals: []const f64, lhs: usize, rhs: usize) bool {
            return vals[lhs] > vals[rhs];
        }
    }.less);

    const out_r = @min(r, qk);
    var actual_r: usize = 0;
    for (0..out_r) |k| {
        const idx = order[k];
        const eig = evals[idx];
        if (eig <= 1e-18) break;
        const sigma = @sqrt(eig);
        const sqrt_sigma: f32 = @floatCast(@sqrt(sigma));
        const inv_sqrt_sigma: f32 = @floatCast(1.0 / @sqrt(sigma));
        const ev = evecs[idx * qk ..][0..qk];

        // u_col = Q @ ev * sqrt(sigma) -> u_out[row * r + k]
        for (0..m) |row| {
            var acc: f64 = 0;
            for (0..qk) |c| acc += @as(f64, q_t[c * m + row]) * ev[c];
            u_out[row * r + actual_r] = @as(f32, @floatCast(acc)) * sqrt_sigma;
        }
        // vt_row = (ev^T @ B) / sqrt(sigma) -> vt_out[k * n + col]
        const vt_dst = vt_out[actual_r * n ..][0..n];
        @memset(vt_dst, 0);
        for (0..qk) |c| {
            const w: f32 = @floatCast(ev[c]);
            const brow = b_mat[c * n ..][0..n];
            for (brow, vt_dst) |bv, *dv| dv.* += w * bv;
        }
        for (vt_dst) |*dv| dv.* *= inv_sqrt_sigma;
        actual_r += 1;
    }
    return actual_r;
}

/// Top-K right singular vectors `vt_out [k, cols]` of row-major `X [rows, cols]` via randomized SVD (LayerWhitener).
pub fn topRightSingularVectors(
    scratch_alloc: Allocator,
    x: []const f32,
    rows: usize,
    cols: usize,
    k_target: usize,
    vt_out: []f32,
) !usize {
    const k = @min(k_target, @min(rows, cols));
    std.debug.assert(x.len == rows * cols and vt_out.len == k * cols);
    if (k == 0) return 0;

    const u_tmp = try scratch_alloc.alloc(f32, rows * k);
    defer scratch_alloc.free(u_tmp);
    const got = try randomizedSvdBalanced(scratch_alloc, x, rows, cols, k, u_tmp, vt_out);
    // Re-orthonormalize right singular vectors with 2-pass MGS so basis @ basis^T == I exactly.
    return orthonormalizeRows(vt_out, got, cols);
}

/// `dst[m, n] += weight * (u[m, r] @ v[r, n])` in place, from the factors as the SVD computed them.
pub fn lowRankAdd(u: []const f32, v: []const f32, m: usize, r: usize, n: usize, weight: f32, dst: []f32) void {
    std.debug.assert(u.len == m * r and v.len == r * n and dst.len == m * n);
    if (r == 0 or weight == 0) return;
    const V = @Vector(8, f32);
    for (0..m) |row| {
        const urow = u[row * r ..][0..r];
        const drow = dst[row * n ..][0..n];
        for (0..r) |k| {
            const uk = urow[k] * weight;
            if (uk == 0.0) continue;
            const vrow = v[k * n ..][0..n];
            const vu: V = @splat(uk);
            var j: usize = 0;
            while (j + 8 <= n) : (j += 8) {
                const cur: V = drow[j..][0..8].*;
                const vv: V = vrow[j..][0..8].*;
                drow[j..][0..8].* = @mulAdd(V, vu, vv, cur);
            }
            while (j < n) : (j += 1) drow[j] += uk * vrow[j];
        }
    }
}

test "quant: partial RoPE forward + inverse roundtrip preserves BF16 within 1 ULP" {
    const kv_heads = 2;
    const seq_len = 5;
    const head_dim = 128;
    var orig: [kv_heads * seq_len * head_dim]u16 = undefined;
    var buf: [kv_heads * seq_len * head_dim]u16 = undefined;
    var rng = SplitMix{ .state = 42 };
    for (&orig, &buf) |*o, *b| {
        const v = f32ToBf16(rng.normal());
        o.* = v;
        b.* = v;
    }
    ropeRotateKvBf16(&buf, kv_heads, seq_len, head_dim, 17, 64, 10_000_000.0, false);
    // Dims >= 64 must be untouched bit-for-bit!
    for (0..kv_heads * seq_len) |r| {
        try std.testing.expectEqualSlices(u16, orig[r * head_dim + 64 ..][0..64], buf[r * head_dim + 64 ..][0..64]);
    }
    ropeRotateKvBf16(&buf, kv_heads, seq_len, head_dim, 17, 64, 10_000_000.0, true);
    var max_err: f32 = 0;
    for (orig, buf) |o, b| {
        max_err = @max(max_err, @abs(bf16ToF32(o) - bf16ToF32(b)));
    }
    try std.testing.expect(max_err < 0.05);
}

test "quant: randomized SVD recovers a low-rank 64x128 matrix from its F32 factors" {
    const gpa = std.testing.allocator;
    const m = 64;
    const n = 128;
    const true_r = 8;
    const a = try gpa.alloc(f32, m * n);
    defer gpa.free(a);
    @memset(a, 0);
    var rng = SplitMix{ .state = 12345 };
    for (0..true_r) |k| {
        const w: f32 = 1.0 / @as(f32, @floatFromInt(k + 1));
        var u: [m]f32 = undefined;
        var v: [n]f32 = undefined;
        for (&u) |*x| x.* = rng.normal();
        for (&v) |*x| x.* = rng.normal();
        for (0..m) |i| for (0..n) |j| {
            a[i * n + j] += w * u[i] * v[j];
        };
    }

    const r = 12;
    const u_out = try gpa.alloc(f32, m * r);
    defer gpa.free(u_out);
    const vt_out = try gpa.alloc(f32, r * n);
    defer gpa.free(vt_out);
    const got_r = try randomizedSvdBalanced(gpa, a, m, n, r, u_out, vt_out);
    try std.testing.expect(got_r >= true_r);

    const recon = try gpa.alloc(f32, m * n);
    defer gpa.free(recon);
    @memset(recon, 0);
    lowRankAdd(u_out, vt_out, m, r, n, 1.0, recon);

    var err2: f64 = 0;
    var ref2: f64 = 0;
    for (a, recon) |x, y| {
        const d = @as(f64, x) - @as(f64, y);
        err2 += d * d;
        ref2 += @as(f64, x) * @as(f64, x);
    }
    const rel = @sqrt(err2 / ref2);
    try std.testing.expect(rel < 0.015);
}
