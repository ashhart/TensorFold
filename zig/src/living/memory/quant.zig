//! BF16/F32 conversion, NaN-safe symmetric Q8 quantization, partial RoPE rotation,
//! and deterministic randomized SVD for low-rank recurrent state deltas.
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

/// NaN-safe symmetric int8 quantization along the last axis with F32 per-row scales.
/// Never underflows to 0/0 NaN -> -128! Exact zeros stay exact zeros.
pub fn quantizeRowsF32(src: []const f32, rows: usize, cols: usize, out_q: []i8, out_s: []f32) void {
    std.debug.assert(src.len == rows * cols);
    std.debug.assert(out_q.len == rows * cols);
    std.debug.assert(out_s.len == rows);
    for (0..rows) |r| {
        const row = src[r * cols ..][0..cols];
        const qrow = out_q[r * cols ..][0..cols];
        var max_abs: f32 = 0;
        for (row) |v| {
            if (std.math.isFinite(v)) {
                const av = @abs(v);
                if (av > max_abs) max_abs = av;
            }
        }
        if (max_abs > 1e-12) {
            const scale = max_abs / 127.0;
            const inv = 127.0 / max_abs;
            out_s[r] = scale;
            for (row, qrow) |v, *q| {
                const f = if (std.math.isFinite(v)) v * inv else 0.0;
                const clamped = std.math.clamp(@round(f), -127.0, 127.0);
                q.* = @intFromFloat(clamped);
            }
        } else {
            out_s[r] = 0.0;
            @memset(qrow, 0);
        }
    }
}

/// NaN-safe symmetric int8 quantization directly from BF16 rows.
pub fn quantizeRowsBf16(src: []const u16, rows: usize, cols: usize, out_q: []i8, out_s: []f32) void {
    std.debug.assert(src.len == rows * cols);
    std.debug.assert(out_q.len == rows * cols);
    std.debug.assert(out_s.len == rows);
    for (0..rows) |r| {
        const row = src[r * cols ..][0..cols];
        const qrow = out_q[r * cols ..][0..cols];
        var max_abs: f32 = 0;
        for (row) |bits| {
            const v = bf16ToF32(bits);
            if (std.math.isFinite(v)) {
                const av = @abs(v);
                if (av > max_abs) max_abs = av;
            }
        }
        if (max_abs > 1e-12) {
            const scale = max_abs / 127.0;
            const inv = 127.0 / max_abs;
            out_s[r] = scale;
            for (row, qrow) |bits, *q| {
                const v = bf16ToF32(bits);
                const f = if (std.math.isFinite(v)) v * inv else 0.0;
                const clamped = std.math.clamp(@round(f), -127.0, 127.0);
                q.* = @intFromFloat(clamped);
            }
        } else {
            out_s[r] = 0.0;
            @memset(qrow, 0);
        }
    }
}

pub fn dequantizeRowsF32(q: []const i8, s: []const f32, rows: usize, cols: usize, out: []f32) void {
    std.debug.assert(q.len == rows * cols and s.len == rows and out.len == rows * cols);
    for (0..rows) |r| {
        const sc = s[r];
        const qrow = q[r * cols ..][0..cols];
        const orow = out[r * cols ..][0..cols];
        if (sc == 0.0) {
            @memset(orow, 0.0);
            continue;
        }
        for (qrow, orow) |qv, *ov| ov.* = @as(f32, @floatFromInt(qv)) * sc;
    }
}

pub fn dequantizeRowsBf16(q: []const i8, s: []const f32, rows: usize, cols: usize, out: []u16) void {
    std.debug.assert(q.len == rows * cols and s.len == rows and out.len == rows * cols);
    for (0..rows) |r| {
        const sc = s[r];
        const qrow = q[r * cols ..][0..cols];
        const orow = out[r * cols ..][0..cols];
        if (sc == 0.0) {
            @memset(orow, 0);
            continue;
        }
        for (qrow, orow) |qv, *ov| ov.* = f32ToBf16(@as(f32, @floatFromInt(qv)) * sc);
    }
}

pub const q4_group: usize = 32;

/// Symmetric 4-bit group-32 quantization directly from BF16 rows (`cols % 32 == 0`).
/// Packs 2 signed nibbles `[-7, 7]` (biased by `+8` into `[1..15]`, zero -> `8`) per byte,
/// with 1 BF16 scale per 32-element group (0.5625 bytes/elem = 1.8x smaller than Q8!).
pub fn quantizeRowsBf16Q4(src: []const u16, rows: usize, cols: usize, out_q: []u8, out_s: []u16) void {
    std.debug.assert(cols % q4_group == 0);
    const groups_per_row = cols / q4_group;
    const packed_per_row = cols / 2;
    std.debug.assert(src.len == rows * cols);
    std.debug.assert(out_q.len == rows * packed_per_row);
    std.debug.assert(out_s.len == rows * groups_per_row);

    for (0..rows) |r| {
        const row = src[r * cols ..][0..cols];
        const qrow = out_q[r * packed_per_row ..][0..packed_per_row];
        const srow = out_s[r * groups_per_row ..][0..groups_per_row];
        for (0..groups_per_row) |g| {
            const gsrc = row[g * q4_group ..][0..q4_group];
            const gq = qrow[g * (q4_group / 2) ..][0 .. q4_group / 2];
            var max_abs: f32 = 0;
            for (gsrc) |bits| {
                const v = bf16ToF32(bits);
                if (std.math.isFinite(v)) {
                    const av = @abs(v);
                    if (av > max_abs) max_abs = av;
                }
            }
            if (max_abs > 1e-12) {
                const sc_bf = f32ToBf16(max_abs / 7.0);
                const sc = bf16ToF32(sc_bf);
                if (sc > 0.0) {
                    const inv = 1.0 / sc;
                    srow[g] = sc_bf;
                    for (0..q4_group / 2) |p| {
                        const v0 = bf16ToF32(gsrc[2 * p]);
                        const v1 = bf16ToF32(gsrc[2 * p + 1]);
                        const f0 = if (std.math.isFinite(v0)) v0 * inv else 0.0;
                        const f1 = if (std.math.isFinite(v1)) v1 * inv else 0.0;
                        const q0: i32 = @intFromFloat(std.math.clamp(@round(f0), -7.0, 7.0));
                        const q1: i32 = @intFromFloat(std.math.clamp(@round(f1), -7.0, 7.0));
                        const lo: u8 = @intCast(q0 + 8);
                        const hi: u8 = @intCast(q1 + 8);
                        gq[p] = (lo & 0x0f) | ((hi & 0x0f) << 4);
                    }
                } else {
                    srow[g] = 0;
                    @memset(gq, 0x88);
                }
            } else {
                srow[g] = 0;
                @memset(gq, 0x88);
            }
        }
    }
}

/// Dequantizes symmetric 4-bit group-32 packed rows back to BF16 (`u16`).
pub fn dequantizeRowsBf16Q4(q: []const u8, s: []const u16, rows: usize, cols: usize, out: []u16) void {
    std.debug.assert(cols % q4_group == 0);
    const groups_per_row = cols / q4_group;
    const packed_per_row = cols / 2;
    std.debug.assert(q.len == rows * packed_per_row);
    std.debug.assert(s.len == rows * groups_per_row);
    std.debug.assert(out.len == rows * cols);

    for (0..rows) |r| {
        const qrow = q[r * packed_per_row ..][0..packed_per_row];
        const srow = s[r * groups_per_row ..][0..groups_per_row];
        const orow = out[r * cols ..][0..cols];
        for (0..groups_per_row) |g| {
            const sc = bf16ToF32(srow[g]);
            const gd = orow[g * q4_group ..][0..q4_group];
            if (sc == 0.0) {
                @memset(gd, 0);
                continue;
            }
            const gq = qrow[g * (q4_group / 2) ..][0 .. q4_group / 2];
            for (0..q4_group / 2) |p| {
                const byte = gq[p];
                const q0: i32 = @as(i32, byte & 0x0f) - 8;
                const q1: i32 = @as(i32, (byte >> 4) & 0x0f) - 8;
                gd[2 * p] = f32ToBf16(@as(f32, @floatFromInt(q0)) * sc);
                gd[2 * p + 1] = f32ToBf16(@as(f32, @floatFromInt(q1)) * sc);
            }
        }
    }
}

/// Half-split partial RoPE matching TensorFold's qwen35_rotate and q27_rope bit-for-bit.
/// `data` has layout `[kv_heads, seq_len, head_dim]` in BF16; rotates dims `[0..rotary_dim)` in place.
/// When `rotary_dim == 0` (e.g. Nemotron NoPE), this is an exact no-op.
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

/// Modified Gram-Schmidt column orthonormalization in-place for column-major-ish `[rows, cols]` row-major matrix
/// stored as `[cols, rows]` (each row of `q_t` is one orthonormal vector of length `rows`).
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

/// Cyclic Jacobi eigendecomposition of an `n x n` symmetric matrix `A` (destroyed in place).
/// Writes eigenvalues into `evals[0..n]` and eigenvectors as rows of `evecs[n, n]` (`evecs[k]` is eigenvector `k`).
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

/// Randomized low-rank SVD of `A [m, n]` (row-major f32) to target rank `r <= min(max_rank, m, n)`.
/// Writes `u_out [m, r]` (= U_r * sqrt(S_r)) and `vt_out [r, n]` (= sqrt(S_r) * V_r^T) so
/// `u_out @ vt_out ≈ A`, with balanced singular values minimizing Q8 quantization error!
/// Returns actual rank `<= r` (0 if A is identically zero).
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
    // Work buffers
    // omega_t: [l, n] (each row is one random probe vector in R^n)
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

/// Top-K right singular vectors `vt_out [k, cols]` of row-major `X [rows, cols]` via randomized SVD.
/// Used by LayerWhitener to strip the top-K background principal components in milliseconds.
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

/// Reconstructs `dst[m, n] += weight * (deq(u_q, u_s)[m, r] @ deq(v_q, v_s)[r, n])` in-place.
pub fn reconstructSvdAdd(
    u_q: []const i8,
    u_s: []const f32,
    v_q: []const i8,
    v_s: []const f32,
    m: usize,
    r: usize,
    n: usize,
    weight: f32,
    dst: []f32,
    v_f32_scratch: []f32,
) void {
    std.debug.assert(u_q.len == m * r and u_s.len == m);
    std.debug.assert(v_q.len == r * n and v_s.len == r);
    std.debug.assert(dst.len == m * n and v_f32_scratch.len >= r * n);
    if (r == 0 or weight == 0) return;

    // Dequantize V once ([r, n])
    dequantizeRowsF32(v_q, v_s, r, n, v_f32_scratch[0 .. r * n]);

    const V = @Vector(8, f32);
    for (0..m) |row| {
        const sc = u_s[row] * weight;
        if (sc == 0.0) continue;
        const urow = u_q[row * r ..][0..r];
        const drow = dst[row * n ..][0..n];
        for (0..r) |k| {
            const uk = @as(f32, @floatFromInt(urow[k])) * sc;
            if (uk == 0.0) continue;
            const vrow = v_f32_scratch[k * n ..][0..n];
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

test "quant: NaN-safe Q8 keeps zeros exact and round-trips finite values accurately" {
    const src = [_]f32{ 0, 0, 0, 0, 1.5, -3.0, 0.25, std.math.nan(f32) };
    var q: [8]i8 = undefined;
    var s: [2]f32 = undefined;
    quantizeRowsF32(&src, 2, 4, &q, &s);
    try std.testing.expectEqual(@as(f32, 0.0), s[0]);
    try std.testing.expectEqualSlices(i8, &.{ 0, 0, 0, 0 }, q[0..4]);
    var out: [8]f32 = undefined;
    dequantizeRowsF32(&q, &s, 2, 4, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out[4], 0.03);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), out[5], 0.03);
    try std.testing.expectEqual(@as(f32, 0.0), out[7]);
}

test "quant: NaN-safe group-32 Q4 BF16 preserves zeros and round-trips accurately" {
    var src: [64]u16 = @splat(0);
    var rng = SplitMix{ .state = 77 };
    for (32..64) |i| src[i] = f32ToBf16(rng.normal() * 1.5);
    var q: [32]u8 = undefined;
    var s: [2]u16 = undefined;
    quantizeRowsBf16Q4(&src, 1, 64, &q, &s);
    try std.testing.expectEqual(@as(u16, 0), s[0]);
    var out: [64]u16 = undefined;
    dequantizeRowsBf16Q4(&q, &s, 1, 64, &out);
    for (0..32) |i| try std.testing.expectEqual(@as(u16, 0), out[i]);
    var err2: f64 = 0;
    var ref2: f64 = 0;
    for (32..64) |i| {
        const a = bf16ToF32(src[i]);
        const b = bf16ToF32(out[i]);
        err2 += @as(f64, a - b) * @as(f64, a - b);
        ref2 += @as(f64, a) * @as(f64, a);
    }
    try std.testing.expect(@sqrt(err2 / ref2) < 0.12);
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

test "quant: randomized SVD + Q8 recovers low-rank 64x128 matrix under 0.5% relative error" {
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

    const uq = try gpa.alloc(i8, m * r);
    defer gpa.free(uq);
    const us = try gpa.alloc(f32, m);
    defer gpa.free(us);
    const vq = try gpa.alloc(i8, r * n);
    defer gpa.free(vq);
    const vs = try gpa.alloc(f32, r);
    defer gpa.free(vs);
    quantizeRowsF32(u_out, m, r, uq, us);
    quantizeRowsF32(vt_out, r, n, vq, vs);

    const recon = try gpa.alloc(f32, m * n);
    defer gpa.free(recon);
    @memset(recon, 0);
    const scratch = try gpa.alloc(f32, r * n);
    defer gpa.free(scratch);
    reconstructSvdAdd(uq, us, vq, vs, m, r, n, 1.0, recon, scratch);

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
