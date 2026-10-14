//! The sparse checks' helpers: fixture access and float64 references for the DSA sparse indexer and its selection.

const std = @import("std");
const kref = @import("glm_kda_ref.zig");

pub const D = 128; // index key dims
pub const H = 32; // index heads
pub const K = 512; // pools a row keeps
pub const pool = 4; // tokens a pool

pub const Manifest = kref.Manifest;

pub fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    return kref.readFile(a, io, dir, name);
}

/// A float64 as the kernel's bf16 rounding (round to nearest even) sees it, widened back to f64.
pub fn bf16rn(x: f64) f64 {
    const u: u32 = @bitCast(@as(f32, @floatCast(x)));
    const rounded = u + 0x7FFF + ((u >> 16) & 1);
    return kref.bf(@intCast(rounded >> 16));
}

/// float64 index LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * w + b over bf16 inputs, rows of ikr.
pub fn indexRef(a: std.mem.Allocator, ikr: []const u16, ikr_stride: usize, lnw: []const u16, lnb: []const u16, eps: f64, rows: usize) ![]f64 {
    const y = try a.alloc(f64, rows * D);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var sum: f64 = 0;
        for (0..D) |i| sum += kref.bf(ikr[r * ikr_stride + i]);
        const mean = sum / 128.0;
        var vsum: f64 = 0;
        for (0..D) |i| {
            const x = kref.bf(ikr[r * ikr_stride + i]);
            vsum += (x - mean) * (x - mean);
        }
        const dv = @sqrt(vsum / 128.0 + eps);
        for (0..D) |i| {
            const x = kref.bf(ikr[r * ikr_stride + i]);
            y[r * D + i] = (x - mean) / dv * f16v(lnw[i]) + f16v(lnb[i]);
        }
    }
    return y;
}

/// An fp16 value as f64 (the k_norm weights' checkpoint dtype).
pub fn f16v(u: u16) f64 {
    return @as(f64, @as(f16, @bitCast(u)));
}

/// float64 pooled keys: the gated softmax over a pool's four gate rows times the keys, bf16 rounded, summed.
pub fn poolRef(a: std.mem.Allocator, ig: []const u16, ape: []const u16, ik: []const u16, p0: usize, p1: usize) ![]f64 {
    const out = try a.alloc(f64, (p1 - p0) * D);
    var p = p0;
    while (p < p1) : (p += 1) {
        var l: [4][D]f64 = undefined;
        var m: [D]f64 = undefined;
        for (0..D) |i| {
            var mx: f64 = 0;
            for (0..4) |j| {
                const g = kref.bf(ig[(4 * p + j) * D + i]);
                const ap = kref.bf(ape[j * D + i]);
                l[j][i] = g + ap;
                if (j == 0 or l[j][i] > mx) mx = l[j][i];
            }
            m[i] = mx;
        }
        for (0..D) |i| {
            var e: [4]f64 = undefined;
            for (0..4) |j| e[j] = @exp(l[j][i] - m[i]);
            const s = ((e[0] + e[1]) + e[2]) + e[3];
            var t: [4]f64 = undefined;
            for (0..4) |j| {
                const k = kref.bf(ik[(4 * p + j) * D + i]);
                t[j] = bf16rn(bf16rn(e[j] / s) * k);
            }
            out[(p - p0) * D + i] = ((t[0] + t[1]) + t[2]) + t[3];
        }
    }
    return out;
}

/// float64 pool scores: s_p = sum_h w_h * max(scale * (q_h . k_p), 0) over bf16 inputs.
pub fn scoresRef(a: std.mem.Allocator, qi: []const u16, qi_stride: usize, wts: []const u16, w_stride: usize, pk: []const u16, rows: usize, np: usize, scale: f64, wscale: f64, pos0: i64) ![]f64 {
    const out = try a.alloc(f64, rows * np);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const npool = @divTrunc(pos0 + @as(i64, @intCast(r)) + 1, 4);
        var p: usize = 0;
        while (p < np) : (p += 1) {
            if (p >= npool) {
                out[r * np + p] = -std.math.inf(f64); // pools past the row's last visible complete pool
                continue;
            }
            var s: f64 = 0;
            for (0..H) |h| {
                var dot: f64 = 0;
                for (0..D) |i| {
                    const q = kref.bf(qi[r * qi_stride + h * D + i]);
                    const k = kref.bf(pk[p * D + i]);
                    dot += q * k;
                }
                const w = kref.bf(wts[r * w_stride + h]);
                s += w * wscale * @max(dot * scale, 0.0);
            }
            out[r * np + p] = s;
        }
    }
    return out;
}

/// The selection's float64-exact reference: order-key sort of the fp32 scores (ties to lower pool), k best ascending.
pub fn selectRef(a: std.mem.Allocator, scores: []const f32, np: usize, k: usize, pos: i64, rows: usize, tokens: []i32, counts: []i32) !void {
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const idx = try a.alloc(usize, np);
        defer a.free(idx);
        for (0..np) |i| idx[i] = i;
        const sb = scores[r * np .. (r + 1) * np];
        const S = struct {
            sb: []const f32,
            fn lessThan(self: @This(), x: usize, y: usize) bool {
                const bx = orderKey(self.sb[x]);
                const by = orderKey(self.sb[y]);
                return if (bx == by) x < y else bx > by; // best first, ties to the lower pool
            }
        };
        std.mem.sort(usize, idx, S{ .sb = sb }, S.lessThan);
        std.mem.sort(usize, idx[0..k], {}, std.sort.asc(usize)); // Python's select returns pools ascending
        const q = pos + @as(i64, @intCast(r));
        const npool = @divTrunc(q + 1, 4);
        for (0..k) |i| {
            const p: i64 = @intCast(idx[i]);
            for (0..pool) |j| tokens[r * 2051 + i * pool + j] = @intCast(p * pool + @as(i64, @intCast(j)));
        }
        var tail_ok: i32 = 0;
        for (0..3) |j| {
            const tail = npool * 4 + @as(i64, @intCast(j));
            tokens[r * 2051 + 2048 + j] = if (tail <= q) @intCast(tail) else -1;
            if (tail <= q) tail_ok += 1;
        }
        counts[r] = if (npool > K) 2048 + tail_ok else 0;
    }
}

/// fp8 e4m3fn (finite, no inf) as f64; 0x7F/0xFF are NaN.
pub fn fp8e4m3(code: u8) f64 {
    const s: f64 = if (code & 0x80 != 0) -1.0 else 1.0;
    const e: u4 = @truncate(code >> 3);
    const m: u3 = @truncate(code);
    return if (e == 15 and m == 7)
        std.math.nan(f64)
    else if (e == 0)
        s * @as(f64, @floatFromInt(m)) * 0.001953125 // subnormal: mantissa / 8 * 2^-6 = m * 2^-9
    else
        s * (1.0 + @as(f64, @floatFromInt(m)) / 8.0) * std.math.pow(f64, 2.0, @as(f64, @floatFromInt(e)) - 7.0);
}

/// float64 sparse attention: the selected tokens' fp8 latent rows, a plain running softmax per (row, head).
pub fn sparseRef(a: std.mem.Allocator, qa: []const u16, cache: []const u8, scales: []const f32, tokens: []const i32, counts: []const i32, heads: usize, lw: usize, scale: f64) ![]f64 {
    const rows = counts.len;
    const out = try a.alloc(f64, rows * heads * lw);
    @memset(out, 0); // rows with count 0 stay untouched by the kernel: the test compares the rest
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        if (counts[r] == 0) continue; // the kernel leaves these rows' out untouched
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var m: f64 = -std.math.inf(f64);
            var l: f64 = 0;
            const o = try a.alloc(f64, lw);
            defer a.free(o);
            @memset(o, 0);
            for (tokens[r * 2051 ..][0..2051]) |tok| {
                if (tok < 0) continue;
                const t: usize = @intCast(tok);
                const qrow = qa[(r * heads + h) * lw ..][0..lw];
                var dot: f64 = 0;
                for (0..lw) |i| dot += kref.bf(qrow[i]) * (fp8e4m3(cache[t * lw + i]) * scales[t]);
                const s = dot * scale;
                const nm = @max(m, s);
                const al = if (m == -std.math.inf(f64)) 0 else @exp(m - nm);
                const p = @exp(s - nm);
                const krow = cache[t * lw ..][0..lw];
                for (0..lw) |i| o[i] = o[i] * al + p * (fp8e4m3(krow[i]) * scales[t]);
                l = l * al + p;
                m = nm;
            }
            if (l > 0) {
                for (0..lw) |i| out[(r * heads + h) * lw + i] = o[i] / l;
            } else {
                for (0..lw) |i| out[(r * heads + h) * lw + i] = 0;
            }
        }
    }
    return out;
}

fn orderKey(s: f32) u32 {
    const b: u32 = @bitCast(s + 0.0);
    const arith: u32 = 0 -% (b >> 31); // 0 or 0xFFFFFFFF: the sign's arithmetic extension
    return b ^ (arith | 0x80000000);
}
