//! Float64 references for the GLM-5.3-Flash latent attention checks (absorb/expand dots and dense attention).

const std = @import("std");
const kref = @import("glm_kda_ref.zig");
const sref = @import("glm_sparse_ref.zig");

pub const fp8c = sref.fp8e4m3;
pub const bf = kref.bf;

/// A dequantized fp8 cache value as the engine sees it: code times the f32 scale, rounded through bf16.
fn cacheVal(cache: []const u8, scales: ?[]const f32, fp8: bool, row: usize, i: usize) f64 {
    if (!fp8) return bf(u16At(cache, row * 512 + i));
    return sref.bf16rn(fp8c(cache[row * 512 + i]) * @as(f64, scales.?[row]));
}

fn u16At(bytes: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
}

/// qa[r,h,n] = sum_k q[r,h,k] * wk[h,k,n] over bf16 operands.
pub fn absorbRef(a: std.mem.Allocator, q: []const u16, wk: []const u16, rows: usize, heads: usize, d: usize, lw: usize) ![]f64 {
    const out = try a.alloc(f64, rows * heads * lw);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var n: usize = 0;
            while (n < lw) : (n += 1) {
                var sum: f64 = 0;
                var k: usize = 0;
                while (k < d) : (k += 1) sum += bf(q[(r * heads + h) * d + k]) * bf(wk[(h * d + k) * lw + n]);
                out[(r * heads + h) * lw + n] = sum;
            }
        }
    }
    return out;
}

/// out[r,h,n] = sum_k ol[r,h,k] * wv[h,n,k] over bf16 operands.
pub fn expandRef(a: std.mem.Allocator, ol: []const u16, wv: []const u16, rows: usize, heads: usize, lw: usize, v: usize) ![]f64 {
    const out = try a.alloc(f64, rows * heads * v);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var n: usize = 0;
            while (n < v) : (n += 1) {
                var sum: f64 = 0;
                var k: usize = 0;
                while (k < lw) : (k += 1) sum += bf(ol[(r * heads + h) * lw + k]) * bf(wv[(h * v + n) * lw + k]);
                out[(r * heads + h) * v + n] = sum;
            }
        }
    }
    return out;
}

/// Dense latent attention: row r attends keys 0..pos+r (the causal window) over the fp8/bf16 cache.
pub fn attnRef(
    a: std.mem.Allocator,
    qa: []const u16,
    cache: []const u8,
    scales: ?[]const f32,
    pos: i64,
    rows: usize,
    heads: usize,
    lw: usize,
    scale: f64,
    fp8: bool,
    cap_rows: usize,
) ![]f64 {
    const nrows = @min(rows, cap_rows);
    const out = try a.alloc(f64, nrows * heads * lw);
    var r: usize = 0;
    while (r < nrows) : (r += 1) {
        const keys: usize = @intCast(pos + @as(i64, @intCast(r)) + 1);
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var m: f64 = -std.math.inf(f64);
            var l: f64 = 0;
            const o = try a.alloc(f64, lw);
            defer a.free(o);
            @memset(o, 0);
            const qrow = qa[(r * heads + h) * lw ..][0..lw];
            var j: usize = 0;
            while (j < keys) : (j += 1) {
                var dot: f64 = 0;
                for (0..lw) |i| dot += bf(qrow[i]) * cacheVal(cache, scales, fp8, j, i);
                const s = dot * scale;
                const nm = @max(m, s);
                const al = if (m == -std.math.inf(f64)) 0 else @exp(m - nm);
                const p = @exp(s - nm);
                for (0..lw) |i| o[i] = o[i] * al + p * cacheVal(cache, scales, fp8, j, i);
                l = l * al + p;
                m = nm;
            }
            for (0..lw) |i| out[(r * heads + h) * lw + i] = o[i] / l;
        }
    }
    return out;
}
