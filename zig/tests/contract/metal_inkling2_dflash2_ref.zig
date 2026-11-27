//! Reference computations and output checks for the Mini DFlash 2 Metal contract suite.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const host = @import("tensorfold").inkling2_dflash2_host;

pub const H = host.hidden;
pub const Q = host.query_width;
pub const KV = host.kv_width;
pub const QH = host.q_heads;
pub const KVH = host.kv_heads;
pub const D = host.head_dim;
pub const G = QH / KVH;
pub const INTER = host.intermediate;
pub const RANK = host.rank;
pub const GROUPS = host.conv_groups;
pub const BLOCK = host.block_size;
pub const PART = 132;
pub const EPS: f32 = 1e-6;
pub const THETA: f32 = 1e6;

pub fn bf(v: f32) f32 {
    return c.fromBf16(c.toBf16(v));
}

pub fn rmsInv(values: []const f32, start: usize, width: usize) f32 {
    var sum: f32 = 0;
    for (0..width) |d| {
        const x = values[start + d];
        sum += x * x;
    }
    return 1.0 / @sqrt(sum / @as(f32, @floatFromInt(width)) + EPS);
}

pub fn headNormRope(src: []const f32, w: []const f32, pos: usize, dst: []f64) void {
    const inv = rmsInv(src, 0, D);
    var nrm: [D]f32 = undefined;
    for (0..D) |d| nrm[d] = bf(w[d] * bf(src[d] * inv));
    for (0..D) |d| {
        const i = d % 64;
        const angle = @as(f32, @floatFromInt(pos)) * (1.0 / std.math.pow(f32, THETA, @as(f32, @floatFromInt(2 * i)) / 128.0));
        const cs = bf(@cos(angle));
        const sn = bf(@sin(angle));
        const rot = if (d < 64) -nrm[d + 64] else nrm[d - 64];
        dst[d] = bf(bf(nrm[d] * cs) + bf(rot * sn));
    }
}

pub fn computeReference(kind: host.Kind, shape: c.Shape, attend_args: host.AttendArgs, f: [6][]f32, u: [3][]u32, want: []f64, mag: []f64) void {
    const rows = shape.rows;
    switch (kind) {
        .embed => for (0..rows) |r| {
            const src = f[0][u[0][r] * H ..][0..H];
            if (shape.k == 0) {
                for (0..H) |d| want[r * H + d] = src[d];
            } else {
                const inv = rmsInv(src, 0, H);
                for (0..H) |d| want[r * H + d] = bf(src[d] * inv * f[1][d]);
            }
        },
        .norm => for (0..rows) |r| {
            var rrow: [H]f32 = undefined;
            for (0..H) |d| {
                var v = f[0][r * H + d];
                if (shape.k != 0) v = bf(v + f[1][r * H + d]);
                rrow[d] = v;
                want[r * H + d] = v;
            }
            const inv = rmsInv(&rrow, 0, H);
            for (0..H) |d| want[rows * H + r * H + d] = bf(f[2][d] * bf(rrow[d] * inv));
        },
        .norm16 => for (0..rows) |r| {
            const inv = rmsInv(f[0], r * H, H);
            for (0..H) |d| want[r * H + d] = bf(f[1][d] * bf(f[0][r * H + d] * inv));
        },
        .conv => for (0..rows) |r| for (0..H) |ch| {
            const grp = ch / (H / GROUPS);
            const side = shape.k;
            const d_base = r * 4 * GROUPS + side * 2 * GROUPS;
            const c0 = bf(f[2][(side * 2) * H + ch] + f[1][d_base + grp]);
            const c1 = bf(f[2][(side * 2 + 1) * H + ch] + f[1][d_base + GROUPS + grp]);
            var y = bf(c0 * f[0][r * H + ch]);
            if (r % BLOCK >= 1) y = bf(y + bf(c1 * f[0][(r - 1) * H + ch]));
            want[r * H + ch] = y;
        },
        .ctx_kv => for (0..rows) |r| {
            const pos = shape.k + r;
            const slot = pos % rows;
            for (0..KVH) |h| {
                headNormRope(f[0][(r * KVH + h) * D ..][0..D], f[2], pos, want[(slot * KVH + h) * D ..][0..D]);
                for (0..D) |d| want[rows * KV + (slot * KVH + h) * D + d] = f[1][(r * KVH + h) * D + d];
            }
        },
        .block_qkv => for (0..rows) |r| {
            const pos = shape.k + (r % BLOCK);
            for (0..QH) |h| headNormRope(f[0][(r * QH + h) * D ..][0..D], f[3], pos, want[(r * QH + h) * D ..][0..D]);
            for (0..KVH) |h| {
                headNormRope(f[1][(r * KVH + h) * D ..][0..D], f[4], pos, want[rows * Q + (r * KVH + h) * D ..][0..D]);
                for (0..D) |d| want[rows * (Q + KV) + (r * KVH + h) * D + d] = f[2][(r * KVH + h) * D + d];
            }
        },
        .attend => {
            const a = attend_args;
            for (0..rows) |r| {
                const reach = a.anchor + @as(u32, @intCast(r)) + 1;
                const row_lo = @max(a.from, if (reach > a.window) reach - a.window else 0);
                for (0..KVH) |kh| for (0..a.chunks) |ch| {
                    const first = ch * 128;
                    const last = @min(a.keys, first + 128);
                    for (0..G) |gg| {
                        const q_off = (r * QH + kh * G + gg) * D;
                        var max_s: f32 = -std.math.inf(f32);
                        var has_nan = false;
                        var scores: [128]f32 = undefined;
                        var live_mask: [128]bool = undefined;
                        for (first..last) |key| {
                            const is_ctx = key < a.retained;
                            const live = !is_ctx or (a.lo + @as(u32, @intCast(key))) >= row_lo;
                            live_mask[key - first] = live;
                            if (!live) {
                                scores[key - first] = -std.math.inf(f32);
                                continue;
                            }
                            const krow = if (is_ctx) (a.lo + @as(u32, @intCast(key))) % a.ring else @as(u32, @intCast(key - a.retained));
                            const ksrc = if (is_ctx) f[3] else f[1];
                            const k_off = (krow * KVH + kh) * D;
                            var dot: f32 = 0;
                            for (0..D) |d| dot += f[0][q_off + d] * ksrc[k_off + d];
                            const sc = dot * a.scale;
                            scores[key - first] = sc;
                            if (std.math.isNan(sc)) has_nan = true else max_s = @max(max_s, sc);
                        }
                        if (has_nan) max_s = std.math.nan(f32);
                        var tot: f32 = 0;
                        var acc: [D]f32 = @splat(0);
                        for (first..last) |key| {
                            if (!live_mask[key - first]) continue;
                            const p = @exp(scores[key - first] - max_s);
                            tot += p;
                            const is_ctx = key < a.retained;
                            const krow = if (is_ctx) (a.lo + @as(u32, @intCast(key))) % a.ring else @as(u32, @intCast(key - a.retained));
                            const vsrc = if (is_ctx) f[4] else f[2];
                            const v_off = (krow * KVH + kh) * D;
                            for (0..D) |d| acc[d] += p * vsrc[v_off + d];
                        }
                        const o_base = (((r * KVH + kh) * a.chunks + ch) * G + gg) * PART;
                        for (0..D) |d| want[o_base + d] = acc[d];
                        want[o_base + D] = max_s;
                        want[o_base + D + 1] = tot;
                    }
                };
            }
        },
        .attend_merge => {
            const chunks = shape.n;
            for (0..rows) |r| for (0..QH) |h| {
                const kh = h / G;
                const gg = h % G;
                var max_s: f32 = -std.math.inf(f32);
                for (0..chunks) |ch| {
                    const m = f[0][(((r * KVH + kh) * chunks + ch) * G + gg) * PART + D];
                    max_s = if (std.math.isNan(max_s) or std.math.isNan(m)) std.math.nan(f32) else @max(max_s, m);
                }
                var tot: f32 = 0;
                var acc: [D]f32 = @splat(0);
                for (0..chunks) |ch| {
                    const base = (((r * KVH + kh) * chunks + ch) * G + gg) * PART;
                    if (f[0][base + D] == -std.math.inf(f32)) continue;
                    const w = @exp(f[0][base + D] - max_s);
                    tot += f[0][base + D + 1] * w;
                    for (0..D) |d| acc[d] += f[0][base + d] * w;
                }
                for (0..D) |d| want[(r * QH + h) * D + d] = bf(if (tot == 0) 0 else acc[d] / tot);
            };
        },
        .silu_mul => for (want, 0..) |*dst, i| {
            const gv = f[0][i];
            const s = 1.0 / (1.0 + @exp(-gv));
            dst.* = bf(bf(gv * s) * f[1][i]);
        },
        .topk16, .walk => @memset(want, 0),
    }
    for (mag, want) |*m, w| m.* = 1 + @abs(w);
}

pub fn checkTopk16(a: std.mem.Allocator, shape: c.Shape, logits: []const f32, bytes: []const u8) !?[]const u8 {
    const rows = shape.rows;
    const vocab = shape.n;
    const stride = vocab + 16;
    const k: usize = host.top_k;
    const id_off = rows * k * 4;
    const Pair = struct { val: f32, id: u32 };
    const pairs = try a.alloc(Pair, vocab);
    defer a.free(pairs);
    for (0..rows) |r| {
        var bad = false;
        for (0..vocab) |tok| {
            const v = logits[r * stride + tok];
            if (!std.math.isFinite(v)) {
                bad = true;
                break;
            }
            pairs[tok] = .{ .val = v, .id = @intCast(tok) };
        }
        if (bad) {
            for (0..k) |j| {
                const got_v = c.read(.f32, bytes, r * k + j);
                const got_id = std.mem.readInt(u32, bytes[id_off + (r * k + j) * 4 ..][0..4], .little);
                if (got_v != -std.math.inf(f32) or got_id != 0xffffffff) return "topk16 did not reject non-finite score row with (-inf, 0xffffffff)";
            }
            continue;
        }
        std.mem.sort(Pair, pairs, {}, struct {
            fn lt(_: void, lhs: Pair, rhs: Pair) bool {
                if (lhs.val != rhs.val) return lhs.val > rhs.val;
                return lhs.id < rhs.id;
            }
        }.lt);
        for (0..k) |j| {
            const got_v = c.read(.f32, bytes, r * k + j);
            const got_id = std.mem.readInt(u32, bytes[id_off + (r * k + j) * 4 ..][0..4], .little);
            if (got_id != pairs[j].id or got_v != pairs[j].val) return "topk16 output differs from exact descending top-16 with lowest-ID tie-break";
        }
    }
    return null;
}

pub fn checkWalk(shape: c.Shape, f: [6][]f32, ui: []const u32, bytes: []const u8) ?[]const u8 {
    const slots = shape.rows;
    const vocab = shape.n;
    const top: usize = host.top_k;
    const softcap: f32 = if (shape.k != 0) 30.0 else 0.0;
    var prev: u32 = 2;
    if (std.mem.readInt(u32, bytes[0..4], .little) != prev) return "walk modified seed token seq[0]";
    for (0..slots) |e| {
        var want_id: u32 = 0xffffffff;
        if (prev < vocab) {
            var best: f32 = -std.math.inf(f32);
            for (0..top) |ci| {
                const cand = ui[e * top + ci];
                if (cand >= vocab) continue;
                var dot: f32 = 0;
                for (0..RANK) |d| {
                    const ah = bf(f[2][prev * RANK + d] * f[1][e * RANK + d]);
                    dot += ah * f[3][cand * RANK + d];
                }
                var s = f[0][e * top + ci] + dot;
                if (softcap > 0) s = softcap * std.math.tanh(s / softcap);
                if (std.math.isFinite(s) and s > best) {
                    best = s;
                    want_id = cand;
                }
            }
        }
        const got_id = std.mem.readInt(u32, bytes[(1 + e) * 4 ..][0..4], .little);
        if (got_id != want_id) return "walk token differs from guarded greedy selector reference";
        prev = want_id;
    }
    return null;
}

pub fn checkNumeric(kind: host.Kind, shape: c.Shape, want: []const f64, bytes: []const u8) ?[]const u8 {
    if (kind == .norm) {
        const split = shape.rows * H;
        for (0..split) |i| {
            const got = c.read(.f32, bytes, i);
            if (std.math.isNan(want[i])) {
                if (!std.math.isNan(got)) return "norm residual NaN was not propagated";
            } else if (@abs(@as(f64, got) - want[i]) > @max(1e-5, 0.01 * @abs(want[i]))) {
                return "norm updated residual differs from reference";
            }
        }
        const y_off = split * 4;
        for (0..split) |i| {
            const bits = std.mem.readInt(u16, bytes[y_off + 2 * i ..][0..2], .little);
            const w = want[split + i];
            if (std.math.isNan(w)) {
                if (!std.math.isNan(c.fromBf16(bits))) return "norm output BF16 NaN was not propagated";
            } else if (bits != c.toBf16(@floatCast(w))) {
                const got = c.fromBf16(bits);
                if (@abs(@as(f64, got) - w) > @max(1e-4, 0.02 * @abs(w))) return "norm BF16 output differs from reference";
            }
        }
        return null;
    }
    const is_f32 = (kind == .embed or kind == .attend);
    for (want, 0..) |w, i| {
        if (is_f32) {
            const got = c.read(.f32, bytes, i);
            if (std.math.isNan(w)) {
                if (!std.math.isNan(got)) return "FP32 NaN was not propagated";
            } else if (std.math.isInf(w)) {
                if (got != @as(f32, @floatCast(w))) return "FP32 infinity differs from reference";
            } else if (@abs(@as(f64, got) - w) > @max(2e-4, 0.02 * @abs(w))) {
                return "FP32 output differs from reference";
            }
        } else {
            const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
            if (std.math.isNan(w)) {
                if (!std.math.isNan(c.fromBf16(bits))) return "BF16 NaN was not propagated";
            } else if (bits != c.toBf16(@floatCast(w))) {
                if (std.math.isInf(w)) return "BF16 infinity differs from reference";
                const got = c.fromBf16(bits);
                const abs_tol: f64 = if (kind == .block_qkv or kind == .ctx_kv) 0.02 else 1e-4;
                const rel: f64 = if (kind == .attend_merge) 0.03 else 0.02;
                if (@abs(@as(f64, got) - w) > @max(abs_tol, rel * @abs(w))) return "BF16 output differs from reference";
            }
        }
    }
    return null;
}
