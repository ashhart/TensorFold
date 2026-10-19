//! Kolibri-1's first layer and head in float64 from the checkpoint's own tensors, rounded where the engine rounds.
const std = @import("std");
const core = @import("core");
const kolibri1 = @import("kolibri1");

const Tensor = core.checkpoint.Tensor;
const names = kolibri1.names;

fn bf16(t: Tensor, i: usize) f64 {
    return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16));
}

fn f32At(t: Tensor, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)));
}

/// x rounded to bf16 (nearest even), as the engine's bf16 stores round.
fn roundBf16(x: f64) f64 {
    const f: f32 = @floatCast(x);
    const b: u32 = @bitCast(f);
    const r = b +% 0x7fff +% ((b >> 16) & 1);
    return @as(f32, @bitCast(r & 0xffff0000));
}

fn roundF32(x: f64) f64 {
    return @as(f32, @floatCast(x));
}

/// The value of an e4m3 code (no NaN in a checkpoint's weights).
fn e4m3(code: u8) f64 {
    const e: i32 = (code >> 3) & 15;
    const m: f64 = @floatFromInt(code & 7);
    const v = if (e == 0) m * 0x1p-9 else (8 + m) * std.math.ldexp(@as(f64, 1), e - 10);
    return if (code & 0x80 != 0) -v else v;
}

/// out[j] = sum over k of x[k] * dequantized w[j, k] (block-FP8: code times its 128 x 128 block's scale).
fn fp8Rows(p: names.Fp8, x: []const f64, out: []f64) void {
    const k = p.w.dim(1);
    const kb = (k + 127) / 128;
    for (out, 0..) |*o, j| {
        var acc: f64 = 0;
        for (0..k) |c| acc += x[c] * e4m3(p.w.bytes[j * k + c]) * f32At(p.s, (j / 128) * kb + c / 128);
        o.* = acc;
    }
}

/// rms(x) * w over `x`, in float64.
fn rms(x: []const f64, w: Tensor, eps: f64, out: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const r = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (out, x, 0..) |*o, v, i| o.* = v * r * bf16(w, i);
}

/// The last row's logits of `tokens` at positions 0.. through layer 0, the final norm and the head (decode arithmetic).
pub fn lastLogits(a: std.mem.Allocator, m: *const names.Map, c: kolibri1.Config, tokens: []const u32) ![]f64 {
    const n = tokens.len;
    const d = c.hidden;
    const h = c.heads;
    const hk = c.kv_heads;
    const dh = c.head_dim;
    const L = m.layers[0];
    const res = try a.alloc(f64, n * d);
    const x = try a.alloc(f64, n * d);
    const q = try a.alloc(f64, n * h * dh);
    const kk = try a.alloc(f64, n * hk * dh);
    const vv = try a.alloc(f64, n * hk * dh);
    const qkv = try a.alloc(f64, c.qkvDim());
    for (tokens, 0..) |t, r| {
        for (0..d) |i| res[r * d + i] = bf16(m.embed, t * d + i);
        rms(res[r * d ..][0..d], L.input_norm, c.eps, x[r * d ..][0..d]);
        for (x[r * d ..][0..d]) |*v| v.* = roundBf16(v.*);
        fp8Rows(L.q, x[r * d ..][0..d], qkv[0 .. h * dh]);
        fp8Rows(L.k, x[r * d ..][0..d], qkv[h * dh ..][0 .. hk * dh]);
        fp8Rows(L.v, x[r * d ..][0..d], qkv[(h + hk) * dh ..][0 .. hk * dh]);
        for (qkv) |*v| v.* = roundBf16(v.*);
        for (0..h + hk) |hd| {
            const src = qkv[hd * dh ..][0..dh];
            const dst = if (hd < h) q[(r * h + hd) * dh ..][0..dh] else kk[(r * hk + hd - h) * dh ..][0..dh];
            var y: [256]f64 = undefined;
            rms(src, if (hd < h) L.q_norm else L.k_norm, c.eps, y[0..dh]);
            for (0..dh) |i| { // neox halves, sliding layers only (layer 0 is sliding)
                const j = i % (dh / 2);
                const ang = @as(f64, @floatFromInt(r)) / std.math.pow(f64, c.rope_theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dh)));
                const rot = if (i < dh / 2) -y[i + dh / 2] else y[i - dh / 2];
                dst[i] = if (c.full[0]) y[i] else roundBf16(y[i] * roundF32(@cos(ang)) + rot * roundF32(@sin(ang)));
            }
            if (c.full[0]) for (dst) |*v| {
                v.* = roundBf16(v.*);
            };
        }
        @memcpy(vv[r * hk * dh ..][0 .. hk * dh], qkv[(h + hk) * dh ..][0 .. hk * dh]);
    }
    // the last row's attention over its window, then o, the residual and the MoE
    const r = n - 1;
    const att = try a.alloc(f64, h * dh);
    const lo = if (r + 1 > c.window) r + 1 - c.window else 0;
    const keys = try a.alloc(usize, r + 1 - lo);
    for (keys, 0..) |*kx, i| kx.* = lo + i;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dh)));
    for (0..h) |hd| {
        const g = hd / (h / hk);
        var sc = try a.alloc(f64, keys.len);
        var most: f64 = -std.math.inf(f64);
        for (keys, 0..) |p, i| {
            var dot: f64 = 0;
            for (0..dh) |e| dot += q[(r * h + hd) * dh + e] * kk[(p * hk + g) * dh + e];
            sc[i] = dot * scale;
            most = @max(most, sc[i]);
        }
        var sum: f64 = 0;
        for (sc) |*v| {
            v.* = @exp(v.* - most);
            sum += v.*;
        }
        for (0..dh) |e| {
            var acc: f64 = 0;
            for (keys, 0..) |p, i| acc += sc[i] * vv[(p * hk + g) * dh + e];
            att[hd * dh + e] = roundBf16(acc / sum);
        }
    }
    const o = try a.alloc(f64, d);
    fp8Rows(L.o, att, o);
    for (o) |*v| v.* = roundBf16(v.*);
    const res1 = res[r * d ..][0..d];
    const t1 = try a.alloc(f64, d);
    rms(o, L.post_attn_norm, c.eps, t1);
    for (res1, t1) |*v, u| v.* = roundF32(v.* + u);
    const x2 = try a.alloc(f64, d);
    rms(res1, L.pre_moe_norm, c.eps, x2);
    for (x2) |*v| v.* = roundBf16(v.*);
    const mo = try moe(a, L, c, x2);
    rms(mo, L.post_moe_norm, c.eps, t1);
    for (res1, t1) |*v, u| v.* = roundF32(v.* + u);
    rms(res1, m.norm, c.eps, x2);
    for (x2) |*v| v.* = roundBf16(v.*);
    const logits = try a.alloc(f64, c.vocab);
    for (logits, 0..) |*lg, j| {
        var acc: f64 = 0;
        for (0..d) |i| acc += x2[i] * bf16(m.head, j * d + i);
        lg.* = acc;
    }
    return logits;
}

/// Top-k experts by sigmoid logit + bias (lower id on ties), weights sigmoid(logit), the shared expert at weight 1.
fn moe(a: std.mem.Allocator, L: names.Layer, c: kolibri1.Config, x: []const f64) ![]f64 {
    const d = c.hidden;
    const ne = c.experts;
    const logit = try a.alloc(f64, ne);
    const key = try a.alloc(f64, ne);
    for (logit, key, 0..) |*lg, *ky, e| {
        var acc: f64 = 0;
        for (0..d) |i| acc += x[i] * bf16(L.router, e * d + i);
        lg.* = roundF32(acc);
        ky.* = lg.* + bf16(L.bias, e);
    }
    const out = try a.alloc(f64, d);
    @memset(out, 0);
    const gate = try a.alloc(f64, c.moe_width);
    const up = try a.alloc(f64, c.moe_width);
    const y = try a.alloc(f64, d);
    for (0..c.top_k + 1) |slot| {
        var e: usize = ne; // the shared expert
        var w: f64 = 1;
        if (slot < c.top_k) {
            e = std.mem.indexOfMax(f64, key);
            w = roundF32(1.0 / (1.0 + @exp(-logit[e])));
            key[e] = -std.math.inf(f64);
        }
        const ex = L.experts[e];
        fp8Rows(ex.gate, x, gate);
        fp8Rows(ex.up, x, up);
        for (gate, up) |*g, u| g.* = roundBf16(g.* / (1.0 + @exp(-g.*)) * u);
        fp8Rows(ex.down, gate, y);
        for (out, y) |*o, v| o.* += w * roundF32(v);
    }
    for (out) |*o| o.* = roundBf16(o.*);
    return out;
}
