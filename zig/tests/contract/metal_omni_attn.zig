//! LFM3-Omni's Metal attention: prompt rows in tiles (om_attention_tiles) and one row over the cache (om_attend).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const k = @import("metal_omni_common.zig");

const c = k.c;
const R = k.R;
const Probe = k.Probe;

/// A stack's attention: heads, kv heads, head width, window (0: none), causal, and LFM3's gate.
const Cfg = struct { h: usize, hkv: usize, hd: usize, window: usize, causal: bool, gate: bool };
const cfgs = [_]Cfg{
    .{ .h = 16, .hkv = 4, .hd = 128, .window = 0, .causal = true, .gate = true },
    .{ .h = 16, .hkv = 4, .hd = 128, .window = 256, .causal = true, .gate = true },
    .{ .h = 8, .hkv = 4, .hd = 96, .window = 32, .causal = true, .gate = false },
    .{ .h = 32, .hkv = 8, .hd = 32, .window = 0, .causal = false, .gate = true },
    .{ .h = 8, .hkv = 2, .hd = 64, .window = 30, .causal = true, .gate = true },
};

fn tiled(p: *const Probe) bool {
    return p.variant & 0x10 == 0;
}

fn cfgOf(p: *const Probe) Cfg {
    return cfgs[p.variant & 0xf];
}

/// The cache's rows: the rows before (shape k) and the new rows (shape rows).
fn kvLen(p: *const Probe) usize {
    return p.shape.k + p.shape.rows;
}

/// A query row's width: its heads, and om_attend's gate logits after them (the projection's row).
fn ldq(p: *const Probe) usize {
    const f = cfgOf(p);
    return f.h * f.hd + if (tiled(p)) 0 else f.h;
}

/// Shape rows: new query rows (om_attend runs one launch a row, as decode steps), n the head width, k rows before.
fn attnPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    const f = cfgOf(p);
    if (s.rows == 0 or s.n != f.hd or (!f.causal and s.k != 0) or kvLen(p) > 4096) return false;
    const nkv = kvLen(p) * f.hkv * f.hd;
    const q = try p.alloc(u16, 0, s.rows * ldq(p));
    const kc = try p.alloc(u16, 1, nkv);
    const vc = try p.alloc(u16, 2, nkv);
    const g = try p.alloc(u16, 3, s.rows * f.h);
    _ = try p.alloc(u16, 4, s.rows * f.h * f.hd);
    if (p.fill == .stress) {
        // equal scores (q = 0); over the last row's keys, integers whose partials grow large then cancel to a few
        @memset(q, c.toBf16(30));
        for (0..s.rows) |r| @memset(q[r * ldq(p) ..][0 .. f.h * f.hd], 0);
        k.randomBf16(rng, kc, -1, 1);
        const n = kvLen(p);
        const lo = if (f.causal and f.window > 0) n -| f.window else 0;
        const half = (n - lo) / 2;
        for (0..n) |jj| {
            const j = jj -| lo;
            // the second half's order turned by one, so 32 lanes' strided partials do not cancel early
            const at = if (j < half) j else (j - half + 1) % @max(half, 1);
            const pos: f32 = if (at % 2 == 0) 127 else 4064;
            const res: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
            const v: f32 = if (jj < lo) 0 else if (j < half) pos else if (j < 2 * half) res - pos else 0;
            @memset(vc[jj * f.hkv * f.hd ..][0 .. f.hkv * f.hd], c.toBf16(v));
        }
        @memset(g, c.toBf16(30));
    } else {
        k.randomBf16(rng, q, -1, 1);
        k.randomBf16(rng, kc, -1, 1);
        k.randomBf16(rng, vc, -2, 2);
        k.randomBf16(rng, g, -3, 3);
    }
    if (p.fill == .nonfinite) {
        // the last key's values: NaN, +Inf, -Inf in three columns and both infinities (two keys) in a fourth
        const last = (kvLen(p) - 1) * f.hkv * f.hd;
        const prev = if (kvLen(p) > 1) last - f.hkv * f.hd else last;
        for ([_]usize{ last, last + 1, last + 2, last + 3, prev + 3 }, [_]u16{ k.nan16, k.inf16, k.ninf16, k.inf16, k.ninf16 }) |at, v| vc[at] = v;
        // a NaN query in the first row's second kv group
        q[f.h / f.hkv * f.hd + 3] = k.nan16;
    }
    for (0..4) |i| try p.in(i);
    try p.out(4, false);
    return true;
}

const TileArgs = extern struct { ldq: i32, ldkv: i32, ldo: i32, T: i32, H: i32, Hkv: i32, hd: i32, pos0: i32, kv_len: i32, window: i32, causal: i32, scale: f32, has_gate: i32, ldg: i32 };
const AttendArgs = extern struct { ldkv: i32, H: i32, Hkv: i32, hd: i32, pos: i32, kv_len: i32, window: i32, causal: i32, has_gate: i32, gate_off: i32, scale: f32 };

fn scale(f: Cfg) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt(f.hd)));
}

fn attnLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const f = cfgOf(p);
    if (tiled(p)) {
        p.bind(e, &.{ 0, 1, 2, 4, if (f.gate) 3 else null }, TileArgs{ .ldq = k.i32_(f.h * f.hd), .ldkv = k.i32_(f.hkv * f.hd), .ldo = k.i32_(f.h * f.hd), .T = k.i32_(s.rows), .H = k.i32_(f.h), .Hkv = k.i32_(f.hkv), .hd = k.i32_(f.hd), .pos0 = k.i32_(s.k), .kv_len = k.i32_(kvLen(p)), .window = k.i32_(f.window), .causal = @intFromBool(f.causal), .scale = scale(f), .has_gate = @intFromBool(f.gate), .ldg = k.i32_(f.h) });
        return Probe.go(e, .{ k.blocks(s.rows, 16), f.h, 1 }, 128);
    }
    e.setPipeline(p.pipe);
    for (0..s.rows) |r| {
        const qr = Probe.at(p.refs[0], r * ldq(p) * 2);
        const o = Probe.at(p.refs[4], r * f.h * f.hd * 2);
        e.setBuffer(qr.buf, qr.off, 0);
        e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
        e.setBuffer(p.refs[2].buf, p.refs[2].off, 2);
        e.setBuffer(o.buf, o.off, 3);
        const a = AttendArgs{ .ldkv = k.i32_(f.hkv * f.hd), .H = k.i32_(f.h), .Hkv = k.i32_(f.hkv), .hd = k.i32_(f.hd), .pos = k.i32_(s.k + r), .kv_len = k.i32_(s.k + r + 1), .window = k.i32_(f.window), .causal = @intFromBool(f.causal), .has_gate = @intFromBool(f.gate), .gate_off = k.i32_(f.h * f.hd), .scale = scale(f) };
        e.setBytes(std.mem.asBytes(&a), 4);
        Probe.go(e, .{ f.h, 1, 1 }, 256);
    }
}

fn attnCount(p: *Probe) usize {
    return p.shape.rows * cfgOf(p).h * cfgOf(p).hd;
}

/// Softmax(q k^T scale) v in float64 a (row, head), bounded by the fp32 steps' spread; then bf16 and the gate.
fn attnEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const s = p.shape;
    const f = cfgOf(p);
    const sc: f64 = scale(f);
    const q = p.get(u16, 0);
    const kc = p.get(u16, 1);
    const vc = p.get(u16, 2);
    const ldkv = f.hkv * f.hd;
    const n_kv = kvLen(p);
    const buf = try p.a.alloc(f64, 4 * n_kv);
    defer p.a.free(buf);
    const sv = buf[0..n_kv];
    const ones = buf[n_kv .. 2 * n_kv];
    const col = buf[2 * n_kv .. 3 * n_kv];
    const unit = c.unit(.f32);
    for (0..s.rows) |r| for (0..f.h) |h| {
        const kh = h / (f.h / f.hkv);
        const pos = s.k + r;
        const hi = if (f.causal) pos + 1 else n_kv;
        const lo = if (f.causal and f.window > 0) (pos + 1) -| f.window else 0;
        const qr = q[r * ldq(p) + h * f.hd ..][0..f.hd];
        var m: f64 = -std.math.inf(f64);
        var nan = false;
        var e_max: f64 = 0;
        for (lo..hi) |j| {
            var dot: f64 = 0;
            var mag: f64 = 0;
            for (qr, kc[j * ldkv + kh * f.hd ..][0..f.hd]) |a, b| {
                const t = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                dot += t;
                mag += @abs(t);
            }
            sv[j - lo] = dot * sc;
            e_max = @max(e_max, c.gamma(@floatFromInt(f.hd + 2), unit) * mag * sc + 2 * unit * @abs(sv[j - lo]));
            if (std.math.isNan(sv[j - lo])) nan = true;
            m = @max(m, sv[j - lo]);
        }
        if (nan) m = std.math.nan(f64);
        const n = hi - lo;
        const tiles: f64 = if (tiled(p)) @floatFromInt((n + 31) / 32 + 1) else 1;
        const e_p = 2 * e_max + (tiles + 2) * 8 * 0x1p-23;
        const e_sum = c.gamma(@floatFromInt(n + 16), unit);
        var l: f64 = 0;
        for (sv[0..n]) |*x| {
            x.* = @exp(x.* - m);
            l += x.*;
        }
        for (0..f.hd) |e| {
            var o: f64 = 0;
            var mag: f64 = 0;
            for (sv[0..n], lo..) |w, j| {
                const v: f64 = c.fromBf16(vc[j * ldkv + kh * f.hd + e]);
                o += w * v;
                mag += w * @abs(v);
                col[j - lo] = v;
            }
            var val = o / l;
            var d = mag / l * (2 * e_p + 2 * e_sum) + @abs(val) * 4 * unit;
            switch (mode) {
                .ref => {},
                .exact => d = @abs(val) * 4 * unit,
                .sim => |x| {
                    @memset(ones[0..n], 1);
                    val = contract.rows.sumIn(x.acc, x.order, col[0..n]) / contract.rows.sumIn(x.acc, x.order, ones[0..n]);
                    d = 0;
                },
            }
            var res = (R{ .v = val, .lo = val - d, .hi = val + d }).bf();
            if (f.gate) {
                const gl = if (tiled(p)) p.get(u16, 3)[r * f.h + h] else q[r * ldq(p) + f.h * f.hd + h];
                res = res.mul(k.sigmR(R.bf16(gl)).bf()).bf();
            }
            out[(r * f.h + h) * f.hd + e] = res;
        }
    };
}

const attn_def: k.Def = .{ .prep = attnPrep, .launch = attnLaunch, .count = attnCount, .eval = attnEval };

fn attnDepth(s: c.Shape) usize {
    return s.n + s.k + s.rows;
}

fn spec(name: []const u8, served: []const c.Shape, tile: c.Tile) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = attnDepth, .ops = 8, .out = .bf16, .tiles = .{ tile, .{}, .{} }, .served = served };
}

const mask = "if (k < nk && r0 + r < a.T && (!a.causal || kp <= pos) && (!(a.causal && a.window > 0) || kp > pos - a.window)) {";
const gate = "if (a.has_gate) ob = b(f(ob) * rb(sigm(f(Q[a.gate_off + h]))));";
const t16: c.Tile = .{ .size = 16 };

pub const entries = [_]k.Entry{
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attention_tiles full, gated (backbone)", &.{ .{ .rows = 64, .n = 128, .k = 37 }, .{ .rows = 5, .n = 128, .k = 0 } }, t16), .file = .omni, .pipe = "om_attention_tiles", .variant = 0, .control = .{ mask, "if (k < nk && r0 + r < a.T && (!a.causal || kp <= pos + 1) && (!(a.causal && a.window > 0) || kp > pos - a.window)) {" }, .what = "sees one key past the row" },
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attention_tiles sliding 256 (backbone)", &.{.{ .rows = 40, .n = 128, .k = 300 }}, t16), .file = .omni, .pipe = "om_attention_tiles", .variant = 1, .control = .{ mask, "if (k < nk && r0 + r < a.T && (!a.causal || kp <= pos) && (!(a.causal && a.window > 0) || kp >= pos - a.window)) {" }, .what = "keeps one key past the window" },
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attention_tiles window 32 (encoder)", &.{ .{ .rows = 94, .n = 96, .k = 0 }, .{ .rows = 20, .n = 96, .k = 0 } }, t16), .file = .omni, .pipe = "om_attention_tiles", .variant = 2, .control = .{ "if (c < dpl) acc[x][c] += p * f(vr[lane + 32 * c]);", "if (c < dpl) acc[x][c] += rb(p) * f(vr[lane + 32 * c]);" }, .what = "adds bf16(p)" },
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attention_tiles bidirectional, gated (depth decoder)", &.{.{ .rows = 12, .n = 32, .k = 0 }}, .{}), .file = .omni, .pipe = "om_attention_tiles", .variant = 3, .control = .{ "const float g = a.has_gate ? rb(sigm(f(G[(long)r * a.ldg + h]))) : 1.0f;", "const float g = a.has_gate ? (sigm(f(G[(long)r * a.ldg + h]))) : 1.0f;" }, .what = "drops bf16(sigmoid(gate))" },
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attention_tiles sliding 30, gated (codec)", &.{ .{ .rows = 48, .n = 64, .k = 50 }, .{ .rows = 6, .n = 64, .k = 50 } }, t16), .file = .omni, .pipe = "om_attention_tiles", .variant = 4, .control = .{ "const bf o = b(acc[x][c] * inv);", "const bf o = b(acc[x][c] * rb(inv));" }, .what = "adds bf16(1 / l)" },
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attend full, gated (backbone)", &.{ .{ .rows = 1, .n = 128, .k = 37 }, .{ .rows = 3, .n = 128, .k = 299 } }, .{}), .file = .rows, .pipe = "om_attend", .variant = 0x10, .control = .{ gate, "if (a.has_gate) ob = b(f(ob) * (sigm(f(Q[a.gate_off + h]))));" }, .what = "drops bf16(sigmoid(gate))" },
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attend sliding 256 (backbone)", &.{.{ .rows = 2, .n = 128, .k = 300 }}, .{}), .file = .rows, .pipe = "om_attend", .variant = 0x11, .control = .{ "const int lo = (a.causal && a.window > 0) ? max(0, a.pos + 1 - a.window) : 0;", "const int lo = (a.causal && a.window > 0) ? max(0, a.pos - a.window) : 0;" }, .what = "keeps one key past the window" },
    .{ .def = &attn_def, .spec = spec("metal lfm3 omni om_attend sliding 30, gated (codec)", &.{.{ .rows = 1, .n = 64, .k = 50 }}, .{}), .file = .rows, .pipe = "om_attend", .variant = 0x14, .control = .{ "bf ob = b(o * (1.0f / l));", "bf ob = b(o * rb(1.0f / l));" }, .what = "adds bf16(1 / l)" },
};
