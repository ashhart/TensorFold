//! The joint schema head on the host (Cloudflare `JointSchemaHead.forward`): FP64, or rounding where BF16 torch does.
const std = @import("std");
const bf16_round = @import("bf16_round");
const act_ref = @import("../act_ref.zig");
const tensors = @import("tensors.zig");
const attn = @import("reference_attn.zig");
const Allocator = std.mem.Allocator;
const Tensor = tensors.Tensor;

/// fp64: every step exact in float64 on the BF16 weights; bf16: each step's output rounded to BF16 as the head's.
pub const Mode = enum { fp64, bf16 };

/// One record's readout: spans are [start, end) rows; options grouped by question, in encoding order.
pub const Record = struct {
    ids: []const u32,
    question_spans: []const [2]u32,
    question_types: []const u8, // noul 0, choice 1, score 2
    option_spans: []const [2]u32,
    option_question: []const u32, // each option's question, non-decreasing
};

/// The output-embedding table (`lm_head.weight`): `rows` rows of hidden-size BF16 values, little-endian.
pub const Table = struct { bytes: []const u8, rows: usize };

pub const Ctx = struct {
    a: Allocator,
    mode: Mode,

    /// A module output: rounded once to BF16 in bf16 mode, exact in fp64 mode.
    pub fn r(c: Ctx, x: f64) f64 {
        return if (c.mode == .bf16) bf16_round.bf16(x) else x;
    }
};

pub const normalize_eps = 1e-12; // F.normalize
pub const cosine_eps = 1e-8; // F.cosine_similarity, each norm clamped on its own

/// A BF16 tensor's element i.
pub fn at(t: Tensor, i: usize) f64 {
    return bf16_round.value(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
}

/// `x Wᵀ + b` over `n` rows with the weight rows first..first+out (and their bias entries), one rounding per output.
pub fn linearRows(c: Ctx, x: []const f64, n: usize, w: Tensor, b: ?Tensor, first: usize, out: usize) ![]f64 {
    const in = w.dim(1);
    std.debug.assert(x.len == n * in);
    const wide = try c.a.alloc(f64, out * in);
    defer c.a.free(wide);
    for (wide, 0..) |*v, i| v.* = at(w, first * in + i);
    const y = try c.a.alloc(f64, n * out);
    for (0..n) |row| for (0..out) |o| {
        var s: f64 = if (b) |bias| at(bias, first + o) else 0;
        for (x[row * in ..][0..in], wide[o * in ..][0..in]) |p, q| s += p * q;
        y[row * out + o] = c.r(s);
    };
    return y;
}

pub fn linear(c: Ctx, x: []const f64, n: usize, l: tensors.Linear) ![]f64 {
    return linearRows(c, x, n, l.weight, l.bias, 0, l.weight.dim(0));
}

fn project(c: Ctx, x: []const f64, n: usize, w: Tensor) ![]f64 {
    return linearRows(c, x, n, w, null, 0, w.dim(0));
}

/// Affine LayerNorm (eps 1e-5) over each row, biased variance; one rounding per output.
pub fn layerNorm(c: Ctx, x: []const f64, n: usize, norm: tensors.Norm) ![]f64 {
    const d = norm.weight.dim(0);
    const y = try c.a.alloc(f64, n * d);
    for (0..n) |row| {
        const v = x[row * d ..][0..d];
        var mean: f64 = 0;
        for (v) |e| mean += e;
        mean /= @floatFromInt(d);
        var vari: f64 = 0;
        for (v) |e| vari += (e - mean) * (e - mean);
        const inv = 1 / @sqrt(vari / @as(f64, @floatFromInt(d)) + 1e-5);
        for (y[row * d ..][0..d], v, 0..) |*o, e, i| o.* = c.r((e - mean) * inv * at(norm.weight, i) + at(norm.bias, i));
    }
    return y;
}

/// Exact (erf) GELU in place.
pub fn gelu(c: Ctx, x: []f64) void {
    for (x) |*v| v.* = c.r(0.5 * v.* * (1 + act_ref.erf(v.* / std.math.sqrt2)));
}

/// `x += y`, each sum rounded (a residual add).
pub fn addInto(c: Ctx, x: []f64, y: []const f64) void {
    for (x, y) |*p, q| p.* = c.r(p.* + q);
}

/// Softmax in float64 (callers round its outputs where the head does).
pub fn softmax(x: []f64) void {
    var m = -std.math.inf(f64);
    for (x) |v| m = @max(m, v);
    var s: f64 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - m);
        s += v.*;
    }
    for (x) |*v| v.* /= s;
}

fn dot(x: []const f64, y: []const f64) f64 {
    var s: f64 = 0;
    for (x, y) |p, q| s += p * q;
    return s;
}

/// The mean of rows span[0]..span[1] of `x` (rows of `d`), one rounding.
fn meanRows(c: Ctx, x: []const f64, d: usize, span: [2]u32, out: []f64) void {
    @memset(out, 0);
    for (span[0]..span[1]) |row| for (out, x[row * d ..][0..d]) |*o, v| {
        o.* += v;
    };
    const n: f64 = @floatFromInt(span[1] - span[0]);
    for (out) |*o| o.* = c.r(o.* / n);
}

/// `F.normalize(x, dim=-1)`: x / max(‖x‖, 1e-12), the norm and each quotient rounded.
pub fn normalize(c: Ctx, x: []const f64, out: []f64) void {
    const n = c.r(@max(c.r(@sqrt(dot(x, x))), normalize_eps));
    for (out, x) |*o, v| o.* = c.r(v / n);
}

/// `F.cosine_similarity(f, o, dim=-1)`: sum((f / max(‖f‖, eps)) ⊙ (o / max(‖o‖, eps))), roundings per element.
pub fn cosine(c: Ctx, f: []const f64, o: []const f64) f64 {
    const nf = c.r(@max(c.r(@sqrt(dot(f, f))), cosine_eps));
    const no = c.r(@max(c.r(@sqrt(dot(o, o))), cosine_eps));
    var s: f64 = 0;
    for (f, o) |p, q| s += c.r(c.r(p / nf) * c.r(q / no));
    return c.r(s);
}

/// `exp(p.clamp(max=ln 100))` on a scalar parameter, each step rounded.
pub fn cappedExp(c: Ctx, p: f64) f64 {
    return c.r(@exp(c.r(@min(p, @log(100.0)))));
}

/// One logit per option, in the record's option order; `hidden` is the record's final-norm rows (rows × hidden).
pub fn logits(c: Ctx, w: *const tensors.Weights, hidden: []const f64, rec: Record, table: Table) ![]f64 {
    const H: usize = w.cfg.hidden_size;
    const W: usize = w.cfg.width;
    const L = hidden.len / H;
    const Q = rec.question_spans.len;
    const T = rec.option_spans.len;
    std.debug.assert(hidden.len == L * H and L > 0 and rec.option_question.len == T and rec.question_types.len == Q);
    // step 1: the normed rows, the memory, the record's last row
    const h = try layerNorm(c, hidden, L, w.hidden_norm);
    const memory = try project(c, h, L, w.memory);
    const g = h[(L - 1) * H ..][0..H];
    // step 2: span means of the normed rows, and of the raw output-embedding rows of each option's tokens
    const qv = try c.a.alloc(f64, Q * H);
    for (rec.question_spans, 0..) |s, k| meanRows(c, h, H, s, qv[k * H ..][0..H]);
    const cv = try c.a.alloc(f64, T * H);
    const ev = try c.a.alloc(f64, T * H);
    const qrep = try c.a.alloc(f64, T * H);
    for (rec.option_spans, rec.option_question, 0..) |s, k, j| {
        meanRows(c, h, H, s, cv[j * H ..][0..H]);
        const e = ev[j * H ..][0..H];
        @memset(e, 0);
        for (rec.ids[s[0]..s[1]]) |id| {
            if (id >= table.rows) return error.BadToken;
            for (e, 0..) |*x, i| x.* += bf16_round.value(std.mem.readInt(u16, table.bytes[(id * H + i) * 2 ..][0..2], .little));
        }
        for (e) |*x| x.* = c.r(x.* / @as(f64, @floatFromInt(s[1] - s[0])));
        @memcpy(qrep[j * H ..][0..H], qv[k * H ..][0..H]);
    }
    // step 3: each option's query, summed in the head's order
    const routed = try project(c, cv, T, w.option_context);
    addInto(c, routed, try project(c, ev, T, w.option_lexical));
    addInto(c, routed, try project(c, qrep, T, w.option_question));
    // step 4: the options route over the memory
    for (w.evidence) |*e| try attn.evidence(c, e, routed, T, memory, L, w.cfg.heads);
    // step 5: each question's field from its projection, its options' weighted summary, the last row and its type
    const fields = try project(c, qv, Q, w.question);
    const summary = try c.a.alloc(f64, Q * W);
    var first: usize = 0;
    for (0..Q) |k| {
        var end = first;
        while (end < T and rec.option_question[end] == k) end += 1;
        const wts = try c.a.alloc(f64, end - first);
        for (wts, first..) |*x, j| x.* = c.r(c.r(dot(routed[j * W ..][0..W], fields[k * W ..][0..W])) / @sqrt(@as(f64, @floatFromInt(W))));
        softmax(wts);
        const s = summary[k * W ..][0..W];
        for (s, 0..) |*x, i| {
            var acc: f64 = 0;
            for (wts, first..) |p, j| acc += c.r(c.r(p) * routed[j * W + i]);
            x.* = c.r(acc);
        }
        first = end;
    }
    addInto(c, fields, try layerNorm(c, summary, Q, w.option_summary_norm));
    const gp = try project(c, g, 1, w.global);
    for (0..Q) |k| {
        addInto(c, fields[k * W ..][0..W], gp);
        const t: usize = rec.question_types[k];
        for (fields[k * W ..][0..W], 0..) |*x, i| x.* = c.r(x.* + at(w.type_embedding, t * W + i));
    }
    // step 6: the decoder layers over the fields, then the field norm
    for (w.layers) |*d| try attn.decoder(c, d, fields, Q, memory, L, w.cfg.heads);
    const f = try layerNorm(c, fields, Q, w.field_norm);
    // step 7: a lexical prior against the question and record, plus the gated joint score
    const opts = try layerNorm(c, routed, T, w.option_norm);
    const prior_scale = cappedExp(c, at(w.prior_logit_scale, 0));
    const joint_scale = cappedExp(c, at(w.joint_logit_scale, 0));
    const gate = c.r(1 / (1 + @exp(-at(w.residual_gate, 0))));
    const anchor_in = try c.a.alloc(f64, H);
    const anchor = try c.a.alloc(f64, H);
    const lex = try c.a.alloc(f64, H);
    const feat = try c.a.alloc(f64, 4 * W);
    const out = try c.a.alloc(f64, T);
    for (out, rec.option_question, 0..) |*logit, k, j| {
        for (anchor_in, qv[k * H ..][0..H], g) |*x, q, gl| x.* = c.r(q + gl);
        normalize(c, anchor_in, anchor);
        normalize(c, ev[j * H ..][0..H], lex);
        const prior = c.r(prior_scale * c.r(dot(lex, anchor)));
        const fk = f[k * W ..][0..W];
        const oj = opts[j * W ..][0..W];
        const cos = cosine(c, fk, oj);
        for (0..W) |i| {
            feat[i] = fk[i];
            feat[W + i] = oj[i];
            feat[2 * W + i] = c.r(fk[i] * oj[i]);
            feat[3 * W + i] = @abs(c.r(fk[i] - oj[i]));
        }
        const hid = try linear(c, feat, 1, w.scorer_up);
        gelu(c, hid);
        const res = (try linear(c, hid, 1, w.scorer_down))[0];
        const joint = c.r(c.r(joint_scale * cos) + res);
        logit.* = c.r(prior + c.r(gate * joint));
    }
    return out;
}
