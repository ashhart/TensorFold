//! Clef's joint schema head (Cloudflare `JointSchemaHead`, Apache-2.0) in fp32 on the host: one logit per option.
const std = @import("std");
const enc = @import("clef_encode.zig");
const Allocator = std.mem.Allocator;

pub const Config = struct { hidden: usize = 5120, width: usize = 1024, routing: usize = 2, layers: usize = 4, heads: usize = 16, ff: usize = 4096 };

const Lin = struct { w: []const f32, b: ?[]const f32, out: usize, in: usize };
const Norm = struct { w: []const f32, b: []const f32 };
const Mha = struct { q: Lin, k: Lin, v: Lin, o: Lin };
const Routing = struct { qn: Norm, mn: Norm, fn_: Norm, attn: Mha, f1: Lin, f2: Lin };
const Decoder = struct { n1: Norm, n2: Norm, n3: Norm, self: Mha, cross: Mha, l1: Lin, l2: Lin };

/// The output-embedding rows a question's options average (`lm_head_rows.safetensors`, bf16).
pub const Lexical = struct {
    table: []const u8,
    rows: usize,
    width: usize,

    pub fn row(l: Lexical, id: u32, out: []f32) !void {
        if (id >= l.rows) return error.BadToken;
        const at = @as(usize, id) * l.width * 2;
        for (out, 0..) |*o, i| o.* = bf16(l.table[at + 2 * i ..][0..2]);
    }
};

fn bf16(b: *const [2]u8) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, b, .little)) << 16);
}

pub const Head = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    hidden_norm: Norm,
    memory: Lin,
    question: Lin,
    option_question: Lin,
    global: Lin,
    option_context: Lin,
    option_lexical: Lin,
    types: []const f32,
    routing: []Routing,
    summary_norm: Norm,
    layers: []Decoder,
    field_norm: Norm,
    option_norm: Norm,
    scorer1: Lin,
    scorer2: Lin,
    prior_scale: f32,
    joint_scale: f32,
    gate: f32,

    /// The head's bf16 tensors widened to f32; `file.get(name)` gives a tensor with `dtype`, `bytes`, `numel()`.
    pub fn load(gpa: Allocator, file: anytype, cfg: Config) !Head {
        var h: Head = undefined;
        h.arena = .init(gpa);
        errdefer h.arena.deinit();
        h.cfg = cfg;
        const L = Loader(@TypeOf(file)){ .a = h.arena.allocator(), .file = file };
        const W = cfg.width;
        const H = cfg.hidden;
        h.hidden_norm = try L.norm("hidden_norm", H);
        h.memory = try L.lin("memory_projection", W, H, false);
        h.question = try L.lin("question_projection", W, H, false);
        h.option_question = try L.lin("option_question_projection", W, H, false);
        h.global = try L.lin("global_projection", W, H, false);
        h.option_context = try L.lin("option_context_projection", W, H, false);
        h.option_lexical = try L.lin("option_lexical_projection", W, H, false);
        h.types = try L.vec("type_embedding.weight", 3 * W);
        h.routing = try h.arena.allocator().alloc(Routing, cfg.routing);
        for (h.routing, 0..) |*r, i| {
            const p = try std.fmt.allocPrint(L.a, "evidence_layers.{d}.", .{i});
            r.* = .{ .qn = try L.norm(try cat(L.a, p, "query_norm"), W), .mn = try L.norm(try cat(L.a, p, "memory_norm"), W), .fn_ = try L.norm(try cat(L.a, p, "feedforward_norm"), W), .attn = try L.mha(try cat(L.a, p, "attention"), W), .f1 = try L.lin(try cat(L.a, p, "feedforward.0"), cfg.ff, W, true), .f2 = try L.lin(try cat(L.a, p, "feedforward.3"), W, cfg.ff, true) };
        }
        h.summary_norm = try L.norm("option_summary_norm", W);
        h.layers = try h.arena.allocator().alloc(Decoder, cfg.layers);
        for (h.layers, 0..) |*d, i| {
            const p = try std.fmt.allocPrint(L.a, "layers.{d}.", .{i});
            d.* = .{ .n1 = try L.norm(try cat(L.a, p, "norm1"), W), .n2 = try L.norm(try cat(L.a, p, "norm2"), W), .n3 = try L.norm(try cat(L.a, p, "norm3"), W), .self = try L.mha(try cat(L.a, p, "self_attn"), W), .cross = try L.mha(try cat(L.a, p, "multihead_attn"), W), .l1 = try L.lin(try cat(L.a, p, "linear1"), cfg.ff, W, true), .l2 = try L.lin(try cat(L.a, p, "linear2"), W, cfg.ff, true) };
        }
        h.field_norm = try L.norm("field_norm", W);
        h.option_norm = try L.norm("option_norm", W);
        h.scorer1 = try L.lin("residual_scorer.0", W, 4 * W, true);
        h.scorer2 = try L.lin("residual_scorer.3", 1, W, true);
        const cap = @log(@as(f32, 100));
        h.prior_scale = @exp(@min((try L.vec("prior_logit_scale", 1))[0], cap));
        h.joint_scale = @exp(@min((try L.vec("joint_logit_scale", 1))[0], cap));
        h.gate = 1 / (1 + @exp(-(try L.vec("residual_gate", 1))[0]));
        return h;
    }

    pub fn deinit(h: *Head) void {
        h.arena.deinit();
    }

    /// One logit per option for each question: `hidden` is the record's final-norm rows (rows x hidden, f32).
    pub fn forward(h: *const Head, a: Allocator, hidden: []const f32, rec: enc.Record, lex: Lexical) ![][]f32 {
        const H = h.cfg.hidden;
        const W = h.cfg.width;
        const n = rec.ids.len;
        if (hidden.len != n * H) return error.BadHidden;
        const nh = try a.alloc(f32, n * H);
        layerNormRows(hidden, nh, h.hidden_norm, H);
        const memory = try a.alloc(f32, n * W);
        try linear(nh, n, h.memory, memory);
        const global = nh[(n - 1) * H ..][0..H];
        const nq = rec.questions.len;
        const qvec = try a.alloc(f32, nq * H);
        var total: usize = 0;
        for (rec.questions, 0..) |q, i| {
            meanRows(nh, H, q.span, qvec[i * H ..][0..H]);
            total += q.options.len;
        }
        // each option: its context rows' mean, its tokens' output-embedding mean, and its question
        const ctx = try a.alloc(f32, total * H);
        const lexv = try a.alloc(f32, total * H);
        const qrep = try a.alloc(f32, total * H);
        const tmp = try a.alloc(f32, H);
        var k: usize = 0;
        for (rec.questions, 0..) |q, qi| for (q.options) |span| {
            meanRows(nh, H, span, ctx[k * H ..][0..H]);
            const lv = lexv[k * H ..][0..H];
            @memset(lv, 0);
            for (rec.ids[span[0]..span[1]]) |id| {
                try lex.row(id, tmp);
                for (lv, tmp) |*x, t| x.* += t;
            }
            const cnt: f32 = @floatFromInt(span[1] - span[0]);
            for (lv) |*x| x.* /= cnt;
            @memcpy(qrep[k * H ..][0..H], qvec[qi * H ..][0..H]);
            k += 1;
        };
        const routed = try a.alloc(f32, total * W);
        const t2 = try a.alloc(f32, total * W);
        try linear(ctx, total, h.option_context, routed);
        try linear(lexv, total, h.option_lexical, t2);
        add(routed, t2);
        try linear(qrep, total, h.option_question, t2);
        add(routed, t2);
        for (h.routing) |*r| try routingLayer(a, r, routed, total, memory, n, h.cfg);
        // fields: the question projection, its options summarised by it, the record's last row and the type
        const fields = try a.alloc(f32, nq * W);
        try linear(qvec, nq, h.question, fields);
        const summaries = try a.alloc(f32, nq * W);
        const gproj = try a.alloc(f32, W);
        try linear(global, 1, h.global, gproj);
        var off: usize = 0;
        for (rec.questions, 0..) |q, qi| {
            const m = q.options.len;
            const f = fields[qi * W ..][0..W];
            const wts = try a.alloc(f32, m);
            for (wts, 0..) |*w, o| w.* = dot(routed[(off + o) * W ..][0..W], f) / @sqrt(@as(f32, @floatFromInt(W)));
            softmax(wts);
            const s = summaries[qi * W ..][0..W];
            @memset(s, 0);
            for (wts, 0..) |w, o| for (s, routed[(off + o) * W ..][0..W]) |*x, y| {
                x.* += w * y;
            };
            off += m;
        }
        layerNormRows(summaries, summaries, h.summary_norm, W);
        for (rec.questions, 0..) |q, qi| {
            const f = fields[qi * W ..][0..W];
            add(f, summaries[qi * W ..][0..W]);
            add(f, gproj);
            add(f, h.types[@backingInt(q.kind) * W ..][0..W]);
        }
        for (h.layers) |*d| try decoderLayer(a, d, fields, nq, memory, n, h.cfg);
        layerNormRows(fields, fields, h.field_norm, W);
        // logits: a lexical prior against the question and record, plus the gated joint score
        const opts = try a.alloc(f32, total * W);
        layerNormRows(routed, opts, h.option_norm, W);
        const out = try a.alloc([]f32, nq);
        const anchor = try a.alloc(f32, H);
        const feat = try a.alloc(f32, 4 * W);
        const hid = try a.alloc(f32, W);
        off = 0;
        for (rec.questions, out, 0..) |q, *lg, qi| {
            for (anchor, qvec[qi * H ..][0..H], global) |*x, y, g| x.* = y + g;
            normalize(anchor);
            const f = fields[qi * W ..][0..W];
            lg.* = try a.alloc(f32, q.options.len);
            for (lg.*, 0..) |*l, o| {
                const lx = lexv[(off + o) * H ..][0..H];
                const prior = h.prior_scale * dot(lx, anchor) / @max(norm2(lx), 1e-12);
                const ov = opts[(off + o) * W ..][0..W];
                const cosine = dot(f, ov) / @max(norm2(f) * norm2(ov), 1e-8);
                for (0..W) |i| {
                    feat[i] = f[i];
                    feat[W + i] = ov[i];
                    feat[2 * W + i] = f[i] * ov[i];
                    feat[3 * W + i] = @abs(f[i] - ov[i]);
                }
                try linear(feat, 1, h.scorer1, hid);
                gelu(hid);
                var res: [1]f32 = undefined;
                try linear(hid, 1, h.scorer2, &res);
                l.* = prior + h.gate * (h.joint_scale * cosine + res[0]);
            }
            off += q.options.len;
        }
        return out;
    }
};

fn Loader(comptime F: type) type {
    return struct {
        a: Allocator,
        file: F,
        const Self = @This();

        fn vec(l: Self, name: []const u8, len: usize) ![]f32 {
            const t = l.file.get(name) orelse return error.MissingTensor;
            if (t.dtype != .bf16 or t.numel() != len) return error.BadTensor;
            const out = try l.a.alloc(f32, len);
            for (out, 0..) |*o, i| o.* = bf16(t.bytes[2 * i ..][0..2]);
            return out;
        }
        fn norm(l: Self, p: []const u8, w: usize) !Norm {
            return .{ .w = try l.vec(try cat(l.a, p, ".weight"), w), .b = try l.vec(try cat(l.a, p, ".bias"), w) };
        }
        fn lin(l: Self, p: []const u8, out: usize, in: usize, bias: bool) !Lin {
            return .{ .w = try l.vec(try cat(l.a, p, ".weight"), out * in), .b = if (bias) try l.vec(try cat(l.a, p, ".bias"), out) else null, .out = out, .in = in };
        }
        /// torch's MultiheadAttention: in_proj packs q, k and v by rows.
        fn mha(l: Self, p: []const u8, w: usize) !Mha {
            const iw = try l.vec(try cat(l.a, p, ".in_proj_weight"), 3 * w * w);
            const ib = try l.vec(try cat(l.a, p, ".in_proj_bias"), 3 * w);
            var m: Mha = undefined;
            for ([_]*Lin{ &m.q, &m.k, &m.v }, 0..) |x, i| x.* = .{ .w = iw[i * w * w ..][0 .. w * w], .b = ib[i * w ..][0..w], .out = w, .in = w };
            m.o = try l.lin(try cat(l.a, p, ".out_proj"), w, w, true);
            return m;
        }
    };
}

fn cat(a: Allocator, x: []const u8, y: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ x, y });
}

fn dot(x: []const f32, y: []const f32) f32 {
    const V = @Vector(16, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= x.len) : (i += 16) acc += @as(V, x[i..][0..16].*) * @as(V, y[i..][0..16].*);
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) s += x[i] * y[i];
    return s;
}

fn norm2(x: []const f32) f32 {
    return @sqrt(dot(x, x));
}

fn normalize(x: []f32) void {
    const s = @max(norm2(x), 1e-12);
    for (x) |*v| v.* /= s;
}

fn add(x: []f32, y: []const f32) void {
    for (x, y) |*a, b| a.* += b;
}

fn softmax(x: []f32) void {
    var m = -std.math.inf(f32);
    for (x) |v| m = @max(m, v);
    var s: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - m);
        s += v.*;
    }
    for (x) |*v| v.* /= s;
}

fn gelu(x: []f32) void {
    for (x) |*v| v.* = @floatCast(0.5 * @as(f64, v.*) * (1 + erf(@as(f64, v.*) / std.math.sqrt2)));
}

fn meanRows(x: []const f32, w: usize, span: [2]u32, out: []f32) void {
    @memset(out, 0);
    for (span[0]..span[1]) |r| add(out, x[r * w ..][0..w]);
    const c: f32 = @floatFromInt(span[1] - span[0]);
    for (out) |*v| v.* /= c;
}

/// LayerNorm over each row of `w` values, eps 1e-5, statistics in f64; `out` may alias `x`.
fn layerNormRows(x: []const f32, out: []f32, nm: Norm, w: usize) void {
    var r: usize = 0;
    while (r * w < x.len) : (r += 1) {
        const row = x[r * w ..][0..w];
        var mean: f64 = 0;
        for (row) |v| mean += v;
        mean /= @floatFromInt(w);
        var varc: f64 = 0;
        for (row) |v| varc += (v - mean) * (v - mean);
        const inv = 1 / @sqrt(varc / @as(f64, @floatFromInt(w)) + 1e-5);
        for (out[r * w ..][0..w], row, nm.w, nm.b) |*o, v, g, b| o.* = @floatCast((v - mean) * inv * g + b);
    }
}

/// `out = x Wᵀ + b` for `n` rows, split over the host's cores by rows.
fn linear(x: []const f32, n: usize, l: Lin, out: []f32) !void {
    const threads = @min(std.Thread.getCpuCount() catch 1, 32, @max(1, n / 8));
    if (threads <= 1) return linearRows(x, l, out, 0, n);
    var pool: [32]std.Thread = undefined;
    const per = (n + threads - 1) / threads;
    var started: usize = 0;
    defer for (pool[0..started]) |t| t.join();
    for (0..threads) |t| {
        const lo = t * per;
        const hi = @min(n, lo + per);
        if (lo >= hi) break;
        pool[started] = try std.Thread.spawn(.{}, linearRows, .{ x, l, out, lo, hi });
        started += 1;
    }
}

fn linearRows(x: []const f32, l: Lin, out: []f32, lo: usize, hi: usize) void {
    for (lo..hi) |r| {
        const xr = x[r * l.in ..][0..l.in];
        for (0..l.out) |o| out[r * l.out + o] = dot(xr, l.w[o * l.in ..][0..l.in]) + if (l.b) |b| b[o] else 0;
    }
}

/// torch MultiheadAttention without masks: `q` (nq rows) attends to `kv` (nk rows); the result goes to `out`.
fn attention(a: Allocator, m: *const Mha, q: []const f32, nq: usize, kv: []const f32, nk: usize, heads: usize, out: []f32) !void {
    const W = m.q.out;
    const dh = W / heads;
    const qp = try a.alloc(f32, nq * W);
    const kp = try a.alloc(f32, nk * W);
    const vp = try a.alloc(f32, nk * W);
    try linear(q, nq, m.q, qp);
    try linear(kv, nk, m.k, kp);
    try linear(kv, nk, m.v, vp);
    const ctx = try a.alloc(f32, nq * W);
    const s = try a.alloc(f32, nk);
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(dh)));
    for (0..nq) |i| for (0..heads) |hh| {
        const qi = qp[i * W + hh * dh ..][0..dh];
        for (s, 0..) |*x, j| x.* = dot(qi, kp[j * W + hh * dh ..][0..dh]) * scale;
        softmax(s);
        const o = ctx[i * W + hh * dh ..][0..dh];
        @memset(o, 0);
        for (s, 0..) |p, j| for (o, vp[j * W + hh * dh ..][0..dh]) |*x, y| {
            x.* += p * y;
        };
    };
    try linear(ctx, nq, m.o, out);
}

fn feedForward(a: Allocator, f1: Lin, f2: Lin, x: []const f32, n: usize, out: []f32) !void {
    const mid = try a.alloc(f32, n * f1.out);
    try linear(x, n, f1, mid);
    gelu(mid);
    try linear(mid, n, f2, out);
}

/// EvidenceRoutingLayer: the options attend to the normed memory, then a feed-forward step, both residual.
fn routingLayer(a: Allocator, r: *const Routing, x: []f32, n: usize, memory: []const f32, nm: usize, cfg: Config) !void {
    const W = cfg.width;
    const qn = try a.alloc(f32, n * W);
    layerNormRows(x, qn, r.qn, W);
    const mn = try a.alloc(f32, nm * W);
    layerNormRows(memory, mn, r.mn, W);
    const t = try a.alloc(f32, n * W);
    try attention(a, &r.attn, qn, n, mn, nm, cfg.heads, t);
    add(x, t);
    layerNormRows(x, qn, r.fn_, W);
    try feedForward(a, r.f1, r.f2, qn, n, t);
    add(x, t);
}

/// TransformerDecoderLayer with norm_first: self-attention, memory attention (memory not normed), feed-forward.
fn decoderLayer(a: Allocator, d: *const Decoder, x: []f32, n: usize, memory: []const f32, nm: usize, cfg: Config) !void {
    const W = cfg.width;
    const y = try a.alloc(f32, n * W);
    const t = try a.alloc(f32, n * W);
    layerNormRows(x, y, d.n1, W);
    try attention(a, &d.self, y, n, y, n, cfg.heads, t);
    add(x, t);
    layerNormRows(x, y, d.n2, W);
    try attention(a, &d.cross, y, n, memory, nm, cfg.heads, t);
    add(x, t);
    layerNormRows(x, y, d.n3, W);
    try feedForward(a, d.l1, d.l2, y, n, t);
    add(x, t);
}

/// erf in f64 to about 1e-15 (core/act_ref.zig's): the Maclaurin series below 3, erfc's continued fraction above.
fn erf(z: f64) f64 {
    if (std.math.isNan(z)) return z;
    if (z < 0) return -erf(-z);
    if (z > 6) return 1;
    if (z < 3) {
        var term = z;
        var sum: f64 = z;
        var n: f64 = 1;
        while (n < 200) : (n += 1) {
            term *= -z * z / n;
            const t = term / (2 * n + 1);
            sum += t;
            if (@abs(t) < 1e-18 * @abs(sum)) break;
        }
        return sum * 2 / @sqrt(std.math.pi);
    }
    var f = z;
    var k: f64 = 60;
    while (k >= 1) : (k -= 1) f = z + (k / 2) / f;
    return 1 - @exp(-z * z) / (@sqrt(std.math.pi) * f);
}
