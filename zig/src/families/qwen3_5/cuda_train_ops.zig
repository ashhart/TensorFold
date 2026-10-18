//! The 27B's input gradients for Living Weights as calls on one stream.
const std = @import("std");
const cuda = @import("cuda");

/// A 4-bit matrix [n, k] in MLX's layout: words [n, k / 8], bf16 scales and biases [n, k / group].
pub const Q4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, group: usize = 64 };

/// Attention's shape (train.cu's Heads): rows, heads, kv heads, head dim, the q|k|v row width, scale, prefix keys.
pub const Heads = extern struct { rows: c_int, heads: c_int, kv_heads: c_int, dim: c_int, qkv: c_int, scale: f32, prefix: c_int = 0 };

/// Neurons a pocket kernel takes at most (qwen27_train.cu's POCKET_MAX).
pub const max_pocket = 128;
/// Keys the attention backward reads at most (attn_train.cu's ATTN_KEYS).
pub const max_keys = 2048;

const own_names = [_][:0]const u8{ "tf_q27_gemm_q4", "tf_q27_gemm_q4_bf", "tf_q27_gemm_q4w", "tf_q27_gemm_q4w_bf", "tf_q27_zero_rows", "tf_q27_fit_dots", "tf_q27_fit_dots_tree", "tf_q27_fit_step", "tf_q27_swiglu_back", "tf_q27_pocket_rows", "tf_q27_pocket_back", "tf_q27_pocket_grad", "tf_q27_cols", "tf_q27_add_bf16" };
/// attn_train.cu's: the gated attention backward, the split products' sum, a nudge's add (shared with Flash Next).
const attn_names = [_][:0]const u8{ "tf_sum_parts", "tf_axpy", "tf_attn_prep", "tf_attn_q", "tf_attn_gate_back", "tf_attn_prep_back" };
const train_names = [_][:0]const u8{ "tf_train_softmax", "tf_train_rms_back", "tf_train_attn_kv", "tf_train_zero" };

fn index(comptime list: []const [:0]const u8, comptime name: []const u8) usize {
    inline for (list, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no kernel " ++ name);
}

/// The forward's pocket kernel alone (qwen27_train.cu), for a server that serves a taught folder.
pub const PocketKernels = struct {
    own: cuda.Module,
    rows: cuda.Function,

    pub fn load(d: *const cuda.Driver) !PocketKernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var m = try cuda.Module.load(d, cuda.kernels.qwen27_train);
        errdefer m.unload();
        return .{ .own = m, .rows = try m.function("tf_q27_pocket_rows") };
    }

    pub fn deinit(k: *PocketKernels) void {
        k.own.unload();
    }
};

pub const Kernels = struct {
    own: cuda.Module,
    attn: cuda.Module,
    train: cuda.Module,
    fo: [own_names.len]cuda.Function,
    fa: [attn_names.len]cuda.Function,
    ft: [train_names.len]cuda.Function,
    wide: bool = true, // products on 128x128 tiles (tf_q27_gemm_q4w, the same bits); TF_Q27_GEMM=narrow: the 64x64 ones

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const narrow = if (@import("sys").c.getenv("TF_Q27_GEMM")) |v| std.mem.eql(u8, std.mem.span(v), "narrow") else false;
        var k: Kernels = .{ .own = try cuda.Module.load(d, cuda.kernels.qwen27_train), .attn = undefined, .train = undefined, .fo = undefined, .fa = undefined, .ft = undefined, .wide = !narrow };
        errdefer k.own.unload();
        k.attn = try cuda.Module.load(d, cuda.kernels.attn_train);
        errdefer k.attn.unload();
        k.train = try cuda.Module.load(d, cuda.kernels.train);
        errdefer k.train.unload();
        for (own_names, &k.fo) |n, *f| f.* = try k.own.function(n);
        for (attn_names, &k.fa) |n, *f| f.* = try k.attn.function(n);
        for (train_names, &k.ft) |n, *f| f.* = try k.train.function(n);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.own.unload();
        k.attn.unload();
        k.train.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn up(n: usize, by: usize) u32 {
    return @intCast((n + by - 1) / by);
}

/// K splits for a product of m rows by n columns over k: enough blocks to fill the GPU, each split 1024 rows or more.
pub fn splitsFor(m: usize, n: usize, k: usize) usize {
    const tiles = ((m + 63) / 64) * ((n + 63) / 64);
    const want = (192 + tiles - 1) / tiles;
    return @max(1, @min(want, k / 1024, 64));
}

/// The rows a backward still needs (every nudged sequence's from its first nudged row on).
pub const Tail = struct { list: u64, mask: u64, n: usize };

/// The pocket: SwiGLU neurons fp32 [n, d]; `scale` (f32 a row, 0: none) scales neurons from `from` on per row.
pub const Pocket = struct { gate: u64, up: u64, down: u64, n: usize, scale: u64 = 0, from: usize = 0 };

/// A layer's MLP as the backward reads it (MLX copies): gate and up [inter, d], down [d, inter].
pub const Mlp = struct { gate: Q4, up: Q4, down: Q4 };

/// What the MLP kept: y [rows, d] its normed input, gate and up [rows, inter] its projections (bf16).
pub const MlpKept = struct { y: u64, gate: u64, up: u64 };

/// Scratch for the MLP backward: dact, dgate, dup [rows, inter] fp32.
pub const MlpScratch = struct { dact: u64, dgate: u64, dup: u64 };

pub const Attn = struct {
    q: Q4, // q_proj [heads 2 dim, d]: each head's q then gate
    k: Q4, // k_proj [kv dim, d]
    v: Q4, // v_proj [kv dim, d]
    o: Q4, // o_proj [d, heads dim]
    qscale: u64, // q_norm's fp32 weight [dim]
    kscale: u64, // k_norm's fp32 weight [dim]
    inv: u64, // RoPE's fp32 inverse frequencies [half]
    heads: usize,
    kv_heads: usize,
    dim: usize,
    half: usize,
    eps: f32,
};

/// What attention kept: pq, pk, pv its bf16 projections, gated o_proj's bf16 input, pos0 the first row's position.
pub const AttnKept = struct { pq: u64, pk: u64, pv: u64, gated: u64, pos0: usize };

/// The attention backward's scratch for `rows` rows (qkv bf16, the rest fp32, probs and dscores [heads, rows, rows]).
pub const AttnScratch = struct { qkv: u64, dg: u64, dout: u64, dpq: u64, dq: u64, dk: u64, dv: u64, dpk: u64, probs: u64, dscores: u64 };

/// Launches on one stream.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn go(o: Ops, comptime name: []const u8, grid: [3]u32, block: u32, args: *cuda.Args) !void {
        const f = if (comptime std.mem.startsWith(u8, name, "tf_q27_")) o.k.fo[index(&own_names, name)] else if (comptime std.mem.startsWith(u8, name, "tf_train_")) o.k.ft[index(&train_names, name)] else o.k.fa[index(&attn_names, name)];
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.s, args);
    }

    pub fn zero(o: Ops, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(n)));
        try o.go("tf_train_zero", .{ up(n, 256), 1, 1 }, 256, &a);
    }

    /// Whether a product takes the wide tiles: whole 128-column tiles, K in slabs of 32, A's rows 16-byte aligned.
    fn wideFits(a: u64, a_bf16: bool, lda: usize, n: usize, k: usize) bool {
        const per: usize = if (a_bf16) 8 else 4;
        return n % 128 == 0 and k % 32 == 0 and a % 16 == 0 and lda % per == 0;
    }

    /// c [m, w.k] (+)= a [m, w.n] w (a fp32 or bf16, rows lda apart); `parts` holds split partials when needed.
    pub fn product(o: Ops, a_: u64, a_bf16: bool, lda: usize, w: Q4, c: u64, ldc: usize, m: usize, accumulate: bool, parts: u64) !void {
        const n = w.k;
        const k = w.n;
        if (n % w.group != 0 or w.group % 8 != 0) return error.UnsupportedShape;
        const splits = if (parts != 0) splitsFor(m, n, k) else 1;
        const kchunk = (((k + splits - 1) / splits) + 31) / 32 * 32;
        const z = (k + kchunk - 1) / kchunk;
        var a: cuda.Args = .{};
        a.add(a_);
        for ([_]u64{ w.w, w.s, w.b, if (z > 1) parts else c }) |v| a.add(v);
        for ([_]usize{ m, n, k, lda, ldc }) |v| a.add(int(v));
        a.add(@as(c_int, @intFromBool(accumulate)));
        a.add(int(kchunk));
        a.add(int(w.group));
        if (o.k.wide and m > 64 and wideFits(a_, a_bf16, lda, n, k)) {
            a.add(@as(u64, 0));
            if (a_bf16) try o.go("tf_q27_gemm_q4w_bf", .{ up(n, 128), up(m, 128), u(z) }, 256, &a) else try o.go("tf_q27_gemm_q4w", .{ up(n, 128), up(m, 128), u(z) }, 256, &a);
        } else if (a_bf16) try o.go("tf_q27_gemm_q4_bf", .{ up(n, 64), up(m, 64), u(z) }, 128, &a) else try o.go("tf_q27_gemm_q4", .{ up(n, 64), up(m, 64), u(z) }, 128, &a);
        if (z == 1) return;
        var b: cuda.Args = .{};
        b.add(parts);
        for ([_]usize{ z, m, n }) |v| b.add(int(v));
        b.add(c);
        b.add(int(ldc));
        b.add(@as(c_int, @intFromBool(accumulate)));
        try o.go("tf_sum_parts", .{ up(m * n, 256), 1, 1 }, 256, &b);
    }

    /// c (+)= a w on the `m` rows listed at `rows`.
    pub fn productRows(o: Ops, a_: u64, lda: usize, w: Q4, c: u64, ldc: usize, rows: u64, m: usize, accumulate: bool) !void {
        return o.productRowsAs(a_, false, lda, w, c, ldc, rows, m, accumulate);
    }

    /// productRows with A fp32 or bf16.
    pub fn productRowsAs(o: Ops, a_: u64, a_bf16: bool, lda: usize, w: Q4, c: u64, ldc: usize, rows: u64, m: usize, accumulate: bool) !void {
        const n = w.k;
        const k = w.n;
        if (n % w.group != 0 or w.group % 8 != 0 or !wideFits(a_, a_bf16, lda, n, k)) return error.UnsupportedShape;
        if (m == 0) return;
        var a: cuda.Args = .{};
        a.add(a_);
        for ([_]u64{ w.w, w.s, w.b, c }) |v| a.add(v);
        for ([_]usize{ m, n, k, lda, ldc }) |v| a.add(int(v));
        a.add(@as(c_int, @intFromBool(accumulate)));
        a.add(int(k));
        a.add(int(w.group));
        a.add(rows);
        if (a_bf16) try o.go("tf_q27_gemm_q4w_bf", .{ up(n, 128), up(m, 128), 1 }, 256, &a) else try o.go("tf_q27_gemm_q4w", .{ up(n, 128), up(m, 128), 1 }, 256, &a);
    }

    /// product on only the backward's tail rows when it has some.
    pub fn productTail(o: Ops, a_: u64, a_bf16: bool, lda: usize, w: Q4, c: u64, ldc: usize, m: usize, accumulate: bool, parts: u64, tail: ?Tail) !void {
        if (tail) |tl| if ((parts == 0 or splitsFor(m, w.k, w.n) == 1) and o.k.wide and wideFits(a_, a_bf16, lda, w.k, w.n))
            return o.productRowsAs(a_, a_bf16, lda, w, c, ldc, tl.list, tl.n, accumulate);
        return o.product(a_, a_bf16, lda, w, c, ldc, m, accumulate, parts);
    }

    /// A device fit step's dot products and hinges for `n` neurons.
    pub fn fitDots(o: Ops, tree: bool, x: u64, m: usize, d: usize, u_: u64, kinds: u64, invs: u64, below: f32, c: u64, wrong: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(m));
        a.add(int(d));
        for ([_]u64{ u_, kinds, invs }) |v| a.add(v);
        a.add(below);
        a.add(c);
        a.add(wrong);
        const f = if (tree) o.k.fo[index(&own_names, "tf_q27_fit_dots_tree")] else o.k.fo[index(&own_names, "tf_q27_fit_dots")];
        try cuda.launch.launch(f, .{ .grid = .{ .x = up(m, 8), .y = u(n) }, .block = .{ .x = 256 }, .shared = @intCast(d * 4) }, o.s, &a);
    }

    /// A device fit step's gradient and Adam for `n` neurons (c1, c2: its bias corrections).
    pub fn fitStep(o: Ops, x: u64, m: usize, d: usize, kinds: u64, c: u64, wrong: u64, invs: u64, u_: u64, mm: u64, vv: u64, rate: f32, c1: f32, c2: f32, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(m));
        a.add(int(d));
        for ([_]u64{ kinds, c, wrong, invs, u_, mm, vv }) |v| a.add(v);
        a.add(rate);
        a.add(c1);
        a.add(c2);
        try o.go("tf_q27_fit_step", .{ up(d, 256), u(n), 1 }, 256, &a);
    }

    /// x [rows, ld] fp32: the rows not in the tail set to zero.
    pub fn zeroRows(o: Ops, x: u64, ld: usize, rows: usize, tail: Tail) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(ld));
        a.add(tail.mask);
        try o.go("tf_q27_zero_rows", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// Each row's softmax over its bf16 logits: (loss, target probability) into stats, then (p - onehot) w in place.
    pub fn softmax(o: Ops, logits: u64, targets: u64, weights: u64, stats: u64, rows: usize, vocab: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, targets, weights, stats }) |v| a.add(v);
        a.add(int(vocab));
        try o.go("tf_train_softmax", .{ u(rows), 1, 1 }, 1024, &a);
    }

    /// g [rows, dim] += the RMS norm y = h rsqrt(mean h^2 + eps) w undone from dy [rows, dim] fp32 (h, w bf16).
    pub fn rmsBack(o: Ops, h: u64, w: u64, dy: u64, g: u64, rows: usize, dim: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ h, w, dy, g }) |v| a.add(v);
        a.add(int(dim));
        a.add(eps);
        try o.go("tf_train_rms_back", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// y[0 .. n] += x[0 .. n] (fp32).
    pub fn axpy(o: Ops, y: u64, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(x);
        a.add(int(n));
        try o.go("tf_axpy", .{ up(n, 256), 1, 1 }, 256, &a);
    }

    /// y[0 .. n] = bf16(y + x) (y bf16, x fp32): a nudge onto a row of the MLP's output.
    pub fn addBf16(o: Ops, y: u64, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(x);
        a.add(int(n));
        try o.go("tf_q27_add_bf16", .{ up(n, 256), 1, 1 }, 256, &a);
    }

    /// dst [rows, ld] columns col0 .. col0 + w = src [rows, w] (bf16).
    pub fn cols(o: Ops, src: u64, w: usize, dst: u64, ld: usize, col0: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(int(w));
        a.add(dst);
        a.add(int(ld));
        a.add(int(col0));
        a.add(int(rows));
        try o.go("tf_q27_cols", .{ up(rows * w, 256), 1, 1 }, 256, &a);
    }

    /// out [rows, d] bf16 += the pocket's output on the rows of y [rows, d] bf16.
    pub fn pocketRows(o: Ops, pk: Pocket, y: u64, out: u64, rows: usize, d: usize) !void {
        if (pk.n > max_pocket) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        a.add(y);
        a.add(int(d));
        for ([_]u64{ pk.gate, pk.up, pk.down }) |v| a.add(v);
        a.add(int(pk.n));
        a.add(out);
        a.add(pk.scale);
        a.add(int(@min(pk.from, pk.n)));
        try o.go("tf_q27_pocket_rows", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// The pocket undone: dx [rows, d] += its input gradient from g = dL/d the MLP output.
    pub fn pocketBack(o: Ops, pk: Pocket, y: u64, g: u64, dx: u64, rows: usize, d: usize) !void {
        if (pk.n > max_pocket) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        a.add(y);
        a.add(int(d));
        for ([_]u64{ pk.gate, pk.up, pk.down }) |v| a.add(v);
        a.add(int(pk.n));
        for ([_]u64{ g, dx, pk.scale }) |v| a.add(v);
        a.add(int(@min(pk.from, pk.n)));
        try o.go("tf_q27_pocket_back", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// grad [n, d] += w act_j push[row_j] for each neuron j at its row rows[j] (i32, negative: none).
    pub fn pocketGrad(o: Ops, pk: Pocket, y: u64, push: u64, row_of: u64, w: f32, grad: u64, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(int(d));
        a.add(pk.gate);
        a.add(pk.up);
        a.add(int(pk.n));
        a.add(push);
        a.add(row_of);
        a.add(w);
        a.add(grad);
        try o.go("tf_q27_pocket_grad", .{ u(pk.n), 1, 1 }, 256, &a);
    }

    /// The dense MLP and its pocket undone from g = dL/d its output: dy = dL/d its input y, [rows, d].
    pub fn mlpBack(o: Ops, m: Mlp, kept: MlpKept, pocket: ?Pocket, g: u64, dy: u64, t: MlpScratch, rows: usize, parts: u64) !void {
        return o.mlpBackTail(m, kept, pocket, g, dy, t, rows, parts, null);
    }

    /// mlpBack whose products take only the tail's rows (null: every row).
    pub fn mlpBackTail(o: Ops, m: Mlp, kept: MlpKept, pocket: ?Pocket, g: u64, dy: u64, t: MlpScratch, rows: usize, parts: u64, tail: ?Tail) !void {
        const d = m.down.n;
        const inter = m.down.k;
        try o.productTail(g, false, d, m.down, t.dact, inter, rows, false, parts, tail);
        var a: cuda.Args = .{};
        for ([_]u64{ kept.gate, kept.up, t.dact, t.dgate, t.dup }) |v| a.add(v);
        a.add(@as(c_longlong, @intCast(rows * inter)));
        try o.go("tf_q27_swiglu_back", .{ up(rows * inter, 256), 1, 1 }, 256, &a);
        try o.productTail(t.dgate, false, inter, m.gate, dy, d, rows, false, parts, tail);
        try o.productTail(t.dup, false, inter, m.up, dy, d, rows, true, parts, tail);
        if (pocket) |pk| if (pk.n > 0) try o.pocketBack(pk, kept.y, g, dy, rows, d);
    }

    /// The prepared heads (normed, rotated q and k; v) into qkv, as the forward made them.
    pub fn attnPrep(o: Ops, at: Attn, kept: AttnKept, qkv: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ kept.pq, kept.pk, kept.pv, at.qscale, at.kscale, at.inv }) |v| a.add(v);
        a.add(int(kept.pos0));
        a.add(qkv);
        for ([_]usize{ at.heads, at.kv_heads, at.dim, at.half }) |v| a.add(int(v));
        a.add(at.eps);
        try o.go("tf_attn_prep", .{ u(at.heads + 2 * at.kv_heads), u(rows), 1 }, u(at.dim), &a);
    }

    /// Causal attention over one sequence's rows undone: dx (+)= its input gradient from g, its output's.
    pub fn attnBack(o: Ops, at: Attn, kept: AttnKept, g: u64, dx: u64, t: AttnScratch, rows: usize, accumulate: bool) !void {
        return o.attnBackFrom(at, kept, g, dx, t, rows, accumulate, 0);
    }

    /// Attention's middle undone for one sequence's rows, from dg (o_proj's input gradient) to dpq.
    pub fn attnCore(o: Ops, at: Attn, kept: AttnKept, t: AttnScratch, rows: usize) !void {
        const qw = at.heads * at.dim;
        const kw = at.kv_heads * at.dim;
        var a: cuda.Args = .{};
        for ([_]u64{ t.dg, kept.gated, kept.pq, t.dout, t.dpq }) |v| a.add(v);
        a.add(int(at.heads));
        a.add(int(at.dim));
        try o.go("tf_attn_gate_back", .{ up(qw, 256), u(rows), 1 }, 256, &a);
        const h: Heads = .{ .rows = int(rows), .heads = int(at.heads), .kv_heads = int(at.kv_heads), .dim = int(at.dim), .qkv = int(qw + 2 * kw), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(at.dim)))) };
        var b: cuda.Args = .{};
        for ([_]u64{ t.qkv, t.dout, t.probs, t.dscores, t.dq }) |v| b.add(v);
        b.add(h);
        b.add(t.qkv);
        b.add(t.qkv);
        try o.go("tf_attn_q", .{ u(at.heads), u(rows), 1 }, 256, &b);
        var c: cuda.Args = .{};
        for ([_]u64{ t.qkv, t.dout, t.probs, t.dscores, t.dk, t.dv }) |v| c.add(v);
        c.add(h);
        try o.go("tf_train_attn_kv", .{ u(at.kv_heads), u(rows), 1 }, u(at.dim), &c);
        var p: cuda.Args = .{};
        for ([_]u64{ kept.pq, kept.pk, at.qscale, at.kscale, at.inv }) |v| p.add(v);
        p.add(int(kept.pos0));
        for ([_]u64{ t.dq, t.dk, t.dpq, t.dpk }) |v| p.add(v);
        for ([_]usize{ at.heads, at.kv_heads, at.dim, at.half }) |v| p.add(int(v));
        p.add(at.eps);
        try o.go("tf_attn_prep_back", .{ u(at.heads + at.kv_heads), u(rows), 1 }, u(at.dim), &p);
    }

    /// attnBack whose products take only rows `from` on.
    pub fn attnBackFrom(o: Ops, at: Attn, kept: AttnKept, g: u64, dx: u64, t: AttnScratch, rows: usize, accumulate: bool, from: usize) !void {
        if (rows > max_keys or at.dim > 256 or at.dim % 32 != 0 or from >= rows) return error.UnsupportedShape;
        const qw = at.heads * at.dim;
        const kw = at.kv_heads * at.dim;
        const d = at.o.n;
        const m = rows - from;
        try o.attnPrep(at, kept, t.qkv, rows);
        if (from > 0) try o.zero(t.dg, from * qw);
        try o.product(g + from * d * 4, false, d, at.o, t.dg + from * qw * 4, qw, m, false, 0);
        try o.attnCore(at, kept, t, rows);
        try o.product(t.dpq + from * 2 * qw * 4, false, 2 * qw, at.q, dx + from * d * 4, d, m, accumulate, 0);
        try o.product(t.dpk + from * kw * 4, false, kw, at.k, dx + from * d * 4, d, m, true, 0);
        try o.product(t.dv + from * kw * 4, false, kw, at.v, dx + from * d * 4, d, m, true, 0);
    }
};

test "splits stay within the K rows and fill small products" {
    try std.testing.expectEqual(@as(usize, 1), splitsFor(1024, 5120, 248320));
    try std.testing.expect(splitsFor(16, 5120, 248320) > 1);
    try std.testing.expectEqual(@as(usize, 1), splitsFor(16, 5120, 512));
}
