//! The training kernels (train.cu) as calls on one stream: the loss, the change, products and each layer's backward.
const std = @import("std");
const cuda = @import("cuda");
const lesson_route = @import("living").lesson_route;

/// A Mamba layer's shape as the mixer kernels read it (train.cu's Shape).
pub const Shape = extern struct { rows: c_int, heads: c_int, dh: c_int, groups: c_int, n: c_int, inner: c_int, conv: c_int, proj: c_int };

/// Attention's shape as its kernels read it, with the q|k|v row width and the keys before the rows (train.cu's Heads).
pub const Heads = extern struct { rows: c_int, heads: c_int, kv_heads: c_int, dim: c_int, qkv: c_int, scale: f32, prefix: c_int = 0 };

/// The tiles of 16 pairs `pairs` pairs cut into at most, each expert's last one part full.
pub fn tileCap(pairs: usize, experts: usize) usize {
    return pairs / 16 + experts;
}

const names = [_][:0]const u8{
    "tf_train_softmax",    "tf_train_rms_back",      "tf_train_lora_in",    "tf_train_gate",       "tf_train_lora_out",
    "tf_train_project",    "tf_train_lora_db",       "tf_train_lora_dxa",   "tf_train_lora_dx",    "tf_train_sketch",
    "tf_train_adam",       "tf_train_gemm",          "tf_train_dequant",    "tf_train_zero",       "tf_train_widen",
    "tf_train_narrow",     "tf_train_add",           "tf_train_rows",       "tf_train_scale_rows", "tf_train_rest_back",
    "tf_train_relu2_back", "tf_train_sort_pairs",    "tf_train_logits",     "tf_train_pairs_in",   "tf_train_pairs_out",
    "tf_train_route_back", "tf_train_experts_tiles", "tf_train_experts_mm", "tf_train_conv_pre",   "tf_train_dt",
    "tf_train_ssm_fwd",    "tf_train_ssm_back",      "tf_train_ssm_bc",     "tf_train_gate_back",  "tf_train_conv_back",
    "tf_train_dt_back",    "tf_train_attn_q",        "tf_train_attn_kv",    "tf_lesson_latch",     "tf_train_pocket_rows",
};

fn index(comptime name: []const u8) usize {
    inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no training kernel " ++ name);
}

pub const Kernels = struct {
    module: cuda.Module,
    fns: [names.len]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .module = try cuda.Module.load(d, cuda.kernels.train), .fns = undefined };
        errdefer k.module.unload();
        for (names, &k.fns) |n, *f| f.* = try k.module.function(n);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn long(x: usize) c_longlong {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn up(n: usize, by: usize) u32 {
    return @intCast((n + by - 1) / by);
}

/// Launches on one stream, each mirroring train_ops.zig's call of the same kernel.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn go(o: Ops, comptime name: []const u8, grid: [3]u32, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(o.k.fns[index(name)], .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.s, args);
    }

    /// Each row's softmax over its bf16 logits: (loss, target probability) into stats, then (p - onehot) w in place.
    pub fn softmax(o: Ops, logits: u64, targets: u64, weights: u64, stats: u64, rows: usize, vocab: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, targets, weights, stats }) |v| a.add(v);
        a.add(int(vocab));
        try o.go("tf_train_softmax", .{ u(rows), 1, 1 }, 1024, &a);
    }

    /// g [rows, dim] += the residual's gradient through an input RMS norm of h (bf16) with weight w, given dx.
    pub fn rmsBack(o: Ops, h: u64, w: u64, dx: u64, g: u64, rows: usize, dim: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ h, w, dx, g }) |v| a.add(v);
        a.add(int(dim));
        a.add(eps);
        try o.go("tf_train_rms_back", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// xa = x a^T over `rank` ranks and xn = |x|^2, x bf16 rows `x_stride` elements apart.
    pub fn loraIn(o: Ops, x: u64, x_stride: usize, a_: u64, xa: u64, rows: usize, in: usize, rank: usize, xn: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(long(x_stride));
        for ([_]u64{ a_, xa }) |v| a.add(v);
        for ([_]usize{ rows, in, rank }) |v| a.add(int(v));
        a.add(xn);
        try o.go("tf_train_lora_in", .{ up(32 * rank, 256), u(rows), 1 }, 256, &a);
    }

    /// Each row's blocks as `route` gives them: none before its decision row, router scores on it, its latch after it.
    pub fn gate(o: Ops, xa: u64, xn: u64, tau: u64, gates: u64, rows: usize, rank: usize, unit: f32, route: *const lesson_route.Route, latches: u64, score: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ xa, xn, tau, gates }) |v| a.add(v);
        for ([_]usize{ rows, rank, rank / 16, lesson_route.max_blocks }) |v| a.add(int(v));
        a.add(unit);
        a.add(route.*);
        a.add(latches);
        a.add(score);
        try o.go("tf_train_gate", .{ 1, u(rows), 1 }, u(rank / 16), &a);
    }

    /// A sequence's lessons at its decision row (slot): cleared before the pass that holds it, latched (`set`) after.
    pub fn latch(o: Ops, score: u64, need: u64, latches: u64, slot: u32, set: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ score, need, latches }) |v| a.add(v);
        a.add(@as(c_int, @intCast(slot)));
        a.add(@as(c_int, lesson_route.max_blocks));
        a.add(@as(c_int, lesson_route.max_blocks));
        a.add(@as(c_int, @intFromBool(set)));
        try o.go("tf_lesson_latch", .{ 1, 1, 1 }, 1, &a);
    }

    /// y += scale xa b over `rank` ranks, y rows `y_stride` elements apart, bf16 or fp32.
    pub fn loraOut(o: Ops, xa: u64, b: u64, y: u64, y_stride: usize, y_f32: bool, rows: usize, out: usize, rank: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ xa, b, y }) |v| a.add(v);
        a.add(long(y_stride));
        a.add(@as(c_int, @intFromBool(y_f32)));
        for ([_]usize{ rows, out, rank }) |v| a.add(int(v));
        a.add(scale);
        try o.go("tf_train_lora_out", .{ up(out, 256), u(rows), 1 }, 256, &a);
    }

    /// Each row of x, made unit, along the k candidate directions f [k, in], into p [rows, k].
    pub fn project(o: Ops, x: u64, x_stride: usize, f: u64, p: u64, rows: usize, in: usize, k: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(long(x_stride));
        for ([_]u64{ f, p }) |v| a.add(v);
        for ([_]usize{ rows, in, k }) |v| a.add(int(v));
        try o.go("tf_train_project", .{ up(32 * k, 256), u(rows), 1 }, 256, &a);
    }

    /// The open block's gradient: db += scale xa[:, first..]^T g.
    pub fn loraDb(o: Ops, xa: u64, g: u64, db: u64, rows: usize, out: usize, rank: usize, first: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ xa, g, db }) |v| a.add(v);
        for ([_]usize{ rows, out, rank, first }) |v| a.add(int(v));
        a.add(scale);
        try o.go("tf_train_lora_db", .{ up(out, 256), u(rank - first), 1 }, 256, &a);
    }

    /// dxa = scale g b^T on each row's open blocks.
    pub fn loraDxa(o: Ops, g: u64, b: u64, dxa: u64, rows: usize, out: usize, rank: usize, scale: f32, gates: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ g, b, dxa }) |v| a.add(v);
        for ([_]usize{ rows, out, rank }) |v| a.add(int(v));
        a.add(scale);
        a.add(gates);
        try o.go("tf_train_lora_dxa", .{ up(32 * rank, 256), u(rows), 1 }, 256, &a);
    }

    /// dx += dxa a: the change's part of its site's input gradient.
    pub fn loraDx(o: Ops, dxa: u64, a_: u64, dx: u64, rows: usize, in: usize, rank: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ dxa, a_, dx }) |v| a.add(v);
        for ([_]usize{ rows, in, rank }) |v| a.add(int(v));
        try o.go("tf_train_lora_dx", .{ up(in, 256), u(rows), 1 }, 256, &a);
    }

    /// rows rows of x added to a random sketch y [k, in] (first: the rows' place in the sketch).
    pub fn sketch(o: Ops, x: u64, x_stride: usize, y: u64, rows: usize, in: usize, k: usize, first: usize, seed: u32, w: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(long(x_stride));
        a.add(y);
        for ([_]usize{ rows, in, k }) |v| a.add(int(v));
        a.add(u(first));
        a.add(seed);
        a.add(w);
        try o.go("tf_train_sketch", .{ up(in, 256), u(k), 1 }, 256, &a);
    }

    /// One Adam step on n parameters; their gradient is cleared after.
    pub fn adam(o: Ops, p: u64, g: u64, m: u64, v: u64, n: usize, hp: [4]f32, corr: [2]f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ p, g, m, v }) |x| a.add(x);
        a.add(long(n));
        for (hp) |x| a.add(x);
        for (corr) |x| a.add(x);
        try o.go("tf_train_adam", .{ up(n, 256), 1, 1 }, 256, &a);
    }

    /// c [m, n] (+)= a [m, k] b [k, n], bf16 row-major in (row strides lda, ldb), fp32 out (ldc).
    pub fn gemm(o: Ops, a_: u64, b: u64, c: u64, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize, accumulate: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ a_, b, c }) |v| a.add(v);
        for ([_]usize{ m, n, k, lda, ldb, ldc }) |v| a.add(int(v));
        a.add(@as(c_int, @intFromBool(accumulate)));
        try o.go("tf_train_gemm", .{ up(n, 64), up(m, 64), 1 }, 128, &a);
    }

    /// A 4-bit MLX matrix of `count` values as bf16 in the same order.
    pub fn dequant(o: Ops, w: [3]u64, out: u64, count: usize) !void {
        var a: cuda.Args = .{};
        for (w) |v| a.add(v);
        a.add(out);
        a.add(long(count));
        try o.go("tf_train_dequant", .{ up(count, 256), 1, 1 }, 256, &a);
    }

    fn unary(o: Ops, comptime name: []const u8, ptrs: []const u64, n: usize) !void {
        var a: cuda.Args = .{};
        for (ptrs) |v| a.add(v);
        a.add(long(n));
        try o.go(name, .{ up(n, 256), 1, 1 }, 256, &a);
    }

    pub fn zero(o: Ops, x: u64, n: usize) !void {
        try o.unary("tf_train_zero", &.{x}, n);
    }

    pub fn widen(o: Ops, x: u64, out: u64, n: usize) !void {
        try o.unary("tf_train_widen", &.{ x, out }, n);
    }

    pub fn narrow(o: Ops, x: u64, out: u64, n: usize) !void {
        try o.unary("tf_train_narrow", &.{ x, out }, n);
    }

    pub fn add(o: Ops, x: u64, y: u64, n: usize) !void {
        try o.unary("tf_train_add", &.{ x, y }, n);
    }

    /// dst row r = src row (idx: idx[r], else r), `width` bf16 each, rows `src_stride` and `dst_stride` elements apart.
    pub fn copyRows(o: Ops, src: u64, src_stride: usize, idx: u64, dst: u64, dst_stride: usize, width: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(long(src_stride));
        a.add(idx);
        a.add(dst);
        a.add(long(dst_stride));
        a.add(int(width));
        a.add(int(n));
        try o.go("tf_train_rows", .{ up(width, 256), u(n), 1 }, 256, &a);
    }

    /// out[r] = wt[r * wt_stride + wt_at] g[r] over rows of `dim`.
    pub fn scaleRows(o: Ops, g: u64, wt: u64, wt_stride: usize, wt_at: usize, out: u64, n: usize, dim: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ g, wt }) |v| a.add(v);
        a.add(int(wt_stride));
        a.add(int(wt_at));
        a.add(out);
        a.add(int(dim));
        try o.go("tf_train_scale_rows", .{ up(dim, 256), u(n), 1 }, 256, &a);
    }

    /// dx [rows, width] (rows dx_stride apart) += g [rows, n] rest [n, width], a projection's bf16 rest.
    pub fn restBack(o: Ops, g: u64, rest: u64, dx: u64, dx_stride: usize, n_rows: usize, n: usize, width: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ g, rest, dx }) |v| a.add(v);
        a.add(long(dx_stride));
        for ([_]usize{ n_rows, n, width }) |v| a.add(int(v));
        try o.go("tf_train_rest_back", .{ up(width, 256), u(n_rows), 1 }, 256, &a);
    }

    /// du = da 2 sqrt(a) from the squared ReLU's bf16 output a.
    pub fn relu2Back(o: Ops, act: u64, da: u64, du: u64, n: usize) !void {
        try o.unary("tf_train_relu2_back", &.{ act, da, du }, n);
    }

    /// The routed pairs sorted by expert: order (slot to pair), inv (row, k to slot) and each expert's first slot.
    pub fn sortPairs(o: Ops, ids: u64, n_rows: usize, slots: usize, top_k: usize, experts: usize, order: u64, inv: u64, offsets: u64) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        for ([_]usize{ n_rows, slots, top_k, experts }) |v| a.add(int(v));
        for ([_]u64{ order, inv, offsets }) |v| a.add(v);
        try o.go("tf_train_sort_pairs", .{ 1, 1, 1 }, 256, &a);
    }

    /// logits [rows, experts] = x gate^T in fp32.
    pub fn routerLogits(o: Ops, x: u64, gate_w: u64, out: u64, n_rows: usize, experts: usize, dim: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, gate_w, out }) |v| a.add(v);
        for ([_]usize{ n_rows, experts, dim }) |v| a.add(int(v));
        try o.go("tf_train_logits", .{ up(32 * experts, 256), u(n_rows), 1 }, 256, &a);
    }

    /// dy [count, dim] at sorted slots: wt[pair] g[pair's row].
    pub fn pairsIn(o: Ops, order: u64, wt: u64, g: u64, dy: u64, dim: usize, slots: usize, count: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ order, wt, g, dy }) |v| a.add(v);
        a.add(int(dim));
        a.add(int(slots));
        try o.go("tf_train_pairs_in", .{ up(dim, 256), u(count), 1 }, 256, &a);
    }

    /// dx [rows, dim] += each row's top_k pairs' input gradients through inv.
    pub fn pairsOut(o: Ops, inv: u64, dxp: u64, dx: u64, dim: usize, top_k: usize, n_rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ inv, dxp, dx }) |v| a.add(v);
        a.add(int(dim));
        a.add(int(top_k));
        try o.go("tf_train_pairs_out", .{ up(dim, 256), u(n_rows), 1 }, 256, &a);
    }

    /// dx += the routing weights' gradient through their normalized sigmoid scores, into the router's input.
    pub fn routeBack(o: Ops, logit: u64, ids: u64, slots: usize, ys: u64, g: u64, gate_w: u64, dx: u64, n_rows: usize, experts: usize, top_k: usize, dim: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        a.add(logit);
        a.add(ids);
        a.add(int(slots));
        for ([_]u64{ ys, g, gate_w, dx }) |v| a.add(v);
        for ([_]usize{ experts, top_k, dim }) |v| a.add(int(v));
        a.add(scale);
        try o.go("tf_train_route_back", .{ u(n_rows), 1, 1 }, 256, &a);
    }

    /// y [count, k_out] = x [count, n_in] (sorted by expert) times each one's expert of the 4-bit w, 16 pairs a tile.
    pub fn expertProducts(o: Ops, x: u64, w: [3]u64, offsets: u64, tiles: u64, y: u64, count: usize, n_in: usize, k_out: usize, n_experts: usize) !void {
        std.debug.assert(n_in % 32 == 0 and k_out % 64 == 0);
        const cap = tileCap(count, n_experts);
        var a: cuda.Args = .{};
        a.add(offsets);
        a.add(tiles);
        for ([_]usize{ count, cap, n_experts }) |v| a.add(int(v));
        try o.go("tf_train_experts_tiles", .{ 1, 1, 1 }, 1, &a);
        var b: cuda.Args = .{};
        b.add(x);
        for (w) |v| b.add(v);
        b.add(tiles);
        b.add(offsets);
        b.add(y);
        for ([_]usize{ count, n_in, k_out, n_experts }) |v| b.add(int(v));
        try o.go("tf_train_experts_mm", .{ u(k_out / 64), u(cap), 1 }, 64, &b);
    }

    /// The conv before its bias and its SiLU with the bias, from the in-projection's xBC columns, as from position 0.
    pub fn convPre(o: Ops, proj: u64, conv_w: u64, conv_b: u64, conv_out: u64, act: u64, s: Shape, taps: usize, hist: ?u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, conv_w, conv_b, conv_out, act }) |v| a.add(v);
        a.add(s);
        a.add(int(taps));
        a.add(hist orelse proj);
        a.add(@as(c_int, @intFromBool(hist != null)));
        try o.go("tf_train_conv_pre", .{ up(@intCast(s.conv), 256), @intCast(s.rows), 1 }, 256, &a);
    }

    pub fn dt(o: Ops, proj: u64, bias: u64, out: u64, s: Shape) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, bias, out }) |v| a.add(v);
        a.add(s);
        try o.go("tf_train_dt", .{ @intCast(s.rows), 1, 1 }, @intCast(s.heads), &a);
    }

    /// The scan forward for every head: y with the D skip, and checkpoints every 8 steps.
    pub fn ssmFwd(o: Ops, act: u64, dt_: u64, a_neg: u64, d_skip: u64, y: u64, ckpt: u64, s: Shape, init: ?u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ act, dt_, a_neg, d_skip, y, ckpt }) |v| a.add(v);
        a.add(s);
        a.add(init orelse ckpt);
        a.add(@as(c_int, @intFromBool(init != null)));
        try o.go("tf_train_ssm_fwd", .{ @intCast(s.heads), 1, 1 }, 1024, &a);
    }

    /// The scan backward for every head: dact's x columns, per-head dB and dC partials, ddt.
    pub fn ssmBack(o: Ops, ptrs: [10]u64, s: Shape) !void {
        var a: cuda.Args = .{};
        for (ptrs) |v| a.add(v);
        a.add(s);
        try o.go("tf_train_ssm_back", .{ @intCast(s.heads), 1, 1 }, 1024, &a);
    }

    pub fn ssmBc(o: Ops, db_part: u64, dc_part: u64, dact: u64, s: Shape) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ db_part, dc_part, dact }) |v| a.add(v);
        a.add(s);
        try o.go("tf_train_ssm_bc", .{ @intCast(s.groups), @intCast(s.rows), 1 }, @intCast(s.n), &a);
    }

    /// The gate and grouped RMS norm undone: dy into the SSM's output, dz into dproj's z columns.
    pub fn gateBack(o: Ops, y: u64, proj: u64, w: u64, dn: u64, dy: u64, dproj: u64, s: Shape, eps: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ y, proj, w, dn, dy, dproj }) |v| a.add(v);
        a.add(s);
        a.add(eps);
        try o.go("tf_train_gate_back", .{ @intCast(s.groups), @intCast(s.rows), 1 }, 256, &a);
    }

    pub fn convBack(o: Ops, dact: u64, conv_out: u64, conv_w: u64, conv_b: u64, dproj: u64, s: Shape, taps: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ dact, conv_out, conv_w, conv_b, dproj }) |v| a.add(v);
        a.add(s);
        a.add(int(taps));
        try o.go("tf_train_conv_back", .{ up(@intCast(s.conv), 256), @intCast(s.rows), 1 }, 256, &a);
    }

    pub fn dtBack(o: Ops, proj: u64, bias: u64, ddt: u64, dproj: u64, s: Shape) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, bias, ddt, dproj }) |v| a.add(v);
        a.add(s);
        try o.go("tf_train_dt_back", .{ @intCast(s.rows), 1, 1 }, @intCast(s.heads), &a);
    }

    /// nemotron_ops' pocket over a pass's normed rows y, the neurons from `from` on times their row's scale.
    pub fn pocketRows(o: Ops, y: u64, d: usize, up_: u64, down: u64, n: usize, out: u64, out_f32: bool, rows: usize, row_mul: usize, row_add: usize, scale: u64, from: u32) !void {
        if (n > 64) return error.PocketTooWide;
        var a: cuda.Args = .{};
        a.add(y);
        a.add(int(d));
        a.add(up_);
        a.add(down);
        a.add(int(n));
        a.add(out);
        a.add(@as(c_int, @intFromBool(out_f32)));
        a.add(int(row_mul));
        a.add(int(row_add));
        a.add(scale);
        a.add(@as(c_int, @intCast(@min(from, std.math.maxInt(c_int)))));
        try o.go("tf_train_pocket_rows", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// Causal attention undone for each query row of each head (probabilities and their gradient kept), then dk, dv.
    pub fn attention(o: Ops, qkv: u64, dout: u64, probs: u64, dscores: u64, dq: u64, dk: u64, dv: u64, h: Heads, prefix: [2]u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ qkv, dout, probs, dscores, dq }) |v| a.add(v);
        a.add(h);
        a.add(prefix[0]);
        a.add(prefix[1]);
        try o.go("tf_train_attn_q", .{ @intCast(h.heads), @intCast(h.rows), 1 }, 256, &a);
        var b: cuda.Args = .{};
        for ([_]u64{ qkv, dout, probs, dscores, dk, dv }) |v| b.add(v);
        b.add(h);
        try o.go("tf_train_attn_kv", .{ @intCast(h.kv_heads), @intCast(h.rows), 1 }, @intCast(h.dim), &b);
    }
};
