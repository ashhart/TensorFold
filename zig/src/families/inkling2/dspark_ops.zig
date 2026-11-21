//! The DSpark drafter's kernels (inkling2/dspark.cu): tap mix, norms, q/k prep, block attention, the Markov head.
const std = @import("std");
const cuda = @import("cuda");

/// Keys a block attention chunk covers (absolute positions, so a row's bits never depend on the call).
pub const chunk = 1024;
/// Query rows x query heads of one key head a block attention call holds at most (dspark.cu's kMaxQ).
pub const max_q = 48;
/// The Markov rank the head's kernels hold (dspark.cu's 256-wide fold rows and rnn's shared rows).
pub const rank = 256;
/// The attention head width dspark.cu's kD fixes.
pub const head_dim = 128;
/// Rows a block drafts at most (dspark.cu's kMaxRows).
pub const max_rows = 8;

/// A draft head these kernels would run wrong is refused: any other Markov rank or head_dim, or a longer block.
pub fn checkHead(markov_rank: usize, dim: usize, block: usize) !void {
    if (markov_rank != rank or dim != head_dim or block == 0 or block > max_rows) return error.Inkling2DraftShape;
}

/// dspark.cu's QkArgs, field for field.
pub const QkArgs = extern struct {
    x: u64,
    qw: u64,
    kw: u64,
    q: u64,
    kc: u64,
    vc: u64,
    ld: i64,
    hq: i32,
    hk: i32,
    ring: i32,
    pos0: i32,
    eps: f32,
    theta: f32,
};

/// dspark.cu's AttnArgs, field for field.
pub const AttnArgs = extern struct {
    q: u64,
    kc: u64,
    vc: u64,
    po: u64,
    pml: u64,
    T: i32,
    hq: i32,
    hk: i32,
    ring: i32,
    lo: i32,
    hi: i32,
    c0: i32,
    scale: f32,
};

pub const Kernels = struct {
    mod: cuda.Module,
    /// act_rows: the SiLU is core's tf_swiglu_rows in torch's form
    rows: cuda.Module,
    mix: cuda.Function,
    addnorm: cuda.Function,
    qk: cuda.Function,
    attn: cuda.Function,
    merge: cuda.Function,
    silu: cuda.Function,
    ctx: cuda.Function,
    rnn: cuda.Function,
    vocab: cuda.Function,
    pick: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_dspark);
        errdefer mod.unload();
        var rows = try cuda.Module.load(d, cuda.kernels.act_rows);
        errdefer rows.unload();
        return .{
            .mod = mod,
            .rows = rows,
            .mix = try mod.function("tf_ink2_dsp_mix"),
            .addnorm = try mod.function("tf_ink2_dsp_addnorm"),
            .qk = try mod.function("tf_ink2_dsp_qk"),
            .attn = try mod.function("tf_ink2_dsp_attn"),
            .merge = try mod.function("tf_ink2_dsp_merge"),
            .silu = try rows.function("tf_swiglu_rows_expf"),
            .ctx = try mod.function("tf_ink2_dsp_ctx"),
            .rnn = try mod.function("tf_ink2_dsp_rnn"),
            .vocab = try mod.function("tf_ink2_dsp_vocab"),
            .pick = try mod.function("tf_ink2_dsp_pick"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.rows.unload();
        k.mod.unload();
        k.* = undefined;
    }
};

fn int(x: anytype) i32 {
    return @intCast(x);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// out bf16 [rows][c] = RMSNorm(alpha . taps[row]) * w; taps bf16 [rows][n][c], alpha fp32 [n].
pub fn mix(k: *const Kernels, s: cuda.Stream, taps: u64, alpha: u64, w: u64, out: u64, rows: usize, n: usize, c: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ taps, alpha, w, out }) |v| a.add(v);
    a.add(int(n));
    a.add(int(c));
    a.add(eps);
    try go(k.mix, s, .{ .x = @intCast(rows) }, 512, &a);
}

/// resid += `nparts` fp32 partials (`stride` elements apart, rank order); out = RMSNorm(resid) * w, both bf16.
pub fn addNorm(k: *const Kernels, s: cuda.Stream, resid: u64, parts: u64, nparts: usize, stride: usize, w: u64, out: u64, rows: usize, c: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    a.add(resid);
    a.add(parts);
    a.add(int(nparts));
    a.add(@as(i64, @intCast(stride)));
    a.add(w);
    a.add(out);
    a.add(int(c));
    a.add(eps);
    try go(k.addnorm, s, .{ .x = @intCast(rows) }, 512, &a);
}

pub fn qk(k: *const Kernels, s: cuda.Stream, q: QkArgs, rows: usize) !void {
    if (q.hq < 0 or q.hk <= 0 or q.ring <= 0 or q.ld < (q.hq + 2 * q.hk) * head_dim) return error.Inkling2DraftShape;
    var a: cuda.Args = .{};
    a.add(q);
    try go(k.qk, s, .{ .x = @intCast(rows), .y = @intCast(q.hq + q.hk) }, 64, &a);
}

/// Chunks a block's keys [lo, hi] span.
pub fn chunks(lo: usize, hi: usize) usize {
    return hi / chunk - lo / chunk + 1;
}

/// Block attention over keys [a.lo, a.hi], then the chunks merged into `out` bf16 [T][hq][128].
pub fn attend(k: *const Kernels, s: cuda.Stream, at: AttnArgs, out: u64) !void {
    const lo: usize = @intCast(at.lo);
    const hi: usize = @intCast(at.hi);
    if (lo > hi or at.T * @divExact(at.hq, at.hk) > max_q) return error.Inkling2DraftShape;
    if (at.ring <= 0 or hi - lo >= @as(usize, @intCast(at.ring))) return error.Inkling2DraftShape; // keys would alias
    var x = at;
    x.c0 = int(lo / chunk);
    const n = chunks(lo, hi);
    var a: cuda.Args = .{};
    a.add(x);
    try go(k.attn, s, .{ .x = @intCast(n), .y = @intCast(at.hk) }, 256, &a);
    var m: cuda.Args = .{};
    m.add(at.po);
    m.add(at.pml);
    m.add(int(n));
    m.add(at.T * at.hq);
    m.add(out);
    try go(k.merge, s, .{ .x = @intCast(at.T * at.hq) }, 128, &m);
}

/// gu bf16 [rows][2 * inter] (gates, then ups) to bf16(silu(gate) * up) by core's tf_swiglu_rows_expf.
pub fn silu(k: *const Kernels, s: cuda.Stream, gu: u64, out: u64, rows: usize, inter: usize) !void {
    if (inter % 64 != 0) return error.Inkling2DraftShape;
    var a: cuda.Args = .{};
    a.add(gu);
    a.add(@as(i64, @intCast(2 * inter)));
    a.add(gu + 2 * inter);
    a.add(@as(i64, @intCast(2 * inter)));
    a.add(out);
    a.add(@as(u64, 0));
    a.add(int(inter));
    try go(k.silu, s, .{ .x = @intCast(rows), .y = @intCast((inter / 8 + 255) / 256) }, 256, &a);
}

/// The Markov head's context rows: fp32 [rows][3r] from the hidden rows and f_rnn.
pub fn ctx(k: *const Kernels, s: cuda.Stream, hidden: u64, frnn: u64, w: u64, bias: u64, out: u64, rows: usize, c: usize, r: usize) !void {
    if (rows > max_rows or r != rank) return error.Inkling2DraftShape;
    var a: cuda.Args = .{};
    for ([_]u64{ hidden, frnn, w, bias, out }) |v| a.add(v);
    for ([_]i32{ int(rows), int(c), int(r) }) |v| a.add(v);
    try go(k.ctx, s, .{ .x = @intCast((3 * r + 7) / 8) }, 256, &a);
}

/// One recurrence step from walk[j]: the state in place and the step's latent.
pub fn rnn(k: *const Kernels, s: cuda.Stream, cx: u64, w: u64, w1: u64, walk: u64, j: usize, state: u64, latent: u64, r: usize, c: usize) !void {
    if (r != rank) return error.Inkling2DraftShape;
    var a: cuda.Args = .{};
    for ([_]u64{ cx, w, w1, walk }) |v| a.add(v);
    a.add(int(j));
    a.add(state);
    a.add(latent);
    a.add(int(r));
    a.add(int(c));
    try go(k.rnn, s, .{ .x = 1 }, @intCast(r), &a);
}

pub fn vocab(k: *const Kernels, s: cuda.Stream, base: u64, latent: u64, fold: u64, fbias: u64, cap: f32, valid: usize, nv: usize, out: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ base, latent, fold, fbias }) |v| a.add(v);
    a.add(cap);
    a.add(int(valid));
    a.add(int(nv));
    a.add(out);
    try go(k.vocab, s, .{ .x = @intCast((nv + 7) / 8) }, 256, &a);
}

pub fn pick(k: *const Kernels, s: cuda.Stream, picks: u64, world: usize, walk: u64, j: usize) !void {
    var a: cuda.Args = .{};
    a.add(picks);
    a.add(int(world));
    a.add(walk);
    a.add(int(j));
    try go(k.pick, s, .{ .x = 1 }, 1, &a);
}

test "dspark ops: argument structs keep dspark.cu's layouts; chunks by absolute position" {
    try std.testing.expectEqual(@as(usize, 6 * 8 + 8 + 4 * 4 + 2 * 4), @sizeOf(QkArgs));
    try std.testing.expectEqual(@as(usize, 5 * 8 + 8 * 4), @sizeOf(AttnArgs));
    try std.testing.expectEqual(@as(usize, 1), chunks(0, 1023));
    try std.testing.expectEqual(@as(usize, 2), chunks(1000, 1030));
}

test "dspark ops: a head, a block or a ring the kernels would run wrong is refused before a launch" {
    const no = error.Inkling2DraftShape;
    try checkHead(256, 128, 8);
    try std.testing.expectError(no, checkHead(8, 128, 8)); // tiny-random's rank
    try std.testing.expectError(no, checkHead(512, 128, 8));
    try std.testing.expectError(no, checkHead(256, 64, 8));
    try std.testing.expectError(no, checkHead(256, 128, 9));
    const k: Kernels = undefined;
    const s: cuda.Stream = undefined;
    try std.testing.expectError(no, silu(&k, s, 0, 0, 8, 4600)); // not a multiple of 64 columns
    try std.testing.expectError(no, rnn(&k, s, 0, 0, 0, 0, 0, 0, 0, 512, 4608));
    try std.testing.expectError(no, ctx(&k, s, 0, 0, 0, 0, 0, 8, 4608, 128));
    try std.testing.expectError(no, ctx(&k, s, 0, 0, 0, 0, 0, 9, 4608, 256));
    const q: QkArgs = .{ .x = 0, .qw = 0, .kw = 0, .q = 0, .kc = 0, .vc = 0, .ld = 32 * 64, .hq = 24, .hk = 4, .ring = 4096, .pos0 = 0, .eps = 1e-6, .theta = 1e6 };
    try std.testing.expectError(no, qk(&k, s, q, 8)); // 64-wide heads
    const at: AttnArgs = .{ .q = 0, .kc = 0, .vc = 0, .po = 0, .pml = 0, .T = 8, .hq = 24, .hk = 4, .ring = 4096, .lo = 0, .hi = 4096, .c0 = 0, .scale = 1 };
    try std.testing.expectError(no, attend(&k, s, at, 0)); // 4,097 keys in a 4,096-row ring
}
