//! Inkling 2 Mini loop math (inkling2/loop_math.cu): raw AR, two-bank messages, rank-64 gates and split QKV inputs.
const std = @import("std");
const cuda = @import("cuda");

pub const rank = 64;
pub const max_blocks = 8;
pub const max_queries = 3;

/// loop_math.cu's ArArgs, field for field. `out[q][row][c]` is FP32 AR output before any controller or RMSNorm.
pub const ArArgs = extern struct {
    blocks: [max_blocks]u64 = @splat(0), // BF16 [rows][c], each a completed block state
    partial: u64, // BF16 when partial_bf16 != 0, otherwise FP32; always the final AR source
    query: u64, // FP32 [nq][c]
    out: u64, // FP32 [nq][rows][c]
    n_blocks: i32,
    partial_bf16: i32,
    nq: i32,
    rows: i32,
    c: i32,
    eps: f32,
    blocks_normed: i32 = 0, // nonzero: `blocks` are FP32 rows that `prenorm` made (its AR output), already normalized
};

pub const Kernels = struct {
    mod: cuda.Module,
    ar_fn: cuda.Function,
    ar_reg_fn: cuda.Function,
    ar_reg_hn_fn: cuda.Function,
    messages_fn: cuda.Function,
    messages_norm_fn: cuda.Function,
    gate_fn: cuda.Function,
    gate_strided_fn: cuda.Function,
    gate_norm_fn: cuda.Function,
    gate_norm_wide_fn: cuda.Function,
    gate_down_fn: cuda.Function,
    gate_apply_fn: cuda.Function,
    gate_down_tc_fn: cuda.Function,
    gate_apply_tc_fn: cuda.Function,
    gate_down_tc3_fn: cuda.Function,
    gate_apply_tc3_fn: cuda.Function,
    rmsnorm_tc3_fn: cuda.Function,
    gate_norm3_fn: cuda.Function,
    gate_down3_fn: cuda.Function,
    gate_apply3_fn: cuda.Function,
    gate_down3_rows_fn: cuda.Function,
    gate_apply3_rows_fn: cuda.Function,
    rmsnorm3_fn: cuda.Function,
    rmsnorm1_fn: cuda.Function,
    project_qkv_fn: cuda.Function,
    round_bf16_fn: cuda.Function,
    prenorm_fn: cuda.Function,
    capture_tap_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop_math);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .ar_fn = try mod.function("tf_ink2_loop_ar"),
            .ar_reg_fn = try mod.function("tf_ink2_loop_ar_reg"),
            .ar_reg_hn_fn = try mod.function("tf_ink2_loop_ar_reg_hn"),
            .messages_fn = try mod.function("tf_ink2_loop_messages"),
            .messages_norm_fn = try mod.function("tf_ink2_loop_messages_norm"),
            .gate_fn = try mod.function("tf_ink2_loop_gate"),
            .gate_strided_fn = try mod.function("tf_ink2_loop_gate_strided"),
            .gate_norm_fn = try mod.function("tf_ink2_loop_gate_norm"),
            .gate_norm_wide_fn = try mod.function("tf_ink2_loop_gate_norm_wide"),
            .gate_down_fn = try mod.function("tf_ink2_loop_gate_down"),
            .gate_apply_fn = try mod.function("tf_ink2_loop_gate_apply"),
            .gate_down_tc_fn = try mod.function("tf_ink2_loop_gate_down_tc"),
            .gate_apply_tc_fn = try mod.function("tf_ink2_loop_gate_apply_tc"),
            .gate_down_tc3_fn = try mod.function("tf_ink2_loop_gate_down_tc3"),
            .gate_apply_tc3_fn = try mod.function("tf_ink2_loop_gate_apply_tc3"),
            .rmsnorm_tc3_fn = try mod.function("tf_ink2_loop_rmsnorm_tc3"),
            .gate_norm3_fn = try mod.function("tf_ink2_loop_gate_norm3"),
            .gate_down3_fn = try mod.function("tf_ink2_loop_gate_down3"),
            .gate_apply3_fn = try mod.function("tf_ink2_loop_gate_apply3"),
            .gate_down3_rows_fn = try mod.function("tf_ink2_loop_gate_down3_rows"),
            .gate_apply3_rows_fn = try mod.function("tf_ink2_loop_gate_apply3_rows"),
            .rmsnorm3_fn = try mod.function("tf_ink2_loop_rmsnorm3"),
            .rmsnorm1_fn = try mod.function("tf_ink2_loop_rmsnorm1"),
            .project_qkv_fn = try mod.function("tf_ink2_loop_project_qkv"),
            .round_bf16_fn = try mod.function("tf_ink2_loop_round_bf16"),
            .prenorm_fn = try mod.function("tf_ink2_loop_prenorm"),
            .capture_tap_fn = try mod.function("tf_ink2_capture_tap"),
        };
    }

    pub fn unload(k: *Kernels) void {
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

fn goShared(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, shared: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, s, a);
}

/// Unnormalized block attention residual. No scratch; nq <= 3 and at most 8 completed blocks plus `partial`.
/// Wide calls (rows > 16) with C <= 2560 take the single-load register-cached path (bit-exact vs tf_ink2_loop_ar,
/// reading the BF16 blocks once); decode rows keep tf_ink2_loop_ar on the prenormed FP32 blocks.
pub fn ar(k: *const Kernels, s: cuda.Stream, a: ArArgs) !void {
    if (a.n_blocks < 0 or a.n_blocks > max_blocks or a.nq <= 0 or a.nq > max_queries or a.rows <= 0 or a.c <= 0)
        return error.Inkling2LoopArShape;
    var args: cuda.Args = .{};
    args.add(a);
    const f = if (arWide(@intCast(a.rows), @intCast(a.c))) k.ar_reg_fn else k.ar_fn;
    try go(f, s, .{ .x = @intCast(a.rows) }, 512, &args);
}

/// Whether `ar` takes the register-cached kernel (and `arHn` takes these rows).
pub fn arWide(rows: usize, c: usize) bool {
    return rows > 16 and c <= 5 * 512;
}

/// `ar` on wide rows (`arWide`) that also writes `hn`, BF16 [nq][rows][c]: each output row's state half as the wide
/// gate normalizes it (its chain, tree and products, `gate_eps`), for `gateTc3`.
pub fn arHn(k: *const Kernels, s: cuda.Stream, a: ArArgs, hn: u64, gate_eps: f32) !void {
    if (a.n_blocks < 0 or a.n_blocks > max_blocks or a.nq <= 0 or a.nq > max_queries or a.rows <= 0 or a.c <= 0 or
        !arWide(@intCast(a.rows), @intCast(a.c)) or hn == 0) return error.Inkling2LoopArShape;
    var args: cuda.Args = .{};
    args.add(a);
    args.add(hn);
    args.add(gate_eps);
    try go(k.ar_reg_hn_fn, s, .{ .x = @intCast(a.rows) }, 512, &args);
}

/// A completed block's RMS normalization, once: `src` BF16 [rows][c] to FP32 rows for `ar` (`out_ar`) and for
/// ops.mix (`out_mix`), each with its consumer's sums and roundings, so a consumer given them skips its own pass and
/// keeps its bits.
pub fn prenorm(k: *const Kernels, s: cuda.Stream, src_bf16: u64, out_ar_f32: u64, out_mix_f32: u64, rows: usize, c: usize, eps: f32) !void {
    if (rows == 0 or c == 0) return error.Inkling2LoopPrenormShape;
    var a: cuda.Args = .{};
    for ([_]u64{ src_bf16, out_ar_f32, out_mix_f32 }) |v| a.add(v);
    a.add(int(c));
    a.add(eps);
    try go(k.prenorm_fn, s, .{ .x = @intCast(rows) }, 512, &a);
}

/// Messages: BF16 banks, even/odd qm/kv FP32 queries/output; row-local q/k/v/mlp, no scratch/position/mask/cache.
pub fn messages(
    k: *const Kernels,
    s: cuda.Stream,
    bank0_bf16: u64,
    bank1_bf16: u64,
    queries_qm_f32: u64,
    queries_kv_f32: u64,
    out_f32: u64,
    rows: usize,
    core_layers: usize,
    c: usize,
    eps: f32,
) !void {
    if (rows == 0 or core_layers == 0 or c == 0) return error.Inkling2LoopMessageShape;
    var a: cuda.Args = .{};
    for ([_]u64{ bank0_bf16, bank1_bf16, queries_qm_f32, queries_kv_f32, out_f32 }) |v| a.add(v);
    for ([_]i32{ int(rows), int(core_layers), int(c) }) |v| a.add(v);
    a.add(eps);
    try go(k.messages_fn, s, .{ .x = @intCast(rows), .y = 4 }, 256, &a);
}

/// `messages` for one core layer (c <= 2560) plus each (row, role) message's half of the wide gate's norm:
/// `mn` BF16 [rows][4][c] (the row and role `out` holds), the bits gate_norm writes from it with `gate_eps`.
pub fn messagesNorm(
    k: *const Kernels,
    s: cuda.Stream,
    bank0_bf16: u64,
    bank1_bf16: u64,
    queries_qm_f32: u64,
    queries_kv_f32: u64,
    out_f32: u64,
    mn_bf16: u64,
    rows: usize,
    c: usize,
    eps: f32,
    gate_eps: f32,
) !void {
    if (rows == 0 or c == 0 or c > 2560) return error.Inkling2LoopMessageShape;
    var a: cuda.Args = .{};
    for ([_]u64{ bank0_bf16, bank1_bf16, queries_qm_f32, queries_kv_f32, out_f32, mn_bf16 }) |v| a.add(v);
    for ([_]i32{ int(rows), int(c) }) |v| a.add(v);
    a.add(eps);
    a.add(gate_eps);
    try go(k.messages_norm_fn, s, .{ .x = @intCast(rows), .y = 4 }, 256, &a);
}

/// Controller: BF16 D[64][2c]/F/G[c][64] and down I/O; FP32 F/G dots/tanh/four-step output; pre-offset msg, no scratch.
pub fn gate(
    k: *const Kernels,
    s: cuda.Stream,
    state_f32: u64,
    message_f32: u64,
    down_bf16: u64,
    forget_bf16: u64,
    gain_bf16: u64,
    out_f32: u64,
    rows: usize,
    c: usize,
    eps: f32,
) !void {
    if (rows == 0 or c == 0) return error.Inkling2LoopGateShape;
    var a: cuda.Args = .{};
    for ([_]u64{ state_f32, message_f32, down_bf16, forget_bf16, gain_bf16, out_f32 }) |v| a.add(v);
    for ([_]i32{ int(rows), int(c) }) |v| a.add(v);
    a.add(eps);
    try go(k.gate_fn, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// Strided-message variant of `gate`: `mr = m + row * m_stride` (for `[row][role][c]` layouts).
pub fn gateStrided(
    k: *const Kernels,
    s: cuda.Stream,
    state_f32: u64,
    message_f32: u64,
    down_bf16: u64,
    forget_bf16: u64,
    gain_bf16: u64,
    out_f32: u64,
    rows: usize,
    c: usize,
    m_stride: usize,
    eps: f32,
) !void {
    if (rows == 0 or c == 0 or m_stride == 0) return error.Inkling2LoopGateShape;
    var a: cuda.Args = .{};
    for ([_]u64{ state_f32, message_f32, down_bf16, forget_bf16, gain_bf16, out_f32 }) |v| a.add(v);
    for ([_]i32{ int(rows), int(c), int(m_stride) }) |v| a.add(v);
    a.add(eps);
    try go(k.gate_strided_fn, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// Fast multi-block gate, bit-identical to `gateStrided`: norm (one block a row), rank-pair zetas
/// (one block a row and rank pair), then channels (one block a row and 256 channels). Same stream order,
/// static pointers only: CUDA-graph safe. `norm_hm` holds BF16 [rows][2*c], `zeta` BF16 [rows][64].
pub fn gateFast(
    k: *const Kernels,
    s: cuda.Stream,
    state_f32: u64,
    message_f32: u64,
    down_bf16: u64,
    forget_bf16: u64,
    gain_bf16: u64,
    out_f32: u64,
    norm_hm_u16: u64,
    zeta_u16: u64,
    rows: usize,
    c: usize,
    m_stride: usize,
    eps: f32,
) !void {
    if (rows == 0 or c == 0 or m_stride == 0) return error.Inkling2LoopGateShape;
    var n: cuda.Args = .{};
    for ([_]u64{ state_f32, message_f32, norm_hm_u16 }) |v| n.add(v);
    for ([_]i32{ int(rows), int(c), int(m_stride) }) |v| n.add(v);
    n.add(eps);
    try go(k.gate_norm_fn, s, .{ .x = @intCast(rows) }, 256, &n);
    if (rowTiled(rows)) {
        // one role through the row-tiled kernels (role 0's rows of norm_hm, zeta and out): the same bits
        const z = [3]u64{ 0, 0, 0 };
        const a: Gate3Args = .{
            .h = .{ state_f32, 0, 0 },
            .m = .{ message_f32, 0, 0 },
            .down = .{ down_bf16, 0, 0 },
            .forget = .{ forget_bf16, 0, 0 },
            .gain = .{ gain_bf16, 0, 0 },
            .norm = z,
            .norm_hm = norm_hm_u16,
            .zeta = zeta_u16,
            .out_raw = out_f32,
            .out_bf16 = 0,
            .rows = int(rows),
            .c = int(c),
            .m_stride = int(m_stride),
            .eps = eps,
        };
        return gateRows(k, s, a, 1);
    }
    var d: cuda.Args = .{};
    for ([_]u64{ norm_hm_u16, down_bf16, zeta_u16 }) |v| d.add(v);
    for ([_]i32{ int(rows), int(c) }) |v| d.add(v);
    try go(k.gate_down_fn, s, .{ .x = @intCast(rows), .y = rank / 2 }, 256, &d);
    var p: cuda.Args = .{};
    for ([_]u64{ state_f32, message_f32, zeta_u16, forget_bf16, gain_bf16, out_f32 }) |v| p.add(v);
    for ([_]i32{ int(rows), int(c), int(m_stride) }) |v| p.add(v);
    try go(k.gate_apply_fn, s, .{ .x = @intCast(rows), .y = @intCast((c + 255) / 256) }, 256, &p);
}

/// Whether `gateWideTc` takes these rows: whole 64-channel tiles, the row staged in at most 48 KB, and 16-byte
/// aligned FP32 rows for the float4/float2 accesses.
pub fn gateWideTcFits(state_f32: u64, message_f32: u64, out_f32: u64, c: usize, m_stride: usize) bool {
    return c > 0 and c % 64 == 0 and 8 * c <= 48 * 1024 and m_stride % 4 == 0 and
        state_f32 % 16 == 0 and message_f32 % 16 == 0 and out_f32 % 16 == 0;
}

/// Wide-row tensor-core gate (prefill rows), the scalar gate within fp tolerance: norm (gate_norm_wide, the exact
/// bits of gate_norm), zetas (mma, one block a 64-row tile), then channels (one block a 64x64 tile).
/// Same stream order, static pointers only: CUDA-graph safe. Requires gateWideTcFits.
pub fn gateWideTc(
    k: *const Kernels,
    s: cuda.Stream,
    state_f32: u64,
    message_f32: u64,
    down_bf16: u64,
    forget_bf16: u64,
    gain_bf16: u64,
    out_f32: u64,
    norm_hm_u16: u64,
    zeta_u16: u64,
    rows: usize,
    c: usize,
    m_stride: usize,
    eps: f32,
) !void {
    if (rows == 0 or m_stride == 0 or !gateWideTcFits(state_f32, message_f32, out_f32, c, m_stride)) return error.Inkling2LoopGateShape;
    var n: cuda.Args = .{};
    for ([_]u64{ state_f32, message_f32, norm_hm_u16 }) |v| n.add(v);
    for ([_]i32{ int(rows), int(c), int(m_stride) }) |v| n.add(v);
    n.add(eps);
    try goShared(k.gate_norm_wide_fn, s, .{ .x = @intCast(rows) }, 256, @intCast(8 * c), &n);
    var d: cuda.Args = .{};
    for ([_]u64{ norm_hm_u16, down_bf16, zeta_u16 }) |v| d.add(v);
    for ([_]i32{ int(rows), int(c) }) |v| d.add(v);
    try go(k.gate_down_tc_fn, s, .{ .x = @intCast((rows + 63) / 64) }, 256, &d);
    var p: cuda.Args = .{};
    for ([_]u64{ zeta_u16, forget_bf16, gain_bf16, state_f32, message_f32, out_f32 }) |v| p.add(v);
    for ([_]i32{ int(rows), int(c), int(m_stride) }) |v| p.add(v);
    try go(k.gate_apply_tc_fn, s, .{ .x = @intCast((rows + 63) / 64), .y = @intCast(c / 64) }, 256, &p);
}

/// loop_math.cu's GateTcArgs, field for field: up to three roles of the wide tensor-core gate, each from its
/// normalized halves (`hn` from `ar`, `mn` from `messagesNorm`), then each gated row's RMSNorm.
pub const GateTcArgs = extern struct {
    hn: [3]u64 = @splat(0),
    mn: [3]u64 = @splat(0),
    down: [3]u64 = @splat(0),
    forget: [3]u64 = @splat(0),
    gain: [3]u64 = @splat(0),
    h: [3]u64 = @splat(0),
    m: [3]u64 = @splat(0),
    out: [3]u64 = @splat(0),
    norm: [3]u64 = @splat(0),
    normed: [3]u64 = @splat(0),
    zeta: u64, // BF16 [roles][rows][64]
    rows: i32,
    c: i32,
    mn_ld: i32,
    m_stride: i32,
    eps: f32,
};

/// Whether `gateTc3` takes these rows: the wide tensor-core gate's tiles (gateWideTcFits) and rms_row's widths.
pub fn gateTc3Fits(c: usize) bool {
    return c > 0 and c % 64 == 0 and c <= 2560;
}

/// The wide tensor-core gate for `roles` roles (grid z), no norm pass: zetas from hn|mn, the gated rows into `out`,
/// then their RMSNorms into `normed`. The bits of `gateWideTc` and `rmsnorm1` a role, for a role's `hn`/`mn` from
/// `ar`/`messagesNorm`. No role's `out` may alias another's `h`. Static pointers only: CUDA-graph safe.
pub fn gateTc3(k: *const Kernels, s: cuda.Stream, a: GateTcArgs, roles: u32) !void {
    if (a.rows <= 0 or roles == 0 or roles > 3 or !gateTc3Fits(@intCast(a.c)) or a.m_stride <= 0 or a.mn_ld <= 0 or
        @mod(a.m_stride, 4) != 0 or @mod(a.mn_ld, 8) != 0) return error.Inkling2LoopGateShape;
    for (0..roles) |r| if (a.h[r] % 16 != 0 or a.m[r] % 16 != 0 or a.out[r] % 16 != 0 or a.hn[r] % 16 != 0 or a.mn[r] % 16 != 0)
        return error.Inkling2LoopGateShape;
    const rows: usize = @intCast(a.rows);
    const c: usize = @intCast(a.c);
    const tiles: u32 = @intCast((rows + 63) / 64);
    var d: cuda.Args = .{};
    d.add(a);
    try go(k.gate_down_tc3_fn, s, .{ .x = tiles, .z = roles }, 256, &d);
    var p: cuda.Args = .{};
    p.add(a);
    try go(k.gate_apply_tc3_fn, s, .{ .x = tiles, .y = @intCast(c / 64), .z = roles }, 256, &p);
    var n: cuda.Args = .{};
    n.add(a);
    try go(k.rmsnorm_tc3_fn, s, .{ .x = @intCast(rows), .y = roles }, 512, &n);
}

/// loop_math.cu's Gate3Args, field for field: the three gated roles' inputs/weights, the [3][rows]
/// scratch (norm pairs, zetas, fp32 gated rows, bf16 normed rows) and the shared geometry.
pub const Gate3Args = extern struct {
    h: [3]u64,
    m: [3]u64,
    down: [3]u64,
    forget: [3]u64,
    gain: [3]u64,
    norm: [3]u64,
    norm_hm: u64,
    zeta: u64,
    out_raw: u64,
    out_bf16: u64,
    rows: i32,
    c: i32,
    m_stride: i32,
    eps: f32,
};

/// Batched 3-role gate + RMSNorm, bit-identical to three single-role gateFast/rmsnorm1 calls:
/// norm (rows x 3), rank-pair zetas (rows x 32 x 3), channels (rows x cols x 3), then norms.
/// Same stream order, static pointers only: CUDA-graph safe.
pub fn gate3(k: *const Kernels, s: cuda.Stream, a: Gate3Args) !void {
    if (a.rows == 0 or a.c == 0 or a.c > 2560 or a.m_stride == 0) return error.Inkling2LoopGateShape;
    const rows: usize = @intCast(a.rows);
    const c: usize = @intCast(a.c);
    var n: cuda.Args = .{};
    n.add(a);
    try go(k.gate_norm3_fn, s, .{ .x = @intCast(rows), .y = 3 }, 256, &n);
    if (rowTiled(rows)) {
        try gateRows(k, s, a, 3);
    } else {
        var d: cuda.Args = .{};
        d.add(a);
        try go(k.gate_down3_fn, s, .{ .x = @intCast(rows), .y = rank / 2, .z = 3 }, 256, &d);
        var p: cuda.Args = .{};
        p.add(a);
        try go(k.gate_apply3_fn, s, .{ .x = @intCast(rows), .y = @intCast((c + 255) / 256), .z = 3 }, 256, &p);
    }
    var r: cuda.Args = .{};
    r.add(a);
    try go(k.rmsnorm3_fn, s, .{ .x = @intCast(rows), .y = 3 }, 512, &r);
}

/// Rows of down and apply blocks in the row-tiled gate (loop_math.cu ink2rt), and its apply block's channels.
pub const down_rows = 4;
pub const apply_rows = 8;
pub const apply_threads = 128;

/// Whether the scalar gate's down and apply take the row-tiled kernels: from 2 rows (verify windows and batched
/// calls), where a weight row read once a tile beats once a row (tests/gate_rows_narrow.log); one-row decode keeps
/// the row-a-block kernels. Either way the same bits.
pub fn rowTiled(rows: usize) bool {
    return rows >= 2;
}

/// The scalar gate's zetas and gated rows for `roles` roles (grid z) over row tiles, from `a.norm_hm`: the bits of
/// gate_down3 / gate_apply3 (roles 3) or gate_down / gate_apply (one role, in role 0's rows).
fn gateRows(k: *const Kernels, s: cuda.Stream, a: Gate3Args, roles: u32) !void {
    const rows: usize = @intCast(a.rows);
    const c: usize = @intCast(a.c);
    var d: cuda.Args = .{};
    d.add(a);
    try go(k.gate_down3_rows_fn, s, .{ .x = @intCast((rows + down_rows - 1) / down_rows), .y = rank / 2, .z = roles }, 256, &d);
    var p: cuda.Args = .{};
    p.add(a);
    try go(k.gate_apply3_rows_fn, s, .{ .x = @intCast((rows + apply_rows - 1) / apply_rows), .y = @intCast((c + apply_threads - 1) / apply_threads), .z = roles }, apply_threads, &p);
}

/// Single RMSNorm over an fp32 row (bit-exact vs `mix` with n_blocks=0 and no query).
pub fn rmsnorm1(k: *const Kernels, s: cuda.Stream, x_f32: u64, norm_bf16: u64, out_bf16: u64, rows: usize, c: usize, eps: f32) !void {
    if (rows == 0 or c == 0 or c > 2560) return error.Inkling2LoopGateShape;
    var a: cuda.Args = .{};
    for ([_]u64{ x_f32, norm_bf16, out_bf16 }) |v| a.add(v);
    for ([_]i32{ int(rows), int(c) }) |v| a.add(v);
    a.add(eps);
    try go(k.rmsnorm1_fn, s, .{ .x = @intCast(rows) }, 512, &a);
}

/// Mini QKV: normalized BF16 uq/uk/uv, query-side u_q; per-visit K/V conv/cache, then wo/post-conv; no history.
pub fn projectQkv(
    k: *const Kernels,
    s: cuda.Stream,
    u_q: u64,
    u_k: u64,
    u_v: u64,
    wq: u64,
    wk: u64,
    wv: u64,
    raw_q: u64,
    raw_k: u64,
    raw_v: u64,
    rows: usize,
    c: usize,
    q_width: usize,
    kv_width: usize,
) !void {
    if (rows == 0 or c == 0 or q_width == 0 or kv_width == 0) return error.Inkling2LoopQkvShape;
    var a: cuda.Args = .{};
    for ([_]u64{ u_q, u_k, u_v, wq, wk, wv, raw_q, raw_k, raw_v }) |v| a.add(v);
    for ([_]i32{ int(rows), int(c), int(q_width), int(kv_width) }) |v| a.add(v);
    try go(k.project_qkv_fn, s, .{ .x = @intCast((rows + 15) / 16), .y = @intCast((@max(q_width, kv_width) + 15) / 16), .z = 3 }, 256, &a);
}

/// BF16-round FP32 then widen, preserving section 6.3 unlike rowdot(false); in/out may alias, no scratch, n=0 no-op.
pub fn roundBf16(k: *const Kernels, s: cuda.Stream, in_f32: u64, out_f32: u64, n: usize) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    for ([_]u64{ in_f32, out_f32 }) |v| a.add(v);
    a.add(@as(i64, @intCast(n)));
    try go(k.round_bf16_fn, s, .{ .x = @intCast((n + 255) / 256) }, 256, &a);
}

/// Copy [rows, C] bf16 into taps[row][channel][C] (4 tap channels per row).
pub fn captureTap(k: *const Kernels, s: cuda.Stream, src_bf16: u64, dst_taps_bf16: u64, rows: usize, c: usize, channel: usize) !void {
    const total = rows * c;
    if (total == 0) return;
    var a: cuda.Args = .{};
    a.add(src_bf16);
    a.add(dst_taps_bf16);
    for ([_]i32{ int(rows), int(c), int(channel) }) |v| a.add(v);
    try go(k.capture_tap_fn, s, .{ .x = @intCast((total + 255) / 256) }, 256, &a);
}

test "loop math argument ABI and message role offsets" {
    try std.testing.expectEqual(@as(usize, 120), @sizeOf(ArArgs));
    try std.testing.expectEqual(@as(usize, 30 * 8 + 8 + 5 * 4 + 4), @sizeOf(GateTcArgs));
    const c = 2304;
    const j = 7;
    const row = 3;
    const core_layers = 12;
    const base = ((row * core_layers + j) * 4) * c;
    for ([_]usize{ 0, 1, 2, 3 }, 0..) |role, i| {
        const at = base + role * c;
        const queries = @as(usize, 2 * j + @as(usize, @intFromBool(i == 2 or i == 3))) * c;
        try std.testing.expect(at >= base and queries < 24 * c);
    }
}

test "the row-tiled gate takes every row of 2..max_batch_rows calls, its tiles as loop_math.cu's ink2rt" {
    try std.testing.expectEqual(@as(usize, 192), @sizeOf(Gate3Args));
    try std.testing.expect(!rowTiled(1));
    for (2..129) |rows| {
        try std.testing.expect(rowTiled(rows));
        // every row in exactly one tile: the last tile's rows past `rows` store nothing
        try std.testing.expect(((rows + down_rows - 1) / down_rows) * down_rows >= rows);
        try std.testing.expect(((rows + apply_rows - 1) / apply_rows) * apply_rows >= rows);
    }
    try std.testing.expectEqual(@as(usize, 18), (2304 + apply_threads - 1) / apply_threads);
}
