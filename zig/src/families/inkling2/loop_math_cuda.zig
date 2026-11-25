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
};

pub const Kernels = struct {
    mod: cuda.Module,
    ar_fn: cuda.Function,
    messages_fn: cuda.Function,
    gate_down_fn: cuda.Function,
    gate_up_fn: cuda.Function,
    project_qkv_fn: cuda.Function,
    qkv_rows_fns: [3]cuda.Function, // decode QKV for at most 1, 4 and 16 rows
    round_bf16_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop_math);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .ar_fn = try mod.function("tf_ink2_loop_ar"),
            .messages_fn = try mod.function("tf_ink2_loop_messages"),
            .gate_down_fn = try mod.function("tf_ink2_loop_gate_down"),
            .gate_up_fn = try mod.function("tf_ink2_loop_gate_up"),
            .project_qkv_fn = try mod.function("tf_ink2_loop_project_qkv"),
            .qkv_rows_fns = .{
                try mod.function("tf_ink2_loop_qkv_rows1"),
                try mod.function("tf_ink2_loop_qkv_rows4"),
                try mod.function("tf_ink2_loop_qkv_rows16"),
            },
            .round_bf16_fn = try mod.function("tf_ink2_loop_round_bf16"),
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

/// Unnormalized block attention residual. No scratch; nq <= 3 and at most 8 completed blocks plus `partial`.
pub fn ar(k: *const Kernels, s: cuda.Stream, a: ArArgs) !void {
    if (a.n_blocks < 0 or a.n_blocks > max_blocks or a.nq <= 0 or a.nq > max_queries or a.rows <= 0 or a.c <= 0)
        return error.Inkling2LoopArShape;
    var args: cuda.Args = .{};
    args.add(a);
    try go(k.ar_fn, s, .{ .x = @intCast(a.rows) }, 512, &args);
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

/// Controller: BF16 D/F/G, FP32 F/G dots/tanh/four-step output; zeta [rows][64] BF16 scratch; out may alias state.
pub fn gate(
    k: *const Kernels,
    s: cuda.Stream,
    state_f32: u64,
    message_f32: u64,
    down_bf16: u64,
    forget_bf16: u64,
    gain_bf16: u64,
    zeta_bf16: u64,
    out_f32: u64,
    rows: usize,
    c: usize,
    eps: f32,
) !void {
    try gateDown(k, s, state_f32, message_f32, down_bf16, zeta_bf16, rows, rank / 2, c, eps);
    try gateUp(k, s, state_f32, message_f32, zeta_bf16, forget_bf16, gain_bf16, out_f32, rows, c);
}

/// The down half alone: `pairs` rank pairs from the first (all 32 in `gate`; fewer only for a contract control).
pub fn gateDown(k: *const Kernels, s: cuda.Stream, state_f32: u64, message_f32: u64, down_bf16: u64, zeta_bf16: u64, rows: usize, pairs: usize, c: usize, eps: f32) !void {
    if (rows == 0 or c == 0 or pairs == 0 or pairs > rank / 2) return error.Inkling2LoopGateShape;
    var a: cuda.Args = .{};
    for ([_]u64{ state_f32, message_f32, down_bf16, zeta_bf16 }) |v| a.add(v);
    a.add(int(c));
    a.add(eps);
    try go(k.gate_down_fn, s, .{ .x = @intCast(rows), .y = @intCast(pairs) }, 256, &a);
}

/// The up half: forget/gain from zeta and the four-step output, 64 columns a block.
pub fn gateUp(k: *const Kernels, s: cuda.Stream, state_f32: u64, message_f32: u64, zeta_bf16: u64, forget_bf16: u64, gain_bf16: u64, out_f32: u64, rows: usize, c: usize) !void {
    if (rows == 0 or c == 0) return error.Inkling2LoopGateShape;
    var a: cuda.Args = .{};
    for ([_]u64{ state_f32, message_f32, zeta_bf16, forget_bf16, gain_bf16, out_f32 }) |v| a.add(v);
    a.add(int(c));
    try go(k.gate_up_fn, s, .{ .x = @intCast(rows), .y = @intCast((c + 63) / 64) }, 64, &a);
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
    const at = [9]u64{ u_q, u_k, u_v, wq, wk, wv, raw_q, raw_k, raw_v };
    return qkvLaunch(k, s, at, rows, c, q_width, kv_width, rows <= 16 and c % 64 == 0);
}

/// tf_ink2_loop_project_qkv's 16x16 tiles at any row count (projectQkv's order): the decode kernels' baseline.
pub fn projectQkvTiles(k: *const Kernels, s: cuda.Stream, at: [9]u64, rows: usize, c: usize, q_width: usize, kv_width: usize) !void {
    return qkvLaunch(k, s, at, rows, c, q_width, kv_width, false);
}

fn qkvLaunch(k: *const Kernels, s: cuda.Stream, at: [9]u64, rows: usize, c: usize, q_width: usize, kv_width: usize, decode: bool) !void {
    if (rows == 0 or c == 0 or q_width == 0 or kv_width == 0) return error.Inkling2LoopQkvShape;
    var a: cuda.Args = .{};
    for (at) |v| a.add(v);
    for ([_]i32{ int(rows), int(c), int(q_width), int(kv_width) }) |v| a.add(v);
    if (decode) { // a warp 32 outputs, the same sums in the same order
        const f = k.qkv_rows_fns[if (rows == 1) 0 else if (rows <= 4) 1 else 2];
        return go(f, s, .{ .x = @intCast((@max(q_width, kv_width) + 31) / 32), .y = 3 }, 32, &a);
    }
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

test "loop math argument ABI and message role offsets" {
    try std.testing.expectEqual(@as(usize, 112), @sizeOf(ArArgs));
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
