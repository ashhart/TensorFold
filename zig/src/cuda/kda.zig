//! Shared CUDA KDA: 128 key/value channels, four conv taps, BF16 rows and FP32 state and sums.

const std = @import("std");
const cuda = @import("root.zig");

/// Key and value channels a head (kda.cuh DK, DV); zig/kda_header_host.zig holds these and taps to the header.
pub const dk = 128;
pub const dv = 128;
/// The conv window's taps (kda.cuh TAPS): three kept rows and the current one.
pub const taps = 4;
/// Windows of this many rows or more (prefill chunks) take the three-kernel chain (kda.WIDE_ROWS).
pub const wide_rows = 64;

/// A model's KDA shape against the one the kernels are built for, refused before anything is allocated or launched.
pub fn geometry(key_dim: usize, value_dim: usize, conv_taps: usize) error{KdaGeometry}!void {
    if (key_dim != dk or value_dim != dv or conv_taps != taps) return error.KdaGeometry;
}

/// glm_kda.cu's entries.
pub const Kernels = struct {
    chain: cuda.Function,
    replay: cuda.Function,
    replay_layers: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (.{ "chain", "replay", "replay_layers", "prep", "step", "out" }) |name| @field(k, name) = try m.function("tf_glm_kda_" ++ name);
        return k;
    }
};

/// One layer's chain inputs, `heads` heads: p rows [q|k|v|f_a|g_a|...] (beta at `b_off`), decay and gate rows, states.
pub const Chain = struct {
    p: u64,
    p_stride: usize,
    b_off: usize,
    a: u64,
    a_stride: usize,
    g: u64,
    g_stride: usize,
    conv_state: u64,
    conv_w: u64,
    state_in: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    heads: usize,
};

/// Where a chain writes: out bf16, state_out like state_in, and replay's rows: k fp32, v bf16, decay and beta fp32.
pub const Saves = struct { out: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };

/// The three-kernel chain's scratch: q fp32 and the read-out bf16, [rows][heads][128] each.
pub const Wide = struct { q: u64, y: u64 };

/// kda.chain: `rows` rows through the fused kernel, or with `wide` through prep, step and out (WARPS 8, TR 16).
pub fn chain(ks: *const Kernels, stream: cuda.Stream, c: Chain, rows: usize, s: Saves, wide: ?Wide) !void {
    try validate(c, rows);
    const h: c_int = @intCast(c.heads);
    if (wide) |w| {
        var a: cuda.Args = .{};
        a.add(h);
        a.add(c.p);
        a.add(@as(c_int, @intCast(c.p_stride)));
        a.add(@as(c_int, @intCast(c.b_off)));
        a.add(c.a);
        a.add(@as(c_int, @intCast(c.a_stride)));
        for ([_]u64{ c.conv_state, c.conv_w, c.a_log, c.dt_bias }) |v| a.add(v);
        a.add(c.lower);
        for ([_]u64{ w.q, s.k, s.v, s.g, s.b }) |v| a.add(v);
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(rows), .y = @intCast(c.heads) };
        try cuda.launch.launch(ks.prep, .{ .grid = grid, .block = .{ .x = 512 } }, stream, &a);
        var b: cuda.Args = .{};
        b.add(h);
        for ([_]u64{ c.state_in, w.q, s.k, s.v, s.g, s.b }) |v| b.add(v);
        b.add(@as(c_int, @intCast(rows)));
        b.add(w.y);
        b.add(s.state_out);
        try cuda.launch.launch(ks.step, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / 8 }, .block = .{ .x = 8 * 32 } }, stream, &b);
        var o: cuda.Args = .{};
        o.add(h);
        o.add(w.y);
        o.add(c.g);
        o.add(@as(c_int, @intCast(c.g_stride)));
        o.add(c.norm_w);
        o.add(c.eps);
        o.add(s.out);
        return cuda.launch.launch(ks.out, .{ .grid = grid, .block = .{ .x = dv } }, stream, &o);
    }
    var a: cuda.Args = .{};
    a.add(h);
    a.add(c.p);
    a.add(@as(c_int, @intCast(c.p_stride)));
    a.add(@as(c_int, @intCast(c.b_off)));
    a.add(c.a);
    a.add(@as(c_int, @intCast(c.a_stride)));
    a.add(c.g);
    a.add(@as(c_int, @intCast(c.g_stride)));
    for ([_]u64{ c.conv_state, c.conv_w, c.state_in, c.a_log, c.dt_bias, c.norm_w }) |v| a.add(v);
    a.add(c.eps);
    a.add(c.lower);
    a.add(@as(c_int, @intCast(rows)));
    for ([_]u64{ s.out, s.state_out, s.k, s.v, s.g, s.b }) |v| a.add(v);
    try cuda.launch.launch(ks.chain, .{ .grid = .{ .x = @intCast(c.heads) }, .block = .{ .x = 1024 } }, stream, &a);
}

/// kda.replay: the state after the first `rows` saved rows, from state_in.
pub fn replay(ks: *const Kernels, stream: cuda.Stream, state_in: u64, k: u64, v: u64, g: u64, b: u64, rows: usize, heads: usize, state_out: u64) !void {
    if (heads == 0 or heads > std.math.maxInt(c_int) or rows > std.math.maxInt(c_int)) return error.KdaShape;
    var a: cuda.Args = .{};
    a.add(@as(c_int, @intCast(heads)));
    for ([_]u64{ state_in, k, v, g, b }) |x| a.add(x);
    a.add(@as(c_int, @intCast(rows)));
    a.add(state_out);
    try cuda.launch.launch(ks.replay, .{ .grid = .{ .x = @intCast(heads) }, .block = .{ .x = 1024 } }, stream, &a);
}

/// The fixed kernel envelope; strides are elements, never bytes. Refuse before narrowing or launching.
fn validate(c: Chain, rows: usize) !void {
    const limit = std.math.maxInt(c_int);
    if (c.heads == 0 or c.heads > limit / (3 * dk) or rows == 0 or rows > limit) return error.KdaShape;
    const n = c.heads * dk;
    if (c.p_stride > limit or c.a_stride > limit or c.g_stride > limit or c.b_off > limit) return error.KdaShape;
    if (c.b_off < 3 * n or c.b_off + c.heads > c.p_stride or c.a_stride < n or c.g_stride < n) return error.KdaShape;
}

/// Batched replay strides are FP32/BF16 elements as declared by the original kernel, not byte offsets.
pub const Replay = struct {
    state_in: u64,
    state_stride: usize,
    k: u64,
    v: u64,
    g: u64,
    b: u64,
    kv_stride: usize,
    b_stride: usize,
    state_out: u64,
};

/// Restore each layer after a rejected suffix, preserving the saved-row order and the original launch ABI.
pub fn replayLayers(ks: *const Kernels, stream: cuda.Stream, s: Replay, rows: usize, heads: usize, layers: usize) !void {
    if (heads == 0 or heads > std.math.maxInt(c_int) or rows > std.math.maxInt(c_int) or
        layers == 0 or layers > std.math.maxInt(u32) / heads) return error.KdaShape;
    var a: cuda.Args = .{};
    a.add(@as(c_int, @intCast(heads)));
    a.add(s.state_in);
    a.add(s.state_stride);
    for ([_]u64{ s.k, s.v, s.g, s.b }) |x| a.add(x);
    a.add(s.kv_stride);
    a.add(s.b_stride);
    a.add(@as(c_int, @intCast(rows)));
    a.add(s.state_out);
    try cuda.launch.launch(ks.replay_layers, .{ .grid = .{ .x = @intCast(layers * heads) }, .block = .{ .x = 1024 } }, stream, &a);
}

test {
    _ = @import("kda_test.zig");
}
