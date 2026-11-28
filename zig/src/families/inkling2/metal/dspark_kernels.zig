//! Mini DSpark's fixed-shape arithmetic: Metal entry points and exact launch ABIs.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
const q4 = @import("draft_head_q4.zig");

pub const hidden: u32 = 2304;
pub const query_width: u32 = 3072;
pub const kv_width: u32 = 1024;
pub const block_rows: u32 = 8;
pub const rank: u32 = 256;
pub const rnn_width: u32 = 768;
pub const intermediate: u32 = 9216;
pub const vocab: u32 = 200058;
pub const Ref = @import("inkling2_metal_ref").Ref;
pub const FuseArgs = extern struct { count: u32, stride_rows: u32 };
pub const PositionArgs = extern struct { base: [8]u32 };
pub const ContextArgs = extern struct { base: u32, rows: u32, ring: u32 };
pub const AttentionArgs = extern struct { local: u32, streams: u32, committed: [8]u32, ring: [8]u32, from: [8]u32 };
pub const KappaArgs = extern struct { rows: u32, stride_s: u32, row_off: u32, streams: u32 };
pub const TokenArgs = extern struct { token: u32, step: u32, use_held: u32 };
pub const ResidualArgs = extern struct { init: u32 };
pub const CountArgs = extern struct { count: u32 };
pub const BaseArgs = extern struct { rows: u32, vocab: u32 };
pub const LogitsArgs = extern struct { count: u32, phi_stride: u32, base_stride: u32, streams: u32, step: u32, mode: u32, flags: u32 };
pub const GroupStat = extern struct { m: f32, l: f32, best_v: f32, best_id: u32 };
pub const LogitsQ4Args = q4.LogitsArgs;

/// Round through binary32 and nearest-even BF16, returning an exactly representable FP64 value.
pub fn roundBf16(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x)) @truncate((bits >> 16) | 0x40) else @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const value: f32 = @bitCast(@as(u32, rounded) << 16);
    return value;
}

/// Rotate-half RoPE partner is d+64 below the midpoint and d-64 above it.
pub fn ropeRotateHalf(x: f32, other: f32, d: u32, angle: f32) f32 {
    return if (d < 64) x * @cos(angle) - other * @sin(angle) else other * @sin(angle) + x * @cos(angle);
}

/// Every recurrent blend operation rounds independently to model precision.
pub fn recurrentBlend(gamma: f32, candidate: f32, previous: f32) f64 {
    const keep: f32 = @floatCast(roundBf16(@as(f32, 1.0 - gamma)));
    const left: f32 = @floatCast(roundBf16(@as(f32, gamma * previous)));
    const right: f32 = @floatCast(roundBf16(@as(f32, keep * candidate)));
    return roundBf16(@as(f32, left + right));
}

/// Folded bias and base each round before the target-style softcap.
pub fn foldedLogit(base_value: f32, beta: f32, dot: f32) f64 {
    const bias: f32 = @floatCast(roundBf16(@as(f32, beta + dot)));
    const logit: f32 = @floatCast(roundBf16(@as(f32, base_value + bias)));
    return roundBf16(@as(f32, 30.0) * std.math.tanh(logit / @as(f32, 30.0)));
}

pub const Kind = enum {
    init_rows,
    fuse_taps,
    block_qkv,
    context_kv,
    attention,
    residual_norm,
    swiglu,
    kappa,
    recurrent,
    base,
    logits,
    logits_q4,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .init_rows => "tf_ink2_dspark_init_rows",
            .fuse_taps => "tf_ink2_dspark_fuse_taps",
            .block_qkv => "tf_ink2_dspark_block_qkv",
            .context_kv => "tf_ink2_dspark_context_kv",
            .attention => "tf_ink2_dspark_attention",
            .residual_norm => "tf_ink2_dspark_residual_norm",
            .swiglu => "tf_ink2_dspark_swiglu",
            .kappa => "tf_ink2_dspark_kappa",
            .recurrent => "tf_ink2_dspark_recurrent",
            .base => "tf_ink2_dspark_base",
            .logits => "tf_ink2_dspark_logits",
            .logits_q4 => "tf_ink2_dspark_logits_q4",
        };
    }

    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            inline else => |v| std.fmt.comptimePrint("metal inkling2 dspark {s} {s}", .{ @tagName(v), comptime v.symbol() }),
        };
    }

    /// Required threadgroup width matching the Metal kernel reservation and host dispatch.
    pub fn requiredThreads(k: Kind) u32 {
        return switch (k) {
            .kappa => 64,
            .block_qkv, .context_kv => 128,
            .attention => 768,
            .logits_q4 => 256,
            else => 256,
        };
    }
};

pub const Pipelines = struct {
    init_rows: mtl.Pipeline,
    fuse_taps: mtl.Pipeline,
    block_qkv: mtl.Pipeline,
    context_kv: mtl.Pipeline,
    attention: mtl.Pipeline,
    residual_norm: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    kappa: mtl.Pipeline,
    recurrent: mtl.Pipeline,
    base: mtl.Pipeline,
    logits: mtl.Pipeline,
    logits_q4: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.inkling2_dspark);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const prefix = "#define TF_SIMD_FRAGS 1\n";
        const owned = if (device.tensorUnits()) null else try std.mem.concat(std.heap.page_allocator, u8, &.{ prefix, text });
        defer if (owned) |buf| std.heap.page_allocator.free(buf);
        const lib = try mtl.Library.fromSource(device, owned orelse text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        var loaded: usize = 0;
        errdefer inline for (@typeInfo(Pipelines).@"struct".field_names, 0..) |name, i| {
            if (loaded > i) @field(out, name).deinit();
        };
        inline for (@typeInfo(Pipelines).@"struct".field_names, 0..) |name, i| {
            const kind: Kind = @fromBackingInt(@intCast(i));
            @field(out, name) = try mtl.Pipeline.init(device, lib, kind.symbol(), false);
            loaded += 1;
            if (@field(out, name).maxThreads() < kind.requiredThreads()) return error.DsparkThreads;
        }
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| @field(p, name).deinit();
        p.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}
fn groups(e: mtl.ComputeEncoder, count: usize, width: usize) void {
    e.dispatchGroups(mtl.Size.of(count, 1, 1), mtl.Size.of(width, 1, 1));
}
fn threads(e: mtl.ComputeEncoder, count: usize, width: usize) void {
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(width, 1, 1));
}

/// Row zero of each stream copies its normalized anchor; rows 1..7 copy the raw mask embedding.
pub fn initializeRows(p: *const Pipelines, e: mtl.ComputeEncoder, anchor: Ref, mask_embed: Ref, out: Ref, streams: u32) void {
    std.debug.assert(streams > 0 and streams <= 8);
    e.setPipeline(p.init_rows);
    bind(e, &.{ anchor, mask_embed, out });
    groups(e, @as(usize, streams) * block_rows, 256);
}

/// Input taps are [row][tap][H], output c is [fusion][row_in_batch][H], and newest_f is one [H] row.
pub fn fuseTaps(p: *const Pipelines, e: mtl.ComputeEncoder, taps: Ref, fuse_weight: Ref, norm_weight: Ref, c: Ref, newest_f: Ref, rows: u32, stride_rows: u32) void {
    std.debug.assert(rows > 0 and stride_rows >= rows);
    e.setPipeline(p.fuse_taps);
    bind(e, &.{ taps, fuse_weight, norm_weight, c, newest_f });
    e.setValue(FuseArgs{ .count = rows, .stride_rows = stride_rows }, 5);
    groups(e, 4 * @as(usize, rows), 256);
}

/// Raw projections become per-head RMS-normalized rotate-half Q/K and unchanged BF16 V across batched streams.
pub fn blockQkv(p: *const Pipelines, e: mtl.ComputeEncoder, q_raw: Ref, k_raw: Ref, v_raw: Ref, q_norm: Ref, k_norm: Ref, q: Ref, k: Ref, v: Ref, bases: []const u32) void {
    std.debug.assert(bases.len > 0 and bases.len <= 8);
    var pos: PositionArgs = .{ .base = @splat(0) };
    @memcpy(pos.base[0..bases.len], bases);
    e.setPipeline(p.block_qkv);
    bind(e, &.{ q_raw, k_raw, v_raw, q_norm, k_norm, q, k, v });
    e.setValue(pos, 8);
    groups(e, bases.len * 8 * 40, 128);
}

/// Context K/V is written at absolute-position modulo ring_capacity for committed rows only.
pub fn commitContext(p: *const Pipelines, e: mtl.ComputeEncoder, k_raw: Ref, v_raw: Ref, k_norm_ctx: Ref, keys: Ref, values: Ref, base_position: u32, rows: u32, ring_capacity: u32) void {
    std.debug.assert(rows > 0 and ring_capacity > 0);
    e.setPipeline(p.context_kv);
    bind(e, &.{ k_raw, v_raw, k_norm_ctx, keys, values });
    e.setValue(ContextArgs{ .base = base_position, .rows = rows, .ring = ring_capacity }, 5);
    groups(e, @as(usize, rows) * 8, 128);
}

/// Online-softmax 768-thread attention across batched streams, with zero intermediate score scratch.
pub fn attend(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, block_k: Ref, block_v: Ref, out: Ref, ctx_k: []const Ref, ctx_v: []const Ref, committed: []const u32, ring_capacity: []const u32, from: []const u32, local: bool) void {
    const n = ctx_k.len;
    std.debug.assert(n > 0 and n <= 8 and ctx_v.len == n and committed.len == n and ring_capacity.len == n and from.len == n);
    e.setPipeline(p.attention);
    bind(e, &.{ q, block_k, block_v, out });
    for (0..8) |i| {
        const idx = if (i < n) i else 0;
        e.setBuffer(ctx_k[idx].buf, ctx_k[idx].off, 4 + i);
        e.setBuffer(ctx_v[idx].buf, ctx_v[idx].off, 12 + i);
    }
    var args: AttentionArgs = .{ .local = @intFromBool(local), .streams = @intCast(n), .committed = @splat(0), .ring = @splat(1), .from = @splat(0) };
    for (0..n) |i| {
        std.debug.assert(ring_capacity[i] > 0 and from[i] <= committed[i]);
        args.committed[i] = committed[i];
        args.ring[i] = ring_capacity[i];
        args.from[i] = from[i];
    }
    e.setValue(args, 20);
    e.dispatchGroups(mtl.Size.of(8, n, 1), mtl.Size.of(768, 1, 1));
}

/// Normalization consumes the FP32 residual sum, independently of the rounded BF16 residual store.
pub fn residualNorm(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, residual: Ref, weight: Ref, new_residual: Ref, normalized: Ref, rows: u32, init: bool) void {
    std.debug.assert(rows > 0);
    e.setPipeline(p.residual_norm);
    bind(e, &.{ x, residual, weight, new_residual, normalized });
    e.setValue(ResidualArgs{ .init = @intFromBool(init) }, 5);
    groups(e, rows, 256);
}

/// Gate and up projections are BF16 rows with 9216 entries; only this SwiGLU product is stored here.
pub fn swiglu(p: *const Pipelines, e: mtl.ComputeEncoder, gate: Ref, up: Ref, out: Ref, count: u32) void {
    std.debug.assert(count > 0 and count % (8 * intermediate) == 0);
    e.setPipeline(p.swiglu);
    bind(e, &.{ gate, up, out });
    e.setValue(CountArgs{ .count = count }, 3);
    threads(e, count, 256);
}

/// Kappa is FP32 [rows][streams][768], reading transposed [2*H, 768] weights across streams.
pub fn kappa(p: *const Pipelines, e: mtl.ComputeEncoder, y: Ref, newest_fs: []const Ref, rnn_kappa: Ref, rnn_bias: Ref, out: Ref, rows: u32, stride_s: u32, row_off: u32) void {
    const n = newest_fs.len;
    std.debug.assert(rows > 0 and rows <= 7 and n > 0 and n <= 8 and stride_s >= row_off + rows);
    e.setPipeline(p.kappa);
    e.setBuffer(y.buf, y.off, 0);
    for (0..8) |i| {
        const f = newest_fs[if (i < n) i else 0];
        e.setBuffer(f.buf, f.off, 1 + i);
    }
    e.setBuffer(rnn_kappa.buf, rnn_kappa.off, 9);
    e.setBuffer(rnn_bias.buf, rnn_bias.off, 10);
    e.setBuffer(out.buf, out.off, 11);
    e.setValue(KappaArgs{ .rows = rows, .stride_s = stride_s, .row_off = row_off, .streams = @intCast(n) }, 12);
    e.dispatchGroups(mtl.Size.of(12, n, 1), mtl.Size.of(64, 1, 1));
}

/// One recurrent step writes rounded Z, the next BF16 state and BF16 lambda across batched streams.
pub fn recurrentStep(p: *const Pipelines, e: mtl.ComputeEncoder, state: Ref, kappa_step: Ref, rnn_rec: Ref, markov_w1: Ref, z: Ref, next_state: Ref, lambda: Ref, held: Ref, streams: u32, token: u32, step: u32, use_held: bool) void {
    std.debug.assert(streams > 0 and streams <= 8);
    e.setPipeline(p.recurrent);
    bind(e, &.{ state, kappa_step, rnn_rec, markov_w1, z, next_state, lambda, held });
    e.setValue(TokenArgs{ .token = token, .step = step, .use_held = @intFromBool(use_held) }, 8);
    groups(e, streams, 256);
}

/// Raw U*y rows round once to BF16: 8x8 simdgroup matrix tiles at rows >= 4, vectorized simd_sum below.
pub fn base(p: *const Pipelines, e: mtl.ComputeEncoder, unembed: Ref, y: Ref, out: Ref, rows: u32, vocabulary_count: u32) void {
    std.debug.assert(rows > 0 and vocabulary_count > 0 and vocabulary_count <= vocab);
    e.setPipeline(p.base);
    bind(e, &.{ unembed, y, out });
    e.setValue(BaseArgs{ .rows = rows, .vocab = vocabulary_count }, 3);
    if (rows < 4) {
        groups(e, (vocabulary_count + 7) / 8, 256);
    } else {
        e.dispatchGroups(mtl.Size.of((vocabulary_count + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
    }
}

/// Mode 0 computes logits + group stats over transposed Phi; mode 1 reduces stats into held token & softmax prob.
pub fn finalLogits(p: *const Pipelines, e: mtl.ComputeEncoder, base_values: Ref, phi: Ref, beta: Ref, lambda: Ref, out: Ref, stats: Ref, held: Ref, probs: Ref, sample_tok: Ref, args: LogitsArgs) void {
    std.debug.assert(args.streams > 0 and args.streams <= 8 and args.count > 0 and args.count <= vocab);
    e.setPipeline(p.logits);
    bind(e, &.{ base_values, phi, beta, lambda, out, stats, held, probs, sample_tok });
    e.setValue(args, 9);
    if (args.mode == 0) {
        groups(e, (args.count + 255) / 256, 256);
    } else {
        groups(e, args.streams, 256);
    }
}

/// Quantized 4-bit/64 unembedding dot rows, delegating to the shared Inkling 2 draft-head helper.
pub fn logitsQ4(p: *const Pipelines, e: mtl.ComputeEncoder, w: Ref, scales: Ref, biases: Ref, y: Ref, out: Ref, rows: u32, vocabulary_count: u32, k: u32, softcap: bool) void {
    std.debug.assert(rows > 0 and vocabulary_count > 0 and vocabulary_count <= vocab and k > 0 and k % q4.group_size == 0);
    q4.dispatch(p.logits_q4, e, w, scales, biases, y, out, rows, vocabulary_count, k, softcap);
}

/// Dispatch the quantized draft head when enabled, otherwise run BF16 base projection.
pub fn baseOrQ4(p: *const Pipelines, e: mtl.ComputeEncoder, head: ?*const q4.DraftHeadQ4, unembed: Ref, x: Ref, out: Ref, rows: u32, vocab_n: u32) void {
    if (head) |h| {
        q4.tf_ink2_draft_head_q4_logits(p.logits_q4, e, h, x, out, rows, false);
    } else {
        base(p, e, unembed, x, out, rows, vocab_n);
    }
}
