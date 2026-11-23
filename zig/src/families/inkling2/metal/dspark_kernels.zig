//! Mini DSpark's fixed-shape arithmetic: Metal entry points and exact launch ABIs.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");

pub const hidden: u32 = 2304;
pub const query_width: u32 = 3072;
pub const kv_width: u32 = 1024;
pub const block_rows: u32 = 8;
pub const rank: u32 = 256;
pub const rnn_width: u32 = 768;
pub const intermediate: u32 = 9216;
pub const vocab: u32 = 200058;
pub const Ref = @import("inkling2_metal_ref").Ref;
pub const PositionArgs = extern struct { base: u32 };
pub const ContextArgs = extern struct { base: u32, rows: u32, ring: u32 };
pub const AttentionArgs = extern struct { committed: u32, ring: u32, local: u32, from: u32 };
pub const TokenArgs = extern struct { token: u32 };
pub const CountArgs = extern struct { count: u32 };
pub const BaseArgs = extern struct { rows: u32, vocab: u32 };
pub const PackArgs = extern struct { rows: u32, ngroups: u32 };
pub const Q4UnitsArgs = extern struct { k: u32, n0: u32, n1: u32, n2: u32 };
pub const SampleArgs = extern struct { step: u32, position: u32, seed_lo: u32, seed_hi: u32, top_k: u32, inv_t: f32, top_p: f32, min_log: f32 };

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
pub const ResidualArgs = extern struct { init: u32 };

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
        };
    }
    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            inline else => |v| std.fmt.comptimePrint("metal inkling2 dspark {s} {s}", .{ @tagName(v), comptime v.symbol() }),
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

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.inkling2_dspark);
    }
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        var loaded: usize = 0;
        errdefer inline for (@typeInfo(Pipelines).@"struct".field_names, 0..) |name, i| {
            if (loaded > i) @field(out, name).deinit();
        };
        inline for (@typeInfo(Pipelines).@"struct".field_names, 0..) |name, i| {
            @field(out, name) = try mtl.Pipeline.init(device, lib, @as(Kind, @fromBackingInt(@intCast(i))).symbol(), false);
            loaded += 1;
        }
        return out;
    }
    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| @field(p, name).deinit();
        p.* = undefined;
    }
};

/// The greedy chain's two kernels, outside `Pipelines`: its entries are the block's `Kind`s one for one.
pub const Chain = struct {
    step: mtl.Pipeline,
    pick: mtl.Pipeline,

    pub fn load(device: mtl.Device, text: []const u8) !Chain {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const step = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_chain_step", false);
        errdefer step.deinit();
        return .{ .step = step, .pick = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_chain_pick", false) };
    }
    pub fn deinit(c: *Chain) void {
        c.pick.deinit();
        c.step.deinit();
        c.* = undefined;
    }
};

/// 2-bit screened greedy head kernels (outside `Pipelines` so `Kind` contracts remain unchanged).
pub const Screen = struct {
    pack: mtl.Pipeline,
    q4_pack: mtl.Pipeline,
    proj: mtl.Pipeline,
    proj_units: ?mtl.Pipeline = null,
    q4_units: ?mtl.Pipeline = null,
    screen: mtl.Pipeline,
    refine: mtl.Pipeline,
    pick: mtl.Pipeline,
    sample: mtl.Pipeline,

    pub fn load(device: mtl.Device, text: []const u8) !Screen {
        return loadSources(device, text, sources.experts_units);
    }
    pub fn loadSources(device: mtl.Device, text: []const u8, units_text: []const u8) !Screen {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pack = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_q2_pack", false);
        errdefer pack.deinit();
        const q4_pack = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_q4_pack", false);
        errdefer q4_pack.deinit();
        const proj = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_q2_proj", false);
        errdefer proj.deinit();
        const screen = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_q2_screen", false);
        errdefer screen.deinit();
        const refine = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_q2_refine", false);
        errdefer refine.deinit();
        const pick = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_q2_pick", false);
        errdefer pick.deinit();
        const sample = try mtl.Pipeline.init(device, lib, "tf_ink2_dspark_q2_sample", false);
        errdefer sample.deinit();
        var proj_units: ?mtl.Pipeline = null;
        var q4_units: ?mtl.Pipeline = null;
        if (device.tensorUnits()) {
            const ulib = try mtl.Library.fromSource(device, units_text, mtl.CompileOptions.mlx());
            defer ulib.deinit();
            proj_units = try mtl.Pipeline.init(device, ulib, "tf_ink2_dspark_q2_proj_units", false);
            errdefer if (proj_units) |p| p.deinit();
            q4_units = try mtl.Pipeline.init(device, ulib, "tf_ink2_dspark_q4_units", false);
        }
        return .{ .pack = pack, .q4_pack = q4_pack, .proj = proj, .proj_units = proj_units, .q4_units = q4_units, .screen = screen, .refine = refine, .pick = pick, .sample = sample };
    }
    pub fn deinit(s: *Screen) void {
        if (s.q4_units) |p| p.deinit();
        if (s.proj_units) |p| p.deinit();
        s.sample.deinit();
        s.pick.deinit();
        s.refine.deinit();
        s.screen.deinit();
        s.proj.deinit();
        s.q4_pack.deinit();
        s.pack.deinit();
        s.* = undefined;
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

/// Row zero copies the target's normalized anchor; the other seven rows copy the raw mask embedding.
pub fn initializeRows(p: *const Pipelines, e: mtl.ComputeEncoder, anchor: Ref, mask_embed: Ref, out: Ref) void {
    e.setPipeline(p.init_rows);
    bind(e, &.{ anchor, mask_embed, out });
    groups(e, block_rows, 256);
}

/// Input taps are [row][tap][H], output c is [fusion][row][H], and newest_f is one [H] row.
pub fn fuseTaps(p: *const Pipelines, e: mtl.ComputeEncoder, taps: Ref, fuse_weight: Ref, norm_weight: Ref, c: Ref, newest_f: Ref, rows: u32) void {
    std.debug.assert(rows > 0);
    e.setPipeline(p.fuse_taps);
    bind(e, &.{ taps, fuse_weight, norm_weight, c, newest_f });
    e.setValue(CountArgs{ .count = rows }, 5);
    groups(e, 4 * @as(usize, rows), 256);
}

/// Raw projections become per-head RMS-normalized rotate-half Q/K and an unchanged BF16 V.
pub fn blockQkv(p: *const Pipelines, e: mtl.ComputeEncoder, q_raw: Ref, k_raw: Ref, v_raw: Ref, q_norm: Ref, k_norm: Ref, q: Ref, k: Ref, v: Ref, base_position: u32) void {
    e.setPipeline(p.block_qkv);
    bind(e, &.{ q_raw, k_raw, v_raw, q_norm, k_norm, q, k, v });
    e.setValue(PositionArgs{ .base = base_position }, 8);
    groups(e, 8 * 40, 128);
}

/// Context K/V is written at absolute-position modulo ring_capacity for committed rows only.
pub fn commitContext(p: *const Pipelines, e: mtl.ComputeEncoder, k_raw: Ref, v_raw: Ref, k_norm_ctx: Ref, keys: Ref, values: Ref, base_position: u32, rows: u32, ring_capacity: u32) void {
    std.debug.assert(rows > 0 and ring_capacity > 0);
    e.setPipeline(p.context_kv);
    bind(e, &.{ k_raw, v_raw, k_norm_ctx, keys, values });
    e.setValue(ContextArgs{ .base = base_position, .rows = rows, .ring = ring_capacity }, 5);
    groups(e, @as(usize, rows) * 8, 128);
}

/// Scores is FP32 scratch sized 8*24*(retained_context+8); global layers retain all context.
pub fn attend(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, block_k: Ref, block_v: Ref, context_k: Ref, context_v: Ref, scores: Ref, out: Ref, committed: u32, ring_capacity: u32, local: bool, from: u32) void {
    std.debug.assert(ring_capacity > 0 and from <= committed);
    e.setPipeline(p.attention);
    bind(e, &.{ q, block_k, block_v, context_k, context_v, scores, out });
    e.setValue(AttentionArgs{ .committed = committed, .ring = ring_capacity, .local = @intFromBool(local), .from = from }, 7);
    groups(e, 8 * 24, 128);
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
    std.debug.assert(count == 8 * intermediate);
    e.setPipeline(p.swiglu);
    bind(e, &.{ gate, up, out });
    e.setValue(CountArgs{ .count = count }, 3);
    threads(e, count, 256);
}

/// Kappa is FP32 [rows][768], using RNN columns 512..2815 for y and 2816..5119 for newest_f.
pub fn kappa(p: *const Pipelines, e: mtl.ComputeEncoder, y: Ref, newest_f: Ref, rnn_weight: Ref, rnn_bias: Ref, out: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= 7);
    e.setPipeline(p.kappa);
    bind(e, &.{ y, newest_f, rnn_weight, rnn_bias, out });
    e.setValue(CountArgs{ .count = rows }, 5);
    threads(e, @as(usize, rows) * rnn_width, 256);
}

/// One recurrent step writes rounded Z, the next BF16 state and BF16 lambda for rows 1..7 in turn.
pub fn recurrentStep(p: *const Pipelines, e: mtl.ComputeEncoder, state: Ref, kappa_row: Ref, rnn_weight: Ref, markov_w1: Ref, z: Ref, next_state: Ref, lambda: Ref, previous_token: u32) void {
    e.setPipeline(p.recurrent);
    bind(e, &.{ state, kappa_row, rnn_weight, markov_w1, z, next_state, lambda });
    e.setValue(TokenArgs{ .token = previous_token }, 7);
    threads(e, rank, 256);
}

/// recurrentStep on the token `held[step]` holds: the chain's previous draft, read on the GPU.
pub fn chainStep(c: *const Chain, e: mtl.ComputeEncoder, state: Ref, kappa_row: Ref, rnn_weight: Ref, markov_w1: Ref, z: Ref, next_state: Ref, lambda: Ref, held: Ref, step: u32) void {
    e.setPipeline(c.step);
    bind(e, &.{ state, kappa_row, rnn_weight, markov_w1, z, next_state, lambda });
    e.setValue(TokenArgs{ .token = step }, 7);
    e.setBuffer(held.buf, held.off, 8);
    threads(e, rank, 256);
}

/// held[step + 1] = the first greatest of a row of final logits: one threadgroup of 1,024.
pub fn chainPick(c: *const Chain, e: mtl.ComputeEncoder, logits: Ref, held: Ref, step: u32) void {
    e.setPipeline(c.pick);
    bind(e, &.{ logits, held });
    e.setValue(TokenArgs{ .token = step }, 2);
    groups(e, 1, 1024);
}

/// Raw U*y rows round once to BF16, with no target unembedding softcap.
pub fn base(p: *const Pipelines, e: mtl.ComputeEncoder, unembed: Ref, y: Ref, out: Ref, rows: u32, vocabulary_count: u32) void {
    std.debug.assert(rows > 0 and vocabulary_count > 0 and vocabulary_count <= vocab);
    e.setPipeline(p.base);
    bind(e, &.{ unembed, y, out });
    e.setValue(BaseArgs{ .rows = rows, .vocab = vocabulary_count }, 3);
    std.debug.assert(rows <= 7);
    groups(e, (@as(usize, vocabulary_count) + 7) / 8, 256);
}
/// Folded Phi/beta correction and target-style 30 tanh softcap, restricted to Mini's 200058 logits.
pub fn finalLogits(p: *const Pipelines, e: mtl.ComputeEncoder, base_values: Ref, phi: Ref, beta: Ref, lambda: Ref, out: Ref) void {
    e.setPipeline(p.logits);
    bind(e, &.{ base_values, phi, beta, lambda, out });
    e.setValue(CountArgs{ .count = vocab }, 5);
    groups(e, (vocab + 7) / 8, 256);
}

pub fn q2Pack(s: *const Screen, e: mtl.ComputeEncoder, mat: Ref, codes: Ref, params: Ref, rows: u32, ngroups: u32) void {
    e.setPipeline(s.pack);
    bind(e, &.{ mat, codes, params });
    e.setValue(PackArgs{ .rows = rows, .ngroups = ngroups }, 3);
    groups(e, (@as(usize, rows) * ngroups + 7) / 8, 256);
}

pub fn q4Pack(s: *const Screen, e: mtl.ComputeEncoder, mat: Ref, codes: Ref, params: Ref, rows: u32, ngroups: u32) void {
    e.setPipeline(s.q4_pack);
    bind(e, &.{ mat, codes, params });
    e.setValue(PackArgs{ .rows = rows, .ngroups = ngroups }, 3);
    groups(e, (@as(usize, rows) * ngroups + 7) / 8, 256);
}

pub fn q4Units(s: *const Screen, e: mtl.ComputeEncoder, codes: Ref, params: Ref, x: Ref, y0: Ref, y1: Ref, y2: Ref, k_dim: u32, n0: u32, n1: u32, n2: u32) void {
    e.setPipeline(s.q4_units.?);
    bind(e, &.{ codes, params, x, y0, y1, y2 });
    e.setValue(Q4UnitsArgs{ .k = k_dim, .n0 = n0, .n1 = n1, .n2 = n2 }, 6);
    groups(e, (@as(usize, n0) + n1 + n2) / 32, 256);
}

pub fn q2Proj(s: *const Screen, e: mtl.ComputeEncoder, u_codes: Ref, u_params: Ref, y: Ref, out: Ref, rows: u32, vocabulary_count: u32) void {
    std.debug.assert(rows > 0 and rows <= 7 and vocabulary_count <= vocab);
    if (s.proj_units) |pu| {
        e.setPipeline(pu);
        bind(e, &.{ u_codes, u_params, y, out });
        e.setValue(BaseArgs{ .rows = rows, .vocab = vocabulary_count }, 4);
        groups(e, (@as(usize, vocabulary_count) + 31) / 32, 256);
        return;
    }
    e.setPipeline(s.proj);
    bind(e, &.{ u_codes, u_params, y, out });
    e.setValue(BaseArgs{ .rows = rows, .vocab = vocabulary_count }, 4);
    groups(e, (@as(usize, vocabulary_count) + 7) / 8, 256);
}

pub fn q2Screen(s: *const Screen, e: mtl.ComputeEncoder, base_approx: Ref, p_codes: Ref, p_params: Ref, beta: Ref, lambda: Ref, candidates: Ref) void {
    q2ScreenCount(s, e, base_approx, p_codes, p_params, beta, lambda, candidates, vocab);
}

pub fn q2ScreenCount(s: *const Screen, e: mtl.ComputeEncoder, base_approx: Ref, p_codes: Ref, p_params: Ref, beta: Ref, lambda: Ref, candidates: Ref, count: u32) void {
    e.setPipeline(s.screen);
    bind(e, &.{ base_approx, p_codes, p_params, beta, lambda, candidates });
    e.setValue(CountArgs{ .count = count }, 6);
    groups(e, 128, 256);
}

pub fn q2Refine(s: *const Screen, e: mtl.ComputeEncoder, candidates: Ref, unembed: Ref, y_row: Ref, phi: Ref, beta: Ref, lambda: Ref, cand_logits: Ref, cand_ids: Ref) void {
    e.setPipeline(s.refine);
    bind(e, &.{ candidates, unembed, y_row, phi, beta, lambda, cand_logits, cand_ids });
    groups(e, 33, 256);
}

pub fn q2Pick(s: *const Screen, e: mtl.ComputeEncoder, cand_logits: Ref, cand_ids: Ref, held: Ref, step: u32) void {
    e.setPipeline(s.pick);
    bind(e, &.{ cand_logits, cand_ids, held });
    e.setValue(TokenArgs{ .token = step }, 3);
    groups(e, 1, 256);
}

pub fn q2Sample(s: *const Screen, e: mtl.ComputeEncoder, cand_logits: Ref, cand_ids: Ref, held: Ref, args: SampleArgs) void {
    e.setPipeline(s.sample);
    bind(e, &.{ cand_logits, cand_ids, held });
    e.setValue(args, 3);
    groups(e, 1, 256);
}
