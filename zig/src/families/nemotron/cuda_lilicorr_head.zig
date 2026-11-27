//! The LiLiCorr correlator head's reference math (vLLM 0.34.0 lilicorr.py): f32 with bf16 round points, the oracle.
const std = @import("std");
const core = @import("core");
const nvfp4 = @import("cuda_lilicorr_nvfp4.zig");
const cuda = @import("cuda");
const DeviceBuffer = cuda.DeviceBuffer;

pub const slots = 7; // block 8 - 1
pub const topk = 8;
pub const hd = 1024; // correlator hidden
pub const model_hidden = 4096; // the target's (and drafter body's) width
pub const heads = 8;
pub const head_dim = hd / heads; // 128
pub const lattice = slots * topk; // 56 rows
pub const factor = 1024;
pub const n_layers = 2;
pub const rms_eps = 1e-6;
pub const ln_eps = 1e-5;
pub const vector_eps = 1e-4;
pub const logit_scale = 8.0;

/// One lattice layer: attention (a prebuilt per-head bias, no KV state) and an mlp, pre-norm residual.
pub const Layer = struct {
    in_proj: []f32, // [3 * hd, hd]
    in_bias: []f32, // [3 * hd]
    out_proj: []f32, // [hd, hd]
    out_bias: []f32, // [hd]
    attn_norm: []f32, // [hd]
    up: []f32, // [2 * hd, hd]
    up_bias: []f32, // [2 * hd]
    down: []f32, // [hd, 2 * hd]
    down_bias: []f32, // [hd]
    mlp_norm: []f32, // [hd]
};

/// A linear's weight [out, in] and its bias (empty: none — every head linear carries one here).
pub const Linear = struct { w: []f32, b: []f32 = &.{} };

/// The head's weights as the math reads them: f32 widened from bf16, dequantized from nvfp4; the arena owns.
pub const Weights = struct {
    token_proj: Linear, // [hd, model_hidden] — the candidate embeddings' projection
    pass_proj: Linear, // [hd, model_hidden] — the pass hidden's projection
    ctx_proj: Linear, // [hd, model_hidden] — the anchor hidden's projection
    feature_norm_w: []f32, // [5]
    feature_norm_b: []f32, // [5]
    feature_up: Linear, // [hd, 5]
    feature_down: Linear, // [hd, hd]
    slot_emb: []f32, // [slots * hd]
    rank_emb: []f32, // [topk * hd]
    out_norm: []f32, // [hd]
    anchor_norm: []f32, // [hd]
    factor: Linear, // [hd, 3 * hd] — the 3-way split is by COLUMNS (see factorSplits)
    rel_bias: []f32, // [heads, 2 * slots + 1]
    same_bias: []f32, // [heads]
    out_head: Linear, // [factor, hd]
    in_head: Linear, // [factor, hd]
    anchor_out_head: Linear, // [factor, hd]
    layers: []Layer, // n_layers entries

    /// The factor projection's 3-way (self/anchor/cross) column split, each [hd, hd] slice materialized contiguous.
    pub fn factorSplits(w: *const Weights, gpa: std.mem.Allocator) ![][3][]f32 {
        const out = try gpa.alloc([3][]f32, 1);
        errdefer gpa.free(out);
        for (0..3) |p| out[0][p] = try gpa.alloc(f32, hd * hd);
        for (0..hd) |r| for (0..hd) |c| for (0..3) |p| {
            out[0][p][r * hd + c] = w.factor.w[r * 3 * hd + p * hd + c];
        };
        return out;
    }
};

/// The staging (DeviceLinear, DeviceLayer, DeviceWeights, loadDevice) lives in cuda_lilicorr_device.zig.
const device = @import("cuda_lilicorr_device.zig");

pub const DeviceLinear = device.DeviceLinear;
pub const DeviceLayer = device.DeviceLayer;
pub const DeviceWeights = device.DeviceWeights;
pub const loadDevice = device.loadDevice;

/// bf16 bytes to f32: the 16 bits are f32's top half (a u16 f16 bitcast reads the wrong exponent).
pub fn widenBits(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn widen(gpa: std.mem.Allocator, t: core.checkpoint.Tensor) ![]f32 {
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const n = t.bytes.len / 2;
    const out = try gpa.alloc(f32, n);
    for (0..n) |i| out[i] = widenBits(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little));
    return out;
}

/// One nvfp4 projection as f32: the packed u8 rows, their e4m3 block scales and the f32 scale_2.
pub fn dequant(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, out_rows: usize, in: usize) ![]f32 {
    var buf: [192]u8 = undefined;
    const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
    if (w.dtype != .u8 or w.rank != 2 or w.dim(0) != out_rows or w.dim(1) != in / 2) return error.UnexpectedTensor;
    const scales = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{name}), .f8_e4m3, &.{ out_rows, in / 16 });
    const t2 = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{name}));
    if (t2.dtype != .f32 or t2.bytes.len != 4) return error.UnexpectedTensor;
    const scale_2: f32 = @bitCast(std.mem.readInt(u32, t2.bytes[0..4], .little));
    const dst = try gpa.alloc(f32, out_rows * in);
    try nvfp4.dequantNvfp4(w.bytes, out_rows, in, scales.bytes, scale_2, dst);
    // the reference stores every dequantized weight in bf16 (gen_fixture w(): deq.to(bfloat16);

    // the goldens are bf16-as-f32) — the load path must mirror that cast or the whole head sits

    // one rounding off the fixture (the S2 A/B caught exactly this: host-vs-gold 5.7e-1 raw).
    for (dst) |*v| v.* = nvfp4.bf16(v.*);
    return dst;
}

/// A bf16 tensor's f32 widening through the checkpoint (the name is the header's, lilicorr. included).
fn bf16Of(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, shape: []const usize) ![]f32 {
    const t = try ck.expect(name, .bf16, shape);
    return widen(gpa, t);
}

/// The head's weights out of a checkpoint (every name under lilicorr.); the caller keeps the checkpoint open.
pub fn load(gpa: std.mem.Allocator, ck: *core.Checkpoint) !Weights {
    var w: Weights = undefined;
    w.token_proj = .{ .w = try dequant(gpa, ck, "lilicorr.token_proj", hd, model_hidden), .b = try bf16Of(gpa, ck, "lilicorr.token_proj.bias", &.{hd}) };
    w.pass_proj = .{ .w = try dequant(gpa, ck, "lilicorr.pass_hidden_proj", hd, model_hidden), .b = try bf16Of(gpa, ck, "lilicorr.pass_hidden_proj.bias", &.{hd}) };
    w.ctx_proj = .{ .w = try dequant(gpa, ck, "lilicorr.context_proj", hd, model_hidden), .b = try bf16Of(gpa, ck, "lilicorr.context_proj.bias", &.{hd}) };
    w.feature_norm_w = try bf16Of(gpa, ck, "lilicorr.feature_norm.weight", &.{5});
    w.feature_norm_b = try bf16Of(gpa, ck, "lilicorr.feature_norm.bias", &.{5});
    w.feature_up = .{ .w = try bf16Of(gpa, ck, "lilicorr.feature_mlp.up_proj.weight", &.{ hd, 5 }), .b = try bf16Of(gpa, ck, "lilicorr.feature_mlp.up_proj.bias", &.{hd}) };
    w.feature_down = .{ .w = try bf16Of(gpa, ck, "lilicorr.feature_mlp.down_proj.weight", &.{ hd, hd }), .b = try bf16Of(gpa, ck, "lilicorr.feature_mlp.down_proj.bias", &.{hd}) };
    w.slot_emb = try bf16Of(gpa, ck, "lilicorr.slot_embedding", &.{ 1, 1, slots, 1, hd });
    w.rank_emb = try bf16Of(gpa, ck, "lilicorr.rank_embedding", &.{ 1, 1, 1, topk, hd });
    w.out_norm = try bf16Of(gpa, ck, "lilicorr.output_norm.weight", &.{hd});
    w.anchor_norm = try bf16Of(gpa, ck, "lilicorr.anchor_norm.weight", &.{hd});
    w.factor = .{ .w = try bf16Of(gpa, ck, "lilicorr.factor_input_proj.weight", &.{ hd, 3 * hd }), .b = try bf16Of(gpa, ck, "lilicorr.factor_input_proj.bias", &.{hd}) };
    w.rel_bias = try bf16Of(gpa, ck, "lilicorr.relative_slot_bias", &.{ heads, 2 * slots + 1 });
    w.same_bias = try bf16Of(gpa, ck, "lilicorr.same_slot_bias", &.{heads});
    w.out_head = .{ .w = try bf16Of(gpa, ck, "lilicorr.out_head.weight", &.{ factor, hd }), .b = try bf16Of(gpa, ck, "lilicorr.out_head.bias", &.{factor}) };
    w.in_head = .{ .w = try bf16Of(gpa, ck, "lilicorr.in_head.weight", &.{ factor, hd }), .b = try bf16Of(gpa, ck, "lilicorr.in_head.bias", &.{factor}) };
    w.anchor_out_head = .{ .w = try bf16Of(gpa, ck, "lilicorr.anchor_out_head.weight", &.{ factor, hd }), .b = try bf16Of(gpa, ck, "lilicorr.anchor_out_head.bias", &.{factor}) };
    w.layers = try gpa.alloc(Layer, n_layers);
    for (w.layers, 0..) |*l, i| {
        var buf: [96]u8 = undefined;
        const pre = try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.", .{i});
        var n2: [128]u8 = undefined;
        l.in_proj = try bf16Of(gpa, ck, try std.fmt.bufPrint(&n2, "{s}attn.in_proj_weight", .{pre}), &.{ 3 * hd, hd });
        l.in_bias = try bf16Of(gpa, ck, try std.fmt.bufPrint(&n2, "{s}attn.in_proj_bias", .{pre}), &.{3 * hd});
        l.out_proj = try dequant(gpa, ck, try std.fmt.bufPrint(&n2, "{s}attn.out_proj", .{pre}), hd, hd);
        l.out_bias = try bf16Of(gpa, ck, try std.fmt.bufPrint(&n2, "{s}attn.out_proj.bias", .{pre}), &.{hd});
        l.attn_norm = try bf16Of(gpa, ck, try std.fmt.bufPrint(&n2, "{s}attn_norm.weight", .{pre}), &.{hd});
        l.up = try dequant(gpa, ck, try std.fmt.bufPrint(&n2, "{s}mlp.up_proj", .{pre}), 2 * hd, hd);
        l.up_bias = try bf16Of(gpa, ck, try std.fmt.bufPrint(&n2, "{s}mlp.up_proj.bias", .{pre}), &.{2 * hd});
        l.down = try dequant(gpa, ck, try std.fmt.bufPrint(&n2, "{s}mlp.down_proj", .{pre}), hd, 2 * hd);
        l.down_bias = try bf16Of(gpa, ck, try std.fmt.bufPrint(&n2, "{s}mlp.down_proj.bias", .{pre}), &.{hd});
        l.mlp_norm = try bf16Of(gpa, ck, try std.fmt.bufPrint(&n2, "{s}mlp_norm.weight", .{pre}), &.{hd});
    }
    return w;
}

/// x (rows, in) @ w.T + b, f32 accumulation, row-major w [out, in] — torch F.linear.
fn linear(x: []const f32, rows: usize, in: usize, l: Linear, dst: []f32) void {
    const out_dim = l.w.len / in;
    for (0..rows) |r| {
        for (0..out_dim) |o| {
            var acc: f32 = if (l.b.len != 0) l.b[o] else 0;
            const xr = x[r * in ..][0..in];
            const wr = l.w[o * in ..][0..in];
            for (0..in) |k| acc += xr[k] * wr[k];
            dst[r * out_dim + o] = acc;
        }
    }
}

/// torch F.rms_norm(x, (d,), w, eps) then the bf16 store the reference's hidden dtype makes.
fn rmsNorm(x: []const f32, rows: usize, d: usize, w: []const f32, dst: []f32) void {
    rmsNormTo(x, rows, d, w, dst, true);
}

/// The same norm kept f32 (the reference's anchor state stays in its f32 dtype).
fn rmsNormF32(x: []const f32, rows: usize, d: usize, w: []const f32, dst: []f32) void {
    rmsNormTo(x, rows, d, w, dst, false);
}

fn rmsNormTo(x: []const f32, rows: usize, d: usize, w: []const f32, dst: []f32, out_bf16: bool) void {
    for (0..rows) |r| {
        var ssq: f32 = 0;
        for (0..d) |k| ssq += x[r * d + k] * x[r * d + k];
        const inv = 1.0 / @sqrt(ssq / @as(f32, @floatFromInt(d)) + rms_eps);
        for (0..d) |k| {
            const v = x[r * d + k] * inv * w[k];
            dst[r * d + k] = if (out_bf16) nvfp4.bf16(v) else v;
        }
    }
}

/// torch F.layer_norm over the last dim with weight and bias, eps 1e-5 (the five features').
fn layerNorm(x: []const f32, rows: usize, d: usize, w: []const f32, b: []const f32, dst: []f32) void {
    for (0..rows) |r| {
        var mean: f32 = 0;
        for (0..d) |k| mean += x[r * d + k];
        mean /= @as(f32, @floatFromInt(d));
        var varr: f32 = 0;
        for (0..d) |k| {
            const dv = x[r * d + k] - mean;
            varr += dv * dv;
        }
        varr /= @as(f32, @floatFromInt(d));
        const inv = 1.0 / @sqrt(varr + ln_eps);
        for (0..d) |k| dst[r * d + k] = (x[r * d + k] - mean) * inv * w[k] + b[k];
    }
}

/// SDPA per head over the whole lattice: softmax(QK^T/sqrt(128) + prebuilt bias[h]) @ V, f32.
pub fn latticeAttention(qkv: []const f32, bias: []const f32, dst: []f32) void {
    const scale = 1.0 / @sqrt(@as(f32, head_dim));
    var scores: [lattice]f32 = undefined;
    for (0..heads) |h| {
        const at = h * head_dim;
        for (0..lattice) |qi| {
            var mx: f32 = -std.math.floatMax(f32);
            for (0..lattice) |ki| {
                var acc: f32 = 0;
                for (0..head_dim) |t| acc += qkv[qi * 3 * hd + at + t] * qkv[ki * 3 * hd + hd + at + t];
                const s = acc * scale + bias[(h * lattice + qi) * lattice + ki];
                scores[ki] = s;
                if (s > mx) mx = s;
            }
            var sum: f32 = 0;
            for (0..lattice) |ki| {
                scores[ki] = @exp(scores[ki] - mx);
                sum += scores[ki];
            }
            for (0..head_dim) |t| dst[qi * hd + at + t] = 0;
            for (0..lattice) |ki| {
                const p = scores[ki] / sum;
                for (0..head_dim) |t| dst[qi * hd + at + t] += p * qkv[ki * 3 * hd + 2 * hd + at + t];
            }
        }
    }
}

/// The attention bias [heads, lattice, lattice]: relative slot offsets plus the same-slot boost, per head.
pub fn buildBias(gpa: std.mem.Allocator, w: *const Weights) ![]f32 {
    const bias = try gpa.alloc(f32, heads * lattice * lattice);
    for (0..heads) |h| for (0..lattice) |qi| for (0..lattice) |ki| {
        const qs = qi / topk;
        const ks = ki / topk;
        const d: i32 = @as(i32, @intCast(qs)) - @as(i32, @intCast(ks));
        var v = w.rel_bias[h * (2 * slots + 1) + @as(usize, @intCast(d + @as(i32, slots)))];
        if (qs == ks) v += w.same_bias[h];
        bias[(h * lattice + qi) * lattice + ki] = v;
    };
    return bias;
}

pub const Scores = struct {
    start: []f32, // start [topk]: slot 0's entry scores
    pair: []f32, // pair [(slots - 1) * topk * topk]

    /// Free the two result slices (the gpa score() was given); every intermediate dies with score()'s internal arena.
    pub fn deinit(self: Scores, gpa: std.mem.Allocator) void {
        gpa.free(self.start);
        gpa.free(self.pair);
    }
};

/// The reference score(): the correlator's start and pair scores; intermediates die with the call's internal arena.
pub fn score(gpa: std.mem.Allocator, w: *const Weights, token_emb: []f32, log_probs: []const f32, pass_hidden: []f32, anchor_hidden: []f32, anchor_valid: bool) !Scores {
    var scratch_arena = std.heap.ArenaAllocator.init(gpa);
    defer scratch_arena.deinit();
    const sc = scratch_arena.allocator();

    const start = try gpa.alloc(f32, topk); // the call's two survivors: see Scores.deinit
    errdefer gpa.free(start); // owner A6, per-acquisition: a pair-alloc failure frees start
    const pair = try gpa.alloc(f32, (slots - 1) * topk * topk);
    errdefer gpa.free(pair); // owner A6, per-acquisition: every later failure frees both, once

    const cand_emb = token_emb; // the reference casts to bf16; the caller hands bf16-widened rows
    for (cand_emb) |*v| v.* = nvfp4.bf16(v.*);
    for (pass_hidden) |*v| v.* = nvfp4.bf16(v.*);
    for (anchor_hidden) |*v| v.* = nvfp4.bf16(v.*);

    const token_states = try sc.alloc(f32, lattice * hd);
    linear(cand_emb, lattice, model_hidden, w.token_proj, token_states);
    const pass_states = try sc.alloc(f32, hd);
    linear(pass_hidden, 1, model_hidden, w.pass_proj, pass_states);

    // the five candidate features: logp, exp(logp), logp - max, rank fraction (over topk - 1), is_top1
    const features = try sc.alloc(f32, lattice * 5);
    var mx: [slots]f32 = undefined;
    for (0..slots) |s| {
        var m: f32 = -std.math.floatMax(f32);
        for (0..topk) |k| m = @max(m, log_probs[s * topk + k]);
        mx[s] = m;
    }
    for (0..slots) |s| for (0..topk) |k| {
        const lp = log_probs[s * topk + k];
        features[(s * topk + k) * 5 + 0] = lp;
        features[(s * topk + k) * 5 + 1] = @exp(lp);
        features[(s * topk + k) * 5 + 2] = lp - mx[s];
        features[(s * topk + k) * 5 + 3] = @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(topk - 1));
        features[(s * topk + k) * 5 + 4] = if (k == 0) 1.0 else 0.0;
    };
    layerNorm(features, lattice, 5, w.feature_norm_w, w.feature_norm_b, features);
    for (features) |*v| v.* = nvfp4.bf16(v.*); // the reference feeds the LayerNorm's bf16 store to the mlp

    // the mlp: silu(up) stored bf16 before down (the down output joins h in f32)
    const fm_up = try sc.alloc(f32, lattice * hd);
    linear(features, lattice, 5, w.feature_up, fm_up);
    for (fm_up) |*v| v.* = nvfp4.bf16(silu(v.*));
    const fm = try sc.alloc(f32, lattice * hd);
    linear(fm_up, lattice, hd, w.feature_down, fm);

    // h = token + pass + feature mlp + slot + rank embeddings, stored to the lattice in bf16
    const lattice_h = try sc.alloc(f32, lattice * hd);
    for (0..slots) |s| for (0..topk) |k| for (0..hd) |t| {
        const i = (s * topk + k) * hd + t;
        lattice_h[i] = nvfp4.bf16(token_states[i] + pass_states[t] + fm[i] + w.slot_emb[s * hd + t] + w.rank_emb[k * hd + t]);
    };

    // the anchor: context projection, validity gate, its own rms norm (kept f32)
    const anchor_state = try sc.alloc(f32, hd);
    linear(anchor_hidden, 1, model_hidden, w.ctx_proj, anchor_state);
    if (!anchor_valid) {
        for (anchor_state) |*v| v.* = 0;
    }
    const anchor_n = try sc.alloc(f32, hd);
    rmsNormF32(anchor_state, 1, hd, w.anchor_norm, anchor_n); // the anchor norm stays f32 (the reference's dtype)

    // the two lattice layers
    const bias = try buildBias(sc, w);
    const a_in = try sc.alloc(f32, lattice * hd);
    const qkv = try sc.alloc(f32, lattice * 3 * hd);
    const att = try sc.alloc(f32, lattice * hd);
    const mlp_out = try sc.alloc(f32, lattice * hd);
    for (w.layers) |l| {
        rmsNorm(lattice_h, lattice, hd, l.attn_norm, a_in);
        linear(a_in, lattice, hd, .{ .w = l.in_proj, .b = l.in_bias }, qkv);
        latticeAttention(qkv, bias, att);
        linear(att, lattice, hd, .{ .w = l.out_proj, .b = l.out_bias }, mlp_out);
        for (0..lattice * hd) |i| lattice_h[i] = nvfp4.bf16(lattice_h[i] + nvfp4.bf16(mlp_out[i]));
        rmsNorm(lattice_h, lattice, hd, l.mlp_norm, a_in);
        linear(a_in, lattice, hd, .{ .w = l.up, .b = l.up_bias }, qkv);
        for (0..lattice * 2 * hd) |i| qkv[i] = nvfp4.bf16(silu(qkv[i]));
        linear(qkv[0 .. lattice * 2 * hd], lattice, 2 * hd, .{ .w = l.down, .b = l.down_bias }, mlp_out);
        for (0..lattice * hd) |i| lattice_h[i] = nvfp4.bf16(lattice_h[i] + nvfp4.bf16(mlp_out[i]));
    }

    // output norm (bf16), then the factor: three column splits, silu, bf16 before the edge heads
    rmsNorm(lattice_h, lattice, hd, w.out_norm, a_in);
    const splits = try w.factorSplits(sc);
    const pre = try sc.alloc(f32, lattice * hd);
    linear(a_in, lattice, hd, .{ .w = splits[0][0], .b = w.factor.b }, pre);
    const prod = try sc.alloc(f32, lattice * hd);
    for (0..lattice) |r| for (0..hd) |t| {
        prod[r * hd + t] = a_in[r * hd + t] * anchor_n[t];
    };
    const tmp = try sc.alloc(f32, lattice * hd);
    for (0..lattice * hd) |r| tmp[r] = anchor_n[r % hd]; // anchor row broadcast
    const scratch = try sc.alloc(f32, lattice * hd);
    linear(tmp, lattice, hd, .{ .w = splits[0][1] }, scratch); // the anchor term
    for (0..lattice * hd) |i| pre[i] += scratch[i];
    linear(prod, lattice, hd, .{ .w = splits[0][2] }, scratch); // the cross term (h * anchor, still intact)
    for (0..lattice * hd) |i| pre[i] += scratch[i];
    for (pre) |*v| v.* = nvfp4.bf16(silu(v.*));

    // the fused edge head, the L2 normalization, the dot scores
    const edges = try sc.alloc(f32, lattice * 2 * factor);
    var fused: Linear = undefined;
    const fused_w = try sc.alloc(f32, 2 * factor * hd);
    @memcpy(fused_w[0 .. factor * hd], w.out_head.w);
    @memcpy(fused_w[factor * hd ..], w.in_head.w);
    const fused_b = try sc.alloc(f32, 2 * factor);
    @memcpy(fused_b[0..factor], w.out_head.b);
    @memcpy(fused_b[factor..], w.in_head.b);
    fused = .{ .w = fused_w, .b = fused_b };
    linear(pre, lattice, hd, fused, edges);
    const out_vec = try sc.alloc(f32, lattice * factor);
    const in_vec = try sc.alloc(f32, lattice * factor);
    for (0..lattice) |r| {
        var so: f32 = 0;
        var si: f32 = 0;
        for (0..factor) |t| {
            so += edges[r * 2 * factor + t] * edges[r * 2 * factor + t];
            si += edges[r * 2 * factor + factor + t] * edges[r * 2 * factor + factor + t];
        }
        so = @max(@sqrt(so), vector_eps);
        si = @max(@sqrt(si), vector_eps);
        for (0..factor) |t| {
            out_vec[r * factor + t] = edges[r * 2 * factor + t] / so;
            in_vec[r * factor + t] = edges[r * 2 * factor + factor + t] / si;
        }
    }
    const anchor_out = try sc.alloc(f32, factor);
    linear(anchor_n, 1, hd, w.anchor_out_head, anchor_out);
    var an: f32 = 0;
    for (anchor_out) |v| an += v * v;
    an = @max(@sqrt(an), vector_eps);
    for (anchor_out) |*v| v.* /= an;

    for (0..topk) |j| { // the entry scores into slot 0's candidates (the reference's in_vec[:, 0, :, :])
        var acc: f32 = 0;
        for (0..factor) |t| acc += anchor_out[t] * in_vec[j * factor + t];
        start[j] = acc;
    }
    for (0..slots - 1) |s| for (0..topk) |i| for (0..topk) |j| {
        var acc: f32 = 0;
        for (0..factor) |t| acc += out_vec[(s * topk + i) * factor + t] * in_vec[((s + 1) * topk + j) * factor + t];
        pair[(s * topk + i) * topk + j] = acc;
    };
    return .{ .start = start, .pair = pair }; // the arena dies here; only these two survive on gpa
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

test "the factor split reads the right columns: each third is its own contiguous matrix" {
    const a = std.testing.allocator;
    var w: Weights = undefined;
    w.factor = .{ .w = try a.alloc(f32, hd * 3 * hd), .b = &.{} };
    defer a.free(w.factor.w);
    for (w.factor.w, 0..) |*v, i| v.* = @floatFromInt(i % 7); // distinct constants by position
    const splits = try w.factorSplits(a);
    // element [r, c] of split p is factor.w[r * 3hd + p * hd + c]
    try std.testing.expectEqual(w.factor.w[5 * 3 * hd + 2 * hd + 9], splits[0][2][5 * hd + 9]);
    try std.testing.expectEqual(w.factor.w[5 * 3 * hd + 0 * hd + 9], splits[0][0][5 * hd + 9]);
    for (splits[0]) |s| a.free(s);
    a.free(splits);
}

test "score holds up under repeated holds: intermediates die with the call, the results free via deinit" {
    const a = std.testing.allocator;

    // The weights ride a throwaway arena: the leak gate targets score()'s per-call gpa use,

    // so the fixture weights must not pollute this test's verdict. Everything score() itself

    // takes from the caller's gpa (the two result slices) frees via Scores.deinit below —

    // std.testing.allocator fails this test on any per-round leak or missed deinit.
    var w_arena = std.heap.ArenaAllocator.init(a);
    defer w_arena.deinit();
    const wg = w_arena.allocator();

    var w: Weights = undefined;
    w.token_proj = .{ .w = try wg.alloc(f32, hd * model_hidden), .b = try wg.alloc(f32, hd) };
    w.pass_proj = .{ .w = try wg.alloc(f32, hd * model_hidden), .b = try wg.alloc(f32, hd) };
    w.ctx_proj = .{ .w = try wg.alloc(f32, hd * model_hidden), .b = try wg.alloc(f32, hd) };
    w.feature_norm_w = try wg.alloc(f32, 5);
    w.feature_norm_b = try wg.alloc(f32, 5);
    w.feature_up = .{ .w = try wg.alloc(f32, hd * 5), .b = try wg.alloc(f32, hd) };
    w.feature_down = .{ .w = try wg.alloc(f32, hd * hd), .b = try wg.alloc(f32, hd) };
    w.slot_emb = try wg.alloc(f32, slots * hd);
    w.rank_emb = try wg.alloc(f32, topk * hd);
    w.out_norm = try wg.alloc(f32, hd);
    w.anchor_norm = try wg.alloc(f32, hd);
    w.factor = .{ .w = try wg.alloc(f32, hd * 3 * hd), .b = try wg.alloc(f32, hd) };
    w.rel_bias = try wg.alloc(f32, heads * (2 * slots + 1));
    w.same_bias = try wg.alloc(f32, heads);
    w.out_head = .{ .w = try wg.alloc(f32, factor * hd), .b = try wg.alloc(f32, factor) };
    w.in_head = .{ .w = try wg.alloc(f32, factor * hd), .b = try wg.alloc(f32, factor) };
    w.anchor_out_head = .{ .w = try wg.alloc(f32, factor * hd), .b = try wg.alloc(f32, factor) };
    const ls = try wg.alloc(Layer, n_layers);
    for (ls) |*l| {
        l.* = .{
            .in_proj = try wg.alloc(f32, 3 * hd * hd),
            .in_bias = try wg.alloc(f32, 3 * hd),
            .out_proj = try wg.alloc(f32, hd * hd),
            .out_bias = try wg.alloc(f32, hd),
            .attn_norm = try wg.alloc(f32, hd),
            .up = try wg.alloc(f32, 2 * hd * hd),
            .up_bias = try wg.alloc(f32, 2 * hd),
            .down = try wg.alloc(f32, hd * 2 * hd),
            .down_bias = try wg.alloc(f32, hd),
            .mlp_norm = try wg.alloc(f32, hd),
        };
    }
    w.layers = ls;

    const token_emb = try a.alloc(f32, slots * topk * model_hidden);
    defer a.free(token_emb);
    @memset(token_emb, 0.125);
    const log_probs = try a.alloc(f32, slots * topk);
    defer a.free(log_probs);
    @memset(log_probs, -1.0);
    const pass_hidden = try a.alloc(f32, model_hidden);
    defer a.free(pass_hidden);
    @memset(pass_hidden, 0.25);
    const anchor_hidden = try a.alloc(f32, model_hidden);
    defer a.free(anchor_hidden);
    @memset(anchor_hidden, 0.25);

    for (0..3) |_| {
        const got = try score(a, &w, token_emb, log_probs, pass_hidden, anchor_hidden, true);
        for (got.start) |v| try std.testing.expect(std.math.isFinite(v));
        for (got.pair) |v| try std.testing.expect(std.math.isFinite(v));
        got.deinit(a);
    }
}
