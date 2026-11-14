//! The decoder groups' shared parts: decoder.metal with a control's edit, value ranges, gate 5's texts.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;
pub const Texts = uocr.Texts;

pub const hidden = uocr.hidden;
pub const heads = uocr.heads;
pub const head_dim = uocr.head_dim;
pub const router_dim = uocr.router_dim;
pub const experts = uocr.experts;
pub const topk = uocr.topk;
pub const expert_ffn = uocr.expert_ffn;
pub const shared_ffn = uocr.shared_ffn;
pub const dense_ffn = uocr.dense_ffn;
pub const vocab = uocr.vocab;
pub const bf = uocr.bf;
pub const noSums = uocr.noSums;
pub const planted = uocr.planted;
pub const nan32 = std.math.nan(f32);
pub const inf32 = std.math.inf(f32);

/// A prompt chunk's rows (serve.zig's prompt_rows), its experts' slots, and the longest context's blocks.
pub const prompt_rows = 2048;
pub const block = 32;
pub const context_blocks = 4096;
pub const read_blocks = 12;
pub const ranked_blocks = 16;
pub const local = 128;
pub const eps: f32 = 1e-6;
/// 1 / sqrt(128), as decoder.metal holds it.
pub const inv_sqrt_hd: f64 = @as(f32, 0.0883883476483184405);
pub const router_scale: f64 = @as(f32, 11.3137084989847604);

/// The source a probe compiles: decoder.metal, or with one planted fault.
pub const Mutation = enum {
    none,
    gather_own_row,
    matvec_drops_tail,
    matvec_half_sums,
    norm_mean_over_threads,
    norm_add_wrong_sign,
    silu_swapped,
    combine_no_shared,
    qk_rotation_sign,
    clamp_drops_nan,
    block_keys_unnormalized,
    route_no_mean,
    attend_no_bias,
    gate_unnormalized,
    expert_gu_ignores_ids,
    expert_down_ignores_ids,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .gather_own_row => "control: metal uocr decoder ud_gather, row i reads row i, not rows[i]",
            .matvec_drops_tail => "control: metal uocr decoder ud_matvec, each lane's last 8 columns are dropped",
            .matvec_half_sums => "control: metal uocr decoder ud_matvec, the dot products sum in half",
            .norm_mean_over_threads => "control: metal uocr decoder ud_norm, the mean of squares divides by the threads",
            .norm_add_wrong_sign => "control: metal uocr decoder ud_norm, the added rows are subtracted",
            .silu_swapped => "control: metal uocr decoder ud_silu_mul, the gate is read from the up rows",
            .combine_no_shared => "control: metal uocr decoder ud_combine, the shared expert is not added",
            .qk_rotation_sign => "control: metal uocr decoder ud_qk, the rotation turns the other way",
            .clamp_drops_nan => "control: metal uocr decoder ud_qk, the router's clamp turns NaN into -1e4",
            .block_keys_unnormalized => "control: metal uocr decoder ud_block_keys, the weights are not divided by their sum",
            .route_no_mean => "control: metal uocr decoder ud_route, the log-weights keep their mean",
            .attend_no_bias => "control: metal uocr decoder ud_attend, the read blocks' biases are dropped",
            .gate_unnormalized => "control: metal uocr decoder ud_moe_gate, the scores are not divided by their sum",
            .expert_gu_ignores_ids => "control: metal uocr decoder ud_expert_gu, pick j reads expert j",
            .expert_down_ignores_ids => "control: metal uocr decoder ud_expert_down, pick j reads expert j",
        };
    }

    /// The gate that must go red.
    pub fn gate(m: Mutation) ?g.Gate {
        return switch (m) {
            .none => null,
            .matvec_half_sums => .sums,
            .clamp_drops_nan => .nonfinite,
            else => .shapes,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .gather_own_row => .{ .from = "x[ulong(rows[at.y]) * UD_H + at.x]", .to = "x[ulong(at.y) * UD_H + at.x]" },
            .matvec_drops_tail => .{ .from = "for (uint k = lane * 8; k < p.k; k += 256) {", .to = "for (uint k = lane * 8; k + 8 < p.k; k += 256) {" },
            .matvec_half_sums => .{ .from = "float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n  for (uint k = lane * 8; k < p.k;", .to = "half acc[4] = {0.0h, 0.0h, 0.0h, 0.0h};\n  for (uint k = lane * 8; k < p.k;" },
            .norm_mean_over_threads => .{ .from = "ss / float(UD_H) + p.eps", .to = "ss / 256.0f + p.eps" },
            .norm_add_wrong_sign => .{ .from = "x = ud_bf(x + float(A[", .to = "x = ud_bf(x - float(A[" },
            .silu_swapped => .{ .from = "const float g = float(G[ulong(at.y) * p.gstride + at.x]);", .to = "const float g = float(U[ulong(at.y) * p.ustride + at.x]);" },
            .combine_no_shared => .{ .from = "Y[o] = bfloat(ud_bf(acc) + float(S[o]));", .to = "Y[o] = bfloat(ud_bf(acc));" },
            .qk_rotation_sign => .{ .from = "const float rot = d < 64 ? -partner : partner;", .to = "const float rot = d < 64 ? partner : -partner;" },
            .clamp_drops_nan => .{ .from = "return isnan(x) ? x : clamp(x, -UD_CLAMP, UD_CLAMP);", .to = "return clamp(x, -UD_CLAMP, UD_CLAMP);" },
            .block_keys_unnormalized => .{ .from = "const float w = precise::divide(e, max(simd_sum(e), 1.0e-6f));", .to = "const float w = e;" },
            .route_no_mean => .{ .from = "bias[i] = i < keep ? logp[i] - mu : 0.0f;", .to = "bias[i] = i < keep ? logp[i] : 0.0f;" },
            .attend_no_bias => .{ .from = "kbias[i] = p.bf16_bias != 0 ? ud_bf(bias[s]) : bias[s];", .to = "kbias[i] = 0.0f;" },
            .gate_unnormalized => .{ .from = "const float mass = simd_sum(v[0] + v[1]);", .to = "const float mass = 1.0f;" },
            .expert_gu_ignores_ids => .{ .from = "const UdExpert ex = E[ids[j]];", .to = "const UdExpert ex = E[j];" },
            .expert_down_ignores_ids => .{ .from = "const device bfloat* W = E[ids[j]].down;", .to = "const device bfloat* W = E[j].down;" },
        };
    }
};

/// A decoder.metal kernel compiled as kernels.zig does, with a control's edit (an edit that finds nothing errors).
pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, m: Mutation, name: []const u8) !mtl.Pipeline {
    if (m == .none) return uocr.pipeline(device, .decoder, name);
    const plain = uocr.source(.decoder);
    const e = m.edit();
    if (std.mem.count(u8, plain, e.from) != 1) return error.ControlEditMissing;
    const text = try std.mem.replaceOwned(u8, a, plain, e.from, e.to);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

/// A probe's kernel under a control's name and gate.
pub fn mark(k: g.Kernel, m: Mutation) g.Kernel {
    var out = k;
    if (m != .none) {
        out.spec.name = m.name();
        out.control = m.gate();
    }
    return out;
}

/// A registration the run's --only selects (all without it).
pub fn wanted(only: ?[]const u8, name: []const u8) bool {
    const o = only orelse return true;
    return std.mem.indexOf(u8, name, o) != null;
}

/// A value's spread through fp32 steps and bf16 rounding points.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// Each end moved out by `ulps` fp32 roundings of its own size (and fp32's least step).
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * c.unit(.f32);
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    /// Each end moved out by `e`.
    pub fn widen(a: Range, e: f64) Range {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [_]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
};

/// The allowance a range gives about `want`: the farther end.
pub fn spread(want: f64, r: Range) f64 {
    return @max(@abs(want - r.lo), @abs(r.hi - want));
}

/// Bounds from a test's intervals: each output's allowance is its interval's farther end (none for non-finite ones).
pub fn boundsFromIntervals(a: std.mem.Allocator, t: anytype, want: []const f64, allow: []f64) !void {
    const n = want.len;
    const buf = try a.alloc(f64, 2 * n);
    defer a.free(buf);
    try t.intervals(buf[0..n], buf[n..]);
    for (allow, want, buf[0..n], buf[n..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) @max(spread(w, .{ .lo = lo, .hi = hi }), 0x1p-133) else 0;
}

/// Gate 5's texts for `uocr decoder ` launches naming one of `functions`: decoder.metal as is, on every chip.
pub fn limitsText(name: []const u8, functions: []const []const u8) ?Texts {
    if (!std.mem.startsWith(u8, name, "uocr decoder ")) return null;
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    for (functions) |f| if (std.mem.eql(u8, f, fun)) return .{ .here = uocr.source(.decoder), .pre_m5 = uocr.source(.decoder) };
    return null;
}

/// Random bf16 values in (-s, s).
pub fn fillBf16(rng: std.Random, out: []u16, s: f32) void {
    for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * s);
}
