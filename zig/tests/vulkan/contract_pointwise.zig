//! Pointwise shader contract (Vulkan): the six pointwise closures of stages 2-4 as
//! host-checkable fixtures, no device code here — contract_pointwise_main.zig drives the
//! real GPU. Stage 2's residual add, SwiGLU and RMSNorm fixtures live in stage2_fixture.zig
//! and are reused, not copied; this file adds the three that fixture cannot state:
//!  - the f32-preserving residual add: binary_vec.comp compiled without -DUSE_BF16 stores
//!    the exact f32 sum back in f32 — no rounding step at all, the strongest claim in the
//!    group ("what is preserved" per the wrapper's stated one-RNE-store pipeline);
//!  - the attention output gate: s = bf16(1/(1+exp(-gate))) rounded to bf16 once, then
//!    out = bf16(f32(ctx) * f32(s)) — ops 4, mids 1, exactly attention_gate.comp's math;
//!  - the gated RMSNorm: out = bf16(silu_f32(z) * widen(bf16(x * rsqrt(mean(x^2)+eps) * w)))
//!    — the rows family bound (the reduction sums only squares, so its relative error stays
//!    under gamma(depth) in any order) plus one mid rounding, the normed store.
//!
//! Denormal honesty (stated, not fitted): Apple GPUs flush fp32 denormals. The preserve-add
//! classes case plants f32-subnormal RESULTS and a flush-to-zero device FAILS those rows —
//! denormal preservation is not claimed without a GPU result. The same policy the stage 2
//! add fixture states for bf16 subnormals; the reference is never bent to the flush.
const std = @import("std");
const contract = @import("contract");
const stage2 = @import("stage2_fixture.zig");

const c = contract.spec;
pub const Report = stage2.Report;
pub const classify = stage2.classify;

/// Element offsets (elements of the binding's kind; 256-byte alignment) of each binding's data.
pub const Offsets = stage2.Offsets;

pub const tail = stage2.tail;
pub const poison16 = stage2.poison16;
pub const poison32: u32 = 0xffffffff;

fn classOfBits(bits: u16) c.Class {
    return c.classOf(c.fromBf16(bits));
}

fn poisonedShell16(a: std.mem.Allocator, at: usize, len: usize) ![]u16 {
    const buf = try a.alloc(u16, at + len + tail);
    @memset(buf, poison16);
    return buf;
}

fn poisonedShell32(a: std.mem.Allocator, at: usize, len: usize) ![]u32 {
    const buf = try a.alloc(u32, at + len + tail);
    @memset(buf, poison32);
    return buf;
}

fn guardIntact(comptime T: type, shell: []const T, at: usize, len: usize, poison: T) !void {
    for (shell, 0..) |g, i| {
        if (i >= at and i < at + len) continue;
        if (g != poison) return error.GuardClobbered;
    }
}

/// A bf16-representable f32 value near `1` scaled by `scale` (a power of two).
fn sampleBits(rng: std.Random, scale: f32) u16 {
    const v = (rng.float(f32) + 0.5) * scale * (if (rng.boolean()) @as(f32, 1) else -1);
    return c.toBf16(v);
}

// ---------------------------------------------------------------------------
// Residual add, preserved: binary_vec.comp without -DUSE_BF16, the exact f32 sum
// ---------------------------------------------------------------------------

pub const PreserveFill = enum { random, cancel, classes };

pub const PreserveCase = struct { name: []const u8, elems: usize, at: Offsets = .{}, fill: PreserveFill = .random };

/// Counts are multiples of four (binary_vec moves four elements per 8-byte load, the f32
/// path included) and offsets are 256-byte aligned (64 f32 elements).
pub const preserve_cases = [_]PreserveCase{
    .{ .name = "random 512", .elems = 512, .at = .{ .a = 64, .b = 128, .out = 192 } },
    .{ .name = "cancel 512", .elems = 512, .at = .{ .a = 64, .b = 128, .out = 192 }, .fill = .cancel },
    .{ .name = "classes 20", .elems = 20, .at = .{ .a = 64, .b = 128, .out = 192 }, .fill = .classes },
};

/// Planted classes-case inputs; every pair is exercised (elems == len). Rows 9-11 and 13
/// produce or carry fp32 SUBNORMALS: a flush-to-zero device (Apple GPUs flush fp32
/// denormals) fails them honestly — that failure is the point, the reference is not fitted.
const f32_class_pairs = [_][2]u32{
    .{ 0x7f800000, 0x3f800000 }, // +inf + 1 -> +inf
    .{ 0xff800000, 0x3f800000 }, // -inf + 1 -> -inf
    .{ 0x7f800000, 0xff800000 }, // +inf + -inf -> nan
    .{ 0x7fc00000, 0x3f800000 }, // nan + 1 -> nan
    .{ 0x00000000, 0x80000000 }, // +0 + -0 -> +0 (round to nearest)
    .{ 0x80000000, 0x80000000 }, // -0 + -0 -> -0
    .{ 0x7f7fffff, 0x7f7fffff }, // f32 max + f32 max -> f32 overflow -> +inf
    .{ 0xff7fffff, 0xff7fffff }, // both min -> -inf
    .{ 0x3f800000, 0xbf800000 }, // 1 + -1 -> +0
    .{ 0x00000001, 0x00000001 }, // smallest f32 DENORMAL doubled -> 0x00000002
    .{ 0x00000001, 0x00000003 }, // 1 + 3 denormal units -> 0x00000004
    .{ 0x00000001, 0x80000000 }, // denormal + -0 -> unchanged
    .{ 0x00800000, 0x00800000 }, // smallest NORMAL doubled -> 0x01000000
    .{ 0x00800000, 0x807fffff }, // smallest normal - largest denormal -> 0x00000001
    .{ 0x00800000, 0x80000000 }, // smallest normal + -0 -> unchanged
    .{ 0x4f000000, 0xb0800000 }, // 2^31 - ~2^-10, huge spread: f32 rounds to 2^31
    .{ 0x42f60000, 0xc2f60000 }, // ±123 -> +0
    .{ 0x3f000000, 0xc0000000 }, // 0.5 + -2 -> -1.5 exact
    .{ 0x34000000, 0x00000001 }, // 2^-23 + 2^-149: the denormal is under half an ulp -> 2^-23
    .{ 0x3f800000, 0x38800000 }, // 1 + 2^-14 exact in f32
};

pub const PreserveAdd = struct {
    a: std.mem.Allocator,
    case: PreserveCase,
    a_bits: []u32,
    b_bits: []u32,
    out: []u32,
    /// The exact bits the shader must produce: the IEEE f32 sum, stored unrounded.
    want: []u32,

    pub fn init(a: std.mem.Allocator, case: PreserveCase, seed: u64) !PreserveAdd {
        const a_bits = try poisonedShell32(a, case.at.a, case.elems);
        errdefer a.free(a_bits);
        const b_bits = try poisonedShell32(a, case.at.b, case.elems);
        errdefer a.free(b_bits);
        const out = try poisonedShell32(a, case.at.out, case.elems);
        errdefer a.free(out);
        const want = try a.alloc(u32, case.elems);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..case.elems) |i| {
            const pair: [2]u32 = switch (case.fill) {
                .random => .{
                    @bitCast((rng.float(f32) + 0.5) * std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * (if (rng.boolean()) @as(f32, 1) else -1)),
                    @bitCast((rng.float(f32) + 0.5) * std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 3)) - 1)) * (if (rng.boolean()) @as(f32, 1) else -1)),
                },
                // v and -(v - k*ulp(v)): the sum is k ulps of v, exact in f32
                .cancel => blk: {
                    const vbits: u32 = @bitCast((rng.float(f32) + 0.5) * (if (rng.boolean()) @as(f32, 1) else -1));
                    const v: f32 = @bitCast(vbits);
                    const k: f32 = @floatFromInt(i % 3);
                    const ulp = std.math.pow(f32, 2, @floor(std.math.log2(@abs(v))) - 23);
                    break :blk .{ vbits, @bitCast(-(v - k * ulp)) };
                },
                .classes => f32_class_pairs[i % f32_class_pairs.len],
            };
            a_bits[case.at.a + i] = pair[0];
            b_bits[case.at.b + i] = pair[1];
            want[i] = @as(u32, @bitCast(@as(f32, @bitCast(pair[0])) + @as(f32, @bitCast(pair[1]))));
        }
        return .{ .a = a, .case = case, .a_bits = a_bits, .b_bits = b_bits, .out = out, .want = want };
    }

    pub fn deinit(f: PreserveAdd) void {
        f.a.free(f.a_bits);
        f.a.free(f.b_bits);
        f.a.free(f.out);
        f.a.free(f.want);
    }

    pub const Mismatch = struct { kind: error{ NonFiniteMismatch, BitMismatch }, index: usize, a: u32, b: u32, want: u32, got: u32 };

    /// The first mismatch under the exact check (class-vs-bit), or null when the whole window
    /// matches. Length/guard failures stay errors: the caller names them.
    pub fn firstMismatch(f: PreserveAdd, got: []const u32) !?Mismatch {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(u32, got, at, f.case.elems, poison32);
        for (f.want, 0..) |w, i| {
            const g = got[at + i];
            if (c.classOf(@as(f32, @bitCast(g))) != c.classOf(@as(f32, @bitCast(w)))) return .{ .kind = error.NonFiniteMismatch, .index = i, .a = f.a_bits[f.case.at.a + i], .b = f.b_bits[f.case.at.b + i], .want = w, .got = g };
            if (std.math.isFinite(@as(f32, @bitCast(w))) and g != w) return .{ .kind = error.BitMismatch, .index = i, .a = f.a_bits[f.case.at.a + i], .b = f.b_bits[f.case.at.b + i], .want = w, .got = g };
        }
        return null;
    }

    pub fn check(f: PreserveAdd, got: []const u32) !void {
        if (try f.firstMismatch(got)) |m| return m.kind;
    }
};

// ---------------------------------------------------------------------------
// Attention output gate: one mid bf16 rounding of the sigmoid, one bf16 store
// ---------------------------------------------------------------------------

/// ops = 4: the f32 chain exp -> 1+ -> divide and the ctx multiply. mids = 1: the sigmoid's
/// own bf16 rounding (attention_gate.comp's "Metal's bfloat(s)"). Its error is bounded by
/// 2^-9 |s|, so scaled by |ctx| it stays within mids*outUnit of |want| = |ctx|*|s| exactly —
/// boundExact needs no per-element repair. This is a FIXED BRING-UP ACCEPTANCE BUDGET, not a
/// Vulkan-spec proof; exp is transcendental and no per-op bound is guaranteed by the spec.
const gate_spec: c.Spec = .{ .name = "vulkan attention_gate bf16 (bf16(sigmoid(gate))*ctx)", .accum = .none, .depth = gateDepth, .ops = 4, .mids = 1, .out = .bf16, .served = &[_]c.Shape{} };
fn gateDepth(_: c.Shape) usize {
    return 0; // elementwise
}

pub const gate_row_elems: usize = 2048; // 8 heads * 256, the gate's output row
pub const gate_qraw_row_elems: usize = 2 * gate_row_elems; // the interleaved q/gate projection row

pub const GateCase = struct { name: []const u8, rows: usize, at: Offsets = .{}, controls: bool = false };

pub const gate_cases = [_]GateCase{
    .{ .name = "random 2x2048", .rows = 2, .at = .{ .a = 128, .b = 256, .out = 384 } },
    .{ .name = "controls 2x2048", .rows = 2, .at = .{ .a = 128, .b = 256, .out = 384 }, .controls = true },
};

/// Gate controls: planted nan/±inf/±0, the model-range ends, and a bf16 subnormal. |gate| <=
/// 100 keeps exp(-gate) inside f32's range so the f64 reference and the f32 shader agree.
const gate_controls = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x3f80, 0xbf80, 0x41a0, 0xc1a0, 0x42c8, 0xc2c8, 0x3e80, 0xbd80, 0x0001, 0x3f00, 0xbf00 };

/// |gate| <= 16 keeps exp(-gate) at most e^16 ~ 8.9e6, safely inside f32.
fn gateBits(rng: std.Random) u16 {
    return c.toBf16((rng.float(f32) * 2 - 1) * 16);
}

pub const Gate = struct {
    a: std.mem.Allocator,
    case: GateCase,
    /// ctx bits: binding 0, rows * 2048 elements.
    ctx: []u16,
    /// qraw bits: binding 1, rows * 4096 elements; element d of row r rides at
    /// r*4096 + (d/256)*512 + 256 + d%256 — the oracle's interleaved layout.
    qraw: []u16,
    out: []u16,
    /// f64(ctx) * sigmoid64(gate) per element; nonfinite only where the gate is planted so.
    want: []f64,

    pub fn init(a: std.mem.Allocator, case: GateCase, seed: u64) !Gate {
        const elems = case.rows * gate_row_elems;
        const ctx = try poisonedShell16(a, case.at.a, elems);
        errdefer a.free(ctx);
        const qraw = try poisonedShell16(a, case.at.b, case.rows * gate_qraw_row_elems);
        errdefer a.free(qraw);
        const out = try poisonedShell16(a, case.at.out, elems);
        errdefer a.free(out);
        const want = try a.alloc(f64, elems);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..elems) |i| {
            const row = i / gate_row_elems;
            const d = i % gate_row_elems;
            const gate: u16 = if (case.controls) gate_controls[i % gate_controls.len] else gateBits(rng);
            // Controls ctx: 1, except ctx = -1 under the -inf gate (a signed zero) and two
            // zero-ctx elements (511: -0, 1023: +0) whose zero outputs carry ctx's sign.
            const ctx_bits: u16 = if (!case.controls) sampleBits(rng, 1) else if (i == 2) 0xbf80 else if (i == 511) 0x8000 else if (i == 1023) 0x0000 else 0x3f80;
            ctx[case.at.a + i] = ctx_bits;
            // The shader's read: head = d/256, gate element = row*4096 + head*512 + 256 + d%256.
            qraw[case.at.b + row * gate_qraw_row_elems + (d / 256) * 512 + 256 + d % 256] = gate;
            const g64: f64 = c.fromBf16(gate);
            const e = @exp(-g64);
            const s = 1 / (1 + e);
            want[i] = c.fromBf16(ctx_bits) * s;
        }
        return .{ .a = a, .case = case, .ctx = ctx, .qraw = qraw, .out = out, .want = want };
    }

    pub fn deinit(f: Gate) void {
        f.a.free(f.ctx);
        f.a.free(f.qraw);
        f.a.free(f.out);
        f.a.free(f.want);
    }

    /// Finite wants within boundExact (ops 4, mids 1, the bf16 store); a zero want (ctx zero
    /// or gate -inf) demands an exactly-zero output WITH the product's sign: the f32 multiply's
    /// sign rule makes it deterministic. Nonfinite wants compare classes.
    pub fn check(f: Gate, got: []const u16) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(u16, got, at, f.case.rows * gate_row_elems, poison16);
        var r: Report = .{};
        for (f.want, 0..) |w, i| {
            const g = got[at + i];
            const gf = c.fromBf16(g);
            if (!std.math.isFinite(w) or !std.math.isFinite(gf)) {
                if (c.classOf(gf) != c.classOf(w)) return error.NonFiniteMismatch;
                continue;
            }
            if (w == 0) {
                if (gf != 0 or std.math.signbit(gf) != std.math.signbit(w)) return error.OutOfBound;
                continue;
            }
            const err = @abs(gf - w);
            const bound = c.boundExact(gate_spec, w);
            if (!(std.math.isFinite(bound) and bound > 0)) return error.BadBound;
            const ratio = err / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .row = i / gate_row_elems, .col = i % gate_row_elems };
        }
        if (r.worst > 1) return error.OutOfBound;
        return r;
    }
};

// ---------------------------------------------------------------------------
// Gated RMSNorm: the rows family bound plus one mid rounding
// ---------------------------------------------------------------------------

/// depth = 20 f32 reduction roundings, the same conservative figure the stage 2 RMSNorm
/// fixture states for the identical 256-thread strided-partial + tree structure at width
/// 2048. ops = 10: the mean divide, inversesqrt (<= 2 ulp), norm*weight, value*norm_weight,
/// the sigmoid's exp/1+/divide, the silu multiply and the final product. mids = 12: ONE
/// mid bf16 rounding of the normed value (outUnit 2^-8) whose error is relative to |normed|,
/// i.e. |want|/|silu(z)| — the fixture's gate floor |z| >= 0.25 keeps |silu(z)| >= 0.109,
/// so the mid costs at most 2^-8/0.109 ~ 3.6e-3 of |want| and mids = 12 covers it with
/// margin. Fixed bring-up acceptance, NOT a Vulkan-spec proof.
const gated_depth = 20;
const gated_spec: c.Spec = .{
    .name = "vulkan fast_norm_gated bf16 gated rms rows (silu(z)*bf16(x*(inv*w)))",
    .accum = .f32,
    .depth = gatedDepth,
    .ops = 10,
    .mids = 12,
    .out = .bf16,
    .served = &[_]c.Shape{ .{ .rows = 1, .n = 1, .k = gated_width }, .{ .rows = 2, .n = 1, .k = gated_width } },
};
fn gatedDepth(_: c.Shape) usize {
    return gated_depth;
}

/// The production wrapper's pinned width (gdn.Model.linear_dim, 2048).
pub const gated_width: usize = 2048;
pub const gated_eps: f32 = 1e-6;

pub const GatedOffsets = struct { x: usize = 0, weight: usize = 0, gate: usize = 0, out: usize = 0 };

pub const GatedCase = struct {
    name: []const u8,
    rows: usize,
    fill: stage2.RmsFill = .flat,
    /// The last row is all zeros: mean square 0, norm = rsqrt(eps), outputs exactly zero.
    zero_row: bool = false,
    /// Row 0 plants z controls (nan/+inf/-inf/±0/tame magnitudes); row 1 poisons x[0] with
    /// +inf (its sum goes infinite, norm 0: signed-zero outputs, nan at x[0]'s inf*0) and a
    /// nan z at column 1; row 2 poisons x[0] with nan (the sum goes nan: a whole nan row).
    controls: bool = false,
    at: GatedOffsets = .{},
};

pub const gated_cases = [_]GatedCase{
    .{ .name = "1x2048", .rows = 1, .at = .{ .x = 128, .weight = 0, .gate = 256, .out = 384 } },
    .{ .name = "2x2048 dynamic zero row", .rows = 2, .fill = .dynamic, .zero_row = true, .at = .{ .x = 128, .weight = 0, .gate = 256, .out = 384 } },
    .{ .name = "controls 3x2048", .rows = 3, .controls = true, .at = .{ .x = 128, .weight = 0, .gate = 256, .out = 384 } },
};

/// The z controls planted at row 0, columns 0..: |silu| >= 0.109 for every finite tame value
/// (the mids budget's stated floor), the signed zeros come from z = ±0, the nonfinite
/// classes from z = nan/±inf. z = -inf gives silu = -inf*0 = nan in f32 exactly as in f64.
const gated_z_controls = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x3f80, 0xbf80, 0x3e80, 0xbe80, 0x4000, 0x4100, 0xc000 };

/// |z| in [0.25, 8] on one side of zero (resampled): the mids budget's stated gate floor.
fn gatedGateBits(rng: std.Random) u16 {
    var v = rng.float(f32) * 10 - 2;
    while (@abs(v) < 0.25) v = rng.float(f32) * 10 - 2;
    return c.toBf16(v);
}

fn sigmoid64(g: f64) f64 {
    return 1 / (1 + @exp(-g));
}

fn silu64(z: f64) f64 {
    return z * sigmoid64(z);
}

pub const GatedNorm = struct {
    a: std.mem.Allocator,
    case: GatedCase,
    x: []u16,
    weight: []u16,
    z: []u16,
    out: []u16,
    /// silu64(z) * widen(bf16(x*(n*w))) per element, f64, mid rounding modeled; nonfinite
    /// where the planted controls make it so. want == 0 carries the product's exact sign.
    want: []f64,

    pub fn init(a: std.mem.Allocator, case: GatedCase, seed: u64) !GatedNorm {
        const elems = case.rows * gated_width;
        const x = try poisonedShell16(a, case.at.x, elems);
        errdefer a.free(x);
        const weight = try poisonedShell16(a, case.at.weight, gated_width);
        errdefer a.free(weight);
        const z = try poisonedShell16(a, case.at.gate, elems);
        errdefer a.free(z);
        const out = try poisonedShell16(a, case.at.out, elems);
        errdefer a.free(out);
        const want = try a.alloc(f64, elems);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..gated_width) |i| weight[case.at.weight + i] = sampleBits(rng, 1);
        for (0..case.rows) |row| {
            const base = case.at.x + row * gated_width;
            const zbase = case.at.gate + row * gated_width;
            var sum_sq: f64 = 0;
            for (0..gated_width) |i| {
                var bits: u16 = if (case.zero_row and row + 1 == case.rows) 0x0000 else switch (case.fill) {
                    .flat => sampleBits(rng, 1),
                    .dynamic => sampleBits(rng, if (i % 4 == 0) 0x1p10 else if (i % 4 == 2) 0x1p-10 else 1),
                };
                if (case.controls and row == 1 and i == 0) bits = 0x7f80; // +inf: the row's sum goes infinite, norm 0
                if (case.controls and row == 2 and i == 0) bits = 0x7fc0; // nan: the whole row's sums go nan
                x[base + i] = bits;
                sum_sq += @as(f64, c.fromBf16(bits)) * @as(f64, c.fromBf16(bits));
            }
            const n: f64 = 1 / @sqrt(sum_sq / @as(f64, @floatFromInt(gated_width)) + gated_eps);
            for (0..gated_width) |i| {
                const zb: u16 = if (case.controls and row == 0 and i < gated_z_controls.len) gated_z_controls[i] else if (case.controls and row == 1 and i == 1) 0x7fc0 else gatedGateBits(rng);
                z[zbase + i] = zb;
                // The shader's association: value * (norm * weight), one mid bf16 rounding.
                const normed = c.toBf16(@floatCast(c.fromBf16(x[base + i]) * (n * c.fromBf16(weight[case.at.weight + i]))));
                want[row * gated_width + i] = silu64(c.fromBf16(zb)) * c.fromBf16(normed);
            }
        }
        return .{ .a = a, .case = case, .x = x, .weight = weight, .z = z, .out = out, .want = want };
    }

    pub fn deinit(f: GatedNorm) void {
        f.a.free(f.x);
        f.a.free(f.weight);
        f.a.free(f.z);
        f.a.free(f.out);
        f.a.free(f.want);
    }

    /// The rows-family budget (gamma(depth) of the output's own magnitude, the reduction
    /// sums only squares) plus the mids/ops terms; a zero want demands the exactly-zero
    /// SIGNED output (a flushed or sign-flipped zero is refused); nonfinite wants compare
    /// classes; a finite want with a nonfinite output is refused outright.
    pub fn check(f: GatedNorm, got: []const u16) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(u16, got, at, f.case.rows * gated_width, poison16);
        var r: Report = .{};
        const shape: c.Shape = .{ .rows = f.case.rows, .n = 1, .k = gated_width };
        for (f.want, 0..) |w, i| {
            const g = got[at + i];
            const gf = c.fromBf16(g);
            if (!std.math.isFinite(w) or !std.math.isFinite(gf)) {
                if (c.classOf(gf) != c.classOf(w)) return error.NonFiniteMismatch;
                continue;
            }
            if (w == 0) {
                if (gf != 0 or std.math.signbit(gf) != std.math.signbit(w)) return error.OutOfBound;
                continue;
            }
            const err = @abs(gf - w);
            const bound = c.bound(gated_spec, shape, w, @abs(w));
            if (!(std.math.isFinite(bound) and bound > 0)) return error.BadBound;
            const ratio = err / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .row = i / gated_width, .col = i % gated_width };
        }
        if (r.worst > 1) return error.OutOfBound;
        return r;
    }
};

const testing = std.testing;

// ---------------------------------------------------------------------------
// Host tests: the references and the checkers, no GPU
// ---------------------------------------------------------------------------

test "preserve wants are the exact f32 sums, subnormal results included" {
    for (preserve_cases) |case| {
        const f = try PreserveAdd.init(testing.allocator, case, 0x58a2);
        defer f.deinit();
        for (0..f.case.elems) |i| {
            const sum = @as(f32, @bitCast(f.a_bits[f.case.at.a + i])) + @as(f32, @bitCast(f.b_bits[f.case.at.b + i]));
            try testing.expectEqual(@as(u32, @bitCast(sum)), f.want[i]);
        }
    }
    // The planted denormal rows survive the host sum: 0x00000001 + 0x00000001 = 0x00000002.
    const g = try PreserveAdd.init(testing.allocator, preserve_cases[2], 0x58a3);
    defer g.deinit();
    try testing.expectEqual(@as(u32, 0x00000002), g.want[9]);
    try testing.expectEqual(@as(u32, 0x00000004), g.want[10]);
    try testing.expectEqual(@as(u32, 0x00000001), g.want[13]); // smallest normal - largest denormal
}

test "preserve classes plants every nonfinite class and the cancel case produces exact tiny sums" {
    const g = try PreserveAdd.init(testing.allocator, preserve_cases[2], 0x58a4);
    defer g.deinit();
    var classes: std.EnumSet(c.Class) = .empty;
    for (g.want) |w| classes.insert(c.classOf(@as(f32, @bitCast(w))));
    try testing.expect(classes.contains(.nan));
    try testing.expect(classes.contains(.pos_inf));
    try testing.expect(classes.contains(.neg_inf));
    try testing.expect(classes.contains(.finite));
    const f = try PreserveAdd.init(testing.allocator, preserve_cases[1], 0x58a5);
    defer f.deinit();
    var tiny: usize = 0;
    for (0..f.case.elems) |i| {
        const sum = @as(f32, @bitCast(f.a_bits[f.case.at.a + i])) + @as(f32, @bitCast(f.b_bits[f.case.at.b + i]));
        if (@abs(sum) < 0x1p-20) tiny += 1;
    }
    try testing.expect(tiny > 0); // v + -(v - 0*ulp) pairs cancel to exact zeros
}

test "preserve checker reports the first bad element and keeps named failures" {
    const f = try PreserveAdd.init(testing.allocator, preserve_cases[0], 0x58a6);
    defer f.deinit();
    const bits = try testing.allocator.dupe(u32, f.out);
    defer testing.allocator.free(bits);
    @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
    try testing.expectEqual(@as(?PreserveAdd.Mismatch, null), try f.firstMismatch(bits));
    bits[f.case.at.out + 3] +%= 1; // one ulp off: a finite bit mismatch
    const m = (try f.firstMismatch(bits)).?;
    try testing.expectEqual(error.BitMismatch, m.kind);
    try testing.expectEqual(@as(usize, 3), m.index);
    try testing.expectEqual(f.want[3], m.want);
    try testing.expectError(error.BitMismatch, f.check(bits));
    try testing.expectError(error.LengthMismatch, f.firstMismatch(f.out[0 .. f.out.len - 1]));
    bits[f.case.at.out + 3] = f.want[3];
    bits[0] = 0x1234; // guard clobbered
    try testing.expectError(error.GuardClobbered, f.firstMismatch(bits));
}

test "gate references stay finite on random cases and the controls plant nan and signed zeros" {
    for (gate_cases) |case| {
        const f = try Gate.init(testing.allocator, case, 0x58a7);
        defer f.deinit();
        if (!case.controls) for (f.want) |w| try testing.expect(std.math.isFinite(w));
    }
    const g = try Gate.init(testing.allocator, gate_cases[1], 0x58a8);
    defer g.deinit();
    try testing.expect(std.math.isNan(g.want[0])); // nan gate
    try testing.expectEqual(@as(f64, 1), g.want[1]); // +inf gate: sigmoid 1, ctx 1
    try testing.expect(std.math.signbit(g.want[2])); // -inf gate with ctx -1: zero carries ctx's sign
    try testing.expectEqual(@as(f64, 0), @abs(g.want[2]));
}

test "gate qraw layout matches the shader's interleaved read" {
    const f = try Gate.init(testing.allocator, gate_cases[0], 0x58a9);
    defer f.deinit();
    // Every gate element d of row r sits at r*4096 + (d/256)*512 + 256 + d%256; nothing else
    // in the qraw window is written, so the poison remains exactly there.
    for (f.qraw, 0..) |g, i| {
        if (i < f.case.at.b) continue; // the poison before the data window
        const w = i - f.case.at.b;
        const row = w / gate_qraw_row_elems;
        const off = w % gate_qraw_row_elems;
        const within = off % 512;
        const planted = within >= 256;
        if (planted != (g != poison16)) return error.TestUnexpectedResult;
        if (row >= f.case.rows and g != poison16) return error.TestUnexpectedResult;
    }
}

test "gate checker refuses wrong finite outputs, wrong-signed zeros and class breaks" {
    const f = try Gate.init(testing.allocator, gate_cases[1], 0x58aa);
    defer f.deinit();
    const bits = try testing.allocator.dupe(u16, f.out);
    defer testing.allocator.free(bits);
    for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
    _ = try f.check(bits);
    // A wrong finite output is out of bound.
    const finite_at = blk: {
        for (f.want, 0..) |w, i| if (w != 0 and std.math.isFinite(w)) break :blk i;
        unreachable;
    };
    bits[f.case.at.out + finite_at] = c.toBf16(@floatCast(f.want[finite_at] * 1.1));
    try testing.expectError(error.OutOfBound, f.check(bits));
    // A sign-flipped zero is refused: the product's sign is deterministic.
    for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
    const zero_at = blk: {
        for (f.want, 0..) |w, i| if (w == 0) break :blk i;
        unreachable;
    };
    bits[f.case.at.out + zero_at] ^= 0x8000;
    try testing.expectError(error.OutOfBound, f.check(bits));
    // A class break at the nan gate.
    for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
    bits[f.case.at.out] = 0x3f80; // nan wanted, 1 got
    try testing.expectError(error.NonFiniteMismatch, f.check(bits));
}

test "gated references are finite off the controls, and the zero row wants exact signed zeros" {
    for (gated_cases) |case| {
        const f = try GatedNorm.init(testing.allocator, case, 0x58ab);
        defer f.deinit();
        if (case.controls) continue;
        for (f.want) |w| try testing.expect(std.math.isFinite(w));
        if (case.zero_row) {
            const zero_at = (case.rows - 1) * gated_width;
            for (f.want[zero_at..]) |w| try testing.expectEqual(@as(f64, 0), @abs(w));
        }
    }
    // The controls: nan and both infinities among the wants, signed zeros from z = ±0.
    const g = try GatedNorm.init(testing.allocator, gated_cases[2], 0x58ac);
    defer g.deinit();
    try testing.expect(std.math.isNan(g.want[0])); // z = nan
    try testing.expect(std.math.isInf(g.want[1])); // z = +inf, silu +inf
    try testing.expect(std.math.isNan(g.want[2])); // z = -inf: silu = -inf*0 = nan
    try testing.expectEqual(@as(f64, 0), @abs(g.want[3])); // z = +0
    // Row 1: the infinite sum gives norm 0, so signed zeros everywhere except the two
    // nonfinite products: x[0]'s inf*0 and the nan z at column 1.
    for (0..gated_width) |i| {
        const w = g.want[gated_width + i];
        if (i == 0 or i == 1) {
            try testing.expect(std.math.isNan(w));
        } else {
            try testing.expectEqual(@as(f64, 0), @abs(w));
        }
    }
    // Row 2: a nan input poisons the row's sums — every output is nan.
    for (g.want[2 * gated_width ..]) |w| try testing.expect(std.math.isNan(w));
}

test "gated checker refuses wrong finite outputs, sign-flipped zeros and class breaks" {
    // Random case: one wrong finite output is out of bound.
    {
        const f = try GatedNorm.init(testing.allocator, gated_cases[0], 0x58ad);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        // Doubling is many bf16 ulps off at EVERY magnitude (an affine perturbation has a
        // fixed point the bf16 rounding can collapse onto — w*1.1+0.05 at w = -0.5).
        bits[f.case.at.out + gated_width / 2] = c.toBf16(@floatCast(f.want[gated_width / 2] * 2));
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // Zero row: one bf16 subnormal unit where an exact zero belongs is refused.
    {
        const f = try GatedNorm.init(testing.allocator, gated_cases[1], 0x58ae);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        bits[f.case.at.out + (f.case.rows - 1) * gated_width] = 0x0001;
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // Controls: a sign-flipped zero and a class break are both refused.
    {
        const f = try GatedNorm.init(testing.allocator, gated_cases[2], 0x58af);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        const zero_at = blk: {
            for (f.want[0..gated_width], 0..) |w, i| if (w == 0) break :blk i;
            unreachable;
        };
        bits[f.case.at.out + zero_at] ^= 0x8000;
        try testing.expectError(error.OutOfBound, f.check(bits));
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        bits[f.case.at.out] = 0x3f80; // nan wanted, 1 got
        try testing.expectError(error.NonFiniteMismatch, f.check(bits));
    }
}

fn preserveAllocRun(a: std.mem.Allocator, case: PreserveCase) !void {
    (try PreserveAdd.init(a, case, 0x58b0)).deinit();
}

fn gateAllocRun(a: std.mem.Allocator, case: GateCase) !void {
    (try Gate.init(a, case, 0x58b1)).deinit();
}

fn gatedAllocRun(a: std.mem.Allocator, case: GatedCase) !void {
    (try GatedNorm.init(a, case, 0x58b2)).deinit();
}

test "every fixture init survives each single allocation failure without leaking" {
    try std.testing.checkAllAllocationFailures(testing.allocator, preserveAllocRun, .{preserve_cases[0]});
    try std.testing.checkAllAllocationFailures(testing.allocator, preserveAllocRun, .{preserve_cases[2]});
    for (gate_cases) |case| try std.testing.checkAllAllocationFailures(testing.allocator, gateAllocRun, .{case});
    for (gated_cases) |case| try std.testing.checkAllAllocationFailures(testing.allocator, gatedAllocRun, .{case});
}

test "guards sit poisoned outside every data window" {
    const f = try GatedNorm.init(testing.allocator, gated_cases[1], 0x58b3);
    defer f.deinit();
    for (f.x, 0..) |g, i| {
        if (i >= f.case.at.x and i < f.case.at.x + f.case.rows * gated_width) continue;
        try testing.expectEqual(poison16, g);
    }
    for (f.z, 0..) |g, i| {
        if (i >= f.case.at.gate and i < f.case.at.gate + f.case.rows * gated_width) continue;
        try testing.expectEqual(poison16, g);
    }
    const p = try PreserveAdd.init(testing.allocator, preserve_cases[0], 0x58b4);
    defer p.deinit();
    for (p.a_bits, 0..) |g, i| {
        if (i >= p.case.at.a and i < p.case.at.a + p.case.elems) continue;
        try testing.expectEqual(poison32, g);
    }
}
