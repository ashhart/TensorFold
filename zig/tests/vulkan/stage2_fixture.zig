//! Stage 2 fixture: residual-add, SwiGLU and RMSNorm cases against the Metal-oracle (ASH_QWEN35)
//! semantics, host-checkable; no device code here. Stage 2 operator bring-up only: no model,
//! no performance claim, and NO cross-backend bitexact claim — only the bounds stated per gate.
//!
//! Acceptance, stated plainly (fixed bring-up budgets, NOT Vulkan-spec proofs, NOT full
//! kernel-contract bounds):
//!  - Residual add: outputs must match the shader's own pipeline bit for bit — the host computes
//!    the scalar f32 add and applies the contract's RNE bf16 cast (the same bit formula as the
//!    shader's bf16_store). Finite outputs compare by bits; nonfinite outputs by class (a driver
//!    may raise a different NaN payload). No FP64 reference is claimed for this gate.
//!  - SwiGLU: |got - want| <= c.boundExact: an independent FP64 reference, the output's own bf16
//!    rounding, and a FIXED budget of 8*2^-24 relative for the f32 exp/1+/div/mul chain. The
//!    8-op figure is an acceptance choice (exp is transcendental; no per-op error bound is proven
//!    here and none is guaranteed by the Vulkan spec).
//!  - RMSNorm: |got - want| <= c.bound(rms_spec, shape, want, |want|): the reduction sums only
//!    squares (a positive sum), so its relative error stays under gamma(20) in ANY order (<= 10
//!    roundings at width 256, 16 at 2048 — 20 is conservative), sqrt halves it, and inversesqrt
//!    plus two f32 multiplies plus one bf16 store fit in ops 4 + outUnit. The norm's output error
//!    is relative for EVERY element because the large-magnitude sum enters only through the
//!    normalized denominator, so the output's own magnitude is the budget's magnitude scale.
//!    want == 0 demands an exactly-zero output (a small nonzero is rejected outright).
//! Every finite reference demands a finite output (planted nan/+inf/-inf must be refused); the
//! nonfinite controls compare value CLASSES against the FP64 reference. Subnormal rows in the add
//! classes case make a flush-to-zero device FAIL honestly — denormal preservation is not claimed
//! without a GPU result.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;

/// Poisoned elements after each binding's data, so an overrun reads poison rather than past the buffer.
pub const tail = 16;
pub const poison16: u16 = 0xffff;

fn classOfBits(bits: u16) c.Class {
    return c.classOf(c.fromBf16(bits));
}

/// The worst output: |got - want| over its bound (nonfinite counts as infinitely far).
pub const Report = struct { worst: f64 = 0, row: usize = 0, col: usize = 0 };

/// Element offsets (bf16 elements; 256-byte alignment = multiples of 128) of each binding's data.
pub const Offsets = struct { a: usize = 0, b: usize = 0, out: usize = 0 };

fn poisonedShell(a: std.mem.Allocator, at: usize, len: usize) ![]u16 {
    const buf = try a.alloc(u16, at + len + tail);
    @memset(buf, poison16);
    return buf;
}

fn guardIntact(shell: []const u16, at: usize, len: usize) !void {
    for (shell, 0..) |g, i| {
        if (i >= at and i < at + len) continue;
        if (g != poison16) return error.GuardClobbered;
    }
}

/// A bf16-representable f32 value near `1` scaled by `scale` (a power of two).
fn sampleBits(rng: std.Random, scale: f32) u16 {
    const v = (rng.float(f32) + 0.5) * scale * (if (rng.boolean()) @as(f32, 1) else -1);
    return c.toBf16(v);
}

// ---------------------------------------------------------------------------
// Residual add: matches the shader pipeline (f32 add + one RNE bf16 store) bit for bit
// ---------------------------------------------------------------------------

pub const AddFill = enum { random, cancel, classes };

pub const AddCase = struct { name: []const u8, elems: usize, at: Offsets = .{}, fill: AddFill = .random };

/// Every count is a multiple of four (binary_vec moves four elements per 8-byte load) and every
/// offset is 256-byte aligned (128 bf16 elements), the largest minStorageBufferOffsetAlignment
/// Vulkan allows.

// Add classes pairs: 0x0001..0x007f are bf16 SUBNORMALS (0x0080 is the
// smallest NORMAL). A device that flushes subnormals to zero fails these rows honestly; denormal
// preservation is NOT claimed without a GPU result.
pub const add_cases = [_]AddCase{
    .{ .name = "random 512", .elems = 512, .at = .{ .a = 128, .b = 256, .out = 384 } },
    .{ .name = "cancel 512", .elems = 512, .at = .{ .a = 128, .b = 256, .out = 384 }, .fill = .cancel },
    .{ .name = "classes 20", .elems = 20, .at = .{ .a = 128, .b = 256, .out = 384 }, .fill = .classes },
};

/// Planted classes-case inputs; every pair is exercised (elems == len).
const class_pairs = [_][2]u16{
    .{ 0x7f80, 0x3f80 }, // +inf + 1 -> +inf
    .{ 0xff80, 0x3f80 }, // -inf + 1 -> -inf
    .{ 0x7f80, 0xff80 }, // +inf + -inf -> nan
    .{ 0x7fc0, 0x3f80 }, // nan + 1 -> nan
    .{ 0x0000, 0x8000 }, // +0 + -0 -> +0 (round to nearest)
    .{ 0x8000, 0x8000 }, // -0 + -0 -> -0
    .{ 0x7f7f, 0x7f7f }, // bf16 max + bf16 max -> f32 overflow -> +inf
    .{ 0xff7f, 0xff7f }, // both-min -> -inf
    .{ 0x3f80, 0xbf80 }, // 1 + -1 -> +0
    .{ 0x4000, 0xc000 }, // 2 + -2 -> +0
    .{ 0x3f80, 0x3f80 }, // 1 + 1 -> 2
    .{ 0x3f81, 0xbd80 }, // 1.0078125 + -0.0625, exact in f32
    .{ 0x0001, 0x0001 }, // smallest bf16 SUBNORMAL doubled -> 0x0002 (honest fail on flush-to-zero)
    .{ 0x0001, 0x0002 }, // 1 + 2 subnormal units -> 0x0003
    .{ 0x0001, 0x8000 }, // subnormal + -0 -> unchanged
    .{ 0x0080, 0x0080 }, // smallest NORMAL doubled -> 0x0100
    .{ 0x0080, 0x8000 }, // smallest normal + -0 -> unchanged
    .{ 0x4f00, 0xb080 }, // 2^31 - 2^-29, huge spread: f32 rounds to 2^31
    .{ 0x42f6, 0xc2f6 }, // ±123 -> +0
    .{ 0x3f00, 0xc000 }, // 0.5 + -2 -> -1.5 exact
};

pub const Add = struct {
    a: std.mem.Allocator,
    case: AddCase,
    /// Bindings as uploaded: shells poisoned on both sides of the data window.
    a_bits: []u16,
    b_bits: []u16,
    out: []u16,
    /// The exact bits the shader pipeline (f32 add, one RNE bf16 store) must produce; a
    /// nonfinite result carries its class bits and is checked by class only.
    want: []u16,

    pub fn init(a: std.mem.Allocator, case: AddCase, seed: u64) !Add {
        const a_bits = try poisonedShell(a, case.at.a, case.elems);
        errdefer a.free(a_bits);
        const b_bits = try poisonedShell(a, case.at.b, case.elems);
        errdefer a.free(b_bits);
        const out = try poisonedShell(a, case.at.out, case.elems);
        errdefer a.free(out);
        const want = try a.alloc(u16, case.elems);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..case.elems) |i| {
            const pair = switch (case.fill) {
                .random => [2]u16{ sampleBits(rng, std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))), sampleBits(rng, std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 3)) - 1))) },
                // v and -(v - k*ulp(v)): a bf16-exact difference, sum tiny but exact
                .cancel => blk: {
                    const vbits = sampleBits(rng, 1);
                    const v = c.fromBf16(vbits);
                    const k: f32 = @floatFromInt(i % 3);
                    const ulp = std.math.pow(f32, 2, @floor(std.math.log2(@abs(v))) - 7);
                    break :blk [2]u16{ vbits, c.toBf16(-(v - k * ulp)) };
                },
                .classes => class_pairs[i % class_pairs.len],
            };
            a_bits[case.at.a + i] = pair[0];
            b_bits[case.at.b + i] = pair[1];
            // The shader's own pipeline, evaluated on the host: f32 add, one RNE bf16 store.
            want[i] = c.toBf16(c.fromBf16(pair[0]) + c.fromBf16(pair[1]));
        }
        return .{ .a = a, .case = case, .a_bits = a_bits, .b_bits = b_bits, .out = out, .want = want };
    }

    pub fn deinit(f: Add) void {
        f.a.free(f.a_bits);
        f.a.free(f.b_bits);
        f.a.free(f.out);
        f.a.free(f.want);
    }

    /// got: the whole out shell read back; guards keep their poison, data bits equal want exactly.
    pub fn check(f: Add, got: []const u16) !void {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(got, at, f.case.elems);
        for (f.want, 0..) |w, i| {
            const g = got[at + i];
            if (classOfBits(g) != classOfBits(w)) return error.NonFiniteMismatch;
            if (std.math.isFinite(c.fromBf16(w)) and g != w) return error.BitMismatch;
        }
    }
};

// ---------------------------------------------------------------------------
// SwiGLU: one bf16 store plus a documented f32 budget
// ---------------------------------------------------------------------------

/// ops = 8: a FIXED BRING-UP ACCEPTANCE BUDGET for the f32 chain exp -> 1+ -> divide -> multiply.
/// exp is transcendental: no per-op error bound is proven here and none is guaranteed by the
/// Vulkan spec; 8*2^-24 is an acceptance choice, and boundExact adds the output's own bf16
/// rounding on top. This is not a full kernel-contract bound.
const swiglu_spec: c.Spec = .{ .name = "vulkan swiglu bf16 (ASH_QWEN35 one-store)", .accum = .none, .depth = no_depth, .ops = 8, .out = .bf16, .served = &[_]c.Shape{} };
fn no_depth(_: c.Shape) usize {
    return 0; // elementwise: boundExact never calls depth
}

pub const SwigluCase = struct { name: []const u8, elems: usize, at: Offsets = .{}, controls: bool = false };

pub const swiglu_cases = [_]SwigluCase{
    .{ .name = "random 512", .elems = 512, .at = .{ .a = 128, .b = 256, .out = 384 } },
    .{ .name = "controls 16", .elems = 16, .at = .{ .a = 128, .b = 256, .out = 384 }, .controls = true },
};

/// The controls: planted nan/+inf/-inf gates and signed zeros, up mostly 1 with two sign flips.
const swiglu_gate_controls = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x3f80, 0xbf80, 0x41a0, 0xc1a0, 0x3f00, 0xbf00, 0x4000, 0xc000, 0x3e80, 0x3d00, 0x4800 };
const swiglu_up_controls = [_]u16{ 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0xbf80, 0x3f80, 0xc000, 0x3f80, 0xbf80, 0x3f80, 0x3f80 };

pub const Swiglu = struct {
    a: std.mem.Allocator,
    case: SwigluCase,
    gate: []u16,
    up: []u16,
    out: []u16,
    /// The f64 reference g / (1 + exp(-g)) * up of each output (finite: bound-checked; else class).
    want: []f64,

    pub fn init(a: std.mem.Allocator, case: SwigluCase, seed: u64) !Swiglu {
        const gate = try poisonedShell(a, case.at.a, case.elems);
        errdefer a.free(gate);
        const up = try poisonedShell(a, case.at.b, case.elems);
        errdefer a.free(up);
        const out = try poisonedShell(a, case.at.out, case.elems);
        errdefer a.free(out);
        const want = try a.alloc(f64, case.elems);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..case.elems) |i| {
            const g: u16 = if (case.controls) swiglu_gate_controls[i % swiglu_gate_controls.len] else sampleBits(rng, std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 4)))));
            const u: u16 = if (case.controls) swiglu_up_controls[i % swiglu_up_controls.len] else sampleBits(rng, 1);
            gate[case.at.a + i] = g;
            up[case.at.b + i] = u;
            const g64: f64 = c.fromBf16(g);
            const u64v: f64 = c.fromBf16(u);
            want[i] = g64 / (1 + @exp(-g64)) * u64v;
        }
        return .{ .a = a, .case = case, .gate = gate, .up = up, .out = out, .want = want };
    }

    pub fn deinit(f: Swiglu) void {
        f.a.free(f.gate);
        f.a.free(f.up);
        f.a.free(f.out);
        f.a.free(f.want);
    }

    /// Finite wants: within boundExact (one bf16 rounding + the fixed f32 budget). want == 0
    /// demands an exactly-zero output. Nonfinite wants: the output's class must equal the
    /// reference's class exactly.
    pub fn check(f: Swiglu, got: []const u16) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(got, at, f.case.elems);
        var r: Report = .{};
        for (f.want, 0..) |w, i| {
            const g = got[at + i];
            const gf = c.fromBf16(g);
            if (!std.math.isFinite(w) or !std.math.isFinite(gf)) {
                if (c.classOf(gf) != c.classOf(w)) return error.NonFiniteMismatch;
                continue;
            }
            if (w == 0) {
                if (gf != 0) return error.OutOfBound; // signed zeros: never a small nonzero
                continue;
            }
            const err = @abs(gf - w);
            const bound = c.boundExact(swiglu_spec, w);
            if (!(std.math.isFinite(bound) and bound > 0)) return error.BadBound;
            const ratio = err / bound;
            if (ratio > r.worst) {
                r = .{ .worst = ratio, .row = i, .col = i };
            }
        }
        if (r.worst > 1) return error.OutOfBound;
        return r;
    }
};

// ---------------------------------------------------------------------------
// RMSNorm: contract bound with the reduction budget named
// ---------------------------------------------------------------------------

/// depth = 20 f32 reduction roundings, CONSERVATIVE at every served width: at 2048 the strided
/// partials take 8 adds and the 256-wide tree 8 levels (16 roundings); at 256 only 1 + 8. The
/// budget applies to the output RELATIVELY (see check): the sum is positive, so gamma(20) bounds
/// its relative error in any order. ops = 4: inversesqrt (<= 2 ulp) and the two f32 multiplies of
/// weight * (value * norm).
const rms_depth = 20;
const rms_spec: c.Spec = .{
    .name = "vulkan fast_norm bf16 rms (ASH_QWEN35 weight*(x*inv))",
    .accum = .f32,
    .depth = rmsDepth,
    .ops = 4,
    .out = .bf16,
    .served = &rms_served,
};
fn rmsDepth(_: c.Shape) usize {
    return rms_depth;
}
const rms_served = [_]c.Shape{
    .{ .rows = 1, .n = 1, .k = 2048 },
    .{ .rows = 4, .n = 1, .k = 2048 },
    .{ .rows = 2, .n = 1, .k = 2048 },
    .{ .rows = 2, .n = 1, .k = 256 },
};

pub const RmsFill = enum { flat, dynamic };

pub const RmsCase = struct {
    name: []const u8,
    rows: usize,
    width: usize,
    fill: RmsFill = .flat,
    /// The last row is all zeros: mean square 0, norm = rsqrt(eps), output exactly zero.
    zero_row: bool = false,
    at: RmsOffsets = .{},
};

/// 256-byte-aligned element offsets (multiples of 128).
pub const RmsOffsets = struct { x: usize = 0, gamma: usize = 0, out: usize = 0 };

pub const rms_cases = [_]RmsCase{
    .{ .name = "1x2048", .rows = 1, .width = 2048 },
    .{ .name = "4x2048 offsets zero row", .rows = 4, .width = 2048, .zero_row = true, .at = .{ .x = 128, .gamma = 0, .out = 384 } },
    .{ .name = "2x2048 dynamic range", .rows = 2, .width = 2048, .fill = .dynamic },
    // The query/key head_norm width Qwen3.5-2B actually serves, at the same aligned offsets.
    .{ .name = "2x256 head_norm offsets", .rows = 2, .width = 256, .at = .{ .x = 128, .gamma = 0, .out = 384 } },
};

pub const rms_eps: f32 = 1e-6;

pub const Rms = struct {
    a: std.mem.Allocator,
    case: RmsCase,
    x: []u16,
    gamma: []u16,
    out: []u16,
    /// gamma64 * (x64 * rsqrt(mean(x^2) + eps)) per element, in f64, independent of the kernel.
    want: []f64,

    pub fn init(a: std.mem.Allocator, case: RmsCase, seed: u64) !Rms {
        const x = try poisonedShell(a, case.at.x, case.rows * case.width);
        errdefer a.free(x);
        const gamma = try poisonedShell(a, case.at.gamma, case.width);
        errdefer a.free(gamma);
        const out = try poisonedShell(a, case.at.out, case.rows * case.width);
        errdefer a.free(out);
        const want = try a.alloc(f64, case.rows * case.width);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..case.width) |i| gamma[case.at.gamma + i] = sampleBits(rng, 1);
        for (0..case.rows) |row| {
            const base = case.at.x + row * case.width;
            var sum_sq: f64 = 0;
            for (0..case.width) |i| {
                const bits: u16 = if (case.zero_row and row + 1 == case.rows) 0x0000 else switch (case.fill) {
                    .flat => sampleBits(rng, 1),
                    // Large and tiny magnitudes interleaved: the norm is dominated by the large
                    // squares while the small entries must still round on their own scale.
                    .dynamic => sampleBits(rng, if (i % 4 == 0) 0x1p10 else if (i % 4 == 2) 0x1p-10 else 1),
                };
                x[base + i] = bits;
                sum_sq += @as(f64, c.fromBf16(bits)) * @as(f64, c.fromBf16(bits));
            }
            const n: f64 = 1 / @sqrt(sum_sq / @as(f64, @floatFromInt(case.width)) + rms_eps);
            for (0..case.width) |i| {
                const x64: f64 = c.fromBf16(x[base + i]);
                const g64: f64 = c.fromBf16(gamma[case.at.gamma + i]);
                want[row * case.width + i] = g64 * (x64 * n);
            }
        }
        for (want) |w| if (!std.math.isFinite(w)) return error.NonFiniteReference;
        return .{ .a = a, .case = case, .x = x, .gamma = gamma, .out = out, .want = want };
    }

    pub fn deinit(f: Rms) void {
        f.a.free(f.x);
        f.a.free(f.gamma);
        f.a.free(f.out);
        f.a.free(f.want);
    }

    /// Fixed bring-up budget, NOT a Vulkan-spec proof: |got - want| <= c.bound with the output's
    /// own magnitude as the summed magnitude. The reduction sums squares (positive), so its
    /// relative error stays under gamma(depth) in any order; the norm divides by that sum's
    /// square root (relative sensitivity ~1/2, kept at gamma(20) conservatively), so the output
    /// error is relative for every element and never inherits the sum's absolute scale. A
    /// deliberately-absolute sum-of-squares budget here would allow ~1e3 of output error on
    /// dynamic rows and nonzero outputs on the zero row — that was wrong and is not this.
    pub fn check(f: Rms, got: []const u16) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(got, at, f.case.rows * f.case.width);
        var r: Report = .{};
        for (f.want, 0..) |w, i| {
            const g = got[at + i];
            const gf = c.fromBf16(g);
            if (!std.math.isFinite(gf)) return error.NonFiniteMismatch;
            if (w == 0) {
                if (gf != 0) return error.OutOfBound; // the zero row: ±0 only, never small nonzero
                continue;
            }
            const err = @abs(gf - w);
            const bound = c.bound(rms_spec, .{ .rows = f.case.rows, .n = 1, .k = f.case.width }, w, @abs(w));
            if (!(std.math.isFinite(bound) and bound > 0)) return error.BadBound;
            const ratio = err / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .row = i / f.case.width, .col = i % f.case.width };
        }
        if (r.worst > 1) return error.OutOfBound;
        return r;
    }
};

const testing = std.testing;

test "add wants are the shader pipeline: f32 add then one RNE bf16 store" {
    for (add_cases) |case| {
        const f = try Add.init(testing.allocator, case, 0x5702);
        defer f.deinit();
        for (0..f.case.elems) |i| {
            const want = c.toBf16(c.fromBf16(f.a_bits[f.case.at.a + i]) + c.fromBf16(f.b_bits[f.case.at.b + i]));
            try testing.expectEqual(want, f.want[i]);
        }
    }
}

test "add cancel case produces exact tiny sums and the classes case plants every nonfinite class" {
    const f = try Add.init(testing.allocator, add_cases[1], 0x5703);
    defer f.deinit();
    var zeros: usize = 0;
    for (0..f.case.elems) |i| {
        const sum = c.fromBf16(f.a_bits[f.case.at.a + i]) + c.fromBf16(f.b_bits[f.case.at.b + i]);
        if (@abs(c.fromBf16(f.want[i]) - sum) == 0) zeros += 1;
    }
    try testing.expect(zeros > 0); // v + -(v - 0*ulp) pairs cancel exactly
    const g = try Add.init(testing.allocator, add_cases[2], 0x5704);
    defer g.deinit();
    var classes: std.EnumSet(c.Class) = .empty;
    for (g.want) |w| classes.insert(classOfBits(w));
    try testing.expect(classes.contains(.nan));
    try testing.expect(classes.contains(.pos_inf));
    try testing.expect(classes.contains(.neg_inf));
    try testing.expect(classes.contains(.finite));
}

test "swiglu controls plant nan, both infinities and signed zeros, and references stay finite elsewhere" {
    for (swiglu_cases) |case| {
        const f = try Swiglu.init(testing.allocator, case, 0x5705);
        defer f.deinit();
        for (f.want, 0..) |w, i| {
            if (case.controls and i < swiglu_gate_controls.len) {
                const gate = c.fromBf16(f.gate[f.case.at.a + i]);
                if (!std.math.isFinite(gate) or gate == 0) continue; // class/plant entries
            }
            if (!case.controls) try testing.expect(std.math.isFinite(w));
        }
    }
    const g = try Swiglu.init(testing.allocator, swiglu_cases[1], 0x5706);
    defer g.deinit();
    try testing.expect(std.math.isNan(g.want[0])); // nan gate
    try testing.expect(std.math.isInf(g.want[1])); // +inf gate, up 1
    try testing.expect(std.math.isNan(g.want[2])); // -inf gate -> -inf/inf
    try testing.expect(g.want[3] == 0); // +0 gate -> exactly zero
    try testing.expect(std.math.signbit(g.want[4])); // -0 gate keeps its zero sign
}

test "rms references are finite and the zero row wants exact zeros" {
    for (rms_cases) |case| {
        const f = try Rms.init(testing.allocator, case, 0x5707);
        defer f.deinit();
        for (f.want) |w| try testing.expect(std.math.isFinite(w));
        if (case.zero_row) {
            const zero_at = (case.rows - 1) * case.width;
            for (f.want[zero_at..]) |w| try testing.expectEqual(@as(f64, 0), w);
        }
    }
}

test "the checkers reject a deliberately wrong finite output, not only accept good ones" {
    // RMS dynamic range: want + 0.1 is far outside the output-relative budget.
    {
        const f = try Rms.init(testing.allocator, rms_cases[2], 0x5710);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        const at = f.case.width / 2;
        bits[f.case.at.out + at] = c.toBf16(@floatCast(f.want[at] + 0.1));
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // RMS zero row: one bf16 subnormal unit of output must be refused outright.
    {
        const f = try Rms.init(testing.allocator, rms_cases[1], 0x5711);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        bits[f.case.at.out + (f.case.rows - 1) * f.case.width] = 0x0001;
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // SwiGLU: a wrong finite output must be refused too.
    {
        const f = try Swiglu.init(testing.allocator, swiglu_cases[0], 0x5712);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        bits[f.case.at.out] = c.toBf16(@floatCast(f.want[0] + 0.1));
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // Residual add: a wrong finite output is a bit mismatch.
    {
        const f = try Add.init(testing.allocator, add_cases[0], 0x5713);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
        try f.check(bits);
        bits[f.case.at.out] = c.toBf16(@floatCast(c.fromBf16(f.want[0]) + 0.1));
        try testing.expectError(error.BitMismatch, f.check(bits));
    }
}

fn addAllocRun(a: std.mem.Allocator, case: AddCase) !void {
    const f = try Add.init(a, case, 0x5714);
    f.deinit();
}

fn swigluAllocRun(a: std.mem.Allocator, case: SwigluCase) !void {
    const f = try Swiglu.init(a, case, 0x5715);
    f.deinit();
}

fn rmsAllocRun(a: std.mem.Allocator, case: RmsCase) !void {
    const f = try Rms.init(a, case, 0x5716);
    f.deinit();
}

test "every fixture init survives each single allocation failure without leaking" {
    // checkAllAllocationFailures counts init's allocations, then fails each one in turn and
    // requires error.OutOfMemory with every earlier byte freed; the staged errdefers pay for this.
    try std.testing.checkAllAllocationFailures(testing.allocator, addAllocRun, .{add_cases[0]});
    try std.testing.checkAllAllocationFailures(testing.allocator, addAllocRun, .{add_cases[2]});
    try std.testing.checkAllAllocationFailures(testing.allocator, swigluAllocRun, .{swiglu_cases[0]});
    try std.testing.checkAllAllocationFailures(testing.allocator, swigluAllocRun, .{swiglu_cases[1]});
    for (rms_cases) |case| try std.testing.checkAllAllocationFailures(testing.allocator, rmsAllocRun, .{case});
}

test "guards sit poisoned outside every data window" {
    const f = try Rms.init(testing.allocator, rms_cases[1], 0x5708);
    defer f.deinit();
    for (f.x, 0..) |g, i| {
        if (i >= f.case.at.x and i < f.case.at.x + f.case.rows * f.case.width) continue;
        try testing.expectEqual(poison16, g);
    }
    for (f.out, 0..) |g, i| {
        if (i >= f.case.at.out and i < f.case.at.out + f.case.rows * f.case.width) continue;
        try testing.expectEqual(poison16, g);
    }
}
