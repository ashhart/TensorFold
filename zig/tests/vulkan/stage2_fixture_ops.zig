// The stage 2 fixture's cases and checkers: residual add, SwiGLU and RMSNorm against the Metal-oracle semantics.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;

/// Poisoned elements after each binding's data, so an overrun reads poison rather than past the buffer.
pub const tail = 16;
pub const poison16: u16 = 0xffff;

pub fn classOfBits(bits: u16) c.Class {
    return c.classOf(c.fromBf16(bits));
}

/// Human name for the value class of raw bf16 bits: sign, then zero/subnormal/normal/inf/nan.
pub fn classify(bits: u16) []const u8 {
    const neg = bits & 0x8000 != 0;
    const e = (bits >> 7) & 0xff;
    const m = bits & 0x7f;
    return switch (e) {
        0xff => if (m == 0) (if (neg) "-inf" else "+inf") else (if (neg) "-nan" else "+nan"),
        0 => if (m == 0) (if (neg) "-zero" else "+zero") else (if (neg) "-subnormal" else "+subnormal"),
        else => if (neg) "-normal" else "+normal",
    };
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

// ---- Residual add: the shader pipeline (f32 add + one RNE bf16 store), bit for bit ----

pub const AddFill = enum { random, cancel, classes };

pub const AddCase = struct { name: []const u8, elems: usize, at: Offsets = .{}, fill: AddFill = .random };

// Counts %4==0 (binary_vec moves 4/8-byte load); offsets 256B aligned, the largest minStorageBufferOffsetAlignment.

// 0x0001..0x007f SUBNORMALs (0x0080 smallest NORMAL); measured properties, never vendor names, allow signed zero.
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
    // The exact bits the shader pipeline (f32 add, one RNE bf16 store) must produce; nonfinite by class only.
    want: []u16,
    allow_subnormal_flush: bool = false,

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

    // First mismatch under the exact check (class-vs-bit), or null when the window matches; guard failures are errors.
    pub const Mismatch = struct { kind: error{ NonFiniteMismatch, BitMismatch }, index: usize, a: u16, b: u16, want: u16, got: u16 };

    pub fn firstMismatch(f: Add, got: []const u16) !?Mismatch {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(got, at, f.case.elems);
        for (f.want, 0..) |w, i| {
            const g = got[at + i];
            if (classOfBits(g) != classOfBits(w)) return .{ .kind = error.NonFiniteMismatch, .index = i, .a = f.a_bits[f.case.at.a + i], .b = f.b_bits[f.case.at.b + i], .want = w, .got = g };
            const flushed = f.allow_subnormal_flush and f.case.fill == .classes and
                (w & 0x7f80) == 0 and (w & 0x007f) != 0 and g == (w & 0x8000);
            if (std.math.isFinite(c.fromBf16(w)) and g != w and !flushed) return .{ .kind = error.BitMismatch, .index = i, .a = f.a_bits[f.case.at.a + i], .b = f.b_bits[f.case.at.b + i], .want = w, .got = g };
        }
        return null;
    }

    /// got: the whole out shell read back; guards keep their poison, data bits equal want exactly.
    pub fn check(f: Add, got: []const u16) !void {
        if (try f.firstMismatch(got)) |m| return m.kind;
    }
};

// ---- SwiGLU: one bf16 store plus a documented f32 budget ----

// The f32 exp/1+/divide/multiply chain uses 8 ops and a fixed 8*2^-24 budget; exp has no Vulkan-spec per-op bound.
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
pub const swiglu_gate_controls = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x3f80, 0xbf80, 0x41a0, 0xc1a0, 0x3f00, 0xbf00, 0x4000, 0xc000, 0x3e80, 0x3d00, 0x4800 };
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

    // Finite wants: within boundExact (bf16 rounding + f32 budget); want == 0 demands exactly-zero; nonfinite: classes.
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

// ---- RMSNorm: contract bound with the reduction budget named ----

// depth = 20: 2048 takes 8 adds + 8 levels (16), 256 1 + 8; the sum is positive, so gamma(20) is relative in any order.
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
                    // Large and tiny magnitudes interleaved: large squares dominate, small ones round on their scale.
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

    // Fixed budget, NOT a spec proof: |got - want| <= c.bound with the output's own magnitude, gamma(depth) relative.
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
