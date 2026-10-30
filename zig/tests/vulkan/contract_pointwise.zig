// Pointwise contract: stages 2-4 closures as host fixtures; preserve add checked STRICTLY (a flushing device fails).
const std = @import("std");
const contract = @import("contract");
const stage2 = @import("stage2_fixture.zig");
const shaders2 = @import("stage2_shaders");
const vk = @import("vulkan");

const c = contract.spec;
pub const Report = stage2.Report;
pub const classify = stage2.classify;

/// Element offsets (elements of the binding's kind; 256-byte alignment) of each binding's data.
pub const Offsets = stage2.Offsets;

pub const tail = stage2.tail;
pub const poison16 = stage2.poison16;

fn poisonedShell16(a: std.mem.Allocator, at: usize, len: usize) ![]u16 {
    const buf = try a.alloc(u16, at + len + tail);
    @memset(buf, poison16);
    return buf;
}

fn guardIntact(shell: []const u16, at: usize, len: usize, poison: u16) !void {
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

// ---- Residual add, preserved: add_preserve_bf16 (binary_vec.comp + -DPRESERVE_F32), checked STRICTLY ----

// Preserved add: stage 2's three shapes, 256B-aligned; norm_mlp selects it where the device declares DenormPreserve.
pub const PreserveCase = stage2.AddCase;
pub const preserve_cases = [_]PreserveCase{
    .{ .name = "random 512", .elems = 512, .at = .{ .a = 128, .b = 256, .out = 384 } },
    .{ .name = "cancel 512", .elems = 512, .at = .{ .a = 128, .b = 256, .out = 384 }, .fill = .cancel },
    .{ .name = "classes 20", .elems = 20, .at = .{ .a = 128, .b = 256, .out = 384 }, .fill = .classes },
};

// Preserve gate reuses stage 2's add fixture; allow_subnormal_flush false: a flushing device fails the subnormal rows.
pub const PreserveAdd = stage2.Add;

// ---- Attention output gate: one mid bf16 rounding of the sigmoid, one bf16 store ----

// ops = 4 (exp, 1+, divide, ctx multiply); mids = 1: sigmoid bf16 rounding, error <= 2^-9|s|; a budget, not a proof.
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

// Gate controls: planted nan/±inf/±0, model-range ends, a bf16 subnormal; |gate| <= 100 keeps exp(-gate) in f32 range.
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
    // qraw bits: binding 1; element d of row r at r*4096 + (d/256)*512 + 256 + d%256 (interleaved layout).
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
            // ctx is 1, except -1 under the -inf gate (signed zero) and zero at 511: -0, 1023: +0; zeros keep the sign.
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

    // Finite wants within boundExact; a zero want (ctx zero, gate -inf) demands an exactly-zero signed output.
    pub fn check(f: Gate, got: []const u16) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(got, at, f.case.rows * gate_row_elems, poison16);
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

// ---- Gated RMSNorm: tokens x 16 head rows of 128, the [128] gain shared by every head, plus one mid rounding ----

// depth = 12 (1 + 8 levels + 1 divide = 10, 2 margin); ops = 10; mids = 12 (2^-8 of |normed|; |z| >= 0.25 floor).
const gated_depth = 12;
const gated_spec: c.Spec = .{
    .name = "vulkan fast_norm_gated bf16 per-head gated rms (silu(z)*bf16(x*(inv*w)), head rows of 128)",
    .accum = .f32,
    .depth = gatedDepth,
    .ops = 10,
    .mids = 12,
    .out = .bf16,
    .served = &[_]c.Shape{ .{ .rows = 1, .n = 1, .k = gated_head_dim }, .{ .rows = 2, .n = 1, .k = gated_head_dim } },
};
fn gatedDepth(_: c.Shape) usize {
    return gated_depth;
}

// gdn.Model geometry: 16 head rows of 128 a token (2048 bf16 elements); one [128] gain shared by every head.
pub const gated_heads: usize = 16;
pub const gated_head_dim: usize = 128;
pub const gated_width: usize = gated_heads * gated_head_dim;
pub const gated_eps: f32 = 1e-6;

pub const GatedOffsets = struct { x: usize = 0, weight: usize = 0, gate: usize = 0, out: usize = 0 };

pub const GatedCase = struct {
    name: []const u8,
    tokens: usize,
    fill: stage2.RmsFill = .flat,
    // The last token's 16 head rows are all zeros: mean square 0, norm = rsqrt(eps), outputs exactly signed zero.
    zero_token: bool = false,
    // Token 0 plants z controls; tokens 1 and 2 poison x[head 0][0] with +inf and nan; the per-head norm must not leak.
    controls: bool = false,
    at: GatedOffsets = .{},
};

// Served tokens: one decode row and a prefill chunk (rows = 1, max_prefill_rows = 32), plus controls on three tokens.
pub const gated_cases = [_]GatedCase{
    .{ .name = "1x16x128", .tokens = 1, .at = .{ .x = 128, .weight = 0, .gate = 256, .out = 384 } },
    .{ .name = "32x16x128 dynamic zero token", .tokens = 32, .fill = .dynamic, .zero_token = true, .at = .{ .x = 128, .weight = 0, .gate = 256, .out = 384 } },
    .{ .name = "controls 3x16x128", .tokens = 3, .controls = true, .at = .{ .x = 128, .weight = 0, .gate = 256, .out = 384 } },
};

// z controls at token 0, head 0: |silu| >= 0.109, zeros from z = ±0, classes from nan/±inf; z = -inf: silu = nan.
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
    // silu64(z) * widen(bf16(x*(n*w))), f64, mid modeled; w the shared [128] gain; nonfinite only at planted controls.
    want: []f64,

    pub fn init(a: std.mem.Allocator, case: GatedCase, seed: u64) !GatedNorm {
        const elems = case.tokens * gated_width;
        const x = try poisonedShell16(a, case.at.x, elems);
        errdefer a.free(x);
        const weight = try poisonedShell16(a, case.at.weight, gated_head_dim);
        errdefer a.free(weight);
        const z = try poisonedShell16(a, case.at.gate, elems);
        errdefer a.free(z);
        const out = try poisonedShell16(a, case.at.out, elems);
        errdefer a.free(out);
        const want = try a.alloc(f64, elems);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..gated_head_dim) |i| weight[case.at.weight + i] = sampleBits(rng, 1);
        for (0..case.tokens) |token| {
            const base = case.at.x + token * gated_width;
            const zbase = case.at.gate + token * gated_width;
            for (0..gated_heads) |head| {
                const head_base = base + head * gated_head_dim;
                const zhead = zbase + head * gated_head_dim;
                var sum_sq: f64 = 0;
                for (0..gated_head_dim) |i| {
                    var bits: u16 = if (case.zero_token and token + 1 == case.tokens) 0x0000 else switch (case.fill) {
                        .flat => sampleBits(rng, 1),
                        .dynamic => sampleBits(rng, if (i % 4 == 0) 0x1p10 else if (i % 4 == 2) 0x1p-10 else 1),
                    };
                    if (case.controls and token == 1 and head == 0 and i == 0) bits = 0x7f80; // +inf: head 0's sum goes infinite, norm 0
                    if (case.controls and token == 2 and head == 0 and i == 0) bits = 0x7fc0; // nan: head 0's sums go nan
                    x[head_base + i] = bits;
                    sum_sq += @as(f64, c.fromBf16(bits)) * @as(f64, c.fromBf16(bits));
                }
                const n: f64 = 1 / @sqrt(sum_sq / @as(f64, @floatFromInt(gated_head_dim)) + gated_eps);
                for (0..gated_head_dim) |i| {
                    const zb: u16 = if (case.controls and token == 0 and head == 0 and i < gated_z_controls.len) gated_z_controls[i] else if (case.controls and token == 1 and head == 0 and i == 1) 0x7fc0 else gatedGateBits(rng);
                    z[zhead + i] = zb;
                    // Shader's association: value * (norm * weight), one mid bf16 rounding; weight rides the column.
                    const normed = c.toBf16(@floatCast(c.fromBf16(x[head_base + i]) * (n * c.fromBf16(weight[case.at.weight + i]))));
                    want[token * gated_width + head * gated_head_dim + i] = silu64(c.fromBf16(zb)) * c.fromBf16(normed);
                }
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

    // Rows-family budget (gamma(depth) of the output's magnitude) plus mids/ops; a zero want demands a signed zero.
    pub fn check(f: GatedNorm, got: []const u16) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        try guardIntact(got, at, f.case.tokens * gated_width, poison16);
        var r: Report = .{};
        const shape: c.Shape = .{ .rows = f.case.tokens * gated_heads, .n = 1, .k = gated_head_dim };
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

// ---- Host tests: the references and the checkers, no GPU ----

test "preserve wants are the stage 2 pipeline bits, and the flushed subnormal rows stay refused" {
    for (preserve_cases) |case| {
        const f = try PreserveAdd.init(testing.allocator, case, 0x58a2);
        defer f.deinit();
        try testing.expect(!f.allow_subnormal_flush); // the preserve gate never qualifies a flush
        for (0..f.case.elems) |i| {
            const sum = c.fromBf16(f.a_bits[f.case.at.a + i]) + c.fromBf16(f.b_bits[f.case.at.b + i]);
            try testing.expectEqual(c.toBf16(sum), f.want[i]); // f32 add, one RNE bf16 store
        }
    }
    // The classes case carries the subnormal rows (0x0001 + 0x0001 wants 0x0002); a signed zero is a BitMismatch.
    const g = try PreserveAdd.init(testing.allocator, preserve_cases[2], 0x58a3);
    defer g.deinit();
    try testing.expectEqual(@as(u16, 0x0002), g.want[12]);
    const bits = try testing.allocator.dupe(u16, g.out);
    defer testing.allocator.free(bits);
    @memcpy(bits[g.case.at.out..][0..g.want.len], g.want);
    try testing.expectEqual(@as(?PreserveAdd.Mismatch, null), try g.firstMismatch(bits));
    bits[g.case.at.out + 12] = 0x0000;
    const m = (try g.firstMismatch(bits)).?;
    try testing.expectEqual(error.BitMismatch, m.kind);
    try testing.expectEqual(@as(u16, 0x0002), m.want);
    try testing.expectError(error.BitMismatch, g.check(bits));
}

test "preserve classes plants every nonfinite class and the cancel case produces exact tiny sums" {
    const g = try PreserveAdd.init(testing.allocator, preserve_cases[2], 0x58a4);
    defer g.deinit();
    var classes: std.EnumSet(c.Class) = .empty;
    for (g.want) |w| classes.insert(c.classOf(c.fromBf16(w)));
    try testing.expect(classes.contains(.nan));
    try testing.expect(classes.contains(.pos_inf));
    try testing.expect(classes.contains(.neg_inf));
    try testing.expect(classes.contains(.finite));
    const f = try PreserveAdd.init(testing.allocator, preserve_cases[1], 0x58a5);
    defer f.deinit();
    var tiny: usize = 0;
    for (0..f.case.elems) |i| {
        const sum = c.fromBf16(f.a_bits[f.case.at.a + i]) + c.fromBf16(f.b_bits[f.case.at.b + i]);
        if (@abs(sum) < 0x1p-20) tiny += 1;
    }
    try testing.expect(tiny > 0); // v + -(v - 0*ulp) pairs cancel to exact zeros
}

test "preserve checker reports the first bad element and keeps named failures" {
    const f = try PreserveAdd.init(testing.allocator, preserve_cases[0], 0x58a6);
    defer f.deinit();
    const bits = try testing.allocator.dupe(u16, f.out);
    defer testing.allocator.free(bits);
    @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
    try testing.expectEqual(@as(?PreserveAdd.Mismatch, null), try f.firstMismatch(bits));
    bits[f.case.at.out + 3] +%= 1; // one bf16 ulp off: a finite bit mismatch
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
    // Gate element d of row r sits at r*4096 + (d/256)*512 + 256 + d%256; nothing else in the qraw window is written.
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

test "gated references are finite off the controls, and the zero token wants exact signed zeros" {
    for (gated_cases) |case| {
        const f = try GatedNorm.init(testing.allocator, case, 0x58ab);
        defer f.deinit();
        if (case.controls) continue;
        for (f.want) |w| try testing.expect(std.math.isFinite(w));
        if (case.zero_token) {
            const zero_at = (case.tokens - 1) * gated_width;
            for (f.want[zero_at..]) |w| try testing.expectEqual(@as(f64, 0), @abs(w));
        }
    }
    // The controls: token 0's z give nan, both infinities and signed zeros; tokens 1 and 2 poison only head 0's row.
    const g = try GatedNorm.init(testing.allocator, gated_cases[2], 0x58ac);
    defer g.deinit();
    try testing.expect(std.math.isNan(g.want[0])); // z = nan
    try testing.expect(std.math.isInf(g.want[1])); // z = +inf, silu +inf
    try testing.expect(std.math.isNan(g.want[2])); // z = -inf: silu = -inf*0 = nan
    try testing.expectEqual(@as(f64, 0), @abs(g.want[3])); // z = +0
    for (0..gated_head_dim) |i| { // token 1, head 0
        const w = g.want[gated_width + i];
        if (i == 0 or i == 1) {
            try testing.expect(std.math.isNan(w));
        } else {
            try testing.expectEqual(@as(f64, 0), @abs(w));
        }
    }
    for (gated_width + gated_head_dim..2 * gated_width) |i| // token 1, heads 1..15
        try testing.expect(std.math.isFinite(g.want[i]));
    for (2 * gated_width..2 * gated_width + gated_head_dim) |i| // token 2, head 0: all nan
        try testing.expect(std.math.isNan(g.want[i]));
    for (2 * gated_width + gated_head_dim..3 * gated_width) |i| // token 2, heads 1..15
        try testing.expect(std.math.isFinite(g.want[i]));
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
        // Doubling is many bf16 ulps off at EVERY magnitude; a fixed point (w*1.1+0.05 at -0.5) bf16 collapses onto.
        bits[f.case.at.out + gated_width / 2] = c.toBf16(@floatCast(f.want[gated_width / 2] * 2));
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // Zero token: one bf16 subnormal unit where an exact zero belongs is refused.
    {
        const f = try GatedNorm.init(testing.allocator, gated_cases[1], 0x58ae);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        bits[f.case.at.out + (f.case.tokens - 1) * gated_width] = 0x0001;
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

test "the production plan accepts every gated case, and the old 2048-gain shape is refused" {
    for (gated_cases) |case| {
        const elems = case.tokens * gated_width;
        const plan = try vk.gdn.gatedNormPlan(elems, elems, elems, gated_head_dim, case.tokens, gated_eps);
        try testing.expectEqual(case.tokens * gated_heads, plan.rows); // one head row a workgroup
    }
    // The old runner's [2048] weight is what the plan refuses: the production gain is the [128] head row.
    try testing.expectError(error.Invalid, vk.gdn.gatedNormPlan(gated_width, gated_width, gated_width, gated_width, 1, gated_eps));
    // A wrong element count and a bad eps are refused too.
    try testing.expectError(error.Invalid, vk.gdn.gatedNormPlan(gated_width + 2, gated_width, gated_width, gated_head_dim, 1, gated_eps));
    try testing.expectError(error.Invalid, vk.gdn.gatedNormPlan(gated_width, gated_width, gated_width, gated_head_dim, 1, 0));
}

test "gated fixture element counts are the production geometry, guards outside" {
    for (gated_cases) |case| {
        const f = try GatedNorm.init(testing.allocator, case, 0x58b6);
        defer f.deinit();
        const elems = case.tokens * gated_width;
        try testing.expectEqual(case.at.x + elems + tail, f.x.len);
        try testing.expectEqual(case.at.gate + elems + tail, f.z.len);
        try testing.expectEqual(case.at.out + elems + tail, f.out.len);
        try testing.expectEqual(case.at.weight + gated_head_dim + tail, f.weight.len);
        try testing.expectEqual(elems, f.want.len);
    }
}

fn preserveAllocRun(a: std.mem.Allocator, case: PreserveCase) !void {
    (try PreserveAdd.init(a, case, 0x58b0)).deinit();
}

// OpCapability (2,17), DenormPreserve (4464): -DPRESERVE_F32 preserves; catches a no-ifdef define (the old -DUSE_BF16).
fn hasCapability(spv: []align(4) const u8, cap: u32) bool {
    const words = std.mem.bytesAsSlice(u32, spv);
    if (words.len < 5 or words[0] != 0x07230203) return false;
    for (words[1 .. words.len - 1], 1..) |w, i| {
        if (w == ((2 << 16) | 17) and words[i + 1] == cap) return true;
    }
    return false;
}

test "the preserve build requests DenormPreserve and the plain add does not" {
    try testing.expect(hasCapability(&shaders2.add_preserve_bf16, 4464));
    try testing.expect(!hasCapability(&shaders2.add_bf16, 4464));
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
        if (i >= f.case.at.x and i < f.case.at.x + f.case.tokens * gated_width) continue;
        try testing.expectEqual(poison16, g);
    }
    for (f.z, 0..) |g, i| {
        if (i >= f.case.at.gate and i < f.case.at.gate + f.case.tokens * gated_width) continue;
        try testing.expectEqual(poison16, g);
    }
    const p = try PreserveAdd.init(testing.allocator, preserve_cases[0], 0x58b4);
    defer p.deinit();
    for (p.a_bits, 0..) |g, i| {
        if (i >= p.case.at.a and i < p.case.at.a + p.case.elems) continue;
        try testing.expectEqual(poison16, g);
    }
}
