//! The CPU delta rule's inputs under the contract: the conv, SiLU, L2 norms and beta of a token row, and its decay.
const std = @import("std");
const contract = @import("contract");
const mx = @import("cpu_mixers");
const rig_ = @import("contract_rig.zig");

const k = mx.ops.kernels;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { inputs, decay, inputs_finite, inputs_writes, inputs_taps, decay_finite, decay_sign };

const hd = 128;
const heads = 16;
const hidden = heads * hd;
const conv_dim = 3 * hidden;
const taps = 4;
/// A row's outputs: q, k and v (2048 each) then the 16 betas, all bf16.
const per_row = 3 * hidden + heads;

fn conv(_: c.Shape) usize {
    return taps;
}

fn flat(_: c.Shape) usize {
    return 0;
}

/// A decode step, a verify window (one stream's segment) and a prompt chunk of token rows.
const served_inputs = [_]c.Shape{ .{ .rows = 1, .n = per_row, .k = taps }, .{ .rows = 16, .n = per_row, .k = taps }, .{ .rows = 128, .n = per_row, .k = taps } };
const served_decay = [_]c.Shape{ .{ .rows = 1, .n = heads, .k = 1 }, .{ .rows = 16, .n = heads, .k = 1 }, .{ .rows = 128, .n = heads, .k = 1 } };
const ctl_inputs = [_]c.Shape{.{ .rows = 9, .n = per_row, .k = taps }};
const ctl_decay = [_]c.Shape{.{ .rows = 9, .n = heads, .k = 1 }};

const sums_why = "the conv's four fused taps are exact on the stress integers in fp32 and in fp16 alike; the L2 norms sum in float64";

fn base(w: Which) Which {
    return switch (w) {
        .inputs, .inputs_finite, .inputs_writes, .inputs_taps => .inputs,
        .decay, .decay_finite, .decay_sign => .decay,
    };
}

pub fn spec(w: Which) c.Spec {
    const name: []const u8 = switch (w) {
        .inputs => "cpu delta inputs",
        .decay => "cpu delta decay",
        .inputs_finite => "control: delta inputs taking a non-finite activation as zero",
        .inputs_writes => "control: delta inputs writing the conv window they read",
        .inputs_taps => "control: delta inputs reading the conv taps in reverse",
        .decay_finite => "control: delta decay taking a non-finite input as zero",
        .decay_sign => "control: delta decay with the sign of a flipped",
    };
    const is_control = w != base(w);
    return if (base(w) == .inputs)
        .{ .name = name, .accum = .f32, .depth = conv, .out = .bf16, .served = if (is_control) &ctl_inputs else &served_inputs, .sums = .{ .f16 = 0, .bf16 = 0, .why = sums_why } }
    else
        .{ .name = name, .accum = .none, .depth = flat, .out = .f32, .served = if (is_control) &ctl_decay else &served_decay, .sums = .{ .f16 = 0, .bf16 = 0, .why = "one float64 expression rounded once to fp32; there is no sum" } };
}

/// The gate each control must fail.
pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .inputs, .decay => null,
        .inputs_finite, .decay_finite => .nonfinite,
        .inputs_writes => .repeat,
        .inputs_taps, .decay_sign => .shapes,
    };
}

pub const Pre = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    shape: c.Shape = undefined,
    saved: []u16 = &.{},
    qkv: []u16 = &.{},
    conv_w: []u16 = &.{},
    a_in: []u16 = &.{},
    b_in: []u16 = &.{},
    a_log: []f32 = &.{},
    dt_bias: []u16 = &.{},
    clean: []u16 = &.{},
    q: []u16 = &.{},
    kk: []u16 = &.{},
    v: []u16 = &.{},
    gate: []f32 = &.{},
    beta: []u16 = &.{},
    co: []u16 = &.{},
    out_bf16: []u16 = &.{},
    out_f32: []f32 = &.{},
    /// Per output of the inputs kernel, the conv's fp32 error carried through SiLU and the norm (float64).
    err: []f64 = &.{},

    pub fn init(a: std.mem.Allocator, which: Which) Pre {
        return .{ .a = a, .rig = .{ .a = a }, .which = which };
    }

    pub fn deinit(t: *Pre) void {
        t.drop();
        t.rig.deinit();
    }

    fn drop(t: *Pre) void {
        if (t.err.len > 0) t.a.free(t.err);
        t.err = &.{};
        t.rig.reset();
    }

    pub fn kernel(t: *Pre) g.Kernel {
        return .{ .spec = spec(t.which), .ctx = t, .vt = &vt, .control = control(t.which) };
    }

    const vt = g.hooks(Pre);

    fn takes(w: Which, shape: c.Shape) bool {
        if (shape.rows == 0 or shape.rows > 128) return false;
        return if (base(w) == .inputs) shape.n == per_row and shape.k == taps else shape.n == heads and shape.k == 1;
    }

    pub fn prepare(t: *Pre, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        if (!takes(t.which, shape)) return false;
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const saved = try t.a.alloc(u16, (taps - 1) * conv_dim);
        defer t.a.free(saved);
        const qkv = try t.a.alloc(u16, shape.rows * conv_dim);
        defer t.a.free(qkv);
        const conv_w = try t.a.alloc(u16, conv_dim * taps);
        defer t.a.free(conv_w);
        const a_in = try t.a.alloc(u16, shape.rows * heads);
        defer t.a.free(a_in);
        const b_in = try t.a.alloc(u16, shape.rows * heads);
        defer t.a.free(b_in);
        const a_log = try t.a.alloc(f32, heads);
        defer t.a.free(a_log);
        const dt_bias = try t.a.alloc(u16, heads);
        defer t.a.free(dt_bias);
        if (fill == .stress) {
            for (saved) |*x| x.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -4, 4)));
            for (qkv) |*x| x.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -4, 4)));
            for (conv_w) |*x| x.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -2, 2)));
        } else {
            for (saved) |*x| x.* = k.bf16of(rnd.floatNorm(f32));
            for (qkv) |*x| x.* = k.bf16of(rnd.floatNorm(f32));
            for (conv_w) |*x| x.* = k.bf16of(rnd.floatNorm(f32) * 0.3);
        }
        for (a_in) |*x| x.* = k.bf16of(rnd.floatNorm(f32));
        for (b_in) |*x| x.* = k.bf16of(rnd.floatNorm(f32));
        for (a_log) |*x| x.* = rnd.floatNorm(f32) * 0.5;
        for (dt_bias) |*x| x.* = k.bf16of(rnd.floatNorm(f32));
        if (fill == .nonfinite) {
            qkv[5] = k.bf16of(std.math.nan(f32));
            qkv[2 * hidden + 300] = k.bf16of(std.math.inf(f32));
            if (shape.rows > 1) a_in[heads + 2] = k.bf16of(std.math.nan(f32));
            if (shape.rows > 2) b_in[2 * heads + 3] = k.bf16of(std.math.inf(f32));
            if (shape.rows > 3) a_in[3 * heads + 4] = k.bf16of(-std.math.inf(f32));
        }
        t.saved = try t.rig.input(u16, saved);
        t.qkv = try t.rig.input(u16, qkv);
        t.conv_w = try t.rig.input(u16, conv_w);
        t.a_in = try t.rig.input(u16, a_in);
        t.b_in = try t.rig.input(u16, b_in);
        t.a_log = try t.rig.input(f32, a_log);
        t.dt_bias = try t.rig.input(u16, dt_bias);
        t.clean = try t.rig.scratch(u16, (taps - 1 + shape.rows) * conv_dim + conv_dim * taps);
        t.co = try t.rig.scratch(u16, shape.rows * (taps - 1) * conv_dim);
        if (base(t.which) == .inputs) {
            t.gate = try t.rig.scratch(f32, shape.rows * heads);
            t.out_bf16 = try t.rig.output(u16, shape.rows * per_row);
            t.err = try t.a.alloc(f64, shape.rows * per_row);
        } else {
            t.q = try t.rig.scratch(u16, hidden);
            t.kk = try t.rig.scratch(u16, hidden);
            t.v = try t.rig.scratch(u16, hidden);
            t.beta = try t.rig.scratch(u16, heads);
            t.out_f32 = try t.rig.output(f32, shape.rows * heads);
        }
        return true;
    }

    fn sanitized(t: *Pre, from: usize, x: []const u16) []const u16 {
        const clean = t.clean[from..][0..x.len];
        for (clean, x) |*o, b| o.* = if (std.math.isFinite(k.f32of(b))) b else 0;
        return clean;
    }

    /// The row's conv, SiLU, norms, v, decay and beta with `in` and `w` as given; the bf16 outputs land in row `r`.
    fn row(t: *Pre, r: usize, in: mx.ConvInput, conv_w: []const u16, a_in: []const u16, b_in: []const u16) void {
        if (base(t.which) == .inputs) {
            const out = t.out_bf16[r * per_row ..][0..per_row];
            mx.gdnPre(in, r, conv_w, a_in, b_in, t.a_log, t.dt_bias, out[0..hidden], out[hidden..][0..hidden], out[2 * hidden ..][0..hidden], t.gate[r * heads ..][0..heads], out[3 * hidden ..][0..heads], t.co[r * (taps - 1) * conv_dim ..][0 .. (taps - 1) * conv_dim]);
        } else {
            mx.gdnPre(in, r, conv_w, a_in, b_in, t.a_log, t.dt_bias, t.q, t.kk, t.v, t.out_f32[r * heads ..][0..heads], t.beta, t.co[r * (taps - 1) * conv_dim ..][0 .. (taps - 1) * conv_dim]);
        }
    }

    pub fn run(t: *Pre) !void {
        const s = t.shape;
        const in: mx.ConvInput = .{ .saved = t.saved, .qkv = t.qkv };
        for (0..s.rows) |r| {
            const a_in = t.a_in[r * heads ..][0..heads];
            const b_in = t.b_in[r * heads ..][0..heads];
            switch (t.which) {
                .inputs, .decay => t.row(r, in, t.conv_w, a_in, b_in),
                .inputs_writes => {
                    t.row(r, in, t.conv_w, a_in, b_in);
                    if (r == 0) t.saved[0] ^= 1;
                },
                .inputs_finite => t.row(r, .{ .saved = t.sanitized(0, t.saved), .qkv = t.sanitized((taps - 1) * conv_dim, t.qkv) }, t.conv_w, a_in, b_in),
                .inputs_taps => {
                    const reversed = t.clean[0 .. conv_dim * taps];
                    for (0..conv_dim) |col| for (0..taps) |tap| {
                        reversed[col * taps + tap] = t.conv_w[col * taps + taps - 1 - tap];
                    };
                    t.row(r, in, reversed, a_in, b_in);
                },
                .decay_finite => t.row(r, in, t.conv_w, t.sanitized(0, a_in), t.sanitized(heads, b_in)),
                .decay_sign => {
                    const negated = t.clean[0..heads];
                    for (negated, a_in) |*o, ab| o.* = ab ^ 0x8000;
                    t.row(r, in, t.conv_w, negated, b_in);
                },
            }
        }
    }

    pub fn count(t: *Pre) usize {
        return t.shape.rows * t.shape.n;
    }

    /// The conv's input at window row `i` (0 to 2 the saved rows, then the chunk's rows) and column `col`.
    fn at(t: *Pre, i: usize, col: usize) f64 {
        return k.f32of(if (i < taps - 1) t.saved[i * conv_dim + col] else t.qkv[(i - (taps - 1)) * conv_dim + col]);
    }

    /// The decay in float64: exp(-exp(A_log) · softplus(a + dt_bias)).
    fn decayOf(a_log: f32, a: f64, dt: f64) f64 {
        const sum = a + dt;
        const softplus = @max(sum, 0.0) + std.math.log1p(@exp(-@abs(sum)));
        return @exp(-@exp(@as(f64, a_log)) * softplus);
    }

    pub fn reference(t: *Pre, want: []f64, mag: []f64) !void {
        const s = t.shape;
        const u = c.unit(.f32);
        const gt = c.gamma(taps, u);
        for (0..s.rows) |r| {
            if (base(t.which) == .decay) {
                for (0..heads) |h| {
                    want[r * heads + h] = decayOf(t.a_log[h], k.f32of(t.a_in[r * heads + h]), k.f32of(t.dt_bias[h]));
                    mag[r * heads + h] = @abs(want[r * heads + h]);
                }
                continue;
            }
            const w = want[r * per_row ..][0..per_row];
            const m = mag[r * per_row ..][0..per_row];
            const e = t.err[r * per_row ..][0..per_row];
            var vals: [hd]f64 = undefined;
            var errs: [hd]f64 = undefined;
            for (0..3 * heads) |head| {
                for (0..hd) |j| {
                    const col = head * hd + j;
                    var acc: f64 = 0;
                    var magn: f64 = 0;
                    for (0..taps) |tap| {
                        const term = @as(f64, k.f32of(t.conv_w[col * taps + tap])) * t.at(r + tap, col);
                        acc += term;
                        magn += @abs(term);
                    }
                    vals[j] = mx.ops.silu(acc);
                    errs[j] = 1.1 * gt * magn;
                }
                if (head < 2 * heads) {
                    var ss: f64 = 0;
                    var cross: f64 = 0;
                    for (vals, errs) |x, ex| {
                        ss += x * x;
                        cross += @abs(x) * ex;
                    }
                    const denom = ss / @as(f64, hd) + 1e-6 / @as(f64, hd);
                    const inv = 1.0 / @sqrt(denom);
                    const scale: f64 = if (head < heads) 1.0 / @as(f64, hd) else 1.0 / @sqrt(@as(f64, hd));
                    for (0..hd) |j| {
                        w[head * hd + j] = scale * (vals[j] * inv);
                        e[head * hd + j] = scale * inv * (errs[j] + @abs(vals[j]) * cross / (@as(f64, hd) * denom));
                        m[head * hd + j] = @abs(w[head * hd + j]);
                    }
                } else {
                    const hv = head - 2 * heads;
                    for (0..hd) |j| {
                        w[2 * hidden + hv * hd + j] = vals[j];
                        e[2 * hidden + hv * hd + j] = errs[j];
                        m[2 * hidden + hv * hd + j] = @abs(vals[j]);
                    }
                    w[3 * hidden + hv] = mx.ops.sigmoid(k.f32of(t.b_in[r * heads + hv]));
                    e[3 * hidden + hv] = 0;
                    m[3 * hidden + hv] = @abs(w[3 * hidden + hv]);
                }
            }
        }
    }

    /// The conv's carried fp32 error, then the output's own bf16 rounding; the decay has only its fp32 rounding.
    pub fn bounds(t: *Pre, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (base(t.which) == .decay) return;
        for (allow, want, t.err) |*l, w, e| l.* = e + 0x1p-8 * (@abs(w) + e) + 0x1p-133;
    }

    /// The conv's four taps summed in `acc` in `order`, the rest exact.
    pub fn simulate(t: *Pre, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        const s = t.shape;
        if (base(t.which) == .decay) return error.NoSums;
        var vals: [hd]f64 = undefined;
        for (0..s.rows) |r| {
            const o = out[r * per_row ..][0..per_row];
            for (0..3 * heads) |head| {
                for (0..hd) |j| {
                    const col = head * hd + j;
                    var terms: [taps]f64 = undefined;
                    for (&terms, 0..) |*term, tap| term.* = @as(f64, k.f32of(t.conv_w[col * taps + tap])) * t.at(r + tap, col);
                    vals[j] = mx.ops.silu(rows_.sumIn(acc, order, &terms));
                }
                if (head < 2 * heads) {
                    var ss: f64 = 0;
                    for (vals) |x| ss += x * x;
                    const inv = 1.0 / @sqrt(ss / @as(f64, hd) + 1e-6 / @as(f64, hd));
                    const scale: f64 = if (head < heads) 1.0 / @as(f64, hd) else 1.0 / @sqrt(@as(f64, hd));
                    for (0..hd) |j| o[head * hd + j] = scale * (vals[j] * inv);
                } else {
                    const hv = head - 2 * heads;
                    for (0..hd) |j| o[2 * hidden + hv * hd + j] = vals[j];
                    o[3 * hidden + hv] = mx.ops.sigmoid(k.f32of(t.b_in[r * heads + hv]));
                }
            }
        }
    }
};

test "each control binds to its gate, and the served rows are the engine's token rows" {
    try std.testing.expectEqual(g.Gate.nonfinite, control(.inputs_finite).?);
    try std.testing.expectEqual(g.Gate.repeat, control(.inputs_writes).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.inputs_taps).?);
    try std.testing.expectEqual(g.Gate.nonfinite, control(.decay_finite).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.decay_sign).?);
    for (std.enums.values(Which)) |w| for (spec(w).served) |s| try std.testing.expect(Pre.takes(w, s));
    try std.testing.expect(!Pre.takes(.inputs, .{ .rows = 1, .n = hidden, .k = taps }));
    try std.testing.expectEqual(@as(usize, 6160), per_row);
}
