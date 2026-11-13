//! The CPU delta rule's recurrence under the contract: one head's fp32 state over a window of tokens, four controls.
const std = @import("std");
const contract = @import("contract");
const mx = @import("cpu_mixers");
const rig_ = @import("contract_rig.zig");

const k = mx.ops.kernels;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { chain, finite, writes, decay, half };

const hd = 128;
const heads = 16;
const stride = heads * hd;
/// Each dot over 128: 32 lanes of four fused terms, then the lanes' five-level tree.
const dot_depth = 4 + 5;

fn depth(_: c.Shape) usize {
    return dot_depth;
}

/// A decode step, a verify window (one stream's segment) and a prompt chunk of tokens, one head's 128 outputs a token.
const served = [_]c.Shape{ .{ .rows = 1, .n = hd, .k = hd }, .{ .rows = 16, .n = hd, .k = hd }, .{ .rows = 128, .n = hd, .k = hd } };
const ctl = [_]c.Shape{.{ .rows = 9, .n = hd, .k = hd }};

const sums_why = "the stress state stays dyadic (beta 1/128, no decay, integer q, k and v), so fp32 sums are exact in every order; the fp16 and bf16 margins are measured, not required";

pub fn spec(w: Which) c.Spec {
    const name: []const u8 = switch (w) {
        .chain => "cpu delta recurrence",
        .finite => "control: recurrence taking a non-finite input as zero",
        .writes => "control: recurrence writing its queries",
        .decay => "control: recurrence skipping the decay",
        .half => "control: recurrence summing its dots in half precision",
    };
    return .{ .name = name, .accum = .f32, .depth = depth, .out = .bf16, .served = if (w == .chain) &served else &ctl, .sums = .{ .f16 = 0, .bf16 = 0, .why = sums_why } };
}

/// The gate each control must fail.
pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .chain => null,
        .finite => .nonfinite,
        .writes => .repeat,
        .decay => .shapes,
        .half => .sums,
    };
}

pub const Chain = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    shape: c.Shape = undefined,
    state0: []f32 = &.{},
    q: []u16 = &.{},
    kk: []u16 = &.{},
    v: []u16 = &.{},
    gate: []f32 = &.{},
    beta: []u16 = &.{},
    state: []f32 = &.{},
    saved: []f32 = &.{},
    ones: []f32 = &.{},
    y: []u16 = &.{},
    clean: []u16 = &.{},
    out: []u16 = &.{},
    /// Per output, the first-order bound of the fp32 arithmetic's error, carried through the state by `reference`.
    err: []f64 = &.{},

    pub fn init(a: std.mem.Allocator, which: Which) Chain {
        return .{ .a = a, .rig = .{ .a = a }, .which = which };
    }

    pub fn deinit(t: *Chain) void {
        t.drop();
        t.rig.deinit();
    }

    fn drop(t: *Chain) void {
        if (t.err.len > 0) t.a.free(t.err);
        t.err = &.{};
        t.rig.reset();
    }

    pub fn kernel(t: *Chain) g.Kernel {
        return .{ .spec = spec(t.which), .ctx = t, .vt = &vt, .control = control(t.which) };
    }

    const vt = g.hooks(Chain);

    fn takes(shape: c.Shape) bool {
        return shape.rows > 0 and shape.rows <= 128 and shape.n == hd and shape.k == hd;
    }

    pub fn prepare(t: *Chain, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        if (!takes(shape)) return false;
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const state0 = try t.a.alloc(f32, hd * hd);
        defer t.a.free(state0);
        const q = try t.a.alloc(u16, shape.rows * stride);
        defer t.a.free(q);
        const kk = try t.a.alloc(u16, shape.rows * stride);
        defer t.a.free(kk);
        const v = try t.a.alloc(u16, shape.rows * stride);
        defer t.a.free(v);
        const gate = try t.a.alloc(f32, shape.rows * heads);
        defer t.a.free(gate);
        const beta = try t.a.alloc(u16, shape.rows * heads);
        defer t.a.free(beta);
        if (fill == .stress) {
            for (state0) |*s| s.* = @floatFromInt(rnd.intRangeAtMost(i32, -2, 2));
            for (q) |*x| x.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -2, 2)));
            for (kk) |*x| x.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -1, 1)));
            for (v) |*x| x.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -4, 4)));
            @memset(gate, 1);
            @memset(beta, k.bf16of(1.0 / 128.0));
        } else {
            for (state0) |*s| s.* = rnd.floatNorm(f32) * 0.1;
            for (q) |*x| x.* = k.bf16of(rnd.floatNorm(f32) * 0.1);
            for (kk) |*x| x.* = k.bf16of(rnd.floatNorm(f32) * 0.1);
            for (v) |*x| x.* = k.bf16of(rnd.floatNorm(f32));
            for (gate) |*x| x.* = 0.9 + rnd.float(f32) * 0.1;
            for (beta) |*x| x.* = k.bf16of(rnd.float(f32));
        }
        if (fill == .nonfinite) {
            q[3] = k.bf16of(std.math.nan(f32));
            kk[(shape.rows - 1) * stride + 7] = k.bf16of(std.math.inf(f32));
        }
        t.state0 = try t.rig.input(f32, state0);
        t.q = try t.rig.input(u16, q);
        t.kk = try t.rig.input(u16, kk);
        t.v = try t.rig.input(u16, v);
        t.gate = try t.rig.input(f32, gate);
        t.beta = try t.rig.input(u16, beta);
        t.state = try t.rig.scratch(f32, hd * hd);
        t.saved = try t.rig.scratch(f32, hd * hd);
        t.ones = try t.rig.scratch(f32, shape.rows * heads);
        @memset(t.ones, 1);
        t.y = try t.rig.scratch(u16, shape.rows * stride);
        t.clean = try t.rig.scratch(u16, 2 * shape.rows * stride);
        t.out = try t.rig.output(u16, shape.rows * hd);
        t.err = try t.a.alloc(f64, shape.rows * hd);
        return true;
    }

    fn sanitized(t: *Chain, at: usize, x: []const u16) []const u16 {
        const clean = t.clean[at..][0..x.len];
        for (clean, x) |*o, b| o.* = if (std.math.isFinite(k.f32of(b))) b else 0;
        return clean;
    }

    /// Head 0's recurrence over the window into `y` (the engine's stride), then its 128 values a token into the output.
    pub fn run(t: *Chain) !void {
        const s = t.shape;
        @memcpy(t.state, t.state0);
        switch (t.which) {
            .chain => mx.gdnChain(t.state, t.q, t.kk, t.v, t.gate, t.beta, s.rows, 0, t.y, t.saved, false),
            .writes => {
                mx.gdnChain(t.state, t.q, t.kk, t.v, t.gate, t.beta, s.rows, 0, t.y, t.saved, false);
                t.q[0] ^= 1;
            },
            .finite => mx.gdnChain(t.state, t.sanitized(0, t.q), t.sanitized(s.rows * stride, t.kk), t.v, t.gate, t.beta, s.rows, 0, t.y, t.saved, false),
            .decay => mx.gdnChain(t.state, t.q, t.kk, t.v, t.ones, t.beta, s.rows, 0, t.y, t.saved, false),
            .half => t.halfChain(),
        }
        for (0..s.rows) |node| @memcpy(t.out[node * hd ..][0..hd], t.y[node * stride ..][0..hd]);
    }

    /// The recurrence with both dots accumulated in fp16; the state updates stay fp32.
    fn halfChain(t: *Chain) void {
        const s = t.shape;
        for (0..s.rows) |node| {
            const gc = t.gate[node * heads];
            const bc = k.f32of(t.beta[node * heads]);
            for (0..hd) |dv| {
                const row = t.state[dv * hd ..][0..hd];
                for (row) |*x| x.* *= gc;
                var kv: f16 = 0;
                for (row, 0..) |x, dk| kv += @floatCast(x * k.f32of(t.kk[node * stride + dk]));
                const delta = (k.f32of(t.v[node * stride + dv]) - @as(f32, @floatCast(kv))) * bc;
                for (row, 0..) |*x, dk| x.* = @mulAdd(f32, k.f32of(t.kk[node * stride + dk]), delta, x.*);
                var y: f16 = 0;
                for (row, 0..) |x, dk| y += @floatCast(x * k.f32of(t.q[node * stride + dk]));
                t.y[node * stride + dv] = k.bf16of(@floatCast(y));
            }
        }
    }

    pub fn count(t: *Chain) usize {
        return t.shape.rows * hd;
    }

    /// The recurrence in float64 on the same inputs; `err` carries a first-order bound of the fp32 kernel's rounding.
    pub fn reference(t: *Chain, want: []f64, mag: []f64) !void {
        const s = t.shape;
        const u = c.unit(.f32);
        const gd = c.gamma(dot_depth, u);
        const S = try t.a.alloc(f64, hd * hd);
        defer t.a.free(S);
        const E = try t.a.alloc(f64, hd * hd);
        defer t.a.free(E);
        for (S, t.state0) |*x, v| x.* = v;
        @memset(E, 0);
        var qc: [hd]f64 = undefined;
        var kc: [hd]f64 = undefined;
        for (0..s.rows) |node| {
            for (0..hd) |i| {
                qc[i] = k.f32of(t.q[node * stride + i]);
                kc[i] = k.f32of(t.kk[node * stride + i]);
            }
            const gc: f64 = t.gate[node * heads];
            const bc: f64 = k.f32of(t.beta[node * heads]);
            for (0..hd) |dv| {
                const row = S[dv * hd ..][0..hd];
                const erow = E[dv * hd ..][0..hd];
                var kv: f64 = 0;
                var kv_mag: f64 = 0;
                var kv_err: f64 = 0;
                for (row, erow, kc) |*x, *e, kk| {
                    x.* *= gc;
                    e.* = e.* * gc + u * @abs(x.*);
                    kv += x.* * kk;
                    kv_mag += @abs(x.* * kk);
                    kv_err += e.* * @abs(kk);
                }
                kv_err += gd * kv_mag;
                const vv: f64 = k.f32of(t.v[node * stride + dv]);
                const delta = (vv - kv) * bc;
                const delta_err = (kv_err + u * @abs(vv - kv)) * bc + u * @abs(delta);
                var y: f64 = 0;
                var y_mag: f64 = 0;
                var y_err: f64 = 0;
                for (row, erow, kc, qc) |*x, *e, kk, qq| {
                    x.* += kk * delta;
                    e.* += @abs(kk) * delta_err + u * @abs(x.*);
                    y += x.* * qq;
                    y_mag += @abs(x.* * qq);
                    y_err += e.* * @abs(qq);
                }
                want[node * hd + dv] = y;
                mag[node * hd + dv] = y_mag;
                t.err[node * hd + dv] = y_err + gd * y_mag;
            }
        }
    }

    /// The carried fp32 error bound, then the output's own bf16 rounding.
    pub fn bounds(t: *Chain, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        for (allow, want, t.err) |*l, w, e| l.* = e + 0x1p-8 * (@abs(w) + e) + 0x1p-133;
    }

    /// Both dots summed in `acc` in `order`, the state's products and updates exact.
    pub fn simulate(t: *Chain, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        const s = t.shape;
        const S = try t.a.alloc(f64, hd * hd);
        defer t.a.free(S);
        for (S, t.state0) |*x, v| x.* = v;
        var terms: [hd]f64 = undefined;
        for (0..s.rows) |node| {
            const gc: f64 = t.gate[node * heads];
            const bc: f64 = k.f32of(t.beta[node * heads]);
            for (0..hd) |dv| {
                const row = S[dv * hd ..][0..hd];
                for (row, &terms, 0..) |*x, *tt, dk| {
                    x.* *= gc;
                    tt.* = x.* * k.f32of(t.kk[node * stride + dk]);
                }
                const kv = rows_.sumIn(acc, order, &terms);
                const delta = (@as(f64, k.f32of(t.v[node * stride + dv])) - kv) * bc;
                for (row, &terms, 0..) |*x, *tt, dk| {
                    x.* += @as(f64, k.f32of(t.kk[node * stride + dk])) * delta;
                    tt.* = x.* * k.f32of(t.q[node * stride + dk]);
                }
                out[node * hd + dv] = rows_.sumIn(acc, order, &terms);
            }
        }
    }
};

test "each control binds to its gate, and the served windows are the engine's token rows" {
    try std.testing.expectEqual(g.Gate.nonfinite, control(.finite).?);
    try std.testing.expectEqual(g.Gate.repeat, control(.writes).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.decay).?);
    try std.testing.expectEqual(g.Gate.sums, control(.half).?);
    for (served ++ ctl) |s| try std.testing.expect(Chain.takes(s));
    try std.testing.expect(!Chain.takes(.{ .rows = 129, .n = hd, .k = hd }));
    try std.testing.expectEqual(@as(usize, 9), depth(served[0]));
}
