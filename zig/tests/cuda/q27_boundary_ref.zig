//! A small FP64 hybrid stack. The real GDN reference and a causal attention layer share a prompt/answer seam.
const std = @import("std");
const gdn = @import("gdn_back_ref.zig");
const D = 4;
const I = 6;
const R = 8;
const L = 8;
const start = 6;
const X = [R][D]f64;
const Mat = [D][D]f64;
const GD: gdn.Dims = .{ .rows = R, .nk = 1, .nv = 1, .dk = D, .dv = D };
const P = 4 * D + 2;
const eps = 1e-3;
const a = std.testing.allocator;
const W = struct {
    input: [P][D]f64,
    output: Mat,
    conv: [3 * D * 4]f64,
    gate: [I][D]f64,
    up: [I][D]f64,
    down: [D][I]f64,
    q: Mat,
    k: Mat,
    v: Mat,
};
const Keep = struct { h: X, norm: X, p: [R * P]f64, mid: X, y: X, gate: [R][I]f64, up: [R][I]f64, prob: [R][R]f64, q: X, k: X, v: X };
const Control = enum { none, adjoint, sigmoid };
const Stack = struct {
    w: [L]W,
    input: X,
    head: Mat,

    fn init(seed: u64) Stack {
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        var s: Stack = undefined;
        const all = std.mem.bytesAsSlice(f64, std.mem.asBytes(&s));
        for (all) |*v| v.* = (random.float(f64) - 0.5) * 1.2;
        for (&s.w) |*w| for (&w.conv, 0..) |*v, i| {
            v.* = if (i % 4 == 3) 0.8 else 0.1;
        };
        return s;
    }

    fn inputs(w: *const W, p: []const f64, qwen27: bool) gdn.Inputs {
        return .{ .p = p, .hist = null, .conv_w = &w.conv, .a_log = &.{-1}, .dt_bias = &.{0.1}, .norm_w = &.{ 1, 0.9, 1.1, 1 }, .init = null, .eps = eps, .qwen27 = qwen27 };
    }

    fn run(s: *const Stack, al: std.mem.Allocator, layer: usize, row: usize, delta: [D]f64, kept: *[L]Keep, last: *X) !f64 {
        var h = s.input;
        for (&s.w, kept, 0..) |*w, *kp, l| {
            kp.h = h;
            var mix: X = @splat(@splat(0));
            for (0..R) |r| norm(&h[r], &kp.norm[r]);
            if (l % 4 != 3) {
                for (0..R) |r| {
                    for (0..P) |j| kp.p[r * P + j] = dot(&w.input[j], &kp.norm[r]);
                }
                const f = try gdn.forward(al, GD, inputs(w, &kp.p, true), false);
                for (0..R) |r| {
                    for (0..D) |j| mix[r][j] = dot(&w.output[j], f.out[r * D ..][0..D]);
                }
            } else {
                for (0..R) |r| for (0..D) |j| {
                    kp.q[r][j] = dot(&w.q[j], &kp.norm[r]);
                    kp.k[r][j] = dot(&w.k[j], &kp.norm[r]);
                    kp.v[r][j] = dot(&w.v[j], &kp.norm[r]);
                };
                for (0..R) |r| {
                    var total: f64 = 0;
                    for (0..r + 1) |j| {
                        kp.prob[r][j] = @exp(dot(&kp.q[r], &kp.k[j]) / @sqrt(@as(f64, D)));
                        total += kp.prob[r][j];
                    }
                    var out: [D]f64 = @splat(0);
                    for (0..r + 1) |j| {
                        kp.prob[r][j] /= total;
                        axpy(&out, kp.prob[r][j], &kp.v[j]);
                    }
                    for (0..D) |j| mix[r][j] = dot(&w.output[j], &out);
                }
            }
            for (0..R) |r| {
                kp.mid[r] = h[r];
                axpy(&kp.mid[r], 1, &mix[r]);
                norm(&kp.mid[r], &kp.y[r]);
                var act: [I]f64 = undefined;
                for (0..I) |j| {
                    kp.gate[r][j] = dot(&w.gate[j], &kp.y[r]);
                    kp.up[r][j] = dot(&w.up[j], &kp.y[r]);
                    act[j] = silu(kp.gate[r][j]) * kp.up[r][j];
                }
                h[r] = kp.mid[r];
                for (0..D) |j| h[r][j] += dot(&w.down[j], &act);
                if (l == layer and r == row) axpy(&h[r], 1, &delta);
            }
        }
        last.* = h;
        var loss: f64 = 0;
        for (start - 1..R) |r| {
            var y: [D]f64 = undefined;
            norm(&h[r], &y);
            var logits: [D]f64 = undefined;
            var sum: f64 = 0;
            for (0..D) |j| {
                logits[j] = dot(&s.head[j], &y);
                sum += @exp(logits[j]);
            }
            loss += (@log(sum) - logits[(r + 1) % D]) / (R - start + 1);
        }
        return loss;
    }

    fn backward(s: *const Stack, al: std.mem.Allocator, kept: *const [L]Keep, last: *const X, layer: usize, control: Control) !X {
        var grad: X = @splat(@splat(0));
        for (start - 1..R) |r| {
            var y: [D]f64 = undefined;
            norm(&last[r], &y);
            var prob: [D]f64 = undefined;
            var total: f64 = 0;
            for (0..D) |j| {
                prob[j] = @exp(dot(&s.head[j], &y));
                total += prob[j];
            }
            var dy: [D]f64 = @splat(0);
            for (0..D) |j| axpy(&dy, (prob[j] / total - @as(f64, if (j == (r + 1) % D) 1 else 0)) / (R - start + 1), &s.head[j]);
            normBack(&last[r], &dy, &grad[r]);
        }
        var l: usize = L - 1;
        while (l > layer) : (l -= 1) {
            const w = &s.w[l];
            const kp = &kept[l];
            for (0..R) |r| {
                var dact: [I]f64 = @splat(0);
                for (0..D) |j| axpy(&dact, grad[r][j], &w.down[j]);
                var dy: [D]f64 = @splat(0);
                for (0..I) |j| {
                    axpy(&dy, dact[j] * kp.up[r][j] * siluGrad(kp.gate[r][j]), &w.gate[j]);
                    axpy(&dy, dact[j] * silu(kp.gate[r][j]), &w.up[j]);
                }
                normBack(&kp.mid[r], &dy, &grad[r]);
            }
            var dy: X = @splat(@splat(0));
            if (l % 4 != 3) {
                const in = inputs(w, &kp.p, control != .sigmoid);
                const f = try gdn.forward(al, GD, in, false);
                var dout: X = @splat(@splat(0));
                for (0..R) |r| for (0..D) |j| axpy(&dout[r], grad[r][j], &w.output[j]);
                const g = try gdn.backward(al, GD, in, f.act, f.beta, f.y, std.mem.bytesAsSlice(f64, std.mem.asBytes(&dout)), false);
                for (0..R) |r| for (0..P) |j| axpy(&dy[r], g.dproj[r * P + j], &w.input[j]);
            } else {
                var dq: X = @splat(@splat(0));
                var dk: X = @splat(@splat(0));
                var dv: X = @splat(@splat(0));
                for (0..R) |r| {
                    var dout: [D]f64 = @splat(0);
                    for (0..D) |j| axpy(&dout, grad[r][j], &w.output[j]);
                    var dp: [R]f64 = undefined;
                    var mix: f64 = 0;
                    for (0..r + 1) |j| {
                        dp[j] = dot(&dout, &kp.v[j]);
                        mix += kp.prob[r][j] * dp[j];
                    }
                    for (0..r + 1) |j| {
                        const ds = kp.prob[r][j] * (dp[j] - mix) / @sqrt(@as(f64, D));
                        axpy(&dq[r], ds, &kp.k[j]);
                        axpy(&dk[j], ds, &kp.q[r]);
                        axpy(&dv[j], kp.prob[r][j], &dout);
                    }
                }
                for (0..R) |r| for (0..D) |j| {
                    axpy(&dy[r], dq[r][j], &w.q[j]);
                    axpy(&dy[r], dk[r][j], &w.k[j]);
                    axpy(&dy[r], dv[r][j], &w.v[j]);
                };
            }
            for (0..R) |r| normBack(&kp.h[r], &dy[r], &grad[r]);
            if (control == .adjoint and l == L / 2) for (&grad) |*row| for (row) |*v| {
                v.* *= 1.2;
            };
        }
        return grad;
    }
};
fn dot(x: []const f64, y: []const f64) f64 {
    var v: f64 = 0;
    for (x, y) |a_, b| v += a_ * b;
    return v;
}
fn axpy(y: []f64, scale: f64, x: []const f64) void {
    for (y, x) |*v, b| v.* += scale * b;
}
fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}
fn siluGrad(x: f64) f64 {
    const s = 1 / (1 + @exp(-x));
    return s * (1 + x * (1 - s));
}
fn norm(x: *const [D]f64, y: *[D]f64) void {
    const r = 1 / @sqrt(dot(x, x) / D + eps);
    for (y, x) |*v, b| v.* = b * r;
}
fn normBack(x: *const [D]f64, dy: *const [D]f64, dx: *[D]f64) void {
    const r = 1 / @sqrt(dot(x, x) / D + eps);
    const t = dot(x, dy) * r * r * r / D;
    for (dx, x, dy) |*v, b, g| v.* += r * g - t * b;
}

test "dense gradient: hybrid prompt/answer boundary matches FD and catches the sigmoid replay" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const s = Stack.init(0x27ba_c004);
    var kept: [L]Keep = undefined;
    var last: X = undefined;
    _ = try s.run(al, L, 0, @splat(0), &kept, &last);
    var caught: usize = 0;
    for ([_]usize{ 0, 3, 6 }) |layer| {
        const g = try s.backward(al, &kept, &last, layer, .none);
        const old = try s.backward(al, &kept, &last, layer, .sigmoid);
        for ([_]usize{ 0, start - 2, start - 1, R - 1 }) |row| {
            const length = @sqrt(dot(&g[row], &g[row]));
            const old_len = @sqrt(dot(&old[row], &old[row]));
            var dir: [D]f64 = undefined;
            for (&dir, old[row]) |*v, x| v.* = x / old_len;
            const true_slope = dot(&g[row], &dir);
            const ratio = old_len / true_slope;
            if (row == start - 1 and layer < 6 and @abs(ratio - 1) > 0.15) caught += 1;
            for ([_]f64{ 1e-6, 2e-6, 4e-6 }) |step| {
                var delta: [D]f64 = undefined;
                for (&delta, g[row]) |*v, x| v.* = step * x / length;
                var tmp: [L]Keep = undefined;
                var out: X = undefined;
                const plus = try s.run(al, layer, row, delta, &tmp, &out);
                for (&delta) |*v| v.* = -v.*;
                const minus = try s.run(al, layer, row, delta, &tmp, &out);
                const fd = (plus - minus) / (2 * step);
                try std.testing.expectApproxEqRel(length, fd, 1e-5);
            }
            std.debug.print("WITNESS hybrid layer {d} row {d}: corrected norm {d:.8}, legacy/slope {d:.6}\n", .{ layer, row, length, ratio });
            if (layer == 6) try std.testing.expectApproxEqAbs(@as(f64, 1), ratio, 1e-12);
        }
    }
    try std.testing.expect(caught > 0);
}

test "dense gradient: the old boundary gate gives a five-thirds state gradient" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const s = Stack.init(0x27ba_c004);
    var p: [R * P]f64 = undefined;
    for (&p, 0..) |*v, i| v.* = 0.2 + @as(f64, @floatFromInt(i % 11)) / 20;
    for (0..R) |r| for (0..D) |j| {
        p[r * P + GD.zAt() + j] = 0.6;
    };
    const in = Stack.inputs(&s.w[0], &p, true);
    const f = try gdn.forward(al, GD, in, false);
    var dout: X = @splat(@splat(0));
    dout[start - 1] = .{ 1, -0.2, 0.7, 0.4 };
    const go = std.mem.bytesAsSlice(f64, std.mem.asBytes(&dout));
    const corrected = try gdn.backward(al, GD, in, f.act, f.beta, f.y, go, false);
    var old_in = in;
    old_in.qwen27 = false;
    const old = try gdn.backward(al, GD, old_in, f.act, f.beta, f.y, go, false);
    for (0..D) |j| {
        const at = (start - 1) * D + j;
        try std.testing.expectApproxEqRel(corrected.dy[at] * (5.0 / 3.0), old.dy[at], 1e-12);
    }
    for ([_]bool{ false, true }) |mode| {
        var check_in = in;
        check_in.qwen27 = mode;
        var rng = std.Random.DefaultPrng.init(0x2715);
        const check = try gdn.finiteCheck(al, GD, check_in, go, rng.random(), 100);
        try std.testing.expect(check.worst < 1e-5);
    }
    const directional = @import("core").directional;
    const row = start - 1;
    const yr = f.y[row * D ..][0..D];
    const true_grad = corrected.dy[row * D ..][0..D];
    const wrong_grad = old.dy[row * D ..][0..D];
    const length = @sqrt(dot(true_grad, true_grad));
    const wrong_length = @sqrt(dot(wrong_grad, wrong_grad));
    var slopes: [6]f64 = undefined;
    for (directional.fractions, &slopes) |fraction, *slope| {
        const h = fraction * @sqrt(dot(yr, yr));
        var losses: [2]f64 = undefined;
        for ([_]f64{ -1, 1 }, &losses) |sign, *loss| {
            var shifted: [D]f64 = undefined;
            for (&shifted, yr, true_grad) |*v, y, g| v.* = y + sign * h * g / length;
            const ri = 1 / @sqrt(dot(&shifted, &shifted) / D + eps);
            loss.* = 0;
            for (shifted, in.norm_w, dout[row]) |y, weight, g| loss.* += y * ri * weight * silu(0.6) * g;
        }
        slope.* = directional.central(losses, h);
    }
    const right = directional.evaluate(slopes, length);
    const wrong = directional.evaluate(slopes, wrong_length);
    try std.testing.expectEqual(directional.Verdict.pass, right.verdict);
    try std.testing.expectEqual(directional.Verdict.mismatch, wrong.verdict);
    std.debug.print("CONTROL boundary row {d}: sigmoid gradient {s}, relative {d:.8}\n", .{ row, @tagName(wrong.verdict), wrong.relative });
    std.debug.print("WITNESS boundary row {d}: sigmoid/SiLU state gradient 1.666667; both adjoints match their own FD\n", .{start - 1});
}

fn profile(s: *const Stack, layer: usize, row: usize, dir: [D]f64, residual_norm: f64) ![6]f64 {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var slopes: [6]f64 = undefined;
    for (@import("core").directional.fractions, &slopes) |fraction, *slope| {
        const h = fraction * residual_norm;
        var losses: [2]f64 = undefined;
        for ([_]f64{ -1, 1 }, &losses) |sign, *loss| {
            var delta = dir;
            for (&delta) |*v| v.* *= sign * h;
            var kept: [L]Keep = undefined;
            var last: X = undefined;
            loss.* = try s.run(arena.allocator(), layer, row, delta, &kept, &last);
        }
        slope.* = @import("core").directional.central(losses, h);
    }
    return slopes;
}

test "dense gradient: v2 counts reported scale actual adjoint scale and sigmoid changed probes" {
    const gate = @import("core").directional;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const s = Stack.init(0x27ba_c004);
    var kept: [L]Keep = undefined;
    var last: X = undefined;
    _ = try s.run(al, L, 0, @splat(0), &kept, &last);
    const Kind = enum { reported, adjoint, sigmoid };
    for ([_]Kind{ .reported, .adjoint, .sigmoid }) |kind| {
        var changed: usize = 0;
        var caught: usize = 0;
        var unit_changed: usize = 0;
        var baseline_pass: usize = 0;
        var eligible: usize = 0;
        for ([_]usize{ 0, 3, 6 }) |layer| {
            const correct = try s.backward(al, &kept, &last, layer, .none);
            const wrong = try s.backward(al, &kept, &last, layer, if (kind == .adjoint) .adjoint else if (kind == .sigmoid) .sigmoid else .none);
            for ([_]usize{ 0, start - 2, start - 1, R - 1 }) |row| {
                const length = @sqrt(dot(&correct[row], &correct[row]));
                var control_grad = wrong[row];
                if (kind == .reported and layer == 3) for (&control_grad) |*v| {
                    v.* *= 1.2;
                };
                const control_norm = @sqrt(dot(&control_grad, &control_grad));
                var dir: [D]f64 = undefined;
                var true_dir: [D]f64 = undefined;
                var delta: f64 = 0;
                for (&dir, &true_dir, control_grad, correct[row]) |*d, *td, g, truth| {
                    d.* = if (kind == .reported) truth / length else g / control_norm;
                    td.* = truth / length;
                    delta += (g - truth) * (g - truth);
                }
                const residual = if (layer + 1 < L) kept[layer + 1].h[row] else last[row];
                const size = @sqrt(dot(&residual, &residual));
                const baseline = gate.evaluate(try profile(&s, layer, row, true_dir, size), length);
                const result = gate.evaluate(try profile(&s, layer, row, dir, size), control_norm);
                const differs = @sqrt(delta) > 1e-8 * length;
                var direction_delta: f64 = 0;
                for (dir, true_dir) |d, truth| direction_delta += (d - truth) * (d - truth);
                if (@sqrt(direction_delta) > 1e-8) unit_changed += 1;
                if (baseline.verdict == .pass) baseline_pass += 1;
                if (differs) {
                    changed += 1;
                    if (baseline.verdict == .pass) {
                        eligible += 1;
                        if (result.verdict == .mismatch) caught += 1;
                    }
                }
                std.debug.print("CONTROL-PROBE {s} layer {d} row {d}: changed {any}, baseline {s}, control {s}, g {e:.8}, estimate {e:.8}, relative {e:.8}\n", .{ @tagName(kind), layer, row, differs, @tagName(baseline.verdict), @tagName(result.verdict), control_norm, result.estimate, result.relative });
            }
        }
        std.debug.print("CONTROL-COUNT {s}: total 12, baseline_pass {d}, changed {d}, unit_changed {d}, eligible {d}, caught {d}, missed {d}\n", .{ @tagName(kind), baseline_pass, changed, unit_changed, eligible, caught, eligible - caught });
        try std.testing.expectEqual(@as(usize, if (kind == .reported) 4 else 8), changed);
        try std.testing.expect(eligible > 0);
        try std.testing.expect(caught > 0);
    }
}
