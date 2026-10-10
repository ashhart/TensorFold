//! The gated delta rule and its gradient on the host, one value head at a time: the reference the GPU's learning kernels
//! are checked against. Per value column c the state s (dk wide) decays, reads the key, and corrects toward the value:
//!   s <- g s;  e = v[c] - s . k;  s <- s + beta e k;  o[c] = s . q
const std = @import("std");

/// One value head over `rows` rows: q, k [rows, dk], v [rows, dv], g, beta [rows], state [dv, dk] in and out.
pub const Head = struct {
    rows: usize,
    dk: usize,
    dv: usize,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    g: []const f32,
    beta: []const f32,
};

/// o [rows, dv] from the head's rows, the state carried through (the state the rows end with is left in `state`).
pub fn forward(h: Head, state: []f32, o: []f32) void {
    for (0..h.rows) |t| {
        const k = h.k[t * h.dk ..][0..h.dk];
        const q = h.q[t * h.dk ..][0..h.dk];
        for (0..h.dv) |c| {
            const s = state[c * h.dk ..][0..h.dk];
            var read: f32 = 0;
            for (s, k) |*x, kx| {
                x.* *= h.g[t];
                read += x.* * kx;
            }
            const e = h.v[t * h.dv + c] - read;
            var out: f32 = 0;
            for (s, k, q) |*x, kx, qx| {
                x.* += h.beta[t] * e * kx;
                out += x.* * qx;
            }
            o[t * h.dv + c] = out;
        }
    }
}

/// The gradients of sum(do . o) for one head that began at `start` [dv, dk]: dq, dk [rows, dk], dv [rows, dv],
/// dg, dbeta [rows] (overwritten). Recomputes each row's state from the start, as the GPU's checkpointed pass does.
pub fn backward(gpa: std.mem.Allocator, h: Head, start: []const f32, do: []const f32, dq: []f32, dk: []f32, dv: []f32, dg: []f32, dbeta: []f32) !void {
    const size = h.dv * h.dk;
    // the state before each row's decay: rows + 1 states, the last the one the rows end with
    const states = try gpa.alloc(f32, (h.rows + 1) * size);
    defer gpa.free(states);
    @memcpy(states[0..size], start);
    for (0..h.rows) |t| {
        const before = states[t * size ..][0..size];
        const after = states[(t + 1) * size ..][0..size];
        @memcpy(after, before);
        const one: Head = .{ .rows = 1, .dk = h.dk, .dv = h.dv, .q = h.q[t * h.dk ..][0..h.dk], .k = h.k[t * h.dk ..][0..h.dk], .v = h.v[t * h.dv ..][0..h.dv], .g = h.g[t..][0..1], .beta = h.beta[t..][0..1] };
        var scratch: [512]f32 = undefined;
        forward(one, after, scratch[0..h.dv]);
    }
    @memset(dq, 0);
    @memset(dk, 0);
    @memset(dv, 0);
    @memset(dg, 0);
    @memset(dbeta, 0);
    const ds = try gpa.alloc(f32, size); // the gradient on the state after the row, per column
    defer gpa.free(ds);
    @memset(ds, 0);
    var t = h.rows;
    while (t > 0) {
        t -= 1;
        const k = h.k[t * h.dk ..][0..h.dk];
        const q = h.q[t * h.dk ..][0..h.dk];
        const before = states[t * size ..][0..size];
        const after = states[(t + 1) * size ..][0..size];
        const g = h.g[t];
        const beta = h.beta[t];
        for (0..h.dv) |c| {
            const d = ds[c * h.dk ..][0..h.dk];
            const s1 = after[c * h.dk ..][0..h.dk];
            const s0 = before[c * h.dk ..][0..h.dk];
            const dout = do[t * h.dv + c];
            // o = s1 . q
            for (d, s1, q, 0..) |*x, sx, qx, i| {
                dq[t * h.dk + i] += dout * sx;
                x.* += dout * qx;
            }
            // s1 = p + beta e k, with p = g s0 and e = v - p . k
            var pk: f32 = 0;
            var kd: f32 = 0;
            for (s0, k, d) |sx, kx, dx| {
                pk += g * sx * kx;
                kd += kx * dx;
            }
            const e = h.v[t * h.dv + c] - pk;
            dv[t * h.dv + c] += beta * kd;
            dbeta[t] += e * kd;
            var sd: f32 = 0; // s0 . dp, for dg
            for (d, s0, k, 0..) |*x, sx, kx, i| {
                const p = g * sx;
                dk[t * h.dk + i] += beta * e * x.* - beta * kd * p;
                const dp = x.* - beta * kd * kx; // through p directly and through e's read of p
                sd += sx * dp;
                x.* = g * dp; // the gradient on s0, for the row before
            }
            dg[t] += sd;
        }
    }
}

test "the delta rule's gradient matches finite differences" {
    const gpa = std.testing.allocator;
    const rows = 5;
    const dk = 4;
    const dv = 3;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var q: [rows * dk]f32 = undefined;
    var k: [rows * dk]f32 = undefined;
    var v: [rows * dv]f32 = undefined;
    var g: [rows]f32 = undefined;
    var beta: [rows]f32 = undefined;
    var start: [dv * dk]f32 = undefined;
    var do: [rows * dv]f32 = undefined;
    for (&q) |*x| x.* = r.floatNorm(f32) * 0.5;
    for (&k) |*x| x.* = r.floatNorm(f32) * 0.5;
    for (&v) |*x| x.* = r.floatNorm(f32);
    for (&g) |*x| x.* = 0.5 + 0.5 * r.float(f32);
    for (&beta) |*x| x.* = r.float(f32);
    for (&start) |*x| x.* = r.floatNorm(f32) * 0.3;
    for (&do) |*x| x.* = r.floatNorm(f32);
    var dq: [rows * dk]f32 = undefined;
    var dk_: [rows * dk]f32 = undefined;
    var dv_: [rows * dv]f32 = undefined;
    var dg: [rows]f32 = undefined;
    var dbeta: [rows]f32 = undefined;
    const h: Head = .{ .rows = rows, .dk = dk, .dv = dv, .q = &q, .k = &k, .v = &v, .g = &g, .beta = &beta };
    try backward(gpa, h, &start, &do, &dq, &dk_, &dv_, &dg, &dbeta);
    const Loss = struct {
        fn of(hh: Head, s0: []const f32, dout: []const f32) f64 {
            var s: [dv * dk]f32 = undefined;
            @memcpy(&s, s0);
            var o: [rows * dv]f32 = undefined;
            forward(hh, &s, &o);
            var sum: f64 = 0;
            for (o, dout) |x, y| sum += x * y;
            return sum;
        }
    };
    const eps: f32 = 1e-3;
    const fields = [_]struct { data: []f32, grad: []const f32 }{ .{ .data = &q, .grad = &dq }, .{ .data = &k, .grad = &dk_ }, .{ .data = &v, .grad = &dv_ }, .{ .data = &g, .grad = &dg }, .{ .data = &beta, .grad = &dbeta } };
    for (fields) |f| for (f.data, f.grad) |*x, want| {
        const keep = x.*;
        x.* = keep + eps;
        const up = Loss.of(h, &start, &do);
        x.* = keep - eps;
        const down = Loss.of(h, &start, &do);
        x.* = keep;
        const got: f32 = @floatCast((up - down) / (2 * eps));
        try std.testing.expectApproxEqAbs(want, got, 2e-3 * @max(1, @abs(want)));
    };
}
