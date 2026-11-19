//! Independent FP64 delta-rule recurrence over the prep's recorded points, with propagated FP32 error.
const std = @import("std");
pub fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}
const eps32 = 0x1p-24;
fn gamma(n: usize) f64 {
    const e = @as(f64, @floatFromInt(n)) * eps32;
    return e / (1 - e);
}
pub const Points = struct {
    rows: usize,
    heads: usize,
    dim: usize,
    initial: []const f32,
    q: []const u16,
    k: []const u16,
    v: []const u16,
    decay: []const f32,
    beta: []const f32,
    sy: []const u16,
    final: []const f32,
};
pub fn verify(a: std.mem.Allocator, p: Points) !void {
    const state = try a.alloc(f64, p.dim * p.dim);
    defer a.free(state);
    const err = try a.alloc(f64, state.len);
    defer a.free(err);
    const width = p.heads * p.dim;
    for (0..p.heads) |h| {
        for (state, p.initial[h * state.len ..][0..state.len]) |*s, v| s.* = v;
        @memset(err, 0);
        for (0..p.rows) |r| for (0..p.dim) |j| {
            const at = r * width + h * p.dim;
            const ss = state[j * p.dim ..][0..p.dim];
            const ee = err[j * p.dim ..][0..p.dim];
            var kv: f64 = 0;
            var ke: f64 = 0;
            var magnitude: f64 = 0;
            for (ss, ee, 0..) |*s, *e, i| {
                const decay: f64 = p.decay[at + i];
                s.* *= decay;
                e.* = @abs(decay) * e.* + gamma(1) * @abs(s.*);
                const ki = bf(p.k[at + i]);
                kv += s.* * ki;
                ke += e.* * @abs(ki);
                magnitude += @abs(s.* * ki);
            }
            ke += gamma(2 * p.dim) * magnitude;
            const value = bf(p.v[at + j]);
            const beta: f64 = p.beta[r * p.heads + h];
            const delta = (value - kv) * beta;
            const de = @abs(beta) * ke + gamma(2) * ((@abs(value) + @abs(kv)) * @abs(beta));
            var out: f64 = 0;
            var oe: f64 = 0;
            magnitude = 0;
            for (ss, ee, 0..) |*s, *e, i| {
                const ki = bf(p.k[at + i]);
                e.* += @abs(ki) * de + gamma(2) * (@abs(s.*) + @abs(ki * delta));
                s.* += ki * delta;
                const qi = bf(p.q[at + i]);
                out += s.* * qi;
                oe += e.* * @abs(qi);
                magnitude += @abs(s.* * qi);
            }
            oe += gamma(2 * p.dim) * magnitude;
            // The store rounds once to BF16; the FP32 recurrence state remains FP32 across cuts.
            const bound = oe + 0x1p-8 * (@abs(out) + oe) + 0x1p-126;
            if (@abs(bf(p.sy[at + j]) - out) > bound) return error.KdaFp64OutputMismatch;
        };
        for (state, err, p.final[h * state.len ..][0..state.len]) |s, e, got| {
            if (@abs(@as(f64, got) - s) > e + 0x1p-126) return error.KdaFp64StateMismatch;
        }
    }
}

test "GLM ordered: recurrence oracle sees nonzero history and rejects an incorrect earlier-row value path" {
    const a = std.testing.allocator;
    const initial = @as([64]f32, @splat(0.25));
    const q = @as([8]u16, @splat(0x3e00));
    const v = @as([8]u16, @splat(0x3f00));
    const decay = @as([8]f32, @splat(0.5));
    const beta = [_]f32{0.5};
    // Eight decayed entries 1/8 dotted with 1/8: kv=1/8, delta=3/16, state=19/128.
    const final = @as([64]f32, @splat(19.0 / 128.0));
    const sy = @as([8]u16, @splat(0x3e18));
    var p: Points = .{ .rows = 1, .heads = 1, .dim = 8, .initial = &initial, .q = &q, .k = &q, .v = &v, .decay = &decay, .beta = &beta, .sy = &sy, .final = &final };
    try verify(a, p);
    p.final = &initial;
    try std.testing.expectError(error.KdaFp64StateMismatch, verify(a, p));
    p.final = &final;
    const bad = @as([8]u16, @splat(0x3e80));
    p.sy = &bad;
    try std.testing.expectError(error.KdaFp64OutputMismatch, verify(a, p));
}
