//! Dense binary64 delta-rule reverse pass, independent of split warps and generated shader bodies.
const std = @import("std");
const ref = @import("learning_reference.zig");
pub fn make(a: std.mem.Allocator) !ref.Fixture {
    const rows = 5;
    const heads = 2;
    const dim = 128;
    const conv = 2 * dim + heads * dim;
    const parts = 4;
    const size = heads * dim * dim;
    const b = try a.alloc(ref.Buffer, 13);
    b[0] = try ref.bfInput(a, try ref.floats(a, rows * conv, 0, 32));
    b[1] = try ref.input(a, try ref.floats(a, rows * dim, 1, 256));
    b[2] = try ref.input(a, try ref.floats(a, rows * dim, 2, 128));
    const decay = try a.alloc(f32, rows * heads);
    const beta = try a.alloc(f32, rows * heads);
    for (decay, beta, 0..) |*g, *mix, i| {
        g.* = @as(f32, @floatFromInt(i % 3 + 5)) / 8;
        mix.* = @as(f32, @floatFromInt(i % 3 + 1)) / 8;
    }
    b[3] = try ref.input(a, decay);
    b[4] = try ref.input(a, beta);
    const checkpoints = try a.alloc(f32, 2 * size);
    const state = try a.alloc(f64, size);
    const states = try a.alloc(f64, rows * size);
    const errors = try a.alloc(f64, rows * heads * dim);
    for (state, 0..) |*v, i| v.* = @as(f64, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 1024;
    for (0..rows) |r| {
        if (r % 4 == 0) for (state, 0..) |*v, i| {
            checkpoints[(r / 4) * size + i] = @floatCast(v.*);
            // The independent tape consumes the same saved F32 checkpoint at each four-row boundary.
            v.* = checkpoints[(r / 4) * size + i];
        };
        @memcpy(states[r * size ..][0..size], state);
        for (0..heads) |h| for (0..dim) |v| {
            const at = (h * dim + v) * dim;
            var prediction: f64 = 0;
            for (0..dim) |d| prediction += state[at + d] * decay[r * heads + h] * ref.scalar(b[2], f32, r * dim + d);
            const error_value = ref.scalar(b[0], u16, r * conv + 2 * dim + h * dim + v) - prediction;
            errors[(r * heads + h) * dim + v] = error_value;
            for (0..dim) |d| state[at + d] = state[at + d] * decay[r * heads + h] + ref.scalar(b[2], f32, r * dim + d) * error_value * beta[r * heads + h];
        };
    }
    b[5] = try ref.input(a, checkpoints);
    b[6] = try ref.input(a, try ref.floats(a, rows * heads * dim, 3, 64));
    b[7] = try ref.scratch(a, heads * parts * 4 * 32 * dim);
    b[8] = try ref.output(a, rows * heads * parts * dim, 0);
    b[9] = try ref.output(a, rows * heads * parts * dim, 0);
    b[10] = try ref.output(a, rows * conv, 0.125);
    b[11] = try ref.output(a, rows * heads * parts, 0);
    b[12] = try ref.output(a, rows * heads * parts, 0);
    const ds = try a.alloc(f64, size);
    @memset(ds, 0);
    var r: usize = rows;
    while (r > 0) {
        r -= 1;
        for (0..heads) |h| for (0..dim) |v| {
            const at = (h * dim + v) * dim;
            const group = (r * heads + h) * parts + v / 32;
            const prior = states[r * size + at ..][0..dim];
            const g: f64 = decay[r * heads + h];
            const mix: f64 = beta[r * heads + h];
            const err = errors[(r * heads + h) * dim + v];
            const upstream = ref.scalar(b[6], f32, (r * heads + h) * dim + v);
            var dd: f64 = 0;
            for (0..dim) |d| {
                const k = ref.scalar(b[2], f32, r * dim + d);
                b[8].want.?[group * dim + d] += upstream * (g * prior[d] + k * mix * err);
                ds[at + d] += upstream * ref.scalar(b[1], f32, r * dim + d);
                dd += ds[at + d] * k;
            }
            b[10].want.?[r * conv + 2 * dim + h * dim + v] = mix * dd;
            b[12].want.?[group] += dd * err;
            for (0..dim) |d| {
                const k = ref.scalar(b[2], f32, r * dim + d);
                b[9].want.?[group * dim + d] += mix * err * ds[at + d] - mix * dd * g * prior[d];
                const before_decay = ds[at + d] - mix * dd * k;
                b[11].want.?[group] += before_decay * prior[d];
                ds[at + d] = g * before_decay;
            }
        };
    }
    return .{ .buffers = b, .params = try ref.params(a, ref.Scan{ .rows = rows, .nk = 1, .nv = heads, .conv = conv, .proj = conv + heads * dim + 2 * heads, .from = 0 }), .groups = .{ heads * parts, 1, 1 }, .bound = 2e-6 };
}
