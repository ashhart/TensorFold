//! FP64 DeltaNet parent-state recurrence, independent of register-slot evaluation order.
const std = @import("std");
const c = @import("contract.zig");
pub const Step = extern struct { row: u32, read_slot: i32, write_slot: i32 };
pub const Params = extern struct {
    rows: u32,
    nk: u32 = 16,
    nv: u32 = 48,
    dk: u32 = 128,
    dv: u32 = 128,
    taps: u32 = 4,
    slots: u32 = 3,
    weight_flags: u32 = 0,
    eps: f32 = 1e-6,
    zba_stride: u32 = 0,
    advance: u32 = 0,
};
pub const Segment = extern struct { first: u32, rows: u32, state_slot: u32, next_slot: u32 };
pub const Fixture = struct {
    p: Params,
    segments: [2]Segment,
    parents: []i32,
    steps: []Step,
    query: []u16,
    key: []u16,
    value: []u16,
    decay: []f32,
    mixing: []u16,
    state: []f32,
    output: []f64,
    bound: []f64,
    advanced: []f64,
    state_bound: []f64,
    pub fn accepts(f: Fixture, bytes: []const u8) bool {
        if (bytes.len != 2 * f.output.len) return false;
        for (f.output, f.bound, 0..) |w, b, i| {
            const got = c.read(.bf16, bytes, i);
            if (!std.math.isFinite(got) or @abs(got - w) > b) return false;
        }
        return true;
    }
};
fn bf(value: f64) u16 {
    return c.toBf16(@floatCast(value));
}
fn v(bits: u16) f64 {
    return c.fromBf16(bits);
}
fn fill(a: std.mem.Allocator, n: usize, seed: u64, scale: f64) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out, 0..) |*x, i| x.* = bf((@as(f64, @floatFromInt((i *% 13 +% seed) % 23)) - 11) * scale);
    return out;
}
pub fn make(comptime Planner: type, a: std.mem.Allocator, limit: usize, n: usize, advance: bool, seed: u64) !Fixture {
    const total = 2 * n;
    const state_size = 48 * 128 * 128;
    var f = Fixture{
        .p = .{ .rows = @intCast(total), .advance = @intFromBool(advance) },
        .segments = .{ .{ .first = 0, .rows = @intCast(n), .state_slot = 2, .next_slot = 1 }, .{ .first = @intCast(n), .rows = @intCast(n), .state_slot = 0, .next_slot = 1 } },
        .parents = try a.alloc(i32, total),
        .steps = try a.alloc(Step, total),
        .query = try fill(a, total * 16 * 128, seed, 0.0005),
        .key = try fill(a, total * 16 * 128, seed +% 7, 0.001),
        .value = try fill(a, total * 48 * 128, seed +% 11, 0.03125),
        .decay = try a.alloc(f32, total * 48),
        .mixing = try fill(a, total * 48, seed +% 19, 0.03125),
        .state = try a.alloc(f32, 3 * state_size),
        .output = try a.alloc(f64, total * 48 * 128),
        .bound = try a.alloc(f64, total * 48 * 128),
        .advanced = try a.alloc(f64, 3 * state_size),
        .state_bound = try a.alloc(f64, 3 * state_size),
    };
    for (f.state, f.advanced, f.state_bound, 0..) |*x, *want, *err, i| {
        x.* = @as(f32, @floatFromInt((i * 17 + 5) % 31)) / 1024 - 0.01;
        want.* = x.*;
        err.* = 0;
    }
    for (f.decay, 0..) |*x, i| x.* = 0.875 + @as(f32, @floatFromInt(i % 3)) / 32;
    for (0..2) |stream| {
        const first = stream * n;
        const parents = f.parents[first..][0..n];
        for (parents, 0..) |*p, row| p.* = if (row == 0) -1 else @intCast(if (limit == 1 or stream == 1) row - 1 else if (limit == 2) 0 else (row - 1) / 2);
        comptime std.debug.assert(@sizeOf(Step) == @sizeOf(Planner.Step) and @alignOf(Step) == @alignOf(Planner.Step));
        const typed_steps: []Planner.Step = @ptrCast(f.steps[first..][0..n]);
        _ = try Planner.build(a, parents, typed_steps, @intCast(first), limit);
        // Exercise high register indices as well as the planner's normal dense slot numbering.
        if (stream == 1) for (f.steps[first..][0..n]) |*step| {
            if (step.read_slot >= 0) step.read_slot = @as(i32, @intCast(limit - 1)) - step.read_slot;
            step.write_slot = @as(i32, @intCast(limit - 1)) - step.write_slot;
        };
        try Planner.validate(parents, typed_steps, @intCast(first), limit);
    }
    try reference(a, &f, n);
    return f;
}
pub fn reference(a: std.mem.Allocator, f: *Fixture, n: usize) !void {
    const states = try a.alloc(f64, n * 128);
    const errors = try a.alloc(f64, n * 128);
    const u = c.unit(.f32);
    const tree = c.gamma(132, u);
    for (f.segments) |span| for (0..48) |head| for (0..128) |dim| {
        for (0..n) |r| {
            const row = span.first + r;
            const kh = head / 3;
            const at = row * 48 + head;
            var dot: f64 = 0;
            var dot_mag: f64 = 0;
            var dot_err: f64 = 0;
            const parent = f.parents[row];
            for (0..128) |k| {
                const si = (span.state_slot * 48 + head) * 128 * 128 + dim * 128 + k;
                const initial: f64 = if (parent < 0) f.state[si] else states[@as(usize, @intCast(parent)) * 128 + k];
                const error_before = if (parent < 0) 0 else errors[@as(usize, @intCast(parent)) * 128 + k];
                const state = initial * f.decay[at];
                const err = error_before * @abs(f.decay[at]) + u * @abs(state);
                const key = v(f.key[(row * 16 + kh) * 128 + k]);
                states[r * 128 + k] = state;
                errors[r * 128 + k] = err;
                dot += state * key;
                dot_mag += @abs(state * key);
                dot_err += err * @abs(key);
            }
            dot_err += tree * dot_mag;
            const incoming = v(f.value[at * 128 + dim]);
            const mix = v(f.mixing[at]);
            const correction = (incoming - dot) * mix;
            const correction_err = (dot_err + u * (@abs(incoming) + @abs(dot))) * @abs(mix) + u * @abs(correction);
            var result: f64 = 0;
            var mag: f64 = 0;
            var err: f64 = 0;
            for (0..128) |k| {
                const key = v(f.key[(row * 16 + kh) * 128 + k]);
                const before = states[r * 128 + k];
                const term = key * correction;
                const next = before + term;
                const next_err = errors[r * 128 + k] + @abs(key) * correction_err + c.gamma(2, u) * (@abs(before) + @abs(term));
                states[r * 128 + k] = next;
                errors[r * 128 + k] = next_err;
                const query = v(f.query[(row * 16 + kh) * 128 + k]);
                result += next * query;
                mag += @abs(next * query);
                err += next_err * @abs(query);
                if (f.p.advance != 0 and n == 1) {
                    const si = (span.state_slot * 48 + head) * 128 * 128 + dim * 128 + k;
                    f.advanced[si] = next;
                    f.state_bound[si] = next_err;
                }
            }
            err += tree * mag;
            f.output[at * 128 + dim] = result;
            f.bound[at * 128 + dim] = err + c.unit(.bf16) * (@abs(result) + err) + 0x1p-133;
        }
    };
}
