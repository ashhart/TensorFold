//! Independent smooth losses check selected coordinates of the local binary64 adjoints.
const std = @import("std");
const ref = @import("learning_reference.zig");
const t = std.testing;
fn value(f: ref.Fixture, slot: usize, i: usize, changed_slot: usize, at: usize, delta: f64) f64 {
    const raw = if (slot == 0 or (f.buffers.len == 9 and slot == 1)) ref.scalar(f.buffers[slot], u16, i) else ref.scalar(f.buffers[slot], f32, i);
    return raw + if (slot == changed_slot and i == at) delta else @as(f64, 0);
}
fn pocketLoss(f: ref.Fixture, at: usize, delta: f64) f64 {
    var loss: f64 = 0;
    for (0..3) |r| for (0..3) |j| {
        var g: f64 = 0;
        var u: f64 = 0;
        var e: f64 = 0;
        for (0..65) |d| {
            const x = value(f, 0, r * 65 + d, 0, at, delta);
            g += x * ref.scalar(f.buffers[1], f32, j * 65 + d);
            u += x * ref.scalar(f.buffers[1], f32, (64 + j) * 65 + d);
            e += ref.scalar(f.buffers[2], f32, r * 65 + d) * ref.scalar(f.buffers[1], f32, (128 + j) * 65 + d);
        }
        loss += e * u * g / (1 + @exp(-g)) * (if (j >= 1) @as(f64, 0.25) else 1);
    };
    return loss;
}
fn prepLoss(f: ref.Fixture, slot: usize, at: usize, delta: f64) f64 {
    var loss: f64 = 0;
    for (0..3) |r| for (0..3) |h| {
        const query = h < 2;
        const head = if (query) h else h - 2;
        const raw_slot: usize = if (query) 0 else 1;
        const raw_at = if (query) r * 1024 + head * 512 else r * 256;
        const grad_at = if (query) r * 512 + head * 256 else r * 256;
        var norm: f64 = 0;
        var y: [256]f64 = undefined;
        for (0..256) |d| {
            const x = value(f, raw_slot, raw_at + d, slot, at, delta);
            norm += x * x;
        }
        const eps: f32 = 1e-6;
        norm = @sqrt(norm / 256 + eps);
        for (&y, 0..) |*out, d| out.* = value(f, raw_slot, raw_at + d, slot, at, delta) * ref.scalar(f.buffers[if (query) 2 else 3], f32, d) / norm;
        for (0..256) |d| {
            var out = y[d];
            if (d < 64) {
                const angle = @as(f64, @floatFromInt(17 + r)) * ref.scalar(f.buffers[4], f32, d % 32);
                out = y[d] * @cos(angle) + (if (d < 32) @as(f64, -1) else 1) * y[if (d < 32) d + 32 else d - 32] * @sin(angle);
            }
            loss += out * ref.scalar(f.buffers[if (query) 5 else 6], f32, grad_at + d);
        }
    };
    return loss;
}
fn scanLoss(a: std.mem.Allocator, f: ref.Fixture, slot: usize, at: usize, delta: f64) !f64 {
    const size = 2 * 128 * 128;
    const state = try a.alloc(f64, size);
    defer a.free(state);
    const base = try a.alloc(f64, size);
    defer a.free(base);
    for (state, base, 0..) |*s, *b, i| {
        s.* = ref.scalar(f.buffers[5], f32, i);
        b.* = s.*;
    }
    var loss: f64 = 0;
    for (0..5) |r| {
        if (r == 4) for (state, base, 0..) |*s, *b, i| {
            // Freeze each saved checkpoint's offset at the base point; cuts have a local straight-through adjoint.
            const checkpoint = ref.scalar(f.buffers[5], f32, size + i);
            s.* += checkpoint - b.*;
            b.* = checkpoint;
        };
        for (0..2) |h| for (0..128) |v| {
            const offset = (h * 128 + v) * 128;
            const g = value(f, 3, r * 2 + h, slot, at, delta);
            const mix = value(f, 4, r * 2 + h, slot, at, delta);
            var prediction: f64 = 0;
            var base_prediction: f64 = 0;
            for (0..128) |d| {
                prediction += state[offset + d] * g * value(f, 2, r * 128 + d, slot, at, delta);
                base_prediction += base[offset + d] * ref.scalar(f.buffers[3], f32, r * 2 + h) * ref.scalar(f.buffers[2], f32, r * 128 + d);
            }
            const error_value = value(f, 0, r * 512 + 256 + h * 128 + v, slot, at, delta) - prediction;
            const base_error = ref.scalar(f.buffers[0], u16, r * 512 + 256 + h * 128 + v) - base_prediction;
            var y: f64 = 0;
            for (0..128) |d| {
                state[offset + d] = state[offset + d] * g + value(f, 2, r * 128 + d, slot, at, delta) * mix * error_value;
                base[offset + d] = base[offset + d] * ref.scalar(f.buffers[3], f32, r * 2 + h) + ref.scalar(f.buffers[2], f32, r * 128 + d) * ref.scalar(f.buffers[4], f32, r * 2 + h) * base_error;
                y += state[offset + d] * value(f, 1, r * 128 + d, slot, at, delta);
            }
            loss += y * ref.scalar(f.buffers[6], f32, (r * 2 + h) * 128 + v);
        };
    }
    return loss;
}
fn close(want: f64, got: f64) !void {
    try t.expect(std.math.isFinite(got));
    try t.expect(@abs(want - got) <= 1e-8 + 1e-6 * @abs(want));
}
test "Qwen27 local adjoints: pocket and weighted partial RoPE RMS match independent central slopes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try ref.make(a, .pocket);
    for ([_]usize{ 0, 32, 64, 65, 194 }) |at| for ([_]f64{ 0x1p-12, 0x1p-15, 0x1p-18 }) |step|
        try close(p.buffers[3].want.?[at] - 0.125, (pocketLoss(p, at, step) - pocketLoss(p, at, -step)) / (2 * step));
    const f = try ref.make(a, .prep);
    for ([_][2]usize{ .{ 0, 0 }, .{ 0, 32 }, .{ 0, 63 }, .{ 0, 64 }, .{ 0, 512 }, .{ 0, 2048 }, .{ 1, 0 }, .{ 1, 32 }, .{ 1, 255 }, .{ 1, 512 } }) |probe| for ([_]f64{ 0x1p-12, 0x1p-15, 0x1p-18 }) |step|
        try close(f.buffers[if (probe[0] == 0) 7 else 8].want.?[probe[1]], (prepLoss(f, probe[0], probe[1], step) - prepLoss(f, probe[0], probe[1], -step)) / (2 * step));
}
test "Qwen27 local adjoints: scan Q K V decay and mixing cross the four-row checkpoint in central slopes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try ref.make(a, .scan);
    for ([_]usize{ 0, 4 }) |row| for (0..5) |slot| {
        const at = switch (slot) {
            0 => row * 512 + 256,
            1, 2 => row * 128,
            else => row * 2,
        };
        var analytic: f64 = 0;
        if (slot == 0) {
            analytic = f.buffers[10].want.?[at];
        } else {
            for (0..4) |part| {
                if (slot == 1 or slot == 2) {
                    for (0..2) |h| analytic += f.buffers[if (slot == 1) 8 else 9].want.?[((row * 2 + h) * 4 + part) * 128];
                } else analytic += f.buffers[if (slot == 3) 11 else 12].want.?[(row * 2) * 4 + part];
            }
        }
        const step = 0x1p-15;
        try close(analytic, (try scanLoss(a, f, slot, at, step) - try scanLoss(a, f, slot, at, -step)) / (2 * step));
    };
}
