//! Offset groups that used to move at the next save, exercised through the real pocket and writer preparation.
const std = @import("std");
const t = std.testing;
const families = @import("families.zig");
const shards = @import("shards.zig");

fn same(a: []const f32, b: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

fn Cases(comptime f: @import("family.zig").Family) type {
    return struct {
        const Trainer = struct {};
        const P = @import("pocket.zig").Pocket(Trainer, f);
        const Host = struct {
            pub fn pocketSave(_: @This(), layers: []const usize, neurons: usize, up: []f32, down: []f32) !usize {
                const n = up.len / layers.len;
                for (layers, 0..) |_, j| try shards.round(f, t.allocator, n / neurons, &.{up[j * n ..][0..n]}, down[j * n ..][0..n]);
                return 1;
            }
            pub fn pocketSaveSwiglu(_: @This(), layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
                const n = up.len / layers.len;
                for (layers, 0..) |_, j| try shards.round(f, t.allocator, n / neurons, &.{ gate[j * n ..][0..n], up[j * n ..][0..n] }, down[j * n ..][0..n]);
                return 1;
            }
        };
        test "learned fixed point: offset rows and output columns survive ordinary saves at checkpoint widths" {
            for ([_]usize{ f.quant.group, 2688, 4096, 5120 }) |d| {
                var p: P = .{ .gpa = t.allocator, .mode = .reply, .hidden = d };
                defer p.deinit();
                const n = f.neurons * d;
                try p.moe.append(t.allocator, 0);
                try p.used.append(t.allocator, f.neurons);
                try p.up.resize(t.allocator, n);
                try p.down.resize(t.allocator, n);
                if (P.swiglu) {
                    try p.gate.resize(t.allocator, n);
                    try p.scratch.resize(t.allocator, n);
                }
                // BF16 bias -1496, scale 3.140625: the old encoder uses no top code, then shrinks its scale.
                for (p.up.items, p.down.items, 0..) |*u, *v, i| {
                    const x = -1499.9329 + 47.0843 * @as(f32, @floatFromInt(i % f.quant.group)) / (f.quant.group - 1);
                    u.* = if (P.swiglu) x * f.neuron.sharp else x;
                    v.* = 1499.9329 - 47.0843 * @as(f32, @floatFromInt((i / d) % f.quant.group)) / (f.quant.group - 1);
                    if (P.swiglu) p.gate.items[i] = x * f.neuron.sharp;
                }
                const original = try t.allocator.dupe(f32, p.up.items);
                defer t.allocator.free(original);
                const down0 = try t.allocator.dupe(f32, p.down.items);
                defer t.allocator.free(down0);
                const gate0 = try t.allocator.dupe(f32, p.gate.items);
                defer t.allocator.free(gate0);
                try shards.roundSlot(&p, 0);
                const old = try t.allocator.dupe(f32, p.up.items);
                defer t.allocator.free(old);
                try shards.roundSlot(&p, 0);
                try t.expect(!same(old, p.up.items));
                @memcpy(p.up.items, original);
                @memcpy(p.down.items, down0);
                @memcpy(p.gate.items, gate0);
                var trainer: Trainer = .{};
                try @import("pocket_round.zig").slot(&p, &trainer, Host{}, 0);
                const up = try t.allocator.dupe(f32, p.up.items);
                defer t.allocator.free(up);
                const down = try t.allocator.dupe(f32, p.down.items);
                defer t.allocator.free(down);
                const gate = try t.allocator.dupe(f32, p.gate.items);
                defer t.allocator.free(gate);
                // Ordinary shard rounding includes both power-of-two conversions, with no tolerance or retry.
                for (0..8) |_| {
                    try shards.roundSlot(&p, 0);
                    try t.expect(same(up, p.up.items));
                    try t.expect(same(down, p.down.items));
                    try t.expect(same(gate, p.gate.items));
                }
                try t.expectEqual(@as(usize, 1), try p.save(&trainer, Host{}));
                try t.expect(!p.saved_mismatch and !p.save_pending);
                // Rounding may move the fit, but cannot erase the group to obtain a fixed point.
                try t.expect(std.mem.min(f32, up) != std.mem.max(f32, up));
            }
        }
    };
}

test {
    inline for (families.all) |f| _ = Cases(f);
}

test "learned fixed point: generated affine groups retain values and packing in the ordinary encoder" {
    const affine = @import("affine.zig");
    var prng = std.Random.DefaultPrng.init(1807);
    const r = prng.random();
    inline for (families.all) |f| {
        const q = f.quant;
        var values: [q.group]f32 = undefined;
        var once: [q.group]f32 = undefined;
        var twice: [q.group]f32 = undefined;
        var w: [affine.wordsOf(q, q.group)]u32 = undefined;
        var w2: @TypeOf(w) = undefined;
        var s: [1]u16 = undefined;
        var b: [1]u16 = undefined;
        var s2: [1]u16 = undefined;
        var b2: [1]u16 = undefined;
        for (0..20000) |trial| {
            const spread = std.math.pow(f32, 10, -38 + 70 * r.float(f32));
            const offset = spread * 256 * r.float(f32) * @as(f32, if (r.boolean()) 1 else -1);
            for (&values) |*v| v.* = switch (trial % 6) {
                0 => spread * r.floatNorm(f32),
                1, 2 => offset + spread * r.floatNorm(f32),
                3 => offset,
                4 => if (r.boolean()) -0.0 else 0.0,
                else => offset + offset * 0x1p-22 * r.floatNorm(f32),
            };
            affine.quantizeFixed(q, &values, &w, &s, &b);
            affine.dequantize(q, &w, &s, &b, &once);
            affine.quantize(q, &once, &w2, &s2, &b2);
            affine.dequantize(q, &w2, &s2, &b2, &twice);
            try t.expect(same(&once, &twice));
            try t.expectEqualSlices(u32, &w, &w2);
            try t.expectEqualSlices(u16, &s, &s2);
            try t.expectEqualSlices(u16, &b, &b2);
            // Endpoint pinning adds at most the grid's BF16 endpoint error to nearest-code rounding error.
            const lo = std.mem.min(f32, &values);
            const hi = std.mem.max(f32, &values);
            const step = affine.f32of(s[0]);
            const endpoint = @abs(affine.f32of(b[0]) - lo) + @abs(15 * step - (hi - lo));
            for (values, once) |v, x| try t.expect(@abs(v - x) <= 0.51 * step + endpoint);
        }
    }
}

test "learned fixed point: exponent seams subnormals and finite overflow stay safe" {
    const affine = @import("affine.zig");
    inline for (families.all) |f| {
        const q = f.quant;
        for ([_][2]f32{ .{ 0x1.fffffep-1, 1 }, .{ -1, -0x1.fffffep-1 }, .{ -2, -0x1.fffffep0 }, .{ 0x1p-149, 0x1p-148 } }) |edges| {
            var values: [q.group]f32 = undefined;
            for (&values, 0..) |*v, i| v.* = edges[i % 2];
            var words: [affine.wordsOf(q, q.group)]u32 = undefined;
            var scales: [1]u16 = undefined;
            var biases: [1]u16 = undefined;
            affine.quantizeFixed(q, &values, &words, &scales, &biases);
            var once: [q.group]f32 = undefined;
            affine.dequantize(q, &words, &scales, &biases, &once);
            affine.quantize(q, &once, &words, &scales, &biases);
            var again: [q.group]f32 = undefined;
            affine.dequantize(q, &words, &scales, &biases, &again);
            try t.expect(same(&once, &again));
        }
        var up: [f.neurons * q.group]f32 = @splat(0);
        var down = up;
        var gate = up;
        up[0] = -std.math.floatMax(f32);
        up[1] = std.math.floatMax(f32);
        try t.expectError(error.NonFinitePocket, @import("pocket_round.zig").layer(f, t.allocator, q.group, f.neurons, if (f.neuron.act == .swiglu) &gate else &.{}, &up, &down));
        if (f.neuron.act == .swiglu) {
            const P = @import("pocket.zig").Pocket(struct {}, f);
            var p: P = .{ .gpa = t.allocator, .mode = .reply, .hidden = q.group };
            defer p.deinit();
            try p.up.appendNTimes(t.allocator, std.math.floatMax(f32), up.len);
            try p.gate.appendNTimes(t.allocator, 0, up.len);
            try p.down.appendNTimes(t.allocator, 0, up.len);
            try p.scratch.resize(t.allocator, up.len);
            var trainer: struct {} = .{};
            try t.expectError(error.NonFinitePocket, @import("pocket_round.zig").slot(&p, &trainer, .{}, 0));
        }
    }
}
