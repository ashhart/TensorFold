//! Device-free tests of LightOn's independent formulas, controls, image isolation and full-geometry reference.
const std = @import("std");
const core = @import("core");
const ref = @import("contract_lighton_ref.zig");
const bf = core.bf16_round;
const c = core.contract.spec;

test "LightOn HIP reference: every production smoke fixture is finite and its ranges contain the point" {
    for (ref.entries) |entry| {
        if (entry.control) continue;
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const shape = ref.spec(entry, .smoke).served[0];
        const d = try ref.Data.init(arena.allocator(), entry, shape, .random, 3);
        for (d.want, d.low, d.high) |want, lo, hi| {
            try std.testing.expect(std.math.isFinite(want));
            try std.testing.expect(want >= lo and want <= hi);
        }
    }
}

test "LightOn HIP reference: nonfinite fixtures have both finite and nonfinite expected outputs" {
    for (ref.entries) |entry| {
        if (entry.control) continue;
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        var shape = ref.spec(entry, .smoke).served[0];
        shape.rows = std.math.clamp(shape.rows, 5, 8);
        const d = try ref.Data.init(arena.allocator(), entry, shape, .nonfinite, 7);
        var count: usize = 0;
        for (d.want) |value| if (!std.math.isFinite(value)) {
            count += 1;
        };
        try std.testing.expect(count > 0 and count < d.want.len);
    }
}

fn bruteAttention(d: ref.Data, head: usize, query: usize, col: usize, leak: bool) f64 {
    const count = if (leak) d.total else d.shape.k;
    const start = if (leak) 0 else d.start;
    var scores: [130]f64 = undefined;
    std.debug.assert(count <= scores.len);
    var top = -std.math.inf(f64);
    for (scores[0..count], 0..) |*score, i| {
        score.* = 0;
        for (0..64) |j| score.* += @as(f64, bf.value(d.b[0][(head * d.total + query) * 64 + j])) * 0.125 * bf.value(d.b[1][(head * d.total + start + i) * 64 + j]);
        top = @max(top, score.*);
    }
    var denominator: f64 = 0;
    var numerator: f64 = 0;
    for (scores[0..count], 0..) |score, i| {
        const probability = @exp(score - top);
        denominator += probability;
        numerator += probability * bf.value(d.b[2][(head * d.total + start + i) * 64 + col]);
    }
    return numerator / denominator;
}

test "LightOn HIP reference: periodic all-output attention equals an independent quadratic small reference" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const d = try ref.Data.init(arena.allocator(), .{ .kind = .attention, .span = true }, .{ .rows = 3, .n = 64, .k = 9 }, .random, 11);
    for (0..d.shape.k) |row| for (0..d.shape.rows) |head| for (0..64) |col| {
        const at = (row * d.shape.rows + head) * 64 + col;
        try std.testing.expectApproxEqAbs(bruteAttention(d, head, d.start + row, col, false), d.want[at], 1e-14);
    };
    var rejected: usize = 0;
    for (0..d.shape.rows) |head| for (0..64) |col| {
        const wrong = bf.bf16(bruteAttention(d, head, d.start, col, true));
        const at = head * 64 + col;
        if (!(wrong >= d.low[at] and wrong <= d.high[at])) rejected += 1;
    };
    try std.testing.expect(rejected > 0);
}

test "LightOn HIP reference: full 12100-patch geometry has a finite reference for every output" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const d = try ref.Data.init(arena.allocator(), .{ .kind = .attention }, .{ .rows = 2, .n = 64, .k = 12100 }, .random, 13);
    try std.testing.expectEqual(@as(usize, 12100 * 2 * 64), d.want.len);
    for (d.want) |value| try std.testing.expect(std.math.isFinite(value));
    try std.testing.expectEqualSlices(f64, d.want[0..128], d.want[8 * 128 .. 9 * 128]);
}

test "LightOn HIP reference: exact GELU and RMS rounding controls leave their independent ranges" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const gelu = try ref.Data.init(a, .{ .kind = .gelu }, .{ .rows = 1, .n = 1024, .k = 1 }, .random, 17);
    var missed: usize = 0;
    for (gelu.b[0], gelu.low, gelu.high) |word, lo, hi| {
        const x: f64 = bf.value(word);
        const wrong = bf.bf16(0.5 * x * (1 + std.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))));
        if (!(wrong >= lo and wrong <= hi)) missed += 1;
    }
    try std.testing.expect(missed > 0);
    const norm = try ref.Data.init(a, .{ .kind = .rms }, .{ .rows = 1, .n = 1024, .k = 1 }, .random, 19);
    var squares: f64 = 0;
    for (norm.b[0]) |word| {
        const x: f64 = bf.value(word);
        squares += x * x;
    }
    const inv = 1 / @sqrt(squares / 1024.0 + 1e-5);
    missed = 0;
    for (norm.b[0], norm.f[0], norm.low, norm.high) |word, weight, lo, hi| {
        const wrong = bf.bf16(@as(f64, bf.value(word)) * inv * weight);
        if (!(wrong >= lo and wrong <= hi)) missed += 1;
    }
    try std.testing.expect(missed > 0);
}

test "LightOn HIP reference: text rotary control loses the fixed BF16 branch witness" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const d = try ref.Data.init(arena.allocator(), .{ .kind = .text_qk }, .{ .rows = 1, .n = 16, .k = 128 }, .random, 23);
    const cs = bf.bf16(@cos(@as(f64, d.f[2][0])));
    const sn = bf.bf16(@sin(@as(f64, d.f[2][0])));
    const wrong = bf.bf16(@as(f64, d.f[0][0]) * cs - @as(f64, d.f[0][64]) * sn);
    try std.testing.expect(!(wrong >= d.low[0] and wrong <= d.high[0]));
}

test "LightOn HIP reference: norm and text stress detect FP16 square overflow in every sum order" {
    for ([_]ref.Kind{ .rms, .text_qk }) |kind| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const shape: c.Shape = if (kind == .rms) .{ .rows = 4, .n = 1024, .k = 1 } else .{ .rows = 4, .n = 16, .k = 128 };
        const d = try ref.Data.init(a, .{ .kind = kind }, shape, .stress, 29);
        const out = try a.alloc(f64, d.want.len);
        const mag = try a.alloc(f64, out.len);
        const low = try a.alloc(f64, out.len);
        const high = try a.alloc(f64, out.len);
        for ([_]core.contract.rows.Order{ .sequential, .pairwise, .strided }) |order| {
            try d.calculate(.f32, order, out, mag, low, high);
            try std.testing.expectEqualSlices(f64, d.want, out);
            try d.calculate(.f16, order, out, mag, low, high);
            var worst: f64 = 0;
            for (d.want, out) |want, got| worst = @max(worst, @abs(got - want) / c.boundExact(ref.spec(d.e, .full), want));
            try std.testing.expect(worst >= 40);
        }
    }
}
