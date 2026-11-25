//! Host-only proof of Mini's loop formulas, launch ABI, source inventory and rounding.
const std = @import("std");
const p = @import("inkling2_loop_plan");
const records = @import("core").contract.inkling2_loop_metal_records.records;
const shader = @import("inkling2_metal").loop;
const t = std.testing;

fn near(want: f64, got: f64) !void {
    try t.expectApproxEqAbs(want, got, 1e-10);
}

test "loop entries have unique symbols, launch records and disabled FP contraction" {
    const kinds = std.enums.values(p.Kind);
    for (kinds, records[0..kinds.len]) |kind, record| {
        try t.expectEqualStrings(kind.label(), record.name);
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, kind.symbol()));
        try t.expectEqual(@as(u32, 256), record.threads);
        try t.expectEqual(record.threads, record.reserved);
        try t.expectEqual(p.geometry(kind, .{ .rows = 1, .n = 1, .k = 1 }).tg_memory, record.tg_memory);
    }
    try t.expect(std.mem.indexOf(u8, shader, "#pragma METAL fp contract(off)") != null);
    try t.expect(std.mem.indexOf(u8, shader, "INKLING_LOOP_SCALE = 0x1.555556p-6f") != null);
    try t.expect(std.mem.indexOf(u8, shader, "LoopMathSpecMissing") == null);
}

test "the split gates' three launches follow the kinds' records, one symbol each" {
    const split = records[std.enums.values(p.Kind).len..];
    try t.expectEqual(@as(usize, 3), split.len);
    const symbols = [_][]const u8{ "tf_inkling2_loop_norm", "tf_inkling2_loop_down", "tf_inkling2_loop_apply" };
    for (split, symbols, [_]u32{ 256, 32, 256 }) |record, symbol, threads| {
        try t.expect(std.mem.endsWith(u8, record.name, symbol));
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, symbol));
        try t.expectEqual(threads, record.threads);
        try t.expectEqual(record.threads, record.reserved);
    }
}

test "the forward takes the split router and gates from two rows, and main's kernels at one" {
    try t.expect(!p.splitRows(1));
    try t.expect(p.splitRows(2));
    try t.expect(p.splitRows(64));
}

test "shape admission and dispatch geometry fix H=2304, rank 64 and 48 messages" {
    const mc = p.Call{ .rows = 7, .n = 48 * p.hidden, .k = p.hidden };
    const gc = p.Call{ .rows = 7, .n = p.hidden, .k = p.rank };
    try t.expect(p.valid(.messages, mc));
    try t.expect(p.valid(.gates, gc));
    try t.expectEqual(@as(usize, 7 * 48), p.geometry(.messages, mc).groups);
    try t.expectEqual(@as(usize, 7), p.geometry(.gates, gc).groups);
    try t.expectEqual(@as(usize, 7 * 48 * p.hidden), p.count(.messages, mc));
    try t.expectEqual(@as(usize, 7 * p.hidden), p.count(.gates, gc));
    try t.expect(!p.valid(.messages, .{ .rows = 1, .n = 47 * p.hidden, .k = p.hidden }));
    try t.expect(!p.valid(.messages, .{ .rows = 1, .n = 48 * p.hidden, .k = 4608 }));
    try t.expect(!p.valid(.gates, .{ .rows = 1, .n = p.hidden, .k = 32 }));
    try t.expect(!p.valid(.gates, .{ .rows = 0, .n = p.hidden, .k = p.rank }));
    try t.expectEqual(@as(usize, 4), @sizeOf(p.RowsArgs));
    try t.expectEqual(@as(usize, 0), @offsetOf(p.RowsArgs, "rows"));
}

test "message with zero query is the exact mean of independently normalized banks" {
    var bank0: [p.hidden]f32 = @splat(1);
    var bank1: [p.hidden]f32 = @splat(2);
    var q_mlp: [2 * p.core_layers * p.hidden]f32 = @splat(0);
    var kv: [2 * p.core_layers * p.hidden]f32 = @splat(0);
    var out: [p.hidden]f64 = undefined;
    p.message(&bank0, &bank1, &q_mlp, &kv, 11, .v, &out);
    const expected = (@as(f64, 1) / @sqrt(@as(f64, 1) + p.eps) +
        @as(f64, 2) / @sqrt(@as(f64, 4) + p.eps)) / 2;
    try near(expected, out[0]);
    try near(expected, out[p.hidden - 1]);
    try t.expectEqual(p.message_scale, @as(f64, @as(f32, @bitCast(@as(u32, 0x3caaaaab)))));
}

test "q/mlp and k/v even-odd layouts select all four roles and scale after the dot" {
    var bank0: [p.hidden]f32 = @splat(1);
    var bank1: [p.hidden]f32 = @splat(2);
    bank1[0] = 4;
    var q_mlp: [2 * p.core_layers * p.hidden]f32 = @splat(0);
    var kv: [2 * p.core_layers * p.hidden]f32 = @splat(0);
    const layer = 5;
    const all = [_]p.Role{ .q, .k, .v, .mlp };
    for (all) |role| {
        const expected_row = @as(usize, 2 * layer) + @as(usize, @intFromBool(role == .v or role == .mlp));
        try t.expectEqual(expected_row, p.queryRow(role, layer));
        const matrix = if (p.usesQueryMessageTensor(role)) &q_mlp else &kv;
        matrix[expected_row * p.hidden] = 48;
    }
    var first: [p.hidden]f64 = undefined;
    p.message(&bank0, &bank1, &q_mlp, &kv, layer, all[0], &first);
    const scale0 = p.rmsScale(&bank0);
    const scale1 = p.rmsScale(&bank1);
    const l0 = 48 * scale0 * p.message_scale;
    const l1 = 48 * 4 * scale1 * p.message_scale;
    const peak = @max(l0, l1);
    const e0 = @exp(l0 - peak);
    const e1 = @exp(l1 - peak);
    const expected = e0 / (e0 + e1) * scale0 + e1 / (e0 + e1) * 4 * scale1;
    try near(expected, first[0]);
    try t.expect(first[0] != (scale0 + 4 * scale1) / 2);
    var out: [p.hidden]f64 = undefined;
    for (all[1..]) |role| {
        p.message(&bank0, &bank1, &q_mlp, &kv, layer, role, &out);
        for (first, out) |a, b| try near(a, b);
    }
}

test "message stress dots are exact in FP32 and discriminate BF16 accumulation" {
    var bank0: [p.hidden]f32 = @splat(256);
    var bank1: [p.hidden]f32 = @splat(-256);
    var q_mlp: [2 * p.core_layers * p.hidden]f32 = @splat(0);
    var kv: [2 * p.core_layers * p.hidden]f32 = @splat(0);
    const query = q_mlp[0..p.hidden];
    for (0..p.hidden / 2) |d| {
        const v: f32 = @floatFromInt(1 + d % 64);
        const nudge: f32 = if (d < 8) 1 else 0;
        query[d] = v;
        query[p.hidden / 2 + d] = nudge - v;
    }
    var fp: f32 = 0;
    var bf: f64 = 0;
    for (query) |v| {
        fp += v;
        bf = p.roundBf16(bf + v);
    }
    try t.expectEqual(@as(f32, 8), fp);
    try t.expect(bf != 8);
    var out: [p.hidden]f64 = undefined;
    p.message(&bank0, &bank1, &q_mlp, &kv, 0, .q, &out);
    const expected = std.math.tanh(8 * p.message_scale);
    try near(expected, out[0]);
    try near(expected, out[p.hidden - 1]);
}

test "rank gate hand witness shares one BF16 zeta and uses FP32 tanh projections" {
    var h: [p.hidden]f32 = @splat(1);
    var m: [p.hidden]f32 = @splat(2);
    var down: [p.rank * 2 * p.hidden]f32 = @splat(0);
    var forget: [p.hidden * p.rank]f32 = @splat(0);
    var gain: [p.hidden * p.rank]f32 = @splat(0);
    down[0] = 1;
    for (0..p.hidden) |d| {
        forget[d * p.rank] = 1;
        gain[d * p.rank] = 1;
    }
    var zeta: [p.rank]f64 = undefined;
    p.downProjection(&h, &m, &down, &zeta);
    try near(1, zeta[0]);
    for (zeta[1..]) |z| try near(0, z);
    var out: [p.hidden]f64 = undefined;
    p.gate(&h, &m, &down, &forget, &gain, &out);
    const expected = p.blend(std.math.tanh(@as(f64, 1)), std.math.tanh(@as(f64, 1)), 1, 2);
    try near(expected, out[0]);
    try near(expected, out[p.hidden - 1]);

    // The message half starts at D column H; it is not interleaved with h.
    @memset(&down, 0);
    down[p.hidden] = 1;
    p.downProjection(&h, &m, &down, &zeta);
    try near(1, zeta[0]);
    down[p.hidden] = 0;
    down[0] = 1;
    @memset(&m, 0);
    p.downProjection(&h, &m, &down, &zeta);
    try near(1, zeta[0]);
}

test "concatenated normalized gate inputs round to BF16 before D accumulation" {
    var h: [p.hidden]f32 = @splat(0);
    var m: [p.hidden]f32 = @splat(0);
    var down: [p.rank * 2 * p.hidden]f32 = @splat(0);
    for (&h, 0..) |*v, d| v.* = @as(f32, @floatFromInt(d % 19 + 1)) / 32;
    for (0..p.hidden) |d| down[d] = 1;
    var zeta: [p.rank]f64 = undefined;
    p.downProjection(&h, &m, &down, &zeta);
    // Fixed witness rounds to 2024; accumulating unrounded normalized values would round to 2016.
    try near(2024, zeta[0]);
}

test "gate stress projections are exact in FP32 and expose BF16 sums" {
    var h: [p.hidden]f32 = @splat(256);
    var m: [p.hidden]f32 = @splat(-256);
    var down: [p.rank * 2 * p.hidden]f32 = @splat(0);
    var forget: [p.hidden * p.rank]f32 = @splat(0);
    var gain: [p.hidden * p.rank]f32 = @splat(0);
    for (0..p.rank) |r| for (0..p.hidden) |d| {
        const w: f32 = @floatFromInt(1 + (r + d) % 64);
        down[r * 2 * p.hidden + d] = w;
        down[r * 2 * p.hidden + p.hidden + d] = if (d == 0) w - 1 else w;
    };
    for ([_][]f32{ &forget, &gain }) |weights| for (0..p.hidden) |d| for (0..p.rank / 2) |r| {
        const w: f32 = @floatFromInt(1 + (d + r) % 7);
        weights[d * p.rank + r] = w;
        weights[d * p.rank + r + p.rank / 2] = if (r == 0) 1 - w else -w;
    };
    var fp: f32 = 0;
    var bf: f64 = 0;
    for (0..2 * p.hidden) |d| {
        const term = @as(f64, down[d]) * (if (d < p.hidden) @as(f64, 1) else -1);
        fp += @as(f32, @floatCast(term));
        bf = p.roundBf16(bf + term);
    }
    try t.expectEqual(@as(f32, 1), fp);
    try t.expect(bf != 1);
    var zeta: [p.rank]f64 = undefined;
    p.downProjection(&h, &m, &down, &zeta);
    for (zeta) |z| try near(1, z);
    var out: [p.hidden]f64 = undefined;
    p.gate(&h, &m, &down, &forget, &gain, &out);
    const expected = p.blend(std.math.tanh(@as(f64, 1)), std.math.tanh(@as(f64, 1)), 256, -256);
    for (out) |v| try near(expected, v);
}

test "BF16 nearest-even and all four binary32 blend steps have exact witnesses" {
    try near(1, p.roundBf16(1.00390625)); // midpoint: even mantissa is zero
    try near(1.015625, p.roundBf16(1.01171875)); // midpoint: two is even
    const f: f32 = 0.33333334;
    const g: f32 = 0.14285715;
    const h: f32 = 1.003;
    const m: f32 = 1.007;
    const keep: f32 = 1.0 - f;
    const left: f32 = keep * h;
    const right: f32 = g * m;
    const rounded: f32 = left + right;
    try t.expectEqual(@as(f64, rounded), p.blend(f, g, h, m));
    try t.expect(p.blend(f, g, h, m) != (1 - @as(f64, f)) * h + @as(f64, g) * m);
}
