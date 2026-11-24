//! Host-only proof of Mini's glue formulas, source inventory and shape admission.
const std = @import("std");
const p = @import("inkling2_glue_plan");
const records = @import("core").contract.inkling2_glue_metal_records.records;
const shader = @import("inkling2_metal").glue;
const c = @import("core").contract.spec;
const t = std.testing;

fn near(want: f64, got: f64) !void {
    try t.expectApproxEqAbs(want, got, 1e-10);
}

fn expectMaxThreads(threads: u32, symbol: []const u8) !void {
    var hdr: [96]u8 = undefined;
    const reserved = try std.fmt.bufPrint(&hdr, "[[max_total_threads_per_threadgroup({d})]]\nkernel void {s}(", .{ threads, symbol });
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, reserved));
}

test "glue entries have one shader, symbol, unique launch record and honest scalar reservation" {
    const kinds = std.enums.values(p.Kind);
    for (kinds, records[0..kinds.len]) |kind, record| {
        try t.expectEqualStrings(kind.label(), record.name);
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, kind.symbol()));
        try t.expectEqual(record.threads, record.reserved);
        try expectMaxThreads(record.threads, kind.symbol());
    }
    try t.expect(std.mem.indexOf(u8, shader, "#pragma METAL fp contract(off)") != null);
    try t.expect(std.mem.indexOf(u8, shader, "INKLING_AR_SCALE = 0x1.555556p-6f") != null);
    try t.expectEqual(@as(u32, 256), records[@backingInt(p.Kind.conv_dynamic)].threads);
    try t.expectEqual(@as(u32, 128), records[@backingInt(p.Kind.conv_dynamic)].tg_memory);
}

test "the split router's two launches follow the kinds' records, one symbol and one scalar record each" {
    const split = records[std.enums.values(p.Kind).len..];
    try t.expectEqual(@as(usize, 2), split.len);
    for (split, [_][]const u8{ "tf_inkling2_route_scores", "tf_inkling2_route_select" }) |record, symbol| {
        try t.expect(std.mem.endsWith(u8, record.name, symbol));
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, symbol));
        try t.expectEqual(@as(u32, 32), record.threads);
        try t.expectEqual(record.threads, record.reserved);
        try expectMaxThreads(record.threads, symbol);
    }
}

test "shape admission fixes Mini's hidden and routed dimensions" {
    try t.expect(p.valid(.embed_norm, .{ .rows = 4, .n = p.hidden, .k = p.hidden }));
    try t.expect(!p.valid(.ar, .{ .rows = 6, .n = p.hidden, .k = p.hidden }));
    try t.expect(p.valid(.cached_ar, .{ .rows = 1, .n = p.hidden, .k = p.hidden }));
    try t.expect(p.valid(.cached_ar, .{ .rows = 5, .n = 3 * p.hidden, .k = p.hidden }));
    try t.expect(!p.valid(.cached_ar, .{ .rows = 0, .n = p.hidden, .k = p.hidden }));
    try t.expect(!p.valid(.cached_ar, .{ .rows = 6, .n = p.hidden, .k = p.hidden }));
    try t.expect(!p.valid(.cached_ar, .{ .rows = 2, .n = 4 * p.hidden, .k = p.hidden }));
    try t.expect(p.valid(.block_inv_rms, .{ .rows = 64, .n = 4, .k = p.hidden }));
    try t.expect(!p.valid(.block_inv_rms, .{ .rows = 1, .n = 0, .k = p.hidden }));
    try t.expect(!p.valid(.block_inv_rms, .{ .rows = 1, .n = 5, .k = p.hidden }));
    try t.expectEqual(@as(usize, 7 * 4), p.count(.block_inv_rms, .{ .rows = 7, .n = 2, .k = p.hidden }));
    try t.expect(!p.valid(.router, .{ .rows = 1, .n = 8, .k = 128 }));
    try t.expect(!p.valid(.conv_dynamic, .{ .rows = 1, .n = 2304, .k = 4 }));
    try t.expect(!p.valid(.swiglu, .{ .rows = 1, .n = p.intermediate + 1, .k = 2 }));
    try t.expect(p.valid(.logits, .{ .rows = 1, .n = 200058, .k = p.hidden }));
    try t.expectEqual(@as(usize, 8 * 1152), p.count(.expert_pick, .{ .rows = 1, .n = 1152, .k = 8 }));
}

test "embedding uses one FP32 RMS, weighted product and final BF16 rounding" {
    var x: [p.hidden]f32 = @splat(1);
    var w: [p.hidden]f32 = @splat(2);
    try near(@as(f64, 2) / @sqrt(@as(f64, 1) + 1e-6), p.embedValue(&x, &w, 17));
    x[3] = 3;
    try t.expect(p.embedValue(&x, &w, 3) != 2);
    try t.expectEqual(@as(f64, 1), p.roundBf16(1.003));
}

test "AR normalizes each source, scales learned FP32 query dots after reduction and softmaxes" {
    const a = t.allocator;
    const blocks = try a.alloc(f32, p.hidden);
    defer a.free(blocks);
    var partial: [p.hidden]f32 = @splat(2);
    var queries: [p.hidden]f32 = @splat(0);
    @memset(blocks, 1);
    const out = try a.alloc(f64, p.hidden);
    defer a.free(out);
    p.blockResidual(blocks, &partial, &queries, 2, 0, out);
    const expected = (1 / @sqrt(@as(f64, 1) + 1e-6) + 2 / @sqrt(@as(f64, 4) + 1e-6)) / 2;
    try near(expected, out[0]);
    try near(expected, out[1]);
    queries[0] = 48;
    p.blockResidual(blocks, &partial, &queries, 2, 0, out);
    // The first dot is 48/sqrt(2304+eps), multiplied by 1/48 only after that dot.
    try t.expect(out[0] < 1.1 and out[0] > 0.9);
    try t.expectEqual(p.ar_scale, @as(f64, @as(f32, @bitCast(@as(u32, 0x3caaaaab)))));
}

test "AR query logits use each normalized source before softmax" {
    var blocks: [p.hidden]f32 = @splat(0);
    var partial: [p.hidden]f32 = @splat(0);
    var queries: [p.hidden]f32 = @splat(0);
    var out: [p.hidden]f64 = undefined;
    blocks[0] = 1;
    partial[1] = 1;
    queries[0] = 1;
    p.blockResidual(&blocks, &partial, &queries, 2, 0, &out);
    const inv = 1 / @sqrt(@as(f64, 1) / p.hidden + 1e-6);
    const first = 1 / (1 + @exp(-inv * p.ar_scale));
    try near(first * inv, out[0]);
    try near((1 - first) * inv, out[1]);
}

test "AR FP32 query-dot model distinguishes partial precision and order" {
    const blocks: [0]f32 = .{};
    var partial: [p.hidden]f32 = @splat(1);
    var queries: [p.hidden]f32 = @splat(0.1);
    const sequential = p.arQueryDot(&blocks, &partial, &queries, 1, 0, 0, 1, .{ .accum = .f32, .order = .sequential });
    var expected: f32 = 0;
    for (0..p.hidden) |d| expected += queries[d] * partial[d];
    try t.expectEqual(@as(f64, expected), sequential);

    const pairwise = p.arQueryDot(&blocks, &partial, &queries, 1, 0, 0, 1, .{ .accum = .f32, .order = .pairwise });
    try t.expect(pairwise != sequential);
    const bf16 = p.arQueryDot(&blocks, &partial, &queries, 1, 0, 0, 1, .{ .accum = .bf16, .order = .sequential });
    var lowered: f64 = 0;
    for (0..p.hidden) |d| {
        const term = p.roundBf16(@as(f64, queries[d]) * partial[d]);
        lowered = p.roundBf16(lowered + term);
    }
    try t.expectEqual(lowered, bf16);
    try t.expect(bf16 != sequential);
}

test "AR integer stress meets the exact FP32 gate and rejects lower-precision sums" {
    const sources = 4;
    var blocks: [(sources - 1) * p.hidden]f32 = undefined;
    for (0..sources - 1) |s| @memset(blocks[s * p.hidden ..][0..p.hidden], if (s % 2 == 0) 256 else -256);
    var partial: [p.hidden]f32 = @splat(-256);
    var queries: [2 * p.hidden]f32 = undefined;
    for (0..2) |query_row| {
        const row = queries[query_row * p.hidden ..][0..p.hidden];
        for (0..p.hidden / 2) |d| {
            const v: f32 = @floatFromInt(1 + d % 64);
            row[d] = v;
            row[p.hidden / 2 + d] = (if (d < 8) @as(f32, 1) else 0) - v;
        }
    }
    var want: [2 * p.hidden]f64 = undefined;
    for (0..2) |q| p.blockResidualIn(&blocks, &partial, &queries, sources, q, null, want[q * p.hidden ..][0..p.hidden]);
    const spec: c.Spec = .{ .name = "AR stress", .accum = .f32, .depth = undefined, .ops = 96, .out = .f32, .served = &.{} };
    var margins = [_]f64{ std.math.inf(f64), std.math.inf(f64) };
    for ([_]p.ArOrder{ .sequential, .pairwise, .strided }) |order| {
        for ([_]p.ArAccum{ .f32, .f16, .bf16 }, 0..) |accum, ai| {
            var got: [2 * p.hidden]f64 = undefined;
            for (0..2) |q| p.blockResidualIn(&blocks, &partial, &queries, sources, q, .{ .accum = accum, .order = order }, got[q * p.hidden ..][0..p.hidden]);
            var most: f64 = 0;
            for (want, got) |w, g| {
                const miss = if (g == w) 0 else if (std.math.isFinite(g) or !std.math.isFinite(w)) @abs(g - w) / c.boundExact(spec, w) else std.math.inf(f64);
                most = @max(most, miss);
            }
            if (ai == 0) try t.expect(most <= 1) else margins[ai - 1] = @min(margins[ai - 1], most);
        }
    }
    try t.expect(margins[0] >= 100);
    try t.expect(margins[1] >= 100);
}

test "block_inv_rms stress meets exact FP32 gate and discriminates FP16/BF16 in all orders" {
    var block: [p.hidden]f32 = @splat(1);
    block[0] = 1024;
    const want = p.blockInvRms(&block);
    const spec: c.Spec = .{ .name = "block_inv_rms stress", .accum = .f32, .depth = undefined, .ops = 8, .out = .f32, .served = &.{} };
    const allow = c.boundExact(spec, want);
    var margins = [_]f64{ std.math.inf(f64), std.math.inf(f64) };
    for ([_]p.ArOrder{ .sequential, .pairwise, .strided }) |order| {
        for ([_]p.ArAccum{ .f32, .f16, .bf16 }, 0..) |accum, ai| {
            const got = p.blockInvRmsIn(&block, .{ .accum = accum, .order = order });
            const miss = if (got == want) 0 else if (std.math.isFinite(got)) @abs(got - want) / allow else std.math.inf(f64);
            if (ai == 0) try t.expect(miss <= 1) else margins[ai - 1] = @min(margins[ai - 1], miss);
        }
    }
    try t.expect(margins[0] >= 100);
    try t.expect(margins[1] >= 100);
}

test "static convolution's stored tap three is current and carries the residual input" {
    const taps = [_]f32{ 1, 2, 3, 4 };
    const weights = [_]f32{ 10, 20, 30, 40 };
    // This addresses channel zero at output row zero with one channel.
    try near((1 * 10 + 2 * 20 + 3 * 30 + 4 * 40 + 4) / @as(f64, @sqrt(5.0)), p.staticConv(&taps, &weights, 1, 0, 0));
}

test "dynamic convolution rounds 32 projections to BF16 and uses lag zero for current" {
    var taps: [7]f32 = @splat(0);
    var input: [p.hidden]f32 = @splat(0);
    var projection: [32 * p.hidden]f32 = @splat(0);
    taps[3] = 2; // r=0, c=0, current
    input[0] = 1;
    projection[0] = 1;
    const static_w = [_]f32{ 5, 0, 0, 0 };
    // lag zero: (5 + BF16(1))*current + current, all divided by three.
    try near(@as(f64, 14.0) / 3, p.dynamicConv(&taps, &input, &static_w, &projection, 1, 0, 0));
    try t.expect(p.dynamicConv(&taps, &input, &static_w, &projection, 1, 0, 0) != p.staticConv(&taps, &static_w, 1, 0, 0));
}

test "router bias affects top-8 only; sigmoid weights normalize to eight after ascending IDs" {
    var x: [p.hidden]f32 = @splat(0);
    var matrix: [p.expert_count * p.hidden]f32 = @splat(0);
    var bias: [p.expert_count]f32 = @splat(0);
    for (120..128) |e| bias[e] = 1;
    matrix[127 * p.hidden] = 1;
    x[0] = 1;
    const r = p.route(&x, &matrix, &bias);
    for (r.ids, 0..) |id, j| try t.expectEqual(@as(u32, @intCast(120 + j)), id);
    var sum: f64 = 0;
    for (r.weights) |w| sum += w;
    try near(8, sum);
    try t.expect(r.weights[7] > r.weights[0]);
    bias[127] = -1;
    const changed = p.route(&x, &matrix, &bias);
    try t.expect(changed.ids[7] != 127);
}

test "interleaved SwiGLU caps gate and up separately with one BF16 activation store" {
    try near(p.cap(3, 4) * p.sigmoid(3) * p.cap(7, 25), p.swiglu(3, 7));
    try t.expect(p.swiglu(40, 100) < 100);
    try near(16.5, p.roundBf16(p.swiglu(3, 7)));
}

test "A9(a) rounds the BF16 unembed before widening to the FP32 softcap30" {
    var h: [p.hidden]f32 = @splat(0);
    var u: [p.hidden]f32 = @splat(0);
    h[0] = 1;
    u[0] = 60;
    try near(30 * std.math.tanh(@as(f64, 2)), p.logit(&h, &u, 0));
    u[0] = 1.003;
    try near(p.cap(p.roundBf16(1.003), 30), p.logit(&h, &u, 0));
    try t.expect(p.logit(&h, &u, 0) != p.cap(1.003, 30));
}

test "expert pick uses sorted IDs and combine accumulates those experts from FP32 zero" {
    var all: [p.expert_count]f32 = undefined;
    for (&all, 0..) |*v, e| v.* = @floatFromInt(e);
    const ids = [_]u32{ 2, 7, 11, 19, 40, 61, 90, 127 };
    try t.expectEqual(@as(f32, 90), p.pick(&all, &ids, 1, 0, 6, 0));
    var picked: [8]f32 = undefined;
    for (&picked, 0..) |*v, j| v.* = @floatFromInt(ids[j]);
    const weights = [_]f32{ 0, 0, 0, 0, 0, 0, 1, 0 };
    try near(90, p.combine(&picked, &weights, 1, 0));
    try near(2 * p.sigmoid(-3), p.sigmoid(-3) * 2);
}
