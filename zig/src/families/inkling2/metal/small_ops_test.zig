//! Host checks for Inkling 2 Small Metal launch geometry and FP64 references (no GPU).
const std = @import("std");
const p = @import("inkling2_small_ops_plan");

const t = std.testing;

test "Small ops geometry validates H=4608, 9 AR sources, 384 experts and 8 block/tap slots" {
    try t.expect(p.valid(.embed_norm, .{ .rows = 1, .n = 4608, .k = 4608 }));
    try t.expect(!p.valid(.embed_norm, .{ .rows = 1, .n = 2304, .k = 2304 }));
    try t.expect(p.valid(.ar, .{ .rows = 9, .n = 4608, .k = 4608 }));
    try t.expect(!p.valid(.ar, .{ .rows = 10, .n = 4608, .k = 4608 }));
    try t.expect(p.valid(.conv_dynamic, .{ .rows = 4, .n = 1024, .k = 4 }));
    try t.expect(p.valid(.router, .{ .rows = 4, .n = 8, .k = 4608 }));
    try t.expect(p.valid(.logits, .{ .rows = 1, .n = 200058, .k = 4608 }));
    try t.expect(p.valid(.attention_partial, .{ .rows = 2, .n = 4608, .k = 8 }));
    try t.expect(!p.valid(.attention_partial, .{ .rows = 2, .n = 4608, .k = 9 }));
    try t.expect(p.valid(.capture_tap, .{ .rows = 2, .n = 4608, .k = 7 }));
    try t.expect(!p.valid(.capture_tap, .{ .rows = 2, .n = 4608, .k = 8 }));
}

test "Small AR scale equals binary32(1/sqrt(4608)) and block/tap offsets match 8-slot layout" {
    const want: f32 = @floatCast(1.0 / @sqrt(@as(f64, 4608.0)));
    try t.expectEqual(@as(f64, want), p.ar_scale);
    try t.expectEqual(@as(usize, 0), p.blockOffset(0, 1));
    try t.expectEqual(@as(usize, 7 * 4608), p.blockOffset(0, 8));
    try t.expectEqual(@as(usize, 8 * 4608), p.blockOffset(1, 1));
    try t.expectEqual(@as(usize, 7 * 4608), p.tapOffset(0, 7));
    try t.expectEqual(@as(usize, 8 * 4608), p.tapOffset(1, 0));
}

test "Small router FP64 reference picks top-8 of 384 experts in ascending index order" {
    const a = t.allocator;
    const x = try a.alloc(f32, p.hidden);
    defer a.free(x);
    const w = try a.alloc(f32, p.expert_count * p.hidden);
    defer a.free(w);
    const bias = try a.alloc(f32, p.expert_count);
    defer a.free(bias);
    @memset(x, 0.0);
    @memset(w, 0.0);
    for (bias) |*b| b.* = -10.0;
    const winners = [8]u32{ 370, 5, 250, 19, 383, 128, 300, 1 };
    for (winners, 0..) |e, j| bias[e] = 1.0 + @as(f32, @floatFromInt(j)) * 0.1;
    var ids: [8]u32 = undefined;
    var weights: [8]f64 = undefined;
    p.routerRef(x, w, bias, 1, &ids, &weights);
    try t.expectEqualSlices(u32, &.{ 1, 5, 19, 128, 250, 300, 370, 383 }, &ids);
    var sum: f64 = 0;
    for (weights) |v| sum += v;
    try t.expectApproxEqAbs(8.0, sum, 1e-6);
}
