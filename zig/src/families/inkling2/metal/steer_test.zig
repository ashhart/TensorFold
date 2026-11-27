//! Host checks of --glp on Inkling 2 Mini: the visit ids a vector may name, the steered rounding order, its launches.
const std = @import("std");
const steer = @import("steer.zig");
const p = @import("runtime_plan.zig");
const records = @import("core").contract.inkling2_steer_metal_records.records;
const shader = @import("inkling2_metal").steer;
const t = std.testing;

fn dir(v: f32) [steer.hidden]f32 {
    return @splat(v);
}

test "a vector names decoder visits 1 to 47, each 2304 wide" {
    const d = dir(1);
    const ok = [_]steer.Visit{ .{ .id = 1, .dir = &d }, .{ .id = 6, .dir = &d }, .{ .id = 18, .dir = &d }, .{ .id = 47, .dir = &d } };
    try steer.check(.{ .alpha = 1, .visits_total = 48, .visits = &ok });
    try t.expectError(error.BadSteerLayer, steer.check(.{ .alpha = 1, .visits_total = 48, .visits = &.{.{ .id = 0, .dir = &d }} }));
    try t.expectError(error.BadSteerLayer, steer.check(.{ .alpha = 1, .visits_total = 48, .visits = &.{.{ .id = 48, .dir = &d }} }));
    try t.expectError(error.BadSteerWidth, steer.check(.{ .alpha = 1, .visits_total = 48, .visits = &.{.{ .id = 3, .dir = d[1..] }} }));
    try t.expectError(error.BadSteerShape, steer.check(.{ .alpha = 1, .visits_total = 48, .visits = &.{} }));
}

test "the steered output rounds the residual first and the product on its own" {
    // a zero direction, or strength 0, leaves main's mlp_partial bits
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var contracted: usize = 0;
    for (0..100_000) |_| {
        const tv = rng.float(f32) - 0.5;
        const x: f32 = @floatCast(p.roundBf16(rng.float(f32) - 0.5));
        const d = rng.float(f32) * 2 - 1;
        try t.expectEqual(p.mlpPartial(tv, x), p.mlpPartialSteer(tv, x, 1, 0));
        try t.expectEqual(p.mlpPartial(tv, x), p.mlpPartialSteer(tv, x, 0, d));
        // the fused op is main's mlp_partial followed by the reference's add
        try t.expectEqual(p.steerAdd(@floatCast(p.mlpPartial(tv, x)), 0.1, d), p.mlpPartialSteer(tv, x, 0.1, d));
        const head: u16 = @truncate(@as(u32, @bitCast(-(@as(f32, 0.1) * d))) >> 16);
        const hx: f32 = @bitCast(@as(u32, head) << 16);
        contracted += @intFromBool(p.mlpPartialSteer(0, hx, 0.1, d) != p.mlpPartialSteerContracted(0, hx, 0.1, d));
    }
    // rows built from alpha*d's own head show the fused multiply-add's difference often enough to plant controls on
    try t.expect(contracted > 1000);
}

test "a zero direction keeps a negative-zero residual's bits, in the unfused sum and the add" {
    const neg: f32 = @bitCast(@as(u32, 0x8000_0000));
    const hv: f32 = @floatCast(p.mlpPartial(neg, neg));
    const want: u32 = @bitCast(hv);
    try t.expectEqual(@as(u32, 0x8000_0000), want);
    // a +0 direction, either as strength zero or as a zero direction
    const by_strength: f32 = @floatCast(p.mlpPartialSteer(neg, neg, 0, 1));
    const by_direction: f32 = @floatCast(p.mlpPartialSteer(neg, neg, 1, 0));
    const added: f32 = @floatCast(p.steerAdd(hv, 1, 0));
    const added_neg: f32 = @floatCast(p.steerAdd(hv, 1, -0.0));
    try t.expectEqual(want, @as(u32, @bitCast(by_strength)));
    try t.expectEqual(want, @as(u32, @bitCast(by_direction)));
    try t.expectEqual(want, @as(u32, @bitCast(added)));
    try t.expectEqual(want, @as(u32, @bitCast(added_neg)));
    // a nonzero step still rounds the sum
    const nonzero: u32 = @bitCast(@as(f32, @floatCast(p.steerAdd(hv, 1, 1))));
    try t.expect(nonzero != 0x8000_0000);
}

test "each steer kernel has one launch record, its thread limit and the ordering pragma" {
    try t.expectEqualStrings(shader, @import("kernel_sources").inkling2_steer);
    try t.expectEqual(@as(usize, 3), records.len);
    for (records, [_][]const u8{ steer.symbol, steer.ar_symbol, steer.add_symbol }, [_]u32{ 0, 240, 0 }) |record, symbol, tg| {
        try t.expect(std.mem.endsWith(u8, record.name, symbol));
        try t.expectEqual(p.threads, record.threads);
        try t.expectEqual(record.threads, record.reserved);
        try t.expectEqual(tg, record.tg_memory);
        var hdr: [96]u8 = undefined;
        const reserved = try std.fmt.bufPrint(&hdr, "[[max_total_threads_per_threadgroup({d})]]\nkernel void {s}(", .{ record.threads, symbol });
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, reserved));
    }
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, "#pragma clang fp contract(off)"));
    try t.expectEqual(@as(usize, 8), @sizeOf(steer.Args));
    try t.expectEqual(@as(usize, 4), @offsetOf(steer.Args, "alpha"));
    try t.expectEqual(@as(usize, 16), @sizeOf(steer.ArArgs));
    try t.expectEqual(@as(usize, 12), @offsetOf(steer.ArArgs, "alpha"));
}

test "DSpark never steers: it reads the taps the steered visits already wrote" {
    // the drafter's kernels and launches name no direction, so a drafted reply applies the vector exactly once
    inline for (.{ "dspark.zig", "dspark_kernels.zig", "dspark_state.zig", "dspark_fold.zig", "dspark_sample.zig", "dspark_snap.zig", "dspark_weights.zig" }) |file| {
        const text = @embedFile(file);
        try t.expect(std.mem.indexOf(u8, text, "steer") == null);
        try t.expect(std.mem.indexOf(u8, text, "mlp_partial") == null);
    }
    try t.expect(std.mem.indexOf(u8, @import("inkling2_metal").inkling2_dspark, "steer") == null);
    // the target's forward swaps in a steered launch at each of main's three fused sites and its unfused one
    const forward = @embedFile("forward.zig");
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, forward, "m.steer.?.partial("));
    try t.expectEqual(@as(usize, 3), std.mem.count(u8, forward, "m.steer.?.partialAr("));
    try t.expectEqual(@as(usize, 3), std.mem.count(u8, forward, "glue.mlpPartialAr("));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, forward, "const prev_fused = vi > 0 and canPreFuse(v);"));
}
