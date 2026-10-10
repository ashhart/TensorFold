//! Native flags, byte counts and non-prefix replay are checked without a Metal device.
const std = @import("std");
const Config = @import("config.zig").Config;
const abi = @import("gdn_contract.zig");
const plan_mod = @import("gdn_plan.zig");
const Window = @import("state.zig").Window;

fn config() Config {
    return .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
}

const native = abi.WeightKinds{ .conv = .bf16, .a_log = .f32, .dt = .bf16, .norm = .f32 };

test "GDN Metal records have fixed offsets without enum padding" {
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(abi.Params));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(abi.Params, "eps"));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(abi.Segment));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(abi.Keep));
}

test "native storage keeps all sixteen combinations and refuses an implicit F16 cast" {
    for (0..16) |flags| {
        const k = abi.WeightKinds{ .conv = @fromBackingInt(@intCast(flags & 1)), .a_log = @fromBackingInt(@intCast((flags >> 1) & 1)), .dt = @fromBackingInt(@intCast((flags >> 2) & 1)), .norm = @fromBackingInt(@intCast((flags >> 3) & 1)) };
        try std.testing.expectEqual(@as(u32, @intCast(flags)), k.flags());
    }
    try std.testing.expectEqual(abi.Storage.f32, try abi.storage(.f32));
    try std.testing.expectEqual(abi.Storage.bf16, try abi.storage(.bf16));
    try std.testing.expectError(error.UnsupportedGdnStorage, abi.storage(.f16));
}

test "actual target widths and snapshot bytes preserve FP32 state" {
    const shape = try abi.Shape.init(config());
    try std.testing.expectEqual(@as(usize, 10240), shape.qkv);
    try std.testing.expectEqual(@as(usize, 6144), shape.value);
    try std.testing.expectEqual(@as(usize, 786432), shape.state);
    const sizes = try abi.Bytes.init(shape, try shape.params(8, 4, native), native);
    try std.testing.expectEqual(@as(usize, 8 * 786432 * 4), sizes.snapshots);
    try std.testing.expectEqual(@as(usize, 4 * 786432 * 4), sizes.state);
    try std.testing.expectEqual(@as(usize, 48 * 4), sizes.a_log);
    try std.testing.expectEqual(@as(usize, 48 * 2), sizes.dt);
}

test "wrong shape flags and dimensions refuse before an encoder exists" {
    var c = config();
    c.dk = 127;
    try std.testing.expectError(error.UnsupportedGdnShape, abi.Shape.init(c));
    c = config();
    c.v_heads = 47;
    try std.testing.expectError(error.UnsupportedGdnShape, abi.Shape.init(c));
    const shape = try abi.Shape.init(config());
    try std.testing.expectError(error.BadGdnDispatch, shape.params(0, 4, native));
    try std.testing.expectError(error.BadGdnDispatch, shape.params(8, 65, native));
    var p = try shape.params(8, 4, native);
    p.dk = 64;
    try std.testing.expectError(error.BadGdnDispatch, abi.Bytes.init(shape, p, native));
    try std.testing.expectError(error.Overflow, abi.countBytes(f32, std.math.maxInt(usize), 2));
}

test "unequal windows flatten conv rows without shifting history or local parents" {
    const a = std.testing.allocator;
    var w0 = try Window.init(a, 20, &.{ -1, 0, 0, 1, 2 }, 4, 128);
    defer w0.deinit();
    var w1 = try Window.init(a, 40, &.{ -1, 0, 1 }, 4, 128);
    defer w1.deinit();
    const shape = try abi.Shape.init(config());
    var plan = try plan_mod.Plan.init(a, try shape.params(8, 4, native), &.{ .{ .window = w0, .state_slot = 2, .next_slot = 0, .path = &.{ 0, 2, 4 } }, .{ .window = w1, .state_slot = 3, .next_slot = 1, .path = &.{ 0, 1 } } });
    defer plan.deinit();
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 1, 2, -1, 0, 1 }, plan.parents);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 8 }, plan.windows[20..24]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 8, 9, 10 }, plan.windows[28..32]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4, 5, 6 }, plan.kept_rows);
    try std.testing.expectEqual(@as(u32, 5), plan.segments[1].first);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2, 2, 2, 2, 3, 3, 3 }, plan.row_slots);
}

test "empty keep retains the base and duplicate destination slots refuse" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 20, &.{ -1, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    const p = try shape.params(2, 2, native);
    var plan = try plan_mod.Plan.init(a, p, &.{.{ .window = w, .state_slot = 1, .next_slot = 0, .path = &.{} }});
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 0), plan.kept_rows.len);
    try std.testing.expectEqual(@as(u32, 0), plan.keeps[0].rows);
    const inputs = [_]plan_mod.Input{ .{ .window = w, .state_slot = 0, .next_slot = 1, .path = &.{0} }, .{ .window = w, .state_slot = 1, .next_slot = 1, .path = &.{0} } };
    try std.testing.expectError(error.OverlappingGdnCommit, plan_mod.Plan.init(a, try shape.params(4, 2, native), &inputs));
}

test "a bad accepted path or input extent cannot become replay indices" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 20, &.{ -1, 0, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    const bad = [_]plan_mod.Input{.{ .window = w, .state_slot = 0, .next_slot = 1, .path = &.{ 0, 1, 2 } }};
    try std.testing.expectError(error.BadPath, plan_mod.Plan.init(a, try shape.params(3, 2, native), &bad));
    const good = [_]plan_mod.Input{.{ .window = w, .state_slot = 0, .next_slot = 1, .path = &.{ 0, 2 } }};
    try std.testing.expectError(error.BadGdnPlan, plan_mod.Plan.init(a, try shape.params(2, 2, native), &good));
}

test "snapshot scratch scales with the aggregate round instead of layers or slots" {
    const Budget = @import("gdn_pool.zig").Budget;
    const shape = try abi.Shape.init(config());
    const one = try Budget.init(shape, 48, 1, 32);
    const four = try Budget.init(shape, 48, 4, 32);
    try std.testing.expectEqual(@as(usize, 96 * 1024 * 1024), one.snapshots);
    try std.testing.expectEqual(one.snapshots, four.snapshots);
    try std.testing.expectEqual(one.logBytes(), four.logBytes());
    try std.testing.expectEqual(@as(usize, 30 * 1024 * 1024), one.projections);
    try std.testing.expectEqual(@as(usize, 144 * 1024 * 1024), one.state);
    try std.testing.expect(one.logBytes() < 31 * 1024 * 1024);
    try std.testing.expectEqual(@as(usize, 786432 * 4), try four.stateOffset(0, 1));
    try std.testing.expectEqual(@as(usize, 48 * 786432 * 4 * 4), four.state);
    try std.testing.expectError(error.BadGdnPoolIndex, four.stateOffset(48, 0));
    try std.testing.expectError(error.BadGdnPool, Budget.init(shape, 48, 4, 129));
}

test "native byte spans accept BF16 alignment and reject unsafe range arithmetic" {
    try abi.checkSpan(1024, 2, 1022, 512, 2);
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 2, 1022, 512, 4));
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 1025, 0, 0, 2));
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 0, 1025, 512, 2));
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 0, 512, 1024, 2));
}

test "a plan from a larger slot pool cannot index the committed state bank" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 0, &.{ -1, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    var plan = try plan_mod.Plan.init(a, try shape.params(2, 8, native), &.{.{ .window = w, .state_slot = 4, .next_slot = 5, .path = &.{ 0, 1 } }});
    defer plan.deinit();
    try @import("gdn_pool.zig").validatePlan(shape, 8, 32, plan);
    try std.testing.expectError(error.BadGdnPoolPlan, @import("gdn_pool.zig").validatePlan(shape, 4, 32, plan));
}

test "cross-stream conv rows and a broken retained path refuse before encoding" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 0, &.{ -1, 0, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    var plan = try plan_mod.Plan.init(a, try shape.params(6, 2, native), &.{ .{ .window = w, .state_slot = 0, .next_slot = 0, .path = &.{ 0, 2 } }, .{ .window = w, .state_slot = 1, .next_slot = 1, .path = &.{ 0, 1 } } });
    defer plan.deinit();
    try @import("gdn_pool.zig").validatePlan(shape, 2, 32, plan);
    plan.windows[12] = 3;
    try std.testing.expectError(error.BadGdnPoolPlan, @import("gdn_pool.zig").validatePlan(shape, 2, 32, plan));
    plan.windows[12] = 0;
    plan.kept_rows[0] = 1;
    try std.testing.expectError(error.BadGdnPoolPlan, @import("gdn_pool.zig").validatePlan(shape, 2, 32, plan));
}
