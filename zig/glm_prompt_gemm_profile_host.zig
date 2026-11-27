const std = @import("std");
const profile = @import("tests/cuda/glm_prompt_gemm_shapes.zig");

test "GLM NVFP4 TP2 dense prompt shapes and geometry" {
    const up = profile.byName("dense-up");
    try std.testing.expectEqual(profile.Kind.dense_up, up.kind);
    try std.testing.expectEqual(@as(usize, 2048), up.rows);
    try std.testing.expectEqual(@as(usize, 6144), up.n);
    try std.testing.expectEqual(@as(usize, 4096), up.k);
    const up_launch = try up.geometry(148);
    try std.testing.expectEqual(@as(u32, 148), up_launch.grid_x);
    try std.testing.expectEqual(@as(u32, 256), up_launch.threads);
    try std.testing.expectEqual(@as(u32, 76848), up_launch.shared);
    const down = profile.byName("dense-down");
    try std.testing.expectEqual(@as(usize, 768), down.rows);
    try std.testing.expectEqual(@as(usize, 4096), down.n);
    try std.testing.expectEqual(@as(usize, 6144), down.k);
    try std.testing.expectEqual(@as(u32, 96), (try down.geometry(148)).grid_x);
    try std.testing.expectEqual(@as(u32, 64), (try profile.byName("dense-down-tail").geometry(148)).grid_x);
}

test "GLM routed expert uniform prompt fixture has exact capacity geometry" {
    const up = profile.byName("routed-up");
    try std.testing.expectEqual(@as(usize, 1024), up.n);
    try std.testing.expectEqual(@as(usize, 4096), up.k);
    try std.testing.expectEqual(@as(usize, 577), up.item_bound);
    try std.testing.expectEqual(@as(usize, 320), up.planned_items);
    try std.testing.expectEqual(@as(usize, 288), profile.routed_experts);
    try std.testing.expectEqual(@as(usize, 289), profile.plan_experts);
    try std.testing.expectEqual(@as(usize, 8), profile.routed_slots);
    try std.testing.expectEqual(@as(usize, 9), profile.slots);
    const launch = try up.geometry(148);
    try std.testing.expectEqual(@as(u32, 4616), launch.grid_x);
    try std.testing.expectEqual(@as(u32, 128), launch.threads);
    try std.testing.expectEqual(@as(u32, 34816), launch.shared);
    const down = profile.byName("routed-down");
    try std.testing.expectEqual(@as(usize, 4096), down.n);
    try std.testing.expectEqual(@as(usize, 1024), down.k);
    const down_launch = try down.geometry(148);
    try std.testing.expectEqual(@as(u32, 9232), down_launch.grid_x);
    try std.testing.expectEqual(@as(u32, 256), down_launch.threads);
}

test "GLM NVFP4 arithmetic ranking distinguishes fused routed gate-up and down pieces" {
    const routed_up = profile.byName("routed-up");
    const routed_down = profile.byName("routed-down");
    const dense_up = profile.byName("dense-up");
    const dense_down = profile.byName("dense-down");
    const dense_tail = profile.byName("dense-down-tail");
    try std.testing.expectEqual(@as(u128, 8192 * 8 * 2 * 1024 * 4096 * 42), profile.promptMacs8K(&routed_up));
    try std.testing.expect(profile.promptMacs8K(&routed_up) > profile.promptMacs8K(&routed_down));
    try std.testing.expect(profile.promptMacs8K(&routed_down) > profile.promptMacs8K(&dense_up));
    try std.testing.expect(profile.promptMacs8K(&dense_up) > profile.promptMacs8K(&dense_down) + profile.promptMacs8K(&dense_tail));
}

test "GLM routed R3 profiles keep launch geometry and halve stage footprint" {
    const up = profile.byName("routed-up-r3");
    try std.testing.expect(up.resident);
    const base = profile.byName("routed-up");
    const up_launch = try up.geometry(148);
    const base_launch = try base.geometry(148);
    try std.testing.expectEqual(base_launch.grid_x, up_launch.grid_x);
    try std.testing.expectEqual(base_launch.threads, up_launch.threads);
    try std.testing.expectEqual(@as(u32, 17408), up_launch.shared);
    const down_launch = try profile.byName("routed-down-r3").geometry(148);
    try std.testing.expectEqual(@as(u32, 9232), down_launch.grid_x);
    try std.testing.expectEqual(@as(u32, 256), down_launch.threads);
    try std.testing.expectEqual(@as(u32, 17408), down_launch.shared);
    try std.testing.expectEqual(@as(?profile.Shape, null), profile.find("dense-up-r3"));
}
