//! The control vector's plan on the host: the layer lookup and the directions a device holds.
const std = @import("std");
const mtl = @import("metal");
const steer = @import("steer.zig");

/// The recording device the host tests use: the buffers live in host memory, no GPU runs.
fn device(allocations: *mtl.Allocations) mtl.Device {
    return .{ .allocations = allocations };
}

test "the plan holds a direction for every layer the vector names and none for the rest" {
    var allocations: mtl.Allocations = .{};
    const dir = [_]f32{ 1, 0, 0, 0 };
    const plan = try steer.init(std.testing.allocator, device(&allocations), .{
        .mode = .project,
        .alpha = 0.5,
        .hidden = 4,
        .layers_total = 8,
        .layers = &.{ .{ .id = 3, .dir = &dir }, .{ .id = 5, .dir = &dir } },
    });
    defer plan.deinit();
    try std.testing.expect(plan.dir(3) != null);
    try std.testing.expect(plan.dir(5) != null);
    try std.testing.expect(plan.dir(4) == null);
    try std.testing.expect(plan.dir(0) == null);
    try std.testing.expect(plan.dir(8) == null);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5 }, plan.covered());
    try std.testing.expectEqualSlices(f32, &dir, plan.dir(3).?.slice(f32, 4));
    try std.testing.expectEqual(@as(f32, 0.5), plan.alpha);
    try std.testing.expectEqual(steer.Mode.project, plan.mode);
}

test "a vector that names a layer the model does not have is refused before any buffer is made" {
    var allocations: mtl.Allocations = .{};
    const dir = [_]f32{ 1, 0, 0, 0 };
    // The blocks are one to `layers_total`, so the last one is the model's last layer.
    try std.testing.expectError(error.BadSteerLayer, steer.init(std.testing.allocator, device(&allocations), .{
        .mode = .add,
        .alpha = 1,
        .hidden = 4,
        .layers_total = 4,
        .layers = &.{.{ .id = 5, .dir = &dir }},
    }));
    try std.testing.expectError(error.BadSteerLayer, steer.init(std.testing.allocator, device(&allocations), .{
        .mode = .add,
        .alpha = 1,
        .hidden = 4,
        .layers_total = 4,
        .layers = &.{.{ .id = 0, .dir = &dir }},
    }));
    try std.testing.expectEqual(@as(usize, 0), allocations.calls);
}

test "a vector that names the model's last layer is taken" {
    var allocations: mtl.Allocations = .{};
    const dir = [_]f32{ 1, 0, 0, 0 };
    const plan = try steer.init(std.testing.allocator, device(&allocations), .{
        .mode = .add,
        .alpha = 1,
        .hidden = 4,
        .layers_total = 4,
        .layers = &.{.{ .id = 4, .dir = &dir }},
    });
    defer plan.deinit();
    try std.testing.expect(plan.dir(4) != null);
}

test "a direction that is not as wide as the model is refused" {
    var allocations: mtl.Allocations = .{};
    const dir = [_]f32{ 1, 0 };
    try std.testing.expectError(error.BadSteerWidth, steer.init(std.testing.allocator, device(&allocations), .{
        .mode = .add,
        .alpha = 1,
        .hidden = 4,
        .layers_total = 4,
        .layers = &.{.{ .id = 1, .dir = &dir }},
    }));
    try std.testing.expectEqual(@as(usize, 0), allocations.calls);
}
