//! Exercise the actual GLM loader and destructor on both Metal capability paths.
const std = @import("std");
const metal = @import("metal");
const kernels = @import("glm_kernels").glm.kernels;
fn check(tensor: bool) !void {
    metal.reset();
    const a = std.testing.allocator;
    const k = try kernels.load(a, .{ .tensor = tensor });
    defer a.destroy(k);
    k.deinit();
    try std.testing.expectEqual(metal.created.load(.monotonic), metal.released.load(.monotonic));
    try std.testing.expect(metal.created.load(.monotonic) > 0);
}
test "GLM kernel ownership: pre-M5 releases every compiled pipeline exactly once" {
    try check(false);
}
test "GLM kernel ownership: M5 releases every compiled pipeline exactly once" {
    try check(true);
}
