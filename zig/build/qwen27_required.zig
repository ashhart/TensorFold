//! All targets here are CPU tests; their dependency graph contains no shader or metallib step.
const std = @import("std");
pub const Modules = struct { fake: *std.Build.Module, masks: *std.Build.Module, draw: *std.Build.Module };
pub fn modules(b: *std.Build, lanes: *std.Build.Module) Modules {
    const masks = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_masks.zig"), .target = b.graph.host });
    const draw = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/draw.zig"), .target = b.graph.host, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const fake = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_mask_fake.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "qwen27_masks", .module = masks }, .{ .name = "qwen_draw", .module = draw } } });
    return .{ .fake = fake, .masks = masks, .draw = draw };
}
pub fn tests(b: *std.Build, lanes: *std.Build.Module) void {
    const m = modules(b, lanes);
    const errors = b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"grammar errors:"} }));
    errors.has_side_effects = true;
    b.step("test-grammar-error-host", "Constraint fill-error allocation balance without shaders").dependOn(&errors.step);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen27_required_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "qwen27_mask_fake", .module = m.fake }, .{ .name = "qwen27_masks", .module = m.masks }, .{ .name = "qwen_draw", .module = m.draw } },
    }), .filters = &.{"qwen27 masks:"} }));
    run.has_side_effects = true;
    b.step("test-qwen27-masks-host", "Production Qwen27 mask policies through the lane fake").dependOn(&run.step);
}
