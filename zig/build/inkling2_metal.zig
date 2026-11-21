//! Opt-in Mini device checks; no checkpoint is fetched by the default suite.
const std = @import("std");

pub fn steps(b: *std.Build, m: anytype) void {
    const api = b.createModule(.{
        .root_source_file = b.path("zig/src/core/engine_api.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = m.lanes }},
    });
    const integration = b.createModule(.{
        .root_source_file = b.path("zig/src/inkling2_metal_checks.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tensorfold", .module = m.engine },
            .{ .name = "metal", .module = m.metal },
            .{ .name = "engine_api", .module = api },
        },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = integration, .filters = &.{"inkling2"} }));
    run.has_side_effects = true;
    const media = b.createModule(.{
        .root_source_file = b.path("zig/src/inkling2_media_checks.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = m.metal },
            .{ .name = "kernel_sources", .module = m.sources },
            .{ .name = "lanes", .module = m.lanes },
        },
    });
    const media_run = b.addRunArtifact(b.addTest(.{ .root_module = media }));
    media_run.has_side_effects = true;
    b.step("test-inkling2-media-metal", "Inkling 2 Mini vision and DSpark Metal checks").dependOn(&media_run.step);
    const loop = b.createModule(.{
        .root_source_file = b.path("zig/src/inkling2_loop_checks.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = m.metal },
            .{ .name = "kernel_sources", .module = m.sources },
        },
    });
    const loop_run = b.addRunArtifact(b.addTest(.{ .root_module = loop }));
    loop_run.has_side_effects = true;
    b.step("test-inkling2-loop-metal", "Original Mini offsets and trained controller checks (TF_INKLING2_MINI)").dependOn(&loop_run.step);
    b.step("test-inkling2-metal", "Inkling 2 Mini device and lane exactness (TF_INKLING2_MINI_SMOKE)").dependOn(&run.step);
}
