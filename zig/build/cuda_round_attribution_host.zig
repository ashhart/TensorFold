//! Focused complete-round record, arithmetic, event ownership and default-off checks without a CUDA driver.
const std = @import("std");
pub fn wire(b: *std.Build, parent: *std.Build.Step) void {
    const cuda = @import("cuda_runtime.zig").runtime(b, b.graph.host, .debug, &.{});
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/round_attribution_test.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"round attribution:"} }));
    run.setEnvironmentVariable("TF_CUDA_ROUND_TRACE", "0");
    b.step("test-cuda-round-attribution-host", "Complete round format, interval budgets and captured event lifetime without devices").dependOn(&run.step);
    parent.dependOn(&run.step);
}

pub fn compile(b: *std.Build, mods: anytype) void {
    const cuda = mods.qwen35.import_table.get("cuda").?;
    const fake = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/round_attribution_fake.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const core_test = b.createModule(.{ .root_source_file = b.path("zig/qwen35_round_attribution_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "cuda", .module = cuda }, .{ .name = "attribution_fake", .module = fake } } });
    const review = b.addRunArtifact(b.addTest(.{ .root_module = core_test, .filters = &.{b.option([]const u8, "round-attribution-review-filter", "Select one attribution review regression") orelse "attribution review:"} }));
    review.setEnvironmentVariable("TF_CUDA_ROUND_TRACE", "0");
    b.step("test-qwen35-round-attribution-review-host", "Focused attribution review regressions without a CUDA device").dependOn(&review.step);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = core_test, .filters = &.{"round attribution core:"} }));
    b.step("test-qwen35-round-attribution-host", "Actual core complete, clipped and shared commit boundaries with a recording CUDA driver").dependOn(&run.step);

    const root = b.createModule(.{ .root_source_file = b.path("zig/cuda_round_attribution_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "lanes", .module = mods.lanes } } });
    b.step("compile-cuda-round-attribution-host", "Compile actual CUDA open, backend, drafter and core round callbacks without linking or executing").dependOn(&b.addObject(.{ .name = "round-attribution-callbacks", .root_module = root }).step);
}
