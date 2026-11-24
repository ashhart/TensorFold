//! macOS: MiniMax H3's Metal tools (metal_targets.zig): FastH3's transformer and MiniMax-H3-Image's tools.
const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, test_step: *std.Build.Step) void {
    @import("h3_host.zig").metalTools(b, target, optimize, metal, sources);
    const h3_kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") });
    @import("h3_image_encoder.zig").tool(b, target, optimize, metal);
    @import("h3_image_prompt.zig").install(b, target, optimize, metal);
    @import("h3_image_decoder.zig").tool(b, target, optimize, metal, test_step);
    @import("h3_image_server.zig").tool(b, target, optimize, metal, h3_kernels, sources, test_step);
}
