//! MiniMax-H3-Image's Metal tools: the edit encoder, the prompt tool and the image decoder.
const std = @import("std");

/// The picture tools on `metal`; `sources` is the Metal kernels' source module, used by the maintained H3 tool.
pub fn tools(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module) void {
    _ = sources;
    @import("h3_image_encoder.zig").tool(b, target, optimize, metal);
    @import("h3_image_prompt.zig").install(b, target, optimize, metal);
    @import("h3_image_decoder.zig").tool(b, target, optimize, metal);
}
