const std = @import("std");
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const key = b.fmt("gpu_memory_{s}", .{target.result.zigTriple(b.allocator) catch @panic("target name")});
    if (b.modules.get(key)) |m| return m;
    return b.addModule(key, .{ .root_source_file = b.path("zig/src/gpu_memory.zig"), .target = target, .optimize = .Debug });
}
