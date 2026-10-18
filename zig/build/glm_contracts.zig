//! Shader-free prompt-logprob owner and maintained projection-policy controls.
const std = @import("std");
pub fn tests(b: *std.Build, glm: *std.Build.Module, lanes: *std.Build.Module, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const host = b.graph.host;
    const prompt_arena = b.createModule(.{ .root_source_file = b.path("zig/glm_prompt_arena_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda } } });
    b.step("test-glm-prompt-arena-host", "Actual sequence arenas retain and account prompt rows").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = prompt_arena })).step);
    const prompt_owner = b.createModule(.{ .root_source_file = b.path("zig/glm_prompt_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm", .module = glm }} });
    b.step("test-glm-prompts-host", "Actual prompt workflow with recording device operations").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = prompt_owner, .filters = &.{ "GLM prompt workflow:", "GLM structured:" } })).step);

    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_prompt_logprobs_host.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = lanes }, .{ .name = "cuda", .module = cuda } },
    }) }));
    b.step("test-glm-logprobs-host", "Actual GLM prompt row ownership and head policy without GPU calls").dependOn(&run.step);
    const structured = b.createModule(.{ .root_source_file = b.path("zig/glm_structured_host.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } } });
    b.step("test-glm-structured-host", "Actual prompt redraw workflow without GPU calls").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = structured })).step);
    const protocol = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{ "flag 1", "the head's columns" } }));
    b.step("test-glm-protocol-host", "Prompt and grammar follow flags without GPU calls").dependOn(&protocol.step);
}
