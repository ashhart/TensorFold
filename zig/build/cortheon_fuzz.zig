//! `zig build fuzz-cortheon`: Cortheon's readers of untrusted bytes under zig/src/server/cortheon/fuzz.zig.
const std = @import("std");
const Module = std.Build.Module;

/// The harness's tests (seeds, 100,000 structure-aware edits, linear time); `--fuzz=N` adds N coverage-guided inputs.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, api: *Module, tokenizer: *Module, template: *Module) void {
    const about = "fuzz-cortheon's build: safe (default) or fast, as shipped";
    const optimize = b.option(std.builtin.OptimizeMode, "fuzz-optimize", about) orelse .safe;
    const harness = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/src/server/root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "engine_api", .module = api },
                .{ .name = "tokenizer", .module = tokenizer },
                .{ .name = "template", .module = template },
            },
        }),
        .filters = &.{"cortheon.fuzz."},
    });
    _ = @import("server.zig").addTo(b, harness.root_module, target, optimize);
    const step = b.step("fuzz-cortheon", "Cortheon's page, link, search and filter readers under the fuzz harness");
    step.dependOn(&b.addRunArtifact(harness).step);
}
