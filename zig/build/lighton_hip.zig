//! HIP LightOnOCR modules, host checks and the zig/tests/lighton_hip tools, on the server's shared engine types.
const std = @import("std");
const Engine = @import("hip.zig").Engine;

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, hip: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module, serve: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lighton_ocr/hip.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api }, .{ .name = "hip_serve", .module = serve } } });
}

pub fn vision(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, hip: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, serve: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_hip.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "hip_serve", .module = serve } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-lighton-hip-host", "LightOnOCR HIP image, state and admission tests without devices").dependOn(&run.step);
    step.dependOn(&run.step);
}

pub fn captureQwen(b: *std.Build, target: std.Build.ResolvedTarget, e: Engine) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "tf-qwen-hip-capture", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/lighton_hip/qwen_capture.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "core", .module = e.hip.import_table.get("core").? }, .{ .name = "qwen3_5", .module = e.family }, .{ .name = "lanes", .module = e.lanes }, .{ .name = "qwen_hip_vision", .module = e.family.import_table.get("qwen_hip_vision").? } },
    }) });
}

pub fn exactTool(b: *std.Build, target: std.Build.ResolvedTarget, e: Engine, family_name: []const u8) *std.Build.Step.Compile {
    const file = if (std.mem.eql(u8, family_name, "lighton")) "exact.zig" else "qwen_exact.zig";
    return b.addExecutable(.{ .name = b.fmt("tf-{s}-hip-exact", .{family_name}), .root_module = b.createModule(.{
        .root_source_file = b.path(b.fmt("zig/tests/lighton_hip/{s}", .{file})),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "core", .module = e.hip.import_table.get("core").? }, .{ .name = "lighton_ocr", .module = e.lighton }, .{ .name = "qwen3_5", .module = e.family }, .{ .name = "lanes", .module = e.lanes }, .{ .name = "qwen_hip_vision", .module = e.family.import_table.get("qwen_hip_vision").? } },
    }) });
}
