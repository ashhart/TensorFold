//! Windows platform decisions exercised on the host, with no GPU images or platform loader execution.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, api: *std.Build.Module) *std.Build.Step {
    const step = b.step("test-windows-port-host", "Windows pool decisions, and DLL and dylib names, without a Windows host or GPU");
    const dll = b.createModule(.{ .root_source_file = b.path("zig/src/platform/windows_library_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dll })).step);
    const dylib = b.createModule(.{ .root_source_file = b.path("zig/src/platform/mac_library.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dylib })).step);
    const shim = b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys_env_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = shim })).step);
    const set = b.createModule(.{ .root_source_file = b.path("zig/src/platform/working_set_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = set })).step);
    const pool = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_family_memory.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api } } });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = pool, .filters = &.{"Windows pool:"} })).step);
    const limits = b.createModule(.{
        .root_source_file = b.path("zig/http_limits_windows_compile.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .windows, .abi = .gnu }),
        .optimize = .debug,
        .link_libc = true,
    });
    b.step("compile-http-limits-windows-host", "Compile production Winsock connection admission without linking or running").dependOn(&b.addObject(.{ .name = "http-limits-windows", .root_module = limits }).step);
    const chat = b.createModule(.{
        .root_source_file = b.path("zig/chat_windows_compile.zig"),
        .target = limits.resolved_target.?,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "json", .module = @import("server.zig").json(b, limits.resolved_target.?, .debug) }},
    });
    const shim_windows = b.createModule(.{
        .root_source_file = b.path("zig/src/platform/sys_env_test.zig"),
        .target = limits.resolved_target.?,
        .optimize = .debug,
        .link_libc = true,
    });
    b.step("compile-sys-env-windows-host", "Compile Windows native filesystem regressions without running them").dependOn(&b.addTest(.{ .root_module = shim_windows }).step);
    _ = @import("product.zig").windowsCompile(b);
    b.step("check-chat-windows-host", "Compile tensorfold chat's --url path for Windows without linking or running").dependOn(&b.addObject(.{ .name = "chat-windows", .root_module = chat }).step);
    return step;
}
