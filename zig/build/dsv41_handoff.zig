const std = @import("std");

fn root(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .optimize = optimize, .link_libc = true });
    fabric.addImport("verbs", @import("tp.zig").verbsModule(b, target, optimize));
    const sessions = @import("sessions.zig").module(b, target, optimize);
    const options = b.addOptions();
    options.addOption(bool, "exl3_kv", exists(b, "zig/src/families/deepseek_v41/metal_kv_state.zig"));
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/handoff_host.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "fabric", .module = fabric },
        .{ .name = "sessions", .module = sessions },
        .{ .name = "handoff_options", .module = options.createModule() },
    } });
}

/// Whether a file exists in the source tree: the served Metal lane's layout guard is built only where that lane is.
fn exists(b: *std.Build, rel: []const u8) bool {
    std.Io.Dir.cwd().access(b.graph.io, rel, .{}) catch return false;
    return true;
}

pub fn steps(b: *std.Build) void {
    const host = b.addTest(.{ .root_module = root(b, b.graph.host, .Debug) });
    b.step("test-dsv41-handoff", "DeepSeek-V4.1 CUDA-to-Metal hand-off: conversions, plan, protocol-2 export/import (host only)").dependOn(&b.addRunArtifact(host).step);
    inline for (.{ .{ "aarch64", .linux, .gnu, "compile-dsv41-handoff-aarch64-linux" }, .{ "aarch64", .macos, .none, "compile-dsv41-handoff-aarch64-macos" } }) |t| {
        const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = t[1], .abi = t[2] });
        const check = b.addTest(.{ .root_module = root(b, target, .Debug) });
        _ = check.getEmittedBin();
        b.step(t[3], "Emit the hand-off tests for the target without running them").dependOn(&check.step);
    }
}
