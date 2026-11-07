//! The CPU engine's build: tensorfold-native over the lane core and the CPU Qwen3.5-2B family, on every OS.
const std = @import("std");
const server_build = @import("server.zig");
const checkpointCli = @import("cuda.zig").checkpointCli;

pub const Modules = struct { core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module, family: *std.Build.Module, engines: *std.Build.Module, tokenizer: *std.Build.Module, living: *std.Build.Module };

/// The module graph the CPU server and its tests share: no `metal`, `cuda`, `hip`, `xpu` or `vulkan` module enters it.
pub fn modules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Modules {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    core.addImport("living", living);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const family = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cpu.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
    const engines = b.createModule(.{ .root_source_file = b.path("zig/src/native/cpu.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "qwen35_cpu", .module = family } } });
    audit(engines);
    return .{ .core = core, .lanes = lanes, .api = api, .family = family, .engines = engines, .tokenizer = tokenizer, .living = living };
}

/// `zig build native-cpu`: tensorfold-native with the CPU engine into zig-out/native-cpu/bin; its tests under `test`.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options, strip: bool, test_step: *std.Build.Step) void {
    const m = modules(b, target, optimize);
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = m.core }, .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = m.tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "living", .module = m.living }, .{ .name = "checkpoint_cli", .module = checkpointCli(b, target, .ReleaseSafe, null, m.engines, m.core) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    exe.root_module.strip = strip;
    _ = server_build.addTo(b, exe.root_module, target, .ReleaseSafe);
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native-cpu/bin" } } });
    b.step("native-cpu", "tensorfold-native with the CPU engine (Qwen3.5-2B) into zig-out/native-cpu/bin, on any OS").dependOn(&install.step);
    const host = modules(b, b.graph.host, .ReleaseFast);
    const tests = b.step("test-cpu", "The CPU engine's tests: the rows against float64, window and chunk exactness, admission, refusals");
    for ([_]*std.Build.Module{ host.family, host.engines }) |mod| tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
    test_step.dependOn(tests);
    const forward = b.addExecutable(.{ .name = "tf-qwen35-cpu-forward", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/qwen35_cpu_forward.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "qwen35_cpu", .module = m.family }, .{ .name = "core", .module = m.core } },
    }) });
    b.step("qwen35-cpu-forward", "Qwen3.5-2B teacher-forced bf16 logits on the CPU for a token fixture (tools/zig/cpu_fidelity.py)").dependOn(&b.addInstallArtifact(forward, .{}).step);
}

/// No device module may enter the CPU engine's graph: a host with no GPU has neither Metal nor CUDA.
fn audit(root: *std.Build.Module) void {
    var seen: std.AutoHashMapUnmanaged(*std.Build.Module, void) = .empty;
    walk(root, &seen);
}

fn walk(m: *std.Build.Module, seen: *std.AutoHashMapUnmanaged(*std.Build.Module, void)) void {
    if ((seen.getOrPut(m.owner.allocator, m) catch @panic("OOM")).found_existing) return;
    var it = m.import_table.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        for ([_][]const u8{ "metal", "cuda", "hip", "xpu", "vulkan" }) |device| if (std.mem.eql(u8, name, device)) @panic("the CPU engine's graph imports a device module");
        walk(e.value_ptr.*, seen);
    }
}
