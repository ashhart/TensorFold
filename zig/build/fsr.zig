//! CPU-only FSR ownership and file fixtures; no shader compiler or GPU driver.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/core/fsr/tests.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-fsr-host", "Exact FSR cache ownership and bounded reader fixtures without devices").dependOn(&run.step);
    all.dependOn(&run.step);
    const source = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/fsr_source_tests.zig"), .target = b.graph.host, .optimize = .debug }) }));
    b.step("test-fsr-source-host", "Recorded expert source address patch and refusal controls without shaders").dependOn(&source.step);
    all.dependOn(&source.step);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const cli = b.createModule(.{ .root_source_file = b.path("zig/fsr_cli_tests.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "engine_api", .module = api }} });
    const cli_run = b.addRunArtifact(b.addTest(.{ .root_module = cli, .filters = &.{"FSR CLI"} }));
    b.step("test-fsr-cli-host", "Actual native FSR CLI forwarding and capability refusals without devices").dependOn(&cli_run.step);
    all.dependOn(&cli_run.step);
}

pub fn deviceSteps(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, sources: *std.Build.Module) void {
    const gate = b.createModule(.{ .root_source_file = b.path("zig/fsr_metal_gate.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } } });
    const device_test = b.addTest(.{ .root_module = gate, .filters = &.{"FSR Metal synthetic"} });
    const run = b.addRunArtifact(device_test);
    b.step("test-fsr-metal", "GPU: uncached four-slot synthetic MoE equals resident at every node and logit").dependOn(&run.step);
    b.step("compile-fsr-metal-gate-host", "Compile the real FSR device gate without shader compilation or execution").dependOn(&device_test.step);
    const contract = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/metal_main.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }, .{ .name = "h3_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") }) } } });
    const compile_contract = b.addObject(.{ .name = "fsr-metal-contract", .root_module = b.createModule(.{ .root_source_file = b.path("zig/fsr_contract_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "contract_main", .module = contract }} }) });
    _ = compile_contract.getEmittedBin();
    b.step("compile-fsr-metal-contract-host", "Compile the actual Metal contract entry with inert embedded bytes and no execution").dependOn(&compile_contract.step);
}
