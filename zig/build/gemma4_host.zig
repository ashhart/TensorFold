//! Safe host targets. Recording fixtures execute no GPU call; compile objects use only embedded source strings.
const std = @import("std");
pub fn tests(b: *std.Build, sources: *std.Build.Module) *std.Build.Step {
    const parts = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/model_parts_test.zig"), .target = b.graph.host }) }));
    parts.has_side_effects = true;
    b.step("test-gemma4-load-parts-host", "Production late constructor with faulting GPU boundary").dependOn(&parts.step);
    const owner_fake = b.createModule(.{ .root_source_file = b.path("zig/gemma4_inert_metal.zig"), .target = b.graph.host });
    const owners = b.createModule(.{ .root_source_file = b.path("zig/gemma4_owners_test.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = owner_fake }, .{ .name = "kernel_sources", .module = sources } } });
    const owner_run = b.addRunArtifact(b.addTest(.{ .root_module = owners }));
    owner_run.has_side_effects = true;
    b.step("test-gemma4-owners-host", "Actual Gemma ownership boundaries without a GPU").dependOn(&owner_run.step);

    const contract_module = @import("kernel_contract.zig").module(b, b.graph.host, .debug);
    const family_facade = b.createModule(.{ .root_source_file = b.path("zig/gemma4_contract_family.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = owner_fake }, .{ .name = "kernel_sources", .module = sources } } });
    const family_test = b.createModule(.{ .root_source_file = b.path("zig/gemma4_contract_test.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = owner_fake }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "contract", .module = contract_module }, .{ .name = "tensorfold", .module = family_facade } } });
    const contract_filter = b.option([]const u8, "gemma4-contract-filter", "One family contract host regression");
    const family_run = b.addRunArtifact(b.addTest(.{ .root_module = family_test, .filters = if (contract_filter) |f| &.{f} else &.{} }));
    family_run.has_side_effects = true;
    const family_step = b.step("test-gemma4-contract-host", "Family contract registration and launch declarations without a GPU");
    family_step.dependOn(&family_run.step);
    const config_run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/gemma4_config_test.zig"), .target = b.graph.host, .link_libc = true }) }));
    config_run.has_side_effects = true;
    b.step("test-gemma4-config", "Real Gemma parser fixtures without GPU dependencies").dependOn(&config_run.step);
    const fake = b.createModule(.{ .root_source_file = b.path("zig/gemma4_recording_metal.zig"), .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/gemma4_shared_affine_test.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "kernel_sources", .module = sources } } });
    const filter = b.option([]const u8, "gemma4-host-filter", "One portable shared affine regression");
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = if (filter) |f| &.{f} else &.{} }));
    run.has_side_effects = true;
    b.step("test-gemma4-affine-host", "Record actual shared dispatch without devices or shader compilation").dependOn(&run.step);
    return family_step;
}

pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module, sources: *std.Build.Module) void {
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/metal_main.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) } } });
    runner.addImport("h3_kernels", b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") }));
    const contract_root = b.createModule(.{ .root_source_file = b.path("zig/gemma4_contract_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "runner", .module = runner }} });
    const contract_object = b.addObject(.{ .name = "gemma4-contract-compile", .root_module = contract_root });
    _ = contract_object.getEmittedBin();
    b.step("compile-gemma4-contract-host", "Compile normal contract runner, family hooks and reflection without a metallib").dependOn(&contract_object.step);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/gemma4.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/gemma4_host_compile.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "host", .module = host }, .{ .name = "tensorfold", .module = engine } } });
    const object = b.addObject(.{ .name = "gemma4-host-compile", .root_module = root });
    _ = object.getEmittedBin();
    b.step("compile-gemma4-host", "Compile actual startup and restored GLM/LFM2 callers without a metallib").dependOn(&object.step);
}
