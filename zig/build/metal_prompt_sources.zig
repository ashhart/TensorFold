//! Runtime source assembly is device-free; the compiler gate requires a device and submits no work.
const std = @import("std");

pub fn steps(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, host_tests: *std.Build.Step) void {
    const recording = b.createModule(.{ .root_source_file = b.path("zig/lfm2_recording_metal.zig"), .imports = &.{.{ .name = "real_metal", .module = metal }} });
    const loader_host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/lfm2_runtime_sources_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = recording },
            .{ .name = "kernel_sources", .module = engine.import_table.get("kernel_sources").? },
        },
    }), .filters = &.{"LFM2 runtime inventory:"} });
    const loader_run = b.addRunArtifact(loader_host);
    host_tests.dependOn(&loader_run.step);
    b.step("test-lfm2-runtime-sources-host", "LFM2 source inventory equals its recording loader without a device").dependOn(&loader_run.step);
    const assembly = b.createModule(.{ .root_source_file = b.path("zig/runtime_sources/assembly_metal.zig"), .imports = &.{.{ .name = "real_metal", .module = metal }} });
    const inventory = b.createModule(.{ .root_source_file = b.path("zig/metal_runtime_inventory.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    var imports = engine.import_table.iterator();
    while (imports.next()) |entry| inventory.addImport(entry.key_ptr.*, entry.value_ptr.*);
    inventory.addImport("metal", assembly);
    inventory.addImport("core", inventory);
    inline for (.{ "bf16_topk_gpu", "tree_round_gpu", "tree_commit_gpu", "draft_ops" }, .{ "bf16_topk_sources", "tree_round_sources", "tree_commit_sources", "draft_ops_sources" }) |owner, name| {
        inventory.addImport(name, engine.import_table.get(owner).?.import_table.get(name).?);
    }
    inventory.addImport("h3_kernels", b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") }));
    const mode_run = b.addRunArtifact(b.addTest(.{ .root_module = inventory, .filters = &.{"Metal runtime source modes:"} }));
    host_tests.dependOn(&mode_run.step);
    b.step("test-metal-runtime-source-modes-host", "Runtime libraries match their loaders' device capability gates").dependOn(&mode_run.step);
    const root = b.createModule(.{
        .root_source_file = b.path("zig/metal_runtime_sources_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "inventory", .module = inventory } },
    });
    const options = b.addOptions();
    options.addOption([]const u8, "family", b.option([]const u8, "metal-source-family", "Diagnostic family prefix; empty compiles all libraries") orelse "");
    root.addOptions("source_options", options);
    const tests = b.addTest(.{ .root_module = root, .filters = &.{"Metal runtime source compile gate:"} });
    const run = b.addRunArtifact(tests);
    run.has_side_effects = true;
    const step = b.step("test-metal-runtime-sources", "Compile every family's runtime source libraries; device required, no dispatch");
    step.dependOn(&loader_run.step);
    step.dependOn(&mode_run.step);
    step.dependOn(&run.step);
    b.step("test-nemotron-prefill-metal-sources", "Alias for the complete runtime Metal source compiler gate").dependOn(step);
    b.step("build-metal-runtime-sources", "Build the source gate executable without opening a device").dependOn(&tests.step);
    const emit_root = b.createModule(.{
        .root_source_file = b.path("zig/metal_runtime_sources_emit.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "inventory", .module = inventory }},
    });
    const emit = b.addExecutable(.{ .name = "tf-metal-runtime-sources", .root_module = emit_root });
    const emit_run = b.addRunArtifact(emit);
    emit_run.addArgs(b.option([]const []const u8, "source-args", "Output directory and tensor-unit mode") orelse &.{ "zig-out/runtime-metal-sources", "0" });
    b.step("emit-metal-runtime-sources", "Write the assembled runtime libraries without opening a device").dependOn(&emit_run.step);
}
