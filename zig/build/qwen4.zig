//! Dense Qwen4 host tests and explicit module ownership never select a shader producer or device runtime.
const std = @import("std");
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core: *std.Build.Module) *std.Build.Module {
    const graph = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/forward_graph.zig"), .target = target, .optimize = optimize });
    const ngram = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/ngram_geometry.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{ .root_source_file = b.path("zig/qwen4_family.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "hybrid_graph", .module = graph }, .{ .name = "hybrid_ngram", .module = ngram } } });
}
const flags: []const []const u8 = &.{ "-std=c++17", "-fno-fast-math", "-ffp-contract=off" };
pub fn metalNative(b: *std.Build, target: std.Build.ResolvedTarget, native: *std.Build.Module) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/qwen4_core_metal.zig"), .target = target, .optimize = native.optimize, .imports = &.{.{ .name = "owner", .module = native.import_table.get("tensorfold").? }} });
    if (native.import_table.get("contract")) |contract| core.addImport("contract", contract);
    native.addImport("qwen4", module(b, target, native.optimize orelse .Debug, core));
}
pub fn steps(b: *std.Build, parent: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/qwen4_core_host.zig"), .target = b.graph.host, .optimize = .Debug });
    core.addImport("lanes", b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host }));
    core.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .imports = &.{.{ .name = "lanes", .module = core.import_table.get("lanes").? }} }));
    const family = module(b, b.graph.host, .Debug, core);
    const m = family;
    m.addCSourceFile(.{ .file = b.path("zig/tests/qwen4/weight_only_host.cpp"), .flags = flags });
    @import("qwen4_hc_host.zig").wire(b, m);
    @import("qwen4_gdn_host.zig").wire(b, m);
    @import("qwen4_gdn_cuda_host.zig").wire(b, m);
    @import("qwen4_ple_host.zig").wire(b, m);
    @import("qwen4_ple_cuda_host.zig").wire(b, m);
    @import("qwen4_qsa_norm_host.zig").wire(b, m);
    @import("qwen4_qsa_scores_host.zig").wire(b, m);
    @import("qwen4_qsa_select_host.zig").wire(b, m);
    @import("qwen4_qsa_attn_host.zig").wire(b, m);
    const step = b.step("test-qwen4-host", "Qwen4 explicit config admission and synthetic graph contracts");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const mixer_step = b.step("test-qwen4-native-mixers-host", "Actual native mixer bodies under CPU collective instruments");
    mixer_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "Qwen4 GDN native contract", "Qwen4 HC native contract", "Qwen4 rounding bounds", "Qwen4 PLE native contract" } })).step);
    const ple_step = b.step("test-qwen4-ple-host", "Generated native lexical bodies against binary64 nodes");
    ple_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"Qwen4 PLE native contract"} })).step);
    const qsa_step = b.step("test-qwen4-qsa-host", "Generated native QSA bodies against independent scalar references");
    qsa_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "Qwen4 QSA native contract", "Qwen4 QSA scores", "Qwen4 QSA selection" } })).step);
    parent.dependOn(step);
    const cu = @import("cuda_runtime.zig").runtime(b, b.graph.host, .debug, &.{});
    const gdn_host = b.createModule(.{ .root_source_file = b.path("zig/qwen4_gdn_host.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cu }, .{ .name = "qwen4", .module = family } } });
    const gdn_step = b.step("test-qwen4-gdn-host", "Real recurrence launch wrappers under owned recording drivers");
    gdn_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = gdn_host })).step);
    parent.dependOn(gdn_step);
    const recording = b.createModule(.{ .root_source_file = b.path("zig/qwen4_recording_metal.zig"), .target = b.graph.host });
    const cproj = b.createModule(.{ .root_source_file = b.path("zig/src/core/weight_only_cuda.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cu } } });
    const mproj = b.createModule(.{ .root_source_file = b.path("zig/src/core/weight_only_metal.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "metal", .module = recording } } });
    const wrapper = b.createModule(.{ .root_source_file = b.path("zig/qwen4_projection_host.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cu }, .{ .name = "metal", .module = recording }, .{ .name = "core", .module = core }, .{ .name = "cuda_projection", .module = cproj }, .{ .name = "metal_projection", .module = mproj } } });
    wrapper.addCSourceFile(.{ .file = b.path("zig/tests/qwen4/weight_only_host.cpp"), .flags = flags });
    const launch = b.step("test-qwen4-projection-host", "Real Metal/CUDA projection wrappers under owned recording devices");
    launch.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = wrapper, .filters = &.{ "launch:", "Metal session:" } })).step);
    parent.dependOn(launch);
    const actual_metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const actual_projection = b.createModule(.{ .root_source_file = b.path("zig/src/core/weight_only_metal.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "metal", .module = actual_metal } } });
    const bodies = b.createModule(.{ .root_source_file = b.path("zig/qwen4_projection_bodies.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cu }, .{ .name = "metal", .module = actual_metal }, .{ .name = "cuda_projection", .module = cproj }, .{ .name = "metal_projection", .module = actual_projection } } });
    bodies.addImport("qwen4", family);
    const lanes = core.import_table.get("lanes").?;
    bodies.addImport("lanes", lanes);
    bodies.addImport("engine_api", core.import_table.get("engine_api").?);
    bodies.addImport("tensorfold", core);
    bodies.addImport("kernel_sources", b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig"), .target = b.graph.host }));
    const contract = b.createModule(.{ .root_source_file = b.path("zig/qwen4_contract_facade.zig"), .target = b.graph.host, .imports = &.{.{ .name = "core", .module = core }} });
    bodies.addImport("contract", contract);
    const body_step = b.step("test-qwen4-projection-bodies", "Emit real Metal/CUDA launch bodies without shaders or execution");
    const object = b.addObject(.{ .name = "qwen4-projection-bodies", .root_module = bodies });
    _ = object.getEmittedBin();
    body_step.dependOn(&object.step);
    parent.dependOn(body_step);
    crossBodies(b, parent);
}

pub fn crossBodies(b: *std.Build, parent: *std.Build.Step) void {
    const step = b.step("test-qwen4-cuda-bodies", "Emit Qwen4 CUDA placement and contract bodies for both Linux architectures");
    for ([_]std.Target.Cpu.Arch{ .aarch64, .x86_64 }) |arch| {
        const target = b.resolveTargetQuery(.{ .cpu_arch = arch, .os_tag = .linux, .abi = .gnu });
        const core = b.createModule(.{ .root_source_file = b.path("zig/qwen4_core_host.zig"), .target = target });
        core.addImport("lanes", b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target }));
        core.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .imports = &.{.{ .name = "lanes", .module = core.import_table.get("lanes").? }} }));
        const family = module(b, target, .Debug, core);
        const cu = @import("cuda_runtime.zig").runtime(b, target, .Debug, &.{});
        const root = b.createModule(.{ .root_source_file = b.path("zig/qwen4_cuda_bodies.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cu }, .{ .name = "qwen4", .module = family } } });
        root.addImport("lanes", core.import_table.get("lanes").?);
        root.addImport("engine_api", core.import_table.get("engine_api").?);
        const object = b.addObject(.{ .name = b.fmt("qwen4-cuda-{t}", .{arch}), .root_module = root });
        _ = object.getEmittedBin();
        step.dependOn(&object.step);
    }
    parent.dependOn(step);
}
