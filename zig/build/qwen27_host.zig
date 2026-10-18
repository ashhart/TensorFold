//! Host regressions use no shaders, model or GPU execution.
const std = @import("std");
pub fn tests(b: *std.Build) void {
    const ane_policy = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/ane_policy.zig"), .target = b.graph.host });
    b.step("test-qwen27-ane-policy-host", "Neural Engine opt-in and tensor-unit chip policy without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ane_policy })).step);
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_speed_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "metal", .module = metal }} });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"qwen27 speed:"} }));
    run.has_side_effects = true;
    b.step("test-qwen27-speed-host", "Prompt tap guards, attention admission and cache failure ownership").dependOn(&run.step);
    const fake = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_speed_metal.zig"), .target = b.graph.host });
    const core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_speed_core.zig"), .target = b.graph.host, .imports = &.{.{ .name = "metal", .module = fake }} });
    const sources = b.createModule(.{ .root_source_file = b.addWriteFiles().add("qwen27-fake-source.zig", "pub const core_deltanet = \"\";\n"), .target = b.graph.host });
    const delta = b.createModule(.{ .root_source_file = b.path("zig/src/core/deltanet_kernels.zig"), .target = b.graph.host, .imports = &.{.{ .name = "metal", .module = fake }} });
    delta.addImport("kernel_sources", sources);
    const keep_module = b.createModule(.{
        .root_source_file = b.path("zig/src/qwen27_keep_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "core", .module = core }, .{ .name = "core_delta", .module = delta } },
    });
    const bank_module = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_bank_test.zig"), .target = b.graph.host, .optimize = .debug, .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "core", .module = core }, .{ .name = "core_delta", .module = delta } } });
    b.step("test-qwen27-bank-host", "Actual DeltaNet bank ownership with a recording device, no GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = bank_module, .filters = &.{"qwen27 bank:"} })).step);
    const keep = b.addRunArtifact(b.addTest(.{ .root_module = keep_module, .filters = &.{"qwen27 speed:"} }));
    keep.has_side_effects = true;
    b.step("test-qwen27-keep-host", "Actual shared round keep with a recording Metal facade").dependOn(&keep.step);
    var names: std.ArrayList([]const u8) = .empty;
    tryNames(b, &names, @embedFile("../src/families/qwen27/gdn_test.zig"));
    tryNames(b, &names, @embedFile("../src/families/qwen27/attention_plan.zig"));
    const plans = b.addRunArtifact(b.addTest(.{ .root_module = keep_module, .filters = names.items }));
    plans.has_side_effects = true;
    b.step("test-qwen27-plans-host", "Shared GDN and attention plans with no shaders or GPU").dependOn(&plans.step);

    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host });
    const equal = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen27_lanes_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    }), .filters = &.{"qwen27 speed:"} }));
    equal.has_side_effects = true;
    b.step("test-qwen27-equality-host", "Shared lanes drafted, resumed and concurrent equality on the fake target").dependOn(&equal.step);
}

pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_served.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    const boundary = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen27_bank_boundary_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "tensorfold", .module = engine }} }), .filters = &.{"qwen27 bank boundary:"} });
    b.step("test-qwen27-bank-boundary-host", "Runner pending-keep guards with no device or shader execution").dependOn(&b.addRunArtifact(boundary).step);
    const object = b.addObject(.{ .name = "qwen27-speed-compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen27_speed_compile.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api }, .{ .name = "qwen27_served", .module = host } },
    }) });
    _ = object.getEmittedBin();
    b.step("check-qwen27-speed-host", "Compile the actual Qwen27 startup and encoding without shaders").dependOn(&object.step);
}

fn tryNames(b: *std.Build, names: *std.ArrayList([]const u8), text: []const u8) void {
    var rest = text;
    while (std.mem.indexOf(u8, rest, "test \"")) |at| {
        rest = rest[at + 6 ..];
        const end = std.mem.indexOfScalar(u8, rest, '"').?;
        names.append(b.allocator, rest[0..end]) catch @panic("OOM");
        rest = rest[end + 1 ..];
    }
}
