//! The shared CUDA dense engine (zig/src/cuda_dense) over the runtime, core and lane core; its host tests.
const std = @import("std");

/// The engine's module, made once a compilation and passed to each dense recipe's family module.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cuda_dense/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    const step = b.step("test-cuda-dense-host", "The CUDA dense engine: attention shapes, pages, RoPE frequencies, Native decls, refusals without a GPU");
    step.dependOn(&run.step);
    all.dependOn(&run.step);
    const offer = b.addRunArtifact(b.addTest(.{ .root_module = offerTest(b, m) }));
    step.dependOn(&offer.step);
    all.dependOn(&offer.step);
}

/// The tower variant through the host's own vision offer and options; the two towers they name by type are stubs.
fn offerTest(b: *std.Build, m: *std.Build.Module) *std.Build.Module {
    const target = m.resolved_target.?;
    const lanes = m.import_table.get("lanes").?;
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const towers = b.createModule(.{ .root_source_file = b.addWriteFiles().add("dense_offer_towers.zig", "pub const native = struct {\n    pub const vision = struct {\n        pub const Vision = struct {};\n    };\n};\n"), .target = target });
    const offer = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_vision_offer.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "qwen35", .module = towers }, .{ .name = "nemotron", .module = towers } } });
    const options = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_options.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "engine_api", .module = api }} });
    return b.createModule(.{ .root_source_file = b.path("zig/src/cuda_dense/offer_test.zig"), .target = target, .optimize = m.optimize, .link_libc = true, .imports = &.{
        .{ .name = "cuda", .module = m.import_table.get("cuda").? },
        .{ .name = "core", .module = m.import_table.get("core").? },
        .{ .name = "lanes", .module = lanes },
        .{ .name = "engine_api", .module = api },
        .{ .name = "vision_offer", .module = offer },
        .{ .name = "cuda_options", .module = options },
    } });
}
