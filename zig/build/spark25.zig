//! Spark2.5's CUDA family over the maintained runtime, the core and the lane core; its host tests.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/spark25/spark25.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-spark25-host", "Spark2.5 config admission, RoPE tables, cache sizes, snapshot runs and refusals without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    const cb = @import("cuda.zig");
    const inert = b.addWriteFiles().add("inert-load-only.fatbin", "INERT DRIVER FIXTURE, NEVER A GPU IMAGE");
    const images: [cb.kernels.len]?std.Build.LazyPath = @splat(inert);
    const fake_cuda = cb.runtime(b, b.graph.host, .debug, &images);
    const fake = module(b, b.graph.host, .debug, fake_cuda, m.import_table.get("core").?, m.import_table.get("lanes").?);
    const root = b.createModule(.{ .root_source_file = b.path("zig/spark25_load_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = fake_cuda }, .{ .name = "spark25", .module = fake } } });
    const load = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Spark load:"} }));
    b.step("test-spark25-load-host", "Actual Spark kernel loading over inert images with every driver lookup observed").dependOn(&load.step);
    all.dependOn(&load.step);
}
