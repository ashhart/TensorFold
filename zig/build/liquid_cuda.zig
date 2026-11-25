//! What the Liquid families (LFM2.5-24B-A2B, LFM3) share on CUDA: one module a runtime, so both import its files once.
const std = @import("std");

const Made = struct { cuda: *std.Build.Module, optimize: std.builtin.OptimizeMode, m: *std.Build.Module };
var made: std.ArrayList(Made) = .empty;

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    for (made.items) |x| if (x.cuda == cuda and x.optimize == optimize) return x.m;
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/liquid/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
    made.append(b.allocator, .{ .cuda = cuda, .optimize = optimize, .m = m }) catch @panic("liquid_cuda: out of memory");
    return m;
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-liquid-cuda-host", "The Liquid families' shared lane backend, kept-state pages and native glue without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
