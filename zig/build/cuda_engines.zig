//! The CUDA engines a native server opens (native/cuda.zig) and the family modules they import.
const std = @import("std");

/// The GLM-5.3-Flash CUDA family over the runtime, the core, the lane core, canon's shares and core/heat.zig.
pub fn glmCuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, heat: *std.Build.Module) *std.Build.Module {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("canon", canon);
    m.addImport("lanes", lanes);
    m.addImport("heat", heat);
    return m;
}

/// LFM2 (d1's decision model) on CUDA: config, weights, kernels and the forward that scores labels.
pub fn lfm2Module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm2/lfm2.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// Flash Next's CUDA family, sharing the maintained runtime and lane core.
pub fn flashnextModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, qwen_vision: *std.Build.Module, api: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_launch.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("qwen_vision", qwen_vision);
    m.addImport("living", core.import_table.get("living").?);
    m.addImport("engine_api", api); // its disk image (cuda_disk.zig) names the rows store's types: family()'s, the server's
    return m;
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
pub fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = qwen35.import_table.get("engine_api").?; // family()'s, so the server and qwen3_5 share one
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "engine_api", .module = api },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "nemotron", .module = nemotron },
            .{ .name = "drama3", .module = @import("drama3_host.zig").module(b, target, optimize, cuda, core) },
            .{ .name = "lfm2", .module = lfm2Module(b, target, optimize, cuda, core) },
            .{ .name = "core", .module = core },
            .{ .name = "flashnext", .module = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision, api) },
            .{ .name = "glm_cuda", .module = glmCuda(b, target, optimize, cuda, core, lanes, heat) },
            .{ .name = "qwen35", .module = qwen35 },
            .{ .name = "kolibri1", .module = @import("kolibri.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "gemma4_cuda", .module = @import("gemma4_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "spark25", .module = @import("spark25.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "dory", .module = @import("dory.zig").module(b, target, optimize, cuda, core, lanes) },
        },
    });
    mod.addImport("qwen4", @import("qwen4.zig").module(b, target, optimize, core));
    return .{ .api = api, .engines = mod };
}
