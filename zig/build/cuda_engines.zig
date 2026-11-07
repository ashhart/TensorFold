//! The CUDA engines a native server opens (native/cuda.zig) and the family modules they import.
const std = @import("std");

/// canon.zig as its own module. A compilation that also builds cluster.zig must pass this same module in.
pub fn canonModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
}

/// The GLM-5.3-Flash CUDA family over the runtime, the core, the lane core, canon's shares, core/heat.zig, engine_api.
pub fn glmCuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, heat: *std.Build.Module, api: *std.Build.Module) *std.Build.Module {
    return glmCudaShared(b, target, optimize, cuda, core, lanes, heat, api, canonModule(b, target, optimize), @import("tp.zig").verbsModule(b, target, optimize));
}

/// GLM with the caller's canon and verbs modules, so cluster and fabric in one binary do not root those files twice.
pub fn glmCudaShared(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, heat: *std.Build.Module, api: *std.Build.Module, canon: *std.Build.Module, verbs: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("canon", canon);
    m.addImport("mesh", @import("mesh.zig").module(b, target, optimize, canon)); // shares declared over that canon
    m.addImport("lanes", lanes);
    m.addImport("heat", heat);
    m.addImport("tp", @import("tp.zig").module(b, target, optimize, cuda)); // over the engine's own cuda module
    m.addImport("engine_api", api); // TF_KV_SHARE's prompt_radix store: family()'s, the server's
    m.addImport("tp", @import("tp.zig").moduleWith(b, target, optimize, cuda, verbs)); // over the engine's own cuda module
    m.addImport("engine_api", api); // TF_GLM_KV_SHARE's prompt_radix store: family()'s, the server's
    return m;
}

/// Qwen3's dense embedding and reranker checkpoints on CUDA: the family module the native server's open dispatches to.
pub fn qwen3Module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3/qwen3_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("engine_api", api); // the server's engine_api: family's (qwen3_5's), the shared one
    return m;
}

/// The family module's GPU-free host tests, without a GPU; `all` (test-cuda-host and so `test`) runs them.
pub fn qwen3HostTest(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-qwen3-cuda-host", "Qwen3 CUDA host tests: config, folder head, prompt lane and reranker labels (no GPU)").dependOn(&run.step);
    all.dependOn(&run.step);
}

/// The Ling-3.1-flash CUDA family over the runtime, the core, the lane core and the GLM kernels it reuses.
pub fn lingCuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, glm: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for ([_]struct { []const u8, *std.Build.Module }{ .{ "glm_cuda", glm }, .{ "cuda", cuda }, .{ "core", core }, .{ "lanes", lanes } }) |x| m.addImport(x[0], x[1]);
    m.addImport("engine_api", glm.import_table.get("engine_api").?); // its disk image names the rows store's types: GLM's, the server's
    return m;
}

/// test-ling: Ling-3.1-flash's CUDA family on the host (loader, graph, engine, lanes and routing; no GPU work).
pub fn lingHost(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, glm: *std.Build.Module, step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = lingCuda(b, host, .debug, cuda, core, lanes, glm) }));
    step.dependOn(&run.step);
    b.step("test-ling", "Ling-3.1-flash's CUDA family on the host: loader, graph, engine and lanes compile, no GPU work").dependOn(&run.step);
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
pub fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module, canon: *std.Build.Module, verbs: *std.Build.Module } {
    const api = qwen35.import_table.get("engine_api").?; // family()'s, so the server and qwen3_5 share one
    const canon = canonModule(b, target, optimize);
    const verbs = @import("tp.zig").verbsModule(b, target, optimize);
    const glm = glmCudaShared(b, target, optimize, cuda, core, lanes, heat, api, canon, verbs);
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
            .{ .name = "glm_cuda", .module = glm },
            .{ .name = "qwen3_cuda", .module = qwen3Module(b, target, optimize, cuda, core, lanes, api) },
            .{ .name = "minimax_m3_cuda", .module = @import("minimax_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "ling_cuda", .module = lingCuda(b, target, optimize, cuda, core, lanes, glm) },
            .{ .name = "qwen35", .module = qwen35 },
            .{ .name = "kolibri1", .module = @import("kolibri.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "gemma4_cuda", .module = @import("gemma4_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "spark25", .module = @import("spark25.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "lfm2_moe_cuda", .module = @import("lfm2_moe_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "dory", .module = @import("dory.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "dots_mocr_cuda", .module = @import("dots_mocr_cuda.zig").module(b, target, optimize, cuda, core, lanes, @import("cuda_dense.zig").module(b, target, optimize, cuda, core, lanes)) },
        },
    });
    mod.addImport("qwen4", @import("qwen4.zig").module(b, target, optimize, core));
    return .{ .api = api, .engines = mod, .canon = canon, .verbs = verbs };
}
