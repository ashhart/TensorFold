//! DeepSeek-V4.1's build: the family's host side (zig/src/families/deepseek_v41) and its host tests on any OS.

const std = @import("std");
const tp = @import("tp.zig");
const kv = @import("kv.zig");
const server_build = @import("server.zig");
const cuda_build = @import("cuda.zig");

/// The backend-neutral core (the safetensors reader the pack headers use) for the family's host side.
fn core(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, tokenizer: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("tokenizer", tokenizer);
    return m;
}

/// The CUDA runtime without kernel images: the family's launch bindings and device loads compile on any host.
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// The family's images in kernels.zig's load order (`dsv41_<file>`), each with its reference extension's nvcc flags.
const units = [_]cuda_build.Kernel{
    .{ .name = "dsv41_exl3_experts", .src = "deepseek_v41/exl3_experts", .flags = &.{ "-O3", "-lineinfo" } }, // tensorfold_exl3_experts_v1
    .{ .name = "dsv41_x3ld", .src = "deepseek_v41/x3ld", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3ld_v1
    .{ .name = "dsv41_x3pf", .src = "deepseek_v41/x3pf", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3pf_v1
    .{ .name = "dsv41_x3gm", .src = "deepseek_v41/x3gm", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3gm_v1
    .{ .name = "dsv41_linear", .src = "deepseek_v41/linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tensorfold_exl3_linear_v4
    .{ .name = "dsv41_x3seg", .src = "deepseek_v41/x3seg", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_x3seg_v3
    .{ .name = "dsv41_dense3", .src = "deepseek_v41/dense3", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_dense3_v1
    .{ .name = "dsv41_attn", .src = "deepseek_v41/attn_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_topk", .src = "deepseek_v41/topk_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_mhc", .src = "deepseek_v41/mhc_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_mhc_cuda_v1
    .{ .name = "dsv41_mhc_pf", .src = "deepseek_v41/mhc_pf", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_mhc_pf_v1
    .{ .name = "dsv41_router_gemv", .src = "deepseek_v41/router_gemv", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_router_gemv_v2
    .{ .name = "dsv41_pfdense_k8", .src = "deepseek_v41/pfdense_k8", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_pfdense_v1
    .{ .name = "dsv41_pfdense_k10", .src = "deepseek_v41/pfdense_k10", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k12", .src = "deepseek_v41/pfdense_k12", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k16", .src = "deepseek_v41/pfdense_k16", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_engram_gate", .src = "deepseek_v41/engram_gate", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_engram_gate_v1
    .{ .name = "dsv41_l2pace", .src = "deepseek_v41/l2pace", .flags = &.{"-O3"} }, // tf_dsv41_l2pace_v1
    .{ .name = "dsv41_l2pf", .src = "deepseek_v41/l2pf", .flags = &.{"-O3"} }, // tf_dsv41_l2pf_v1
    .{ .name = "dsv41_x3gm_plan", .src = "deepseek_v41/x3gm_plan", .flags = &.{"-O3"} }, // ours: x3gm.plan's torch ops
    .{ .name = "dsv41_topk_keys", .src = "deepseek_v41/topk_keys", .flags = &.{"-O3"} }, // ours: pick.top's torch.topk
    .{ .name = "dsv41_pointwise", .src = "deepseek_v41/pointwise", .flags = &.{ "-O3", "--fmad=false" } }, // ours: prefill MoE's torch pointwise ops
    .{ .name = "dsv41_kvsplit", .src = "deepseek_v41/kvsplit", .flags = &.{"-O3"} }, // ours: split KV's exchange copies
    .{ .name = "dsv41_glue", .src = "deepseek_v41/glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the window's torch glue
    .{ .name = "dsv41_x3gm3", .src = "deepseek_v41/x3gm3", .flags = &.{ "-O3", "-lineinfo" } }, // ours: x3gm v3 (x3gm.cu's device code)
    .{ .name = "dsv41_pfglue", .src = "deepseek_v41/prefill_glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the prefill segment's torch glue
    .{ .name = "dsv41_router_glue", .src = "deepseek_v41/router_glue", .flags = &.{ "-O3", "-lineinfo" } }, // histogram group + upstream rotation
    .{ .name = "dsv41_kv_glue", .src = "deepseek_v41/kv_glue", .flags = &.{ "-O3", "-lineinfo" } }, // KV RMS in rounded FP64; store keeps Triton's FMA
    .{ .name = "dsv41_x3ld_epi", .src = "deepseek_v41/x3ld_epi", .flags = &.{ "-O3", "-lineinfo" } }, // ours: x3ld + the expert epilogues
    .{ .name = "dsv41_topk_b", .src = "deepseek_v41/topk_b", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // ours: topk_cuda.cu row-bounded
};

/// Optional images: the keyed sampler (TF_DSV41_SAMPLING) and two-tile linears (TF_DSV41_LIN2); empty, they refuse.
const optional = [_]cuda_build.Kernel{
    .{ .name = "dsv41_sampling", .src = "deepseek_v41/sampling", .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "dsv41_lin2", .src = "deepseek_v41/lin2", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
};

/// SASS targets of the family's images: GB10 (sm_121), the only target its kernels are qualified on.
const family_sms = "121";

/// A build's family images: every unit's (an optional one may be absent), embedded by `withImages`.
pub const Images = struct { units: [units.len]std.Build.LazyPath, optional: [optional.len]?std.Build.LazyPath };

/// The family's images (and `fatbins-dsv41`): -Dnvcc with -Dsm=121, or a -Dfatbins dir's full set (part fails install).
pub fn images(b: *std.Build, nvcc: ?[]const u8, prebuilt: ?[]const u8, version: ?std.Build.LazyPath, sms: []const u8) ?*const Images {
    const step = b.step("fatbins-dsv41", "Build and install DeepSeek-V4.1's kernel fatbins alone (-Dnvcc with -Dsm=121, or -Dfatbins)");
    // else null: the family's binaries keep host-only bindings; other families' fatbins and `fatbins` are unchanged
    const im = b.allocator.create(Images) catch @panic("OOM");
    if (prebuilt) |dir| {
        var missing: usize = 0;
        for (units, &im.units) |k, *f| {
            const file = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) });
            std.Io.Dir.cwd().access(b.graph.io, file, .{}) catch {
                missing += 1;
            };
            f.* = b.graph.cwdRelativePath(file);
        }
        for (optional, &im.optional) |k, *f| {
            const file = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) });
            f.* = if (std.Io.Dir.cwd().access(b.graph.io, file, .{})) |_| b.graph.cwdRelativePath(file) else |_| null;
        }
        if (missing > 0) {
            const fail = &b.addFail(b.fmt("-Dfatbins={s}: {d} of DeepSeek-V4.1's {d} fatbins (dsv41_*.fatbin) missing", .{ dir, missing, units.len })).step;
            step.dependOn(fail);
            if (missing < units.len) b.getInstallStep().dependOn(fail); // part of the set: a mistake, not another family's dir
            return null;
        }
    } else if (nvcc) |compiler| {
        if (!std.mem.eql(u8, sms, family_sms)) {
            step.dependOn(&b.addFail(b.fmt("fatbins-dsv41: DeepSeek-V4.1's kernels build for -Dsm={s} only (got {s})", .{ family_sms, sms })).step);
            return null;
        }
        for (units, &im.units) |k, *f| f.* = cuda_build.fatbin(b, compiler, version.?, k, sms);
        for (optional, &im.optional) |k, *f| f.* = cuda_build.fatbin(b, compiler, version.?, k, sms);
    } else {
        step.dependOn(&b.addFail("fatbins-dsv41 needs -Dnvcc=NVCC (with -Dsm=121) or -Dfatbins=DIR").step);
        return null;
    }
    for (units, im.units) |k, f| step.dependOn(&b.addInstallFile(f, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    for (optional, im.optional) |k, f| if (f) |file| step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    return im;
}

/// The family's kernel bindings and loader (`dsv41_kernels`, kernels.zig) over `cuda`, without fatbins (host-only).
pub fn kernels(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return withImages(b, target, optimize, cuda, null);
}

/// `kernels` with the family's images embedded; null: host-only (every image empty, Kernels.load refuses).
pub fn withImages(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, embedded: ?*const Images) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", embedded != null);
    options.addOption(usize, "dsv41_images", if (embedded != null) units.len else 0); // also keeps this options file apart from the runtime's
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kernels.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addOptions("dsv41_kernel_options", options);
    if (embedded) |im| for (units, im.units) |k, f| m.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{k.name}), .{ .root_source_file = f });
    // an optional image the build has not got is embedded as an empty file
    const empty = b.addWriteFiles().add("empty.fatbin", "");
    for (optional, 0..) |k, i| {
        const f = if (embedded) |im| im.optional[i] orelse empty else empty;
        m.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{k.name}), .{ .root_source_file = f });
    }
    return m;
}

/// The family's host side (config, pack headers, weight plan, kernel bindings, the forward over `tp` and `kv`).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return rooted(b, target, optimize, "deepseek_v41.zig");
}

/// The family's sources with their imports, rooted at `file` (a file of zig/src/families/deepseek_v41).
fn rooted(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, file: []const u8) *std.Build.Module {
    const cuda = runtime(b, target, optimize);
    return over(b, target, optimize, file, cuda, kernels(b, target, optimize, cuda));
}

/// The family's sources rooted at `file` over the given CUDA runtime and kernel module (`dk`).
fn over(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, file: []const u8, cuda: *std.Build.Module, dk: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(b.fmt("zig/src/families/deepseek_v41/{s}", .{file})), .target = target, .optimize = optimize, .link_libc = true });
    // one tokenizer module for the core and the serving module
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("core", core(b, target, optimize, tokenizer));
    m.addImport("cuda", cuda);
    m.addImport("dsv41_kernels", dk);
    const tp_module = tp.module(b, target, optimize, cuda);
    m.addImport("tp", tp_module);
    m.addImport("kv", kv.module(b, target, optimize, cuda, tp_module));
    // the lane core's sampling types, shared with the serving module's exact sampler
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("lanes", lanes);
    m.addImport("dsv41_serve", server_build.family(b, target, optimize, tokenizer, lanes).serve);
    // the served engine: the server's engine seam over the same lanes, and the tokenizer for calibration text
    m.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
    m.addImport("tokenizer", tokenizer);
    return m;
}

/// `tf-dsv41` (tool_main.zig): replay, host and GPU checks, the served engine without HTTP, over `cuda` and `dk`.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, dk: *std.Build.Module, strip: bool) *std.Build.Step.Compile {
    const m = over(b, target, optimize, "tool_main.zig", cuda, dk);
    m.strip = strip;
    return b.addExecutable(.{ .name = "tf-dsv41", .root_module = m });
}

/// `tensorfold-dsv41`: the native server with the family's engine (serve_engine.zig) on rank 0, over `cuda` and `dk`.
pub fn server(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, dk: *std.Build.Module, build_options: *std.Build.Step.Options) *std.Build.Step.Compile {
    const engine = over(b, target, optimize, "serve_engine.zig", cuda, dk);
    return server_build.withEngine(b, target, engine, build_options, "tensorfold-dsv41");
}

/// `tf-dsv41-test`, the device test runner and benches (zig/tests/cuda/dsv41_main.zig), over `cuda` and `dk`.
pub fn runner(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, dk: *std.Build.Module, strip: bool) *std.Build.Step.Compile {
    const m = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/dsv41_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    m.addImport("cuda", cuda);
    m.addImport("dsv41_kernels", dk);
    return b.addExecutable(.{ .name = "tf-dsv41-test", .root_module = m });
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const step = b.step("test-dsv41", "DeepSeek-V4.1's host-side unit tests (no GPU work)");
    const family = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug) }));
    const bindings = b.addRunArtifact(b.addTest(.{ .root_module = kernels(b, host, .debug, runtime(b, host, .debug)) }));
    for ([_]*std.Build.Step.Run{ family, bindings }) |run| {
        step.dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
    // the prefill emitter's and DSpark's own suites, each its own test root (also in test-dsv41 and `zig build test`)
    const suites = [_][3][]const u8{
        .{ "test-dsv41-gm2pf", "gm2pf_test.zig", "DeepSeek-V4.1 x3gm v2 (TF_DSV41_GM_V2) on the real prefill emitter, host only" },
        .{ "test-dsv41-longpf", "longpf_test.zig", "DeepSeek-V4.1 long prompts (stream top-k, split KV's row blocks) on the real prefill emitter, host only" },
        .{ "test-dsv41-slot-drafts", "slot_drafts_test.zig", "DeepSeek-V4.1 DSpark drafting over several slots on the real emitter, host only" },
        .{ "test-dsv41-rows", "rows_test.zig", "DeepSeek-V4.1 row mode (several slots) on the real emitter, host only" },
        .{ "test-dsv41-wide", "wide_test.zig", "DeepSeek-V4.1 decode windows past 16 rows on the real emitter, host only" },
        .{ "test-dsv41-knobs", "serve_knobs_test.zig", "DeepSeek-V4.1 serving knobs (fused dense prefill, rows, read-ahead) on the real prefill emitter, host only" },
    };
    for (suites) |x| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = rooted(b, host, .debug, x[1]) }));
        b.step(x[0], x[2]).dependOn(&run.step);
        step.dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
    // drafting (DSpark depth, trees, the lanes backend) on the CPU twin and the reference's golden traces
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const draft = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/draft/draft.zig"), .target = host, .optimize = .debug, .link_libc = true });
    draft.addImport("lanes", lanes);
    const draft_run = b.addRunArtifact(b.addTest(.{ .root_module = draft }));
    b.step("test-dsv41-draft", "DeepSeek-V4.1's drafting and lanes backend on the CPU twin (no GPU work)").dependOn(&draft_run.step);
    step.dependOn(&draft_run.step);
    test_step.dependOn(&draft_run.step);
    // the device runner compiles on any host; running it needs a GPU and every family image
    const rt = runtime(b, host, .debug);
    const host_dk = kernels(b, host, .debug, rt);
    const compiled = &runner(b, host, .debug, rt, host_dk, false).step;
    step.dependOn(compiled);
    test_step.dependOn(compiled);
    // the tool and the family's server compile on any host too; running them needs a GPU, the images and a pack
    const options = b.addOptions();
    options.addOption([]const u8, "version", @import("../../build.zig.zon").version);
    options.addOption(bool, "slide_test_hooks", false); // main.zig reads it, as build.zig's options carry it
    for ([_]*std.Build.Step.Compile{ tool(b, host, .debug, rt, host_dk, false), server(b, host, .debug, rt, host_dk, options) }) |c| {
        step.dependOn(&c.step);
        test_step.dependOn(&c.step);
    }
}
