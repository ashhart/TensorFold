//! Flash Next's device checks and explicit model diagnostics, each under its own build step.

const std = @import("std");

/// tf-flashnext-widths: every product class at every oQ width against float64, every row's bits in every window.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-flashnext-widths", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_widths.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } },
    }) });
    const about = "Flash Next's products at every oQ width against float64 (worst column), every row's bits in every window";
    b.step("tf-flashnext-widths", about).dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// Explicit model diagnostics; these tools do not qualify a checkpoint merely by compiling.
pub fn cudaTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, flashnext: *std.Build.Module, opts: @import("options.zig").Cuda) void {
    // not installed: one embedding row through the captured _embed cubin
    const embed_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_embed_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    embed_check.addImport("cuda", cuda);
    embed_check.addImport("core", core);
    embed_check.addImport("flashnext", flashnext);
    const embed_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-embed", .root_module = embed_check }));
    const embed_model = opts.flashnext_model;
    const embed_kernels = opts.flashnext_kernels;
    const embed_token = opts.flashnext_token;
    if (embed_model != null and embed_kernels != null) embed_run.addArgs(&.{ embed_model.?, embed_kernels.?, embed_token });
    b.step("flashnext-embed", "Upload the Flash Next embedding table and run _embed for one token").dependOn(&embed_run.step);
    const prompt_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_prompt_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    prompt_check.addImport("cuda", cuda);
    prompt_check.addImport("core", core);
    prompt_check.addImport("flashnext", flashnext);
    const prompt_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-prompt", .root_module = prompt_check }));
    const prompt_ids = opts.flashnext_ids;
    const prompt_count = opts.flashnext_count;
    if (embed_model != null and embed_kernels != null and prompt_ids != null) prompt_run.addArgs(&.{ embed_model.?, embed_kernels.?, prompt_ids.?, prompt_count });
    b.step("flashnext-prompt", "Run one Flash Next prompt on the captured CUDA kernels").dependOn(&prompt_run.step);
    // not installed: our own glue against a captured set on one loaded engine (scores of every prompt row, replies)
    const compare_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_compare_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    compare_check.addImport("cuda", cuda);
    compare_check.addImport("flashnext", flashnext);
    const compare_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-compare", .root_module = compare_check }));
    if (embed_model != null and embed_kernels != null and prompt_ids != null) compare_run.addArgs(&.{ embed_model.?, embed_kernels.?, prompt_ids.? });
    b.step("flashnext-compare", "Our Flash Next glue against a captured set: every prompt row's scores, perplexity, greedy replies").dependOn(&compare_run.step);
    // not installed: the native block selection against Triton's lists (capture_sparse.sh's fixtures)
    const select_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_select_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    select_check.addImport("cuda", cuda);
    const select_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-select", .root_module = select_check }));
    if (opts.flashnext_fixtures) |dir| select_run.addArgs(&.{dir});
    b.step("flashnext-select", "fn_qsa_select against Triton's _select / _select_tiles lists on captured score rows").dependOn(&select_run.step);
    const learn_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_learn_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    learn_check.addImport("cuda", cuda);
    learn_check.addImport("flashnext", flashnext);
    const learn_exe = b.addExecutable(.{ .name = "flashnext-learn", .root_module = learn_check });
    const learn_run = b.addRunArtifact(learn_exe);
    if (embed_model != null and prompt_ids != null) learn_run.addArgs(&.{ embed_model.?, prompt_ids.? });
    b.step("flashnext-learn", "Trainer forward identity and all-path finite-difference gates").dependOn(&learn_run.step);
    b.step("compile-flashnext-learn", "Compile the real trainer runner without executing it").dependOn(&learn_exe.step);
}
