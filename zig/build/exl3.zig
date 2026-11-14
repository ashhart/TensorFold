const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, host_tests: *std.Build.Step) void {
    const files = b.addWriteFiles();
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_core.metal"), "exl3_projection_core.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_wide.metal"), "exl3_projection_wide.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_chain.metal"), "exl3_projection_chain.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_base.metal"), "exl3_projection_nax_base.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_prefill_body.metal"), "exl3_projection_nax_prefill_body.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_prefill_ku.metal"), "exl3_projection_nax_prefill_ku.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_prefill.metal"), "exl3_projection_nax_prefill.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_variants_a.metal"), "exl3_projection_nax_variants_a.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_variants_b.metal"), "exl3_projection_nax_variants_b.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_variants_c.metal"), "exl3_projection_nax_variants_c.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_fma.metal"), "exl3_projection_nax_fma.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_fma_static_a.metal"), "exl3_projection_nax_fma_static_a.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_fma_static_b.metal"), "exl3_projection_nax_fma_static_b.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_fma_t.metal"), "exl3_projection_nax_fma_t.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_nax_fp8.metal"), "exl3_projection_nax_fp8.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/exl3_projection_i8.metal"), "exl3_projection_i8.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/nax.h"), "nax.h");
    // The engine activation glue, for the fused SwiGLU+rotate differential (its precise silu is the reference).
    _ = files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/ops.metal"), "deepseek_v41_ops.metal");
    // The query RoPE, for the fused wq_b epilogue differential (dsv41_rope is the reference).
    _ = files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/rope.metal"), "deepseek_v41_rope.metal");
    _ = files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/kv_glue.metal"), "deepseek_v41_kv_glue.metal");
    const sources = b.createModule(.{
        .root_source_file = files.add("sources.zig", "pub const core_exl3_projection = @embedFile(\"exl3_projection_core.metal\") ++ @embedFile(\"exl3_projection_wide.metal\") ++ @embedFile(\"exl3_projection_chain.metal\");\n" ++
            "pub const core_exl3_projection_i8 = @embedFile(\"exl3_projection_i8.metal\");\n" ++
            "pub const core_exl3_projection_nax = @embedFile(\"exl3_projection_nax_base.metal\") ++ @embedFile(\"exl3_projection_nax_prefill_body.metal\") ++ @embedFile(\"exl3_projection_nax_prefill_ku.metal\") ++ @embedFile(\"exl3_projection_nax_prefill.metal\") ++ @embedFile(\"exl3_projection_nax_variants_a.metal\") ++ @embedFile(\"exl3_projection_nax_variants_b.metal\") ++ @embedFile(\"exl3_projection_nax_variants_c.metal\") ++ @embedFile(\"exl3_projection_nax_fma.metal\") ++ @embedFile(\"exl3_projection_nax_fma_static_a.metal\") ++ @embedFile(\"exl3_projection_nax_fma_static_b.metal\") ++ @embedFile(\"exl3_projection_nax_fma_t.metal\") ++ @embedFile(\"exl3_projection_nax_fp8.metal\");\n" ++
            "pub const nax = @embedFile(\"nax.h\");\n" ++
            "pub const deepseek_v41_ops = @embedFile(\"deepseek_v41_ops.metal\");\n" ++
            "pub const deepseek_v41_rope = @embedFile(\"deepseek_v41_rope.metal\");\n" ++
            "pub const deepseek_v41_kv_glue = @embedFile(\"deepseek_v41_kv_glue.metal\");\n"),
    });
    const imports: []const std.Build.Module.Import =
        &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } };
    const projection = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exl3_projection.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = imports,
    });
    const bench = b.addExecutable(.{ .name = "tf-exl3-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/exl3_projection_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "exl3_projection", .module = projection } },
    }) });
    b.step("tf-exl3-bench", "Standalone deterministic EXL3 FP64 oracle and relative operator benchmark")
        .dependOn(&b.addInstallArtifact(bench, .{}).step);
    const host = b.addRunArtifact(b.addTest(.{ .root_module = projection }));
    host.setCwd(b.path("."));
    const oracle_tests = b.addRunArtifact(b.addTest(.{ .root_module = bench.root_module }));
    oracle_tests.setCwd(b.path("."));
    const host_step = b.step("test-exl3", "EXL3 extraction, codebook, shape, storage and oracle host contracts");
    const load_check = b.addObject(.{ .name = "exl3_metal_loader", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/exl3_metal_compile.zig"),
        .target = target,
        .link_libc = true,
        .imports = &.{ .{ .name = "exl3_projection", .module = projection }, .{ .name = "metal", .module = metal } },
    }) });
    host_step.dependOn(&load_check.step);
    host_step.dependOn(&host.step);
    host_step.dependOn(&oracle_tests.step);
    host_tests.dependOn(host_step);
    const gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exl3_projection_gpu_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = imports,
    }) });
    const gpu_run = b.addRunArtifact(gpu);
    gpu_run.setCwd(b.path("."));
    b.step("test-exl3-gpu", "EXL3 Metal arithmetic, row windows, routes and validation regressions")
        .dependOn(&gpu_run.step);
}
