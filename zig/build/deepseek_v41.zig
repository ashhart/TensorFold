//! Experimental DS4.1 weight-loading qualification targets.
const std = @import("std");

pub fn weights(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
    core: *std.Build.Module,
) *std.Build.Step {
    const host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_weights_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "core", .module = core },
        },
    }) });
    const step = b.step("test-deepseek-v41-weights", "V4.1 configuration, Engram table plans and resident weight staging");
    step.dependOn(&b.addRunArtifact(host).step);
    return step;
}

pub fn checks(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
    sources: *std.Build.Module,
    engine: *std.Build.Module,
    core: *std.Build.Module,
) struct { device: *std.Build.Step, reference: *std.Build.Step } {
    const device = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "core", .module = core },
        },
    }) });
    const device_step = b.step("test-deepseek-v41", "Experimental DS4.1 EXL3, Engram, mHC and sparse attention on Metal");
    device_step.dependOn(&b.addRunArtifact(device).step);
    const rows_probe = b.addTest(.{ .root_module = device.root_module, .filters = &.{"batched decode windows"} });
    b.step("test-deepseek-v41-rows", "Per-slot attention, rotation and KV publication at supported head counts")
        .dependOn(&b.addRunArtifact(rows_probe).step);
    const contract = @import("kernel_contract.zig").module(b, target, optimize);
    const row_files = b.addWriteFiles();
    _ = row_files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/rope.metal"), "rope.metal");
    _ = row_files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/kv.metal"), "kv.metal");
    _ = row_files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/kv_glue.metal"), "kv_glue.metal");
    _ = row_files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/csa2.metal"), "csa2.metal");
    _ = row_files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/csa2_nax.metal"), "csa2_nax.metal");
    _ = row_files.addCopyFile(b.path("zig/kernels/metal/deepseek_v41/csa2_rows.metal"), "csa2_rows.metal");
    _ = row_files.addCopyFile(b.path("zig/kernels/metal/nax.h"), "nax.h");
    const row_sources = b.createModule(.{ .root_source_file = row_files.add("sources.zig", "pub const deepseek_v41_rope = @embedFile(\"rope.metal\");\n" ++
        "pub const deepseek_v41_kv = @embedFile(\"kv.metal\");\n" ++
        "pub const deepseek_v41_kv_glue = @embedFile(\"kv_glue.metal\");\n" ++
        "pub const deepseek_v41_csa2 = @embedFile(\"csa2.metal\");\n" ++
        "pub const deepseek_v41_csa2_nax = @embedFile(\"csa2_nax.metal\");\n" ++
        "pub const deepseek_v41_csa2_rows = @embedFile(\"csa2_rows.metal\");\n" ++
        "pub const nax = @embedFile(\"nax.h\");\n") });
    const rows_contract = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_dsv41_native_rows.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = row_sources },
            .{ .name = "contract", .module = contract },
        },
    });
    const rows_gates = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_rows_contract.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = row_sources },
            .{ .name = "contract", .module = contract },
            .{ .name = "rows_contract", .module = rows_contract },
        },
    }) });
    b.step("test-deepseek-v41-rows-contract", "Native row position, BF16 rotation and packed-cache numeric contracts")
        .dependOn(&b.addRunArtifact(rows_gates).step);
    const attention_rows_host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/dsv41_attention_rows_ref.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "contract", .module = contract }},
    }), .filters = &.{ "native attention value cancellation", "native packed decoder" } });
    b.step("test-deepseek-v41-attention-rows-host", "Native packed-cache decoding and attention accumulator discrimination")
        .dependOn(&b.addRunArtifact(attention_rows_host).step);
    const attention_contract = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_dsv41_attention_rows.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = row_sources },
            .{ .name = "contract", .module = contract },
        },
    });
    const attention_rows_gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_attention_rows_contract.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = row_sources },
            .{ .name = "contract", .module = contract },
            .{ .name = "attention_rows_contract", .module = attention_contract },
        },
    }), .filters = &.{"native packed attention dense and scan row contracts"} });
    b.step("test-deepseek-v41-attention-rows-contract", "Native packed dense/scan row gates and real accumulator controls")
        .dependOn(&b.addRunArtifact(attention_rows_gpu).step);
    const lifetime_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_lifetime_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "metal", .module = metal },
            .{ .name = "lanes", .module = engine.import_table.get("lanes").? },
        },
    });
    const lifetime_step = b.step("test-deepseek-v41-lifetime", "Backend ownership, adaptive depth and proposal selection");
    for ([_]struct { lookup: []const u8, filters: []const []const u8 }{
        .{ .lookup = "0", .filters = &.{
            "releases all per-slot host state",
            "drafter ID growth preserves pending Markov picks",
            "trained-only draft depth",
        } },
        .{ .lookup = "3", .filters = &.{
            "suffix drafts follow committed context",
            "lookup proposals take precedence over bound trained drafts",
        } },
    }) |config| {
        const lifetime = b.addTest(.{ .root_module = lifetime_module, .filters = config.filters });
        const lifetime_run = b.addRunArtifact(lifetime);
        lifetime_run.setEnvironmentVariable("TF_DSV41_LOOKUP_DRAFTS", config.lookup);
        lifetime_run.setEnvironmentVariable("TF_DSV41_MTP_DEPTH_MAX", "3");
        lifetime_run.setEnvironmentVariable("TF_DSV41_MTP_ADAPT", "0.7");
        lifetime_run.setEnvironmentVariable("TF_DSV41_MTP_ADAPT_SIGNAL", "first");
        lifetime_run.setEnvironmentVariable("TF_DSV41_MTP_ADAPT_RATCHET", "0");
        lifetime_run.setEnvironmentVariable("TF_DSV41_MTP_ADAPT_CONFIRM", "1");
        lifetime_step.dependOn(&lifetime_run.step);
    }
    // The family's Metal sources carry struct-size static_asserts; compile them the way the engine joins them.
    const mirrors = b.addSystemCommand(&.{ "sh", "tools/zig/check_dsv41_metal_mirrors.sh" });
    mirrors.setCwd(b.path("."));
    device_step.dependOn(&mirrors.step);
    // A one-test slice of the same module: the index select's determinism probe.
    const probe = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "core", .module = core },
        },
    }), .filters = &.{"index one-shot select"} });
    const probe_run = b.addRunArtifact(probe);
    b.step("test-deepseek-v41-index-probe", "The index one-shot select's same-input determinism, one test").dependOn(&probe_run.step);
    const reference = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/deepseek_v41_reference_checks.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "tensorfold", .module = engine },
                .{ .name = "metal", .module = metal },
                .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, optimize) },
            },
        }),
        .filters = &.{"tests.deepseek_v41_reference."},
    });
    const reference_step = b.step("test-deepseek-v41-reference", "Independent CPU teacher-oracle arithmetic boundaries");
    reference_step.dependOn(&b.addRunArtifact(reference).step);
    const qkv = b.addTest(.{ .root_module = reference.root_module, .filters = &.{"one-row fused wq_a+wkv"} });
    b.step("test-deepseek-v41-qkv", "Fused QKV projection parity without the unrelated reference corpus")
        .dependOn(&b.addRunArtifact(qkv).step);
    return .{ .device = device_step, .reference = reference_step };
}

/// The qualification runner on real weights: `tf-dsv41-run`, installed with the native tools.
pub fn program(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    engine: *std.Build.Module,
    metal: *std.Build.Module,
) void {
    const run_mod = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_run.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "metal", .module = metal },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, optimize) },
        },
    });
    const run_exe = b.addExecutable(.{ .name = "tf-dsv41-run", .root_module = run_mod });
    b.installArtifact(run_exe);
    b.step("tf-dsv41-run", "DSV41 qualification receipts on real weights").dependOn(&b.addInstallArtifact(run_exe, .{}).step);
}
