//! Recording controls for served family rounds, copies, graph shapes and startup warmups.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    const host = b.graph.host;
    const startup = b.createModule(.{ .root_source_file = b.path("zig/cuda_startup_costs_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    b.step("test-cuda-startup-costs-host", "Eager measured CUDA costs and transactional failure controls without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = startup })).step);

    const mtp_policy = b.createModule(.{ .root_source_file = b.path("zig/nemotron_mtp_policy_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "nemotron", .module = mods.nemotron }, .{ .name = "lanes", .module = mods.lanes } } });
    const mtp_run = b.addRunArtifact(b.addTest(.{ .root_module = mtp_policy, .filters = &.{"mtp policy:"} }));
    b.step("test-nemotron-mtp-policy-host", "Nemotron controls through shared rounds, resumes and handoff without a GPU").dependOn(&mtp_run.step);
    b.step("test-nemotron-restored-head-host", "Actual Nemotron cache restore, bind and captured draft dispatch without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.nemotron, .filters = &.{"restored head:"} })).step);
    b.step("test-nemotron-subset-head-host", "Actual subset-head ownership and row gathering without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.nemotron, .filters = &.{"subset head:"} })).step);
    step.dependOn(&mtp_run.step);
    const decode_host = b.createModule(.{ .root_source_file = b.path("zig/qwen35_decode_host.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "core", .module = mods.core }} });
    b.step("test-qwen35-decode-records-host", "Optimized Qwen decode launch records on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = decode_host })).step);
    b.step("test-shared-lane-regressions-host", "Every shared lane regression without device libraries").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.lanes })).step);
    b.step("test-qwen35-draft-trees-host", "Grown trees through the actual shared draft frame and wrapper").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen35_draft_frame_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "core", .module = mods.core }, .{ .name = "lanes", .module = mods.lanes } } }), .filters = &.{"Qwen CUDA contracts:"} })).step);
    const copies = b.createModule(.{ .root_source_file = b.path("zig/qwen35_copies_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = mods.lanes }, .{ .name = "qwen35", .module = mods.qwen35 } } });
    const copies_run = b.addRunArtifact(b.addTest(.{ .root_module = copies, .filters = &.{"Suffix copies:"} }));
    copies_run.has_side_effects = true;
    b.step("test-qwen35-copies-host", "CUDA family copy facts and shared lane equality, no GPU or shaders").dependOn(&copies_run.step);
    const rounds = b.createModule(.{ .root_source_file = b.path("zig/qwen35_rounds_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "core", .module = mods.core }, .{ .name = "lanes", .module = mods.lanes } } });
    const rounds_run = b.addRunArtifact(b.addTest(.{ .root_module = rounds, .filters = &.{"Verify width calibration:"} }));
    rounds_run.has_side_effects = true;
    b.step("test-qwen35-rounds-host", "CUDA 64-row metadata and shared register-slot plans without devices").dependOn(&rounds_run.step);
    const shared = b.createModule(.{ .root_source_file = b.path("zig/qwen35_shared_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "core", .module = mods.core }, .{ .name = "lanes", .module = mods.lanes } } });
    const shared_run = b.addRunArtifact(b.addTest(.{ .root_module = shared, .filters = &.{"Shared verification:"} }));
    shared_run.has_side_effects = true;
    b.step("test-qwen35-shared-host", "Shared CUDA slots and rollback without device execution").dependOn(&shared_run.step);
    const draft_frame = b.createModule(.{ .root_source_file = b.path("zig/qwen35_draft_frame_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "core", .module = mods.core }, .{ .name = "lanes", .module = mods.lanes } } });
    b.step("test-qwen35-frame-invalidation-host", "A failed shared frame invalidates held tokens and grown trees").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = draft_frame, .filters = &.{"Draft frames: the real hold adapter"} })).step);
    const frame_run = b.addRunArtifact(b.addTest(.{ .root_module = draft_frame, .filters = &.{"Draft frames:"} }));
    frame_run.has_side_effects = true;
    b.step("test-qwen35-draft-frame-host", "Batched CUDA drafter frame bookkeeping and lane equality without devices").dependOn(&frame_run.step);
    const cache = b.createModule(.{ .root_source_file = b.path("zig/qwen35_cost_cache_host.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "lanes", .module = mods.lanes } } });
    const cost_run1 = b.addRunArtifact(b.addTest(.{ .root_module = cache, .filters = &.{"window cost cache:"} }));
    b.step("test-qwen35-cost-cache-host", "Real width-loop cache misses and hits without a GPU").dependOn(&cost_run1.step);
    step.dependOn(&cost_run1.step);
    const verify_contract = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_verify_host.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    b.step("test-qwen35-verify-contract-host", "Wider affine verify contracts, FP64 and exact gate controls without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = verify_contract })).step);
    const verify = b.createModule(.{ .root_source_file = b.path("zig/qwen35_verify_host.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = mods.qwen35 } } });
    b.step("test-qwen35-verify-host", "Actual verify matmul launches, row buckets and weight-pass counts without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = verify })).step);
    const live = b.createModule(.{ .root_source_file = b.path("zig/qwen35_live_host.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = mods.qwen35 } } });
    const live_run = b.addRunArtifact(b.addTest(.{ .root_module = live }));
    live_run.has_side_effects = true;
    b.step("test-qwen35-live-host", "Live CUDA launch buckets and startup warmups without devices").dependOn(&live_run.step);
}
