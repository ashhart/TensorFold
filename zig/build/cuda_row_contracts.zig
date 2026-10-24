//! CPU reference fixtures for CUDA row contracts; no shader producer or device Run.
const std = @import("std");

pub fn host(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, step: *std.Build.Step) void {
    const q35 = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_qwen_bf16.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const q35_run = b.addRunArtifact(b.addTest(.{ .root_module = q35 }));
    b.step("test-qwen35-bf16-contract-host", "BF16 copy and FP64 dense references plus maintained controls without devices").dependOn(&q35_run.step);
    step.dependOn(&q35_run.step);
    const radio = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_vit_round_host.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const radio_run = b.addRunArtifact(b.addTest(.{ .root_module = radio }));
    b.step("test-radio-fused-contract-host", "RADIO fused rounding registration and finite tie control without devices").dependOn(&radio_run.step);
    step.dependOn(&radio_run.step);
    const stress = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_group_stress_host.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    b.step("test-group-sums-stress-host", "Actual group-sum stress inputs distinguish reduced sums at every seed").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = stress })).step);
    for ([_][]const u8{ "contract_norms_tail.zig", "contract_l2.zig", "contract_act_round.zig", "contract_embed.zig", "contract_qk_round.zig", "contract_window_rows.zig", "contract_hc_mix.zig", "contract_qk_l2.zig", "contract_act_l2.zig" }) |name| {
        const m = b.createModule(.{
            .root_source_file = b.path(b.pathJoin(&.{ "zig/tests/cuda", name })),
            .target = target,
            .optimize = .debug,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
        });
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        step.dependOn(&run.step);
        if (std.mem.eql(u8, name, "contract_window_rows.zig")) b.step("test-window-rows-host", "Wide attention reference, admission coverage and diagnostic failure ownership on CPU").dependOn(&run.step);
    }
}
