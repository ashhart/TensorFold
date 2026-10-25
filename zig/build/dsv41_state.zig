const std = @import("std");
pub fn kvModule(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, tp: *std.Build.Module) *std.Build.Module {
    return kvModuleAt(b, target, .Debug, cuda, tp);
}
pub fn kvModuleAt(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, tp: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/kv.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("sessions", @import("sessions.zig").module(b, target, optimize));
    m.addImport("cuda", cuda);
    m.addImport("tp", tp);
    return m;
}
pub fn steps(b: *std.Build) void {
    const state = @import("dsv41_shared.zig").module(b, b.graph.host, "kv_state.zig");
    b.step("test-dsv41-state", "Shared KV transactions and state seam regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = state })).step);
    inline for (.{ false, true }) |cross| {
        const target = if (cross) b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }) else b.graph.host;
        const root = @import("dsv41_cuda.zig").module(b, target, "state_test.zig", cross);
        const check = b.addTest(.{ .root_module = root });
        if (cross) {
            _ = check.getEmittedBin();
            b.step("compile-dsv41-forward", "Emit Linux forward, graphs and prefill bodies").dependOn(&check.step);
        } else b.step("test-dsv41-forward", "Host forward and prompt plan checks").dependOn(&b.addRunArtifact(check).step);
    }
    const linux = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
    inline for (.{ false, true }) |cross| {
        const target = if (cross) linux else b.graph.host;
        const cuda = @import("dsv41_cuda.zig").runtime(b, target, false);
        const tp = @import("tp.zig").module(b, target, .Debug, cuda);
        const t = b.addTest(.{ .root_module = kvModule(b, target, cuda, tp) });
        if (cross) {
            _ = t.getEmittedBin();
            b.step("compile-dsv41-kv", "Emit Linux paged/split KV host tests").dependOn(&t.step);
        } else b.step("test-dsv41-kv", "Paged/split KV and session host tests").dependOn(&b.addRunArtifact(t).step);
    }
}
