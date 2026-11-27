//! Focused GPU-free tests and semantic compilation for the opt-in Qwen prompt pipeline.
const std = @import("std");

fn root(b: *std.Build, file: []const u8, parent: *std.Build.Module) *std.Build.Module {
    const module = b.createModule(.{ .root_source_file = b.path(file), .target = b.graph.host, .link_libc = true });
    for (parent.import_table.keys(), parent.import_table.values()) |name, value| module.addImport(name, value);
    return module;
}

pub fn host(b: *std.Build, cuda: *std.Build.Module) void {
    const module = root(b, "zig/src/cuda/prompt_stages2_host_test.zig", cuda);
    const test_ = b.addTest(.{ .root_module = module, .filters = &.{"prompt stages2:"} });
    b.step("test-qwen-prompt-stages2-host", "Exact prompt dispatch and two-CTA admission without a GPU").dependOn(&b.addRunArtifact(test_).step);
}

pub fn consumers(b: *std.Build, runner: *std.Build.Module) void {
    const object = root(b, "zig/tests/cuda/prompt_stages2_compile.zig", runner);
    b.step("compile-qwen-prompt-stages2-host", "Compile real prompt profile, contract and Qwen startup without devices").dependOn(&b.addObject(.{ .name = "qwen-prompt-stages2", .root_module = object }).step);
    const selectors = root(b, "zig/tests/cuda/qwen_prompt_gemm_profile.zig", runner);
    b.step("test-qwen-prompt-profile-host", "Profile variants and shape refusals without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = selectors, .filters = &.{"Qwen prompt profile:"} })).step);
    const record = root(b, "zig/qwen_prompt_stages2_record_host.zig", runner);
    b.step("test-qwen-prompt-stages2-record-host", "Missing register receipts cannot qualify the opt-in kernel").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = record, .filters = &.{"Qwen stages2 records:"} })).step);
}
