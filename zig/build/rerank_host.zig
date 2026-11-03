//! test-rerank-host: the /v1/rerank route and the reranker's labels on the CPU; TF_QWEN3_RERANK_* checks real ids.
const std = @import("std");

pub fn steps(b: *std.Build, server_tests: *std.Build.Module) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"rerank:"} }));
    run.has_side_effects = true; // TF_QWEN3_RERANK_DIR and _REF are not cached inputs
    b.step("test-rerank-host", "Reranker labels and the /v1/rerank route; TF_QWEN3_RERANK_* checks real ids").dependOn(&run.step);
}
