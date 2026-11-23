//! test-rerank-host: the rerank and score routes and the reranker's labels; TF_QWEN3_RERANK_* checks real ids.
const std = @import("std");

pub fn steps(b: *std.Build, server_tests: *std.Build.Module) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "rerank:", "score shapes:" } }));
    run.has_side_effects = true; // TF_QWEN3_RERANK_DIR and _REF are not cached inputs
    b.step("test-rerank-host", "Reranker labels, /v1/rerank and its /score, TEI and /v2/rerank shapes; TF_QWEN3_RERANK_* checks real ids").dependOn(&run.step);
}
