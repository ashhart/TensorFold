//! Compile the production file copies into one host identity fixture; no GPU or shader dependency.
const std = @import("std");
pub fn add(b: *std.Build) void {
    const files = b.addWriteFiles();
    var lines = std.mem.tokenizeScalar(u8, @embedFile("../tests/fixtures/gradient-serving.tsv"), '\n');
    while (lines.next()) |line| {
        const tab = std.mem.indexOfScalar(u8, line, '\t').?;
        const path = line[0..tab];
        _ = files.addCopyFile(b.path(path), path);
    }
    _ = files.addCopyFile(b.path("zig/tests/fixtures/gradient-serving.tsv"), "manifest.tsv");
    _ = files.addCopyFile(b.path("zig/tests/fixtures/pocket-serving-approved.tsv"), "approved.tsv");
    const root = files.addCopyFile(b.path("zig/tests/gradient_serving_pin.zig"), "main.zig");
    const test_ = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = root, .target = b.graph.host, .optimize = .debug }) });
    b.step("test-gradient-serving", "Pin the serving kernels' sources and the explicit learning extensions").dependOn(&b.addRunArtifact(test_).step);
}
