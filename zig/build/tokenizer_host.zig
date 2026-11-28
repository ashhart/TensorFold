//! The tokenizer's own tests, every one and unfiltered, without a GPU, a shader compiler or a checkpoint.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    // libc: two tests run their body in a forked child (sys.isolate); debug keeps the safety checks they rely on
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tokenizer }));
    b.step("test-tokenizer-host", "The tokenizer modules' own tests, unfiltered, without a GPU or shader compiler").dependOn(&run.step);
}
