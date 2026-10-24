//! Explicit real-tokenizer erase checks always run, rather than reusing an environment-blind test cache.
const std = @import("std");

pub fn add(b: *std.Build, server: *std.Build.Module) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"real-tokenizer erase spans"} }));
    run.has_side_effects = true;
    run.setEnvironmentVariable("TF_NOTES_TOKENIZER_REQUIRED", "1");
    b.step("test-notes-tokenizer", "Always run erased-span checks; requires TF_NOTES_TOKENIZER_DIR").dependOn(&run.step);
}
