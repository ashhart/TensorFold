//! `zig build dev-tools`: every tf-* developer executable a macOS target declares, compiled and linked, none run.

const std = @import("std");

/// Once every step is declared: each tf-* step's executables, linked into the cache rather than installed.
pub fn wire(b: *std.Build, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag != .macos) return;
    const all = b.step("dev-tools", "Compile and link every tf-* developer tool (macOS), installing and running none");
    for (b.top_level_steps.values()) |top| {
        if (!std.mem.startsWith(u8, top.step.name, "tf-")) continue;
        for (top.step.dependencies.items) |dep| {
            const install = dep.cast(std.Build.Step.InstallArtifact) orelse {
                all.dependOn(dep);
                continue;
            };
            all.dependOn(&install.artifact.step);
        }
    }
}
