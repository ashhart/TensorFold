//! The GPTQ int4 host tests: the packer, the stored codes, the binary64 reference and the launch guards, no device.
const std = @import("std");

/// `test-int4-host` over the cuda module's `int4:` tests, also run by `test-cuda-host`.
pub fn steps(b: *std.Build, cuda: *std.Build.Module) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"int4:"} }));
    b.step("test-int4-host", "The GPTQ int4 packer, its stored codes and the binary64 reference (cuda/int4.zig)")
        .dependOn(&run.step);
    if (b.top_level_steps.get("test-cuda-host")) |host| host.step.dependOn(&run.step);
}
