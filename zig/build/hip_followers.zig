//! The HIP followers' step parsers: what rank 0 sends a rank above it, taken or refused (#3042, #3081), no device.
const std = @import("std");

/// Tests by name: the shared loop's and each family worker's step parsers, and the sender they pair with.
pub const parsers = [_][]const u8{
    "security: a step from the link", // protocol.zig and flashnext/hip/worker.zig (#3042)
    "security: rank 0's page and slot", // protocol.zig's bounds the Nemotron and Qwen workers use (#3081)
    "security: a begin's cache slot", // flashnext/hip/worker.zig (#3081)
    "a sender ", // protocol.zig: what rank 0's Sender puts in a message
    "the words of a prefill message", // nemotron/hip/worker.zig
};

/// `test-hip-followers-host` over the host modules (hip_serve, Flash Next HIP, Nemotron HIP) with inert kernels.
pub fn steps(b: *std.Build, modules: []const *std.Build.Module) void {
    const step = b.step("test-hip-followers-host", "HIP follower step parsers and their refusals without device execution");
    for (modules) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &parsers })).step);
}
