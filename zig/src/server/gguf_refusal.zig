//! The host adapter does not admit a GGUF file into the checkpoint directory's serving path.
const std = @import("std");
pub const message = "GGUF serving is coming. Use a supported checkpoint directory for this build.";
pub fn forPath(path: []const u8) ?[]const u8 {
    return if (path.len >= 5 and std.ascii.eqlIgnoreCase(path[path.len - 5 ..], ".gguf")) message else null;
}
