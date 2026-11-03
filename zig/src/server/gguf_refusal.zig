//! A GGUF model argument must resolve to an existing regular file before admission.
const std = @import("std");
pub const message = "GGUF file is missing or is not a regular file; name an existing .gguf file.";
pub fn forPath(path: []const u8) ?[]const u8 {
    return if (path.len >= 5 and std.ascii.eqlIgnoreCase(path[path.len - 5 ..], ".gguf")) message else null;
}
