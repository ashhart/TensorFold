//! Accepted verify paths retained by the CUDA lane backend.
const std = @import("std");
const be = @import("lanes").backend;
pub fn keep(self: anytype, windows: []const be.Window, paths: []const []const u32) !void {
    const t0 = std.Io.Clock.awake.now(self.io);
    defer @import("cuda_draft.zig").timing("keep", self.io, t0);
    for (windows, paths) |w, path| {
        const cache = try self.cacheOf(w.stream);
        if (cache.last == null) continue; // already kept: nothing to roll back
        try cache.keep(self.e, path);
    }
}
