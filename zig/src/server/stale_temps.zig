//! A crash between a note's or the graph's write and its rename leaves a temporary: cleared at server open.
const std = @import("std");
const platform = @import("sys");
const Config = @import("server.zig").Config;

/// Each temporary left in the stored note's folder and beside the fact graph, removed.
pub fn discard(io: std.Io, config: Config) void {
    if (config.compact_memory) |dir| platform.atomic_file.discardStale(io, dir, null);
    if (config.slide_graph) |path| platform.atomic_file.discardStale(io, std.fs.path.dirname(path) orelse ".", std.fs.path.basename(path));
}
