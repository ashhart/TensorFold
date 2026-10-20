//! Select the applicable startup phase from validated pack metadata before creating the engine or its resources.
const std = @import("std");
const pio = @import("pack_io.zig");
const cache = @import("pack_cache.zig");

pub const Route = struct {
    identity: []u8,
    dir: ?[]u8,
    root: ?[]u8,
    needs_build: bool,

    pub fn deinit(p: Route, a: std.mem.Allocator) void {
        a.free(p.identity);
        if (p.dir) |d| a.free(d);
        if (p.root) |d| a.free(d);
    }
};

pub fn plan(a: std.mem.Allocator, io: std.Io, model: []const u8, supplied: ?[]const u8, home: []const u8) !Route {
    const identity = try pio.sourceIdentity(a, io, model);
    errdefer a.free(identity);
    if (supplied) |dir| {
        const path = try std.fs.path.join(a, &.{ dir, "pack.safetensors" });
        defer a.free(path);
        try pio.checkIdentityPath(a, io, path, identity);
        return .{ .identity = identity, .dir = try a.dupe(u8, dir), .root = null, .needs_build = false };
    }
    const root = try cache.defaultRoot(a, home);
    errdefer a.free(root);
    const found = try cache.lookup(a, io, model, root, identity);
    return .{ .identity = identity, .dir = found, .root = root, .needs_build = found == null };
}
