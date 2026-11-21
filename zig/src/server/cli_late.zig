//! Serve flag checks of one flag's own values that run once every flag is read; a config file runs them on its own.
const std = @import("std");
const cli = @import("cli.zig");

/// The idle unload's range, each --name-priority entry's form and the host, origin and CORS list sizes.
pub fn check(a: std.mem.Allocator, out: cli.Args, repeated: *const cli.Repeated, u: *cli.Usage) error{ Usage, OutOfMemory }!void {
    if (out.idle_unload < 0 or out.idle_unload > std.math.maxInt(i64) / std.time.ns_per_s) return cli.fail(u, a, "argument --idle-unload: expected a seconds count from 0 (off) under {d}: '{d}'", .{ std.math.maxInt(i64) / std.time.ns_per_s, out.idle_unload });
    for (repeated.priorities.items) |entry| {
        const eq = std.mem.indexOfScalar(u8, entry, '=') orelse
            return cli.fail(u, a, "argument --name-priority: takes ID=background, not '{s}'", .{entry});
        const id = std.mem.trim(u8, entry[0..eq], " \t");
        const word = std.mem.trim(u8, entry[eq + 1 ..], " \t");
        if (id.len == 0 or !std.ascii.eqlIgnoreCase(word, "background"))
            return cli.fail(u, a, "argument --name-priority: takes ID=background, not '{s}'", .{entry});
    }
    if (repeated.hosts.items.len > 64 or repeated.origins.items.len > 64) return cli.fail(u, a, "at most 64 --allowed-host and 64 --allowed-origin entries are allowed", .{});
    if (repeated.cors.items.len > 64) return cli.fail(u, a, "at most 64 --cors-origin entries are allowed", .{});
}
