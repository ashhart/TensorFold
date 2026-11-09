//! The --recall, --recall-dir, --recall-gib and --recall-read-only flags.
const std = @import("std");
const Allocator = std.mem.Allocator;
const cli = @import("cli.zig");

/// --recall and the flags that imply it; false for any other flag.
pub fn flag(a: Allocator, out: *cli.Args, name: []const u8, v: []const u8, u: *cli.Usage) error{ Usage, OutOfMemory }!bool {
    if (std.mem.eql(u8, name, "--recall-dir")) {
        out.recall_dir = v;
    } else if (std.mem.eql(u8, name, "--recall-gib")) {
        out.recall_gib = try cli.gib(u, a, name, v);
        const least = "argument --recall-gib: at least 1/128 GiB (two 4 MiB chunks): '{s}'";
        if (out.recall_gib < 1.0 / 128.0) return cli.fail(u, a, least, .{v});
    } else if (std.mem.eql(u8, name, "--recall-read-only")) {
        out.recall_read_only = true;
    } else if (!std.mem.eql(u8, name, "--recall")) return false;
    out.recall = true;
    return true;
}
