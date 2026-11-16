//! The serve parser's scalar helpers (usage errors carry argparse's shape).
const std = @import("std");
const Allocator = std.mem.Allocator;
const cli = @import("cli.zig");
const Usage = cli.Usage;
const Args = cli.Args;
const fail = cli.fail;

/// --spill-gib writes to the conversation store: beside --learn's shared prefixes only with --learn-conversations.
pub fn spillWith(u: *Usage, a: Allocator, out: Args) error{ Usage, OutOfMemory }!void {
    if (spills(out) and out.learn and !out.learn_conversations) return fail(u, a, "--spill-gib writes evicted states to the conversation store, which --learn's shared prefixes do not use: use --prompt-cache-disk DIR without --learn-dir, add --learn-conversations, or serve without --learn", .{});
}

pub fn spills(args: Args) bool {
    return (args.spill_gib orelse 0) > 0;
}

pub fn int(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!i64 {
    return std.fmt.parseInt(i64, std.mem.trim(u8, v, " "), 10) catch fail(u, a, "argument {s}: invalid int value: '{s}'", .{ name, v });
}

pub fn float(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    return @import("fields.zig").pyFloat(v) orelse fail(u, a, "argument {s}: invalid float value: '{s}'", .{ name, v });
}

/// A finite GiB count from 0 (off) to below 2^34, whose bytes fit a u64.
pub fn gib(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    const g = try float(u, a, name, v);
    if (!std.math.isFinite(g) or g < 0 or g >= 1 << 34) return fail(u, a, "argument {s}: expected GiB from 0 (off) to below 2^34: '{s}'", .{ name, v });
    return g;
}
