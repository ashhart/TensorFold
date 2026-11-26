//! --confidential: serve starts only when nothing keeps request content on disk, sends it out or dumps it.
const std = @import("std");
const builtin = @import("builtin");
const cli = @import("cli.zig");
const Allocator = std.mem.Allocator;

/// TensorFold's settings and the loader's injection variables; HOME, PATH, TMPDIR and HF_* stay allowed.
const prefixes = [_][]const u8{ "TENSORFOLD_", "TF_", "FZ_", "GLM_", "MCDMA_", "DYLD_", "LD_PRELOAD", "LD_AUDIT" };

/// The first flag that keeps request content on disk or sends it out of the process; null: none.
pub fn kept(args: cli.Args) ?[]const u8 {
    const checks = [_]struct { []const u8, bool }{
        .{ "--record", args.record != null },
        .{ "--learn-conversations", args.learn_conversations }, // --prompt-cache-disk sets it
        .{ "--learn", args.learn },
        .{ "--recall", args.recall },
        .{ "--slide", args.slide or args.slide_graph != null },
        .{ "--spill-gib", (args.spill_gib orelse 0) > 0 },
        .{ "--compact-memory", args.compact_memory != null },
        .{ "--fsr-route-trace", args.fsr_trace != null },
        .{ "--cortheon", args.cortheon },
    };
    for (checks) |c| if (c[1]) return c[0];
    return null;
}

/// The refused variables `env` sets, sorted by name; their values are never read.
pub fn settings(a: Allocator, env: *const std.process.Environ.Map) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    for (env.keys()) |name| if (refused(name)) try names.append(a, name);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return names.items;
}

fn refused(name: []const u8) bool {
    for (prefixes) |p| if (std.mem.startsWith(u8, name, p)) return true;
    return false;
}

/// Why a confidential start is refused, naming flags and variables but never a value; null: it may start.
pub fn refusal(a: Allocator, args: cli.Args, env: *const std.process.Environ.Map) !?[]const u8 {
    if (!args.confidential) return null;
    if (kept(args)) |flag| return try std.fmt.allocPrint(a, "--confidential refuses {s}: it keeps or sends request content", .{flag});
    const names = try settings(a, env);
    if (names.len == 0) return null;
    const list = try std.mem.join(a, ", ", names);
    return try std.fmt.allocPrint(a, "--confidential reads no TensorFold or loader setting from the environment: unset {s}", .{list});
}

/// No core file can hold the process's memory: the soft and the hard limit both 0, so no child raises it.
pub fn noCoreDumps() !void {
    if (comptime builtin.os.tag == .windows) return error.Unsupported;
    try std.posix.setrlimit(.CORE, .{ .cur = 0, .max = 0 });
}
