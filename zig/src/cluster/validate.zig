//! `tensorfold config validate FILE`: parse a cluster file, print every problem, no planning or probes.
const std = @import("std");
const config = @import("config.zig");

const Allocator = std.mem.Allocator;
const W = std.Io.Writer;

pub const Env = struct {
    io: std.Io,
    a: Allocator,
    out: *W,
    err: *W,
};

pub const usage =
    \\usage: tensorfold config validate FILE
    \\
;

/// True when argv starts with the config command.
pub fn wants(args: []const []const u8) bool {
    return args.len > 0 and std.mem.eql(u8, args[0], "config");
}

/// Exit: 0 valid, 1 unreadable or invalid, 2 usage.
pub fn run(env: Env, args: []const []const u8) !u8 {
    if (args.len != 3 or !std.mem.eql(u8, args[0], "config") or !std.mem.eql(u8, args[1], "validate")) return bad(env);
    const path = args[2];
    if (std.mem.startsWith(u8, path, "-")) return bad(env);
    const json = std.Io.Dir.cwd().readFileAlloc(env.io, path, env.a, .limited(16 << 20)) catch |err| {
        try env.err.print("{s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    var probs: config.Problems = .{ .a = env.a };
    const c = config.parse(env.a, json, &probs) catch {
        for (probs.list.items) |p| try env.err.print("{s}\n", .{p});
        return 1;
    };
    try env.out.print("{s}: {d} nodes, {d} models\n", .{ path, c.nodes.len, c.models.len });
    return 0;
}

fn bad(env: Env) !u8 {
    try env.err.writeAll(usage);
    return 2;
}

/// The entry both binaries share: argv after the program name, stdout and stderr.
pub fn main(init: std.process.Init, argv: []const [:0]const u8) !u8 {
    const a = init.arena.allocator();
    var out_buf: [64 << 10]u8 = undefined;
    var err_buf: [4 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    var err = std.Io.File.stderr().writerStreaming(init.io, &err_buf);
    const args = try a.alloc([]const u8, argv.len);
    for (args, argv) |*x, y| x.* = y;
    const code = try run(.{ .io = init.io, .a = a, .out = &out.interface, .err = &err.interface }, args);
    try out.interface.flush();
    try err.interface.flush();
    return code;
}

test {
    _ = @import("validate_test.zig");
}
