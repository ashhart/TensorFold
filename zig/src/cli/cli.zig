//! The tensorfold checkpoint subcommands: `models` and `info` (pull adds downloading in its own commit).
const std = @import("std");
const models = @import("models.zig");
const info = @import("info.zig");
const pull = @import("pull.zig");

const usage =
    \\usage: tensorfold models
    \\       tensorfold info MODEL
    \\       tensorfold pull REPO[@REVISION]
    \\
;

/// True when argv starts with a checkpoint subcommand.
pub fn wants(args: []const []const u8) bool {
    if (args.len == 0) return false;
    const cmd = args[0];
    return std.mem.eql(u8, cmd, "models") or std.mem.eql(u8, cmd, "info") or std.mem.eql(u8, cmd, "pull");
}

/// Runs a checkpoint subcommand; `argv` is argv[1..] as in cluster.cli.main.
pub fn main(init: std.process.Init, argv: []const [:0]const u8) !u8 {
    const a = init.arena.allocator();
    var out_buf: [64 << 10]u8 = undefined;
    var err_buf: [4 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &out_buf);
    var err = std.Io.File.stderr().writer(init.io, &err_buf);
    defer out.interface.flush() catch {};
    defer err.interface.flush() catch {};
    const args = try a.alloc([]const u8, argv.len);
    for (args, argv) |*x, y| x.* = y;
    const env = init.environ_map;
    if (std.mem.eql(u8, args[0], "models")) {
        if (args.len != 1) {
            try err.interface.writeAll(usage);
            return 2;
        }
        return models.run(a, init.io, &out.interface, env, null);
    }
    if (std.mem.eql(u8, args[0], "pull")) {
        if (args.len != 2) {
            try err.interface.writeAll(usage);
            return 2;
        }
        return pull.run(a, init.io, &out.interface, &err.interface, env, null, args[1]);
    }
    if (args.len != 2) {
        try err.interface.writeAll(usage);
        return 2;
    }
    return info.run(a, init.io, &out.interface, env, null, args[1]);
}

test "wants only the checkpoint subcommands" {
    try std.testing.expect(wants(&.{"models"}));
    try std.testing.expect(wants(&.{ "info", "Org/Flash" }));
    try std.testing.expect(!wants(&.{"run"}) and !wants(&.{}) and !wants(&.{"serve"}));
}
