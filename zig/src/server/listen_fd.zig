//! ``--listen-fd N``: serve the bound Unix socket the parent passed, the only way in: no path, port or key beside it.
const std = @import("std");
const builtin = @import("builtin");
const cli = @import("cli.zig");
const Listener = @import("listener.zig").Listener;

/// The descriptor's range, and the flags that open another way in or name a key, refused beside it.
pub fn check(u: *cli.Usage, a: std.mem.Allocator, out: cli.Args, repeated: *const cli.Repeated, tcp_flag: ?[]const u8) error{ Usage, OutOfMemory }!void {
    const fd = out.listen_fd orelse return;
    if (builtin.os.tag == .windows) return cli.fail(u, a, "argument --listen-fd: not served on Windows", .{});
    if (fd < 0 or fd > std.math.maxInt(i32)) return cli.fail(u, a, "argument --listen-fd: expected a descriptor number from 0: '{d}'", .{fd});
    const other: ?[]const u8 = if (out.unix_socket != null) "--unix-socket" else if (tcp_flag) |t| t else if (repeated.keys.items.len > 0) "--api-key" else if (out.api_key_file != null) "--api-key-file" else null;
    if (other) |o| return cli.fail(u, a, "argument --listen-fd: not allowed with {s}", .{o});
}

/// TENSORFOLD_API_KEY beside --listen-fd (`env_keys`), refused: only the process that passed the socket connects.
pub fn keyRefusal(args: cli.Args, env_keys: []const u8) ?[]const u8 {
    if (args.listen_fd == null or std.mem.trim(u8, env_keys, " \t\r\n\x0b\x0c,").len == 0) return null;
    return "--listen-fd: not allowed with TENSORFOLD_API_KEY; only the process that passed the socket can connect";
}

/// The inherited socket, or null after one line naming why the descriptor cannot be served.
pub fn open(fd: i64) ?Listener {
    return Listener.inherit(@intCast(fd)) catch |e| {
        const why = switch (e) {
            error.Unsupported => "not served on Windows",
            error.NotOpen => "not an open descriptor; the parent must leave it open across exec",
            error.NotSocket => "not a socket",
            error.NotUnix => "not a Unix-domain socket",
            error.NotBound => "not bound to an address; the parent must bind it first",
            error.NotStream => "not a stream socket",
            error.CannotListen => "cannot listen on it; pass a bound socket that is not connected",
        };
        std.debug.print("tensorfold: --listen-fd {d}: {s}\n", .{ fd, why });
        return null;
    };
}
