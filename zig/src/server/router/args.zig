//! ``tensorfold serve --router`` flags: the router's own, the serve flags it shares, and the rest refused by name.
const std = @import("std");
const cli = @import("../cli.zig");
const Allocator = std.mem.Allocator;

pub const usage_line = "usage: tensorfold serve --router [-h] [--router-config FILE.toml] [--max-loaded N] [--idle-unload SECONDS]" ++
    " [--host HOST] [--port PORT] [--api-key API_KEY] [--api-key-file API_KEY_FILE] [--allowed-host NAME] [--allowed-origin ORIGIN]\n";

/// The most children ``--max-loaded`` allows at once.
pub const max_loaded_limit = 64;

pub const Settings = struct {
    config: ?[]const u8 = null,
    max_loaded: u32 = 1,
    /// Seconds a child may sit idle before the router stops it (0: never), as ``--idle-unload`` reads them.
    idle_s: i64 = 0,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    api_keys: []const []const u8 = &.{},
    api_key_file: ?[]const u8 = null,
    allowed_hosts: []const []const u8 = &.{},
    allowed_origins: []const []const u8 = &.{},
};

/// The serve flags the router reads as ``tensorfold serve`` does, with serve's own checks and messages.
const shared = [_][]const u8{ "--host", "--port", "--api-key", "--api-key-file", "--idle-unload", "--allowed-host", "--allowed-origin" };

/// Whether a serve command line asks for the router.
pub fn wanted(argv: []const []const u8) bool {
    for (argv) |arg| if (std.mem.eql(u8, arg, "--router")) return true;
    return false;
}

/// Whether the command line asks for help, which needs no models.
pub fn help(argv: []const []const u8) bool {
    for (argv) |arg| if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return true;
    return false;
}

/// The router's settings from its command line; a usage error names the flag, as ``cli.parse`` does.
pub fn parse(a: Allocator, argv: []const []const u8, u: *cli.Usage) error{ Usage, OutOfMemory }!Settings {
    var out: Settings = .{};
    var scratch: cli.Args = .{};
    var repeated: cli.Repeated = .{};
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (!std.mem.startsWith(u8, token, "--"))
            return cli.fail(u, a, "unrecognized arguments: {s}: --router serves the models of --router-config or the cache and takes no MODEL", .{token});
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        if (std.mem.eql(u8, name, "--router")) {
            if (eq != null) return cli.fail(u, a, "argument --router: ignored explicit argument '{s}'", .{token[eq.? + 1 ..]});
            continue;
        }
        const own = std.mem.eql(u8, name, "--router-config") or std.mem.eql(u8, name, "--max-loaded");
        const flag = cli.find(name);
        const is_shared = for (shared) |s| {
            if (std.mem.eql(u8, s, name)) break true;
        } else false;
        if (!own and !is_shared) {
            if (std.mem.eql(u8, name, "--config")) return cli.fail(u, a, "argument --config: with --router, models and their serve flags are listed in --router-config FILE.toml", .{});
            if (std.mem.eql(u8, name, "--name") or std.mem.eql(u8, name, "--alias")) return cli.fail(u, a, "argument {s}: with --router, each model answers to its id and aliases in --router-config FILE.toml", .{name});
            if (std.mem.eql(u8, name, "--unix-socket")) return cli.fail(u, a, "argument --unix-socket: the router listens on --host and --port only", .{});
            if (flag != null) return cli.fail(u, a, "argument {s}: a model's serve flag; with --router, set it on the model in --router-config FILE.toml", .{name});
            return cli.fail(u, a, "unrecognized arguments: {s}", .{token});
        }
        const value = if (eq) |e| token[e + 1 ..] else blk: {
            i += 1;
            if (i >= argv.len) return cli.fail(u, a, "argument {s}: expected one argument", .{name});
            break :blk argv[i];
        };
        if (std.mem.eql(u8, name, "--router-config")) {
            if (!std.ascii.endsWithIgnoreCase(value, ".toml")) return cli.fail(u, a, "argument --router-config: expected a .toml file: '{s}'", .{value});
            if (out.config != null) return cli.fail(u, a, "argument --router-config: given twice; the router reads one config file", .{});
            out.config = value;
        } else if (std.mem.eql(u8, name, "--max-loaded")) {
            const n = try cli.int(u, a, name, value);
            if (n < 1 or n > max_loaded_limit) return cli.fail(u, a, "argument --max-loaded: expected 1 to {d} models: '{s}'", .{ max_loaded_limit, value });
            out.max_loaded = @intCast(n);
        } else try cli.one(a, &scratch, &repeated, flag.?, value, u);
    }
    try @import("../cli_late.zig").check(a, scratch, &repeated, u);
    out.idle_s = scratch.idle_unload;
    out.host = scratch.host;
    out.port = scratch.port;
    out.api_keys = repeated.keys.items;
    out.api_key_file = scratch.api_key_file;
    out.allowed_hosts = repeated.hosts.items;
    out.allowed_origins = repeated.origins.items;
    return out;
}
