//! ``--router-config FILE.toml``: one ``[[models]]`` table a model, its id, aliases, model and serve flags.
const std = @import("std");
const cli = @import("../cli.zig");
const cli_config = @import("../cli_config.zig");
const toml = @import("../toml_flat.zig");
const catalog = @import("catalog.zig");
const Allocator = std.mem.Allocator;

const Error = error{ Usage, OutOfMemory };
const flag = "--router-config";

/// The longest id or alias, in bytes.
pub const name_limit = 256;

/// The serve flags the router sets on every child itself, and why a model's table cannot.
const owned = [_][2][]const u8{
    .{ "name", "a model answers to its id and aliases" },
    .{ "alias", "a model answers to its id and aliases: list them in aliases" },
    .{ "host", "each model's server listens on 127.0.0.1 at a port the router picks" },
    .{ "port", "each model's server listens on 127.0.0.1 at a port the router picks" },
    .{ "unix-socket", "each model's server listens on 127.0.0.1 at a port the router picks" },
    .{ "api-key", "each model's server takes a key of its own; give the router --api-key or --api-key-file" },
    .{ "api-key-file", "each model's server takes a key of its own; give the router --api-key or --api-key-file" },
    .{ "idle-unload", "the router stops idle models: give it --idle-unload SECONDS" },
    .{ "allowed-host", "the router checks the Host header: give it --allowed-host" },
    .{ "allowed-origin", "the router checks the Origin header: give it --allowed-origin" },
};

/// The models a router config file lists, read and checked as ``serve --config`` reads its file.
pub fn load(a: Allocator, path: []const u8, u: *cli.Usage) Error![]const catalog.Model {
    const file = try cli_config.read(a, path, flag, u);
    return parse(a, path, file.text, u);
}

/// The models in ``text``, the file at ``path``: each table's keys checked, its serve flags as a command line's.
pub fn parse(a: Allocator, path: []const u8, text: []const u8, u: *cli.Usage) Error![]const catalog.Model {
    var problem: toml.Problem = .{};
    const shaped = toml.parseShaped(a, text, .{ .table = "models" }, &problem) catch |e| switch (e) {
        error.OutOfMemory => |x| return x,
        error.Invalid => return cli.fail(u, a, "argument {s}: {s} line {d}: {s}", .{ flag, path, problem.line, problem.message }),
    };
    for (shaped.entries) |e| if (e.table == 0)
        return at(u, a, path, e.line, "top-level keys are not read: each model is a [[models]] table, and the router's flags go on its command line", .{});
    if (shaped.tables.len == 0) return cli.fail(u, a, "argument {s}: {s} lists no models: add a [[models]] table with an id and a model", .{ flag, path });
    var out: std.ArrayList(catalog.Model) = .empty;
    for (shaped.tables, 1..) |header, n| {
        const m = try table(a, path, header, @intCast(n), shaped.entries, u);
        for (out.items) |other| for ([_][]const []const u8{ &.{m.id}, m.aliases }) |names| for (names) |name| {
            if (named(other, name)) return at(u, a, path, header, "'{s}' already names model '{s}'", .{ name, other.id });
        };
        try out.append(a, m);
    }
    return out.items;
}

/// One ``[[models]]`` table whose header is on ``header``: the router's keys, then serve's for the rest.
fn table(a: Allocator, path: []const u8, header: u32, n: u32, entries: []const toml.Entry, u: *cli.Usage) Error!catalog.Model {
    var id: ?[]const u8 = null;
    var aliases: std.ArrayList([]const u8) = .empty;
    var rest: std.ArrayList(toml.Entry) = .empty;
    var model = false;
    var first: u32 = 0;
    for (entries) |e| {
        if (e.table != n) continue;
        if (first == 0) first = e.line;
        const key = try a.dupe(u8, e.key);
        std.mem.replaceScalar(u8, key, '_', '-');
        if (std.mem.eql(u8, key, "id")) {
            if (e.value != .scalar or e.value.scalar != .string) return at(u, a, path, e.line, "id takes a string", .{});
            id = try checkName(a, path, e.line, "id", e.value.scalar.string, u);
            continue;
        }
        if (std.mem.eql(u8, key, "aliases")) {
            const list: []const []const u8 = switch (e.value) {
                .array => |items| items,
                .scalar => |s| if (s == .string) &.{s.string} else return at(u, a, path, e.line, "aliases takes strings", .{}),
            };
            for (list) |alias| try aliases.append(a, try checkName(a, path, e.line, "alias", alias, u));
            continue;
        }
        for (owned) |o| if (std.mem.eql(u8, key, o[0])) return at(u, a, path, e.line, "{s} is the router's: {s}", .{ o[0], o[1] });
        model = model or std.mem.eql(u8, key, "model");
        try rest.append(a, e);
    }
    const name = id orelse return at(u, a, path, header, "a [[models]] table needs an id", .{});
    if (!model) return at(u, a, path, header, "'{s}' has no model: add model = a folder or a Hugging Face repo id", .{name});
    for (aliases.items, 0..) |alias, i| {
        const twice = std.mem.eql(u8, alias, name) or for (aliases.items[0..i]) |earlier| {
            if (std.mem.eql(u8, earlier, alias)) break true;
        } else false;
        if (twice) return at(u, a, path, header, "'{s}' is named twice in model '{s}'", .{ alias, name });
    }
    const found: cli_config.Found = .{ .path = path, .at = 0, .width = 0, .model = false, .flag = flag };
    const merged = try cli_config.mergeEntries(a, &.{}, found, rest.items, u);
    return .{ .id = name, .aliases = aliases.items, .model = merged.argv[0], .flags = merged.argv[1..], .line = first };
}

fn named(m: catalog.Model, name: []const u8) bool {
    if (std.mem.eql(u8, m.id, name)) return true;
    for (m.aliases) |alias| if (std.mem.eql(u8, alias, name)) return true;
    return false;
}

/// A name a request's ``model`` field can carry: 1 to 256 bytes, no spaces or control characters, no leading dash.
fn checkName(a: Allocator, path: []const u8, line: u32, what: []const u8, text: []const u8, u: *cli.Usage) Error![]const u8 {
    const plain = for (text) |c| {
        if (c <= ' ' or c == 0x7f) break false;
    } else true;
    if (text.len == 0 or text.len > name_limit or !plain or text[0] == '-')
        return at(u, a, path, line, "{s} '{s}': expected 1 to {d} bytes without spaces or control characters, not starting with -", .{ what, text, name_limit });
    return text;
}

fn at(u: *cli.Usage, a: Allocator, path: []const u8, line: u32, comptime fmt: []const u8, args: anytype) Error {
    return cli.fail(u, a, "argument {s}: {s} line {d}: " ++ fmt, .{ flag, path, line } ++ args);
}
