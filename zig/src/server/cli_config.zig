//! ``serve --config FILE.toml``: the file's keys become serve flags ahead of the command line's, which then win.
const std = @import("std");
const cli = @import("cli.zig");
const toml = @import("toml_flat.zig");
const platform = @import("sys");
const file_meta = @import("file_meta.zig").meta;
const Allocator = std.mem.Allocator;

const Error = error{ Usage, OutOfMemory };

/// The largest file read, in bytes.
pub const limit = 1 << 20;

pub const Expanded = struct {
    /// What ``cli.parse`` reads.
    argv: []const []const u8,
    /// The file, when one was read.
    path: ?[]const u8 = null,
    /// The settings it held.
    options: usize = 0,
};

/// ``--config`` on the command line: its file, its place (`width` tokens from `at`) and whether a model is named.
pub const Found = struct { path: []const u8, at: usize, width: usize, model: bool };

/// ``serve``'s arguments with ``--config FILE.toml`` replaced by the file's flags, ahead of the command line's.
pub fn expand(a: Allocator, io: std.Io, argv: []const []const u8, u: *cli.Usage) Error!Expanded {
    const found = try locate(a, argv, u) orelse return .{ .argv = argv };
    if (try unsafe(a, found.path)) |why| return cli.fail(u, a, "argument --config: {s} {s}", .{ found.path, why });
    const text = std.Io.Dir.cwd().readFileAlloc(io, found.path, a, .limited(limit + 1)) catch |e| switch (e) {
        error.OutOfMemory => |x| return x,
        error.StreamTooLong => return cli.fail(u, a, "argument --config: {s} is over 1 MiB", .{found.path}),
        else => return cli.fail(u, a, "argument --config: cannot read {s}: {s}", .{ found.path, @errorName(e) }),
    };
    if (text.len > limit) return cli.fail(u, a, "argument --config: {s} is over 1 MiB", .{found.path});
    return merge(a, argv, found, text, u);
}

/// Where ``--config`` is, walking the arguments as ``cli.parse`` consumes them; null without one.
pub fn locate(a: Allocator, argv: []const []const u8, u: *cli.Usage) Error!?Found {
    var found: ?Found = null;
    var model = false;
    var valueless: ?[]const u8 = null; // the first flag left without its value: removing --config must not repair it
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (!std.mem.startsWith(u8, token, "--")) {
            model = true;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        const flag = cli.find(name) orelse continue;
        if (flag.kind == .store_true) continue;
        const at = i;
        const value = if (eq) |e| token[e + 1 ..] else value: {
            const disk = std.mem.eql(u8, name, "--prompt-cache-disk"); // parse refuses a flag as this one's directory
            if (i + 1 >= argv.len or (disk and std.mem.startsWith(u8, argv[i + 1], "--"))) {
                if (valueless == null) valueless = name;
                continue;
            }
            i += 1;
            break :value argv[i];
        };
        if (!std.mem.eql(u8, name, "--config")) continue;
        if (found != null) return cli.fail(u, a, "argument --config: given twice; a command reads one config file", .{});
        if (!std.ascii.endsWithIgnoreCase(value, ".toml"))
            return cli.fail(u, a, "argument --config: expected a .toml file: '{s}'", .{value});
        found = .{ .path = value, .at = at, .width = i - at + 1, .model = false };
    }
    if (found == null) return null;
    if (valueless) |name| return cli.fail(u, a, "argument {s}: expected one argument", .{name});
    found.?.model = model;
    return found;
}

/// The file's text checked against the flag table and merged with the command line; it reads no file.
pub fn merge(a: Allocator, argv: []const []const u8, found: Found, text: []const u8, u: *cli.Usage) Error!Expanded {
    var problem: toml.Problem = .{};
    const entries = toml.parse(a, text, &problem) catch |e| switch (e) {
        error.OutOfMemory => |x| return x,
        error.Invalid => return cli.fail(u, a, "argument --config: {s} line {d}: {s}", .{ found.path, problem.line, problem.message }),
    };
    var out: std.ArrayList([]const u8) = .empty;
    var seen: std.ArrayList([]const u8) = .empty;
    var model: ?[]const u8 = null;
    const base = std.fs.path.dirname(found.path); // the file's folder: where its relative paths start
    for (entries) |entry| {
        const at: Where = .{ .a = a, .u = u, .path = found.path, .line = entry.line };
        const key = try a.dupe(u8, entry.key);
        std.mem.replaceScalar(u8, key, '_', '-');
        if (std.mem.eql(u8, key, "model")) {
            if (entry.value != .scalar or entry.value.scalar != .string) return at.fail("model takes a string", .{});
            model = try anchored(a, base, entry.value.scalar.string);
            if (std.mem.startsWith(u8, model.?, "--")) return at.fail("model cannot start with --, which parse reads as a flag: write a folder as ./{s}", .{model.?});
            continue;
        }
        if (std.mem.eql(u8, key, "config")) return at.fail("a config file cannot name another config file", .{});
        const flag = cli.find(try std.fmt.allocPrint(a, "--{s}", .{key})) orelse return at.fail("unknown key '{s}'", .{entry.key});
        if (!flag.native) return at.fail("{s} is not served by this build", .{flag.name});
        for (seen.items) |name| if (std.mem.eql(u8, name, flag.name)) return at.fail("{s} is set twice ('{s}' and its other spelling)", .{ flag.name, entry.key });
        try seen.append(a, flag.name);
        switch (flag.kind) {
            .store_true => {
                if (entry.value != .scalar or entry.value.scalar != .boolean) return at.fail("{s} is a switch: use true or false", .{flag.name});
                if (try positive(a, flag)) |name| return at.fail("{s} has the switch {s}: write {s} = {s}", .{ flag.name, name, name[2..], if (entry.value.scalar.boolean) "false" else "true" });
                if (entry.value.scalar.boolean) try out.append(a, flag.name) else if (try negative(a, flag)) |other| try out.append(a, other);
            },
            .value => switch (entry.value) {
                .array => return at.fail("{s} takes one value, not a list", .{flag.name}),
                .scalar => |s| try out.append(a, try valued(a, flag, try placed(a, base, flag, s)) orelse return at.fail("{s} takes a value, not true or false", .{flag.name})),
            },
            .append => {
                switch (entry.value) {
                    .array => |list| for (list) |item| try out.append(a, (try valued(a, flag, .{ .string = item })).?),
                    .scalar => |s| try out.append(a, try valued(a, flag, s) orelse return at.fail("{s} takes values, not true or false", .{flag.name})),
                }
            },
        }
    }
    var merged: std.ArrayList([]const u8) = .empty;
    if (!found.model) if (model) |m| try merged.append(a, m);
    try merged.appendSlice(a, out.items);
    try merged.appendSlice(a, argv[0..found.at]);
    try merged.appendSlice(a, argv[found.at + found.width ..]);
    return .{ .argv = merged.items, .path = found.path, .options = entries.len };
}

/// The line ``serve`` adds under a usage error when a file was read, so a flag nobody typed has a source.
pub fn note(a: Allocator, e: Expanded) error{OutOfMemory}!?[]const u8 {
    const path = e.path orelse return null;
    return try std.fmt.allocPrint(a, "--config {s} supplied {d} option{s}, read before the command line's", .{ path, e.options, if (e.options == 1) "" else "s" });
}

/// A file's line, for its refusals.
const Where = struct {
    a: Allocator,
    u: *cli.Usage,
    path: []const u8,
    line: u32,

    fn fail(w: Where, comptime fmt: []const u8, args: anytype) Error {
        return cli.fail(w.u, w.a, "argument --config: {s} line {d}: " ++ fmt, .{ w.path, w.line } ++ args);
    }
};

/// ``--name=text`` for a string or a number, so a value that starts with ``-`` stays a value; null for a boolean.
fn valued(a: Allocator, flag: cli.Flag, s: toml.Scalar) error{OutOfMemory}!?[]const u8 {
    const text = switch (s) {
        .string, .integer, .float => |t| t,
        .boolean => return null,
    };
    return try std.fmt.allocPrint(a, "{s}={s}", .{ flag.name, text });
}

/// The served ``--no-x`` that undoes the switch ``--x``; null without one.
fn negative(a: Allocator, flag: cli.Flag) error{OutOfMemory}!?[]const u8 {
    return served(try std.fmt.allocPrint(a, "--no-{s}", .{flag.name[2..]}));
}

/// The served ``--x`` that ``--no-x`` undoes, which is then the one key for the pair; null for any other flag.
fn positive(a: Allocator, flag: cli.Flag) error{OutOfMemory}!?[]const u8 {
    if (!std.mem.startsWith(u8, flag.name, "--no-")) return null;
    return served(try std.fmt.allocPrint(a, "--{s}", .{flag.name["--no-".len..]}));
}

fn served(name: []const u8) ?[]const u8 {
    const other = cli.find(name) orelse return null;
    return if (other.kind == .store_true and other.native) other.name else null;
}

/// A path flag's relative string under the file's folder; ``--drafter`` (a folder or a repo id) only from ./ or ../.
fn placed(a: Allocator, base: ?[]const u8, flag: cli.Flag, s: toml.Scalar) error{OutOfMemory}!toml.Scalar {
    const text = switch (s) {
        .string, .integer, .float => |t| t, // a folder can be named 123
        .boolean => return s,
    };
    if (std.mem.eql(u8, flag.name, "--drafter")) return .{ .string = try anchored(a, base, text) };
    if (!flag.path or text.len == 0 or std.fs.path.isAbsolute(text) or text[0] == '~') return s;
    return .{ .string = if (base) |b| try std.fs.path.join(a, &.{ b, text }) else text };
}

/// A model or drafter that starts with ./ or ../ is a folder beside the file; any other text is left as written.
fn anchored(a: Allocator, base: ?[]const u8, text: []const u8) error{OutOfMemory}![]const u8 {
    const b = base orelse return text;
    if (!std.mem.startsWith(u8, text, "./") and !std.mem.startsWith(u8, text, "../")) return text;
    const joined = try std.fs.path.join(a, &.{ b, if (text[1] == '/') text[2..] else text });
    return if (std.mem.startsWith(u8, joined, "--")) try std.fs.path.join(a, &.{ ".", joined }) else joined; // not a flag
}

/// Why the file is not read, or null: it can hold keys and it chooses what is served, so only its owner may write it.
fn unsafe(a: Allocator, path: []const u8) error{OutOfMemory}!?[]const u8 {
    const m = file_meta(-1, try a.dupeSentinel(u8, path, 0)) orelse return null; // the read names a file it cannot open
    if (m.mode & platform.c.S.IFMT != platform.c.S.IFREG) return "is not a regular file";
    if (m.mode & 0o022 == 0) return null;
    return try std.fmt.allocPrint(a, "can be written by other users (mode 0{o}): it can hold keys and it chooses what is served; use chmod go-w", .{m.mode & 0o7777});
}
