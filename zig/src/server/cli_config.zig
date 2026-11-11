//! ``serve --config FILE.toml``: the file's keys become serve flags ahead of the command line's, which then win.
const std = @import("std");
const cli = @import("cli.zig");
const toml = @import("toml_flat.zig");
const auth = @import("auth.zig");
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
    /// The line that sets ``api-key``, or 0: such a file must be unreadable by other users.
    secret: u32 = 0,
};

/// ``--config`` on the command line: its file, its place (`width` tokens from `at`) and whether a model is named.
pub const Found = struct { path: []const u8, at: usize, width: usize, model: bool };

/// ``serve``'s arguments with ``--config FILE.toml`` replaced by the file's flags, ahead of the command line's.
pub fn expand(a: Allocator, argv: []const []const u8, u: *cli.Usage) Error!Expanded {
    const found = try locate(a, argv, u) orelse return .{ .argv = argv };
    const path = found.path;
    // opened once: the handle that is checked is the one that is read, and a link is not followed
    const fd = auth.openKeys(try a.dupeSentinel(u8, path, 0)) catch |e|
        return cli.fail(u, a, "argument --config: cannot open {s} ({s}); name an existing file, not a link", .{ path, @errorName(e) });
    defer _ = platform.c.close(fd);
    const st = file_meta(fd, null) orelse return cli.fail(u, a, "argument --config: cannot read the mode of {s}", .{path});
    if (st.mode & platform.c.S.IFMT != platform.c.S.IFREG) return cli.fail(u, a, "argument --config: {s} is not a regular file", .{path});
    if (st.mode & 0o022 != 0)
        return cli.fail(u, a, "argument --config: {s} can be written by other users (mode 0{o}): it chooses what is served; use chmod go-w", .{ path, st.mode & 0o7777 });
    const text = try a.alloc(u8, limit + 1);
    var got: usize = 0;
    while (got < text.len) {
        const n = auth.readKeys(fd, text[got..]) catch |e| return cli.fail(u, a, "argument --config: cannot read {s} ({s})", .{ path, @errorName(e) });
        if (n == 0) break;
        got += n;
    }
    if (got > limit) return cli.fail(u, a, "argument --config: {s} is over 1 MiB", .{path});
    const out = try merge(a, argv, found, text[0..got], u);
    if (out.secret != 0 and st.mode & 0o044 != 0)
        return cli.fail(u, a, "argument --config: {s} line {d}: api-key is set in a file other users can read (mode 0{o}); use chmod 600, or api-key-file", .{ path, out.secret, st.mode & 0o7777 });
    return out;
}

/// One argument of ``serve`` as ``cli.parse`` consumes it: a positional (no name), or a flag with its value.
const Step = struct { at: usize, width: usize = 1, name: ?[]const u8 = null, value: ?[]const u8 = null, valueless: bool = false };

/// ``serve``'s arguments, one flag or positional a step; an unknown flag takes no value (parse refuses it later).
const Walk = struct {
    argv: []const []const u8,
    i: usize = 0,

    fn next(w: *Walk) ?Step {
        if (w.i >= w.argv.len) return null;
        const at = w.i;
        const token = w.argv[at];
        w.i += 1;
        if (!std.mem.startsWith(u8, token, "--")) return .{ .at = at };
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        const flag = cli.find(name) orelse return .{ .at = at, .name = name };
        if (flag.kind == .store_true) return .{ .at = at, .name = name };
        if (eq) |e| return .{ .at = at, .name = name, .value = token[e + 1 ..] };
        const disk = std.mem.eql(u8, name, "--prompt-cache-disk"); // parse refuses a flag as this one's directory
        if (w.i >= w.argv.len or (disk and std.mem.startsWith(u8, w.argv[w.i], "--"))) return .{ .at = at, .name = name, .valueless = true };
        w.i += 1;
        return .{ .at = at, .width = 2, .name = name, .value = w.argv[at + 1] };
    }
};

/// Where ``--config`` is, walking the arguments as ``cli.parse`` consumes them; null without one.
pub fn locate(a: Allocator, argv: []const []const u8, u: *cli.Usage) Error!?Found {
    var found: ?Found = null;
    var model = false;
    var valueless: ?[]const u8 = null; // the first flag left without its value: removing --config must not repair it
    var walk: Walk = .{ .argv = argv };
    while (walk.next()) |step| {
        const name = step.name orelse {
            model = true;
            continue;
        };
        if (step.valueless and valueless == null) valueless = name;
        const value = step.value orelse continue;
        if (!std.mem.eql(u8, name, "--config")) continue;
        if (found != null) return cli.fail(u, a, "argument --config: given twice; a command reads one config file", .{});
        if (!std.ascii.endsWithIgnoreCase(value, ".toml"))
            return cli.fail(u, a, "argument --config: expected a .toml file: '{s}'", .{value});
        found = .{ .path = value, .at = step.at, .width = step.width, .model = false };
    }
    if (found == null) return null;
    if (valueless) |name| return cli.fail(u, a, "argument {s}: expected one argument", .{name});
    found.?.model = model;
    return found;
}

/// The command line names `flag`, so a repeatable flag's list in the file gives way to it.
fn given(argv: []const []const u8, flag: cli.Flag) bool {
    var walk: Walk = .{ .argv = argv };
    while (walk.next()) |step| if (step.name) |name| if (std.mem.eql(u8, name, flag.name)) return true;
    return false;
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
    var secret: u32 = 0;
    const base = std.fs.path.dirname(found.path); // the file's folder: where its relative paths start
    for (entries) |entry| {
        var scratch: cli.Args = .{}; // one entry's values, read as parse reads them, so its checks see the whole list
        var repeated: cli.Repeated = .{};
        const at: Where = .{ .a = a, .u = u, .path = found.path, .line = entry.line, .scratch = &scratch, .repeated = &repeated };
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
        if (std.mem.eql(u8, flag.name, "--api-key")) secret = entry.line;
        const start = out.items.len;
        switch (flag.kind) {
            .store_true => {
                if (entry.value != .scalar or entry.value.scalar != .boolean) return at.fail("{s} is a switch: use true or false", .{flag.name});
                if (try positive(a, flag)) |name| return at.fail("{s} has the switch {s}: write {s} = {s}", .{ flag.name, name, name[2..], if (entry.value.scalar.boolean) "false" else "true" });
                const on = if (entry.value.scalar.boolean) flag else try negative(a, flag) orelse continue;
                try at.check(on, null);
                try out.append(a, on.name);
            },
            .value => switch (entry.value) {
                .array => return at.fail("{s} takes one value, not a list", .{flag.name}),
                .scalar => |s| try out.append(a, try at.valued(flag, base, s) orelse return at.fail("{s} takes a value, not true or false", .{flag.name})),
            },
            .append => switch (entry.value) {
                .array => |list| for (list) |item| try out.append(a, (try at.valued(flag, base, .{ .string = item })).?),
                .scalar => |s| try out.append(a, try at.valued(flag, base, s) orelse return at.fail("{s} takes values, not true or false", .{flag.name})),
            },
        }
        // a repeatable flag the command line names replaces the file's list, so an old key or host does not stay
        if (flag.kind == .append and given(argv, flag)) out.shrinkRetainingCapacity(start);
    }
    var merged: std.ArrayList([]const u8) = .empty;
    if (!found.model) if (model) |m| try merged.append(a, m);
    try merged.appendSlice(a, out.items);
    try merged.appendSlice(a, argv[0..found.at]);
    try merged.appendSlice(a, argv[found.at + found.width ..]);
    return .{ .argv = merged.items, .path = found.path, .options = entries.len, .secret = secret };
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
    scratch: *cli.Args,
    repeated: *cli.Repeated,

    fn fail(w: Where, comptime fmt: []const u8, args: anytype) Error {
        return cli.fail(w.u, w.a, "argument --config: {s} line {d}: " ++ fmt, .{ w.path, w.line } ++ args);
    }

    /// The flag's own check of this value, as on the command line, so its refusal names the file's line.
    fn check(w: Where, flag: cli.Flag, value: ?[]const u8) Error!void {
        const checked = if (cli.one(w.a, w.scratch, w.repeated, flag, value, w.u)) |_| @import("cli_late.zig").check(w.a, w.scratch.*, w.repeated, w.u) else |e| e;
        checked catch |e| switch (e) {
            error.OutOfMemory => |x| return x,
            error.Usage => return w.fail("{s}", .{w.u.message}),
        };
    }

    /// ``--name=text`` for a string or a number, placed and checked; null for a boolean.
    fn valued(w: Where, flag: cli.Flag, base: ?[]const u8, s: toml.Scalar) Error!?[]const u8 {
        var text = switch (s) {
            .string, .integer, .float => |t| t, // a folder can be named 123
            .boolean => return null,
        };
        if (std.mem.eql(u8, flag.name, "--drafter")) text = try anchored(w.a, base, text) else if (flag.path and text.len > 0) {
            if (text[0] == '~') return w.fail("{s}: no shell expands ~ in a config file; write the whole path", .{flag.name});
            if (base) |b| if (!std.fs.path.isAbsolute(text)) {
                text = try std.fs.path.join(w.a, &.{ b, text });
            };
        }
        try w.check(flag, text);
        return try std.fmt.allocPrint(w.a, "{s}={s}", .{ flag.name, text }); // one token: a value that starts with - stays one
    }
};

/// The served ``--no-x`` that undoes the switch ``--x``; null without one.
fn negative(a: Allocator, flag: cli.Flag) error{OutOfMemory}!?cli.Flag {
    return served(try std.fmt.allocPrint(a, "--no-{s}", .{flag.name[2..]}));
}

/// The served ``--x`` that ``--no-x`` undoes, which is then the one key for the pair; null for any other flag.
fn positive(a: Allocator, flag: cli.Flag) error{OutOfMemory}!?[]const u8 {
    if (!std.mem.startsWith(u8, flag.name, "--no-")) return null;
    return if (served(try std.fmt.allocPrint(a, "--{s}", .{flag.name["--no-".len..]}))) |f| f.name else null;
}

fn served(name: []const u8) ?cli.Flag {
    const other = cli.find(name) orelse return null;
    return if (other.kind == .store_true and other.native) other else null;
}

/// A model or drafter that starts with ./ or ../ is a folder beside the file; any other text is left as written.
fn anchored(a: Allocator, base: ?[]const u8, text: []const u8) error{OutOfMemory}![]const u8 {
    const b = base orelse return text;
    if (!std.mem.startsWith(u8, text, "./") and !std.mem.startsWith(u8, text, "../")) return text;
    const joined = try std.fs.path.join(a, &.{ b, if (text[1] == '/') text[2..] else text });
    return if (std.mem.startsWith(u8, joined, "--")) try std.fs.path.join(a, &.{ ".", joined }) else joined; // not a flag
}
