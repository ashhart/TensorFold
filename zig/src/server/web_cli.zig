//! Model-free entry to the separately installed TensorFold Web application.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Environ = std.process.Environ.Map;

pub const Version = struct { version: []const u8, is_release: bool };
const description = "TensorFold Web, a small local web app for picking, running and chatting with models\n";

/// The terminal and launch boundaries are injectable for host-only command tests.
pub const Context = struct {
    a: Allocator,
    io: std.Io,
    environ: *const Environ,
    out: *std.Io.Writer,
    input: *std.Io.Reader,
    interactive: bool,
    self_exe: ?[]const u8 = null,
    launch: *const fn (std.Io, []const []const u8, *const Environ) anyerror!u8 = replace,
};

pub fn main(init: std.process.Init, args: []const []const u8, version: Version) !u8 {
    var out_buf: [1024]u8 = undefined;
    var in_buf: [256]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    var input = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    defer out.interface.flush() catch {};
    return command(.{
        .a = init.arena.allocator(),
        .io = init.io,
        .environ = init.environ_map,
        .out = &out.interface,
        .input = &input.interface,
        .interactive = (std.Io.File.stdin().isTty(init.io) catch false) and
            (std.Io.File.stdout().isTty(init.io) catch false),
    }, args, version);
}

pub fn command(ctx: Context, args: []const []const u8, version: Version) !u8 {
    if (try onPath(ctx, "tensorfold-web")) |app| return launch(ctx, app, args);
    try ctx.out.writeAll(description);
    const spec = try std.fmt.allocPrint(ctx.a, "tensorfold-web @ git+https://github.com/ashhart/TensorFold@v{s}#subdirectory=web", .{version.version});
    const uv = try onPath(ctx, "uv");
    if (!version.is_release) {
        try ctx.out.print("Development build without a release tag; the command below targets the manifest version v{s}.\n", .{version.version});
    } else if (uv == null) {
        try ctx.out.writeAll("uv is not on PATH.\n");
    } else if (!ctx.interactive or std.mem.eql(u8, ctx.environ.get("TENSORFOLD_WEB_NO_PROMPT") orelse "", "1")) {
        try ctx.out.writeAll("Automatic installation needs an interactive terminal and prompting enabled.\n");
    } else {
        try ctx.out.writeAll("Install it now with uv? [y/N] ");
        try ctx.out.flush();
        const answer = ctx.input.takeDelimiterExclusive('\n') catch "";
        const trimmed = std.mem.trim(u8, answer, " \t\r");
        if (std.ascii.eqlIgnoreCase(trimmed, "y") or std.ascii.eqlIgnoreCase(trimmed, "yes")) {
            const status = spawnAndWait(ctx.io, &.{ uv.?, "tool", "install", spec }, ctx.environ) catch |err| {
                try ctx.out.print("Could not run uv: {s}.\n", .{@errorName(err)});
                return instructions(ctx, spec, false);
            };
            if (status != 0) {
                try ctx.out.print("uv tool install failed (exit {d}).\n", .{status});
                return instructions(ctx, spec, false);
            }
            if (try onPath(ctx, "tensorfold-web")) |app| return launch(ctx, app, args);
            // uv's tool-bin directory need not already be on this process's PATH.
            const result = std.process.run(ctx.a, ctx.io, .{
                .argv = &.{ uv.?, "tool", "dir", "--bin" },
                .environ_map = ctx.environ,
                .stdout_limit = .limited(4096),
                .stderr_limit = .limited(4096),
            }) catch return instructions(ctx, spec, false);
            defer ctx.a.free(result.stdout);
            defer ctx.a.free(result.stderr);
            const dir = std.mem.trim(u8, result.stdout, "\r\n");
            if (exitCode(result.term) == 0 and std.fs.path.isAbsolute(dir)) {
                if (try inDir(ctx, dir, "tensorfold-web")) |app| return launch(ctx, app, args);
            }
            try ctx.out.writeAll("uv finished, but its tensorfold-web executable could not be found.\n");
        }
    }
    return instructions(ctx, spec, uv == null);
}

fn instructions(ctx: Context, spec: []const u8, need_uv: bool) !u8 {
    try ctx.out.writeAll("Run:\n");
    if (need_uv) try ctx.out.writeAll(if (builtin.os.tag == .windows) "  py -m pip install --user uv\n" else "  python3 -m pip install --user uv\n");
    try ctx.out.print("  uv tool install \"{s}\"\nThen run tensorfold web again.\n", .{spec});
    return 1;
}

fn launch(ctx: Context, app: []const u8, args: []const []const u8) !u8 {
    var env = try ctx.environ.clone(ctx.a);
    defer env.deinit();
    try env.put("TENSORFOLD_BIN", ctx.self_exe orelse try std.process.executablePathAlloc(ctx.io, ctx.a));
    const argv = try ctx.a.alloc([]const u8, args.len + 1);
    argv[0] = app;
    @memcpy(argv[1..], args);
    try ctx.out.flush();
    return ctx.launch(ctx.io, argv, &env) catch |err| {
        try ctx.out.print("Could not launch tensorfold-web: {s}.\n", .{@errorName(err)});
        return 1;
    };
}

fn replace(io: std.Io, argv: []const []const u8, env: *const Environ) !u8 {
    // Windows cannot replace an image; retain the child's status there.
    if (builtin.os.tag == .windows) return spawnAndWait(io, argv, env);
    return std.process.replace(io, .{ .exe = .{ .path = .cwd() }, .argv = argv, .environ_map = env });
}

pub fn spawnAndWait(io: std.Io, argv: []const []const u8, env: *const Environ) !u8 {
    var child = try std.process.spawn(io, .{ .exe = .{ .path = .cwd() }, .argv = argv, .environ_map = env });
    defer child.kill(io);
    return exitCode(try child.wait(io));
}

fn exitCode(term: std.process.Child.Term) u8 {
    return switch (term) {
        .exited => |code| @intCast(@min(code, 255)),
        .signal => |sig| 128 + @as(u8, @intCast(@min(@backingInt(sig), 127))),
        else => 1,
    };
}

fn onPath(ctx: Context, name: []const u8) !?[]const u8 {
    const path = ctx.environ.get("PATH") orelse return null;
    var dirs = std.mem.splitScalar(u8, path, std.fs.path.delimiter);
    while (dirs.next()) |dir| if (try inDir(ctx, if (dir.len == 0) "." else dir, name)) |app| return app;
    return null;
}

fn inDir(ctx: Context, dir: []const u8, name: []const u8) !?[]const u8 {
    const extensions: []const []const u8 = if (builtin.os.tag == .windows) &.{ ".exe", ".com", ".cmd", ".bat", "" } else &.{""};
    for (extensions) |extension| {
        const file = try std.fmt.allocPrint(ctx.a, "{s}{s}", .{ name, extension });
        const path = try std.fs.path.join(ctx.a, &.{ dir, file });
        std.Io.Dir.cwd().access(ctx.io, path, .{ .execute = true }) catch continue;
        const stat = std.Io.Dir.cwd().statFile(ctx.io, path, .{}) catch continue;
        if (stat.kind == .file) return path;
    }
    return null;
}

test {
    _ = @import("web_cli_test.zig");
}
