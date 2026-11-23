//! The server ``tensorfold chat MODEL`` starts through the benchmark's serve child: its flags checked, then ready.
const std = @import("std");
const cli = @import("cli.zig");
const cli_config = @import("cli_config.zig");
const hub = @import("hub.zig");
const model = @import("benchmark_model.zig");
const server = @import("benchmark_server.zig");
const Allocator = std.mem.Allocator;

/// What a chat's own server says when it fails to start or stop.
pub const words: server.Words = .{
    .log = "Model server could not start; check the checkpoint and directory permissions",
    .spawn = "Could not start the model server; check the TensorFold installation",
    .exited = "Model server exited during startup; check the checkpoint and backend toolchain",
    .timed_out = "Model server startup timed out after",
    .stuck = "Model server did not exit after termination; cleanup is incomplete",
};

/// Managed child setup: request ``key`` (fresh by default), and optional full-list ``environment_key`` for the child.
pub const Setup = struct { a: Allocator, io: std.Io, environ: *const std.process.Environ.Map, exe: ?[]const u8 = null, key: ?[]const u8 = null, environment_key: ?[]const u8 = null, log_dir: []const u8 };

/// The started server, the name it serves and the key every request to it carries.
pub const Child = struct { managed: server.Managed, served: []const u8, key: []const u8 };

/// The serve child's arguments: the checkpoint, loopback, a free port and the model's own name, then ``extra``.
pub fn argv(a: Allocator, exe: []const u8, dir: []const u8, served: []const u8, extra: []const []const u8) ![]const []const u8 {
    const head = [_][]const u8{ exe, "serve", dir, "--host", "127.0.0.1", "--port", "0", "--name", served };
    return std.mem.concat(a, []const u8, &.{ &head, extra });
}

/// The name the child will serve, read as serve reads its flags (a ``--name`` after ``--`` wins); null: refused.
pub fn check(a: Allocator, child_argv: []const []const u8, problem: *[]const u8) !?[]const u8 {
    var u: cli.Usage = .{};
    const expanded = cli_config.expand(a, child_argv[2..], &u) catch |e| switch (e) {
        error.Usage => return refuse(a, problem, u.message),
        else => |x| return x,
    };
    const parsed = cli.parse(a, expanded.argv, &u) catch |e| switch (e) {
        error.Usage => return refuse(a, problem, u.message),
        else => |x| return x,
    };
    if (!std.mem.eql(u8, parsed.host, "127.0.0.1") or parsed.port != 0) {
        problem.* = "--host and --port: tensorfold chat gives the server it starts a free port on 127.0.0.1";
        return null;
    }
    if (parsed.api_key.len > 0 or parsed.api_key_file != null) {
        problem.* = "--api-key and --api-key-file: tensorfold chat sends no key to the server it starts";
        return null;
    }
    return parsed.name;
}

fn refuse(a: Allocator, problem: *[]const u8, why: []const u8) !?[]const u8 {
    problem.* = try std.fmt.allocPrint(a, "tensorfold serve refuses the flags after --: {s}", .{why});
    return null;
}

/// A child ready to start: its command line, the name it will serve, how long it gets to load and its key.
pub const Plan = struct { argv: []const []const u8, served: []const u8, timeout_s: f64, key: []const u8 };

/// A fresh key for one child, 32 random bytes as hex, so the server answers this chat and no other local user.
pub fn freshKey(a: Allocator, io: std.Io) ![]const u8 {
    var raw: [32]u8 = undefined;
    io.random(&raw);
    return a.dupe(u8, &std.fmt.bytesToHex(raw, .lower));
}

/// The plan for ``model_arg``: a cached checkpoint, serving flags and the caller-supplied or fresh child key.
pub fn plan(s: Setup, model_arg: []const u8, extra: []const []const u8, startup_timeout: ?f64, problem: *[]const u8) !?Plan {
    const dir = try hub.resolve(s.a, s.io, s.environ, model_arg, problem) orelse return null;
    const exe = s.exe orelse try std.process.executablePathAlloc(s.io, s.a);
    const child_argv = try argv(s.a, exe, dir, hub.servedName("", model_arg, dir), extra);
    const served = try check(s.a, child_argv, problem) orelse return null;
    const timeout_s = startup_timeout orelse server.startupTimeout(model.weightBytes(s.io, dir));
    return .{ .argv = child_argv, .served = served, .timeout_s = timeout_s, .key = s.key orelse try freshKey(s.a, s.io) };
}

/// ``p`` started and ready; null with ``problem`` set ("" when stopped) and ``tail`` the child's last log lines.
pub fn start(s: Setup, p: Plan, tail: *[]const []const u8, problem: *[]const u8) !?Child {
    const o: server.Options = .{ .startup_timeout_s = p.timeout_s, .bench_dir = s.log_dir, .tail = tail, .served = p.served, .lock = false, .own_group = true, .key = p.key, .environment_key = s.environment_key, .words = words };
    const m = try server.Managed.spawn(s.a, s.io, s.environ, p.argv, o, problem) orelse return null;
    return .{ .managed = m, .served = p.served, .key = p.key };
}

test "chat CLI: the started server's flags are serve's, checked before a spawn; its address and keys stay the chat's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const head = [_][]const u8{ "/opt/example/tensorfold-native", "/models/example" };
    const plain = try argv(a, head[0], head[1], "example", &.{ "--parallel", "1" });
    try std.testing.expectEqualStrings("serve", plain[1]);
    try std.testing.expectEqualStrings("example", (try check(a, plain, &problem)).?);
    try std.testing.expectEqualStrings("mine", (try check(a, try argv(a, head[0], head[1], "example", &.{ "--name", "mine" }), &problem)).?);
    const refused = [_]struct { []const []const u8, []const u8 }{
        .{ &.{ "--port", "9000" }, "--host and --port: tensorfold chat gives the server it starts a free port on 127.0.0.1" },
        .{ &.{"--host=localhost"}, "--host and --port: tensorfold chat gives the server it starts a free port on 127.0.0.1" },
        .{ &.{ "--api-key", "example-key" }, "--api-key and --api-key-file: tensorfold chat sends no key to the server it starts" },
        .{ &.{"--paralel"}, "tensorfold serve refuses the flags after --: unrecognized arguments: --paralel" },
        .{ &.{ "--unix-socket", "/tmp/example.sock" }, "tensorfold serve refuses the flags after --: argument --unix-socket: not allowed with --port" },
    };
    for (refused) |c| {
        try std.testing.expect(try check(a, try argv(a, head[0], head[1], "example", c[0]), &problem) == null);
        try std.testing.expectEqualStrings(c[1], problem);
    }
}

test "chat CLI: each child's key is 64 hex characters of fresh randomness" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = try freshKey(a, std.testing.io);
    const two = try freshKey(a, std.testing.io);
    try std.testing.expect(one.len == 64 and !std.mem.eql(u8, one, two));
    for (one) |ch| try std.testing.expect(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'));
}
