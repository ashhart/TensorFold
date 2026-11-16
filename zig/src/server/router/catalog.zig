//! The router's models: the names each answers to, the serve child that loads it and what loading it needs.
const std = @import("std");
const cli = @import("../cli.zig");
const hub = @import("../hub.zig");
const benchmark_server = @import("../benchmark_server.zig");
const log = @import("../log.zig");
const Allocator = std.mem.Allocator;

/// One model as a config table or the cache names it: its id, aliases, model folder or repo id and serve flags.
pub const Model = struct {
    id: []const u8,
    aliases: []const []const u8 = &.{},
    model: []const u8,
    flags: []const []const u8 = &.{},
    /// The config line of its table's first key, for messages (0: from the cache).
    line: u32 = 0,
};

/// One loadable model: its names, its child's whole command line and its memory floor.
pub const Entry = struct {
    id: []const u8,
    aliases: []const []const u8 = &.{},
    argv: []const []const u8,
    /// Bytes it needs loaded (weights, drafter and runtime margin); null when there is no estimate.
    floor: ?u64 = null,
    /// Seconds its child gets to become ready, as the benchmark's child gets them for the same weights.
    startup_s: f64 = 600,
};

/// One checkpoint's memory floor and weights, as ``tensorfold info`` computes them.
pub const Memory = struct { floor: u64, weights: u64 };

/// The checkpoint facts the router reads through ``tensorfold models`` and ``tensorfold info``.
pub const Checkpoints = struct {
    repos: *const fn (Allocator, std.Io, ?*const std.process.Environ.Map) anyerror![]const []const u8,
    memory: *const fn (Allocator, std.Io, []const u8) ?Memory,
};

/// The entry whose id or alias is ``name``.
pub fn find(entries: []const Entry, name: []const u8) ?usize {
    for (entries, 0..) |e, i| {
        if (std.mem.eql(u8, e.id, name)) return i;
        for (e.aliases) |alias| if (std.mem.eql(u8, alias, name)) return i;
    }
    return null;
}

/// Every cached checkpoint ``tensorfold models`` lists, each answering to its repo id.
pub fn cached(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, checkpoints: Checkpoints) ![]const Model {
    const repos = try checkpoints.repos(a, io, env);
    const out = try a.alloc(Model, repos.len);
    for (repos, out) |repo, *m| m.* = .{ .id = repo, .model = repo };
    return out;
}

/// The child's command line: this binary serving ``m`` on 127.0.0.1 at a free port, named by its id and aliases.
pub fn childArgv(a: Allocator, exe: []const u8, m: Model) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    try out.appendSlice(a, &.{ exe, "serve", m.model, "--host", "127.0.0.1", "--port", "0", "--name", m.id });
    for (m.aliases) |alias| try out.appendSlice(a, &.{ "--alias", alias });
    try out.appendSlice(a, m.flags);
    return out.items;
}

/// The entries for ``models``: each one resolved, its command line checked as serve checks it, its floor read.
pub fn build(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, exe: []const u8, models: []const Model, checkpoints: Checkpoints, problem: *[]const u8) !?[]const Entry {
    const out = try a.alloc(Entry, models.len);
    for (models, out) |m, *e| e.* = try entry(a, io, env, exe, m, checkpoints, problem) orelse return null;
    return out;
}

/// ``build`` for the cache's models: one that ``serve`` would refuse is left out and logged, so the rest still serve.
pub fn buildCached(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, exe: []const u8, models: []const Model, checkpoints: Checkpoints) ![]const Entry {
    var out: std.ArrayList(Entry) = .empty;
    for (models) |m| {
        var problem: []const u8 = "";
        if (try entry(a, io, env, exe, m, checkpoints, &problem)) |e| try out.append(a, e) else log.line("not routing {s}", .{problem});
    }
    return out.items;
}

fn entry(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, exe: []const u8, m: Model, checkpoints: Checkpoints, problem: *[]const u8) !?Entry {
    const argv = try childArgv(a, exe, m);
    var u: cli.Usage = .{};
    const parsed = cli.parse(a, argv[2..], &u) catch |err| switch (err) {
        error.Usage => {
            problem.* = try std.fmt.allocPrint(a, "model '{s}'{s}: {s}", .{ m.id, try where(a, m), u.message });
            return null;
        },
        else => |x| return x,
    };
    var why: []const u8 = "";
    const dir = try hub.resolve(a, io, env, m.model, &why) orelse {
        problem.* = try std.fmt.allocPrint(a, "model '{s}'{s}: {s}", .{ m.id, try where(a, m), why });
        return null;
    };
    var drafter: u64 = 0;
    var known = true;
    if (parsed.drafter) |d| {
        const ddir = try hub.resolve(a, io, env, d, &why) orelse {
            problem.* = try std.fmt.allocPrint(a, "model '{s}'{s}: drafter: {s}", .{ m.id, try where(a, m), why });
            return null;
        };
        if (checkpoints.memory(a, io, ddir)) |dm| drafter = dm.weights else known = false;
    }
    var e: Entry = .{ .id = m.id, .aliases = m.aliases, .argv = argv };
    if (checkpoints.memory(a, io, dir)) |mem| {
        if (known) e.floor = mem.floor + drafter;
        e.startup_s = benchmark_server.startupTimeout(mem.weights + drafter);
    }
    return e;
}

fn where(a: Allocator, m: Model) ![]const u8 {
    return if (m.line > 0) std.fmt.allocPrint(a, " (line {d})", .{m.line}) else "";
}

test "serve --router: a name finds its model by id or alias, and the child serves it on loopback under those names" {
    const entries = [_]Entry{
        .{ .id = "glm", .aliases = &.{ "glm-flash", "g" }, .argv = &.{} },
        .{ .id = "org/qwen", .argv = &.{} },
    };
    try std.testing.expectEqual(@as(?usize, 0), find(&entries, "glm"));
    try std.testing.expectEqual(@as(?usize, 0), find(&entries, "g"));
    try std.testing.expectEqual(@as(?usize, 1), find(&entries, "org/qwen"));
    try std.testing.expectEqual(@as(?usize, null), find(&entries, "qwen"));
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const argv = try childArgv(a, "/opt/example/tensorfold", .{ .id = "glm", .aliases = &.{"g"}, .model = "org/glm", .flags = &.{"--context=8192"} });
    const want = [_][]const u8{ "/opt/example/tensorfold", "serve", "org/glm", "--host", "127.0.0.1", "--port", "0", "--name", "glm", "--alias", "g", "--context=8192" };
    try std.testing.expectEqual(want.len, argv.len);
    for (want, argv) |w, g| try std.testing.expectEqualStrings(w, g);
    var u: cli.Usage = .{};
    const parsed = try cli.parse(a, argv[2..], &u);
    try std.testing.expect(parsed.port == 0 and std.mem.eql(u8, parsed.host, "127.0.0.1") and parsed.context.? == 8192);
    try std.testing.expectEqualStrings("glm", parsed.name);
    try std.testing.expectEqualStrings("g", parsed.alias[0]);
}

test "serve --router: an entry's floor adds its drafter, a cached model serve refuses is left out, a config's stops the start" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "model");
    try tmp.dir.createDirPath(io, "drafter");
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    const gib: u64 = 1 << 30;
    const Disk = struct {
        fn repos(_: Allocator, _: std.Io, _: ?*const std.process.Environ.Map) anyerror![]const []const u8 {
            return &.{};
        }
        fn memory(_: Allocator, _: std.Io, dir: []const u8) ?Memory {
            if (std.mem.endsWith(u8, dir, "/model")) return .{ .floor = 30 * gib, .weights = 28 * gib };
            if (std.mem.endsWith(u8, dir, "/drafter")) return .{ .floor = 3 * gib, .weights = gib };
            return null;
        }
    };
    const checkpoints: Checkpoints = .{ .repos = Disk.repos, .memory = Disk.memory };
    var env: std.process.Environ.Map = .init(a);
    const drafter = try std.fmt.allocPrint(a, "--drafter={s}", .{try std.fs.path.join(a, &.{ root, "drafter" })});
    const models = [_]Model{
        .{ .id = "big", .model = try std.fs.path.join(a, &.{ root, "model" }), .flags = &.{drafter}, .line = 3 },
        .{ .id = "gone", .model = try std.fs.path.join(a, &.{ root, "missing" }), .line = 9 },
    };
    var problem: []const u8 = "";
    try std.testing.expect(try build(a, io, &env, "/opt/example/tensorfold", &models, checkpoints, &problem) == null);
    try std.testing.expect(std.mem.startsWith(u8, problem, "model 'gone' (line 9): "));
    const kept = try buildCached(a, io, &env, "/opt/example/tensorfold", &models, checkpoints);
    try std.testing.expectEqual(@as(usize, 1), kept.len);
    try std.testing.expectEqualStrings("big", kept[0].id);
    try std.testing.expectEqual(31 * gib, kept[0].floor.?);
    try std.testing.expectEqual(benchmark_server.startupTimeout(29 * gib), kept[0].startup_s);
}
