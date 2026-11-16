//! config.zig and args.zig: what a router config and command line give, and each refusal with its line.
const std = @import("std");
const cli = @import("../cli.zig");
const config = @import("config.zig");
const args = @import("args.zig");
const toml = @import("../toml_flat.zig");
const t = std.testing;

const Fixture = struct {
    arena: std.heap.ArenaAllocator = .init(t.allocator),
    u: cli.Usage = .{},

    fn deinit(f: *Fixture) void {
        f.arena.deinit();
    }

    fn models(f: *Fixture, text: []const u8) ![]const @import("catalog.zig").Model {
        return config.parse(f.arena.allocator(), "conf/router.toml", text, &f.u) catch |e| {
            if (e == error.Usage) std.debug.print("refused: {s}\n", .{f.u.message});
            return e;
        };
    }

    /// ``text`` is refused with exactly ``message`` after the ``argument --router-config: FILE line N: `` prefix.
    fn refused(f: *Fixture, text: []const u8, line: u32, message: []const u8) !void {
        try t.expectError(error.Usage, config.parse(f.arena.allocator(), "conf/router.toml", text, &f.u));
        try t.expectEqualStrings(try std.fmt.allocPrint(f.arena.allocator(), "argument --router-config: conf/router.toml line {d}: {s}", .{ line, message }), f.u.message);
    }

    fn settings(f: *Fixture, argv: []const []const u8) !args.Settings {
        return args.parse(f.arena.allocator(), argv, &f.u);
    }

    fn badArgs(f: *Fixture, argv: []const []const u8, message: []const u8) !void {
        try t.expectError(error.Usage, args.parse(f.arena.allocator(), argv, &f.u));
        try t.expectEqualStrings(message, f.u.message);
    }
};

test "serve --router: each [[models]] table gives an id, aliases, a model beside the file and serve flags as typed" {
    var f: Fixture = .{};
    defer f.deinit();
    const got = try f.models(
        \\# two models behind one address
        \\[[models]]
        \\id = "glm"
        \\model = "org/GLM-Example"
        \\aliases = ["glm-flash", "g"]
        \\context = 131072
        \\thinking = false
        \\chat_template = "templates/glm.jinja"
        \\
        \\[[ models ]]   # a folder beside the file
        \\id = "small"
        \\model = "./weights/small"
        \\aliases = "s"
        \\parallel = "auto"
    );
    try t.expectEqual(@as(usize, 2), got.len);
    try t.expectEqualStrings("glm", got[0].id);
    try t.expectEqualStrings("org/GLM-Example", got[0].model);
    try t.expectEqual(@as(usize, 2), got[0].aliases.len);
    try t.expectEqualStrings("g", got[0].aliases[1]);
    const want = [_][]const u8{ "--context=131072", "--no-thinking", "--chat-template=conf/templates/glm.jinja" };
    try t.expectEqual(want.len, got[0].flags.len);
    for (want, got[0].flags) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqual(@as(u32, 3), got[0].line);
    try t.expectEqualStrings("conf/weights/small", got[1].model);
    try t.expectEqualStrings("s", got[1].aliases[0]);
    try t.expectEqualStrings("--parallel=auto", got[1].flags[0]);
    var u: cli.Usage = .{};
    const parsed = try cli.parse(f.arena.allocator(), try std.mem.concat(f.arena.allocator(), []const u8, &.{ &.{got[0].model}, got[0].flags }), &u);
    try t.expect(parsed.context.? == 131072 and !parsed.thinking);
}

test "serve --router: a router config refuses top-level keys, missing ids and models, owned flags and repeated names" {
    var f: Fixture = .{};
    defer f.deinit();
    try f.refused("max-loaded = 2\n[[models]]\nid = \"a\"\nmodel = \"o/a\"\n", 1, "top-level keys are not read: each model is a [[models]] table, and the router's flags go on its command line");
    try f.refused("[[models]]\nmodel = \"o/a\"\n", 1, "a [[models]] table needs an id");
    try f.refused("[[models]]\nid = \"a\"\n", 1, "'a' has no model: add model = a folder or a Hugging Face repo id");
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\nport = 9000\n", 4, "port is the router's: each model's server listens on 127.0.0.1 at a port the router picks");
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\nalias = [\"b\"]\n", 4, "alias is the router's: a model answers to its id and aliases: list them in aliases");
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\napi_key_file = \"/etc/k\"\n", 4, "api-key-file is the router's: each model's server takes a key of its own; give the router --api-key or --api-key-file");
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\nidle-unload = 60\n", 4, "idle-unload is the router's: the router stops idle models: give it --idle-unload SECONDS");
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\n[[models]]\nid = \"b\"\nmodel = \"o/b\"\naliases = [\"a\"]\n", 4, "'a' already names model 'a'");
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\naliases = [\"x\", \"x\"]\n", 1, "'x' is named twice in model 'a'");
    try f.refused("[[models]]\nid = \"has space\"\nmodel = \"o/a\"\n", 2, "id 'has space': expected 1 to 256 bytes without spaces or control characters, not starting with -");
    try f.refused("[[models]]\nid = 7\nmodel = \"o/a\"\n", 2, "id takes a string");
    // a serve key keeps serve's own refusal, and the table and file rules are the flat reader's
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\nprot = 1\n", 4, "unknown key 'prot'");
    try f.refused("[[models]]\nid = \"a\"\nmodel = \"o/a\"\ncontext = \"x\"\n", 4, "argument --context: invalid int value: 'x'");
    try f.refused("[models]\nid = \"a\"\n", 1, "a [table] is not read: each item is a [[models]] table");
    try f.refused("[[servers]]\nid = \"a\"\n", 1, "unknown table [[servers]]: this file reads [[models]] tables");
    try f.refused("[[models]]\nid = \"a\"\nid = \"b\"\n", 3, "key defined twice");
    try t.expectError(error.Usage, config.parse(f.arena.allocator(), "conf/router.toml", "# nothing\n", &f.u));
    try t.expectEqualStrings("argument --router-config: conf/router.toml lists no models: add a [[models]] table with an id and a model", f.u.message);
}

test "serve --router: the flat reader still refuses every table, and a shaped one keys duplicates by table" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: toml.Problem = .{};
    try t.expectError(error.Invalid, toml.parse(a, "[[models]]\nid = \"a\"\n", &problem));
    try t.expectEqualStrings("tables are not read: every key is a top-level setting", problem.message);
    const shaped = try toml.parseShaped(a, "top = 1\n[[models]]\nid = \"a\"\n\n[[models]]\nid = \"b\"\n", .{ .table = "models" }, &problem);
    try t.expectEqual(@as(usize, 3), shaped.entries.len);
    try t.expectEqualSlices(u32, &.{ 2, 5 }, shaped.tables);
    try t.expectEqual(@as(u32, 2), shaped.entries[2].table);
    try t.expectError(error.Invalid, toml.parseShaped(a, "models = \"x\"\n[[models]]\n", .{ .table = "models" }, &problem));
    try t.expectEqualStrings("key defined twice", problem.message);
    try t.expectError(error.Invalid, toml.parseShaped(a, "[[models]] x\n", .{ .table = "models" }, &problem));
    try t.expectEqualStrings("unexpected text after the table header", problem.message);
}

test "serve --router: the command line takes the router's flags and serve's shared ones, and refuses a model's" {
    var f: Fixture = .{};
    defer f.deinit();
    const s = try f.settings(&.{ "--router", "--router-config", "r.toml", "--max-loaded=2", "--idle-unload", "600", "--host", "::1", "--port", "9000", "--api-key-file", "/etc/keys", "--api-key", "sk-example" });
    try t.expectEqualStrings("r.toml", s.config.?);
    try t.expect(s.max_loaded == 2 and s.idle_s == 600 and s.port == 9000);
    try t.expectEqualStrings("::1", s.host);
    try t.expectEqualStrings("/etc/keys", s.api_key_file.?);
    try t.expectEqualStrings("sk-example", s.api_keys[0]);
    const plain = try f.settings(&.{"--router"});
    try t.expect(plain.config == null and plain.max_loaded == 1 and plain.idle_s == 0 and plain.port == 8080);
    try t.expect(args.wanted(&.{ "--port", "1", "--router" }) and !args.wanted(&.{"org/model"}));
    try f.badArgs(&.{ "--router", "--context", "8192" }, "argument --context: a model's serve flag; with --router, set it on the model in --router-config FILE.toml");
    try f.badArgs(&.{ "--router", "org/model" }, "unrecognized arguments: org/model: --router serves the models of --router-config or the cache and takes no MODEL");
    try f.badArgs(&.{ "--router", "--config", "a.toml" }, "argument --config: with --router, models and their serve flags are listed in --router-config FILE.toml");
    try f.badArgs(&.{ "--router", "--alias", "x" }, "argument --alias: with --router, each model answers to its id and aliases in --router-config FILE.toml");
    try f.badArgs(&.{ "--router", "--unix-socket", "/run/x.sock" }, "argument --unix-socket: the router listens on --host and --port only");
    try f.badArgs(&.{ "--router", "--max-loaded", "0" }, "argument --max-loaded: expected 1 to 64 models: '0'");
    try f.badArgs(&.{ "--router", "--router-config", "r.json" }, "argument --router-config: expected a .toml file: 'r.json'");
    try f.badArgs(&.{ "--router", "--port", "99999" }, "argument --port: invalid port: '99999'");
    try f.badArgs(&.{ "--router", "--idle-unload", "-1" }, "argument --idle-unload: expected a seconds count from 0 (off) under 9223372036: '-1'");
    try f.badArgs(&.{ "--router", "--bogus" }, "unrecognized arguments: --bogus");
    try f.badArgs(&.{ "--router", "--port" }, "argument --port: expected one argument");
}

test "serve --router: the config file is opened as serve opens its own: no link, nothing others can write" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const c = @import("sys").c;
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fs.path.join(a, &.{ dir, "router.toml" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "router.toml", .data = "[[models]]\nid = \"a\"\nmodel = \"./w\"\n" });
    const pathz = try a.dupeSentinel(u8, path, 0);
    try t.expectEqual(@as(c_int, 0), c.fchmodat(std.posix.AT.FDCWD, pathz, 0o644, 0));
    var u: cli.Usage = .{};
    const got = try config.load(a, path, &u);
    try t.expectEqualStrings(try std.fs.path.join(a, &.{ dir, "w" }), got[0].model);
    try t.expectEqual(@as(c_int, 0), c.fchmodat(std.posix.AT.FDCWD, pathz, 0o664, 0));
    try t.expectError(error.Usage, config.load(a, path, &u));
    try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --router-config: {s} can be written by other users (mode 0664): it chooses what is served; use chmod go-w", .{path}), u.message);
    try tmp.dir.symLink(t.io, "router.toml", "link.toml", .{});
    const link = try std.fs.path.join(a, &.{ dir, "link.toml" });
    try t.expectError(error.Usage, config.load(a, link, &u));
    try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --router-config: cannot open {s} (SymLinkLoop); name an existing file, not a link", .{link}), u.message);
}
