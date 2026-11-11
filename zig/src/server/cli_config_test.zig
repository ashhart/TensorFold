//! cli_config.zig: where ``--config`` is found, what a file's keys become, and that the command line wins.
const std = @import("std");
const cli = @import("cli.zig");
const config = @import("cli_config.zig");
const t = std.testing;

const Fixture = struct {
    arena: std.heap.ArenaAllocator = .init(t.allocator),
    u: cli.Usage = .{},

    fn deinit(f: *Fixture) void {
        f.arena.deinit();
    }

    /// The merged arguments for `text` as ``f.toml``, given before `rest` on a command line.
    fn merged(f: *Fixture, text: []const u8, rest: []const []const u8) ![]const []const u8 {
        const a = f.arena.allocator();
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(a, &.{ "--config", "f.toml" });
        try argv.appendSlice(a, rest);
        const found = (try config.locate(a, argv.items, &f.u)).?;
        const out = config.merge(a, argv.items, found, text, &f.u) catch |e| {
            if (e == error.Usage) std.debug.print("refused: {s}\n", .{f.u.message});
            return e;
        };
        return out.argv;
    }

    fn tokens(f: *Fixture, text: []const u8, want: []const []const u8) !void {
        const got = try f.merged(text, &.{});
        errdefer std.debug.print("file: {s}\n", .{text});
        try t.expectEqual(want.len, got.len);
        for (want, got) |w, g| try t.expectEqualStrings(w, g);
    }

    /// `text` is refused with exactly `message` after the ``argument --config: f.toml line N: `` prefix.
    fn refused(f: *Fixture, text: []const u8, line: u32, message: []const u8) !void {
        const a = f.arena.allocator();
        const argv: []const []const u8 = &.{ "--config", "f.toml" };
        const found = (try config.locate(a, argv, &f.u)).?;
        try t.expectError(error.Usage, config.merge(a, argv, found, text, &f.u));
        try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: f.toml line {d}: {s}", .{ line, message }), f.u.message);
    }

    fn locateFails(f: *Fixture, argv: []const []const u8, message: []const u8) !void {
        try t.expectError(error.Usage, config.locate(f.arena.allocator(), argv, &f.u));
        try t.expectEqualStrings(message, f.u.message);
    }
};

test "serve --config: locate finds the flag as parse consumes arguments" {
    var f: Fixture = .{};
    defer f.deinit();
    const a = f.arena.allocator();
    try t.expect(try config.locate(a, &.{ "m", "--port", "1" }, &f.u) == null);
    const spaced = (try config.locate(a, &.{ "m", "--port", "1", "--config", "a.toml", "--dashboard" }, &f.u)).?;
    try t.expectEqualDeep(config.Found{ .path = "a.toml", .at = 3, .width = 2, .model = true }, spaced);
    const joined = (try config.locate(a, &.{ "--config=dir/a.TOML", "m" }, &f.u)).?;
    try t.expectEqualDeep(config.Found{ .path = "dir/a.TOML", .at = 0, .width = 1, .model = true }, joined);
    try t.expect(!(try config.locate(a, &.{ "--config", "a.toml", "--port", "1" }, &f.u)).?.model);
    // another flag's value is neither --config nor the model
    try t.expect(try config.locate(a, &.{ "m", "--name", "--config" }, &f.u) == null);
    try t.expect(!(try config.locate(a, &.{ "--name", "served", "--config", "a.toml" }, &f.u)).?.model);
    try t.expect(try config.locate(a, &.{ "m", "--name=--config", "a.toml" }, &f.u) == null);
    // without --config nothing is refused here: parse reports it, as before
    try t.expect(try config.locate(a, &.{ "m", "--port" }, &f.u) == null);
    try t.expect(try config.locate(a, &.{ "m", "--bogus", "--config" }, &f.u) == null);
    try f.locateFails(&.{ "--config", "a.toml", "--config=b.toml" }, "argument --config: given twice; a command reads one config file");
    // a trailing --config has no file to read: parse reports it, in its usual words
    try t.expect(try config.locate(a, &.{ "m", "--config" }, &f.u) == null);
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--config" }, &f.u));
    try t.expectEqualStrings("argument --config: expected one argument", f.u.message);
    try f.locateFails(&.{ "--config", "a.yaml" }, "argument --config: expected a .toml file: 'a.yaml'");
    try f.locateFails(&.{"--config="}, "argument --config: expected a .toml file: ''");
    try f.locateFails(&.{ "--config", "toml" }, "argument --config: expected a .toml file: 'toml'");
    // removing --config must not repair a flag left without its value
    try f.locateFails(&.{ "m", "--prompt-cache-disk", "--config", "a.toml", "cache" }, "argument --prompt-cache-disk: expected one argument");
    try f.locateFails(&.{ "m", "--config", "a.toml", "--port" }, "argument --port: expected one argument");
}

test "serve --config: each kind of flag takes its values and refuses the rest by line" {
    var f: Fixture = .{};
    defer f.deinit();
    try f.tokens("", &.{});
    try f.tokens("host = \"0.0.0.0\"\nport = 9000\ntemperature = 0.5\n", &.{ "--host=0.0.0.0", "--port=9000", "--temperature=0.5" });
    try f.tokens("max_tokens = 1_024\ntop-k = +40\nparallel = \"auto\"\n", &.{ "--max-tokens=1024", "--top-k=40", "--parallel=auto" });
    try f.tokens("chat-template = \"-x\"\nname = \"\"\n", &.{ "--chat-template=-x", "--name=" });
    try f.tokens("alias = \"a\"\n", &.{"--alias=a"});
    try f.tokens("alias = [\"a\", \"b\"]\napi_key = []\n", &.{ "--alias=a", "--alias=b" });
    try f.tokens("dashboard = true\nmetrics-open = false\n", &.{"--dashboard"});
    try f.tokens("thinking = false\n", &.{"--no-thinking"});
    try f.tokens("no-thinking = false\n", &.{"--thinking"});
    try f.tokens("thinking = true\nno_drafts = true\n", &.{ "--thinking", "--no-drafts" });
    try f.tokens("no-drafts = false\n", &.{});
    try f.refused("port = 1\nprot = 2\n", 2, "unknown key 'prot'");
    try f.refused("\"--port\" = 2\n", 1, "unknown key '--port'");
    try f.refused("config = \"other.toml\"\n", 1, "a config file cannot name another config file");
    try f.refused("port = true\n", 1, "--port takes a value, not true or false");
    try f.refused("port = [1, 2]\n", 1, "--port takes one value, not a list");
    try f.refused("alias = true\n", 1, "--alias takes values, not true or false");
    try f.refused("# aliases\nalias = [\"a\",\n  false]\n", 2, "--alias takes values, not true or false");
    try f.refused("dashboard = \"yes\"\n", 1, "--dashboard is a switch: use true or false");
    try f.refused("dashboard = 1\n", 1, "--dashboard is a switch: use true or false");
    try f.refused("dashboard = [true]\n", 1, "--dashboard is a switch: use true or false");
    try f.refused("max-tokens = 1\nmax_tokens = 2\n", 2, "--max-tokens is set twice ('max_tokens' and its other spelling)");
    try f.refused("model = 7\n", 1, "model takes a string");
    try f.refused("model = [\"m\"]\n", 1, "model takes a string");
    try f.refused("model = \"--name\"\n", 1, "model cannot start with --, which parse reads as a flag: write a folder as ./--name");
    try f.tokens("model = \"-m\"\n", &.{"-m"});
    try f.refused("port = 1\n[serve]\n", 2, "tables are not read: every key is a top-level setting");
    // a flag this build does not serve is refused even where its value would emit nothing
    try t.expect(!cli.find("--vision-offload").?.native and !cli.find("--policy").?.native);
    try f.refused("vision-offload = false\n", 1, "--vision-offload is not served by this build");
    try f.refused("port = 1\npolicy = []\n", 2, "--policy is not served by this build");
    try f.refused("lane-kernels = \"auto\"\n", 1, "--lane-kernels is not served by this build");
}

test "serve --config: the file's model yields to the command line's, and its flags come first" {
    var f: Fixture = .{};
    defer f.deinit();
    const text = "port = 1\nmodel = \"file/model\"\nalias = [\"a\"]\n";
    const alone = try f.merged(text, &.{ "--port", "2" });
    try t.expectEqual(@as(usize, 5), alone.len);
    for ([_][]const u8{ "file/model", "--port=1", "--alias=a", "--port", "2" }, alone) |w, g| try t.expectEqualStrings(w, g);
    const named = try f.merged(text, &.{ "cli/model", "--alias", "z" });
    for ([_][]const u8{ "--port=1", "--alias=a", "cli/model", "--alias", "z" }, named) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqual(@as(usize, 5), named.len);
    // --config between other arguments leaves them in order
    const a = f.arena.allocator();
    const argv: []const []const u8 = &.{ "m", "--dashboard", "--config=f.toml", "--port", "3" };
    const out = try config.merge(a, argv, (try config.locate(a, argv, &f.u)).?, "port = 1\n", &f.u);
    for ([_][]const u8{ "--port=1", "m", "--dashboard", "--port", "3" }, out.argv) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqualStrings("f.toml", out.path.?);
    try t.expectEqual(@as(usize, 1), out.options);
}

test "serve --config: a file parses as its flags typed by hand, and the command line wins" {
    var f: Fixture = .{};
    defer f.deinit();
    const a = f.arena.allocator();
    const text =
        \\model = "org/Model"
        \\host = "0.0.0.0"
        \\port = 9000
        \\context = 131072
        \\temperature = 0.6
        \\thinking = false
        \\dashboard = true
        \\alias = ["a", "b"]
        \\compact-at = "auto"
        \\prompt_cache_gib = 2.5
    ;
    const typed: []const []const u8 = &.{ "org/Model", "--host", "0.0.0.0", "--port", "9000", "--context", "131072", "--temperature", "0.6", "--no-thinking", "--dashboard", "--alias", "a", "--alias", "b", "--compact-at", "auto", "--prompt-cache-gib", "2.5" };
    const from_file = try cli.parse(a, try f.merged(text, &.{}), &f.u);
    const by_hand = try cli.parse(a, typed, &f.u);
    try t.expectEqualDeep(by_hand, from_file);
    try t.expectEqualDeep(try cli.request(a, "d", "t", by_hand), try cli.request(a, "d", "t", from_file));
    try t.expectEqual(@as(u16, 9000), from_file.port);
    try t.expect(!from_file.thinking and from_file.dashboard and from_file.compact_auto);

    const over = try cli.parse(a, try f.merged(text, &.{ "other/model", "--port", "9001", "--thinking", "--alias", "z", "--compact-at", "0.5", "--temperature=0" }), &f.u);
    try t.expectEqualStrings("other/model", over.model);
    try t.expectEqual(@as(u16, 9001), over.port);
    try t.expect(over.thinking);
    try t.expectEqual(@as(?f64, 0), over.temperature);
    try t.expect(!over.compact_auto);
    try t.expectEqual(@as(?f64, 0.5), over.compact_fraction);
    try t.expectEqual(@as(usize, 3), over.alias.len); // an append flag accumulates: the file's, then the command line's
    for ([_][]const u8{ "a", "b", "z" }, over.alias) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqualStrings("0.0.0.0", over.host); // what the command line does not name stays the file's
    try t.expect(over.dashboard);

    // flags that refuse each other do so across the file and the command line
    if (@import("builtin").os.tag != .windows) {
        try t.expectError(error.Usage, cli.parse(a, try f.merged("unix-socket = \"/tmp/s.sock\"\n", &.{ "m", "--port", "1" }), &f.u));
        try t.expectEqualStrings("argument --unix-socket: not allowed with --port", f.u.message);
    }
    // a value the flag refuses is refused by parse, in its own words
    try t.expectError(error.Usage, cli.parse(a, try f.merged("port = 99999\n", &.{"m"}), &f.u));
    try t.expectEqualStrings("argument --port: invalid port: '99999'", f.u.message);
    try t.expectError(error.Usage, cli.parse(a, try f.merged("backend = \"vulkan\"\n", &.{"m"}), &f.u));
    try t.expectEqualStrings("argument --backend: invalid choice: 'vulkan'", f.u.message);
    try t.expectError(error.Usage, cli.parse(a, try f.merged("port = 1\n", &.{}), &f.u));
    try t.expectEqualStrings("the following arguments are required: model", f.u.message);
}

test "serve --config: expand reads the file, passes other commands through and names what it cannot read" {
    var f: Fixture = .{};
    defer f.deinit();
    const a = f.arena.allocator();
    const plain: []const []const u8 = &.{ "m", "--port", "1" };
    const same = try config.expand(a, t.io, plain, &f.u);
    try t.expect(same.argv.ptr == plain.ptr and same.argv.len == plain.len and same.path == null);
    try t.expect(try config.note(a, same) == null);

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(t.io, .{ .sub_path = "serve.toml", .data = "model = \"org/Model\"\nport = 9000\n" });
    const path = try std.fs.path.join(a, &.{ dir, "serve.toml" });
    const read = try config.expand(a, t.io, &.{ "--config", path, "--dashboard" }, &f.u);
    try t.expectEqual(@as(usize, 3), read.argv.len);
    for ([_][]const u8{ "org/Model", "--port=9000", "--dashboard" }, read.argv) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqualStrings(try std.fmt.allocPrint(a, "--config {s} supplied 2 options, read before the command line's", .{path}), (try config.note(a, read)).?);
    try t.expectEqualStrings("--config x.toml supplied 1 option, read before the command line's", (try config.note(a, .{ .argv = &.{}, .path = "x.toml", .options = 1 })).?);

    const missing = try std.fs.path.join(a, &.{ dir, "none.toml" });
    try t.expectError(error.Usage, config.expand(a, t.io, &.{ "m", "--config", missing }, &f.u));
    try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: cannot read {s}: FileNotFound", .{missing}), f.u.message);

    // exactly the limit is read; one byte more, and far more, get the size message
    const filler = try a.alloc(u8, 3 * config.limit);
    @memset(filler, '#');
    for ([_]usize{ config.limit, config.limit + 1, 3 * config.limit }) |size| {
        try tmp.dir.writeFile(t.io, .{ .sub_path = "big.toml", .data = filler[0..size] });
        const big = try std.fs.path.join(a, &.{ dir, "big.toml" });
        if (size == config.limit) {
            try t.expectEqual(@as(usize, 0), (try config.expand(a, t.io, &.{ "m", "--config", big }, &f.u)).options);
        } else {
            try t.expectError(error.Usage, config.expand(a, t.io, &.{ "m", "--config", big }, &f.u));
            try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: {s} is over 1 MiB", .{big}), f.u.message);
        }
    }
}

test "serve --config: parse refuses the flag itself and capabilities lists it" {
    var f: Fixture = .{};
    defer f.deinit();
    const a = f.arena.allocator();
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--config", "f.toml" }, &f.u));
    try t.expectEqualStrings("argument --config: serve reads it before its other flags; it cannot appear here", f.u.message);
    var out: std.Io.Writer.Allocating = .init(a);
    try cli.capabilities(&out.writer, .{ .version = "test" });
    try t.expect(std.mem.indexOf(u8, out.written(), "\"--config\": {}") != null);
}
