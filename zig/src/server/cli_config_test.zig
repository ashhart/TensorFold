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
    // a pair of switches has one key, the positive one
    try f.refused("no-thinking = false\n", 1, "--no-thinking has the switch --thinking: write thinking = true");
    try f.refused("port = 1\nno_thinking = true\n", 2, "--no-thinking has the switch --thinking: write thinking = false");
    try f.tokens("thinking = true\nno_drafts = true\n", &.{ "--thinking", "--no-drafts" });
    try f.tokens("no-drafts = false\n", &.{});
    try f.refused("port = 1\nprot = 2\n", 2, "unknown key 'prot'");
    try f.refused("\"--port\" = 2\n", 1, "unknown key '--port'");
    try f.refused("config = \"other.toml\"\n", 1, "a config file cannot name another config file");
    try f.refused("port = true\n", 1, "--port takes a value, not true or false");
    try f.refused("port = [\"1\", \"2\"]\n", 1, "--port takes one value, not a list");
    try f.refused("alias = true\n", 1, "--alias takes values, not true or false");
    try f.refused("# aliases\nalias = [\"a\",\n  false]\n", 3, "an array holds strings only: quote each item");
    try f.refused("alias = [1]\n", 1, "an array holds strings only: quote each item");
    try f.refused("dashboard = \"yes\"\n", 1, "--dashboard is a switch: use true or false");
    try f.refused("dashboard = 1\n", 1, "--dashboard is a switch: use true or false");
    try f.refused("dashboard = [\"true\"]\n", 1, "--dashboard is a switch: use true or false");
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
    for ([_][]const u8{ "--port=1", "cli/model", "--alias", "z" }, named) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqual(@as(usize, 4), named.len); // --alias on the command line replaces the file's list
    // --config between other arguments leaves them in order
    const a = f.arena.allocator();
    const argv: []const []const u8 = &.{ "m", "--dashboard", "--config=f.toml", "--port", "3" };
    const out = try config.merge(a, argv, (try config.locate(a, argv, &f.u)).?, "port = 1\n", &f.u);
    for ([_][]const u8{ "--port=1", "m", "--dashboard", "--port", "3" }, out.argv) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqualStrings("f.toml", out.path.?);
    try t.expectEqual(@as(usize, 1), out.options);
}

test "serve --config: a relative path in the file starts at the file's folder" {
    var f: Fixture = .{};
    defer f.deinit();
    const a = f.arena.allocator();
    const text =
        \\model = "./weights/m"
        \\chat-template = "t.jinja"
        \\api-key-file = "../keys"
        \\learn-dir = "/abs/learn"
        \\drafter = "./drafter"
        \\name = "plain/name"
        \\kv-cache = "bf16"
    ;
    const sep = std.fs.path.sep_str;
    const argv: []const []const u8 = &.{ "--config", "conf" ++ sep ++ "f.toml" };
    const out = try config.merge(a, argv, (try config.locate(a, argv, &f.u)).?, text, &f.u);
    const want = [_][]const u8{ "conf" ++ sep ++ "weights/m", "--chat-template=conf" ++ sep ++ "t.jinja", "--api-key-file=conf" ++ sep ++ "../keys", "--learn-dir=/abs/learn", "--drafter=conf" ++ sep ++ "drafter", "--name=plain/name", "--kv-cache=bf16" };
    try t.expectEqual(want.len, out.argv.len);
    for (want, out.argv) |w, g| try t.expectEqualStrings(w, g);
    // a repo id, or a folder named without ./, is left as written: the hub resolves it as it does a typed model
    try f.tokens("model = \"org/Model\"\ndrafter = \"org/Drafter\"\n", &.{ "org/Model", "--drafter=org/Drafter" });
    const rel = try config.merge(a, argv, (try config.locate(a, argv, &f.u)).?, "model = \"org/Model\"\ndrafter = \"../d\"\n", &f.u);
    for ([_][]const u8{ "org/Model", "--drafter=conf" ++ sep ++ "../d" }, rel.argv) |w, g| try t.expectEqualStrings(w, g);
    // a number names a folder too, and a folder that starts with -- does not turn the model into a flag
    const odd: []const []const u8 = &.{"--config=--conf" ++ sep ++ "f.toml"};
    const dashed = try config.merge(a, odd, (try config.locate(a, odd, &f.u)).?, "model = \"./weights\"\ncompact-memory = 123\n", &f.u);
    for ([_][]const u8{ "." ++ sep ++ "--conf" ++ sep ++ "weights", "--compact-memory=--conf" ++ sep ++ "123" }, dashed.argv) |w, g| try t.expectEqualStrings(w, g);
    // a file in the working directory has no folder to add
    try f.tokens("model = \"./m\"\nchat-template = \"t.jinja\"\n", &.{ "./m", "--chat-template=t.jinja" });
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
    try t.expectEqual(@as(usize, 1), over.alias.len); // a repeatable flag on the command line replaces the file's list
    try t.expectEqualStrings("z", over.alias[0]);
    // so a key, a host or an origin the file names does not stay in force beside the command line's
    const keys = try cli.parse(a, try f.merged("api-key = [\"old\", \"older\"]\nallowed-host = \"a.example\"\nalias = \"kept\"\n", &.{ "m", "--api-key=new", "--allowed-host", "b.example" }), &f.u);
    try t.expectEqual(@as(usize, 1), keys.api_key.len);
    try t.expectEqualStrings("new", keys.api_key[0]);
    try t.expectEqual(@as(usize, 1), keys.allowed_host.len);
    try t.expectEqualStrings("b.example", keys.allowed_host[0]);
    try t.expectEqualStrings("kept", keys.alias[0]); // a list the command line does not name stays the file's
    try t.expectEqualStrings("0.0.0.0", over.host); // what the command line does not name stays the file's
    try t.expect(over.dashboard);

    // flags that refuse each other do so across the file and the command line
    if (@import("builtin").os.tag != .windows) {
        try t.expectError(error.Usage, cli.parse(a, try f.merged("unix-socket = \"/tmp/s.sock\"\n", &.{ "m", "--port", "1" }), &f.u));
        try t.expectEqualStrings("argument --unix-socket: not allowed with --port", f.u.message);
    }
    // a value the flag refuses is refused in the flag's own words, with the file's line
    try f.refused("model = \"m\"\n\nport = 99999\n", 3, "argument --port: invalid port: '99999'");
    try f.refused("backend = \"vulkan\"\n", 1, "argument --backend: invalid choice: 'vulkan'");
    try f.refused("port = 1\ntemperature = \"warm\"\n", 2, "argument --temperature: invalid float value: 'warm'");
    try f.refused("alias = [\"a\"]\nallowed-host = [\"ok.example\",\n  \"bad host:1\"]\n", 2, "argument --allowed-host: expected a hostname or bracketed IP without a port: 'bad host:1'");
    try f.refused("compact-at = 1.5\n", 1, "argument --compact-at: expected auto or a fraction in (0, 1]: '1.5'");
    // checks that parse runs once every flag is read name the line too
    try f.refused("port = 1\nidle-unload = -1\n", 2, "argument --idle-unload: expected a seconds count from 0 (off) under 9223372036: '-1'");
    try f.refused("name-priority = [\"a=background\", \"broken\"]\n", 1, "argument --name-priority: takes ID=background, not 'broken'");
    {
        var hosts: std.ArrayList(u8) = .empty;
        try hosts.appendSlice(a, "allowed-host = [\n");
        for (0..65) |n| try hosts.print(a, "  \"h{d}.example\",\n", .{n});
        try hosts.appendSlice(a, "]\n");
        try f.refused(hosts.items, 1, "at most 64 --allowed-host and 64 --allowed-origin entries are allowed");
        const argv: []const []const u8 = &.{ "--config", "f.toml", "--name-priority", "x=background" };
        try t.expectError(error.Usage, config.merge(a, argv, (try config.locate(a, argv, &f.u)).?, "name-priority = [\"broken\"]\n", &f.u));
        try t.expectEqualStrings("argument --config: f.toml line 1: argument --name-priority: takes ID=background, not 'broken'", f.u.message);
    }
    // a replaced list is still checked, so a bad entry does not hide behind the command line
    {
        const argv: []const []const u8 = &.{ "--config", "f.toml", "--allowed-host", "b.example" };
        try t.expectError(error.Usage, config.merge(a, argv, (try config.locate(a, argv, &f.u)).?, "allowed-host = \"bad host:1\"\n", &f.u));
        try t.expectEqualStrings("argument --config: f.toml line 1: argument --allowed-host: expected a hostname or bracketed IP without a port: 'bad host:1'", f.u.message);
    }
    // no shell expands ~ in the file: a path flag refuses it, where the hub expands a model's or drafter's ~/
    try f.refused("port = 1\nrecall-dir = \"~/recall\"\n", 2, "--recall-dir: no shell expands ~ in a config file; write the whole path");
    try f.tokens("model = \"~/models/m\"\ndrafter = \"~/d\"\n", &.{ "~/models/m", "--drafter=~/d" });
    try t.expectError(error.Usage, cli.parse(a, try f.merged("port = 1\n", &.{}), &f.u));
    try t.expectEqualStrings("the following arguments are required: model", f.u.message);
}

test "serve --config: expand reads the file, passes other commands through and names what it cannot read" {
    var f: Fixture = .{};
    defer f.deinit();
    const a = f.arena.allocator();
    const plain: []const []const u8 = &.{ "m", "--port", "1" };
    const same = try config.expand(a, plain, &f.u);
    try t.expect(same.argv.ptr == plain.ptr and same.argv.len == plain.len and same.path == null);
    try t.expect(try config.note(a, same) == null);

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(t.io, .{ .sub_path = "serve.toml", .data = "model = \"org/Model\"\nport = 9000\n" });
    const path = try std.fs.path.join(a, &.{ dir, "serve.toml" });
    const read = try config.expand(a, &.{ "--config", path, "--dashboard" }, &f.u);
    try t.expectEqual(@as(usize, 3), read.argv.len);
    for ([_][]const u8{ "org/Model", "--port=9000", "--dashboard" }, read.argv) |w, g| try t.expectEqualStrings(w, g);
    try t.expectEqualStrings(try std.fmt.allocPrint(a, "--config {s} supplied 2 options, read before the command line's", .{path}), (try config.note(a, read)).?);
    try t.expectEqualStrings("--config x.toml supplied 1 option, read before the command line's", (try config.note(a, .{ .argv = &.{}, .path = "x.toml", .options = 1 })).?);

    // a file that other users can write is refused before it is read; a folder is not a file
    if (@import("builtin").os.tag != .windows) {
        const c = @import("sys").c;
        const pathz = try a.dupeSentinel(u8, path, 0);
        for ([_]u32{ 0o664, 0o646, 0o666 }) |mode| {
            try t.expectEqual(@as(c_int, 0), c.fchmodat(std.posix.AT.FDCWD, pathz, @intCast(mode), 0));
            try t.expectError(error.Usage, config.expand(a, &.{ "--config", path }, &f.u));
            try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: {s} can be written by other users (mode 0{o}): it chooses what is served; use chmod go-w", .{ path, mode }), f.u.message);
        }
        for ([_]u32{ 0o600, 0o644, 0o640 }) |mode| {
            try t.expectEqual(@as(c_int, 0), c.fchmodat(std.posix.AT.FDCWD, pathz, @intCast(mode), 0));
            try t.expectEqual(@as(usize, 2), (try config.expand(a, &.{ "--config", path }, &f.u)).options);
        }
        // a file that sets api-key is refused while other users can read it; any other file may be world-readable
        try tmp.dir.writeFile(t.io, .{ .sub_path = "key.toml", .data = "model = \"m\"\n\napi-key = \"k\"\n" });
        const key = try std.fs.path.join(a, &.{ dir, "key.toml" });
        const keyz = try a.dupeSentinel(u8, key, 0);
        for ([_]u32{ 0o644, 0o640, 0o604 }) |mode| {
            try t.expectEqual(@as(c_int, 0), c.fchmodat(std.posix.AT.FDCWD, keyz, @intCast(mode), 0));
            try t.expectError(error.Usage, config.expand(a, &.{ "--config", key }, &f.u));
            try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: {s} line 3: api-key is set in a file other users can read (mode 0{o}); use chmod 600, or api-key-file", .{ key, mode }), f.u.message);
        }
        try t.expectEqual(@as(c_int, 0), c.fchmodat(std.posix.AT.FDCWD, keyz, 0o600, 0));
        try t.expectEqual(@as(u32, 3), (try config.expand(a, &.{ "--config", key }, &f.u)).secret);
        // a link is not followed: the file that is checked is the file that is read
        try tmp.dir.symLink(t.io, "serve.toml", "link.toml", .{});
        const link = try std.fs.path.join(a, &.{ dir, "link.toml" });
        try t.expectError(error.Usage, config.expand(a, &.{ "--config", link }, &f.u));
        try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: cannot open {s} (SymLinkLoop); name an existing file, not a link", .{link}), f.u.message);
        try tmp.dir.createDir(t.io, "folder.toml", .default_dir);
        const folder = try std.fs.path.join(a, &.{ dir, "folder.toml" });
        try t.expectError(error.Usage, config.expand(a, &.{ "--config", folder }, &f.u));
        try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: {s} is not a regular file", .{folder}), f.u.message);
    }

    const missing = try std.fs.path.join(a, &.{ dir, "none.toml" });
    try t.expectError(error.Usage, config.expand(a, &.{ "m", "--config", missing }, &f.u));
    const absent = if (@import("builtin").os.tag == .windows) "OpenFailed" else "FileNotFound";
    try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --config: cannot open {s} ({s}); name an existing file, not a link", .{ missing, absent }), f.u.message);

    // exactly the limit is read; one byte more, and far more, get the size message
    const filler = try a.alloc(u8, 3 * config.limit);
    @memset(filler, '#');
    for ([_]usize{ config.limit, config.limit + 1, 3 * config.limit }) |size| {
        try tmp.dir.writeFile(t.io, .{ .sub_path = "big.toml", .data = filler[0..size] });
        const big = try std.fs.path.join(a, &.{ dir, "big.toml" });
        if (size == config.limit) {
            try t.expectEqual(@as(usize, 0), (try config.expand(a, &.{ "m", "--config", big }, &f.u)).options);
        } else {
            try t.expectError(error.Usage, config.expand(a, &.{ "m", "--config", big }, &f.u));
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

test "serve --config: CORS list overrides and file-line limits" {
    var f: Fixture = .{};
    defer f.deinit();
    const a = f.arena.allocator();
    const merged = try f.merged("cors-origin = [\"https://file.invalid\"]\n", &.{ "m", "--cors-origin", "https://cli.invalid" });
    const got = try cli.parse(a, merged, &f.u);
    try t.expectEqual(@as(usize, 1), got.cors_origin.len);
    try t.expectEqualStrings("https://cli.invalid", got.cors_origin[0]);
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "cors-origin = [");
    for (0..65) |i| {
        if (i > 0) try text.appendSlice(a, ", ");
        try text.appendSlice(a, "\"https://ui.invalid\"");
    }
    try text.appendSlice(a, "]\n");
    try f.refused(text.items, 1, "at most 64 --cors-origin entries are allowed");
}
