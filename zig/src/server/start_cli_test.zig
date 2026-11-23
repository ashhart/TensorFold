//! `tensorfold start` against fake devices, registries, checkpoints and HTTP servers.
const std = @import("std");
const start = @import("start_cli.zig");
const registry = @import("checkpoint_cli").start;
const machine_mod = @import("sys").machine;
const fake = @import("benchmark_fake.zig");
const stop = @import("benchmark_stop.zig");
const stream = @import("benchmark_stream.zig");
const Allocator = std.mem.Allocator;

const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const draft_revision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const gib = machine_mod.gib;
const entries = [_]registry.Entry{
    .{ .title = "Fixture", .format = "4-bit", .repo = "Org/Model", .revision = revision, .drafter = "Org/Draft", .drafter_revision = draft_revision, .download = 5 * gib, .metal = true, .cuda = true, .m5 = .drafts, .recipe = "fixture" },
};

const Run = struct {
    a: Allocator,
    env: std.process.Environ.Map,
    dir: []const u8,
    root: []const u8,
    out: std.Io.Writer.Allocating,
    err: std.Io.Writer.Allocating,
    stdin: std.Io.Reader = .fixed(""),
    input: []const u8 = "",
    interactive: bool = false,
    self_exe: ?[]const u8 = null,
    machine: machine_mod.Machine = .{ .backend = .metal, .memory = 96 * gib, .gpu_limit = 64 * gib },
    pull_code: u8 = 0,
    cancel_in_pull: bool = false,
    pulls: std.ArrayList([]const u8) = .empty,

    fn init(a: Allocator, tmp: *std.testing.TmpDir) !Run {
        var r: Run = .{ .a = a, .env = .init(a), .dir = try fake.tmpPath(a, tmp), .root = undefined, .out = .init(a), .err = .init(a) };
        r.machine.setChip("Apple M5 Max");
        r.root = try r.path("hub");
        try r.env.put("PATH", "/usr/bin:/bin");
        try r.env.put("HOME", r.dir);
        try r.env.put("TMPDIR", r.dir);
        try r.env.put("HF_HUB_CACHE", r.root);
        return r;
    }

    fn path(r: *const Run, name: []const u8) ![]const u8 {
        return std.fs.path.join(r.a, &.{ r.dir, name });
    }

    fn command(r: *Run, argv: []const []const u8) !u8 {
        r.out.clearRetainingCapacity();
        r.err.clearRetainingCapacity();
        r.stdin = .fixed(r.input);
        const ctx: start.Context = .{
            .a = r.a,
            .io = std.testing.io,
            .environ = &r.env,
            .out = &r.out.writer,
            .err = &r.err.writer,
            .input = &r.stdin,
            .interactive = r.interactive,
            .self_exe = r.self_exe,
            .signals = r.cancel_in_pull,
            .given = r.machine,
            .registry = &entries,
            .override = r.root,
            .pull = .{ .ctx = r, .run = fakePull },
        };
        return start.command(&ctx, argv);
    }

    fn fakePull(ctx: *anyopaque, spec: []const u8) !u8 {
        const r: *Run = @ptrCast(@alignCast(ctx));
        if (r.cancel_in_pull) {
            var action: std.posix.Sigaction = undefined;
            std.posix.sigaction(.INT, null, &action);
            if (action.handler.handler == std.posix.SIG.DFL) return error.DownloadSignalNotInstalled;
            stop.request(.INT);
        }
        try r.pulls.append(r.a, spec);
        try r.out.writer.print("stored {s} at {s}/private-snapshot\n", .{ spec, r.root });
        const model = std.mem.eql(u8, spec, "Org/Model@" ++ revision);
        const repo = if (model) "models--Org--Model" else "models--Org--Draft";
        const rev = if (model) revision else draft_revision;
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ r.root, repo, "snapshots", rev }));
        return if (r.pull_code != 0 and r.pulls.items.len >= 2) r.pull_code else 0;
    }

    fn cache(r: *Run) !void {
        const repo = try std.fs.path.join(r.a, &.{ r.root, "models--Org--Model" });
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ repo, "refs" }));
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ repo, "snapshots", revision }));
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ repo, "refs/main" }), .data = revision });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ repo, "snapshots", revision, "config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}}" });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ repo, "snapshots", revision, "model.safetensors" }), .data = "fixture" });
        const draft = try std.fs.path.join(r.a, &.{ r.root, "models--Org--Draft" });
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ draft, "refs" }));
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ draft, "snapshots", draft_revision }));
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ draft, "refs/main" }), .data = draft_revision });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ draft, "snapshots", draft_revision, "config.json" }), .data = "{\"model_type\": \"qwen4_exp_draft\"}" });
    }

    fn standIn(r: *Run, port: u16) !void {
        return standInAt(r, port, "serve-stand-in");
    }

    fn standInAt(r: *Run, port: u16, name: []const u8) !void {
        const script = try r.path(name);
        const text = try std.fmt.allocPrint(r.a, "#!/bin/sh\necho $$ > {s}\nprintf '%s\\n' \"$@\" > {s}\nprintf '%s' \"$TENSORFOLD_API_KEY\" > {s}\nprintf '%s' \"$START_TEST_UNRELATED\" > {s}\nprintf '[tensorfold] serving Model at http://127.0.0.1:{d}/v1 (x)\\n'\nif [ \"$START_TEST_SHORT_LIVED\" = 1 ]; then exec sleep 1; else exec sleep 30; fi\n", .{ try r.path("pid"), try r.path("args"), try r.path("child-key"), try r.path("unrelated"), port });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = script, .data = text, .flags = .{ .permissions = .fromMode(0o755) } });
        r.self_exe = script;
    }

    fn cudaStandIn(r: *Run, port: u16) !void {
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try r.path("bin"));
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try r.path("native/bin"));
        try standInAt(r, port, "native/bin/tensorfold-native");
        const low_level = try r.path("bin/tensorfold");
        const text = try std.fmt.allocPrint(r.a, "#!/bin/sh\ntouch {s}\nexit 2\n", .{try r.path("low-level-called")});
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = low_level, .data = text, .flags = .{ .permissions = .fromMode(0o755) } });
        r.self_exe = low_level;
    }

    fn childGone(r: *Run) bool {
        const text = std.Io.Dir.cwd().readFileAlloc(std.testing.io, r.path("pid") catch return false, r.a, .limited(64)) catch return false;
        const pid = std.fmt.parseInt(std.posix.pid_t, std.mem.trim(u8, text, " \n"), 10) catch return false;
        stop.releaseChild(pid);
        std.posix.kill(pid, @fromBackingInt(0)) catch |e| return e == error.ProcessNotFound;
        return false;
    }
};

fn events(a: Allocator, content: []const u8) ![]const []const fake.Write {
    const head = try fake.headOf(a, "200 OK", "text/event-stream");
    const body = try fake.event(a, try std.fmt.allocPrint(a, "{{\"choices\": [{{\"index\": 0, \"delta\": {{\"content\": \"{s}\"}}, \"finish_reason\": \"stop\"}}]}}", .{content}));
    const writes = try a.dupe(fake.Write, &.{.{ .bytes = try std.mem.concat(a, u8, &.{ head, body, try fake.event(a, "[DONE]") }) }});
    return a.dupe([]const fake.Write, &.{writes});
}

fn world(r: *Run, replies: []const []const fake.Write) !*fake.Fake {
    return worldKeyed(r, replies, try r.path("child-key"));
}

fn worldKeyed(r: *Run, replies: []const []const fake.Write, key_file: []const u8) !*fake.Fake {
    return fake.Fake.startKeyed(std.testing.allocator, try r.a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/health", .replies = try r.a.dupe([]const fake.Write, &.{try fake.jsonReply(r.a, "200 OK", "{\"status\": \"ok\", \"warming\": false}")}) },
        .{ .method = "GET", .path = "/v1/models", .replies = try r.a.dupe([]const fake.Write, &.{try fake.jsonReply(r.a, "200 OK", "{\"data\": [{\"id\": \"Model\"}]}")}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = replies },
    }), key_file);
}

test "start CLI: a missing fit or a refused download never invokes pull or serve" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.env.put("TENSORFOLD_API_KEY", "fixture-consent-key");
    r.machine.gpu_limit = gib;
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{}));
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "no catalog model that this machine can start automatically") != null);
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);

    r.machine.gpu_limit = 64 * gib;
    try std.testing.expectEqual(@as(u8, 2), try r.command(&.{}));
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "pass --yes to consent in a non-interactive run; nothing was downloaded") != null);
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
    r.interactive = true;
    r.input = "no\n";
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{}));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "download declined; nothing was downloaded") != null);
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
}

test "start CLI: explicit consent shows sizes and pulls both pinned specs before a failed pull stops" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.env.put("TENSORFOLD_API_KEY", "fixture-consent-key");
    r.pull_code = 1;
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{"--yes"}));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "5.0 GiB (5368709120 bytes) of model weights") != null);
    try std.testing.expectEqual(@as(usize, 2), r.pulls.items.len);
    try std.testing.expectEqualStrings("Org/Model@" ++ revision, r.pulls.items[0]);
    try std.testing.expectEqualStrings("Org/Draft@" ++ draft_revision, r.pulls.items[1]);
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "pull Org/Draft@" ++ draft_revision ++ " failed") != null);
    for ([_][]const u8{ "Org/Model@" ++ revision, "Org/Draft@" ++ draft_revision }) |spec| {
        try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "Pulling") != null);
        try std.testing.expect(std.mem.indexOf(u8, r.out.written(), spec) != null);
    }
    for ([_][]const u8{ r.out.written(), r.err.written() }) |text|
        try std.testing.expect(std.mem.indexOf(u8, text, r.root) == null);

    r.pulls.clearRetainingCapacity();
    r.interactive = true;
    r.input = "yes\n";
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{}));
    try std.testing.expectEqual(@as(usize, 2), r.pulls.items.len);
}

test "start CLI: Ctrl-C during a deferred download stops before a drafter or server" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    r.cancel_in_pull = true;
    stop.reset();
    defer stop.reset();
    try std.testing.expectEqual(@as(u8, 130), try r.command(&.{ "--yes", "--check" }));
    try std.testing.expectEqual(@as(usize, 1), r.pulls.items.len);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(std.testing.io, try r.path("pid"), .{}));
}

test "start CLI: an unpublished snapshot is not complete and cannot skip its pull" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.cache();
    const repo = try std.fs.path.join(a, &.{ r.root, "models--Org--Model" });
    try std.Io.Dir.cwd().deleteFile(std.testing.io, try std.fs.path.join(a, &.{ repo, "refs/main" }));
    const f = try world(&r, try events(a, "Ready."));
    defer f.stop();
    try r.standIn(f.port);
    try std.testing.expectEqual(@as(u8, 2), try r.command(&.{"--check"}));
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "nothing was downloaded") != null);
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
}

test "start CLI: a cached model answers once, prints no key and --check reaps its serve child" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.env.put("TENSORFOLD_API_KEY", "user-key-must-not-leak");
    try r.cache();
    const f = try world(&r, try events(a, "Ready."));
    defer f.stop();
    try r.standIn(f.port);
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{"--check"}));
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "First reply:\nReady.\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "first-reply check passed; server stopped") != null);
    try std.testing.expect(r.childGone());
    const key = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("child-key"), a, .limited(256));
    try std.testing.expect(key.len == 64 and !std.mem.eql(u8, key, "user-key-must-not-leak"));
    for ([_][]const u8{ r.out.written(), r.err.written() }) |text| try std.testing.expect(std.mem.indexOf(u8, text, key) == null);
    const args = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("args"), a, .limited(4096));
    const want = try std.fmt.allocPrint(a, "serve\n{s}/models--Org--Model/snapshots/{s}\n--host\n127.0.0.1\n--port\n0\n--name\n{s}\n--name\nModel\n--drafter\n{s}/models--Org--Draft/snapshots/{s}\n", .{ r.root, revision, revision, r.root, draft_revision });
    try std.testing.expect(std.mem.indexOf(u8, args, want) != null);
    for ([_][]const u8{ key, "user-key-must-not-leak" }) |secret|
        for ([_][]const u8{ r.out.written(), r.err.written(), args }) |text| try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
}

test "start CLI: a failed first reply is fatal and still reaps its serve child" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.cache();
    const refusal = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "500 Internal Server Error", "{\"error\": {\"message\": \"generation failed\", \"type\": \"server_error\"}}")});
    const f = try world(&r, refusal);
    defer f.stop();
    try r.standIn(f.port);
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{"--check"}));
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "first reply failed: HTTP 500: generation failed") != null);
    try std.testing.expect(r.childGone());
}

test "start CLI: CUDA spawns its native sibling instead of the unsupported low-level serve child" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.cache();
    const f = try world(&r, try events(a, "Ready."));
    defer f.stop();
    try r.cudaStandIn(f.port);
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{"--check"}));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "first-reply check passed") != null);
    try std.testing.expect(r.childGone());
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(std.testing.io, try r.path("low-level-called"), .{}));
}

test "start CLI: pre-M5 Metal starts Gemma-style rows without resolving pulling or passing a drafter" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    r.machine.setChip("Apple M4 Max");
    const f = try world(&r, try events(a, "Ready."));
    defer f.stop();
    try r.standIn(f.port);
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--yes", "--check" }));
    try std.testing.expectEqual(@as(usize, 1), r.pulls.items.len);
    try std.testing.expectEqualStrings("Org/Model@" ++ revision, r.pulls.items[0]);
    const args = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("args"), a, .limited(4096));
    try std.testing.expect(std.mem.indexOf(u8, args, "--drafter") == null);
    try std.testing.expect(std.mem.indexOf(u8, args, "Org/Draft") == null);
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "Drafter download") == null);
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "Pulling Org/Model@" ++ revision) != null);
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), r.root) == null);
}

test "start CLI: CUDA still resolves pulls and passes the selected drafter" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    r.machine.backend = .cuda;
    r.machine.setChip("NVIDIA RTX 4090");
    const f = try world(&r, try events(a, "Ready."));
    defer f.stop();
    try r.standIn(f.port);
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--yes", "--check" }));
    try std.testing.expectEqual(@as(usize, 2), r.pulls.items.len);
    const args = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("args"), a, .limited(4096));
    const drafter = try std.fs.path.join(a, &.{ r.root, "models--Org--Draft", "snapshots", draft_revision });
    const want_drafter = try std.fmt.allocPrint(a, "--drafter\n{s}", .{drafter});
    try std.testing.expect(std.mem.indexOf(u8, args, want_drafter) != null);
}

test "start CLI: persistent mode requires a valid environment credential before model work" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    r.machine.gpu_limit = gib;
    try std.testing.expectEqual(@as(u8, 2), try r.command(&.{}));
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "persistent start requires") != null);
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
    for ([_][]const u8{ "   ", ",", " , ", "\t,\r\n", "" }) |key| {
        try r.env.put("TENSORFOLD_API_KEY", key);
        try std.testing.expectEqual(@as(u8, 2), try r.command(&.{}));
        try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "persistent start requires a nonempty TENSORFOLD_API_KEY entry") != null);
        try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
        try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "Selected ") == null);
    }
    for ([_][]const u8{ "\xc2\xa0", "good-key,bad key", "good-key,\xc2\xa0" }) |key| {
        try r.env.put("TENSORFOLD_API_KEY", key);
        try std.testing.expectEqual(@as(u8, 2), try r.command(&.{}));
        try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "expected keys without spaces or control characters") != null);
        try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
        try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "Selected ") == null);
    }
}

test "start CLI: persistent caller keys preserve the child list and send only the first key" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const caller_key = "fixture-persistent-caller-key";
    const second_key = "fixture-second-caller-key";
    const key_list = " , " ++ caller_key ++ "," ++ second_key;
    try r.env.put("TENSORFOLD_API_KEY", key_list);
    try r.env.put("START_TEST_UNRELATED", "caller-value-retained");
    try r.env.put("START_TEST_SHORT_LIVED", "1");
    try r.cache();
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try r.path("credential"), .data = caller_key });
    const f = try worldKeyed(&r, try events(a, "Ready."), try r.path("credential"));
    defer f.stop();
    try r.standIn(f.port);

    var client: std.http.Client = .{ .allocator = a, .io = std.testing.io };
    defer client.deinit();
    const unauthenticated = stream.fetch(a, std.testing.io, &client, .{ .url = try f.url(a, "/v1/models"), .limit = 4096, .timeout_s = 1, .token = null });
    try std.testing.expectEqual(.none, unauthenticated.failed);
    try std.testing.expectEqual(@as(u16, 401), unauthenticated.status);

    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--startup-timeout", "2" }));
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "the model server exited after its successful first reply") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "Serving Model at") != null);
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
    try std.testing.expect(r.childGone());
    const child_key = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("child-key"), a, .limited(256));
    try std.testing.expectEqualStrings(key_list, child_key);
    const unrelated = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("unrelated"), a, .limited(256));
    try std.testing.expect(std.mem.eql(u8, unrelated, "caller-value-retained"));
    const bearer = try std.fmt.allocPrint(a, "Bearer {s}", .{caller_key});
    try std.testing.expect(f.count() > 0 and std.mem.eql(u8, f.header(f.count() - 1, "authorization") orelse "", bearer));
    try std.testing.expectEqual(@as(usize, 1), f.refusals());
    const args = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("args"), a, .limited(4096));
    for ([_][]const u8{ caller_key, second_key, key_list }) |secret|
        for ([_][]const u8{ args, r.out.written(), r.err.written() }) |text| try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
}
