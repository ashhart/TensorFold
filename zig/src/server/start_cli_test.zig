//! `tensorfold start` against fake devices, registries, checkpoints and HTTP servers.
const std = @import("std");
const start = @import("start_cli.zig");
const registry = @import("checkpoint_cli").start;
const machine_mod = @import("sys").machine;
const fake = @import("benchmark_fake.zig");
const stop = @import("benchmark_stop.zig");
const Allocator = std.mem.Allocator;

const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const draft_revision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const gib = machine_mod.gib;
const entries = [_]registry.Entry{
    .{ .title = "Fixture", .format = "4-bit", .repo = "Org/Model", .revision = revision, .drafter = "Org/Draft", .drafter_revision = draft_revision, .download = 5 * gib, .metal = true, .cuda = true, .recipe = "fixture" },
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
    pulls: std.ArrayList([]const u8) = .empty,

    fn init(a: Allocator, tmp: *std.testing.TmpDir) !Run {
        var r: Run = .{ .a = a, .env = .init(a), .dir = try fake.tmpPath(a, tmp), .root = undefined, .out = .init(a), .err = .init(a) };
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
            .given = r.machine,
            .registry = &entries,
            .override = r.root,
            .pull = .{ .ctx = r, .run = fakePull },
        };
        return start.command(&ctx, argv);
    }

    fn fakePull(ctx: *anyopaque, spec: []const u8) !u8 {
        const r: *Run = @ptrCast(@alignCast(ctx));
        try r.pulls.append(r.a, spec);
        return if (r.pull_code != 0 and r.pulls.items.len >= 2) r.pull_code else 0;
    }

    fn cache(r: *Run) !void {
        const repo = try std.fs.path.join(r.a, &.{ r.root, "models--Org--Model" });
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ repo, "refs" }));
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ repo, "snapshots", revision }));
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ repo, "refs/main" }), .data = "cccccccccccccccccccccccccccccccccccccccc" });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ repo, "snapshots", revision, "config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}}" });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ repo, "snapshots", revision, "model.safetensors" }), .data = "fixture" });
        const draft = try std.fs.path.join(r.a, &.{ r.root, "models--Org--Draft" });
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ draft, "refs" }));
        try std.Io.Dir.cwd().createDirPath(std.testing.io, try std.fs.path.join(r.a, &.{ draft, "snapshots", draft_revision }));
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ draft, "refs/main" }), .data = "dddddddddddddddddddddddddddddddddddddddd" });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ draft, "snapshots", draft_revision, "config.json" }), .data = "{\"model_type\": \"qwen4_exp_draft\"}" });
    }

    fn standIn(r: *Run, port: u16) !void {
        const script = try r.path("serve-stand-in");
        const text = try std.fmt.allocPrint(r.a, "#!/bin/sh\necho $$ > {s}\nprintf '%s\\n' \"$@\" > {s}\nprintf '%s' \"$TENSORFOLD_API_KEY\" > {s}\nprintf '[tensorfold] serving Model at http://127.0.0.1:{d}/v1 (x)\\n'\nexec sleep 30\n", .{ try r.path("pid"), try r.path("args"), try r.path("key"), port });
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = script, .data = text, .flags = .{ .permissions = .fromMode(0o755) } });
        r.self_exe = script;
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
    return fake.Fake.startKeyed(std.testing.allocator, try r.a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/health", .replies = try r.a.dupe([]const fake.Write, &.{try fake.jsonReply(r.a, "200 OK", "{\"status\": \"ok\", \"warming\": false}")}) },
        .{ .method = "GET", .path = "/v1/models", .replies = try r.a.dupe([]const fake.Write, &.{try fake.jsonReply(r.a, "200 OK", "{\"data\": [{\"id\": \"Model\"}]}")}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = replies },
    }), try r.path("key"));
}

test "start CLI: a missing fit or a refused download never invokes pull or serve" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
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
    r.pull_code = 1;
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{"--yes"}));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "5.0 GiB (5368709120 bytes) of model weights") != null);
    try std.testing.expectEqual(@as(usize, 2), r.pulls.items.len);
    try std.testing.expectEqualStrings("Org/Model@" ++ revision, r.pulls.items[0]);
    try std.testing.expectEqualStrings("Org/Draft@" ++ draft_revision, r.pulls.items[1]);
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "pull Org/Draft@" ++ draft_revision ++ " failed") != null);

    r.pulls.clearRetainingCapacity();
    r.interactive = true;
    r.input = "yes\n";
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{}));
    try std.testing.expectEqual(@as(usize, 2), r.pulls.items.len);
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
    const key = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("key"), a, .limited(256));
    try std.testing.expect(key.len == 64 and !std.mem.eql(u8, key, "user-key-must-not-leak"));
    for ([_][]const u8{ r.out.written(), r.err.written() }) |text| try std.testing.expect(std.mem.indexOf(u8, text, key) == null);
    const args = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("args"), a, .limited(4096));
    const want = try std.fmt.allocPrint(a, "serve\n{s}/models--Org--Model/snapshots/{s}\n--host\n127.0.0.1\n--port\n0\n--name\n{s}\n--name\nModel\n--drafter\n{s}/models--Org--Draft/snapshots/{s}\n", .{ r.root, revision, revision, r.root, draft_revision });
    try std.testing.expect(std.mem.indexOf(u8, args, want) != null);
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
