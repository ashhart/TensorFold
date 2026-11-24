//! KR67 managed-log containment through the start command's local fake server and child.
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
    .{ .title = "Fixture", .format = "4-bit", .repo = "Org/Model", .revision = revision, .drafter = "Org/Draft", .drafter_revision = draft_revision, .download = 5 * gib, .metal = true, .cuda = true, .m5 = .drafts, .recipe = "fixture" },
};
const marker_input = "[redacted] %5Bredacted%5D %255Bredacted%255D redacted %72%65%64%61%63%74%65%64 %2572%2565%2564%2561%2563%2574%2565%2564";
const marker_secrets = [_][]const u8{
    "[redacted]",
    "redacted",
    "%5Bredacted%5D",
    "%255Bredacted%255D",
    "%72%65%64%61%63%74%65%64",
    "%2572%2565%2564%2561%2563%2574%2565%2564",
};

const Run = struct {
    a: Allocator,
    env: std.process.Environ.Map,
    dir: []const u8,
    root: []const u8,
    out: std.Io.Writer.Allocating,
    err: std.Io.Writer.Allocating,
    stdin: std.Io.Reader = .fixed(""),
    self_exe: ?[]const u8 = null,
    machine: machine_mod.Machine = .{ .backend = .metal, .memory = 96 * gib, .gpu_limit = 64 * gib },
    catalog: []const registry.Entry = &entries,
    pulls: std.ArrayList([]const u8) = .empty,

    fn init(a: Allocator, tmp: *std.testing.TmpDir) !Run {
        var r: Run = .{ .a = a, .env = .init(a), .dir = try fake.tmpPath(a, tmp), .root = undefined, .out = .init(a), .err = .init(a) };
        r.machine.setChip("Apple M5 Max");
        r.root = try r.path("hub");
        try r.env.put("PATH", "/usr/bin:/bin");
        for ([_][]const u8{ "TMPDIR", "HOME", "HF_HUB_CACHE" }) |name| try r.env.put(name, if (std.mem.eql(u8, name, "HF_HUB_CACHE")) r.root else r.dir);
        return r;
    }

    fn path(r: *const Run, name: []const u8) ![]const u8 {
        return std.fs.path.join(r.a, &.{ r.dir, name });
    }

    fn command(r: *Run) !u8 {
        r.out.clearRetainingCapacity();
        r.err.clearRetainingCapacity();
        const ctx: start.Context = .{
            .a = r.a,
            .io = std.testing.io,
            .environ = &r.env,
            .out = &r.out.writer,
            .err = &r.err.writer,
            .input = &r.stdin,
            .interactive = false,
            .self_exe = r.self_exe,
            .given = r.machine,
            .registry = r.catalog,
            .override = r.root,
            .pull = .{ .ctx = r, .run = fakePull },
        };
        return start.command(&ctx, &.{ "--startup-timeout", "2" });
    }

    fn fakePull(ctx: *anyopaque, spec: []const u8) !u8 {
        const r: *Run = @ptrCast(@alignCast(ctx));
        try r.pulls.append(r.a, spec);
        return 1;
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
        const script = try r.path("serve-stand-in");
        const text = try std.fmt.allocPrint(r.a, "#!/bin/sh\necho $$ > {s}\nprintf '%s\\n' \"$@\" > {s}\nprintf '[tensorfold] serving Model at http://127.0.0.1:{d}/v1 (x)\\n'\nprintf '%s\\n' \"$START_TEST_LOG_ENCODED\" >&2\nexec sleep 1\n", .{ try r.path("pid"), try r.path("args"), port });
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

fn ready(a: Allocator) ![]const []const fake.Write {
    const head = try fake.headOf(a, "200 OK", "text/event-stream");
    const body = try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"Ready.\"}, \"finish_reason\": \"stop\"}]}");
    const writes = try a.dupe(fake.Write, &.{.{ .bytes = try std.mem.concat(a, u8, &.{ head, body, try fake.event(a, "[DONE]") }) }});
    return a.dupe([]const fake.Write, &.{writes});
}

fn world(r: *Run) !*fake.Fake {
    return fake.Fake.startKeyed(std.testing.allocator, try r.a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/health", .replies = try r.a.dupe([]const fake.Write, &.{try fake.jsonReply(r.a, "200 OK", "{\"status\": \"ok\", \"warming\": false}")}) },
        .{ .method = "GET", .path = "/v1/models", .replies = try r.a.dupe([]const fake.Write, &.{try fake.jsonReply(r.a, "200 OK", "{\"data\": [{\"id\": \"Model\"}]}")}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = try ready(r.a) },
    }), try r.path("credential"));
}

fn managedRun(a: Allocator, tmp: *std.testing.TmpDir, keys: []const u8, first_key: []const u8, log: []const u8, secrets: []const []const u8) !void {
    var r = try Run.init(a, tmp);
    try r.env.put("TENSORFOLD_API_KEY", keys);
    try r.env.put("START_TEST_LOG_ENCODED", log);
    try r.cache();
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try r.path("credential"), .data = first_key });
    const f = try world(&r);
    defer f.stop();
    try r.standIn(f.port);
    try std.testing.expectEqual(@as(u8, 1), try r.command());
    try std.testing.expect(r.childGone());
    try std.testing.expectEqual(@as(usize, 0), r.pulls.items.len);
    for ([_][]const u8{ r.out.written(), r.err.written() }) |text|
        for (secrets) |secret| try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
}

fn controlMaterials(a: Allocator) !struct { key: []u8, raw: []const u8 } {
    const key = try a.alloc(u8, 256);
    for (key, 0..) |*c, i| c.* = 'a' + @as(u8, @intCast(i % 26));
    const printable = "123456789012345678901234567890123456789012345";
    return .{ .key = key, .raw = try std.mem.concat(a, u8, &.{ printable, key[0..128], "\x01", key[128..] }) };
}

test "start CLI: KR67 managed logs contain neither control-split keys nor their 255-byte prefixes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const control = try controlMaterials(a);
    const keys = try std.mem.concat(a, u8, &.{ "AB,", control.key });
    const log = try std.mem.concat(a, u8, &.{ "A\x01B\n", control.raw });
    try managedRun(a, &tmp, keys, "AB", log, &.{ "AB", control.key, control.key[0..255] });
}

test "start CLI: KR67 managed logs contain no exact marker, substring or encoded marker credentials" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try managedRun(a, &tmp, "[redacted],redacted", "[redacted]", marker_input, &marker_secrets);
}

test "start CLI: KR69 managed logs contain raw and nested already-truncated credentials" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const key = try a.alloc(u8, 256);
    for (key, 0..) |*c, i| c.* = 'a' + @as(u8, @intCast(i % 26));
    const keys = try std.mem.concat(a, u8, &.{ key, ",fixture-secret" });
    const log = try std.mem.concat(a, u8, &.{ key[0..255], "\n%66ixture-secre" });
    try managedRun(a, &tmp, keys, key, log, &.{ key[0..255], "fixture-secre", "%66ixture-secre" });
}

test "start CLI: KR71 managed logs contain raw and decoded exact two-thirds prefixes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const hostile = "abcdef\n%61%62%63%64%65%66\n%2561%2562%2563%2564%2565%2566";
    try managedRun(a, &tmp, "abcdefghi", "abcdefghi", hostile, &.{ "abcdef", "%61%62%63%64%65%66", "%2561%2562%2563%2564%2565%2566" });
}

test "start CLI: KR71 a short readiness credential prevents every first-reply POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.env.put("TENSORFOLD_API_KEY", "short");
    try r.env.put("START_TEST_LOG_ENCODED", "");
    try r.cache();
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try r.path("credential"), .data = "short" });
    const f = try world(&r);
    defer f.stop();
    try r.standIn(f.port);
    try std.testing.expectEqual(@as(u8, 1), try r.command());
    try std.testing.expect(r.childGone());
    var posts: usize = 0;
    for (0..f.count()) |i| posts += @intFromBool(std.mem.eql(u8, f.request(i).method, "POST"));
    try std.testing.expectEqual(@as(usize, 0), posts);
    for ([_][]const u8{ r.out.written(), r.err.written() }) |text| try std.testing.expect(std.mem.indexOf(u8, text, "short") == null);
}
