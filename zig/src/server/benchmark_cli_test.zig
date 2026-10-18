//! The command end to end against fakes: attached run, consent, login, publish, cancel and the checks.
const std = @import("std");
const platform = @import("sys");
const json = @import("json");
const suite = @import("benchmark_suite.zig");
const receipt = @import("benchmark_receipt.zig");
const benchmark = @import("benchmark.zig");
const stop = @import("benchmark_stop.zig");
const server = @import("benchmark_server.zig");
const fake = @import("benchmark_fake.zig");
const receipts = @import("benchmark_receipt_test.zig");
const check = @import("benchmark_check.zig");

pub const token = "synthetic-contributor-token-0001";
const now_ms: i64 = 1767225600000 + 60_000;

pub const Run = struct {
    a: std.mem.Allocator,
    env: std.process.Environ.Map,
    home: []const u8,
    out: std.Io.Writer.Allocating,
    err: std.Io.Writer.Allocating,
    stdin: std.Io.Reader,
    tty: bool = false,
    publication: benchmark.Publication = .{},
    families: []const @import("engine_api").Family = &.{.{ .model_type = "qwen3_5", .formats = &.{} }},
    hf_base: []const u8 = "",
    self_exe: ?[]const u8 = null,
    fd: ?std.posix.fd_t = null,
    io: ?std.Io = null,
    reader: ?*std.Io.Reader = null,

    pub fn init(a: std.mem.Allocator, tmp: *std.testing.TmpDir) !Run {
        return .{ .a = a, .env = .init(a), .home = try fake.tmpPath(a, tmp), .out = .init(a), .err = .init(a), .stdin = .fixed("") };
    }

    pub fn command(r: *Run, argv: []const []const u8) !u8 {
        r.out.clearRetainingCapacity();
        r.err.clearRetainingCapacity();
        const ctx: benchmark.Context = .{
            .a = r.a,
            .io = r.io orelse std.testing.io,
            .environ = &r.env,
            .home = r.home,
            .out = &r.out.writer,
            .err = &r.err.writer,
            .input = .{ .tty = r.tty, .reader = r.reader orelse &r.stdin, .fd = r.fd },
            .engines = .{ .version = "0.0.0-test", .families = r.families },
            .hf_base = r.hf_base,
            .self_exe = r.self_exe,
            .publication = r.publication,
        };
        return benchmark.command(&ctx, argv);
    }

    pub fn has(r: *Run, text: []const u8) bool {
        return std.mem.indexOf(u8, r.out.written(), text) != null or std.mem.indexOf(u8, r.err.written(), text) != null;
    }

    pub fn path(r: *const Run, name: []const u8) ![]const u8 {
        return std.fs.path.join(r.a, &.{ r.home, name });
    }
};

/// An attached server that streams 4 tokens per reply, a public checkpoint lookup and a receiver.
pub fn world(a: std.mem.Allocator, error_stream: bool, upload_status: []const u8) !*fake.Fake {
    const err_reply = try a.dupe(fake.Write, &.{.{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), try fake.event(a, "{\"error\": {\"message\": \"private upstream details must not be printed\"}}") }) }});
    const ack = if (std.mem.startsWith(u8, upload_status, "202")) "{\"id\": \"synthetic-submission\", \"status\": \"pending\"}" else "{\"error\": {\"message\": \"private upstream details must not be printed\"}}";
    const routes = try a.dupe(fake.Route, &.{
        .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{if (error_stream) err_reply else try fake.reply(a, false, 4, 15)}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{if (error_stream) err_reply else try fake.reply(a, true, 4, 15)}) },
        .{ .method = "GET", .path = "/api/models/example-org/test-model", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"private\": false}")}) },
        .{ .method = "GET", .path = "/me", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"display_name\": \"example-contributor\"}")}) },
        .{ .method = "POST", .path = "/submissions", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, upload_status, ack)}) },
    });
    return fake.Fake.start(std.testing.allocator, routes);
}

pub fn runArgs(r: *Run, base: []const u8, output: []const u8) ![]const []const u8 {
    return backendArgs(r, "cuda", base, output);
}

fn backendArgs(r: *Run, backend: []const u8, base: []const u8, output: []const u8) ![]const []const u8 {
    return r.a.dupe([]const u8, &.{ "example-org/test-model", "--server", base, "--backend", backend, "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", output });
}

pub fn posts(f: *fake.Fake, route: []const u8) usize {
    var n: usize = 0;
    for (0..f.count()) |i| n += @intFromBool(std.mem.eql(u8, f.request(i).path, route));
    return n;
}

test "an attached run saves an unranked receipt that publishes with explicit consent and the token" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    r.hf_base = try f.url(a, "/api/models/");
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 0), try r.command(try runArgs(&r, try f.url(a, ""), out)));
    try std.testing.expect(r.has("unranked") and r.has("Receipt saved") and r.has("code T=0 repeat 1/1: ok"));
    try std.testing.expectEqual(@as(usize, 5), f.count()); // the manifest it asked for first, then the four replies
    try std.testing.expectEqualStrings("/v1/benchmark/manifest", f.request(0).path);
    for (0..5) |i| try std.testing.expect(std.mem.startsWith(u8, f.request(i).path, "/v1/"));
    var problem: []const u8 = "";
    const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem)).?;
    try std.testing.expect(v.get("samples").?.array.len == 2 and !v.get("settings").?.get("managed").?.bool and v.get("settings").?.get("rank_count").? == .null);
    try std.testing.expectEqualStrings("unavailable", v.get("hardware").?.get("source").?.string);
    try std.testing.expectEqualStrings("unknown", v.get("runtime").?.get("tensorfold_version").?.string);
    try std.testing.expectEqualStrings("unknown", v.get("runtime").?.get("python_version").?.string);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, out, a, .limited(1 << 20));
    try std.testing.expect(std.mem.indexOf(u8, text, "synthetic") == null and std.mem.indexOf(u8, text, "127.0.0.1") == null);
    _ = try receipts.asLegacy(a, v);
    try std.testing.expect(try receipt.save(a, std.testing.io, v, out, now_ms + 100000000000, &problem));
    try r.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "publish", out, "--yes", "--upload-url", try f.url(a, "/") }));
    try std.testing.expect(r.has("Public upload preview") and r.has("Submission synthetic-submission: pending") and !r.has(token));
    try std.testing.expectEqual(@as(usize, 1), posts(f, "/submissions"));
    const i = f.count() - 1;
    try std.testing.expectEqualStrings("Bearer " ++ token, f.header(i, "authorization").?);
    try std.testing.expectEqualStrings(try receipt.canonical(a, v), f.request(i).body);
}

test "without --yes a pipe is refused before any upload, and a terminal's no keeps the receipt local" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    r.hf_base = try f.url(a, "/api/models/");
    try r.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    const out = try r.path("run.json");
    var problem: []const u8 = "";
    try std.testing.expect(try receipt.save(a, std.testing.io, try receipts.legacy(a), out, now_ms, &problem));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "publish", out, "--upload-url", try f.url(a, "") }));
    try std.testing.expect(r.has("Public upload preview") and r.has("explicit consent") and !r.has(token));
    r.tty = true;
    r.stdin = .fixed("n\n");
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "publish", out, "--upload-url", try f.url(a, "") }));
    try std.testing.expect(r.has("Upload cancelled"));
    try std.testing.expectEqual(@as(usize, 0), posts(f, "/submissions"));
}

test "login reads the token from stdin and never prints it; a refused upload shows no server text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "401 Unauthorized");
    defer f.stop();
    r.hf_base = try f.url(a, "/api/models/");
    r.stdin = .fixed(token ++ "\n");
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "login", "--token-stdin", "--upload-url", try f.url(a, "") }));
    try std.testing.expect(r.has("Benchmark uploads enabled for example-contributor.") and !r.has(token));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "login", "--upload-url", try f.url(a, "") }));
    try std.testing.expect(r.has("--token-stdin"));
    const out = try r.path("run.json");
    var problem: []const u8 = "";
    try std.testing.expect(try receipt.save(a, std.testing.io, try receipts.legacy(a), out, now_ms, &problem));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "publish", out, "--yes", "--upload-url", try f.url(a, "") }));
    try std.testing.expect(r.has("Upload token was refused") and !r.has("private upstream") and !r.has(token));
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{"logout"}));
    try std.testing.expect(r.has("Local benchmark upload credentials removed."));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, try r.path(".tensorfold/benchmarks/auth.json"), .{}));
}

test "failed warm-ups and repeats stay in the receipt with fixed codes and exit 2" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, true, "202 Accepted");
    defer f.stop();
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 2), try r.command(try runArgs(&r, try f.url(a, "/v1/"), out)));
    var problem: []const u8 = "";
    const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem)).?;
    const samples = v.get("samples").?.array;
    try std.testing.expectEqual(@as(usize, 4), samples.len);
    for (samples, [_]i64{ -1, 0, -1, 0 }) |s, rep| {
        try std.testing.expectEqual(rep, s.get("repeat").?.int64().?);
        try std.testing.expectEqualStrings("server_error", s.get("error_code").?.string);
    }
    try std.testing.expect(!r.has("private upstream"));
}

test "a cancelled local run saves its receipt, exits 130 and never publishes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    r.hf_base = try f.url(a, "/api/models/");
    try r.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    const out = try r.path("run.json");
    stop.reset();
    defer stop.reset();
    stop.request(.INT);
    const argv = try runArgs(&r, try f.url(a, ""), out);
    try std.testing.expectEqual(@as(u8, 130), try r.command(argv));
    try std.testing.expect(f.count() == 0 and !r.has("Public upload preview"));
    stop.reset();
    var problem: []const u8 = "";
    const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem)).?;
    try std.testing.expectEqualStrings("cancelled", v.get("samples").?.array[0].get("error_code").?.string);
}

test "attach and managed checks refuse before any request, and parse errors exit 2" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "example-org/test-model", "--server", try f.url(a, ""), "--output", out }));
    try std.testing.expect(r.has("explicit --backend mlx, cuda or rocm") and f.count() == 0);
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "example-org/test-model", "--server", "file:///tmp/x", "--backend", "mlx" }));
    try std.testing.expect(r.has("Server URL must be HTTP(S)"));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "example-org/test-model", "--model-id", "x" }));
    try std.testing.expect(r.has("--model-id applies only to an attached server"));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "example-org/test-model", "--startup-timeout", "3601" }));
    try std.testing.expect(r.has("at most 3600 seconds"));
    const ckpt = try r.path("gpt");
    try std.Io.Dir.cwd().createDirPath(std.testing.io, ckpt);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ ckpt, "config.json" }), .data = "{\"model_type\": \"gpt2\"}" });
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ckpt}));
    try std.testing.expect(r.has("This build does not serve model_type gpt2"));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{"example-org/not-cached"}));
    try std.testing.expect(r.has("run tensorfold pull example-org/not-cached first"));
    try std.testing.expectEqual(@as(u8, 2), try r.command(&.{ "example-org/test-model", "--download" }));
    try std.testing.expect(r.has("usage: tensorfold benchmark") and r.has("run tensorfold pull MODEL first"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{"suites"}));
    try std.testing.expectEqualStrings(suite.suites_text, r.out.written());
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{"-h"}));
    try std.testing.expect(r.has("--token-stdin"));
    try std.testing.expectEqual(@as(usize, 0), f.count());
}

test "a checkpoint that is not public stops publishing before the preview, with one unauthenticated lookup" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "GET", .path = "/api/models/example-org/test-model", .replies = &.{&.{.{ .bytes = "HTTP/1.1 404 Not Found\r\nContent-Length: 15\r\nConnection: close\r\n\r\nprivate details" }}} }});
    defer f.stop();
    r.hf_base = try f.url(a, "/api/models/");
    try r.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    const out = try r.path("run.json");
    var problem: []const u8 = "";
    try std.testing.expect(try receipt.save(a, std.testing.io, try receipts.legacy(a), out, now_ms, &problem));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "publish", out, "--yes", "--upload-url", "https://example.test/api" }));
    try std.testing.expect(r.has("publicly accessible") and !r.has("Public upload preview") and !r.has("private details") and !r.has(token));
    try std.testing.expect(f.count() == 1 and f.header(0, "authorization") == null);
}

/// A serve stand-in: a script that records its pid and arguments, prints serve's line with the fake's port and sleeps.
pub fn standIn(r: *Run, port: u16) ![]const u8 {
    const script = try r.path("serve-stand-in");
    const text = try std.fmt.allocPrint(r.a, "#!/bin/sh\necho $$ > {s}\nprintf '%s\\n' \"$@\" > {s}\nprintf '[tensorfold] serving .tensorfold-benchmark at http://127.0.0.1:%s/v1 (x)\\n' {d}\nexec sleep 30\n", .{ try r.path("pid"), try r.path("args"), port });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = script, .data = text, .flags = .{ .permissions = .fromMode(0o755) } });
    const ckpt = try r.path("private-checkpoint");
    try std.Io.Dir.cwd().createDirPath(std.testing.io, ckpt);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(r.a, &.{ ckpt, "config.json" }), .data = "{\"model_type\": \"qwen3_5\", \"quantization\": {\"bits\": 4, \"group_size\": 64}}" });
    try r.env.put("PATH", "/usr/bin:/bin");
    return script;
}

pub fn managedWorld(a: std.mem.Allocator, chat_trickles: bool) !*fake.Fake {
    var trickle: std.ArrayList(fake.Write) = .empty;
    try trickle.append(a, .{ .bytes = try fake.headOf(a, "200 OK", "text/event-stream"), .pause_ms = 4 });
    for (0..300) |_| try trickle.append(a, .{ .bytes = "x", .pause_ms = 4 });
    return fake.Fake.start(std.testing.allocator, try a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/health", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"status\": \"ok\", \"warming\": false}")}) },
        .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"data\": [{\"id\": \".tensorfold-benchmark\"}]}")}) },
        .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, false, 4, 15)}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{if (chat_trickles) trickle.items else try fake.reply(a, true, 4, 15)}) },
    }));
}

pub fn childGone(r: *Run) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("pid"), r.a, .limited(64));
    const pid = try std.fmt.parseInt(std.posix.pid_t, std.mem.trim(u8, text, " \n"), 10);
    std.posix.kill(pid, @fromBackingInt(0)) catch |e| return e == error.ProcessNotFound;
    return false;
}

test "a managed run starts its own serve, measures through it, stops it and saves a native receipt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try managedWorld(a, false);
    defer f.stop();
    r.self_exe = try standIn(&r, f.port);
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ try r.path("private-checkpoint"), "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", out }));
    try std.testing.expect(r.has("| local model") and !r.has("unranked") and try childGone(&r));
    var problem: []const u8 = "";
    const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem)).?;
    const rt = v.get("runtime").?;
    try std.testing.expect(std.mem.eql(u8, rt.get("python_version").?.string, "none") and std.mem.eql(u8, rt.get("tensorfold_version").?.string, "0.0.0-test"));
    try std.testing.expectEqualStrings(server.buildBackend(), rt.get("backend").?.string);
    const m = v.get("model").?;
    try std.testing.expect(m.get("local").?.bool and m.get("repo_id").? == .null and std.mem.eql(u8, m.get("quantization").?.string, "mlx-4bit-g64"));
    try std.testing.expect(v.get("settings").?.get("managed").?.bool and std.mem.eql(u8, v.get("settings").?.get("rank_count").?.int, "1"));
    try std.testing.expect(std.mem.indexOf(u8, try std.Io.Dir.cwd().readFileAlloc(std.testing.io, out, a, .limited(1 << 20)), "private-checkpoint") == null);
    const again = server.lockRun(std.testing.io, try r.path(".tensorfold/benchmarks"), &problem) orelse return error.TestUnexpectedResult;
    again.close(std.testing.io);
}

test "Ctrl-C mid-run saves the receipt and exits 130; SIGTERM and SIGHUP stop the child and save nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try managedWorld(a, true);
    defer f.stop();
    r.self_exe = try standIn(&r, f.port);
    defer stop.reset();
    for ([_]std.posix.SIG{ .INT, .TERM, .HUP }, [_]u8{ 130, 143, 129 }) |sig, code| {
        stop.reset();
        const out = try r.path(@tagName(sig));
        const seen = posts(f, "/v1/chat/completions");
        const got = blk: {
            const t = try std.Thread.spawn(.{}, struct {
                fn onceChatStarts(fk: *fake.Fake, before: usize, s: std.posix.SIG) void {
                    var waited: u32 = 0;
                    while (posts(fk, "/v1/chat/completions") == before and waited < 10_000) : (waited += 5) fake.nap(5);
                    stop.request(s);
                }
            }.onceChatStarts, .{ f, seen, sig });
            defer t.join();
            break :blk try r.command(&.{ try r.path("private-checkpoint"), "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", out });
        };
        try std.testing.expectEqual(code, got);
        try std.testing.expect(try childGone(&r));
        var problem: []const u8 = "";
        const saved = try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem);
        if (sig == .INT) {
            const samples = saved.?.get("samples").?.array;
            try std.testing.expectEqualStrings("cancelled", samples[samples.len - 1].get("error_code").?.string);
        } else try std.testing.expect(saved == null);
    }
}

test "a drafter directory named like owner/name is never recorded as a public drafter" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try managedWorld(a, false);
    defer f.stop();
    r.self_exe = try standIn(&r, f.port);
    try std.Io.Dir.cwd().createDirPath(io, try r.path("example-org/test-draft"));
    var back = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer back.close(io);
    var home = try std.Io.Dir.cwd().openDir(io, r.home, .{});
    defer home.close(io);
    try std.process.setCurrentDir(io, home);
    defer std.process.setCurrentDir(io, back) catch {};
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ try r.path("private-checkpoint"), "--drafter", "example-org/test-draft", "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", out }));
    var problem: []const u8 = "";
    const v = (try receipt.load(a, io, out, std.Io.Clock.real.now(io).toMilliseconds(), &problem)).?;
    try std.testing.expect(v.get("model").?.get("drafter_repo_id").? == .null);
    try std.testing.expect(std.mem.indexOf(u8, try std.Io.Dir.cwd().readFileAlloc(io, out, a, .limited(1 << 20)), "test-draft") == null);
}

test "--publish with rocm is refused before the run starts, with no request and no receipt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    r.hf_base = try f.url(a, "/api/models/");
    try r.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    const out = try r.path("run.json");
    const argv = try std.mem.concat(a, []const u8, &.{ try backendArgs(&r, "rocm", try f.url(a, ""), out), &.{ "--publish", "--yes", "--upload-url", try f.url(a, "") } });
    try std.testing.expectEqual(@as(u8, 1), try r.command(argv));
    try std.testing.expect(r.has(check.rocm_local) and !r.has("This generates model replies") and f.count() == 0);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, out, .{}));
}

test "an attached rocm run saves a local receipt that inspect accepts and publish refuses before any request" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    const rx = try world(a, false, "202 Accepted");
    defer rx.stop();
    r.hf_base = try rx.url(a, "/api/models/");
    try r.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 0), try r.command(try backendArgs(&r, "rocm", try f.url(a, ""), out)));
    try std.testing.expect(r.has("| rocm |") and r.has("Receipt saved") and r.has("rocm receipt is saved locally") and f.count() == 5);
    var problem: []const u8 = "";
    const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem)).?;
    try std.testing.expectEqualStrings("rocm", v.get("runtime").?.get("backend").?.string);
    try std.testing.expectEqualStrings("linux", v.get("runtime").?.get("platform").?.string);
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "inspect", out }));
    try std.testing.expect(r.has("| rocm |") and r.has("cannot be published until the board accepts rocm"));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "publish", out, "--yes", "--upload-url", try rx.url(a, "") }));
    try std.testing.expect(r.has(check.rocm_local) and !r.has("Public upload preview") and rx.count() == 0);
    const cuda = try r.path("cuda.json");
    try std.testing.expect(try receipt.save(a, std.testing.io, try receipts.legacy(a), cuda, now_ms, &problem));
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "publish", cuda, "--yes", "--upload-url", try rx.url(a, "") }));
    try std.testing.expectEqual(@as(usize, 1), posts(rx, "/submissions"));
}

test "a receipt path that cannot be written is refused before any request or serve child" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const blocker = try r.path("blocker");
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = blocker, .data = "" });
    const out = try std.fs.path.join(a, &.{ blocker, "run.json" });
    const wanted = try std.fmt.allocPrint(a, "Could not create the receipt directory {s}; choose another --output", .{blocker});
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    try std.testing.expectEqual(@as(u8, 1), try r.command(try runArgs(&r, try f.url(a, ""), out)));
    try std.testing.expect(r.has(wanted) and !r.has("This generates model replies") and f.count() == 0);
    const m = try managedWorld(a, false);
    defer m.stop();
    r.self_exe = try standIn(&r, m.port);
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ try r.path("private-checkpoint"), "--output", out }));
    try std.testing.expect(r.has(wanted) and m.count() == 0);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, try r.path("pid"), .{}));
}

test "v2 publish is off before the run, and legacy publish needs a token before lookup or prompt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    r.hf_base = try f.url(a, "/api/models/");
    const missing = "An upload token is required; run tensorfold benchmark login";
    const out = try r.path("run.json");
    const argv = try std.mem.concat(a, []const u8, &.{ try runArgs(&r, try f.url(a, ""), out), &.{ "--publish", "--yes", "--upload-url", try f.url(a, "") } });
    try std.testing.expectEqual(@as(u8, 1), try r.command(argv));
    try std.testing.expect(r.has("community-v2 submissions are disabled") and !r.has("This generates model replies") and f.count() == 0);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, out, .{}));
    var problem: []const u8 = "";
    try std.testing.expect(try receipt.save(a, std.testing.io, try receipts.legacy(a), out, now_ms, &problem));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "publish", out, "--yes", "--upload-url", try f.url(a, "") }));
    try std.testing.expect(r.has(missing) and !r.has("Public upload preview"));
    r.tty = true;
    r.stdin = .fixed("y\n");
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "publish", out, "--upload-url", try f.url(a, "") }));
    try std.testing.expect(r.has(missing) and !r.has("Upload these public fields"));
    try std.testing.expectEqual(@as(usize, 0), f.count());
}

test "the hidden login prompt reads with echo off and a guarded terminal, then restores and releases it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const pty = fake.Pty.open() orelse return error.SkipZigTest;
    defer pty.close();
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    var fds: [2]std.posix.fd_t = undefined;
    if (platform.c.pipe(&fds) != 0) return error.SkipZigTest;
    defer for (fds) |fd| {
        _ = platform.c.close(fd);
    };
    var buf: [256]u8 = undefined;
    var piped = (std.Io.File{ .handle = fds[0], .flags = .{ .nonblocking = false } }).readerStreaming(std.testing.io, &buf);
    r.tty = true;
    r.fd = pty.term;
    r.reader = &piped.interface;
    const Typist = struct {
        term: std.posix.fd_t,
        pipe: std.posix.fd_t,
        quiet: bool = false,
        guarded: std.posix.fd_t = -1,
        fn run(t: *@This()) void {
            var waited: u32 = 0;
            while (waited < 3000 and !t.quiet) : (waited += 2) {
                t.quiet = if (std.posix.tcgetattr(t.term)) |now| !now.lflag.ECHO else |_| false;
                if (t.quiet) t.guarded = stop.guardedTerminal() else fake.nap(2);
            }
            _ = platform.c.write(t.pipe, token ++ "\n", token.len + 1);
        }
    };
    var typist: Typist = .{ .term = pty.term, .pipe = fds[1] };
    const code = blk: {
        const t = try std.Thread.spawn(.{}, Typist.run, .{&typist});
        defer t.join();
        break :blk try r.command(&.{ "login", "--upload-url", try f.url(a, "") });
    };
    try std.testing.expectEqual(@as(u8, 0), code);
    try std.testing.expect(typist.quiet and typist.guarded == pty.term);
    try std.testing.expect((try std.posix.tcgetattr(pty.term)).lflag.ECHO and stop.guardedTerminal() == -1);
    try std.testing.expect(r.has("Upload token: ") and r.has("Benchmark uploads enabled for example-contributor.") and !r.has(token));
}

test "with no task to read input on, login fails with a message instead of a cancel" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    var single: std.Io.Threaded = .init_single_threaded;
    r.io = single.io();
    r.stdin = .fixed(token ++ "\n");
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "login", "--token-stdin", "--upload-url", "http://127.0.0.1:9" }));
    try std.testing.expect(r.has("Could not read the upload token") and !r.has("Benchmark cancelled"));
}

test "issue 969 v2: disabled submissions refuse saved, run and direct uploads without requests" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, false, "202 Accepted");
    defer f.stop();
    const api = try f.url(a, "");
    r.hf_base = try f.url(a, "/api/models/");
    try r.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    const out = try r.path("v2.json");
    var why: []const u8 = "";
    const v = try receipts.valid(a);
    try std.testing.expect(try receipt.save(a, std.testing.io, v, out, now_ms, &why));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "publish", out, "--yes", "--upload-url", api }));
    try std.testing.expect(r.has("community-v2 submissions are disabled until the client and receiver both support v2"));
    try std.testing.expectEqual(@as(usize, 0), f.count());
    const run_out = try r.path("new.json");
    const argv = try std.mem.concat(a, []const u8, &.{ try runArgs(&r, api, run_out), &.{ "--publish", "--yes", "--upload-url", api } });
    try std.testing.expectEqual(@as(u8, 1), try r.command(argv));
    try std.testing.expect(r.has("community-v2 submissions are disabled"));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, run_out, .{}));
    const up = @import("benchmark_upload.zig");
    try std.testing.expect(try up.upload(a, std.testing.io, .{ .api = api, .hf_base = r.hf_base, .environ = &r.env, .home = r.home, .now_ms = now_ms }, v, &why) == null);
    try std.testing.expectEqual(@as(usize, 0), f.count());
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "inspect", out }));
    try std.testing.expect(r.has("community-v2"));
}
