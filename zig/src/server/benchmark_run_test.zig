//! A run's edges against fakes: a failed serve child's own reason, lost-run paths, refusals and the drafter it passes.
const std = @import("std");
const fake = @import("benchmark_fake.zig");
const cli = @import("benchmark_cli_test.zig");
const Run = cli.Run;
const benchmark = @import("benchmark.zig");
const receipt = @import("benchmark_receipt.zig");

const short = [_][]const u8{ "--tokens", "4", "--reps", "1", "--temperatures", "0" };

fn argv(r: *Run, head: []const []const u8) ![]const []const u8 {
    return std.mem.concat(r.a, []const u8, &.{ head, &short });
}

fn writeScript(r: *Run, name: []const u8, text: []const u8) ![]const u8 {
    const script = try r.path(name);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = script, .data = text, .flags = .{ .permissions = .fromMode(0o755) } });
    return script;
}

fn missing(path: []const u8) !void {
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, path, .{}));
}

test "a serve child that exits during startup has its last log line printed after the message, and nothing is saved" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    _ = try cli.standIn(&r, 9);
    r.self_exe = try writeScript(&r, "failing-serve", "#!/bin/sh\nprintf 'loading\\n\\033[31merror: cannot read tokenizer.json\\033[0m\\n\\n'\nexit 3\n");
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 1), try r.command(try argv(&r, &.{ try r.path("private-checkpoint"), "--output", out })));
    const wanted = "tensorfold benchmark: Benchmark server exited during startup; check the checkpoint and backend toolchain\nLast server log line: error: cannot read tokenizer.json\n";
    try std.testing.expect(std.mem.endsWith(u8, r.err.written(), wanted));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "tokenizer") == null);
    try missing(out);
}

test "an --output naming a directory is refused before the run; a save that fails after the run still shows the table" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const routes = try a.dupe(fake.Route, &.{
        .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, false, 4, 15)}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, true, 4, 400)}) },
    });
    const f = try fake.Fake.start(std.testing.allocator, routes);
    defer f.stop();
    const base = try f.url(a, "");
    try std.Io.Dir.cwd().createDirPath(io, try r.path("existing"));
    for ([_][]const u8{ try std.mem.concat(a, u8, &.{ try r.path("newdir"), "/" }), try r.path("existing") }) |out| {
        try std.testing.expectEqual(@as(u8, 1), try r.command(try cli.runArgs(&r, base, out)));
        try std.testing.expect(r.has(try std.fmt.allocPrint(a, "Could not save the benchmark receipt to {s}; choose another --output", .{out})));
        try std.testing.expect(!r.has("This generates model replies") and f.count() == 0);
    }
    try missing(try r.path("newdir"));
    const later = try r.path("later");
    const Mover = struct {
        fn run(fk: *fake.Fake, dir: []const u8, aside: []const u8, moved: *bool) void {
            var waited: u32 = 0;
            while (fk.count() == 0 and waited < 10_000) : (waited += 5) fake.nap(5);
            std.Io.Dir.cwd().rename(dir, std.Io.Dir.cwd(), aside, std.testing.io) catch return;
            std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = dir, .data = "" }) catch return;
            moved.* = true;
        }
    };
    var moved = false;
    const code = blk: {
        const t = try std.Thread.spawn(.{}, Mover.run, .{ f, later, try r.path("later-aside"), &moved });
        defer t.join();
        break :blk try r.command(try cli.runArgs(&r, base, try std.fs.path.join(a, &.{ later, "run.json" })));
    };
    try std.testing.expect(moved);
    try std.testing.expectEqual(@as(u8, 1), code);
    try std.testing.expect(r.has(try std.fmt.allocPrint(a, "Could not create the receipt directory {s}", .{later})));
    try std.testing.expect(r.has("Workload       Temp") and r.has("Stream delivery includes transport buffering") and !r.has("Receipt saved"));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "chat           0") != null);
}

test "a run refused before it starts leaves no benchmarks directory; the default path is checked before the serve child" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const m = try cli.managedWorld(a, false);
    defer m.stop();
    _ = try cli.standIn(&r, m.port);
    r.self_exe = try r.path("serve-stand-in");
    const gpt = try r.path("gpt");
    try std.Io.Dir.cwd().createDirPath(io, gpt);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ gpt, "config.json" }), .data = "{\"model_type\": \"gpt2\"}" });
    const refused = [_][]const []const u8{
        &.{"example-org/not-cached"},
        &.{gpt},
        &.{ "example-org/test-model", "--server", "http://127.0.0.1:9" },
        &.{ try r.path("private-checkpoint"), "--startup-timeout", "3601" },
        &.{ try r.path("private-checkpoint"), "--drafter", "example-org/not-cached-drafter" },
    };
    for (refused) |head| {
        try std.testing.expectEqual(@as(u8, 1), try r.command(head));
        try missing(try r.path(".tensorfold"));
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try r.path(".tensorfold"), .data = "" });
    try std.testing.expectEqual(@as(u8, 1), try r.command(try argv(&r, &.{try r.path("private-checkpoint")})));
    try std.testing.expect(r.has("Could not create the receipt directory") and r.has("; choose another --output"));
    try std.testing.expect(m.count() == 0 and !r.has("This generates model replies"));
    try missing(try r.path("pid"));
    try std.Io.Dir.cwd().deleteFile(io, try r.path(".tensorfold"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{try r.path("private-checkpoint")})));
    const made = try std.Io.Dir.cwd().statFile(io, try r.path(".tensorfold/benchmarks"), .{});
    try std.testing.expect(made.kind == .directory and made.permissions.toMode() & 0o777 == 0o700 and r.has("Receipt saved"));
}

const sha = "0123456789abcdef0123456789abcdef01234567";

/// Caches ``repo`` in the Hugging Face layout under ``hub``.
fn cache(a: std.mem.Allocator, hub: []const u8, repo: []const u8) !void {
    const io = std.testing.io;
    const slash = std.mem.indexOfScalar(u8, repo, '/').?;
    const root = try std.fmt.allocPrint(a, "{s}/models--{s}--{s}", .{ hub, repo[0..slash], repo[slash + 1 ..] });
    try std.Io.Dir.cwd().createDirPath(io, try std.fs.path.join(a, &.{ root, "snapshots", sha }));
    try std.Io.Dir.cwd().createDirPath(io, try std.fs.path.join(a, &.{ root, "refs" }));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "refs", "main" }), .data = sha });
}

fn passed(r: *Run, flag: []const u8) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("args"), r.a, .limited(1 << 16));
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |x| if (std.mem.eql(u8, x, flag)) return true;
    return false;
}

/// The value the stand-in got after ``flag``, else null.
fn passedValue(r: *Run, flag: []const u8) !?[]const u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try r.path("args"), r.a, .limited(1 << 16));
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |x| if (std.mem.eql(u8, x, flag)) return it.next();
    return null;
}

/// The saved receipt's text holds ``field`` exactly.
fn receiptHas(r: *Run, out: []const u8, field: []const u8) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, out, r.a, .limited(1 << 20));
    return std.mem.indexOf(u8, text, field) != null;
}

test "--drafter auto passes the family's drafter only when it is cached and says which; only --serial adds --no-drafts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const m = try cli.managedWorld(a, false);
    defer m.stop();
    r.self_exe = try cli.standIn(&r, m.port);
    const hub = try r.path("hub");
    try r.env.put("HF_HUB_CACHE", hub);
    const big = try r.path("big-checkpoint");
    try std.Io.Dir.cwd().createDirPath(io, big);
    const config = "{\"model_type\": \"qwen3_5\", \"hidden_size\": 5120, \"num_hidden_layers\": 64}";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ big, "config.json" }), .data = config });
    const drafter = "z-lab/Qwen3.8-27B-DFlash2";
    const uncached = "This run passes no drafter: " ++ drafter ++ " is not in the Hugging Face cache; " ++
        "the native engine does not download: run tensorfold pull " ++ drafter ++ " first.\n";
    const named = "This run passes the cached drafter " ++ drafter ++ "; --drafter none runs without it.\n";
    const out = try r.path("run.json");
    const repo_null = "\"drafter_repo_id\":null";
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ big, "--output", out })));
    try std.testing.expect(std.mem.count(u8, r.out.written(), uncached) == 1 and !r.has(named));
    try std.testing.expect(!try passed(&r, "--drafter") and !try passed(&r, "--no-drafts"));
    try std.testing.expect(try receiptHas(&r, out, repo_null) and try receiptHas(&r, out, "\"serial\":false"));
    try cache(a, hub, drafter);
    const snapshot = try std.fs.path.join(a, &.{ hub, "models--z-lab--Qwen3.8-27B-DFlash2", "snapshots", sha });
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ big, "--output", out })));
    const text = r.out.written();
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, named));
    try std.testing.expect(std.mem.indexOf(u8, text, named).? < std.mem.indexOf(u8, text, "code T=0 warm-up").?);
    try std.testing.expectEqualStrings(snapshot, (try passedValue(&r, "--drafter")).?);
    try std.testing.expect(!try passed(&r, "--no-drafts") and !r.has("passes no drafter"));
    try std.testing.expect(try receiptHas(&r, out, "\"drafter_repo_id\":\"" ++ drafter ++ "\""));
    try std.testing.expect(try receiptHas(&r, out, "\"serial\":false"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ big, "--serial", "--output", out })));
    try std.testing.expect(try passed(&r, "--no-drafts") and !try passed(&r, "--drafter") and !r.has("drafter"));
    try std.testing.expect(try receiptHas(&r, out, repo_null) and try receiptHas(&r, out, "\"serial\":true"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ big, "--drafter", "none" })));
    try std.testing.expect(!try passed(&r, "--drafter") and !try passed(&r, "--no-drafts") and !r.has("drafter"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ big, "--drafter", drafter })));
    try std.testing.expectEqualStrings(snapshot, (try passedValue(&r, "--drafter")).?);
    try std.testing.expect(!try passed(&r, "--no-drafts") and !r.has("drafter"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{try r.path("private-checkpoint")})));
    try std.testing.expect(!try passed(&r, "--drafter") and !try passed(&r, "--no-drafts") and !r.has("drafter"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ big, "--serial", "--drafter", drafter })));
    const unused = "--serial passes no drafter, so --drafter " ++ drafter ++ " is not used.";
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, r.out.written(), unused));
    try std.testing.expect(!r.has(named) and !try passed(&r, "--drafter") and try passed(&r, "--no-drafts"));
}

test "a greedy-only engine's refusal is printed once with its reason and what to run instead, never saved" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const reason = "this engine decodes greedily only and this server samples by default: send temperature 0, or serve with --temperature 0";
    const refused = try fake.jsonReply(a, "400 Bad Request", "{\"error\": {\"message\": \"" ++ reason ++ "\", \"type\": \"invalid_request_error\"}}");
    const f = try fake.Fake.start(std.testing.allocator, try a.dupe(fake.Route, &.{
        .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{ refused, refused, try fake.reply(a, false, 4, 15) }) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{ refused, refused, try fake.reply(a, true, 4, 15) }) },
    }));
    defer f.stop();
    const out = try r.path("run.json");
    const head = try cli.runArgs(&r, try f.url(a, ""), out);
    const args = try std.mem.concat(a, []const u8, &.{ head[0 .. head.len - 4], &.{ "--temperatures", "1,0", "--output", out } });
    try std.testing.expectEqual(@as(u8, 2), try r.command(args));
    const text = r.out.written();
    const answered = "Server answered HTTP 400: " ++ reason ++ "\n";
    const greedy = "This engine decodes greedily only, so the cells above temperature 0 fail and the run cannot be ranked; for a local run, use --temperatures 0.\n";
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, answered));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, greedy));
    try std.testing.expect(std.mem.indexOf(u8, text, "code T=1 warm-up: error\n" ++ answered ++ greedy ++ "code T=1 repeat 1/1: error\n") != null);
    try std.testing.expect(f.count() == 9 and r.has("Receipt saved")); // the manifest, then eight replies
    const saved = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, out, a, .limited(1 << 20));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, saved, "\"error_code\":\"http_error\""));
    try std.testing.expect(std.mem.indexOf(u8, saved, "greedily") == null and std.mem.indexOf(u8, saved, "Bad Request") == null);
    const server_source = @embedFile("server.zig");
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, server_source, "\"this engine " ++ benchmark.greedy_only));
}

test "distinct refusals are printed at most three times, and a greedy-only reason at temperature 0 adds no advice" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    var replies: [5][]const fake.Write = undefined;
    for (&replies, 0..) |*w, i| w.* = try fake.jsonReply(a, "500 Internal Server Error", try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"engine decodes greedily only, case {d}\"}}}}", .{i}));
    const f = try fake.Fake.start(std.testing.allocator, try a.dupe(fake.Route, &.{
        .{ .method = "POST", .path = "/v1/completions", .replies = &replies },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, true, 4, 15)}) },
    }));
    defer f.stop();
    const head = try cli.runArgs(&r, try f.url(a, ""), try r.path("run.json"));
    try std.testing.expectEqual(@as(u8, 2), try r.command(try std.mem.concat(a, []const u8, &.{ head, &.{ "--reps", "4" } })));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, r.out.written(), "Server answered HTTP 500: engine decodes greedily only, case "));
    try std.testing.expect(r.has("case 2\n") and !r.has("case 3") and !r.has("This engine decodes greedily only"));
}

test "a serve child's load-limit hint is printed with its error, and a big checkpoint gets a longer startup" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    _ = try cli.standIn(&r, 9);
    const hint = "tensorfold: weights + caches need 184 GB; the limit is 179.2 GB (70% of RAM); raise it with GLM_LOAD_LIMIT_GB=190";
    const failed = "tensorfold: the native engine cannot load X (OverMemoryLimit)";
    r.self_exe = try writeScript(&r, "failing-serve", "#!/bin/sh\nprintf '%s\\n' loading '" ++ hint ++ "' '" ++ failed ++ "' 'error(SafeAllocator): leaked' '(empty stack trace)'\nexit 1\n");
    const ckpt = try r.path("private-checkpoint");
    try std.testing.expectEqual(@as(u8, 1), try r.command(try argv(&r, &.{ckpt})));
    try std.testing.expect(std.mem.endsWith(u8, r.err.written(), "toolchain\nLast server log lines:\n  " ++ hint ++ "\n  " ++ failed ++ "\n"));
    try std.testing.expect(!r.has("--startup-timeout sets another limit"));
    const m = try cli.managedWorld(a, false);
    defer m.stop();
    r.self_exe = try cli.standIn(&r, m.port);
    for ([_]struct { []const u8, u64 }{ .{ "model-00001.safetensors", 100_000_000_000 }, .{ "blob", 70_000_000_000 }, .{ "optimizer.bin", 50_000_000_000 } }) |w| {
        const file = try std.Io.Dir.cwd().createFile(io, try std.fs.path.join(a, &.{ ckpt, w[0] }), .{});
        defer file.close(io);
        try file.setLength(io, w[1]);
    }
    try std.Io.Dir.cwd().symLink(io, "blob", try std.fs.path.join(a, &.{ ckpt, "model-00002.safetensors" }), .{});
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ckpt})));
    const wait = "The server gets up to 1500 s to load 170 GB of weights; --startup-timeout sets another limit.\n";
    const at = std.mem.indexOf(u8, r.out.written(), wait) orelse return error.TestExpectedWaitLine;
    try std.testing.expect(at < std.mem.indexOf(u8, r.out.written(), "code T=0 warm-up").?);
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ ckpt, "--startup-timeout", "900" })));
    try std.testing.expect(!r.has("--startup-timeout sets another limit"));
}

const two_ranks =
    \\{"schema_version":1,"rank_count":2,"runtime":{"tensorfold_version":"9.8.7","backend":"cuda","python_version":"none",
    \\"platform":"linux","driver":"1.2.3"},"model":{"revision":"0123456789abcdef0123456789abcdef01234567"},
    \\"hardware":{"system_memory_bytes":1000,"memory_type":"unified","gpus":[{"name":"Example GPU","memory_bytes":null,
    \\"cores":null}],"gpu_count":1,"source":"partial"}}
;

test "an attached run copies an accepted server manifest into its receipt and is not called unranked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const routes = try a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/v1/benchmark/manifest", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", two_ranks)}) },
        .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, false, 4, 15)}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, true, 4, 15)}) },
    });
    const f = try fake.Fake.start(std.testing.allocator, routes);
    defer f.stop();
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 0), try r.command(try cli.runArgs(&r, try f.url(a, ""), out)));
    try std.testing.expect(r.has("Receipt saved") and !r.has("unranked") and f.count() == 5);
    var problem: []const u8 = "";
    const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem)).?;
    const settings = v.get("settings").?;
    try std.testing.expect(!settings.get("managed").?.bool and settings.get("rank_count").?.int64() == 2);
    try std.testing.expectEqualStrings("9.8.7", v.get("runtime").?.get("tensorfold_version").?.string);
    try std.testing.expectEqualStrings("partial", v.get("hardware").?.get("source").?.string);
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef01234567", v.get("model").?.get("revision").?.string);
}

const no_rank =
    \\{"schema_version":1,"runtime":{"tensorfold_version":"9.8.7","backend":"cuda","python_version":"none",
    \\"platform":"linux"},"model":{"revision":"0123456789abcdef0123456789abcdef01234567"},
    \\"hardware":{"system_memory_bytes":1000,"memory_type":"unified","gpus":[{"name":"Example GPU","memory_bytes":null,
    \\"cores":null}],"gpu_count":1,"source":"partial"}}
;
const no_revision =
    \\{"schema_version":1,"rank_count":2,"runtime":{"tensorfold_version":"9.8.7","backend":"cuda","python_version":"none",
    \\"platform":"linux"},"model":{},
    \\"hardware":{"system_memory_bytes":1000,"memory_type":"unified","gpus":[{"name":"Example GPU","memory_bytes":null,
    \\"cores":null}],"gpu_count":1,"source":"partial"}}
;

test "an attached run whose accepted manifest lacks its rank count or revision is called unranked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ no_rank, no_revision }, 0..) |doc, i| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try Run.init(a, &tmp);
        const routes = try a.dupe(fake.Route, &.{
            .{ .method = "GET", .path = "/v1/benchmark/manifest", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", doc)}) },
            .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, false, 4, 15)}) },
            .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, true, 4, 15)}) },
        });
        const f = try fake.Fake.start(std.testing.allocator, routes);
        defer f.stop();
        const out = try r.path("run.json");
        try std.testing.expectEqual(@as(u8, 0), try r.command(try cli.runArgs(&r, try f.url(a, ""), out)));
        try std.testing.expect(r.has("does not attest a rank count and checkpoint revision; this result is unranked"));
        try std.testing.expect(r.has("Receipt saved") and !r.has("hardware and checkpoint revisions are unknown"));
        var problem: []const u8 = "";
        const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &problem)).?;
        try std.testing.expectEqualStrings("partial", v.get("hardware").?.get("source").?.string);
        const missing_field = if (i == 0) v.get("settings").?.get("rank_count").? else v.get("model").?.get("revision").?;
        try std.testing.expect(missing_field == .null);
    }
}
