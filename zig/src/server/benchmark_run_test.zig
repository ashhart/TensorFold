//! A run's edges against fakes: a failed serve child's own reason, lost-run paths, refusals and drafter hints.
const std = @import("std");
const fake = @import("benchmark_fake.zig");
const cli = @import("benchmark_cli_test.zig");
const Run = cli.Run;
const benchmark = @import("benchmark.zig");

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

/// Caches ``repo`` in the Hugging Face layout under ``hub``.
fn cache(a: std.mem.Allocator, hub: []const u8, repo: []const u8) !void {
    const io = std.testing.io;
    const sha = "0123456789abcdef0123456789abcdef01234567";
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

test "with its known drafter cached, a default run names the drafter flag and passes none; --serial says it drops a drafter" {
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
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ big, "config.json" }), .data = "{\"model_type\": \"qwen3_5\", \"hidden_size\": 5120, \"num_hidden_layers\": 64}" });
    const drafter = "z-lab/Qwen3.8-27B-DFlash2";
    const hint = "This run passes no drafter; add --drafter " ++ drafter ++ " (cached) to measure drafted speed.";
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{big})));
    try std.testing.expect(!r.has(hint));
    try cache(a, hub, drafter);
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{big})));
    const out = r.out.written();
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, hint));
    try std.testing.expect(std.mem.indexOf(u8, out, hint).? < std.mem.indexOf(u8, out, "code T=0 warm-up").?);
    try std.testing.expect(!try passed(&r, "--drafter") and !try passed(&r, "--no-drafts"));
    for ([_][]const []const u8{ &.{try r.path("private-checkpoint")}, &.{ big, "--serial" }, &.{ big, "--drafter", "none" }, &.{ big, "--drafter", drafter } }) |head| {
        try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, head)));
        try std.testing.expect(!r.has(hint) and !r.has("--serial passes no drafter"));
    }
    try std.testing.expect(try passed(&r, "--drafter"));
    try std.testing.expectEqual(@as(u8, 0), try r.command(try argv(&r, &.{ big, "--serial", "--drafter", drafter })));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, r.out.written(), "--serial passes no drafter, so --drafter " ++ drafter ++ " is not used."));
    try std.testing.expect(!r.has(hint) and !try passed(&r, "--drafter") and try passed(&r, "--no-drafts"));
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
    try std.testing.expect(f.count() == 8 and r.has("Receipt saved"));
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
