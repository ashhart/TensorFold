//! The serve child's life with a shell stand-in: ready, exit, timeout, stubborn child, lock, Ctrl-C.
const std = @import("std");
const cli = @import("cli.zig");
const server = @import("benchmark_server.zig");
const stop = @import("benchmark_stop.zig");
const fake = @import("benchmark_fake.zig");
const posix = std.posix;

const plan: server.Plan = .{ .exe = "/opt/example/tensorfold-native", .dir = "/models/x", .backend = "", .tokens = 256, .context = 4096, .drafter_dir = null, .serial = true };

test "the child's flags parse as serve flags and carry none that native serve refuses; adding --tp fails" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p = plan;
    p.backend = server.buildBackend();
    const args = try server.argv(a, p);
    var u: cli.Usage = .{};
    const parsed = try cli.parse(a, args[2..], &u);
    try std.testing.expect(parsed.port == 0 and parsed.no_drafts and parsed.prompt_cache_gib.? == 0 and parsed.max_tokens == 256);
    for (args) |x| for ([_][]const u8{ "--tp", "--rank", "--checkpoint-slots", "--spill-gib" }) |refused| try std.testing.expect(!std.mem.eql(u8, x, refused));
    const with_tp = try std.mem.concat(a, []const u8, &.{ args[2..], &.{ "--tp", "1" } });
    try std.testing.expectError(error.Usage, cli.parse(a, with_tp, &u));
    try std.testing.expect(std.mem.indexOf(u8, u.message, "not served") != null);
}

test "drafts stay on unless serial: only serial adds --no-drafts, and a drafter directory goes to --drafter" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p = plan;
    p.backend = server.buildBackend();
    for ([_]bool{ false, true }) |serial| for ([_]?[]const u8{ null, "/models/draft" }) |drafter| {
        p.serial = serial;
        p.drafter_dir = drafter;
        const args = try server.argv(a, p);
        var u: cli.Usage = .{};
        const parsed = try cli.parse(a, args[2..], &u);
        try std.testing.expectEqual(serial, parsed.no_drafts);
        if (drafter) |d| try std.testing.expectEqualStrings(d, parsed.drafter.?);
        if (drafter == null) try std.testing.expect(parsed.drafter == null);
        var n: usize = 0;
        for (args) |x| n += @intFromBool(std.mem.eql(u8, x, "--no-drafts"));
        try std.testing.expectEqual(@as(usize, @intFromBool(serial)), n);
    };
}

test "the child's environment drops the key, log, salt and token and turns off the live line" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var parent: std.process.Environ.Map = .init(a);
    for ([_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_SEED_SALT", "TENSORFOLD_BENCHMARK_TOKEN", "PATH" }) |k| try parent.put(k, "x");
    const env = try server.environ(a, &parent);
    for ([_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_SEED_SALT", "TENSORFOLD_BENCHMARK_TOKEN" }) |k| try std.testing.expect(env.get(k) == null);
    for ([_][]const u8{ "TENSORFOLD_NO_LIVE", "TENSORFOLD_NO_UPDATE_CHECK", "HF_HUB_OFFLINE" }) |k| try std.testing.expectEqualStrings("1", env.get(k).?);
    try std.testing.expectEqualStrings("x", env.get("PATH").?);
}

test "the port comes from serve's serving line, whose format serve.zig still prints" {
    try std.testing.expectEqual(@as(?u16, 51234), server.servingPort("x\n[tensorfold] serving .tensorfold-benchmark at http://127.0.0.1:51234/v1 (sampling: greedy)\n"));
    try std.testing.expectEqual(@as(?u16, null), server.servingPort("[tensorfold] serving other at http://127.0.0.1:5/v1 ("));
    try std.testing.expectEqual(@as(?u16, null), server.servingPort("[tensorfold] serving .tensorfold-benchmark at http://127.0.0.1:"));
    const serve_source = @embedFile("serve.zig");
    try std.testing.expect(std.mem.indexOf(u8, serve_source, "\"serving {s} at http://{s}:{d}/v1 (") != null);
    try std.testing.expect(std.mem.indexOf(u8, serve_source, "\"serving {s} on http://{s}:{d}/v1 (") == null);
}

const Setup = struct {
    a: std.mem.Allocator,
    env: std.process.Environ.Map,
    dir: []const u8,
    bench: []const u8,

    fn init(a: std.mem.Allocator, tmp: *std.testing.TmpDir) !Setup {
        var env: std.process.Environ.Map = .init(a);
        try env.put("PATH", "/usr/bin:/bin");
        const dir = try fake.tmpPath(a, tmp);
        return .{ .a = a, .env = env, .dir = dir, .bench = try std.fs.path.join(a, &.{ dir, "benchmarks" }) };
    }

    fn start(s: *Setup, script: []const u8, timeout_s: f64, grace_ms: u32, problem: *[]const u8) !?server.Managed {
        const sh = try s.a.dupe([]const u8, &.{ "/bin/sh", "-c", script });
        return server.Managed.start(s.a, std.testing.io, &s.env, plan, .{ .startup_timeout_s = timeout_s, .grace_ms = grace_ms, .bench_dir = s.bench, .argv = sh }, problem);
    }
};

fn healthy(a: std.mem.Allocator, health: []const u8, model_id: []const u8) ![2]fake.Route {
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{model_id});
    return .{
        .{ .method = "GET", .path = "/health", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", health)}) },
        .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", models)}) },
    };
}

fn serving(a: std.mem.Allocator, port: u16, tail: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "printf '[tensorfold] serving .tensorfold-benchmark at http://127.0.0.1:%s/v1 (x)\\n' {d}; {s}", .{ port, tail });
}

fn gone(pid: posix.pid_t) bool {
    posix.kill(pid, @fromBackingInt(0)) catch |e| return e == error.ProcessNotFound;
    return false;
}

/// Passes when ``start`` gave no child; a child it did give is stopped before the test fails.
fn none(got: ?server.Managed) !void {
    var m = got orelse return;
    _ = m.stop();
    return error.TestUnexpectedResult;
}

test "a ready child is served, then stopped and reaped; its log is private and unlinked, the lock stays" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const routes = try healthy(a, "{\"status\": \"ok\", \"warming\": false}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    var problem: []const u8 = "";
    var m = (try s.start(try serving(a, f.port, "exec sleep 30"), 10, 10_000, &problem)) orelse return error.TestUnexpectedResult;
    defer _ = m.stop();
    try std.testing.expectEqualStrings(try f.url(a, ""), m.base());
    const st = try m.log.stat(std.testing.io);
    try std.testing.expect(st.nlink == 0 and st.permissions.toMode() & 0o777 == 0o600);
    try std.testing.expect(m.stop() == null);
    try std.testing.expect(m.stop() == null);
    try std.testing.expect(gone(m.pid));
    const lock = try std.Io.Dir.cwd().statFile(std.testing.io, try std.fs.path.join(a, &.{ s.bench, ".lock" }), .{});
    try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(lock.permissions.toMode() & 0o777)));
}

test "warming, a missing name or a starting status time out; an exited child is named without paths" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    for ([_][]const u8{ "{\"status\": \"ok\", \"warming\": true}", "{\"status\": \"starting\"}", "{\"ok\": true, \"models\": \"other\"}" }, 0..) |health, i| {
        const routes = try healthy(a, health, if (i == 2) "another-server" else ".tensorfold-benchmark");
        const f = try fake.Fake.start(std.testing.allocator, &routes);
        defer f.stop();
        var problem: []const u8 = "";
        try none(try s.start(try serving(a, f.port, "exec sleep 30"), 0.3, 10_000, &problem));
        try std.testing.expect(std.mem.indexOf(u8, problem, "timed out") != null);
    }
    var problem: []const u8 = "";
    try none(try s.start("exit 3", 10, 10_000, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "exited during startup") != null and std.mem.indexOfScalar(u8, problem, '/') == null);
}

test "a child that exits during startup hands back its last log lines, cleaned for a terminal; a timeout names its limit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    var problem: []const u8 = "";
    var tail: []const []const u8 = &.{};
    const exits = try a.dupe([]const u8, &.{ "/bin/sh", "-c", "printf 'loading\\nerror: cannot read tokenizer.json\\n\\n  \\n'; exit 3" });
    try none(try server.Managed.start(a, std.testing.io, &s.env, plan, .{ .startup_timeout_s = 10, .bench_dir = s.bench, .argv = exits, .tail = &tail }, &problem));
    try std.testing.expectEqualStrings("Benchmark server exited during startup; check the checkpoint and backend toolchain", problem);
    try lines(&.{"error: cannot read tokenizer.json"}, tail);
    tail = &.{};
    const hangs = try a.dupe([]const u8, &.{ "/bin/sh", "-c", "printf 'still loading\\n'; exec sleep 30" });
    try none(try server.Managed.start(a, std.testing.io, &s.env, plan, .{ .startup_timeout_s = 0.3, .bench_dir = s.bench, .argv = hangs, .tail = &tail }, &problem));
    try std.testing.expectEqualStrings("Benchmark server startup timed out after 0.3 s (--startup-timeout, at most 3600); check the checkpoint, GPU memory and backend toolchain", problem);
    try std.testing.expect(tail.len == 0);
    try lines(&.{"red alert bell"}, try server.lastLines(a, "ok\n\x1b[1;31mred\x1b[0m alert\x07 bell\r\n\x1b[2K\n"));
    try lines(&.{}, try server.lastLines(a, "\n \t\n\x1b[0m\n"));
    try lines(&.{"tab here"}, try server.lastLines(a, "tab\there"));
    const leaks = "error: cannot open w.safetensors\ntensorfold: the native engine cannot load m (OpenFailed)\nerror(SafeAllocator): leaked [len: 20] allocated at: \n(empty stack trace)\n\n";
    try lines(&.{ "error: cannot open w.safetensors", "tensorfold: the native engine cannot load m (OpenFailed)" }, try server.lastLines(a, leaks));
    try lines(&.{"loading"}, try server.lastLines(a, "loading\nerror(gpa): memory address 0x10 leaked:\n(empty stack trace)\n"));
    try lines(&.{"caf\xc3\xa9 ok"}, try server.lastLines(a, "caf\xc3\xa9\xc2\x9b\xff\x80 ok"));
    const accents: [200][2]u8 = @splat(.{ 0xc3, 0xa9 });
    const cut = (try server.lastLines(a, try std.mem.concat(a, u8, &.{ "a", std.mem.asBytes(&accents) })))[0];
    try std.testing.expect(cut.len == 302 and std.mem.endsWith(u8, cut, "...") and std.unicode.utf8ValidateSlice(cut));
}

fn lines(want: []const []const u8, got: []const []const u8) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
}

test "an engine's load-limit hint before its error is kept with the error, three lines at most, in their order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const hint = "tensorfold: weights + caches need 184 GB; the limit is 179.2 GB (70% of RAM); raise it with GLM_LOAD_LIMIT_GB=190";
    const failed = "tensorfold: the native engine cannot load X (OverMemoryLimit)";
    try lines(&.{ hint, failed }, try server.lastLines(a, hint ++ "\n" ++ failed ++ "\n"));
    const xs: [200]u8 = @splat('x');
    const long_path = try std.mem.concat(a, u8, &.{ "tensorfold: the native GLM-5.3-Flash engine cannot load /models/", &xs, " (OverMemoryLimit)" });
    try lines(&.{ hint, long_path }, try server.lastLines(a, try std.mem.concat(a, u8, &.{ "info: loading\n", hint, "\n", long_path, "\nerror(SafeAllocator): leaked\n" })));
    const glm = "error: glm: 169.4 GB of weights and 14.6 GB of caches pass this Mac's 179.2 GB load limit (70% of RAM); load a layer subset or the expert-parallel pair, or set GLM_LOAD_LIMIT_GB";
    const many = glm ++ "\nerror: one\nerror: two\nerror: three\n" ++ failed ++ "\nTF_AFTER_ERROR=1 is not a hint\n";
    try lines(&.{ glm, "error: three", failed }, try server.lastLines(a, many));
    try lines(&.{ "error: two", "error: three", failed }, try server.lastLines(a, "error: one\nerror: two\nerror: three\n" ++ failed));
    try lines(&.{ "tensorfold: GLM_LOAD_LIMIT_GB is not a load limit", "error: a", "error: b" }, try server.lastLines(a, "set GLM_LOAD_LIMIT_GB=1\ntensorfold: GLM_LOAD_LIMIT_GB is not a load limit\nerror: a\nerror: b\n"));
}

test "the startup timeout grows with the weights past 20 GB, 6 s per GB from 600 s, to 3600 s" {
    try std.testing.expectEqual(@as(f64, 600), server.startupTimeout(0));
    try std.testing.expectEqual(@as(f64, 600), server.startupTimeout(20_000_000_000));
    try std.testing.expectEqual(@as(f64, 606), server.startupTimeout(21_000_000_000));
    try std.testing.expectEqual(@as(f64, 1500), server.startupTimeout(170_000_000_000));
    try std.testing.expectEqual(@as(f64, 3600), server.startupTimeout(520_000_000_000));
    try std.testing.expectEqual(@as(f64, 3600), server.startupTimeout(std.math.maxInt(u64)));
}

test "a child that ignores SIGTERM is killed and reaped after the grace period" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const routes = try healthy(a, "{\"ok\": true}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    var problem: []const u8 = "";
    var m = (try s.start(try serving(a, f.port, "trap '' TERM; i=0; while [ $i -lt 400 ]; do sleep 0.05; i=$((i+1)); done"), 10, 200, &problem)) orelse return error.TestUnexpectedResult;
    defer _ = m.stop();
    try std.testing.expect(m.stop() == null);
    try std.testing.expect(gone(m.pid));
}

test "a second benchmark finds the lock busy and spawns nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    var problem: []const u8 = "";
    const held = server.lockRun(std.testing.io, s.bench, &problem) orelse return error.TestUnexpectedResult;
    defer held.close(std.testing.io);
    const marker = try std.fs.path.join(a, &.{ s.dir, "spawned" });
    try none(try s.start(try std.fmt.allocPrint(a, "touch {s}", .{marker}), 10, 10_000, &problem));
    try std.testing.expectEqualStrings("Another TensorFold benchmark is running; wait for it to finish", problem);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, marker, .{}));
}

test "Ctrl-C during startup stops and reaps the child and frees the lock" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    stop.reset();
    defer stop.reset();
    const pidfile = try std.fs.path.join(a, &.{ s.dir, "pid" });
    const script = try std.fmt.allocPrint(a, "echo $$ > {s}; exec sleep 30", .{pidfile});
    var problem: []const u8 = "x";
    {
        const t = try std.Thread.spawn(.{}, struct {
            fn later() void {
                fake.nap(300);
                stop.request(.INT);
            }
        }.later, .{});
        defer t.join();
        try none(try s.start(script, 10, 10_000, &problem));
    }
    try std.testing.expect(problem.len == 0 and stop.requested().? == .INT);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, pidfile, a, .limited(64));
    try std.testing.expect(gone(try std.fmt.parseInt(posix.pid_t, std.mem.trim(u8, text, " \n"), 10)));
    const again = server.lockRun(std.testing.io, s.bench, &problem) orelse return error.TestUnexpectedResult;
    again.close(std.testing.io);
}

test "a lock path that is a symlink is refused and its target is never created" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const s = try Setup.init(a, &tmp);
    const io = std.testing.io;
    try std.Io.Dir.cwd().createDirPath(io, s.bench);
    const target = try std.fs.path.join(a, &.{ s.dir, "elsewhere" });
    try std.Io.Dir.cwd().symLink(io, target, try std.fs.path.join(a, &.{ s.bench, ".lock" }), .{});
    var problem: []const u8 = "";
    try std.testing.expect(server.lockRun(io, s.bench, &problem) == null);
    try std.testing.expectEqualStrings("Could not create the benchmark lock; check the TensorFold directory permissions", problem);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, target, .{}));
}

test "an allocation failure before the spawn leaves the lock free" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const sh = try a.dupe([]const u8, &.{ "/bin/sh", "-c", "exit 0" });
    for (0..3) |fail_index| {
        var failing: std.testing.FailingAllocator = .init(a, .{ .fail_index = fail_index });
        var problem: []const u8 = "";
        try std.testing.expectError(error.OutOfMemory, server.Managed.start(failing.allocator(), std.testing.io, &s.env, plan, .{ .startup_timeout_s = 1, .bench_dir = s.bench, .argv = sh }, &problem));
        const again = server.lockRun(std.testing.io, s.bench, &problem) orelse return error.TestUnexpectedResult;
        again.close(std.testing.io);
    }
}

test "a second Ctrl-C kills the running serve child before the benchmark exits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const routes = try healthy(a, "{\"ok\": true}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    var problem: []const u8 = "";
    var m = (try s.start(try serving(a, f.port, "trap '' TERM; exec sleep 30"), 10, 10_000, &problem)) orelse return error.TestUnexpectedResult;
    defer _ = m.stop();
    stop.abandon();
    var waited: u32 = 0;
    while (!gone(m.pid) and waited < 2000) : (waited += 10) {
        var status: c_int = 0;
        _ = std.c.waitpid(m.pid, &status, std.c.W.NOHANG);
        fake.nap(10);
    }
    try std.testing.expect(waited < 2000);
}

test "a lock path that is a symlink to an existing file, or a FIFO, is refused and left as it was" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const s = try Setup.init(a, &tmp);
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const lock = try std.fs.path.join(a, &.{ s.bench, ".lock" });
    try cwd.createDirPath(io, s.bench);
    const target = try std.fs.path.join(a, &.{ s.dir, "elsewhere" });
    try cwd.writeFile(io, .{ .sub_path = target, .data = "kept" });
    try cwd.symLink(io, target, lock, .{});
    var problem: []const u8 = "";
    try std.testing.expect(server.lockRun(io, s.bench, &problem) == null);
    try std.testing.expectEqualStrings("Could not create the benchmark lock; check the TensorFold directory permissions", problem);
    try std.testing.expectEqualStrings("kept", try cwd.readFileAlloc(io, target, a, .limited(16)));
    try cwd.deleteFile(io, lock);
    try std.testing.expectEqual(@as(c_int, 0), mkfifo(try a.dupeSentinel(u8, lock, 0), 0o600));
    try std.testing.expect(server.lockRun(io, s.bench, &problem) == null);
    try std.testing.expectEqualStrings("Could not create the benchmark lock; check the TensorFold directory permissions", problem);
}

extern "c" fn mkfifo(path: [*:0]const u8, mode: std.c.mode_t) c_int;

const Racer = struct {
    bench: []const u8,
    go: *std.atomic.Value(bool),
    problem: []const u8 = "",
    file: ?std.Io.File = null,

    fn run(r: *Racer) void {
        while (!r.go.load(.acquire)) std.atomic.spinLoopHint();
        r.file = server.lockRun(std.testing.io, r.bench, &r.problem);
    }
};

test "benchmarks that race to create the first lock: one wins and every other one is told a benchmark is running" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const s = try Setup.init(a, &tmp);
    for (0..100) |round| {
        const bench = try std.fmt.allocPrint(a, "{s}/b{d}", .{ s.bench, round });
        try std.Io.Dir.cwd().createDirPath(std.testing.io, bench);
        var go: std.atomic.Value(bool) = .init(false);
        var racers: [4]Racer = undefined;
        {
            var threads: [4]std.Thread = undefined;
            var started: usize = 0;
            defer {
                go.store(true, .release);
                for (threads[0..started]) |t| t.join();
            }
            for (&racers, &threads) |*r, *t| {
                r.* = .{ .bench = bench, .go = &go };
                t.* = try std.Thread.spawn(.{}, Racer.run, .{r});
                started += 1;
            }
        }
        var won: usize = 0;
        for (racers) |r| if (r.file) |f| {
            won += 1;
            f.close(std.testing.io);
        } else try std.testing.expectEqualStrings("Another TensorFold benchmark is running; wait for it to finish", r.problem);
        try std.testing.expectEqual(@as(usize, 1), won);
    }
}

test "an allocation failure while waiting for readiness stops the child and frees the lock" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const io = std.testing.io;
    const routes = try healthy(a, "{\"ok\": true}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    const pidfile = try std.fs.path.join(a, &.{ s.dir, "pid" });
    const sh = try a.dupe([]const u8, &.{ "/bin/sh", "-c", try std.fmt.allocPrint(a, "echo $$ > {s}; {s}", .{ pidfile, try serving(a, f.port, "exec sleep 30") }) });
    var after_spawn: usize = 0;
    var fail_index: usize = 0;
    while (after_spawn < 3 and fail_index < 400) : (fail_index += 1) {
        var failing: std.testing.FailingAllocator = .init(a, .{ .fail_index = fail_index });
        var problem: []const u8 = "";
        if (server.Managed.start(failing.allocator(), io, &s.env, plan, .{ .startup_timeout_s = 10, .bench_dir = s.bench, .argv = sh }, &problem)) |got| {
            var m = got orelse return error.TestUnexpectedResult;
            _ = m.stop();
            break;
        } else |e| try std.testing.expectEqual(error.OutOfMemory, e);
        const text = std.Io.Dir.cwd().readFileAlloc(io, pidfile, a, .limited(64)) catch continue;
        try std.Io.Dir.cwd().deleteFile(io, pidfile);
        after_spawn += 1;
        try std.testing.expect(gone(try std.fmt.parseInt(posix.pid_t, std.mem.trim(u8, text, " \n"), 10)));
        const again = server.lockRun(io, s.bench, &problem) orelse return error.TestUnexpectedResult;
        again.close(io);
    }
    try std.testing.expect(after_spawn > 0);
}

test "an exited child is seen without being reaped, and once stopped it is released from a second Ctrl-C" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const io = std.testing.io;
    const quick = try std.process.spawn(io, .{ .argv = &.{ "/bin/sh", "-c", "exit 0" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
    var waited: u32 = 0;
    while (!server.ended(quick.id.?) and waited < 2000) : (waited += 10) fake.nap(10);
    var status: c_int = 0;
    try std.testing.expectEqual(quick.id.?, std.c.waitpid(quick.id.?, &status, std.c.W.NOHANG));
    const routes = try healthy(a, "{\"ok\": true}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    var problem: []const u8 = "";
    var m = (try s.start(try serving(a, f.port, "exec sleep 30"), 10, 10_000, &problem)) orelse return error.TestUnexpectedResult;
    defer _ = m.stop();
    try std.testing.expectEqual(m.pid, stop.guardedChild());
    try std.testing.expect(m.stop() == null);
    try std.testing.expectEqual(@as(posix.pid_t, 0), stop.guardedChild());
    const other = try std.process.spawn(io, .{ .argv = &.{ "/bin/sh", "-c", "exec sleep 30" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
    defer {
        _ = std.c.kill(other.id.?, .KILL);
        _ = std.c.waitpid(other.id.?, &status, 0);
    }
    stop.abandon();
    fake.nap(100);
    try std.testing.expectEqual(@as(posix.pid_t, 0), std.c.waitpid(other.id.?, &status, std.c.W.NOHANG));
}
