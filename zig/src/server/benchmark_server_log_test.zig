//! A managed child's log with keys configured: no key in the file, readiness, and the startup tail.
const std = @import("std");
const server = @import("benchmark_server.zig");
const fake = @import("benchmark_fake.zig");
const credential = @import("sys").credential;
const t = @import("benchmark_server_test.zig");
const Setup = t.Setup;
const plan = t.plan;
const healthy = t.healthy;
const serving = t.serving;
const none = t.none;
const lines = t.lines;

test "managed log: a managed child's key never reaches its log file, raw or encoded across writes, only its replacement" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const routes = try healthy(a, "{\"status\": \"ok\"}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    const leak = "printf 'echo %s\\n' \"$TENSORFOLD_API_KEY\"; printf 'enc child%%2Dlog'; sleep 0.2; printf '%%2Dkey\\n'; ";
    const sh = try a.dupe([]const u8, &.{ "/bin/sh", "-c", try std.mem.concat(a, u8, &.{ leak, try serving(a, f.port, "exec sleep 30") }) });
    var problem: []const u8 = "";
    var m = (try server.Managed.spawn(a, std.testing.io, &s.env, sh, .{ .startup_timeout_s = 10, .bench_dir = s.bench, .key = "child-log-key" }, &problem)) orelse return error.TestUnexpectedResult;
    defer _ = m.stop();
    // the file itself, not a cleaned tail: what reached the disk
    var buf: [4096]u8 = undefined;
    const got = buf[0..try m.log.readPositionalAll(std.testing.io, &buf, 0)];
    try std.testing.expect(std.mem.indexOf(u8, got, "child-log-key") == null);
    try std.testing.expect(std.mem.indexOf(u8, got, "child%2Dlog%2Dkey") == null);
    try std.testing.expect(std.mem.indexOf(u8, got, "echo [redacted]") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "[tensorfold] serving .tensorfold-benchmark at http://127.0.0.1:") != null);
}

test "managed log: a valid key that overlaps the serving line still lets a healthy managed child become ready, and stays off disk" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const routes = try healthy(a, "{\"status\": \"ok\"}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    for ([_][]const u8{ "serving", ".tensorfold-benchmark", "127.0.0.1" }) |key| {
        const sh = try a.dupe([]const u8, &.{ "/bin/sh", "-c", try serving(a, f.port, "exec sleep 30") });
        var problem: []const u8 = "";
        var m = (try server.Managed.spawn(a, std.testing.io, &s.env, sh, .{ .startup_timeout_s = 5, .bench_dir = s.bench, .key = key }, &problem)) orelse return error.TestUnexpectedResult;
        var buf: [4096]u8 = undefined;
        const got = buf[0..try m.log.readPositionalAll(std.testing.io, &buf, 0)];
        try std.testing.expect(std.mem.indexOf(u8, got, key) == null);
        try std.testing.expect(m.stop() == null);
    }
}

test "managed log: an exited child's last diagnostic without a newline reaches its startup tail, also while a descendant holds the pipe" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    for ([_][]const u8{
        "sleep 4 & printf 'loading\\nerror: bad model child-log-key'; exit 3", // a descendant keeps the pipe open
        "printf 'loading\\nerror: bad model child-log-key'; exit 3", // ordinary end of stream
    }) |script| {
        var problem: []const u8 = "";
        var tail: []const []const u8 = &.{};
        const argv = try a.dupe([]const u8, &.{ "/bin/sh", "-c", script });
        const t0 = std.Io.Clock.awake.now(std.testing.io);
        try none(try server.Managed.start(a, std.testing.io, &s.env, plan, .{ .startup_timeout_s = 10, .bench_dir = s.bench, .argv = argv, .tail = &tail, .key = "child-log-key" }, &problem));
        try std.testing.expectEqualStrings("Benchmark server exited during startup; check the checkpoint and backend toolchain", problem);
        try lines(&.{"error: bad model [redacted]"}, tail);
        const took = t0.durationTo(std.Io.Clock.awake.now(std.testing.io)).toMilliseconds();
        try std.testing.expect(took < 4000); // bounded: no wait for the descendant
    }
}

test "managed log: a key whose encoded form is printed beside the serving line still lets the child become ready, and stays off disk" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var s = try Setup.init(a, &tmp);
    const routes = try healthy(a, "{\"status\": \"ok\"}", ".tensorfold-benchmark");
    const f = try fake.Fake.start(std.testing.allocator, &routes);
    defer f.stop();
    const sh = try a.dupe([]const u8, &.{ "/bin/sh", "-c", try std.mem.concat(a, u8, &.{ "printf 's%%65rving s%%2565rving\\n'; ", try serving(a, f.port, "exec sleep 30") }) });
    var problem: []const u8 = "";
    var m = (try server.Managed.spawn(a, std.testing.io, &s.env, sh, .{ .startup_timeout_s = 5, .bench_dir = s.bench, .key = "serving" }, &problem)) orelse return error.TestUnexpectedResult;
    var buf: [4096]u8 = undefined;
    const got = buf[0..try m.log.readPositionalAll(std.testing.io, &buf, 0)];
    try std.testing.expect(!try credential.contains(a, got, &.{"serving"}));
    try std.testing.expect(m.stop() == null);
}
