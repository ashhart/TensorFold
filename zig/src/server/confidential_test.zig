//! --confidential: the flag, the flags and variables it refuses, and refusals that never show a value.
const std = @import("std");
const builtin = @import("builtin");
const cli = @import("cli.zig");
const confidential = @import("confidential.zig");
const t = std.testing;

fn parse(a: std.mem.Allocator, argv: []const []const u8) !cli.Args {
    var u: cli.Usage = .{};
    return cli.parse(a, argv, &u) catch |e| {
        std.debug.print("{s}\n", .{u.message});
        return e;
    };
}

test "confidential: off by default, and a plain start with HOME, PATH and HF_* set is not refused" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    try env.put("TENSORFOLD_REQUEST_LOG", "/tmp/log");
    try t.expect(!(try parse(a, &.{"m"})).confidential);
    try t.expectEqual(@as(?[]const u8, null), try confidential.refusal(a, try parse(a, &.{"m"}), &env));
    var clean: std.process.Environ.Map = .init(a);
    for ([_][]const u8{ "HOME", "PATH", "TMPDIR", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "HF_HOME" }) |name| try clean.put(name, "x");
    const on = try parse(a, &.{ "m", "--confidential", "--dashboard", "--prompt-cache-gib", "0" });
    try t.expect(on.confidential);
    try t.expectEqual(@as(?[]const u8, null), try confidential.refusal(a, on, &clean));
}

test "confidential: each flag that keeps or sends request content names itself" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    const cases = [_]struct { flag: []const u8, argv: []const []const u8 }{
        .{ .flag = "--record", .argv = &.{ "--record", "/tmp/rows" } },
        .{ .flag = "--learn-conversations", .argv = &.{ "--prompt-cache-disk", "/tmp/kept" } },
        .{ .flag = "--learn-conversations", .argv = &.{"--learn-conversations"} },
        .{ .flag = "--learn", .argv = &.{"--learn"} },
        .{ .flag = "--learn", .argv = &.{ "--learn-dir", "/tmp/kept" } },
        .{ .flag = "--recall", .argv = &.{"--recall"} },
        .{ .flag = "--recall", .argv = &.{ "--recall-dir", "/tmp/recall" } },
        .{ .flag = "--slide", .argv = &.{"--slide"} },
        .{ .flag = "--slide", .argv = &.{ "--slide-graph", "/tmp/graph" } },
        .{ .flag = "--compact-memory", .argv = &.{ "--compact-memory", "/tmp/notes" } },
        .{ .flag = "--fsr-route-trace", .argv = &.{ "--fsr-route-trace", "/tmp/trace" } },
        .{ .flag = "--cortheon", .argv = &.{"--cortheon"} },
    };
    for (cases) |c| {
        const args = try parse(a, try std.mem.concat(a, []const u8, &.{ &.{ "m", "--confidential" }, c.argv }));
        try t.expectEqualStrings(c.flag, confidential.kept(args).?);
        const why = (try confidential.refusal(a, args, &env)).?;
        try t.expect(std.mem.indexOf(u8, why, c.flag) != null);
    }
}

test "confidential: every TensorFold and loader variable set is named, sorted, and its value never shown" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    try env.put("HOME", "/tmp/home");
    try env.put("TF_TIMING", "secret-value-1");
    try env.put("TENSORFOLD_REQUEST_LOG", "secret-value-2");
    try env.put("MCDMA_RPC_LIBRARY", "secret-value-3");
    try env.put("DYLD_INSERT_LIBRARIES", "secret-value-4");
    try env.put("HF_HUB_CACHE", "/tmp/hub");
    const why = (try confidential.refusal(a, try parse(a, &.{ "m", "--confidential" }), &env)).?;
    try t.expect(std.mem.endsWith(u8, why, "unset DYLD_INSERT_LIBRARIES, MCDMA_RPC_LIBRARY, TENSORFOLD_REQUEST_LOG, TF_TIMING"));
    try t.expect(std.mem.indexOf(u8, why, "secret-value") == null);
    for ([_][]const u8{ "FZ_X", "GLM_X", "LD_PRELOAD", "LD_AUDIT" }) |name| {
        var one: std.process.Environ.Map = .init(a);
        try one.put(name, "v");
        const named = try confidential.settings(a, &one);
        try t.expectEqual(@as(usize, 1), named.len);
        try t.expectEqualStrings(name, named[0]);
    }
}

test "confidential: no core dumps, soft and hard" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    try confidential.noCoreDumps(); // lowers this test process's own limits only, which a test run never needs
    const after = try std.posix.getrlimit(.CORE);
    try t.expect(after.cur == 0 and after.max == 0);
}
