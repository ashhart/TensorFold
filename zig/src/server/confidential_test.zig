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
        .{ .flag = "--prompt-cache-disk or --learn-conversations", .argv = &.{ "--prompt-cache-disk", "/tmp/kept" } },
        .{ .flag = "--prompt-cache-disk or --learn-conversations", .argv = &.{"--learn-conversations"} },
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
    for ([_][]const u8{ "FZ_X", "GLM_X", "XPU_DUMP", "TF_NCCL_LIB", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT" }) |name| {
        var one: std.process.Environ.Map = .init(a);
        try one.put(name, "v");
        const named = try confidential.settings(a, &one);
        try t.expectEqual(@as(usize, 1), named.len);
        try t.expectEqualStrings(name, named[0]);
    }
}

test "confidential: no core dumps, soft and hard, and on Linux not dumpable" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    try confidential.harden(); // lowers this test process's own limits and attach rights, which a test run never needs
    const after = try std.posix.getrlimit(.CORE);
    try t.expect(after.cur == 0 and after.max == 0);
    if (builtin.os.tag == .linux) try t.expectEqual(@as(usize, 0), std.os.linux.prctl(@backingInt(std.os.linux.PR.GET_DUMPABLE), 0, 0, 0, 0));
}

test "confidential: the GPU drivers' dump, injection and capture variables are refused" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names = [_][]const u8{ "CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "CUDA_ENABLE_USER_TRIGGERED_COREDUMP", "CUDA_COREDUMP_FILE", "CUDA_INJECTION64_PATH", "MTL_CAPTURE_ENABLED", "METAL_CAPTURE_ENABLED", "MTL_DEBUG_LAYER", "METAL_DEVICE_WRAPPER_TYPE", "MTL_SHADER_VALIDATION" };
    for (names) |name| {
        var one: std.process.Environ.Map = .init(a);
        try one.put(name, "1");
        try t.expectEqual(@as(usize, 1), (try confidential.settings(a, &one)).len);
    }
    var allowed: std.process.Environ.Map = .init(a);
    for ([_][]const u8{ "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "MTL_HUD_ENABLED" }) |name| try allowed.put(name, "0");
    try t.expectEqual(@as(usize, 0), (try confidential.settings(a, &allowed)).len);
}

/// An engine nothing may reach: the refusal comes before serve.run uses it.
const Untouched = struct {
    fn engine(u: *Untouched) @import("engine_api").Engine {
        return .{ .ctx = u, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) @import("engine_api").Info {
        @panic("serve.run reached the engine");
    }
    fn submit(_: *anyopaque, _: @import("engine_api").Id, _: *const @import("engine_api").Request, _: @import("engine_api").Sink) @import("engine_api").SubmitError!void {
        @panic("serve.run reached the engine");
    }
    fn cancel(_: *anyopaque, _: @import("engine_api").Id) void {}
    fn status(_: *anyopaque, _: *@import("engine_api").Status, _: []u32) void {}
    fn memory(_: *anyopaque, _: bool) ?@import("engine_api").Memory {
        return null;
    }
    fn close(_: *anyopaque) void {
        @panic("serve.run closed the engine");
    }
};

test "confidential: the shared serving path refuses too, before it listens or touches the engine" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var env: std.process.Environ.Map = .init(arena.allocator());
    try env.put("TENSORFOLD_REQUEST_LOG", "/tmp/never");
    var engine: Untouched = .{};
    var text: @import("drain_test.zig").TestText = .{};
    const setup: @import("serve.zig").Setup = .{ .engine = engine.engine(), .text = text.text(), .served = "m", .environ = &env, .stop = .{ .ctx = &engine, .close = Untouched.close } };
    try t.expectEqual(@as(u8, 1), @import("serve.zig").run(t.allocator, t.io, .{ .confidential = true, .port = 0 }, setup));
    try t.expectEqual(@as(?[]const u8, null), try confidential.enforce(arena.allocator(), .{}, &env));
}
