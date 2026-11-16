//! Serve and benchmark share one drafter table; serve pairs one only from local files and never fails startup for it.
const std = @import("std");
const json = @import("json");
const cli = @import("cli.zig");
const log = @import("log.zig");
const drafter = @import("drafter.zig");
const t = std.testing;
const target_config = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120,\"num_hidden_layers\":64}";
const repo = "z-lab/Qwen3.8-27B-DFlash2";
const revision = "0123456789abcdef0123456789abcdef01234567";
const cache_repo = "cache/models--z-lab--Qwen3.8-27B-DFlash2";

test "serve drafter: the shared table matches only its target family and complete shape" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { body: []const u8, found: bool }{
        .{ .body = target_config, .found = true },
        .{ .body = "{\"text_config\":{\"model_type\":\"qwen3_5\",\"hidden_size\":5120,\"num_hidden_layers\":64}}", .found = true },
        .{ .body = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"hidden_size\":5120,\"num_hidden_layers\":64}}", .found = true },
        .{ .body = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120,\"num_hidden_layers\":63}", .found = false },
        .{ .body = "{\"model_type\":\"qwen3_5\",\"hidden_size\":4096,\"num_hidden_layers\":64}", .found = false },
        .{ .body = "{\"model_type\":\"other\",\"hidden_size\":5120,\"num_hidden_layers\":64}", .found = false },
        .{ .body = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120}", .found = false },
        .{ .body = "[]", .found = false },
    };
    for (cases) |c| {
        const selected = drafter.knownDrafter((try json.parse(a, c.body)).ok);
        if (c.found) try t.expectEqualStrings(repo, selected.?) else try t.expect(selected == null);
    }
}

test "serve drafter: local cache hits pair, misses and unreadable configs stay plain, and explicit choices win" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    try tmp.dir.createDirPath(io, "target");
    try tmp.dir.writeFile(io, .{ .sub_path = "target/config.json", .data = target_config });
    const target = try std.fs.path.join(a, &.{ root, "target" });
    var env: std.process.Environ.Map = .init(a);
    try env.put("HF_HUB_CACHE", try std.fs.path.join(a, &.{ root, "cache" }));
    var problem: []const u8 = "";
    const absent = (try drafter.select(a, io, &env, target, null, false, "cuda", &problem)).?;
    try t.expect(absent.dir == null and absent.name == null and !absent.automatic);
    try t.expectEqualStrings("", problem);
    try tmp.dir.createDirPath(io, cache_repo ++ "/refs");
    try tmp.dir.createDirPath(io, cache_repo ++ "/snapshots/" ++ revision);
    try tmp.dir.writeFile(io, .{ .sub_path = cache_repo ++ "/refs/main", .data = revision ++ "\n" });
    for ([_][]const u8{ "cuda", "mlx" }) |backend| {
        const found = (try drafter.select(a, io, &env, target, null, false, backend, &problem)).?;
        try t.expect(found.automatic);
        try t.expectEqualStrings(repo, found.name.?);
        try t.expectEqualStrings(try std.fs.path.join(a, &.{ root, "cache", "models--z-lab--Qwen3.8-27B-DFlash2", "snapshots", revision }), found.dir.?);
    }
    // no table entry, no config.json, or one that cannot be read: no drafter, and startup goes on to name the problem
    try tmp.dir.createDirPath(io, "other/config.json");
    try tmp.dir.createDirPath(io, "plain");
    try tmp.dir.writeFile(io, .{ .sub_path = "plain/config.json", .data = "{\"model_type\":\"qwen3_5\",\"hidden_size\":2048}" });
    for ([_][]const u8{ "plain", "other", "other/config.json" }) |sub| {
        const none = (try drafter.select(a, io, &env, try std.fs.path.join(a, &.{ root, sub }), null, false, "cuda", &problem)).?;
        try t.expect(none.dir == null and !none.automatic);
    }
    for ([_]bool{ false, true }) |no_drafts| {
        const off = (try drafter.select(a, io, &env, target, "none", no_drafts, "cuda", &problem)).?;
        try t.expect(off.dir == null and off.name == null and !off.automatic);
    }
    try t.expect((try drafter.select(a, io, &env, target, null, true, "cuda", &problem)).?.dir == null);
    try t.expect((try drafter.select(a, io, &env, target, null, false, "rocm", &problem)).?.dir == null);
    try tmp.dir.createDirPath(io, "override");
    const explicit = try std.fs.path.join(a, &.{ root, "override" });
    const chosen = (try drafter.select(a, io, &env, target, explicit, false, "cuda", &problem)).?;
    try t.expect(!chosen.automatic);
    try t.expectEqualStrings(explicit, chosen.dir.?);
    try t.expectEqualStrings(explicit, chosen.name.?);
    try t.expectEqualStrings("", problem);
    try t.expect((try drafter.select(a, io, &env, target, "example/missing-drafter", false, "cuda", &problem)) == null);
    try t.expect(std.mem.indexOf(u8, problem, "run tensorfold pull example/missing-drafter first") != null);
    problem = "";
    for ([_][]const u8{ "../../outside", "ffffffffffffffffffffffffffffffffffffffff" }) |ref| {
        try tmp.dir.writeFile(io, .{ .sub_path = cache_repo ++ "/refs/main", .data = ref });
        try t.expect((try drafter.select(a, io, &env, target, null, false, "cuda", &problem)).?.dir == null);
        try t.expectEqualStrings("", problem);
    }
}

const Fake = struct {
    const Mode = enum { opens, refuses, fails };
    var calls: u32 = 0;
    var lines: u32 = 0;
    var line: [256]u8 = undefined;
    var line_len: usize = 0;

    /// 9 with a drafter in `mode`'s way, 7 without one; a refusal says why in `problem`.
    fn open(mode: Mode, args: cli.Args, problem: *[]const u8) error{DrafterFailed}!?u8 {
        calls += 1;
        if (args.drafter == null) return 7;
        return switch (mode) {
            .opens => 9,
            .refuses => {
                problem.* = "it does not fit";
                return null;
            },
            .fails => error.DrafterFailed,
        };
    }

    fn capture(_: *anyopaque, text: []const u8) void {
        lines += 1;
        line_len = @min(text.len, line.len);
        @memcpy(line[0..line_len], text[0..line_len]);
    }
};

test "serve drafter: a paired drafter that does not open leaves the target serving alone with one line" {
    log.Test.sink = .{ .ctx = undefined, .call = Fake.capture };
    defer log.Test.sink = null;
    const cases = [_]struct { mode: Fake.Mode, automatic: bool, got: ?u8, calls: u32, line: []const u8 }{
        .{ .mode = .opens, .automatic = true, .got = 9, .calls = 1, .line = "" },
        .{ .mode = .refuses, .automatic = true, .got = 7, .calls = 2, .line = "[tensorfold] drafter example/d did not open (it does not fit): serving without it\n" },
        .{ .mode = .fails, .automatic = true, .got = 7, .calls = 2, .line = "[tensorfold] drafter example/d did not open (DrafterFailed): serving without it\n" },
        .{ .mode = .refuses, .automatic = false, .got = null, .calls = 1, .line = "" },
    };
    for (cases) |c| {
        Fake.calls = 0;
        Fake.lines = 0;
        var args: cli.Args = .{ .drafter = "/cache/d", .drafter_name = "example/d" };
        var problem: []const u8 = "";
        try t.expectEqual(c.got, try drafter.openOrPlain(Fake.open, .{c.mode}, &args, c.automatic, &problem));
        try t.expectEqual(c.calls, Fake.calls);
        try t.expectEqual(@as(u32, @intFromBool(c.line.len > 0)), Fake.lines);
        if (c.line.len > 0) try t.expectEqualStrings(c.line, Fake.line[0..Fake.line_len]);
        try t.expectEqual(c.calls == 2, args.drafter == null and args.drafter_name == null);
        try t.expectEqualStrings(if (c.got == null) "it does not fit" else "", problem);
    }
    var named: cli.Args = .{ .drafter = "d" };
    var problem: []const u8 = "";
    try t.expectError(error.DrafterFailed, drafter.openOrPlain(Fake.open, .{Fake.Mode.fails}, &named, false, &problem));
    try t.expectEqualStrings("d", named.drafter.?);
}
