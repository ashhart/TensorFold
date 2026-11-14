//! The prompt cache under CUDA rules: no fork marks unless a family opts in, and where they sit when it does.
const std = @import("std");
const api = @import("engine_api");
const cacheRules = @import("cuda_snaps.zig").cacheRules;

/// Copies that hold nothing: a store keys its states by tokens and position, all that its plans read.
const Held = struct {
    var one: u8 = 0;
    const vtable: api.prompt_cache.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
        return &one;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
    fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
};

const Asked = struct { from: [3]u32 = .{ 0, 0, 0 }, marks: [3][]const u32 = undefined, kept: []u32 = &.{} };

/// Three questions of `q` tokens on one document under `F`'s rules: where each resumes, its marks, every state kept.
fn questions(comptime F: type, a: std.mem.Allocator, sys: u32, doc: u32, q: u32, rewind: bool) !Asked {
    var store = api.prompt_cache.Store.init(std.testing.allocator, .{ .ptr = &Held.one, .vtable = &Held.vtable }, cacheRules(F), 1 << 30);
    defer store.deinit();
    var asked: Asked = .{};
    for (0..3) |i| {
        const prompt = try a.alloc(u32, sys + doc + q + 2); // the question, then the template's two-token tail
        for (prompt, 0..) |*t, j| t.* = @intCast(if (j >= sys + doc and j < sys + doc + q) 1_000_000 * (i + 1) + j else j + 1);
        const p = try store.beginRewind(a, prompt, sys + doc + q, if (rewind) sys else 0, &.{}, &.{}, null, &.{});
        for (p.marks) |m| try std.testing.expect(store.keep(prompt, m, null, &.{}, &.{}));
        asked.from[i] = p.from;
        asked.marks[i] = p.marks;
    }
    asked.kept = try a.alloc(u32, store.entries.items.len);
    for (asked.kept, store.entries.items) |*at, e| at.* = e.at;
    std.mem.sort(u32, asked.kept, {}, std.sort.asc(u32));
    return asked;
}

test "snap fork: without it a family keeps the states it kept before the rule, the stable mark at the part" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Other = struct {};
    const Grid = struct {
        pub const snap_grid: u32 = 2048;
    };
    try std.testing.expect(!cacheRules(Other).fork and !cacheRules(Grid).fork);
    // A document re-sent with a new question: each pass misses and keeps only its history's mark.
    const missed = try questions(Other, a, 4, 1000, 2, false);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0 }, &missed.from);
    for (missed.marks) |m| try std.testing.expectEqualSlices(u32, &.{1006}, m);
    try std.testing.expectEqualSlices(u32, &.{ 1006, 1006, 1006 }, missed.kept);
    // Resuming the rewind state, the stable mark sits at the part, keyed with the question's first token.
    const resumed = try questions(Other, a, 300, 1000, 300, true);
    try std.testing.expectEqualSlices(u32, &.{ 0, 299, 299 }, &resumed.from);
    try std.testing.expectEqualSlices(u32, &.{ 299, 1600 }, resumed.marks[0]);
    for (resumed.marks[1..]) |m| try std.testing.expectEqualSlices(u32, &.{ 1300, 1600 }, m);
    try std.testing.expectEqualSlices(u32, &.{ 299, 1300, 1300, 1600, 1600, 1600 }, resumed.kept);
}

test "snap fork: a family that opts in keeps where the questions part, and the next question resumes there" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Fork = struct {
        pub const snap_fork = true;
    };
    const GridFork = struct {
        pub const snap_grid: u32 = 4;
        pub const snap_fork = true;
    };
    try std.testing.expect(cacheRules(Fork).fork and cacheRules(GridFork).fork);
    // A miss keeps one before the part, 3 tokens from the history's mark (under min_gap), and the next resumes it.
    const missed = try questions(Fork, a, 4, 1000, 2, false);
    try std.testing.expectEqualSlices(u32, &.{ 1003, 1006 }, missed.marks[1]);
    try std.testing.expectEqual(@as(u32, 1003), missed.from[2]);
    // Resuming the rewind state, the stable mark sits one before the part, keyed with the document alone.
    const resumed = try questions(Fork, a, 300, 1000, 300, true);
    try std.testing.expectEqualSlices(u32, &.{ 1299, 1600 }, resumed.marks[1]);
    try std.testing.expectEqual(@as(u32, 1299), resumed.from[2]);
    // A planned family keeps the fork on its grid start (16), however near the history's (20).
    const planned = try questions(GridFork, a, 4, 16, 2, false);
    try std.testing.expectEqualSlices(u32, &.{ 16, 20 }, planned.marks[1]);
    try std.testing.expectEqual(@as(u32, 16), planned.from[2]);
}

test "GLM warm turns: only the request-start opt-in turns them on, with every message start a cut" {
    const snaps = @import("cuda_snaps.zig");
    const env = @import("sys").env;
    const Family = struct {
        pub const snap_grid: u32 = 2048;
        pub const request_starts = true;
        pub const request_starts_switch = "TF_GLM_WARM_TEST_STARTS";
        pub const request_starts_warm_quiet: u64 = 123;
    };
    defer _ = env.unsetenv(Family.request_starts_switch);
    for ([_]?[:0]const u8{ null, "0", "1" }) |v| {
        _ = if (v) |x| env.setenv(Family.request_starts_switch, x, 1) else env.unsetenv(Family.request_starts_switch);
        const on = v != null and std.mem.eql(u8, v.?, "1");
        try std.testing.expectEqual(if (on) @as(u64, 123) else 0, snaps.warmQuiet(Family));
        try std.testing.expectEqual(on, snaps.warmMessages(Family));
        try std.testing.expectEqual(if (on) @as(u32, 1) else 256, snaps.requestMinimum(Family));
        try std.testing.expectEqual(on, snaps.cacheRules(Family).warm != .off);
    }
    const Other = struct { // Nemotron's shape: its own quiet, warm whenever a cache is kept, its cuts its own
        pub const warm_quiet: u64 = 777;
    };
    try std.testing.expectEqual(@as(u64, 777), snaps.warmQuiet(Other));
    try std.testing.expect(!snaps.warmMessages(Other));
    try std.testing.expectEqual(@as(u64, 150 * std.time.ns_per_ms), snaps.warmQuiet(@import("nemotron").native));
    const glm: [:0]const u8 = "TF_GLM_REQUEST_STARTS";
    const saved = if (@import("sys").getenv(glm)) |x| try std.testing.allocator.dupeSentinel(u8, x, 0) else null;
    defer if (saved) |x| {
        _ = env.setenv(glm, x, 1);
        std.testing.allocator.free(x);
    } else {
        _ = env.unsetenv(glm);
    };
    inline for (@import("cuda.zig").registry) |F| if (comptime std.mem.eql(u8, F.model_type, "glm5_next")) {
        _ = env.unsetenv(glm); // GLM's default: no warm pass, its cuts and 256-token spacing as on main
        try std.testing.expect(snaps.warmQuiet(F) == 0 and !snaps.warmMessages(F) and snaps.cacheRules(F).warm == .off);
        try std.testing.expectEqual(@as(u32, 256), snaps.requestMinimum(F));
        _ = env.setenv(glm, "1", 1);
        try std.testing.expect(snaps.warmQuiet(F) == 150 * std.time.ns_per_ms and snaps.warmMessages(F));
        try std.testing.expect(snaps.requestMinimum(F) == 1 and snaps.cacheRules(F).warm == .waits);
    };
}
