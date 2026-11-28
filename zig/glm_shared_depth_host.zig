//! Real GLM facts and lane scheduling with a history-dependent target and counted native head requests.
const std = @import("std");
const glm = @import("glm");
const lanes = @import("lanes");
const policy = glm.native.shared_depth;
const a = std.testing.allocator;
const be = lanes.backend;

fn facts(ceiling: u32, world: usize, copies: bool, block: bool) lanes.Model {
    var native: glm.engine.Engine = undefined;
    native.mtp = true;
    native.drafter = if (block) @ptrFromInt(16) else null;
    var backend: glm.lanes.Lanes = .{ .x = .{ .e = &native, .link = undefined } };
    return policy.apply(backend.factsWithCopies(copies), ceiling, world, false);
}

const Counted = struct {
    target: lanes.fake.Fake = .{ .gpa = a, .head_accepts = 1000 },
    core: ?*lanes.Engine = null,
    first: ?*lanes.Stream = null,
    reject: ?*lanes.Stream = null,
    maximum: [2]u32 = .{ 0, 0 },
    trace: std.ArrayList(u32) = .empty,
    widest: usize = 0,

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("target", target);
        for (requests) |r| {
            if (c.core) |core| if (!core.alone) {
                const i: usize = if (r.stream == c.first) 0 else 1;
                c.maximum[i] = @max(c.maximum[i], r.depth);
                try c.trace.append(a, r.depth);
            };
            target.head_accepts = if (r.stream == c.reject) 0 else 1000;
            try target.backend().vtable.draft(ptr, &.{r});
        }
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("target", target);
        var rows: usize = 0;
        for (windows) |w| {
            rows += w.rows();
            try std.testing.expect(w.held <= 15);
        }
        try std.testing.expect(rows <= 32);
        c.widest = @max(c.widest, rows);
        try target.backend().vtable.verify(ptr, windows, out);
    }
};

const Result = struct {
    maximum: [2]u32,
    widest: usize,
    trace: []u32,
    tokens: [2][]u32,
    fn deinit(r: Result) void {
        a.free(r.trace);
        for (r.tokens) |t| a.free(t);
    }
};
const Options = struct { ceiling: u32 = 0, streams: usize = 2, seed: ?u64 = null, reject: bool = false, restored: bool = false, replace: bool = false, drafts: bool = true, tokens: u32 = 128 };
const prompts = [_][]const u32{ &.{ 7, 13, 29, 31 }, &.{ 3, 19, 41, 59 } };

fn run(o: Options) !Result {
    var m = facts(o.ceiling, 2, false, false);
    var costs: [32]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, n| c.* = .{ .width = @intCast(n), .ms = 1 };
    m.window_costs = &costs;
    m.shared_costs = &costs;
    var cfg = try lanes.Config.init(a, m, 32, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var counted: Counted = .{};
    defer counted.target.deinit();
    defer counted.trace.deinit(a);
    var table = counted.target.backend().vtable.*;
    table.draft = Counted.draft;
    table.verify = Counted.verify;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = &counted.target, .vtable = &table }, clock.clock());
    defer core.deinit();
    counted.core = &core;
    var streams: [2]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var saved: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&saved) |*s| s.deinit(a);
    for (streams[0..o.streams], 0..) |*s, i| {
        try saved[i].appendSlice(a, prompts[i][0..2]);
        s.* = try lanes.Stream.init(a, .{ .id = if (i == 0) "first" else "second", .prompt = prompts[i], .max_new = o.tokens, .drafts = o.drafts, .sampling = if (o.seed) |seed| .{ .seed = seed + i } else null, .reuse = if (o.restored) .{ .saved = &saved[i], .at = 2 } else .{} });
        made += 1;
        if (i == 0) counted.first = s;
        if (i == 1 and o.reject) counted.reject = s;
        try core.addStream(s);
        try std.testing.expectEqual(@as(u32, if (o.restored) 2 else 0), s.cached);
    }
    var replaced = false;
    while (core.activeCount() > 0) {
        try std.testing.expect(core.steps < 1024);
        if (o.replace and !replaced and core.steps == 4) {
            core.discard(&streams[1]);
            streams[1].deinit(a);
            streams[1] = try lanes.Stream.init(a, .{ .id = "replacement", .prompt = prompts[1], .max_new = o.tokens, .sampling = if (o.seed) |seed| .{ .seed = seed + 1 } else null });
            try std.testing.expectEqual(@as(?u32, null), streams[1].shared_draft_limit);
            try core.addStream(&streams[1]);
            replaced = true;
        }
        try core.step();
        for (core.live.items) |s| {
            const cache = counted.target.cache(s) orelse return error.NoCache;
            try std.testing.expect(cache.len == s.context.items.len or cache.len + 1 == s.context.items.len);
            try std.testing.expectEqualSlices(u32, s.context.items[0..cache.len], cache);
        }
    }
    try std.testing.expect(!o.replace or replaced);
    var result: Result = .{ .maximum = counted.maximum, .widest = counted.widest, .trace = try a.dupe(u32, counted.trace.items), .tokens = .{ try a.dupe(u32, streams[0].emitted()), if (o.streams == 2) try a.dupe(u32, streams[1].emitted()) else try a.alloc(u32, 0) } };
    errdefer result.deinit();
    for (result.tokens[0..o.streams], 0..) |got, i| {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(a);
        try history.appendSlice(a, prompts[i]);
        const sampling: ?lanes.sampling.Sampling = if (o.seed) |seed| .{ .seed = seed + i } else null;
        for (0..o.tokens) |_| try history.append(a, lanes.fake.next(history.items, sampling, history.items.len));
        try std.testing.expectEqualSlices(u32, history.items[prompts[i].len..], got);
    }
    return result;
}

test "glm shared depth: perfect shared native heads grow beyond three only with the switch" {
    for ([_]u32{ 7, 15 }) |limit| for ([_]?u64{ null, 1234 }) |seed| {
        const old = try run(.{ .seed = seed });
        defer old.deinit();
        const grown = try run(.{ .ceiling = limit, .seed = seed });
        defer grown.deinit();
        try std.testing.expectEqual(@as(u32, 3), old.maximum[0]);
        try std.testing.expect(grown.maximum[0] > 3);
        for (grown.maximum) |maximum| try std.testing.expect(maximum <= limit);
        try std.testing.expect(grown.widest <= 32);
        for (old.tokens, grown.tokens) |before, after| try std.testing.expectEqualSlices(u32, before, after);
    };
}

test "glm shared depth: rejects stay local while restores and replacements keep the fresh reply" {
    for ([_]?u64{ null, 9 }) |seed| {
        const asymmetric = try run(.{ .ceiling = 15, .seed = seed, .reject = true, .restored = true, .tokens = 256 });
        defer asymmetric.deinit();
        try std.testing.expect(asymmetric.maximum[0] > 3);
        try std.testing.expect(asymmetric.maximum[1] <= 3);
        const replaced = try run(.{ .ceiling = 7, .seed = seed, .replace = true, .tokens = 256 });
        defer replaced.deinit();
        const solo = try run(.{ .ceiling = 15, .seed = seed, .streams = 1 });
        defer solo.deinit();
        const plain = try run(.{ .seed = seed, .streams = 1, .drafts = false });
        defer plain.deinit();
        try std.testing.expectEqualSlices(u32, solo.tokens[0], plain.tokens[0]);
        const short = try run(.{ .ceiling = 15, .seed = seed, .tokens = 2 });
        defer short.deinit();
    }
}

test "glm shared depth: actual environment and unsupported modes retain current facts" {
    const env = @import("sys").environment;
    for ([_]?[]const u8{ null, "", " \t" }) |blank| try std.testing.expectEqual(@as(?u64, null), try env.parse(policy.name, blank));
    for ([_]u64{ 0, 7, 15 }, [_][]const u8{ "0", "7", "15" }) |want, raw| try std.testing.expectEqual(@as(?u64, want), try env.parse(policy.name, raw));
    for ([_][]const u8{ "1", "3", "8", "16", "-1", "true", " 7", "7 " }) |bad| try std.testing.expectError(error.InvalidEnvironment, env.parse(policy.name, bad));
    for ([_]usize{ 1, 3 }) |world| try std.testing.expectEqual(@as(u32, 0), facts(15, world, false, false).shared_draft_growth);
    try std.testing.expectEqual(@as(u32, 0), facts(15, 2, true, false).shared_draft_growth);
    try std.testing.expectEqual(@as(u32, 0), facts(15, 2, false, true).shared_draft_growth);
    var no_drafts = facts(0, 2, false, false);
    no_drafts.mtp = false;
    try std.testing.expectEqual(@as(u32, 0), policy.apply(no_drafts, 15, 2, false).shared_draft_growth);
    try std.testing.expectEqual(@as(u32, 0), policy.apply(facts(0, 2, false, false), 15, 2, true).shared_draft_growth);
    var alone = facts(0, 2, false, false);
    alone.hidden_rows = false;
    try std.testing.expectEqual(@as(u32, 0), policy.apply(alone, 15, 2, false).shared_draft_growth);
}
