//! Simultaneous follow-ups claim their own reply-end state before any admission can reserve it away.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const a = std.testing.allocator;

const Reply = struct {
    mutex: std.Io.Mutex = .init,
    cached: ?u32 = null,
    tokens: [4]u32 = undefined,
    count: usize = 0,
    reason: ?api.Reason = null,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const r: *Reply = @ptrCast(@alignCast(ptr));
        r.mutex.lockUncancelable(std.testing.io);
        defer r.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |at| r.cached = at,
            .tokens => |tokens| for (tokens) |token| {
                if (r.count < r.tokens.len) r.tokens[r.count] = token;
                r.count += 1;
            },
            .finished => |done| r.reason = done.reason,
            else => {},
        }
    }
    fn wait(r: *Reply) !void {
        const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (true) {
            r.mutex.lockUncancelable(std.testing.io);
            const done = r.reason;
            r.mutex.unlock(std.testing.io);
            if (done) |reason| return std.testing.expectEqual(api.Reason.length, reason);
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began > 3 * std.time.ns_per_s)
                return error.FollowupTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

const Snapshots = struct {
    target: *lanes.fake.Fake,
    seed: []const u32 = &.{},
    growing: bool = false,
    late: ?struct { engine: api.Engine, request: *const api.Request, reply: *Reply, sent: bool = false } = null,
    fn of(ptr: *anyopaque) *Snapshots {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        if (of(ptr).late) |*late| if (!late.sent and at == 16) {
            late.sent = true;
            late.engine.submit(2, late.request, .{ .ctx = late.reply, .event = Reply.event }) catch unreachable;
        };
        return if (of(ptr).growing) 64 + 4 * @as(u64, at) else 100;
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const self = of(ptr);
        if (owner) |stream| return self.target.save(@ptrCast(@alignCast(stream)), at);
        const saved = try a.create(std.ArrayList(u32));
        errdefer a.destroy(saved);
        saved.* = .empty;
        try saved.appendSlice(a, self.seed[0..at]);
        return saved;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        of(ptr).target.drop(saved);
    }
};

fn followups(n: usize, late: bool, shorter: bool) !void {
    return followupsRules(n, late, shorter, .{ .min_prompt = 0 });
}

fn followupsRules(n: usize, late: bool, shorter: bool, rules: pc.Rules) !void {
    return followupsSized(n, late, shorter, rules, false);
}

fn followupsSized(n: usize, late: bool, shorter: bool, rules: pc.Rules, growing: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var snapshots: Snapshots = .{ .target = &target, .growing = growing };
    var store = pc.Store.init(a, .{ .ptr = &snapshots, .vtable = &.{ .bytes = Snapshots.bytes, .save = Snapshots.save, .restore = Snapshots.restore, .drop = Snapshots.drop } }, rules, if (growing) n * 112 + 16 else (n + @intFromBool(shorter)) * 100);
    defer store.deinit();
    var words: [16][24]u32 = undefined;
    var requests: [16]api.Request = undefined;
    var replies: [16]Reply = @splat(.{});
    for (0..n) |i| {
        for (&words[i], 0..) |*word, j| word.* = @intCast(1000 * (i + 1) + j);
        snapshots.seed = words[i][0..14];
        try std.testing.expect(store.keepBorn(snapshots.seed, 12, 12, null, &.{}, &.{}));
        requests[i] = .{ .prompt = words[i][0..20], .max_tokens = 4, .history_len = 16, .chunks = &.{ 4, 8, 12, 16 } };
    }
    if (shorter) {
        snapshots.seed = &words[1];
        try std.testing.expect(store.keepBorn(snapshots.seed, 12, 24, null, &.{}, &.{}));
    }
    try std.testing.expectEqual(@as(u64, @intCast(if (growing) n * 112 else (n + @intFromBool(shorter)) * 100)), store.held);
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = @intCast(@min(n, 8)) });
    host.cache = &store;
    const engine = host.engine();
    if (late) snapshots.late = .{ .engine = engine, .request = &requests[1], .reply = &replies[1] };
    for (0..if (late) 1 else n) |i| try engine.submit(i + 1, &requests[i], .{ .ctx = &replies[i], .event = Reply.event });
    try host.start();
    defer host.stop();
    for (replies[0..n], 0..) |*reply, i| {
        try reply.wait();
        try std.testing.expectEqual(@as(?u32, 12), reply.cached);
        try std.testing.expectEqual(@as(usize, 4), reply.count);
        var history: [24]u32 = undefined;
        @memcpy(history[0..20], words[i][0..20]);
        for (20..24) |at| history[at] = lanes.fake.next(history[0..at], null, at);
        try std.testing.expectEqualSlices(u32, history[20..24], &reply.tokens);
    }
    if (!growing) try std.testing.expectEqual(@as(u64, @intCast(2 * n + @intFromBool(shorter))), store.counts.kept);
    try std.testing.expectEqual(@as(u64, 0), store.counts.misses);
    try std.testing.expect(store.held <= store.budget);
    for (words[0..n]) |*conversation| {
        const kept = store.find(conversation[0..20], &.{ 4, 8, 12, 16 }, &.{}) orelse return error.ConversationLost;
        if (growing) try std.testing.expect(kept.at >= 12) else try std.testing.expectEqual(@as(u32, 16), kept.at);
    }
    try std.testing.expect(store.reservation_guard == null);
}

test "queued cache: 1 4 8 and 16 simultaneous follow-ups all resume with exactly one state each" {
    for ([_]usize{ 1, 4, 8, 16 }) |n| try followups(n, false, false);
}

test "queued cache: a follow-up arriving during lookup is protected before the reservation chooses a victim" {
    try followups(2, true, false);
}

test "queued cache: a shorter queued fork keeps its redundant resume prefix beside a newer end" {
    try followups(2, false, true);
}

test "queued cache: progress a full one-state store keeps a new conversation beside an idle one" {
    try coldBefore(false, true);
}

test "queued cache: a cold arrival cannot reserve away the following queued conversation" {
    try coldBefore(true, true);
}

fn coldBefore(protected: bool, history: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var snapshots: Snapshots = .{ .target = &target };
    var store = pc.Store.init(a, .{ .ptr = &snapshots, .vtable = &.{ .bytes = Snapshots.bytes, .save = Snapshots.save, .restore = Snapshots.restore, .drop = Snapshots.drop } }, .{ .min_prompt = 0, .min_gap = 0 }, 100);
    defer store.deinit();
    const idle = @as([24]u32, @splat(91));
    const words = @as([24]u32, @splat(82));
    snapshots.seed = &idle;
    try std.testing.expect(store.keepBorn(&idle, 12, 12, null, &.{}, &.{}));
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    host.cache = &store;
    const engine = host.engine();
    const request: api.Request = .{ .prompt = words[0..20], .max_tokens = 4, .history_len = if (history) 16 else 0, .shared_prefixes = if (history) &.{} else &.{16} };
    var reply: Reply = .{};
    var follow: Reply = .{};
    const follow_request: api.Request = .{ .prompt = idle[0..20], .max_tokens = 4, .history_len = 16 };
    try engine.submit(1, &request, .{ .ctx = &reply, .event = Reply.event });
    if (protected) try engine.submit(2, &follow_request, .{ .ctx = &follow, .event = Reply.event });
    try host.start();
    defer host.stop();
    try reply.wait();
    try std.testing.expectEqual(@as(?u32, 0), reply.cached);
    if (protected) {
        try follow.wait();
        try std.testing.expectEqual(@as(?u32, 12), follow.cached);
        try std.testing.expectEqual(@as(u32, 16), store.find(&idle, &.{}, &.{}).?.at);
        return;
    }
    const kept = store.find(words[0..20], &.{}, &.{}) orelse return error.NewConversationNotKept;
    try std.testing.expectEqual(@as(u32, 16), kept.at);
    try std.testing.expect(store.find(&idle, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u64, 100), store.held);
}

test "queued cache: mark-only families advance with warm, lookahead and planned histories" {
    const rules = [_]pc.Rules{
        .{ .min_prompt = 0, .min_gap = 0, .warm = .on },
        .{ .min_prompt = 0, .min_gap = 0, .lookahead = 1 },
        .{ .min_prompt = 0, .min_gap = 0, .lookahead = 1, .planned = true, .warm = .on },
        .{ .min_prompt = 0, .min_gap = 0, .lookahead = 1, .planned = true, .grid = 4 },
    };
    for (rules) |rule| {
        try followupsRules(1, false, false, rule);
        try followupsRules(4, false, false, rule);
    }
}

test "queued cache: progress a follow-up advances from twelve to sixteen in one state" {
    try followups(1, false, false);
}

test "queued cache: progress a shared-cut request without history replaces an idle state" {
    try coldBefore(false, false);
}

test "queued cache: growing states do not spend the same sixteen free bytes twice" {
    for ([_]usize{ 2, 4, 8 }) |n| try followupsSized(n, false, false, .{ .min_prompt = 0 }, true);
}
