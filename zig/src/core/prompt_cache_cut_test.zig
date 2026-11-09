//! Exact cut identity and warmed follow-ups on the real host with a deterministic test target.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const api = @import("engine_api.zig");
const lanes = @import("lanes");
const a = std.testing.allocator;

const Reply = struct {
    mutex: std.Io.Mutex = .init,
    done: ?api.Reason = null,
    cached: u32 = 0,
    ids: [4]u32 = undefined,
    n: usize = 0,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const r: *Reply = @ptrCast(@alignCast(ptr));
        r.mutex.lockUncancelable(std.testing.io);
        defer r.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |at| r.cached = at,
            .tokens => |ids| for (ids) |id| {
                if (r.n < r.ids.len) r.ids[r.n] = id;
                r.n += 1;
            },
            .finished => |f| r.done = f.reason,
            else => {},
        }
    }
    fn wait(r: *Reply) !void {
        const start = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (true) {
            r.mutex.lockUncancelable(std.testing.io);
            const done = r.done;
            r.mutex.unlock(std.testing.io);
            if (done) |v| return std.testing.expectEqual(api.Reason.length, v);
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - start > 3 * std.time.ns_per_s) return error.TurnTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};
const Snaps = struct {
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return target.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        target.drop(saved);
    }
};

fn turn(sample: ?lanes.Sampling, changed: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(a, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .planned = true, .grid = 4, .min_prompt = 0, .warm = .waits, .warm_usable = true, .cut_sensitive = true }, 300);
    defer store.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.cache = &store;
    host.warm_quiet = std.time.ns_per_ms;
    try host.start();
    defer host.stop();
    const words = [_]u32{ 2, 7, 1, 8, 3, 9, 4, 5, 6, 10 };
    const warm: api.Request = .{ .prompt = words[0..6], .max_tokens = 0, .background = true, .history_len = 6, .born = 4, .chunks = &.{ 2, 6 } };
    var first: Reply = .{};
    try host.engine().submit(1, &warm, .{ .ctx = &first, .event = Reply.event });
    try first.wait();
    const request: api.Request = .{ .prompt = &words, .max_tokens = 4, .history_len = 6, .sampling = sample, .chunks = if (changed) &.{ 3, 6, 10 } else &.{ 2, 6, 10 } };
    var reply: Reply = .{};
    try host.engine().submit(2, &request, .{ .ctx = &reply, .event = Reply.event });
    try reply.wait();
    try std.testing.expectEqual(@as(u32, if (changed) 0 else 6), reply.cached);
    try std.testing.expectEqual(@as(usize, 4), reply.n);
    var expected: [14]u32 = undefined;
    @memcpy(expected[0..words.len], &words);
    for (10..14) |i| expected[i] = lanes.fake.next(expected[0..i], sample, i);
    try std.testing.expectEqualSlices(u32, expected[10..14], &reply.ids);
    try std.testing.expect(store.held <= store.budget);
}

test "turn ends: a warmed off-grid endpoint resumes deepest with greedy and seeded fresh-equal replies" {
    try turn(null, false);
    try turn(.{ .seed = 123, .temperature = 0.7, .top_k = 5 }, false);
}

test "turn ends: different cuts before the same endpoint refuse RAM reuse with fresh-equal replies" {
    try turn(null, true);
    try turn(.{ .seed = 123, .temperature = 0.7, .top_k = 5 }, true);
}
