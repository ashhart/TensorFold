//! The same-document/new-question request pattern through the real lane host and its recording backend.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = api.prompt_cache;
const gpa = std.testing.allocator;
const io = std.testing.io;

const Snaps = struct {
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
};
const Box = struct {
    mutex: std.Io.Mutex = .init,
    changed: std.Io.Condition = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?api.Reason = null,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .prefilled => |n| b.cached = n,
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch unreachable,
            .finished => |f| b.done = f.reason,
            else => {},
        }
        b.changed.broadcast(io);
    }
    fn wait(b: *Box) !void {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        while (b.done == null) b.changed.waitUncancelable(io, &b.mutex);
        try std.testing.expectEqual(api.Reason.length, b.done.?);
    }
};

test "lane interior state reports cached tokens, fresh output and budgeted occupancy" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 128 * 1024 + 64);
    for (prompt, 0..) |*t, i| t.* = @intCast(i % 19);
    const edited = try a.dupe(u32, prompt);
    edited[edited.len - 512] += 1;
    // Flash Next CUDA's loaded pass and Qwen27 Metal's unplanned pass, both on the shared store path.
    for ([_]pc.Rules{ .{ .chunk = 1536, .chunk_starts = false, .lookahead = 1 }, .{ .chunk = 2048, .warm = .on, .chunk_marks = false } }) |rules| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, rules, 2 << 20);
        defer store.deinit();
        var host = api.LaneHost.init(gpa, io, &core, .{ .lanes = 2, .prompt_cache = true });
        host.cache = &store;
        try host.start();
        var stopped = false;
        defer if (!stopped) host.stop();
        var boxes: [2]Box = .{ .{}, .{} };
        defer for (&boxes) |*b| b.tokens.deinit(gpa);
        const e = host.engine();
        var starts: std.ArrayList(u32) = .empty;
        defer starts.deinit(a);
        if (!rules.chunk_marks) {
            var cut: u32 = rules.chunk;
            while (cut < prompt.len) : (cut += rules.chunk) try starts.append(a, cut);
        }
        const first: api.Request = .{ .prompt = prompt, .max_tokens = 4, .history_len = @intCast(prompt.len - 8), .rewind_len = 16, .user_end = @intCast(prompt.len - 8), .chunks = starts.items };
        try e.submit(1, &first, .{ .ctx = &boxes[0], .event = Box.event });
        try boxes[0].wait();
        const followup: api.Request = .{ .prompt = edited, .max_tokens = 4, .history_len = @intCast(edited.len - 8), .rewind_len = 16, .user_end = @intCast(prompt.len - 8), .chunks = starts.items };
        try e.submit(2, &followup, .{ .ctx = &boxes[1], .event = Box.event });
        try boxes[1].wait();
        e.drain(); // joins the cache owner while the status reader's storage remains alive
        var status: api.Status = .{};
        e.status(&status, &.{});
        host.stop();
        stopped = true;
        try std.testing.expectEqual(@as(?u32, 0), boxes[0].cached);
        const expected: u32 = if (rules.chunk == 1536) 129039 else 129024;
        try std.testing.expectEqual(@as(?u32, expected), boxes[1].cached);
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, edited);
        for (boxes[1].tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
        try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
        try std.testing.expect(store.held + store.spare() <= store.budget);
        const published = status.prompt_cache.?;
        try std.testing.expectEqual(@as(u64, store.entries.items.len), published.states);
        try std.testing.expectEqual(store.held + store.spare(), published.bytes);
    }
}
