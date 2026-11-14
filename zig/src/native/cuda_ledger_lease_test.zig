//! #1943 step 2: with a prepaid stream the lane host admits the window plan's full-window requests, not one fewer.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const ledger = @import("cuda_ledger.zig");
const PerRequest = @import("core").kv_pool.PerRequest;
const gpa = std.testing.allocator;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: usize = 0,
    done: ?api.Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn state(b: *@This()) struct { usize, ?api.Reason } {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return .{ b.tokens, b.done };
    }
    fn startsWithin(b: *@This(), ms: usize) bool {
        for (0..ms) |_| {
            if (b.state()[0] > 0) return true;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return false;
    }
    fn endsWithin(b: *@This(), ms: usize) ?api.Reason {
        for (0..ms) |_| {
            if (b.state()[1]) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return null;
    }
};

const Bare = struct {}; // no prompt cache: the room is all streams
const each: PerRequest = .{ .layout = .{ .fixed_bytes = 1000, .token_bytes = 10, .page_tokens = 0, .snap_bytes = 0 }, .lookahead = 8, .scratch = 400 };
var unused: u8 = 0;

test "kv ledger: a prepaid stream admits as many full-window requests as the window plan, the next one waits" {
    const o: api.Open = .{ .dir = "", .model_type = "f", .prompt_cache_gib = 0, .lanes = 4 };
    const full: u64 = 1000 + (4096 + 8) * 10;
    const room = 400 + 2 * full + 5; // the window plan: the engine's own and two more
    const loaded = .{ .ctx = @as(*anyopaque, &unused), .stream_bytes = @as(u64, 7), .per_request = @as(?PerRequest, each), .free_streams = @as(u32, 1) };
    var short: ?ledger.Short = null;
    const p = ledger.plan(Bare, loaded, o, .{ .room = room, .context = 4096, .host_budget = 0, .host_cache = false }, &short).?;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4, .ledger = ledger.info(&p) });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop(); // a failed check must not free the engine under a running thread
    const e = host.engine();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const long: api.Request = .{ .prompt = &prompt, .max_tokens = 100000 }; // past the context: a full window each
    var boxes: [4]Box = .{ .{}, .{}, .{}, .{} };
    for (&boxes, 1..) |*b, id| try e.submit(id, &long, .{ .ctx = b, .event = Box.event });
    for (boxes[0..3]) |*b| try std.testing.expect(b.startsWithin(2000)); // main: the third waited for bytes
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    try std.testing.expectEqual(@as(usize, 0), boxes[3].state()[0]); // the fourth waits, not refused
    try std.testing.expectEqual(@as(?api.Reason, null), boxes[3].state()[1]);
    e.cancel(1);
    try std.testing.expectEqual(@as(?api.Reason, .cancelled), boxes[0].endsWithin(2000));
    try std.testing.expect(boxes[3].startsWithin(2000)); // a lease came back
    for (2..5) |id| e.cancel(id);
    for (boxes[1..]) |*b| try std.testing.expectEqual(@as(?api.Reason, .cancelled), b.endsWithin(2000));
    host.stop();
    stopped = true;
    try std.testing.expect(host.kv.?.idle()); // the standing lease is not left at stop
}
