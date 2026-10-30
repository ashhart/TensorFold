//! Actual native prompt cancellation and lane reuse without a GPU.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");

pub fn check(comptime Host: type, comptime serve: anytype) !void {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 10 };
    defer target.deinit();
    const Nothing = struct {
        fn release(_: *anyopaque) void {}
    };
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = null, .family = &target, .release = Nothing.release, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    try serve(h, std.testing.io, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 7, .hidden_rows = true }, 8, .{ .lanes = 2 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    const e = h.host.engine();
    try std.testing.expect(e.memory(false) == null); // no device to count
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?api.Reason = null,
        fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (ev.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) api.Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const Breaker = struct {
        engine: api.Engine,
        id: api.Id,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == 2 and c.id != 0) c.engine.cancel(c.id);
        }
    };
    const prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 40 };
    var breaker: Breaker = .{ .engine = e, .id = 1 };
    target.prefill_hook = Breaker.call;
    target.prefill_hook_ctx = &breaker;
    var broken: Box = .{};
    defer broken.tokens.deinit(gpa);
    try e.submit(1, &request, .{ .ctx = &broken, .event = Box.event });
    try std.testing.expectEqual(api.Reason.cancelled, broken.wait());
    try std.testing.expect(target.prefill_count <= 3); // stopped at the next chunk, not after all ten
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released
    try std.testing.expectEqual(@as(usize, 0), broken.tokens.items.len);
    breaker.id = 0;
    var restored: Box = .{};
    defer restored.tokens.deinit(gpa);
    try e.submit(2, &request, .{ .ctx = &restored, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, restored.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (restored.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 40), restored.tokens.items.len);
}
