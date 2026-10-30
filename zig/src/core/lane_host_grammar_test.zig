//! A request's grammar the server compiled reaches the lane thread as is: no compile there, the same masked reply.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const fake = lanes.grammar_fake;
const a = std.testing.allocator;

const Box = struct {
    reason: api.Reason = .failed,
    tokens: std.ArrayList(u32) = .empty,
    done: std.atomic.Value(bool) = .init(false),
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(a, t) catch {},
            .finished => |f| {
                b.reason = f.reason;
                b.done.store(true, .release);
            },
            else => {},
        }
    }
    fn wait(b: *Box) !api.Reason {
        const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (!b.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began > 2 * std.time.ns_per_s) return error.RequestTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.reason;
    }
};

/// One reply under the request's structure; returns its tokens (owned) after `finished`.
fn reply(host: *api.LaneHost, id: api.Id, st: api.Structure) ![]u32 {
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 8, .structure = st };
    var box: Box = .{};
    try host.engine().submit(id, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, try box.wait());
    return box.tokens.toOwnedSlice(a);
}

test "a compiled grammar handed in compiles nothing on the lane thread and masks the reply as a compile does" {
    const old = fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    var c = fake.compiler();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    var err: [512]u8 = undefined;
    const text = "{\"type\":\"object\"}";
    const g = try c.compile(std.testing.io, .json_schema, text, &err); // the server's compile
    defer g.free();
    const handed = try reply(&host, 1, .{ .kind = .json_schema, .text = text, .compiler = &c, .compiled = g });
    defer a.free(handed);
    try std.testing.expectEqual(@as(usize, 1), fake.compiles);
    try std.testing.expectEqual(@as(usize, 0), fake.frees); // the handle stays the server's
    const compiled = try reply(&host, 2, .{ .kind = .json_schema, .text = text, .compiler = &c });
    defer a.free(compiled);
    try std.testing.expectEqual(@as(usize, 2), fake.compiles); // without a handle the lane thread compiles, as before
    try std.testing.expectEqual(@as(usize, 8), handed.len);
    try std.testing.expectEqualSlices(u32, compiled, handed);
    try std.testing.expectEqual(@as(usize, 0), fake.arms + fake.narrows); // grammar_budget off: never consulted
    const budgeted = try reply(&host, 3, .{ .kind = .json_schema, .text = text, .compiler = &c, .compiled = g, .budget = true });
    defer a.free(budgeted);
    try std.testing.expectEqual(@as(usize, 1), fake.arms); // armed once, before its first token
    try std.testing.expect(fake.narrows > 0);
    try std.testing.expectEqualSlices(u32, handed, budgeted); // the stand-in narrows nothing, so nothing else changed
}
