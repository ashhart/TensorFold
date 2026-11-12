//! serve_lanes over the CPU twin: replies through LaneHost equal the twin's serial greedy, together and drafted or not.
const std = @import("std");
const api = @import("engine_api");
const twin = @import("draft/twin.zig");
const iface = @import("draft/iface.zig");
const calib_env = @import("draft/calib_env.zig");
const calib_lanes = @import("calib_lanes.zig");
const serve_lanes = @import("serve_lanes.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

/// A reply's tokens and finish, written on LaneHost's thread and read on the test's.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    reason: ?api.Reason = null,
    drafted: u64 = 0,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |ts| b.out.appendSlice(gpa, ts) catch {},
            .finished => |fin| {
                b.done = true;
                b.reason = fin.reason;
                b.drafted = fin.stats.drafted;
            },
            else => {},
        }
    }

    fn wait(b: *Box) !void {
        for (0..60_000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    fn deinit(b: *Box) void {
        b.out.deinit(gpa);
    }
};

/// The twin as a serve_lanes.Model: its CPU target and DSpark pass; the cost table is the defaults, never measured.
const TwinModel = struct {
    t: *twin.Twin,
    d: *twin.Drafter,

    fn open(slots: u32) !serve_lanes.Model {
        const b = try gpa.create(TwinModel);
        errdefer gpa.destroy(b);
        b.t = try twin.Twin.init(gpa, .{}, slots);
        errdefer b.t.deinit();
        b.d = try twin.Drafter.init(gpa, b.t, slots);
        return .{ .ptr = b, .name = "dsv41-twin", .slots = slots, .limit = 4096, .world = 1, .prefill_step = 16, .forward_rows = 16, .shape = b.d.shape, .target = target, .pass = pass, .calibrate = calibrate, .close = close, .calib = .{ .mode = "default" } };
    }

    fn target(ctx: *anyopaque) iface.Target {
        const b: *TwinModel = @ptrCast(@alignCast(ctx));
        return b.t.target();
    }

    fn pass(ctx: *anyopaque) ?iface.Pass {
        const b: *TwinModel = @ptrCast(@alignCast(ctx));
        return b.d.pass();
    }

    fn calibrate(_: *anyopaque, _: std.mem.Allocator, _: std.Io, _: []const u8, _: calib_env.Knobs, _: calib_env.Shape) anyerror!calib_lanes.Boot {
        return error.Unmeasured; // mode "default" never measures
    }

    fn close(ctx: *anyopaque) void {
        const b: *TwinModel = @ptrCast(@alignCast(ctx));
        b.d.deinit();
        b.t.deinit();
        gpa.destroy(b);
    }
};

const max_new = 40;
const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 }, &.{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 } };

/// Each request on its own, then all together; every reply returned in `out` (gpa-owned).
fn serve(e: api.Engine, drafts: bool, together: bool, out: *[prompts.len][]u32, drafted: *u64) !void {
    var boxes: [prompts.len]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var reqs: [prompts.len]api.Request = undefined;
    for (&reqs, prompts) |*r, p| r.* = .{ .prompt = p, .max_tokens = max_new, .drafts = drafts };
    for (&reqs, &boxes, 0..) |*r, *b, i| {
        try e.submit(@intCast(i + 1 + @as(usize, if (together) 100 else 0) + @as(usize, if (drafts) 0 else 1000)), r, .{ .ctx = b, .event = Box.event });
        if (!together) try b.wait();
    }
    for (&boxes, out) |*b, *o| {
        try b.wait();
        try std.testing.expectEqual(api.Reason.length, b.reason.?);
        drafted.* += b.drafted;
        o.* = try b.out.toOwnedSlice(gpa);
    }
}

fn freeAll(xs: *[prompts.len][]u32) void {
    for (xs) |x| gpa.free(x);
}

test "serve lanes: replies through LaneHost over the twin, concurrent == solo == serial, drafted == draft:false" {
    const t = std.testing;
    // the reference: the twin's serial greedy decode of each prompt, no lanes
    var serial: [prompts.len][]u32 = undefined;
    {
        const ref = try twin.Twin.init(gpa, .{}, 1);
        defer ref.deinit();
        for (&serial, prompts) |*s, p| s.* = try ref.serial(0, p, max_new, null, null);
    }
    defer freeAll(&serial);
    const s = try serve_lanes.Served.open(gpa, io, try TwinModel.open(4), ".", null);
    defer serve_lanes.Served.close(s);
    const e = s.engine();
    var drafted: u64 = 0;
    var plain_drafted: u64 = 0;
    var solo: [prompts.len][]u32 = undefined;
    try serve(e, true, false, &solo, &drafted);
    defer freeAll(&solo);
    var together: [prompts.len][]u32 = undefined;
    try serve(e, true, true, &together, &drafted);
    defer freeAll(&together);
    var plain: [prompts.len][]u32 = undefined;
    try serve(e, false, true, &plain, &plain_drafted);
    defer freeAll(&plain);
    for (serial, solo, together, plain) |want, a, b, c| {
        try t.expectEqualSlices(u32, want, a);
        try t.expectEqualSlices(u32, want, b);
        try t.expectEqualSlices(u32, want, c);
    }
    try t.expect(drafted > 0); // the DSpark pass drafted: the drafted replies were not one token a round
    try t.expectEqual(@as(u64, 0), plain_drafted);
}
