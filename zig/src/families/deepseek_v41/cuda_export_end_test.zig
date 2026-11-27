//! A handoff request cancelled in its prefill (a deadline, a disconnect): the lane host fails its announced stream.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const fabric = @import("fabric");
const serve = @import("cuda_export_serve.zig");
const cx = @import("cuda_export.zig");
const pc = api.prompt_cache;
const Handoff = api.LaneHost.Export.Handoff;
const t = std.testing;

const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const id: [16]u8 = @splat(0x7c);

/// The family's exporter over a real producer: `begin` announces the stream as cuda_export_lane.zig's does.
const Rig = struct {
    p: serve.Produce,
    host: *api.LaneHost = undefined,
    began: u32 = 0,
    fed: u32 = 0,
    stopped: u32 = 0,
    done: std.atomic.Value(bool) = .init(false),
    reason: ?api.Reason = null,

    fn exporter(r: *Rig) api.LaneHost.Export.Exporter {
        return .{ .ptr = r, .snapshots = .{ .ptr = r, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .exported = exported, .failed = failed, .begin = begin };
    }

    fn self(ptr: *anyopaque) *Rig {
        return @ptrCast(@alignCast(ptr));
    }

    fn begin(ptr: *anyopaque, h: Handoff, tokens: []const u32) void {
        const r = self(ptr);
        const built = cx.skeleton(t.allocator, h.id, tokens, cx.shapeOf(tokens.len, @intCast(tokens.len))) catch return;
        r.p.beginStream(h.id, built, .{ .ptr = r, .progress = progress, .drain = drain, .stop = stop }) catch return built.deinit(t.allocator);
        r.began += 1;
    }

    fn progress(ptr: *anyopaque, _: u64) void {
        self(ptr).fed += 1;
    }

    fn drain(_: *anyopaque) bool {
        return false;
    }

    fn stop(ptr: *anyopaque) void {
        self(ptr).stopped += 1;
    }

    fn failed(ptr: *anyopaque, h: Handoff, why: []const u8) void {
        self(ptr).p.failed(h.id, why);
    }

    fn exported(_: *anyopaque, _: Handoff, _: []const u32, _: pc.Saved) void {}

    fn bytes(_: *anyopaque, _: u32) u64 {
        return 0;
    }

    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
        return error.NoCutInThisTest;
    }

    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}

    fn drop(_: *anyopaque, _: pc.Saved) void {}

    /// The server's deadline or a disconnect, as the engine hears it: a cancel while the prompt prefills.
    fn cancelInPrefill(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
        if (chunk == 0) self(ctx).host.engine().cancel(1);
    }

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const r = self(ctx);
        switch (ev.*) {
            .finished => |f| {
                r.reason = f.reason;
                r.done.store(true, .release);
            },
            else => {},
        }
    }
};

test "handoff: a request cancelled in its prefill fails the stream it announced, releasing its bytes and its feeder" {
    var rig: Rig = .{ .p = .{ .gpa = t.allocator, .io = t.io, .source = undefined, .opts = .{ .link = "tf-test", .model = "dsv41-test" }, .table = .{ .gpa = t.allocator, .ttl_ns = std.time.ns_per_hour, .clock = fabric.words.nowNs } } };
    defer {
        var it = rig.p.held.valueIterator();
        while (it.next()) |h| {
            h.*.built.deinit(t.allocator);
            t.allocator.destroy(h.*);
        }
        rig.p.held.deinit(t.allocator);
        rig.p.table.deinit();
    }
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .prefill_hook = Rig.cancelInPrefill, .prefill_hook_ctx = &rig };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    rig.host = &host;
    host.exporter = rig.exporter();
    try host.start();
    defer host.stop();
    try host.engine().submit(1, &.{ .prompt = &prompt, .max_tokens = 8, .handoff = .{ .id = id } }, .{ .ctx = &rig, .event = Rig.event });
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!rig.done.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.HandoffFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    try t.expectEqual(api.Reason.cancelled, rig.reason.?);
    try t.expectEqual(@as(u32, 1), rig.began);
    const entry = rig.p.table.get(id).?;
    try t.expect(entry == .failed); // the decoder's next OPEN or PULL hears it, not WAIT until the export expires
    try t.expect(std.mem.indexOf(u8, entry.failed, "cancelled") != null);
    try t.expectEqual(@as(u64, 0), rig.p.table.held_bytes);
    try t.expectEqual(@as(u32, 1), rig.stopped);
}
