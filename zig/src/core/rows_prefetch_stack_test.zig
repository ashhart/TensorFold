//! Read-ahead and write-behind share publication rules while keeping independent I/O workers.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const rows = @import("rows_store.zig");
const fetch = @import("lane_host_fetch.zig");
const prefetch = @import("rows_prefetch.zig");
const fixture = @import("rows_async_fixture.zig");
const t = std.testing;

test "prefetch: a queued write publishes before the next request reads its exact image ahead" {
    const x = try fixture.Fixture.open("fetch-queued", .evicted, false, 1 << 20);
    defer x.close();
    var gate: fixture.Gate = .{};
    defer gate.release.store(true, .release);
    x.r.file_io = gate.io();
    const first = fixture.prompt(1);
    try x.keep(&first, 900);
    x.s.remove(0);
    try fixture.Gate.wait(&gate.entered);
    const next = fixture.prompt(2);
    try x.keep(&next, 900);
    const kept: *@import("rows_fake.zig").Kept = @ptrCast(@alignCast(x.s.entries.items[0].saved));
    const expected = try t.allocator.dupe(u8, kept.img);
    defer t.allocator.free(expected);
    x.s.remove(0);
    try t.expectEqual(@as(usize, 2), x.r.worker.?.pending());
    var core: lanes.Engine = undefined;
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 2 });
    host.cache = &x.s;
    defer host.queued.deinit(t.allocator);
    defer host.admitted.deinit(t.allocator);
    const request: api.Request = .{ .prompt = &next, .max_tokens = 1 };
    var job: api.LaneHost.Job = .{ .host = &host, .id = 1, .request = &request, .sink = undefined };
    try host.admitted.append(t.allocator, &job);
    try host.queued.append(t.allocator, &job);
    const f = try prefetch.make(x.r, t.io, .{ .reads = 1 });
    defer f.stop();
    fetch.queue(&host);
    try t.expect(fetch.held(&host, &job));
    try t.expectEqual(@as(u64, 0), f.counts.made);
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".st"));
    gate.release.store(true, .release);
    const end = rows.write.nowNs() + 3 * std.time.ns_per_s;
    while (job.fetch.name == null or fetch.held(&host, &job)) {
        if (rows.write.nowNs() >= end) return error.PrefetchTimedOut;
        fetch.queue(&host);
        rows.write.io.nap();
    }
    const saved = try rows.read.read(x.r, x.r.find(job.fetch.name.?).?);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    const restored: *@import("rows_fake.zig").Kept = @ptrCast(@alignCast(saved));
    try t.expectEqualSlices(u8, expected, restored.img);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
    try t.expectEqual(@as(u64, 1), f.counts.used);
    try t.expectEqual(@as(u64, 2), x.r.counts.writes);
}

test "prefetch: closing the store does not join a blocked read past its deadline" {
    const x = try fixture.Fixture.open("fetch-stop", .evicted, false, 1 << 20);
    defer x.close();
    const tokens = fixture.prompt(3);
    try x.keep(&tokens, 900);
    x.s.remove(0);
    try x.settle();
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    const f = try prefetch.make(x.r, t.io, .{ .reads = 1 });
    const Hold = struct {
        var entered: std.atomic.Value(bool) = .init(false);
        var release: std.atomic.Value(bool) = .init(false);
        fn read(_: *prefetch.Prefetch, _: *prefetch.Read) !void {
            entered.store(true, .release);
            while (!release.load(.acquire)) rows.write.io.nap();
        }
        fn later() void {
            std.Io.sleep(t.io, .fromMilliseconds(250), .awake) catch {};
            release.store(true, .release);
        }
    };
    Hold.entered.store(false, .release);
    Hold.release.store(false, .release);
    defer Hold.release.store(true, .release);
    f.reader = Hold.read;
    _ = f.start(x.r, 0).?;
    try fixture.Gate.wait(&Hold.entered);
    const thread = try std.Thread.spawn(.{}, Hold.later, .{});
    defer thread.join();
    const start = rows.write.nowNs();
    rows.write.stop(x.r, start + 20 * std.time.ns_per_ms);
    try t.expect(rows.write.nowNs() - start < 150 * std.time.ns_per_ms);
    f.mutex.lockUncancelable(f.io);
    const cancelled = f.counts.cancelled;
    f.mutex.unlock(f.io);
    try t.expectEqual(@as(u64, 1), cancelled);
    try t.expectEqual(@as(u32, 2), f.refs.load(.acquire));
    x.r.fetch = null;
    f.destroy(); // the reader is still blocked after the store releases its last reference
}

test "prefetch: a failed or late staged write releases its request without reading private files" {
    for ([_]bool{ false, true }) |late| {
        const x = try fixture.Fixture.open("fetch-fallback", .evicted, false, 1 << 20);
        defer x.close();
        var gate: fixture.Gate = .{ .mode = .full };
        defer gate.release.store(true, .release);
        x.r.file_io = gate.io();
        const tokens = fixture.prompt(4);
        try x.keep(&tokens, 900);
        x.s.remove(0);
        try fixture.Gate.wait(&gate.entered);
        var core: lanes.Engine = undefined;
        var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 2 });
        host.cache = &x.s;
        defer host.queued.deinit(t.allocator);
        defer host.admitted.deinit(t.allocator);
        const request: api.Request = .{ .prompt = &tokens, .max_tokens = 1 };
        var job: api.LaneHost.Job = .{ .host = &host, .id = 1, .request = &request, .sink = undefined };
        try host.admitted.append(t.allocator, &job);
        try host.queued.append(t.allocator, &job);
        const f = try prefetch.make(x.r, t.io, .{ .reads = 1, .wait_ns = if (late) 0 else std.time.ns_per_s });
        defer f.stop();
        fetch.queue(&host);
        try t.expectEqual(!late, fetch.held(&host, &job));
        if (late) {
            fetch.queue(&host);
        } else {
            gate.release.store(true, .release);
            try x.settle();
            fetch.queue(&host);
        }
        try t.expect(job.fetch.looked and !fetch.held(&host, &job));
        try t.expectEqual(@as(u64, 0), f.counts.made);
        try t.expectEqual(@as(usize, 0), x.r.states.items.len);
        gate.release.store(true, .release);
        try x.settle();
    }
}
