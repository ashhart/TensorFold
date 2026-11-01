//! The load ledger's handles: attach and detach epochs, counters kept across hosts, and gauges gone at detach.
const std = @import("std");
const api = @import("engine_api.zig");
const ledger = @import("load_ledger.zig");

/// An engine that keeps the producer handle the ledger gives it.
const Fake = struct {
    producer: ?ledger.Producer = null,
    calls: u32 = 0,

    fn info(_: *anyopaque) api.Info {
        return .{ .lanes = 4 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, _: *api.Status, _: []u32) void {}
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn load(ctx: *anyopaque, p: ?ledger.Producer) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.producer = p;
        f.calls += 1;
    }
    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .load = load };

    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &vtable };
    }
};

test "a producer's writes reach the ledger until it detaches; a second detach calls no host" {
    var l: ledger.Ledger = .init(std.testing.io);
    var f: Fake = .{};
    l.attach(f.engine());
    try std.testing.expectEqual(@as(u32, 4), l.read().lanes);
    const p = f.producer.?;
    p.publish(.{ .running = 2, .started = 5 }, .{ .decode = 7 }, .{ .mode = .window, .lanes_free = 2 });
    const live = l.read();
    try std.testing.expectEqual(@as(u32, 2), live.requests.?.running);
    try std.testing.expectEqual(@as(u64, 7), live.work.?.decode);
    try std.testing.expectEqual(ledger.Mode.window, live.capacity.?.mode);
    l.detach(true);
    try std.testing.expect(f.producer == null);
    try std.testing.expectEqual(@as(u32, 2), f.calls);
    p.publish(.{ .running = 9 }, null, null); // a stale handle writes nothing
    const parked = l.read();
    try std.testing.expect(parked.parked and parked.requests == null and parked.work == null and parked.capacity == null);
    l.detach(false);
    try std.testing.expectEqual(@as(u32, 2), f.calls);
}

test "counters continue across a detach and the next host's attach; the instance stays" {
    var l: ledger.Ledger = .init(std.testing.io);
    const id = l.instance;
    try std.testing.expectEqual(@as(usize, 32), std.mem.indexOfNone(u8, &id, "0123456789abcdef") orelse id.len);
    var first: Fake = .{};
    l.attach(first.engine());
    first.producer.?.publish(.{ .running = 3, .started = 10, .requeued = 1, .yields = 2 }, .{ .prefill_compute = 100, .prefill_busy_ns = 50 }, null);
    l.detach(true);
    var second: Fake = .{};
    l.attach(second.engine());
    const empty = l.read();
    try std.testing.expect(!empty.parked and empty.requests == null);
    second.producer.?.publish(.{ .running = 1, .started = 4 }, .{ .prefill_compute = 20, .prefill_busy_ns = 5 }, null);
    const s = l.read();
    try std.testing.expectEqual(@as(u32, 1), s.requests.?.running); // a gauge is the live host's alone
    try std.testing.expectEqual(@as(u64, 14), s.requests.?.started);
    try std.testing.expectEqual(@as(u64, 1), s.requests.?.requeued);
    try std.testing.expectEqual(@as(u64, 2), s.requests.?.yields);
    try std.testing.expectEqual(@as(u64, 120), s.work.?.prefill_compute);
    try std.testing.expectEqual(@as(u64, 55), s.work.?.prefill_busy_ns);
    try std.testing.expectEqualSlices(u8, &id, &s.instance);
}

test "attaching a second engine stops the first one's handle" {
    var l: ledger.Ledger = .init(std.testing.io);
    var first: Fake = .{};
    var second: Fake = .{};
    l.attach(first.engine());
    const stale = first.producer.?;
    stale.publish(.{ .started = 3 }, null, null);
    l.attach(second.engine());
    try std.testing.expect(first.producer == null);
    stale.publish(.{ .running = 8, .started = 99 }, null, null);
    second.producer.?.publish(.{ .running = 1 }, null, null);
    const s = l.read();
    try std.testing.expectEqual(@as(u32, 1), s.requests.?.running);
    try std.testing.expectEqual(@as(u64, 3), s.requests.?.started);
}

test "a paged capacity reads absent until its pool is published" {
    var l: ledger.Ledger = .init(std.testing.io);
    var f: Fake = .{};
    l.attach(f.engine());
    const p = f.producer.?;
    p.publish(.{}, null, .{ .mode = .paged, .lanes_free = 4 });
    try std.testing.expect(l.read().capacity == null);
    p.publishPool(.{ .capacity = 1024, .available = 512, .reclaimable = 256, .promised = 64 });
    const s = l.read();
    try std.testing.expectEqual(ledger.Mode.paged, s.capacity.?.mode);
    try std.testing.expectEqual(@as(u64, 512), s.pool.?.available);
}

test "route groups count requests in progress" {
    var l: ledger.Ledger = .init(std.testing.io);
    l.enter(.chat_completions);
    l.enter(.chat_completions);
    l.enter(.responses);
    l.leave(.chat_completions);
    const s = l.read();
    try std.testing.expectEqual(@as(u32, 1), s.active(.chat_completions));
    try std.testing.expectEqual(@as(u32, 1), s.active(.responses));
    try std.testing.expectEqual(@as(u32, 0), s.active(.other));
}
