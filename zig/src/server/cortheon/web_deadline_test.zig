//! Real web Client.get with a fake DNS backend; neither test can reach a socket or an external resolver.
const std = @import("std");
const Io = std.Io;
const net = Io.net;
const web = @import("web.zig");
const transport = @import("../media_fetch_net.zig");
const t = std.testing;
const Fake = struct {
    var table: Io.VTable = undefined;
    var addresses: []const net.IpAddress = &.{};
    var delay_ms: i64 = 0;
    var abandon: std.atomic.Value(bool) = .init(false);
    var active: std.atomic.Value(bool) = .init(false);
    var calls: std.atomic.Value(u32) = .init(0);
    fn lookup(_: ?*anyopaque, _: net.HostName, q: *Io.Queue(net.HostName.LookupResult), _: net.HostName.LookupOptions) net.HostName.LookupError!void {
        const io = t.io;
        active.store(true, .release);
        defer active.store(false, .release);
        defer q.close(io);
        _ = calls.fetchAdd(1, .monotonic);
        if (delay_ms > 0) {
            const end = Io.Clock.awake.now(io).addDuration(.fromMilliseconds(delay_ms));
            while (Io.Clock.awake.now(io).nanoseconds < end.nanoseconds) io.sleep(.fromMilliseconds(5), .awake) catch {};
            return error.NameServerFailure;
        }
        for (addresses) |addr| while ((q.put(io, &.{.{ .address = addr }}, 0) catch return error.NameServerFailure) == 0) {
            if (abandon.load(.acquire)) return error.NameServerFailure;
            io.sleep(.fromMilliseconds(1), .awake) catch {};
        };
    }
    fn system() Io {
        table = t.io.vtable.*;
        table.netLookup = lookup;
        table.async = Io.noAsync;
        abandon.store(false, .release);
        calls.store(0, .monotonic);
        return .{ .userdata = t.io.userdata, .vtable = &table };
    }
    fn drain() void {
        while (active.load(.acquire) or transport.lookups.load(.acquire) != 0) t.io.sleep(.fromMilliseconds(5), .awake) catch {};
    }
    fn stop() void {
        t.io.sleep(.fromMilliseconds(250), .awake) catch {};
        abandon.store(true, .release);
    }
};
test "request access audit: web deadline returns before a DNS backend that ignores cancellation" {
    Fake.addresses = &.{};
    Fake.delay_ms = 1000;
    defer {
        Fake.drain();
        Fake.delay_ms = 0;
    }
    const io = Fake.system();
    var client: web.Client = .init(t.allocator, io, "fixture");
    defer client.deinit();
    const start = Io.Clock.awake.now(io);
    try t.expectError(error.Timeout, client.get(t.allocator, "http://slow.audit.invalid/", &.{}, .{ .timeout_ms = 100 }));
    const elapsed_ms = start.durationTo(Io.Clock.awake.now(io)).toMilliseconds();
    try t.expectEqual(@as(u32, 1), Fake.calls.load(.monotonic));
    try t.expect(elapsed_ms < 750);
}
test "request access audit: all DNS answers including a private address past the queue are checked" {
    var addresses: [40]net.IpAddress = undefined;
    for (&addresses, 0..) |*addr, i| addr.* = .{ .ip4 = .{ .bytes = .{ 8, 8, 4, @intCast(i + 1) }, .port = 80 } };
    addresses[39] = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 80 } };
    Fake.addresses = &addresses;
    Fake.delay_ms = 0;
    defer Fake.drain();
    const io = Fake.system();
    const stopper = try std.Thread.spawn(.{}, Fake.stop, .{});
    defer stopper.join();
    var client: web.Client = .init(t.allocator, io, "fixture");
    defer client.deinit();
    try t.expectError(error.NotPublic, client.get(t.allocator, "http://many.audit.invalid/", &.{}, .{ .timeout_ms = 5000 }));
    try t.expectEqual(@as(u32, 1), Fake.calls.load(.monotonic));
}
