//! Background reply passes on planned, one-token-lookahead caches; ordinary request cuts stay unchanged.
const std = @import("std");
const api = @import("engine_api");

/// A reply's immediate follow-up wins before a background pass starts its first chunk.
pub const quiet_ns = 150 * std.time.ns_per_ms;

pub fn enable(host: *api.LaneHost) void {
    const store = host.cache orelse return;
    std.debug.assert(store.rules.planned and store.rules.lookahead == 1);
    store.rules.warm = .waits;
    store.rules.warm_usable = true;
    host.warm_quiet = quiet_ns;
    host.info_.warm_turns = true;
}

test "planned warm: no cache advertises no pass; a cache keeps its lookahead and normal cuts" {
    const lanes = @import("lanes");
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    enable(&host);
    try std.testing.expect(!host.info_.warm_turns);
    try std.testing.expectEqual(@as(u64, 0), host.warm_quiet);
    const S = struct {
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 8;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!*anyopaque {
            unreachable;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
            unreachable;
        }
        fn drop(_: *anyopaque, _: *anyopaque) void {
            unreachable;
        }
    };
    var dummy: u8 = 0;
    var store = api.prompt_cache.Store.init(a, .{ .ptr = &dummy, .vtable = &.{ .bytes = S.bytes, .save = S.save, .restore = S.restore, .drop = S.drop } }, .{ .lookahead = 1, .planned = true }, 1024);
    defer store.deinit();
    const prompt: [9]u32 = @splat(7);
    const starts = [_]u32{ 4, 8 };
    const cold = try store.marks(a, &prompt, 0, 4, &.{}, &starts, &.{});
    defer a.free(cold);
    host.cache = &store;
    enable(&host);
    try std.testing.expect(host.info_.warm_turns);
    try std.testing.expectEqual(@as(u64, 150_000_000), host.warm_quiet);
    try std.testing.expect(store.rules.planned and store.rules.warm_usable);
    try std.testing.expectEqual(@as(u32, 1), store.rules.lookahead);
    try std.testing.expectEqual(.waits, store.rules.warm);
    const after = try store.marks(a, &prompt, 0, 4, &.{}, &starts, &.{});
    defer a.free(after);
    try std.testing.expectEqualSlices(u32, cold, after);
    const warm = try store.marks(a, &prompt, 4, prompt.len, &.{}, &starts, &.{});
    defer a.free(warm);
    try std.testing.expectEqualSlices(u32, &.{8}, warm);
    try std.testing.expect(store.usable(8, &starts));
    try std.testing.expect(!store.usable(9, &starts));
}
