//! The retained-history integration case shares the existing stepped target and drafter fixtures.
const std = @import("std");
pub fn run(comptime sm: type, comptime fake: type, comptime FakeDrafter: type, comptime Drafted: type, gpa: std.mem.Allocator, comptime long: anytype) !void {
    const be = @import("backend.zig");
    const dr = @import("drafter.zig");
    const Probe = struct {
        var from: u64 = std.math.maxInt(u64);
        var inner_step: *const fn (*anyopaque, *sm.Stream, u32) anyerror!bool = undefined;
        fn step(ptr: *anyopaque, stream: *sm.Stream, limit: u32) !bool {
            const done = try inner_step(ptr, stream, limit);
            stream.cached = stream.reuse.at;
            return done;
        }
        var inner: *const fn (*anyopaque, *sm.Stream, []const u32, u64, u32) anyerror!be.Features = undefined;
        fn features(ptr: *anyopaque, stream: *sm.Stream, taps_: []const u32, start: u64, count: u32) !be.Features {
            from = start;
            return inner(ptr, stream, taps_, start, count);
        }
    };
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var early = try sm.Stream.init(gpa, .{ .id = "cache-primer", .prompt = long[0..8], .max_new = 1 });
    defer early.deinit(gpa);
    try target.backend().prefill(&early);
    const saved = try target.save(&early, 8);
    defer target.drop(saved);
    target.backend().release(&early);
    var backend = target.stepped();
    var table = backend.vtable.*;
    Probe.inner = table.features.?;
    Probe.inner_step = table.prefill_step.?;
    table.prefill_step = Probe.step;
    table.features = Probe.features;
    backend.vtable = &table;
    if (@hasField(be.Backend, "restored_features")) backend.restored_features = true;
    var facts: dr.Facts = .{ .depth = 6, .step_ms = 0.5 };
    if (@hasField(dr.Facts, "history_rows")) facts.history_rows = 4;
    var head: FakeDrafter = .{ .facts_ = facts };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, backend, head.drafter());
    defer wrapped.deinit();
    var stream = try sm.Stream.init(gpa, .{ .id = "cached", .prompt = &long, .max_new = 8, .reuse = .{ .saved = saved, .at = 8 } });
    defer stream.deinit(gpa);
    const b = wrapped.backend();
    defer b.release(&stream);
    _ = try b.vtable.prefill_step.?(b.ptr, &stream, 0);
    try std.testing.expectEqual(@as(u32, 8), stream.cached);
    try std.testing.expectEqual(@as(u64, 4), Probe.from);
}
