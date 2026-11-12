//! Keep-side reservations live only through the prompt pass and fallback checkpoints use unclaimed room.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const queue = @import("lane_cache_queue.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;

const Mode = enum { reduce, last, prefilled, protect, fallback, fallback_room, fallback_spare, oom };

fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}

fn exercise(mode: Mode) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var family: Fake = .{ .gpa = a };
    var snapshots = family.snapshots();
    var vtable = snapshots.vtable.*;
    const Spare = struct {
        fn held(_: *anyopaque) u64 {
            return 200;
        }
    };
    if (mode == .fallback_spare) {
        vtable.spare = Spare.held;
        snapshots.vtable = &vtable;
    }
    var store = pc.Store.init(a, snapshots, .{ .min_prompt = 0, .min_gap = 0 }, switch (mode) {
        .protect => 220,
        .fallback, .fallback_spare => 320,
        .prefilled => 300,
        else => 600,
    });
    defer store.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    defer host.admitted.deinit(a);
    defer host.queued.deinit(a);
    const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    const other = [_]u32{ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 };
    const idle = [_]u32{ 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 };
    const request: api.Request = .{ .prompt = &words, .history_len = 12, .max_tokens = 1 };
    const next: api.Request = .{ .prompt = &other, .history_len = 16, .max_tokens = 1 };
    var job: api.LaneHost.Job = .{ .host = &host, .id = 1, .request = &request, .sink = .{ .ctx = &family, .event = event }, .marks = &.{ 4, 8, 12 }, .cache_promise = .{ .bytes = 324, .slots = 3 } };
    var later: api.LaneHost.Job = .{ .host = &host, .id = 2, .request = &next, .sink = .{ .ctx = &family, .event = event } };
    job.stream = try lanes.Stream.init(a, .{ .id = "promise", .prompt = &words, .max_new = 1 });
    defer job.stream.deinit(a);
    try host.admitted.append(a, &job);
    if (mode == .protect or mode == .oom or mode == .fallback or mode == .fallback_spare) {
        _ = family.pass(&store, other[0..8], .{});
        try std.testing.expect(store.keepBorn(&other, 8, 8, null, &.{}, &.{}));
    }
    switch (mode) {
        .reduce => {
            _ = family.pass(&store, words[0..4], .{});
            queue.keep(&host, &job, &job.stream, 4);
            try std.testing.expectEqual(@as(u64, 220), job.cache_promise.bytes);
            try std.testing.expectEqual(@as(usize, 2), job.cache_promise.slots);
        },
        .last => {
            _ = family.pass(&store, words[0..12], .{});
            queue.keep(&host, &job, &job.stream, 12);
            try std.testing.expectEqual(@as(u64, 0), job.cache_promise.bytes);
            try std.testing.expectEqual(@as(usize, 0), job.cache_promise.slots);
        },
        .prefilled => {
            _ = family.pass(&store, words[0..4], .{});
            queue.keep(&host, &job, &job.stream, 4);
            // The backend skipped both later marks; the job still occupies a decode slot at the next admission.
            host.prefilled(&job, 0);
            const found = try queue.lookup(&host, &later);
            defer a.free(found.marks);
            try std.testing.expectEqualSlices(u32, &.{16}, found.marks);
            if (found.entry) |e| store.release(e);
        },
        .protect => {
            _ = family.pass(&store, idle[0..8], .{});
            try std.testing.expect(store.keepBorn(&idle, 8, 8, null, &.{}, &.{}));
            try host.queued.append(a, &later);
            _ = family.pass(&store, words[0..12], .{});
            queue.keep(&host, &job, &job.stream, 12);
            try std.testing.expect(store.find(&other, &.{}, &.{}) != null);
            try std.testing.expect(store.find(&idle, &.{}, &.{}) == null);
            try std.testing.expect(store.find(&words, &.{}, &.{}) != null);
        },
        .fallback, .fallback_room, .fallback_spare => {
            job.marks = &.{12};
            job.cache_promise = .{ .bytes = 112, .slots = 1 };
            later.cache_promise = .{ .bytes = 100, .slots = 1 };
            if (mode == .fallback_spare) {
                job.cache_promise = .{};
                later.cache_promise = .{};
            }
            const promised = job.cache_promise;
            if (mode != .fallback_spare) try host.admitted.append(a, &later);
            _ = family.pass(&store, words[0..6], .{});
            const held = store.held;
            queue.keep(&host, &job, &job.stream, 6);
            try std.testing.expectEqual(promised.bytes, job.cache_promise.bytes);
            try std.testing.expectEqual(promised.slots, job.cache_promise.slots);
            try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
            try std.testing.expectEqual(held + @as(u64, if (mode == .fallback_room) 106 else 0), store.held);
        },
        .oom => {
            try host.queued.append(a, &later);
            var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
            host.gpa = failing.allocator();
            queue.keep(&host, &job, &job.stream, 12);
            try std.testing.expectEqual(@as(u64, 0), job.cache_promise.bytes);
            try std.testing.expectEqual(@as(usize, 0), job.cache_promise.slots);
            try std.testing.expectEqual(@as(u64, 1), store.counts.failed);
            try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
        },
    }
}

test "queued cache: each kept mark reduces its job's promised bytes and slots" {
    try exercise(.reduce);
}

test "queued cache: the last mark releases every remaining promised byte and slot" {
    try exercise(.last);
}

test "queued cache: a later admission ignores a finished prefill's skipped marks" {
    try exercise(.prefilled);
}

test "queued cache: a keep spares a queued conversation while evicting an idle one" {
    try exercise(.protect);
}

test "queued cache: fallback checkpoints neither evict nor spend either job's promise" {
    try exercise(.fallback);
    try exercise(.fallback_room);
    try exercise(.fallback_spare);
}

test "queued cache: failed queue snapshot releases the abandoned keep promise" {
    try exercise(.oom);
}
