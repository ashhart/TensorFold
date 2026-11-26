//! A producer's streamed export beside its feeder: fed, drained at the cut, failed, released once, never while fed.
const std = @import("std");
const fabric = @import("fabric");
const cx = @import("cuda_export.zig");
const serve = @import("cuda_export_serve.zig");
const t = @import("cuda_export_serve_test.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const upto: u64 = 300;

/// A feeder that counts what the producer asks of it, and what the export held when it was asked.
const Spy = struct {
    p: *serve.Produce,
    id: [16]u8,
    built: *cx.Built,
    answer: bool = false, // what `drain` reports: every paged part converted
    discard_in_progress: bool = false,
    fed: u32 = 0,
    last: u64 = 0,
    drained: u32 = 0,
    whole_at_drain: bool = true,
    stopped: u32 = 0,
    stopped_in_progress: u32 = 0,

    fn feeder(s: *Spy) serve.Feeder {
        return .{ .ptr = s, .progress = progress, .drain = drain, .stop = stop };
    }

    fn progress(ptr: *anyopaque, pos: u64) void {
        const s: *Spy = @ptrCast(@alignCast(ptr));
        s.fed += 1;
        s.last = pos;
        if (s.discard_in_progress) {
            _ = s.p.discard(s.id); // the decoder closes the export while the engine feeds it
            s.stopped_in_progress = s.stopped;
        }
    }

    fn drain(ptr: *anyopaque) bool {
        const s: *Spy = @ptrCast(@alignCast(ptr));
        s.drained += 1;
        s.whole_at_drain = s.built.exp.progress.?.whole.load(.acquire);
        return s.answer;
    }

    fn stop(ptr: *anyopaque) void {
        const s: *Spy = @ptrCast(@alignCast(ptr));
        s.stopped += 1;
    }
};

fn producer(fake: *t.Fake) serve.Produce {
    return .{ .gpa = gpa, .io = io, .source = fake.source(), .opts = .{ .link = "tf-test", .model = "dsv41-test" }, .table = .{ .gpa = gpa, .ttl_ns = std.time.ns_per_hour, .clock = fabric.words.nowNs } };
}

fn cleanup(p: *serve.Produce) void {
    var it = p.held.valueIterator();
    while (it.next()) |h| {
        h.*.built.deinit(gpa);
        gpa.destroy(h.*);
    }
    p.held.deinit(gpa);
    p.table.deinit();
}

fn begin(p: *serve.Produce, spy: *Spy, id: [16]u8, tokens: []const u32) !void {
    const built = try cx.skeleton(gpa, id, tokens, cx.shapeOf(upto, 128));
    spy.* = .{ .p = p, .id = id, .built = built };
    p.beginStream(id, built, spy.feeder()) catch |err| {
        built.deinit(gpa);
        return err;
    };
}

test "a streamed export is fed, drained before its cut converts, completed whole, and its feeder stopped once at release" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try t.Fake.init(upto);
    defer fake.deinit();
    var p = producer(fake);
    defer cleanup(&p);
    const tokens = try t.tokensOf(a, upto);
    const id: [16]u8 = @splat(0x41);
    var spy: Spy = undefined;
    try begin(&p, &spy, id, tokens);
    const entry = p.table.get(id).?.ready;
    try std.testing.expect(!entry.progress.?.whole.load(.acquire));
    try std.testing.expectError(error.HandoffHeld, p.beginStream(id, spy.built, spy.feeder()));

    p.feedStream(id, 100);
    p.feedStream(@splat(0x42), 100); // an id nothing streams: nobody is fed
    try std.testing.expectEqual(@as(u32, 1), spy.fed);
    try std.testing.expectEqual(@as(u64, 100), spy.last);
    try std.testing.expectEqual(@as(?bool, false), p.drainStream(id));
    try std.testing.expectEqual(@as(?bool, null), p.drainStream(@splat(0x42)));
    spy.answer = true;
    try std.testing.expectEqual(@as(?bool, true), p.drainStream(id));
    spy.drained = 0;

    // the cut: the feeder is quiet before the rest converts, and the export that results is the whole one's
    p.exportedBg(id, tokens, try p.save(upto));
    try std.testing.expectEqual(@as(u32, 1), spy.drained);
    try std.testing.expect(!spy.whole_at_drain);
    try std.testing.expect(entry.progress.?.whole.load(.acquire));
    const saved = try p.save(upto);
    const want = try cx.build(gpa, id, tokens, saved.cap.state);
    defer want.deinit(gpa);
    p.drop(saved);
    for (want.exp.sources, entry.sources) |x, y| for (x.parts, y.parts) |u, v| try std.testing.expectEqualSlices(u8, u, v);
    try std.testing.expectEqual(@as(u32, 0), spy.stopped);
    try std.testing.expect(p.discard(id));
    try std.testing.expectEqual(@as(u32, 1), spy.stopped);
    try std.testing.expectEqual(@as(usize, 0), p.held.count());
}

test "a streamed export whose dump fails at any step is served all the same, and leaves no file behind" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try t.Fake.init(upto);
    defer fake.deinit();
    var p = producer(fake);
    defer cleanup(&p);
    p.opts.dump = try tmp.dir.realPathFileAlloc(io, ".", a);
    const tokens = try t.tokensOf(a, upto);
    for ([_]serve.Produce.DumpStep{ .create, .write, .flush, .rename }, 0..) |step, i| {
        const id: [16]u8 = @splat(0x60 + @as(u8, @intCast(i)));
        var spy: Spy = undefined;
        try begin(&p, &spy, id, tokens);
        p.dump_fault = step;
        p.exportedBg(id, tokens, try p.save(upto));
        p.dump_fault = null;
        try std.testing.expect(p.table.get(id).?.ready.progress.?.whole.load(.acquire));
        var it = tmp.dir.iterate();
        if (try it.next(io)) |left| {
            std.debug.print("left after a failed {t} step: {s}\n", .{ step, left.name });
            return error.DumpLeftAFile;
        }
        try std.testing.expect(p.discard(id));
    }
    try std.testing.expectEqual(fake.taken, fake.released);
}

test "a streamed export that failed tells its decoder why, and its feeder stops at release" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try t.Fake.init(upto);
    defer fake.deinit();
    var p = producer(fake);
    defer cleanup(&p);
    const tokens = try t.tokensOf(a, upto);
    const id: [16]u8 = @splat(0x43);
    var spy: Spy = undefined;
    try begin(&p, &spy, id, tokens);
    p.failed(id, "DeviceLost");
    const progress = p.table.get(id).?.ready.progress.?;
    try std.testing.expect(progress.broken.load(.acquire));
    try std.testing.expectEqualStrings("DeviceLost", progress.reason());
    try std.testing.expectEqual(@as(u32, 0), spy.stopped); // held until the decoder closes it or it expires
    try std.testing.expect(!progress.whole.load(.acquire));
    try std.testing.expect(p.discard(id));
    try std.testing.expectEqual(@as(u32, 1), spy.stopped);
}

test "an export released while the engine feeds it is freed when the feed returns, not before" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try t.Fake.init(upto);
    defer fake.deinit();
    var p = producer(fake);
    defer cleanup(&p);
    const tokens = try t.tokensOf(a, upto);
    const id: [16]u8 = @splat(0x44);
    var spy: Spy = undefined;
    try begin(&p, &spy, id, tokens);
    spy.discard_in_progress = true;
    p.feedStream(id, 50);
    try std.testing.expectEqual(@as(u32, 0), spy.stopped_in_progress);
    try std.testing.expectEqual(@as(u32, 1), spy.stopped);
    try std.testing.expectEqual(@as(usize, 0), p.held.count());
}

/// A streamed export begun while the engine drains its id again and again, from the moment both queue on the lock.
const Race = struct {
    p: *serve.Produce,
    spy: Spy = undefined,
    id: [16]u8,
    tokens: []const u32,
    refused: bool = false,
    done: std.atomic.Value(bool) = .init(false),
    /// the engine leased the export and asked its feeder: it reached a refused export
    reached: bool = false,

    fn start(r: *Race) void {
        begin(r.p, &r.spy, r.id, r.tokens) catch {
            r.refused = true;
        };
        r.done.store(true, .release);
    }

    fn drain(r: *Race) void {
        while (!r.reached) {
            const last = r.done.load(.acquire);
            if (r.p.drainStream(r.id) != null) r.reached = true;
            if (last) return;
        }
    }
};

test "a streamed export refused at the cap is never leased by the engine racing its admission, so none frees under a pin" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try t.Fake.init(upto);
    defer fake.deinit();
    var p = producer(fake);
    defer cleanup(&p);
    const tokens = try t.tokensOf(a, upto);
    var first: Spy = undefined;
    try begin(&p, &first, @splat(0x50), tokens);
    p.table.cap_bytes = p.table.held_bytes; // full: every later export of this size is refused
    for (0..40) |i| {
        var race: Race = .{ .p = &p, .id = @splat(0x51), .tokens = tokens };
        race.id[0] = @intCast(i);
        while (!p.lock.tryLock()) std.atomic.spinLoopHint(); // both threads queue on it, then race when it opens
        const starter = try std.Thread.spawn(.{}, Race.start, .{&race});
        const engine = try std.Thread.spawn(.{}, Race.drain, .{&race});
        try std.Io.sleep(io, .fromMilliseconds(30), .awake);
        p.lock.unlock();
        starter.join();
        engine.join();
        try std.testing.expect(race.refused);
        if (race.reached) return error.EngineReachedARefusedExport;
    }
    try std.testing.expectEqual(@as(usize, 1), p.held.count());
}
