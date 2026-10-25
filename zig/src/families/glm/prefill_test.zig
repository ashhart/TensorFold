//! Real lane-host scheduling and GLM's production chunk driver, with a fake engine at GPU boundaries.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("../../core/engine_api.zig");
const Backend = @import("backend.zig").Backend;
const mac = @import("builtin").os.tag == .macos;
const fill = @import("prefill.zig");
const a = std.testing.allocator;
const Stream = lanes.Stream;

test "glm prefill: exhausted pool releases the waiting owner and preserves permanent refusals" {
    if (comptime !mac) return error.SkipZigTest;
    const pool = @import("pool.zig");
    const slots = @import("slots.zig");
    var engine: @import("engine.zig").Engine = undefined;
    engine.ep = null;
    engine.c = .{};
    engine.c.layers = 0;
    engine.c.mtp = 0;
    var storage: [2]slots.Slot = undefined;
    for (&storage) |*s| s.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl: slots.Slots = .{
        .gpa = a,
        .e = &engine,
        .slots = &storage,
        .rows = undefined,
        .ids = undefined,
        .lasts = undefined,
        .picks = undefined,
        .ring = undefined,
        .pool = .{ .base = undefined, .blocks = 2, .kpool = 4 },
    };
    defer sl.snaps.deinit(a);
    var backend: Backend = .{ .gpa = a, .sl = &sl };
    defer backend.deinit();
    const prompt: [100]u32 = @splat(7);
    var waiting = try Stream.init(a, .{ .id = "waiting", .prompt = &prompt, .max_new = 200 });
    defer waiting.deinit(a);
    var oversized = try Stream.init(a, .{ .id = "oversized", .prompt = &prompt, .max_new = 400 });
    defer oversized.deinit(a);

    try std.testing.expect(Backend.fits(&backend, 100, 200));
    try pool.reserve(&sl, 0, 317, null);
    storage[0].used = true;
    const occupied = storage[0].region.?;
    try std.testing.expect(!Backend.fits(&backend, 100, 200));
    try std.testing.expectError(error.LaneMemory, fill.step(&backend, &waiting, 0));
    try std.testing.expectEqual(@as(u32, 0), backend.by.count());
    try std.testing.expectEqual(@as(u32, 0), backend.filling.count());
    try std.testing.expect(storage[1].region == null and !storage[1].used);
    try std.testing.expectEqual(occupied, storage[0].region.?);

    try std.testing.expectError(error.ContextFull, fill.step(&backend, &oversized, 0));
    sl.release(0);
    try std.testing.expect(Backend.fits(&backend, 100, 200));
    try std.testing.expectError(error.ContextFull, fill.step(&backend, &oversized, 0));
    try pool.reserve(&sl, 1, 317, null);
    sl.release(1);
    for (storage) |s| try std.testing.expect(s.region == null and !s.used);
}

const harness = @import("prefill_harness.zig");
const Fake = harness.Fake;
const Box = harness.Box;
fn run(drafts: bool, concurrent: bool) !Box {
    var fake: Fake = .{ .barrier = concurrent };
    defer fake.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 4, .hidden_rows = true, .streams_exact = true, .batch_rows = 8, .max_streams = 2, .draft_streams = true }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    var box: Box = .{};
    errdefer box.deinit();
    const long = try a.alloc(u32, 102400);
    defer a.free(long);
    for (long, 0..) |*t, i| t.* = @intCast(i % 97);
    const first: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 4096, .drafts = drafts, .sampling = .{ .seed = 1235, .temperature = 1 } };
    const second: api.Request = .{ .prompt = long, .max_tokens = 32, .drafts = drafts, .sampling = .{ .seed = 1236, .temperature = 1 } };
    const e = host.engine();
    try e.submit(1, &first, .{ .ctx = &box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer {
        fake.go.store(true, .release);
        if (!stopped) host.stop();
    }
    if (concurrent) {
        while (!fake.ready.load(.acquire)) std.Thread.yield() catch {};
        try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
        fake.go.store(true, .release);
    } else {
        while (box.done.load(.acquire) < 1) std.Thread.yield() catch {};
        try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
    }
    while (box.done.load(.acquire) < 2) std.Thread.yield() catch {};
    host.stop();
    stopped = true;
    try std.testing.expect(!box.bad);
    try std.testing.expectEqual(@as(usize, 4096), box.tokens[0].items.len);
    try std.testing.expectEqual(@as(usize, 32), box.tokens[1].items.len);
    try std.testing.expectEqual(@as(usize, 2), fake.begins);
    try std.testing.expectEqual(@as(usize, 2), fake.ends);
    if (concurrent) {
        try std.testing.expect(fake.chunks >= 102400 / fill.decode_rows);
        try std.testing.expect(fake.max_rows <= fill.decode_rows);
        const rounds = fake.last_round - fake.first_round;
        try std.testing.expect(rounds >= fake.chunks - 1);
        // A simulated 100 ms per chunk still permits at least nine decode rounds per second.
        try std.testing.expect(rounds * 1000 >= fake.chunks * 100 * 9);
    }
    return box;
}

test "glm prefill: resumed planned cuts and marks survive yields and independent filling streams" {
    var f: Fake = .{ .restored_at = 1536 };
    defer f.deinit();
    const tokens = try a.alloc(u32, 6000);
    defer a.free(tokens);
    @memset(tokens, 7);
    var first = try Stream.init(a, .{ .id = "a", .prompt = tokens, .max_new = 4, .chunks = &.{ 512, 2048, 4096 } });
    defer first.deinit(a);
    var second = try Stream.init(a, .{ .id = "b", .prompt = tokens, .max_new = 4, .chunks = &.{3000} });
    defer second.deinit(a);
    const Marks = struct {
        at: [2]u32 = undefined,
        n: usize = 0,
        fn hit(p: *anyopaque, _: *Stream, at: u32) void {
            const m: *@This() = @ptrCast(@alignCast(p));
            m.at[m.n] = at;
            m.n += 1;
        }
    };
    var marks: Marks = .{};
    first.reuse = .{ .marks = &.{ 2048, 4096 }, .hook = .{ .ptr = &marks, .at = Marks.hit } };
    try std.testing.expect(!try fill.step(&f, &first, 0));
    try std.testing.expectEqual(@as(u32, 2048), first.fill_at.load(.monotonic));
    f.restored_at = 0;
    try std.testing.expect(!try fill.step(&f, &second, 0));
    try std.testing.expectEqual(@as(u32, 3000), second.fill_at.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 2), f.filling.count());
    try std.testing.expect(!try fill.step(&f, &first, 0));
    try std.testing.expectEqual(@as(u32, 4096), first.fill_at.load(.monotonic));
    try std.testing.expect(try fill.step(&f, &first, 0));
    while (!try fill.step(&f, &second, 0)) {}
    try std.testing.expectEqualSlices(u32, &.{ 2048, 4096 }, &marks.at);
    try std.testing.expectEqual(@as(usize, 0), f.filling.count());
}

test "glm prefill: a failed chunk, cancellation or state allocation releases just that filling owner" {
    var f: Fake = .{};
    defer f.deinit();
    var s = try Stream.init(a, .{ .id = "a", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 4 });
    defer s.deinit(a);
    var survivor = try Stream.init(a, .{ .id = "other", .prompt = &.{ 2, 7, 1, 8 }, .max_new = 4, .chunks = &.{2} });
    defer survivor.deinit(a);
    try std.testing.expect(!try fill.step(&f, &survivor, 0));
    f.fail_chunk = true;
    try std.testing.expectError(error.GpuFailed, fill.step(&f, &s, 0));
    try std.testing.expectEqual(@as(usize, 1), f.filling.count());
    try std.testing.expectEqual(@as(u32, 2), f.filling.get(&survivor).?.at);
    f.abortPrompt(&survivor);
    try std.testing.expectEqual(@as(usize, 0), f.target.lanes.count());
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    f.filling.deinit(a);
    f.filling = .empty;
    f.gpa = failing.allocator();
    f.fail_chunk = false;
    try std.testing.expectError(error.OutOfMemory, fill.step(&f, &s, 0));
    try std.testing.expectEqual(@as(usize, 0), f.filling.count());
    f.gpa = a;
    s.chunks = &.{2};
    try std.testing.expect(!try fill.step(&f, &s, 0));
    const Cancel = struct {
        fn yes(_: *anyopaque) bool {
            return true;
        }
    };
    var dummy: u8 = 0;
    s.cancel_check = .{ .ptr = &dummy, .check = Cancel.yes };
    try std.testing.expectError(error.Cancelled, fill.step(&f, &s, 0));
    try std.testing.expectEqual(@as(usize, 0), f.filling.count());
}

test "glm prefill: shared prompt input restores each owner's IDs and lookahead after settling decoder scratch" {
    const Input = struct {
        shared: [16]u32 = @splat(99),
        order: [3]u8 = undefined,
        n: usize = 0,
        fail: bool = false,
        pub fn settle(p: *@This()) void {
            p.order[p.n] = 1;
            p.n += 1;
        }
        pub fn flush(p: *@This()) !void {
            if (p.fail) return error.GpuFailed;
            p.order[p.n] = 2;
            p.n += 1;
        }
        pub fn stage(p: *@This(), at: u32, ts: []const u32) void {
            p.order[p.n] = 3;
            p.n += 1;
            @memcpy(p.shared[at..][0..ts.len], ts);
        }
    };
    const input = @import("prompt_input.zig");
    const a_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const b_ids = [_]u32{ 9, 8, 7, 6, 5, 4 };
    var io: Input = .{};
    try input.prepare(&io, &a_ids, 0, 2);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, &io.order);
    try std.testing.expectEqualSlices(u32, a_ids[0..3], io.shared[0..3]);
    io.n = 0;
    try input.prepare(&io, &b_ids, 0, 4);
    io.n = 0;
    try input.prepare(&io, &a_ids, 2, 4);
    try std.testing.expectEqualSlices(u32, a_ids[2..], io.shared[2..6]);
    const before = io.shared;
    io.n = 0;
    io.fail = true;
    try std.testing.expectError(error.GpuFailed, input.prepare(&io, &b_ids, 0, 2));
    try std.testing.expectEqualSlices(u32, &before, &io.shared);
    try std.testing.expectEqual(@as(usize, 1), io.n);
}

test "glm prefill: an image handoff failure invalidates the old owner before changing shared image rows" {
    const Images = struct {
        image_owner: ?*Stream,
        inject: bool = true,
        writes: usize = 0,
        fail: bool = true,
        pub fn endImages(i: *@This()) void {
            i.inject = false;
        }
        pub fn encodeImages(i: *@This(), _: u32, _: []const lanes.stream.Image) !void {
            i.writes += 1;
            if (i.fail) return error.GpuFailed;
            i.inject = true;
        }
    };
    var first = try Stream.init(a, .{ .id = "a", .prompt = &.{1}, .max_new = 1 });
    defer first.deinit(a);
    var second = try Stream.init(a, .{ .id = "b", .prompt = &.{2}, .max_new = 1 });
    defer second.deinit(a);
    const images = [_]lanes.stream.Image{.{ .pixels = &.{}, .gh = 0, .gw = 0, .at = 0, .tokens = 1 }};
    first.images = &images;
    second.images = &images;
    var io: Images = .{ .image_owner = &first };
    const input = @import("prompt_images.zig");
    try std.testing.expectError(error.GpuFailed, input.prepare(&io, &second, 1));
    try std.testing.expect(io.image_owner == null and !io.inject);
    io.fail = false;
    try input.prepare(&io, &first, 0);
    try std.testing.expect(io.image_owner == &first and io.inject);
    try std.testing.expectEqual(@as(usize, 2), io.writes);
    try input.prepare(&io, &first, 0);
    try std.testing.expectEqual(@as(usize, 2), io.writes);
}
test "glm prefill: a resumed image prompt sends the tower only the images its restored state does not hold" {
    const Images = struct {
        image_owner: ?*Stream = null,
        sent: []const lanes.stream.Image = &.{},
        writes: usize = 0,
        pub fn endImages(_: *@This()) void {}
        pub fn encodeImages(i: *@This(), _: u32, imgs: []const lanes.stream.Image) !void {
            i.writes += 1;
            i.sent = imgs;
        }
    };
    var s = try Stream.init(a, .{ .id = "a", .prompt = &.{ 1, 9, 9, 2, 9, 9, 3, 4 }, .max_new = 1 });
    defer s.deinit(a);
    var other = try Stream.init(a, .{ .id = "b", .prompt = &.{1}, .max_new = 1 });
    defer other.deinit(a);
    const images = [_]lanes.stream.Image{ .{ .pixels = &.{}, .gh = 0, .gw = 0, .at = 1, .tokens = 2 }, .{ .pixels = &.{}, .gh = 0, .gw = 0, .at = 4, .tokens = 2 } };
    s.images = &images;
    const input = @import("prompt_images.zig");
    // fresh, kept through the first image, kept inside the second: whole images from the first one not held
    for ([_]u32{ 0, 3, 5 }, [_]usize{ 2, 1, 1 }) |cached, n| {
        var io: Images = .{};
        s.cached = cached;
        try input.prepare(&io, &s, 0);
        try std.testing.expectEqual(@as(usize, 1), io.writes);
        try std.testing.expectEqual(n, io.sent.len);
        try std.testing.expectEqual(images[2 - n].at, io.sent[0].at);
        try std.testing.expect(io.image_owner == &s);
    }
    // kept past every image: no tower run, and another prompt's rows stay its own
    var io: Images = .{ .image_owner = &other };
    s.cached = 7;
    try input.prepare(&io, &s, 0);
    try std.testing.expectEqual(@as(usize, 0), io.writes);
    try std.testing.expect(io.image_owner == &other);
}
test "glm prefill: GLM advertises one-chunk prompt work to the shared lane core" {
    if (comptime !mac) return error.SkipZigTest;
    var b: Backend = undefined;
    try std.testing.expect(b.backend().vtable.prefill_step != null);
}
test "glm prefill: a decoding stream keeps its round rate during a 102k prefill and both streams finish" {
    var concurrent = try run(false, true);
    defer concurrent.deinit();
}
test "glm prefill: sampled drafted equals plain and concurrent equals solo through GLM's chunk driver" {
    var plain = try run(false, true);
    defer plain.deinit();
    var drafted = try run(true, true);
    defer drafted.deinit();
    var solo = try run(false, false);
    defer solo.deinit();
    var solo_drafted = try run(true, false);
    defer solo_drafted.deinit();
    for (0..2) |i| {
        try std.testing.expectEqualSlices(u32, plain.tokens[i].items, drafted.tokens[i].items);
        try std.testing.expectEqualSlices(u32, plain.tokens[i].items, solo.tokens[i].items);
        try std.testing.expectEqualSlices(u32, plain.tokens[i].items, solo_drafted.tokens[i].items);
    }
}

fn history(f: *Fake, s: *Stream) !void {
    try std.testing.expect(f.target.lanes.contains(s));
    const at = s.fill_at.load(.monotonic);
    try std.testing.expectEqualSlices(u32, s.prompt()[0..at], f.target.lanes.get(s).?.history.items);
}
fn draws(f: *Fake, s: *Stream) ![32]u32 {
    const b = f.target.backend();
    var handle = try b.first(s, s.prompt_len);
    var tokens: [32]u32 = undefined;
    for (&tokens, 0..) |*t, i| {
        t.* = try b.read(handle);
        if (i + 1 < tokens.len) handle = try b.queue(s, .{ .value = t.* }, s.prompt_len + i + 1);
    }
    return tokens;
}

test "glm prefill: resumed equals fresh with real fake histories and saved prefixes across chunk seams" {
    var tokens: [1300]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast((i * 13 + i / 7) % 97);
    for ([_]u64{ 1235, 1236 }) |seed| {
        var f: Fake = .{};
        defer f.deinit();
        var s = try Stream.init(a, .{ .id = "fresh", .prompt = &tokens, .max_new = 32, .sampling = .{ .seed = seed, .temperature = 1 }, .chunks = &.{ 256, 513, 769, 1025 } });
        defer s.deinit(a);
        var saved: [4]*anyopaque = undefined;
        var made: usize = 0;
        defer for (saved[0..made]) |snap| f.target.drop(snap);
        while (true) {
            const done = try fill.step(&f, &s, 0);
            try history(&f, &s);
            if (done) break;
            saved[made] = try f.target.save(&s, s.fill_at.load(.monotonic));
            made += 1;
        }
        try std.testing.expectEqual(@as(usize, 4), made);
        const expected = try draws(&f, &s);
        for (saved) |snap| {
            const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(snap));
            var resumed: Fake = .{};
            defer resumed.deinit();
            var r = try Stream.init(a, .{ .id = "resumed", .prompt = &tokens, .max_new = 32, .sampling = s.sampling, .chunks = &.{ 128, 257, 512, 700, 1024, 1200 }, .reuse = .{ .saved = snap, .at = @intCast(kept.items.len) } });
            defer r.deinit(a);
            var other = try Stream.init(a, .{ .id = "other", .prompt = &.{ 9, 2, 6, 5, 3 }, .max_new = 1, .chunks = &.{ 2, 4 } });
            defer other.deinit(a);
            var other_done = try fill.step(&resumed, &other, 0);
            while (true) {
                const done = try fill.step(&resumed, &r, 0);
                try history(&resumed, &r);
                if (!other_done) other_done = try fill.step(&resumed, &other, 0);
                try history(&resumed, &other);
                if (done) break;
            }
            try std.testing.expectEqual(@as(u32, @intCast(kept.items.len)), r.cached);
            try std.testing.expectEqualSlices(u32, &expected, &try draws(&resumed, &r));
        }
    }
}

test "glm prefill: fake completion cannot repair a corrupted earlier chunk by replaying the prompt" {
    var f: Fake = .{};
    defer f.deinit();
    var s = try Stream.init(a, .{ .id = "control", .prompt = &.{ 3, 1, 4, 1, 5 }, .max_new = 1, .chunks = &.{2} });
    defer s.deinit(a);
    try std.testing.expect(!try fill.step(&f, &s, 0));
    try history(&f, &s);
    f.target.lanes.getPtr(&s).?.history.items[0] += 1;
    try std.testing.expect(try fill.step(&f, &s, 0));
    try std.testing.expectEqual(@as(u32, 4), f.target.lanes.get(&s).?.history.items[0]);
    const expected = lanes.fake.next(s.prompt(), s.sampling, s.prompt_len);
    const b = f.target.backend();
    try std.testing.expect(expected != try b.read(try b.first(&s, s.prompt_len)));
}

test "glm prefill: the live lane host finishes a long fill amid continuing short arrivals with unchanged tokens" {
    var f: Fake = .{};
    defer f.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .streams_exact = true }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 3 });
    const tokens: [1024]u32 = @splat(7);
    const long: api.Request = .{ .prompt = &tokens, .max_tokens = 1, .sampling = .{ .seed = 1235, .temperature = 1 }, .chunks = &.{ 128, 256, 384, 512, 640, 768, 896 } };
    const short: api.Request = .{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 1, .sampling = .{ .seed = 1236, .temperature = 1 } };
    const Arrivals = struct {
        engine: api.Engine,
        short: *const api.Request,
        expected: [2]u32,
        done: std.atomic.Value(usize) = .init(0),
        bad: std.atomic.Value(bool) = .init(false),
        sent: usize = 0,
        round: usize = 0,
        last: usize = 0,
        max_gap: usize = 0,
        chunks: usize = 0,
        ended_with_arrivals: bool = false,
        fn chunk(ptr: *anyopaque, s: *Stream) void {
            const q: *@This() = @ptrCast(@alignCast(ptr));
            q.round += 1;
            if (s.prompt_len == 1024) {
                q.max_gap = @max(q.max_gap, q.round - q.last);
                q.last = q.round;
                q.chunks += 1;
                if (q.chunks == 8) q.ended_with_arrivals = q.sent < 128;
            }
            if (q.sent == 128) return;
            // Submit queues the next arrival; it does not reenter the backend or run an engine round.
            q.engine.submit(q.sent + 3, q.short, .{ .ctx = q, .event = event }) catch {
                q.bad.store(true, .release);
                return;
            };
            q.sent += 1;
        }
        fn event(ptr: *anyopaque, id: api.Id, e: *const api.Event) void {
            const q: *@This() = @ptrCast(@alignCast(ptr));
            switch (e.*) {
                .tokens => |ts| {
                    if (ts.len != 1 or ts[0] != q.expected[@intFromBool(id != 1)]) q.bad.store(true, .release);
                },
                .finished => |end| {
                    if (end.reason != .length) q.bad.store(true, .release);
                    _ = q.done.fetchAdd(1, .release);
                },
                else => {},
            }
        }
    };
    var q: Arrivals = .{ .engine = host.engine(), .short = &short, .expected = .{ lanes.fake.next(long.prompt, long.sampling, long.prompt.len), lanes.fake.next(short.prompt, short.sampling, short.prompt.len) } };
    f.chunk_hook = Arrivals.chunk;
    f.chunk_ctx = &q;
    try q.engine.submit(1, &long, .{ .ctx = &q, .event = Arrivals.event });
    try q.engine.submit(2, &short, .{ .ctx = &q, .event = Arrivals.event });
    try host.start();
    while (q.done.load(.acquire) < 130 and !q.bad.load(.acquire)) std.Thread.yield() catch {};
    host.stop();
    try std.testing.expect(!q.bad.load(.acquire));
    try std.testing.expectEqual(@as(usize, 8), q.chunks);
    try std.testing.expect(q.max_gap <= 8 + 3);
    try std.testing.expect(q.ended_with_arrivals);
    try std.testing.expectEqual(@as(usize, 130), q.done.load(.acquire));
}

test "glm prefill: the flag share keeps every CUDA chunk start on the grid while the family is gated out" {
    const hostfill = @import("../../core/lane_host_fill.zig");
    var f: Fake = .{ .grid = 2048, .decode_shrink = false }; // CUDA's 2048 grid, full-grid chunks (no Metal shrink)
    defer f.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2, .decode_share = 0.5 }); // the explicit flag share
    try std.testing.expect(host.decode_cap);
    var dec = try Stream.init(a, .{ .id = "dec", .prompt = &.{ 3, 1, 4 }, .max_new = 8 });
    defer dec.deinit(a);
    try core.addStream(&dec); // a live decoder: the policy sees concurrent load
    defer core.backend.release(&dec);
    const tokens = try a.alloc(u32, 5000);
    defer a.free(tokens);
    @memset(tokens, 7);
    var s = try Stream.init(a, .{ .id = "grid", .prompt = tokens, .max_new = 4, .chunks = &.{ 2048, 4096 } });
    defer s.deinit(a);
    var hint: u32 = hostfill.capRows(&host); // the host's prefill_step hint: 0 for a gated-off family
    try std.testing.expectEqual(@as(u32, 0), hint);
    while (!try fill.step(&f, &s, hint)) {
        hint = hostfill.capRows(&host);
        try std.testing.expectEqual(@as(u32, 0), hint); // every step arrives uncapped: l.fill rides the grid
    }
    try std.testing.expectEqualSlices(u32, &.{ 0, 2048, 4096 }, f.starts.items); // every start on 2048; BadChunk never raised
}

test "glm prefill: a family that opts into chunk invariance gets the cap, and the grid guard raises BadChunk" {
    const hostfill = @import("../../core/lane_host_fill.zig");
    var f: Fake = .{ .grid = 2048, .decode_shrink = false, .invariant = true };
    defer f.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2, .decode_share = 0.5 });
    var dec = try Stream.init(a, .{ .id = "dec", .prompt = &.{ 3, 1, 4 }, .max_new = 8 });
    defer dec.deinit(a);
    try core.addStream(&dec);
    defer core.backend.release(&dec);
    const hint = hostfill.capRows(&host);
    try std.testing.expectEqual(hostfill.decode_cap_rows, hint); // opted in: the cap now reaches the driver
    const tokens = try a.alloc(u32, 5000);
    defer a.free(tokens);
    @memset(tokens, 7);
    var s = try Stream.init(a, .{ .id = "capped", .prompt = tokens, .max_new = 4, .chunks = &.{ 2048, 4096 } });
    defer s.deinit(a);
    try std.testing.expect(!try fill.step(&f, &s, hint)); // one 512-row chunk goes out...
    try std.testing.expectError(error.BadChunk, fill.step(&f, &s, hint)); // ...and the next start, 512, breaks the grid
    try std.testing.expectEqualSlices(u32, &.{ 0, 512 }, f.starts.items); // Bizuayeu's failure, reproduced only under opt-in
}
