//! A queued follow-up takes the completed warm prefix instead of paying for the same chunk again.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const t = std.testing;
const a = t.allocator;
const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };

const Reply = struct {
    driver: *Driver,
    ids: std.ArrayList(u32) = .empty,
    cached: u32 = 0,
    at: usize = 0,
    reason: api.Reason = .failed,
    done: std.atomic.Value(bool) = .init(false),
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const r: *Reply = @ptrCast(@alignCast(ptr));
        switch (e.*) {
            .prefilled => |n| {
                r.cached = n;
                r.at = r.driver.completed;
            },
            .tokens => |ids| r.ids.appendSlice(a, ids) catch @panic("test allocation failed"),
            .finished => |f| {
                r.reason = f.reason;
                r.done.store(true, .release);
            },
            else => {},
        }
    }
    fn wait(r: *Reply) !void {
        const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
        while (!r.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(t.io).toNanoseconds() >= until) return error.RequestTimedOut;
            try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
        }
    }
};

const Driver = struct {
    target: *lanes.fake.Fake,
    engine: api.Engine,
    foreground: ?*const api.Request = null,
    reply: ?*Reply = null,
    completed: usize = 0,
    warm_chunks: usize = 0,
    arrived: bool = false,
    restored: u32 = 0,
    fn of(ptr: *anyopaque) *Driver {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @ptrCast(@alignCast(f.prefill_hook_ctx.?));
    }
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) !void {
        const d = of(ptr);
        while (!s.yields()) if (try step(ptr, s, 0)) return;
        const at = if (d.target.lanes.getPtr(s)) |l| l.history.items.len else 0;
        s.keepInterruptedPrefill(@intCast(at));
        return error.Yielded;
    }
    fn step(ptr: *anyopaque, s: *lanes.Stream, _: u32) !bool {
        const d = of(ptr);
        const got = try d.target.lanes.getOrPut(a, s);
        if (!got.found_existing) {
            got.value_ptr.* = .{};
            if (s.reuse.saved) |saved| {
                const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
                try got.value_ptr.history.appendSlice(a, kept.items);
                s.cached = s.reuse.at;
            }
            if (s.max_new > 0) d.restored = s.cached;
        }
        const h = &got.value_ptr.history;
        const end = @min(h.items.len + 4, s.prompt_len);
        while (h.items.len < end) {
            try h.append(a, s.prompt()[h.items.len]);
            if (s.max_new == 0 and !d.arrived and h.items.len == 6) {
                d.arrived = true;
                try d.engine.submit(2, d.foreground.?, .{ .ctx = d.reply.?, .event = Reply.event });
            }
            const at: u32 = @intCast(h.items.len);
            if (std.mem.indexOfScalar(u32, s.reuse.marks, at) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, at);
        }
        d.completed += 1;
        d.warm_chunks += @intFromBool(s.max_new == 0);
        s.fill_at.store(@intCast(end), .monotonic);
        return end == s.prompt_len;
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const d: *Driver = @ptrCast(@alignCast(ptr));
        return d.target.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const d: *Driver = @ptrCast(@alignCast(ptr));
        d.target.drop(saved);
    }
};

const Result = struct { ids: [8]u32, cached: u32, completed: usize, restored: u32, warm_chunks: usize };
const Options = struct { image: bool = false, reserved: bool = false, warm_images: bool = false };
fn run(warming: bool, sampled: bool, options: Options) !Result {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var vt = target.backend().vtable.*;
    vt.prefill = Driver.prefill;
    vt.prefill_step = Driver.step;
    vt.reports_fill = true;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt, .warm_images = options.warm_images }, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, t.io, &core, .{ .lanes = 2, .warm_turns = warming });
    var d: Driver = .{ .target = &target, .engine = host.engine() };
    target.prefill_hook_ctx = &d;
    var store = pc.Store.init(a, .{ .ptr = &d, .vtable = &.{ .bytes = Driver.bytes, .save = Driver.save, .restore = Driver.restore, .drop = Driver.drop } }, .{ .min_prompt = 0, .min_gap = 0, .lookahead = if (options.reserved) 0 else 1, .warm = .waits }, if (options.reserved) 224 else 128);
    defer store.deinit();
    host.cache = &store;
    var seed = try lanes.Stream.init(a, .{ .id = "seed", .prompt = &words, .max_new = 0 });
    defer seed.deinit(a);
    const initial = try target.lanes.getOrPut(a, &seed);
    initial.value_ptr.* = .{};
    try initial.value_ptr.history.appendSlice(a, words[0..4]);
    const pictures: []const api.Image = if (options.image) &.{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 2, .tokens = 1, .hash = 17 }} else &.{};
    try t.expect(store.keepBornFrame(words[0..20], 4, 4, &seed, &.{}, &.{}, pictures, 0));
    target.backend().release(&seed);
    const request: api.Request = .{ .prompt = &words, .max_tokens = 8, .history_len = 20, .born = 4, .images = pictures, .sampling = if (sampled) .{ .seed = 19, .temperature = 1 } else null };
    var real: Reply = .{ .driver = &d };
    defer real.ids.deinit(a);
    var warm: Reply = .{ .driver = &d };
    defer warm.ids.deinit(a);
    const pass: api.Request = .{ .prompt = words[0..20], .max_tokens = 0, .background = true, .history_len = 20, .born = 4, .images = pictures };
    d.foreground = &request;
    d.reply = &real;
    const other_words: [24]u32 = @splat(99);
    const other_request: api.Request = .{ .prompt = &other_words, .max_tokens = 1, .history_len = 20 };
    var other: Reply = .{ .driver = &d };
    defer other.ids.deinit(a);
    if (options.reserved) {
        try host.engine().submit(3, &other_request, .{ .ctx = &other, .event = Reply.event });
        try t.expect(host.admitOne());
        try t.expectEqual(@as(usize, 1), host.filling.items.len);
        try t.expectEqual(@as(u64, 120), host.filling.items[0].cache_promise.bytes);
        try t.expectEqual(store.budget, store.held + host.filling.items[0].cache_promise.bytes);
    }
    var before_work: usize = 0;
    var before_promise: u64 = 120;
    var before_held: u64 = 104;
    if (warming) {
        try host.engine().submit(1, &pass, .{ .ctx = &warm, .event = Reply.event });
        if (options.reserved) {
            try t.expect(host.admitOne());
            try warm.wait();
            before_work = d.completed;
            before_promise = host.filling.items[0].cache_promise.bytes;
            before_held = store.held;
        }
    }
    if (options.reserved and !d.arrived) try host.engine().submit(2, &request, .{ .ctx = &real, .event = Reply.event });
    try host.start();
    defer host.stop();
    if (!options.reserved) {
        if (warming and options.image) {
            try warm.wait();
            if (!d.arrived) try host.engine().submit(2, &request, .{ .ctx = &real, .event = Reply.event });
        } else if (!warming) try host.engine().submit(2, &request, .{ .ctx = &real, .event = Reply.event });
    }
    try real.wait();
    try t.expectEqual(api.Reason.length, real.reason);
    if (warming) {
        try warm.wait();
        try t.expectEqual(api.Reason.cancelled, warm.reason);
        if (!options.image and !options.reserved) try t.expect(d.arrived);
    }
    if (options.reserved) {
        try other.wait();
        try t.expectEqual(@as(usize, 0), before_work);
        try t.expectEqual(@as(u64, 120), before_promise);
        try t.expectEqual(@as(u64, 104), before_held);
    }
    try t.expectEqual(@as(usize, 8), real.ids.items.len);
    var out: Result = .{ .ids = undefined, .cached = real.cached, .completed = real.at, .restored = d.restored, .warm_chunks = d.warm_chunks };
    @memcpy(&out.ids, real.ids.items);
    return out;
}

test "queued cache: progress a follow-up arriving mid-warm-chunk uses that chunk before further prefill" {
    for ([_]bool{ false, true }) |sampled| {
        const main = try run(false, sampled, .{});
        const warm = try run(true, sampled, .{});
        errdefer std.debug.print("warm handoff: main cached={d} chunks={d}; warm cached={d} chunks={d}\n", .{ main.cached, main.completed, warm.cached, warm.completed });
        try t.expectEqual(@as(u32, 4), main.cached);
        try t.expectEqual(@as(usize, 5), main.completed);
        try t.expectEqual(@as(u32, 8), warm.cached);
        try t.expectEqual(@as(u32, 8), warm.restored);
        try t.expect(warm.completed <= main.completed);
        try t.expectEqualDeep(main.ids, warm.ids);
    }
}

test "queued cache: progress a warm handoff keeps other promises and never replaces an unrelated key" {
    const queue = @import("lane_cache_queue.zig");
    const Mode = enum { matching, different, reserved, full, unrelated };
    for ([_]Mode{ .matching, .different, .reserved, .full, .unrelated }) |mode| {
        var cfg = try lanes.Config.init(a, .{ .exact_width = 8 }, 8, 7);
        defer cfg.deinit(a);
        var target: lanes.fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var family: @import("prompt_cache_fake.zig").Fake = .{ .gpa = a };
        var store = pc.Store.init(a, family.snapshots(), .{ .min_prompt = 0, .min_gap = 0 }, if (mode == .reserved or mode == .full) 224 else 128);
        defer store.deinit();
        store.most = if (mode == .reserved or mode == .full) 2 else 1;
        var host = api.LaneHost.init(a, t.io, &core, .{ .lanes = 3, .warm_turns = true });
        host.cache = &store;
        defer host.admitted.deinit(a);
        defer host.queued.deinit(a);
        const other: [24]u32 = @splat(99);
        const pass: api.Request = .{ .prompt = &words, .background = true, .max_tokens = 0, .history_len = 20 };
        const request: api.Request = .{ .prompt = &words, .cache_key = if (mode == .different) &other else &.{}, .max_tokens = 1 };
        const unrelated: api.Request = .{ .prompt = &other, .max_tokens = 1 };
        var warm: api.LaneHost.Job = .{ .host = &host, .id = 1, .request = &pass, .sink = undefined, .started = true, .marks = if (mode == .full) &.{} else &.{20}, .cache_promise = .{ .bytes = if (mode == .full) 0 else 16 } };
        warm.stream = try lanes.Stream.init(a, .{ .id = "handoff", .prompt = &words, .max_new = 0 });
        defer warm.stream.deinit(a);
        var next: api.LaneHost.Job = .{ .host = &host, .id = 2, .request = &request, .sink = undefined };
        var later: api.LaneHost.Job = .{ .host = &host, .id = 3, .request = &unrelated, .sink = undefined, .cache_promise = .{ .bytes = if (mode == .full) 120 else 80, .slots = 1 } };
        try host.admitted.append(a, &warm);
        if (mode == .reserved or mode == .full) try host.admitted.append(a, &later);
        try host.queued.append(a, &next);
        const resident = if (mode == .unrelated) &other else &words;
        _ = family.pass(&store, resident[0..4], .{});
        try t.expect(store.keepBorn(resident, 4, 4, null, &.{}, &.{}));
        _ = family.pass(&store, words[0..8], .{});
        queue.interrupted(&host, &warm, &warm.stream, 8);
        if (mode == .unrelated) {
            try t.expectEqual(@as(u32, 4), store.find(&other, &.{}, &.{}).?.at);
            try t.expect(store.find(&words, &.{}, &.{}) == null);
        } else {
            // Main's cancelled keep can deepen its own key even without a matching queued follow-up.
            const want: u32 = if (mode == .full) 4 else 8;
            try t.expectEqual(want, store.find(&words, &.{}, &.{}).?.at);
        }
        try t.expectEqual(@as(u64, 0), warm.cache_promise.bytes); // cancellation abandons it even without a warm handoff
        try t.expectEqual(@as(u64, if (mode == .full) 120 else 80), later.cache_promise.bytes);
        try t.expectEqual(@as(usize, 1), later.cache_promise.slots);
        const promised: u64 = if (mode == .reserved or mode == .full) later.cache_promise.bytes else 0;
        try t.expect(store.held + promised <= store.budget);
    }
}

test "queued cache: progress image warm passes run no chunk and preserve ordinary image follow-ups" {
    for ([_]bool{ false, true }) |sampled| {
        const main = try run(false, sampled, .{ .image = true });
        const warm = try run(true, sampled, .{ .image = true });
        errdefer std.debug.print("image warm: main cached={d} chunks={d}; warm cached={d} chunks={d}\n", .{ main.cached, main.completed, warm.cached, warm.completed });
        try t.expectEqual(@as(u32, 4), main.cached);
        try t.expectEqual(@as(usize, 0), warm.warm_chunks);
        try t.expectEqual(main.cached, warm.cached);
        try t.expectEqual(main.completed, warm.completed);
        try t.expectEqualDeep(main.ids, warm.ids);
    }
}

test "queued cache: progress another admitted cache promise yields warm work before its first chunk" {
    for ([_]bool{ false, true }) |sampled| {
        const main = try run(false, sampled, .{ .reserved = true });
        const warm = try run(true, sampled, .{ .reserved = true });
        try t.expectEqual(@as(u32, 4), main.cached);
        try t.expectEqual(@as(usize, 0), warm.warm_chunks);
        try t.expectEqual(main.cached, warm.cached);
        try t.expectEqual(main.completed, warm.completed);
        try t.expectEqualDeep(main.ids, warm.ids);
    }
}

test "queued cache: progress image warm policy belongs to the target through wrappers" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    try t.expect(target.backend().warm_images);
    for ([_]bool{ false, true }) |allowed| {
        var backend = target.backend();
        backend.warm_images = allowed;
        var recorded: lanes.recorder.Recorded = .{ .inner = backend, .recorder = undefined };
        try t.expectEqual(allowed, recorded.backend().warm_images);
        var drafted: lanes.drafted.Drafted = .{ .gpa = a, .target = recorded.backend(), .drafter = .{ .ptr = &target, .vtable = undefined } };
        try t.expectEqual(allowed, drafted.backend().warm_images);
    }
    const unchanged = try run(true, false, .{ .image = true, .warm_images = true });
    try t.expectEqual(@as(usize, 1), unchanged.warm_chunks);
}
