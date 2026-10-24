//! Interrupted prompt passes retain completed state through the real host and prompt store.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const t = std.testing;
const a = t.allocator;

const Reply = struct {
    ids: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    reason: api.Reason = .failed,
    done: std.atomic.Value(bool) = .init(false),
    cancel_engine: ?api.Engine = null,

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .prefilled => |n| r.cached = n,
            .tokens => |ids| {
                r.ids.appendSlice(a, ids) catch @panic("test allocation failed");
                if (r.cancel_engine) |engine| engine.cancel(id);
                r.cancel_engine = null;
            },
            .finished => |f| {
                r.reason = f.reason;
                r.done.store(true, .release);
            },
            else => {},
        }
    }

    fn wait(r: *Reply) !api.Reason {
        const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
        while (!r.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(t.io).toNanoseconds() >= until) return error.RequestTimedOut;
            try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
        }
        return r.reason;
    }

    fn exact(r: *const Reply, input: []const u32) !void {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(a);
        try history.appendSlice(a, input);
        try t.expectEqual(@as(usize, 12), r.ids.items.len);
        for (r.ids.items) |id| {
            try t.expectEqual(lanes.fake.next(history.items, null, history.items.len), id);
            try history.append(a, id);
        }
    }
};

const Driver = struct {
    target: *lanes.fake.Fake,
    engine: api.Engine,
    id: api.Id = 1,
    stop_at: ?usize = 4,
    started_at: u32 = 0,
    saves: usize = 0,
    fail_prefill: bool = false,
    fail_save: bool = false,
    fail_cancel: bool = false,

    fn of(ctx: *anyopaque) *Driver {
        return @ptrCast(@alignCast(ctx));
    }

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) !void {
        while (!try step(ptr, s)) {}
    }

    fn step(ptr: *anyopaque, s: *lanes.Stream) !bool {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const d = of(f.prefill_hook_ctx.?);
        const got = try f.lanes.getOrPut(a, s);
        if (!got.found_existing) got.value_ptr.* = .{};
        const h = &got.value_ptr.history;
        if (!got.found_existing) {
            if (s.reuse.saved) |saved| {
                const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
                try h.appendSlice(a, kept.items);
                s.cached = s.reuse.at;
            }
            d.started_at = s.cached;
        }
        if (d.stop_at == h.items.len) {
            if (d.fail_prefill) return error.GpuFailed;
            d.engine.cancel(d.id);
        }
        if (s.isCancelled()) {
            s.keepInterruptedPrefill(@intCast(h.items.len));
            return error.Cancelled;
        }
        const end = @min(h.items.len + 4, s.prompt_len);
        try h.appendSlice(a, s.prompt()[h.items.len..end]);
        s.fill_at.store(@intCast(end), .monotonic);
        if (d.stop_at == end and !d.fail_prefill) d.engine.cancel(d.id);
        return end == s.prompt_len;
    }

    fn cancelPrefill(ptr: *anyopaque, s: *lanes.Stream) !void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        if (of(f.prefill_hook_ctx.?).fail_cancel) return error.PeerFailed;
        const l = f.lanes.getPtr(s) orelse return;
        s.keepInterruptedPrefill(@intCast(l.history.items.len));
    }

    fn bytes(_: *anyopaque, _: u32) u64 {
        return 100;
    }
    fn save(ctx: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const d = of(ctx);
        if (d.fail_save) return error.SnapshotFailed;
        const saved = try d.target.save(@ptrCast(@alignCast(owner.?)), at);
        d.saves += 1;
        return saved;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ctx: *anyopaque, saved: pc.Saved) void {
        of(ctx).target.drop(saved);
    }

    fn submit(d: *Driver, request: *const api.Request, reply: *Reply) !void {
        try d.engine.submit(d.id, request, .{ .ctx = reply, .event = Reply.event });
    }
};

const snapshots: pc.Snapshots.VTable = .{
    .bytes = Driver.bytes,
    .save = Driver.save,
    .restore = Driver.restore,
    .drop = Driver.drop,
};
const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };

fn repeated(stepped: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var vt = target.backend().vtable.*;
    vt.prefill = Driver.prefill;
    vt.prefill_step = if (stepped) Driver.step else null;
    vt.cancel_prefill = Driver.cancelPrefill;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, t.io, &core, .{ .lanes = 2 });
    var driver: Driver = .{ .target = &target, .engine = host.engine() };
    target.prefill_hook_ctx = &driver;
    var store = pc.Store.init(a, .{ .ptr = &driver, .vtable = &snapshots }, .{ .min_prompt = 0 }, 200);
    defer store.deinit();
    host.cache = &store;
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 12 };
    const stops = [_]usize{ 4, 4, 8 };
    const starts = [_]u32{ 0, 4, 4 };
    const saves = [_]usize{ 1, 1, 2 };
    for (stops, starts, saves) |stop, start, count| {
        driver.stop_at = stop;
        var reply: Reply = .{};
        defer reply.ids.deinit(a);
        try driver.submit(&request, &reply);
        try t.expectEqual(api.Reason.cancelled, try reply.wait());
        try t.expectEqual(start, driver.started_at);
        try t.expectEqual(count, driver.saves);
        try t.expectEqual(@as(usize, 0), reply.ids.items.len);
        driver.id += 1;
    }
    driver.stop_at = null;
    var resumed: Reply = .{};
    defer resumed.ids.deinit(a);
    var plain: Reply = .{};
    defer plain.ids.deinit(a);
    var plain_request = request;
    plain_request.drafts = false;
    try driver.submit(&request, &resumed);
    driver.id += 1;
    try driver.submit(&plain_request, &plain);
    try t.expectEqual(api.Reason.length, try resumed.wait());
    try t.expectEqual(api.Reason.length, try plain.wait());
    try t.expectEqual(@as(?u32, 8), resumed.cached);
    try t.expectEqual(@as(?u32, 8), plain.cached);
    try resumed.exact(&prompt);
    try t.expectEqualSlices(u32, resumed.ids.items, plain.ids.items);
    var edited = prompt;
    edited[0] += 1;
    var changed = request;
    changed.prompt = &edited;
    var reply: Reply = .{};
    defer reply.ids.deinit(a);
    driver.id += 1;
    try driver.submit(&changed, &reply);
    try t.expectEqual(api.Reason.length, try reply.wait());
    try t.expectEqual(@as(?u32, 0), reply.cached);
    try reply.exact(&edited);
}

test "interrupted prefill: whole passes retain repeated progress and concurrent retries equal fresh" {
    try repeated(false);
}

test "interrupted prefill: cancellation between host steps retains progress and concurrent retries equal fresh" {
    try repeated(true);
}

test "interrupted prefill: cache refusals, zero progress and backend failures leave clean retries" {
    for (0..7) |mode| {
        var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(a);
        var target: lanes.fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var vt = target.backend().vtable.*;
        vt.prefill = Driver.prefill;
        vt.prefill_step = if (mode >= 5) Driver.step else null;
        vt.cancel_prefill = Driver.cancelPrefill;
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt }, clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(a, t.io, &core, .{ .lanes = 2 });
        var driver: Driver = .{
            .target = &target,
            .engine = host.engine(),
            .stop_at = if (mode == 6) null else if (mode == 3) 0 else 4,
            .fail_save = mode == 2,
            .fail_prefill = mode == 4,
            .fail_cancel = mode >= 5,
        };
        target.prefill_hook_ctx = &driver;
        var store = pc.Store.init(a, .{ .ptr = &driver, .vtable = &snapshots }, .{ .min_prompt = 0 }, if (mode == 1) 0 else 200);
        defer store.deinit();
        if (mode != 0) host.cache = &store;
        try host.start();
        defer host.stop();
        const request: api.Request = .{ .prompt = &prompt, .max_tokens = 12 };
        var stopped: Reply = .{ .cancel_engine = if (mode == 6) driver.engine else null };
        defer stopped.ids.deinit(a);
        try driver.submit(&request, &stopped);
        try t.expectEqual(if (mode == 4 or mode == 5) api.Reason.failed else .cancelled, try stopped.wait());
        try t.expectEqual(@as(usize, 0), store.entries.items.len);
        try t.expectEqual(@as(usize, 0), driver.saves);
        driver.stop_at = null;
        driver.id += 1;
        var retry: Reply = .{};
        defer retry.ids.deinit(a);
        try driver.submit(&request, &retry);
        try t.expectEqual(api.Reason.length, try retry.wait());
        try t.expectEqual(@as(?u32, 0), retry.cached);
        try retry.exact(&prompt);
    }
}
