//! LaneHost with Recall's record on disk: a request read again after a restart, read-only, and a resumed turn's writes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const Store = lanes.recall_store.Store;
const gpa = std.testing.allocator;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    rounds: u64 = 0,
    telemetry: [256]u8 = undefined,
    telemetry_len: usize = 0,
    done: ?Reason = null,

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .logprobs, .prompt_logprobs, .audio => {},
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.rounds = f.stats.rounds;
                b.telemetry_len = @min(b.telemetry.len, f.stats.telemetry_json.len);
                @memcpy(b.telemetry[0..b.telemetry_len], f.stats.telemetry_json[0..b.telemetry_len]);
                b.done = f.reason;
            },
        }
    }

    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
};

/// A served fake model: its lane core with a draft head, a host, and optionally a prompt cache.
const Served = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: LaneHost,

    fn init(s: *Served) !void {
        var costs: [16]lanes.config.Cost = undefined;
        for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
        s.cfg = try lanes.Config.init(gpa, .{
            .exact_width = 16,
            .gpu_tokens = true,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .drafts = 4,
            .window_costs = &costs,
            .mtp_step_ms = 0.5,
            .hidden_rows = true,
            .batch_rows = 32,
            .max_streams = 8,
            .draft_streams = true,
        }, 16, 15);
        s.target = .{ .gpa = gpa };
        s.clock = .{};
        s.core = lanes.Engine.init(gpa, &s.cfg, s.target.backend(), s.clock.clock());
        s.host = LaneHost.init(gpa, std.testing.io, &s.core, .{ .lanes = 2 });
    }

    fn deinit(s: *Served) void {
        s.host.stop();
        s.core.deinit();
        s.target.deinit();
        s.cfg.deinit(gpa);
    }

    fn run(s: *Served, r: *const Request, id: Id, b: *Box) !Reason {
        try s.host.engine().submit(id, r, .{ .ctx = b, .event = Box.event });
        return b.wait();
    }
};

fn openStore(path: []const u8, read_only: bool) !*Store {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var problem: []const u8 = "";
    const id: lanes.recall_store.Identity = .{ .digest = @splat(9), .vocab = lanes.fake.vocab, .name = "fake" };
    const o: lanes.recall_store.Options = .{ .dir = path, .cap_bytes = 64 << 20, .read_only = read_only };
    const store = (try Store.open(gpa, std.testing.io, id, o, &problem, arena.allocator())) orelse return error.TestRefused;
    store.admission.floor = 1 << 20;
    return store;
}

fn prompt(comptime n: usize, seed: u32) [n]u32 {
    var out: [n]u32 = undefined;
    var x: u32 = seed;
    for (&out) |*t| {
        x = x *% 1103515245 +% 12345;
        t.* = (x >> 16) % 90;
    }
    return out;
}

test "a request read again after a restart copies its earlier reply: the same tokens, fewer rounds, counters in stats" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/recall", .{tmp.sub_path});
    defer gpa.free(path);
    const p = prompt(200, 5);
    const r: Request = .{ .prompt = &p, .max_tokens = 120 };
    var first: Box = .{};
    defer first.deinit();
    {
        const store = try openStore(path, false);
        defer store.close();
        var s: Served = undefined;
        try s.init();
        defer s.deinit();
        try std.testing.expect(s.host.engine().attachRecall(store));
        try s.host.start();
        try std.testing.expectEqual(Reason.length, try s.run(&r, 1, &first));
        try std.testing.expectEqual(@as(u64, 200 + 64 + 120), store.added);
    }
    const store = try openStore(path, false); // a new server process: the record is all it has
    defer store.close();
    var s: Served = undefined;
    try s.init();
    defer s.deinit();
    try std.testing.expect(s.host.engine().attachRecall(store));
    try s.host.start();
    var again: Box = .{};
    defer again.deinit();
    try std.testing.expectEqual(Reason.length, try s.run(&r, 2, &again));
    try std.testing.expectEqualSlices(u32, first.tokens.items, again.tokens.items);
    try std.testing.expect(again.rounds < first.rounds);
    try std.testing.expect(std.mem.indexOf(u8, again.telemetry[0..again.telemetry_len], "\"accepted\": 0") == null);
    try std.testing.expectEqual(@as(u64, 0), store.added); // all of it was held already
    var plain: Box = .{};
    defer plain.deinit();
    const ref: Request = .{ .prompt = &p, .max_tokens = 120, .drafts = false };
    try std.testing.expectEqual(Reason.length, try s.run(&ref, 3, &plain));
    try std.testing.expectEqualSlices(u32, first.tokens.items, plain.tokens.items);
}

test "a read-only record serves copies and adds nothing" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/recall", .{tmp.sub_path});
    defer gpa.free(path);
    const p = prompt(150, 6);
    const r: Request = .{ .prompt = &p, .max_tokens = 100 };
    var first: Box = .{};
    defer first.deinit();
    {
        const store = try openStore(path, false);
        defer store.close();
        var s: Served = undefined;
        try s.init();
        defer s.deinit();
        _ = s.host.engine().attachRecall(store);
        try s.host.start();
        _ = try s.run(&r, 1, &first);
    }
    const store = try openStore(path, true);
    defer store.close();
    var s: Served = undefined;
    try s.init();
    defer s.deinit();
    _ = s.host.engine().attachRecall(store);
    try s.host.start();
    var again: Box = .{};
    defer again.deinit();
    const other = prompt(120, 7);
    const r2: Request = .{ .prompt = &other, .max_tokens = 40 };
    var fresh: Box = .{};
    defer fresh.deinit();
    _ = try s.run(&r, 2, &again);
    _ = try s.run(&r2, 3, &fresh);
    try std.testing.expectEqualSlices(u32, first.tokens.items, again.tokens.items);
    try std.testing.expect(again.rounds < first.rounds);
    try std.testing.expectEqual(@as(u64, 0), store.added);
}

test "a turn resumed from the prompt cache adds the ids past its resume point and the overlap before it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/recall", .{tmp.sub_path});
    defer gpa.free(path);
    const store = try openStore(path, false);
    defer store.close();
    var s: Served = undefined;
    try s.init();
    defer s.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    const snaps: pc.Snapshots.VTable = .{
        .bytes = Snaps.bytes,
        .save = Snaps.save,
        .restore = Snaps.restore,
        .drop = Snaps.drop,
    };
    const family: pc.Snapshots = .{ .ptr = &s.target, .vtable = &snaps };
    var cache = pc.Store.init(gpa, family, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer cache.deinit();
    s.host.cache = &cache;
    _ = s.host.engine().attachRecall(store);
    try s.host.start();
    const t1 = prompt(300, 8);
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 50, .history_len = 290 };
    _ = try s.run(&r1, 1, &b1);
    const after1 = store.added;
    var t2: std.ArrayList(u32) = .empty;
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..40]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 20, .history_len = 340 };
    _ = try s.run(&r2, 2, &b2);
    try std.testing.expectEqual(@as(?u32, 290), b2.cached);
    // the prompt from 290 less the overlap (118 ids), then the prompt's last 64 with the reply's 20
    try std.testing.expectEqual(after1 + (344 - (290 - 64)) + 64 + 20, store.added);
}

test "a prompt with an image gets the same reply drafted and plain, each prefill holding the image" {
    var s: Served = undefined;
    try s.init();
    defer s.deinit();
    const Seen = struct {
        n: usize = 0,
        fn hook(ptr: *anyopaque, stream: *lanes.Stream, _: usize) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (stream.images.len == 1 and stream.images[0].at == 2 and
                stream.images[0].pixels[0] == 0.25) self.n += 1;
        }
    };
    var seen: Seen = .{};
    s.target.prefill_hook = Seen.hook;
    s.target.prefill_hook_ctx = &seen;
    try s.host.start();
    var drafted: Box = .{};
    defer drafted.deinit();
    var plain: Box = .{};
    defer plain.deinit();
    const image = api.Image{ .pixels = &.{0.25}, .gh = 2, .gw = 2, .at = 2, .tokens = 1 };
    const request: Request = .{ .prompt = &.{ 3, 4, 9, 5, 6 }, .images = &.{image}, .max_tokens = 24 };
    try std.testing.expectEqual(Reason.length, try s.run(&request, 1, &drafted));
    var serial = request;
    serial.drafts = false;
    try std.testing.expectEqual(Reason.length, try s.run(&serial, 2, &plain));
    try std.testing.expectEqualSlices(u32, drafted.tokens.items, plain.tokens.items);
    try std.testing.expectEqual(@as(usize, 2), seen.n);
    try std.testing.expect(drafted.rounds < plain.rounds);
}
