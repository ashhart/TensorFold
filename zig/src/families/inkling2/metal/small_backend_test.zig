//! End-to-end Metal serving tests for Inkling 2 Small EXL3: lane engine, prompt cache restore and 4-stream concurrency.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const api = @import("engine_api");
const st = @import("small_state.zig");
const backend = @import("small_backend.zig");
const synth = @import("small_forward_test.zig");

const t = std.testing;
const gpa = t.allocator;
const io = t.io;

const Reply = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    rows: std.ArrayList(lanes.LogprobRow) = .empty,
    cached: ?u32 = null,
    done: ?api.Reason = null,

    fn event(ptr: *anyopaque, _: api.Id, item: *const api.Event) void {
        const self: *Reply = @ptrCast(@alignCast(ptr));
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        switch (item.*) {
            .tokens => |ids| self.tokens.appendSlice(gpa, ids) catch {},
            .logprobs => |rows| self.rows.appendSlice(gpa, rows) catch {},
            .prefilled => |n| self.cached = n,
            .finished => |f| self.done = f.reason,
            else => {},
        }
    }

    fn wait(self: *Reply) !api.Reason {
        for (0..60_000) |_| {
            self.mutex.lockUncancelable(io);
            const done = self.done;
            self.mutex.unlock(io);
            if (done) |reason| return reason;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    fn deinit(self: *Reply) void {
        self.tokens.deinit(gpa);
        self.rows.deinit(gpa);
    }
};

fn serialDecode(b: lanes.backend.Backend, ids: []const u32, max_new: usize, out: []u32) !lanes.LogprobRow {
    var s = try lanes.Stream.init(gpa, .{ .id = "serial", .prompt = ids, .max_new = @intCast(max_new), .drafts = false, .logprobs = 2 });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    var next = try b.read(try b.first(&s, ids.len));
    const first = try b.firstRow(&s);
    out[0] = next;
    for (1..max_new) |i| {
        next = try b.read(try b.queue(&s, .{ .value = next }, ids.len + i));
        out[i] = next;
    }
    return first;
}

test "inkling2 small: LaneHost serving (solo, prompt-cache resume, and 4 concurrent streams) is bit-exact" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = device.queue() catch {
        device.deinit();
        return error.SkipZigTest;
    };
    const m = try synth.buildSyntheticModel(gpa, device, queue);
    defer m.deinit();

    const max_new = 6;
    const capacity = 64;
    const batch_rows = 16;
    const prompt_rows = 8;
    const budget: u64 = 1 << 30;

    const metal = try backend.Metal.initFor(gpa, m, .{ .capacity = capacity, .streams = 4, .budget = budget }, batch_rows, prompt_rows);
    defer metal.deinit();

    // 1. Serial baseline for 4 distinct prompts (lengths 4, 5, 6, 7).
    const p0 = [_]u32{ 3, 17, 42, 7 };
    const p1 = [_]u32{ 3, 17, 42, 7, 55 };
    const p2 = [_]u32{ 3, 17, 42, 7, 55, 12 };
    const p3 = [_]u32{ 9, 21, 33, 44, 51, 2, 18 };
    const prompts = [_][]const u32{ &p0, &p1, &p2, &p3 };

    var want_toks: [4][max_new]u32 = undefined;
    var want_first: [4]lanes.LogprobRow = undefined;
    for (prompts, 0..) |pr, i| {
        want_first[i] = try serialDecode(metal.backend(), pr, max_new, &want_toks[i]);
    }

    // 2. Spin up LaneHost + prompt cache store (grid = 4 so p0 caches at 4 and p1/p2 resume from it).
    var cfg = try lanes.Config.init(gpa, metal.facts(), st.window_rows, st.window_rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(gpa, &cfg, metal.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, io, &core, .{
        .lanes = 4,
        .context_window = capacity,
        .prefill_step = prompt_rows,
        .decode_share = 0.5,
        .structures = true,
        .logprobs = true,
    });
    const M = backend.Metal;
    var store = api.prompt_cache.Store.init(gpa, .{
        .ptr = metal,
        .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop },
    }, .{ .lookahead = 0, .planned = true, .grid = 4, .min_prompt = 0 }, budget);
    defer store.deinit();
    host.cache = &store;
    try host.start();
    defer host.stop();

    // 3. Seed prompt cache at history_len = 4 on p2 (fresh -> cached 0), then resume p2 (cached 4).
    {
        var r_fresh: Reply = .{};
        defer r_fresh.deinit();
        const req_fresh: api.Request = .{ .prompt = &p2, .max_tokens = max_new, .drafts = false, .logprobs = 2, .history_len = 4 };
        try host.engine().submit(1, &req_fresh, .{ .ctx = &r_fresh, .event = Reply.event });
        try t.expectEqual(api.Reason.length, try r_fresh.wait());
        try t.expectEqual(@as(?u32, 0), r_fresh.cached);
        try t.expectEqualSlices(u32, &want_toks[2], r_fresh.tokens.items);
        try t.expectEqual(@as(u32, @bitCast(want_first[2].logprob)), @as(u32, @bitCast(r_fresh.rows.items[0].logprob)));

        var r_resumed: Reply = .{};
        defer r_resumed.deinit();
        const req_resumed: api.Request = .{ .prompt = &p2, .max_tokens = max_new, .drafts = false, .logprobs = 2, .history_len = 4 };
        try host.engine().submit(2, &req_resumed, .{ .ctx = &r_resumed, .event = Reply.event });
        try t.expectEqual(api.Reason.length, try r_resumed.wait());
        try t.expectEqual(@as(?u32, 4), r_resumed.cached);
        try t.expectEqualSlices(u32, &want_toks[2], r_resumed.tokens.items);
        try t.expectEqual(@as(u32, @bitCast(want_first[2].logprob)), @as(u32, @bitCast(r_resumed.rows.items[0].logprob)));
    }

    // 4. Concurrent 4-stream batch: all 4 prompts served concurrently must match solo bit-for-bit.
    var reqs: [4]api.Request = undefined;
    var replies: [4]Reply = .{ .{}, .{}, .{}, .{} };
    defer for (&replies) |*r| r.deinit();
    for (prompts, 0..) |pr, i| {
        reqs[i] = .{ .prompt = pr, .max_tokens = max_new, .drafts = false, .logprobs = 2 };
        try host.engine().submit(10 + i, &reqs[i], .{ .ctx = &replies[i], .event = Reply.event });
    }
    for (&replies, 0..) |*r, i| {
        try t.expectEqual(api.Reason.length, try r.wait());
        try t.expectEqualSlices(u32, &want_toks[i], r.tokens.items);
        try t.expectEqual(@as(u32, @bitCast(want_first[i].logprob)), @as(u32, @bitCast(r.rows.items[0].logprob)));
    }
}
