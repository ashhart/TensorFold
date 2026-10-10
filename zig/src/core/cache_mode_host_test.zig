//! Cache identity follows consumed-row arithmetic while the real host drives snapshots and replies.
const std = @import("std");
const api = @import("engine_api.zig");
const serial = @import("serial_host.zig");
const pc = @import("prompt_cache.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const State = struct { at: u32 = 0, value: u64 = 1 };
const Fixture = struct {
    state: State = .{},
    restores: usize = 0,
    resets: usize = 0,
    callbacks: usize = 0,
    feed_stop: bool = false,
    hash_draw: bool = false,
    batch_words: [16]u32 = undefined,
    fn self(ptr: *anyopaque) *Fixture {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        self(ptr).state = .{};
        self(ptr).resets += 1;
    }
    fn rows(f: *Fixture, ids: []const u32, is_decoded: bool) void {
        f.callbacks += 1;
        for (ids) |token| {
            f.state.value = (f.state.value *% 1099511628211) ^ token ^ (if (is_decoded) @as(u64, 0x9e3779b97f4a7c15) else @as(u64, 1));
            f.state.at += 1;
        }
    }
    fn prompt(ptr: *anyopaque, ids: []const u32, _: bool) !void {
        self(ptr).rows(ids, false);
    }
    fn decoded(ptr: *anyopaque, ids: []const u32, _: bool) !void {
        self(ptr).rows(ids, true);
    }
    fn draw(ptr: *anyopaque, _: ?api.Sampling) !u32 {
        const f = self(ptr);
        return if (f.hash_draw) @truncate(f.state.value) else 1000 + f.state.at;
    }
    fn advance(ptr: *anyopaque, token: u32) !void {
        self(ptr).rows(&.{token}, true);
    }
    fn fed(ptr: *anyopaque) u64 {
        return self(ptr).state.at;
    }
    fn driver(f: *Fixture, store: ?*pc.Store, supports_decode: bool) serial.Driver {
        return .{ .ctx = f, .info = .{ .name = "mode-fixture", .context_window = 512, .prefill_step = 128 }, .reset = reset, .prompt_chunk = prompt, .decode_chunk = if (supports_decode) decoded else null, .fed = fed, .draw = draw, .advance = advance, .draft_batch = batch, .cache = store };
    }
    fn batch(ptr: *anyopaque, budget: u32, eos: []const u32) !serial.Batch {
        const f = self(ptr);
        if (budget == 0 or budget > f.batch_words.len) return error.BadFixtureBatch;
        var count: usize = 0;
        while (count < budget) {
            if (count > 0) try advance(ptr, f.batch_words[count - 1]);
            const token = try draw(ptr, null);
            f.batch_words[count] = token;
            count += 1;
            if (std.mem.indexOfScalar(u32, eos, token) != null) {
                if (f.feed_stop) try advance(ptr, token);
                break;
            }
        }
        return .{ .tokens = f.batch_words[0..count], .stats = .{ .rounds = 1 } };
    }
    fn snapshots(f: *Fixture) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return @sizeOf(State);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        if (self(ptr).state.at != at) return error.BadPosition;
        const state = try a.create(State);
        state.* = self(ptr).state;
        return state;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) !void {
        const state: *const State = @ptrCast(@alignCast(saved));
        self(ptr).state = state.*;
        self(ptr).restores += 1;
    }
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        const state: *State = @ptrCast(@alignCast(saved));
        a.destroy(state);
    }
};
const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?api.Reason = null,
    expected: api.Reason = .length,
    from: u32 = 0,
    tokens: [32]u32 = undefined,
    count: usize = 0,
    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (value.*) {
            .prefilled => |from| b.from = @intCast(from),
            .tokens => |ids| {
                @memcpy(b.tokens[b.count..][0..ids.len], ids);
                b.count += ids.len;
            },
            .finished => |finish| b.done = finish.reason,
        }
    }
    fn wait(b: *Box) !void {
        for (0..2000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) |reason| return std.testing.expectEqual(b.expected, reason);
            try std.Io.sleep(io, .fromMilliseconds(1), .awake);
        }
        return error.HostTimeout;
    }
};
const Result = struct { from: u32, tokens: [32]u32, count: usize };
fn request(host: *serial.Host, ids: []const u32, spans: []const [2]u32, history: u32, count: u32) !Result {
    return requestImpl(host, ids, spans, history, count, false);
}
fn requestImpl(host: *serial.Host, ids: []const u32, spans: []const [2]u32, history: u32, count: u32, drafts: bool) !Result {
    return requestWithStop(host, ids, spans, history, count, drafts, &.{}, .length);
}
fn requestWithStop(host: *serial.Host, ids: []const u32, spans: []const [2]u32, history: u32, count: u32, drafts: bool, eos: []const u32, expected: api.Reason) !Result {
    var box: Box = .{ .expected = expected };
    const r = api.Request{ .prompt = ids, .max_tokens = count, .decode_spans = spans, .history_len = history, .drafts = drafts, .eos = eos };
    try host.engine().submit(1, &r, .{ .ctx = &box, .event = Box.event });
    try box.wait();
    return .{ .from = box.from, .tokens = box.tokens, .count = box.count };
}
fn seedTokens() [32]u32 {
    var out: [32]u32 = undefined;
    for (&out, 1..) |*token, i| token.* = @intCast(i);
    return out;
}
fn fresh(tokens: []const u32, spans: []const [2]u32, supports_decode: bool) !State {
    var f: Fixture = .{};
    const host = try serial.Host.init(a, io, f.driver(null, supports_decode));
    defer host.deinit();
    _ = try request(host, tokens, spans, 0, 0);
    return f.state;
}

test "serial host restores a user edit from rewind with fresh state and output" {
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const edited = [_]u32{ 1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11, 12 };
    var f: Fixture = .{ .hash_draw = true };
    var store = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 2 * @sizeOf(State));
    defer store.deinit();
    const host = try serial.Host.init(a, io, f.driver(&store, false));
    defer host.deinit();
    var box: Box = .{};
    const r1 = api.Request{ .prompt = &first, .max_tokens = 1, .history_len = 10, .rewind_len = 4 };
    try host.engine().submit(1, &r1, .{ .ctx = &box, .event = Box.event });
    try box.wait();
    box = .{};
    const r2 = api.Request{ .prompt = &edited, .max_tokens = 1, .history_len = 10, .rewind_len = 4 };
    try host.engine().submit(2, &r2, .{ .ctx = &box, .event = Box.event });
    try box.wait();
    try std.testing.expectEqual(@as(u32, 3), box.from);
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);

    var cold: Fixture = .{ .hash_draw = true };
    const cold_host = try serial.Host.init(a, io, cold.driver(null, false));
    defer cold_host.deinit();
    var cold_box: Box = .{};
    try cold_host.engine().submit(3, &r2, .{ .ctx = &cold_box, .event = Box.event });
    try cold_box.wait();
    try std.testing.expectEqual(@as(u32, 0), cold_box.from);
    try std.testing.expectEqualDeep(cold.state, f.state);
    try std.testing.expectEqualSlices(u32, cold_box.tokens[0..cold_box.count], box.tokens[0..box.count]);
}

test "serial warm pass keeps both its endpoint and the earlier user rewind" {
    const warm = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const edited = [_]u32{ 1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11 };
    var f: Fixture = .{ .hash_draw = true };
    var store = pc.Store.init(a, f.snapshots(), .{ .warm = true, .min_prompt = 0 }, 2 * @sizeOf(State));
    defer store.deinit();
    const host = try serial.Host.init(a, io, f.driver(&store, false));
    defer host.deinit();
    var box: Box = .{};
    const warm_request = api.Request{ .prompt = &warm, .max_tokens = 0, .background = true, .history_len = warm.len, .rewind_len = 4 };
    try host.engine().submit(1, &warm_request, .{ .ctx = &box, .event = Box.event });
    try box.wait();
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
    box = .{};
    const edited_request = api.Request{ .prompt = &edited, .max_tokens = 1, .history_len = 10, .rewind_len = 4 };
    try host.engine().submit(2, &edited_request, .{ .ctx = &box, .event = Box.event });
    try box.wait();
    try std.testing.expectEqual(@as(u32, 4), box.from);
    var cold: Fixture = .{ .hash_draw = true };
    const cold_host = try serial.Host.init(a, io, cold.driver(null, false));
    defer cold_host.deinit();
    var cold_box: Box = .{};
    try cold_host.engine().submit(3, &edited_request, .{ .ctx = &cold_box, .event = Box.event });
    try cold_box.wait();
    try std.testing.expectEqualDeep(cold.state, f.state);
    try std.testing.expectEqualSlices(u32, cold_box.tokens[0..cold_box.count], box.tokens[0..box.count]);
}

test "cache rejects a changed arithmetic map for identical prefix tokens" {
    const tokens = seedTokens();
    var f: Fixture = .{};
    var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    const host = try serial.Host.init(a, io, f.driver(&store, true));
    defer host.deinit();
    _ = try request(host, tokens[0..8], &.{.{ 2, 6 }}, 0, 0);
    const result = try request(host, tokens[0..12], &.{}, 0, 0);
    try std.testing.expectEqual(@as(u32, 0), result.from);
    try std.testing.expectEqualDeep(try fresh(tokens[0..12], &.{}, true), f.state);
}

test "canonical cache modes retain identical prefixes and ignore suffix changes" {
    const tokens = seedTokens();
    const cases = [_]struct { saved: []const [2]u32, current: []const [2]u32 }{
        .{ .saved = &.{.{ 1, 6 }}, .current = &.{ .{ 1, 6 }, .{ 9, 11 } } },
        .{ .saved = &.{.{ 1, 20 }}, .current = &.{ .{ 1, 10 }, .{ 25, 30 } } },
    };
    for (cases) |case| {
        var f: Fixture = .{};
        var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        defer store.deinit();
        const host = try serial.Host.init(a, io, f.driver(&store, true));
        defer host.deinit();
        _ = try request(host, tokens[0..8], case.saved, 0, 0);
        const result = try request(host, tokens[0..12], case.current, 0, 0);
        try std.testing.expectEqual(@as(u32, 8), result.from);
        try std.testing.expectEqualDeep(try fresh(tokens[0..12], case.current, true), f.state);
    }
}

test "a changed later map can retain a shorter matching cache mark" {
    const tokens = seedTokens();
    var f: Fixture = .{};
    var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    const host = try serial.Host.init(a, io, f.driver(&store, true));
    defer host.deinit();
    _ = try request(host, tokens[0..12], &.{.{ 8, 11 }}, 5, 0);
    const result = try request(host, tokens[0..16], &.{}, 0, 0);
    try std.testing.expectEqual(@as(u32, 5), result.from);
    try std.testing.expectEqualDeep(try fresh(tokens[0..16], &.{}, true), f.state);
}

test "mode identity stops at consumed rows and excludes lookahead" {
    const tokens = seedTokens();
    var f: Fixture = .{};
    var store = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    const host = try serial.Host.init(a, io, f.driver(&store, true));
    defer host.deinit();
    _ = try request(host, tokens[0..8], &.{.{ 1, 3 }}, 5, 0);
    const current = [_][2]u32{ .{ 1, 3 }, .{ 5, 7 } };
    const result = try request(host, tokens[0..12], &current, 0, 0);
    try std.testing.expectEqual(@as(u32, 5), result.from);
    try std.testing.expectEqualDeep(try fresh(tokens[0..12], &current, true), f.state);
}

test "drivers without a decoded callback reuse despite inert span metadata" {
    const tokens = seedTokens();
    var f: Fixture = .{};
    var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    const host = try serial.Host.init(a, io, f.driver(&store, false));
    defer host.deinit();
    _ = try request(host, tokens[0..8], &.{.{ 2, 6 }}, 0, 0);
    const result = try request(host, tokens[0..12], &.{.{ 0, 12 }}, 0, 0);
    try std.testing.expectEqual(@as(u32, 8), result.from);
    try std.testing.expectEqualDeep(try fresh(tokens[0..12], &.{}, false), f.state);
}

test "finished reply cache preserves prompt modes and excludes the pending token" {
    for ([_]bool{ false, true }) |drafted| for ([_]u32{ 1, 3 }) |count| {
        var tokens = seedTokens();
        var f: Fixture = .{};
        var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        defer store.deinit();
        const host = try serial.Host.init(a, io, f.driver(&store, true));
        defer host.deinit();
        const reply = try requestImpl(host, tokens[0..20], &.{.{ 2, 5 }}, 0, count, drafted);
        try std.testing.expectEqual(@as(usize, count), reply.count);
        @memcpy(tokens[20..][0..count], reply.tokens[0..count]);
        const next_spans = [_][2]u32{ .{ 2, 5 }, .{ 20, 20 + count } };
        const result = try request(host, tokens[0 .. 22 + count], &next_spans, 0, 0);
        try std.testing.expectEqual(@as(u32, 19 + count), result.from);
        try std.testing.expectEqualDeep(try fresh(tokens[0 .. 22 + count], &next_spans, true), f.state);
    };
}

test "finished stop replies resume at the engine position including a fed stop row" {
    for ([_]bool{ false, true }) |fed_stop| for ([_]u32{ 1, 3 }) |count| {
        var tokens = seedTokens();
        var f: Fixture = .{ .feed_stop = fed_stop };
        var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        defer store.deinit();
        const host = try serial.Host.init(a, io, f.driver(&store, true));
        defer host.deinit();
        const stop = 1000 + 20 + count - 1;
        const reply = try requestWithStop(host, tokens[0..20], &.{.{ 2, 5 }}, 0, count + 3, fed_stop, &.{stop}, .stop);
        try std.testing.expectEqual(@as(usize, count), reply.count);
        try std.testing.expectEqual(stop, reply.tokens[count - 1]);
        @memcpy(tokens[20..][0..count], reply.tokens[0..count]);
        const end = 20 + count;
        const stands = end - @as(u32, @intFromBool(!fed_stop));
        try std.testing.expectEqual(stands, f.state.at);
        const next_spans = [_][2]u32{ .{ 2, 5 }, .{ 20, end } };
        const result = try request(host, tokens[0 .. end + 2], &next_spans, 0, 0);
        try std.testing.expectEqual(stands, result.from);
        try std.testing.expectEqualDeep(try fresh(tokens[0 .. end + 2], &next_spans, true), f.state);
    };
}

test "invalid request spans fail before serial queue or cache mutation" {
    const tokens = seedTokens();
    const invalid = [_][]const [2]u32{ &.{.{ 2, 2 }}, &.{.{ 3, 2 }}, &.{ .{ 1, 4 }, .{ 3, 5 } }, &.{ .{ 1, 4 }, .{ 4, 5 } }, &.{ .{ 7, 8 }, .{ 1, 3 } } };
    for ([_]bool{ false, true }) |callback| {
        var f: Fixture = .{};
        var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        defer store.deinit();
        const host = try serial.Host.init(a, io, f.driver(&store, callback));
        defer host.deinit();
        const engine = host.engine();
        for (invalid) |spans| for ([_]bool{ false, true }) |direct| {
            var box: Box = .{};
            const r = api.Request{ .prompt = tokens[0..8], .max_tokens = 0, .decode_spans = spans };
            const sink = api.Sink{ .ctx = &box, .event = Box.event };
            try std.testing.expectError(error.InvalidSpans, if (direct) engine.vtable.submit(engine.ctx, 1, &r, sink) else engine.submit(1, &r, sink));
            try std.testing.expect(box.done == null and box.count == 0);
            var status: api.Status = undefined;
            engine.status(&status, &.{});
            try std.testing.expectEqual(@as(u32, 0), status.waiting);
            try std.testing.expectEqual(@as(u32, 0), status.running);
            try std.testing.expectEqual(@as(usize, 0), f.callbacks);
            try std.testing.expectEqual(@as(usize, 0), f.resets);
            try std.testing.expectEqual(@as(u64, 0), store.counts.misses);
            try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
        };
        _ = try request(host, tokens[0..8], &.{}, 0, 0);
        try std.testing.expectEqualDeep(try fresh(tokens[0..8], &.{}, callback), f.state);
    }
}

test "reply modes coalesce a decoded prompt suffix after clipping future rows" {
    for ([_]u32{ 1, 3 }) |count| {
        var tokens = seedTokens();
        var f: Fixture = .{};
        var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        defer store.deinit();
        const host = try serial.Host.init(a, io, f.driver(&store, true));
        defer host.deinit();
        const reply = try requestImpl(host, tokens[0..20], &.{.{ 2, 30 }}, 0, count, false);
        @memcpy(tokens[20..][0..count], reply.tokens[0..count]);
        const next_spans = [_][2]u32{.{ 2, 20 + count }};
        const result = try request(host, tokens[0 .. 22 + count], &next_spans, 0, 0);
        try std.testing.expectEqual(@as(u32, 19 + count), result.from);
        try std.testing.expectEqualDeep(try fresh(tokens[0 .. 22 + count], &next_spans, true), f.state);
    }
}
