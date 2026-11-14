//! The real CUDA prompt cancellation hook needs no device to name the last completed prefix.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const prefill = @import("cuda_prefill.zig");
const readout = @import("cuda_readout.zig");
const t = std.testing;

pub fn cancelled(comptime Lanes: type, backend: anytype) !void {
    const Probe = struct {
        calls: usize = 0,
        at: u32 = 0,
        fn kept(ctx: *anyopaque, _: *lanes.Stream, at: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            p.calls += 1;
            p.at = at;
        }
    };
    var state: Lanes = .{ .gpa = t.allocator, .io = t.io, .e = undefined, .capacity = 32, .target = undefined };
    defer state.fills.deinit(t.allocator);
    var probe: Probe = .{};
    var s = try lanes.Stream.init(t.allocator, .{ .id = "prefix", .prompt = &(@as([16]u32, @splat(3))), .max_new = 1, .reuse = .{ .hook = .{ .ptr = &probe, .at = Probe.kept } } });
    defer s.deinit(t.allocator);
    const b = backend(&state);
    const cancel = b.vtable.cancel_prefill orelse return error.MissingCancelHook;
    try cancel(b.ptr, &s);
    try t.expectEqual(@as(usize, 0), probe.calls);
    s.cached = 4;
    s.fill_at.store(12, .monotonic);
    try state.fills.put(t.allocator, &s, .{ .at = 8, .media = null });
    try cancel(b.ptr, &s);
    try t.expectEqual(@as(usize, 1), probe.calls);
    try t.expectEqual(@as(u32, 8), probe.at);
    try t.expect(state.fills.contains(&s)); // the host still owns the fill until its snapshot and release complete
    for ([_]usize{ 0, 4, 16 }) |at| {
        state.fills.getPtr(&s).?.at = at;
        try cancel(b.ptr, &s);
    }
    try t.expectEqual(@as(usize, 1), probe.calls);
}

/// Lanes' parts a stepped pass touches, with a forward that records each cut's rows and nothing on a device.
const Fake = struct {
    const reuse = @import("cuda_reuse.zig");
    const Cache = struct {
        untaught: bool = false,
        delta: i32 = 0,
        len: usize = 0,
        written: usize = 0,
        pub fn reset(_: *Cache, e: *Engine) !void {
            if (e.reset_fails) return error.FakeReset;
        }
    };
    const Plan = struct { from: usize, marks: []u32 };
    const Prefix = struct {
        pub fn open(_: *Prefix, _: []const u32, _: usize, _: u32, _: []const u32) !?struct { cache: *Cache, plan: Plan } {
            return null;
        }
        pub fn keep(_: *Prefix, _: *Cache, _: []const u32, _: usize) error{QwenLaneTable}!void {}
    };
    const Pages = struct {
        pub fn log(_: *Pages, _: usize, _: usize) void {}
    };
    const Engine = struct {
        prompt_rows: usize = E.grid_rows,
        rows: std.ArrayList(usize) = .empty,
        x: u64 = 0,
        stream: cuda.Stream = .{ .d = undefined, .handle = null },
        whole_calls: usize = 0,
        whole_rows: usize = 0,
        piece_calls: usize = 0,
        piece_rows: usize = 0,
        reset_fails: bool = false,
        pub fn forward(e: *Engine, parts: anytype, ids: []const u32, keep: E.Keep) !void {
            var rows: usize = 0;
            inline for (parts) |p| rows += p.rows;
            try t.expectEqual(rows, ids.len);
            try e.rows.append(t.allocator, ids.len);
            if (keep == .rows) {
                e.piece_calls += 1;
                e.piece_rows += rows;
            } else {
                e.whole_calls += 1;
                e.whole_rows += rows;
            }
        }
    };
    gpa: std.mem.Allocator = t.allocator,
    capacity: usize = 1 << 16,
    e: *Engine,
    cache: Cache = .{},
    vision: ?*@import("cuda_vision.zig").Vision = null,
    prefix: ?*Prefix = null,
    pages: ?*Pages = null,
    readout: readout.Side = .{},
    restore_ok: bool = true,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *Cache) = .empty,
    seen: std.AutoHashMapUnmanaged(*lanes.Stream, []reuse.Seen) = .empty,
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, prefill.Fill) = .empty,
    pub fn seenOf(f: *Fake, _: anytype) ![]reuse.Seen {
        return f.gpa.alloc(reuse.Seen, 0);
    }
    pub fn take(f: *Fake, _: usize, _: usize) !*Cache {
        return &f.cache;
    }
    pub fn give(_: *Fake, _: *Cache) void {}
    pub fn cacheOf(f: *Fake, s: *lanes.Stream) !*Cache {
        return f.caches.get(s) orelse error.UnknownQwenStream;
    }
    pub fn restoreKept(f: *Fake, _: *Cache, _: *anyopaque, _: []const reuse.Seen) !void {
        if (!f.restore_ok) return error.FakeRestore;
    }
    fn deinit(f: *Fake) void {
        var it = f.seen.valueIterator();
        while (it.next()) |v| f.gpa.free(v.*);
        f.seen.deinit(f.gpa);
        f.caches.deinit(f.gpa);
        f.fills.deinit(f.gpa);
        f.readout.open.deinit(f.gpa);
    }
};

/// One stepped pass at the caps in turn, a forward a step of at most its cap's rows: where each forward ended.
fn pass(prompt: []const u32, chunks: []const u32, caps: []const u32, ends: *std.ArrayList(usize)) !void {
    var e: Fake.Engine = .{};
    defer e.rows.deinit(t.allocator);
    var f: Fake = .{ .e = &e };
    defer f.deinit();
    var s = try lanes.Stream.init(t.allocator, .{ .id = "cap", .prompt = prompt, .max_new = 1, .chunks = chunks });
    defer s.deinit(t.allocator);
    var i: usize = 0;
    while (true) : (i += 1) {
        const cap = caps[i % caps.len];
        const done = try prefill.step(&f, &s, cap);
        try t.expectEqual(i + 1, e.rows.items.len);
        if (cap > 0) try t.expect(e.rows.items[i] <= @max(128, cap));
        if (done) break;
    }
    var at: usize = 0;
    for (e.rows.items) |rows| {
        at += rows;
        try ends.append(t.allocator, at);
    }
}

pub fn capped(comptime Lanes: type, backend: anytype) !void {
    var state: Lanes = .{ .gpa = t.allocator, .io = t.io, .e = undefined, .capacity = 32, .target = undefined };
    try t.expect(backend(&state).chunk_invariant); // the lane host offers its cap only to a family that declares it
    const m = @import("cuda_engine.zig").grid_rows;
    const prompt = try t.allocator.alloc(u32, 3 * m);
    defer t.allocator.free(prompt);
    @memset(prompt, 3);
    const start: u32 = m + 1000; // a request chunk start: a cold cut off the max_rows grid
    var cold: std.ArrayList(usize) = .empty;
    defer cold.deinit(t.allocator);
    try pass(prompt, &.{start}, &.{0}, &cold);
    try t.expectEqualSlices(usize, &.{ m, start, start + m, prompt.len }, cold.items);
    for ([_][]const u32{ &.{512}, &.{ 512, 0, 128 }, &.{ 1000, 4096 }, &.{64} }) |caps| {
        var ends: std.ArrayList(usize) = .empty;
        defer ends.deinit(t.allocator);
        try pass(prompt, &.{start}, caps, &ends);
        try t.expect(ends.items.len > cold.items.len); // the cap reached the pass
        var from: usize = 0;
        var landed: usize = 0;
        for (ends.items) |end| {
            const cold_cut = std.mem.indexOfScalar(usize, cold.items, end) != null;
            if (cold_cut) landed += 1 else try t.expect((end - from) % 128 == 0); // inside a cold chunk: on the grid
            from = end;
        }
        try t.expectEqual(cold.items.len, landed);
    }
    try marks();
    try policy();
}

/// A store mark on the 128 grid inside a cold chunk and one on a cold cut: a capped pass keeps only the cold one.
fn marks() !void {
    const Probe = struct {
        at: [8]u32 = undefined,
        n: usize = 0,
        fn kept(ctx: *anyopaque, _: *lanes.Stream, at: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            p.at[p.n] = at;
            p.n += 1;
        }
    };
    const m = @import("cuda_engine.zig").grid_rows;
    const prompt = try t.allocator.alloc(u32, 2 * m + 100);
    defer t.allocator.free(prompt);
    @memset(prompt, 3);
    for ([_]u32{ 0, 512 }) |cap| {
        var e: Fake.Engine = .{};
        defer e.rows.deinit(t.allocator);
        var f: Fake = .{ .e = &e };
        defer f.deinit();
        var probe: Probe = .{};
        const want = [_]u32{ 1024, m }; // 1024: a capped step ends there, but no pass resumes inside a cold chunk
        var s = try lanes.Stream.init(t.allocator, .{
            .id = "marks",
            .prompt = prompt,
            .max_new = 1,
            .reuse = .{ .hook = .{ .ptr = &probe, .at = Probe.kept }, .marks = &want },
        });
        defer s.deinit(t.allocator);
        while (!try prefill.step(&f, &s, cap)) {}
        if (cap > 0) try t.expect(std.mem.indexOfScalar(usize, e.rows.items, 512) != null); // the cap cut at 1024 too
        try t.expectEqualSlices(u32, &.{m}, probe.at[0..probe.n]);
    }
}

/// fcfs's TOKENS rows reach any family beside a decoder (capRows): Qwen cuts at TOKENS floored to 128 in cold chunks.
fn policy() !void {
    const m = @import("cuda_engine.zig").grid_rows;
    const prompt = try t.allocator.alloc(u32, 2 * m + 100);
    defer t.allocator.free(prompt);
    @memset(prompt, 3);
    var ends: std.ArrayList(usize) = .empty;
    defer ends.deinit(t.allocator);
    try pass(prompt, &.{}, &.{1000}, &ends); // TOKENS=1000: 896-row steps
    const want = [_]usize{ 896, 1792, 2688, 3584, m, m + 896, m + 1792, m + 2688, m + 3584, 2 * m, 2 * m + 100 };
    try t.expectEqualSlices(usize, &want, ends.items);
}

const ReadoutProbe = struct {
    opened: usize = 0,
    pieces: usize = 0,
    rows: usize = 0,

    fn open(ctx: *anyopaque, _: *const lanes.readout.Spec, _: []const u32) anyerror!*readout.Readout.State {
        const p: *@This() = @ptrCast(@alignCast(ctx));
        p.opened += 1;
        return @ptrCast(@alignCast(ctx));
    }

    fn consume(ctx: *anyopaque, _: *readout.Readout.State, _: u64, _: usize, count: usize, _: cuda.Stream) anyerror!void {
        const p: *@This() = @ptrCast(@alignCast(ctx));
        p.pieces += 1;
        p.rows += count;
    }

    fn finish(_: *anyopaque, _: *readout.Readout.State, _: []u16) anyerror!void {}

    fn close(_: *anyopaque, _: *readout.Readout.State) void {}
};

/// A text stream runs its whole pass and hits its mark; a readout stream runs its pieces and no whole pass.
pub fn whole() !void {
    const Hook = struct {
        calls: usize = 0,
        at: u32 = 0,
        fn kept(ctx: *anyopaque, _: *lanes.Stream, mark: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            p.calls += 1;
            p.at = mark;
        }
    };
    var eng: Fake.Engine = .{};
    defer eng.rows.deinit(t.allocator);
    var toks: [E.grid_rows + 8]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(i % 7);
    var fake: Fake = .{ .e = &eng, .capacity = toks.len + 1 };
    defer fake.deinit();
    const keep = [_]u32{E.grid_rows};
    var hook: Hook = .{};
    var text = try lanes.Stream.init(t.allocator, .{ .id = "text", .prompt = &toks, .max_new = 1, .reuse = .{ .marks = &keep, .hook = .{ .ptr = &hook, .at = Hook.kept } } });
    defer text.deinit(t.allocator);
    try prefill.whole(&fake, &text);
    try t.expectEqual(@as(usize, 2), fake.e.whole_calls);
    try t.expectEqual(toks.len, fake.e.whole_rows);
    try t.expectEqual(@as(usize, 0), fake.e.piece_calls);
    try t.expectEqual(@as(usize, 1), hook.calls);
    try t.expectEqual(@as(u32, E.grid_rows), hook.at);
    try t.expectEqual(@as(u32, @intCast(toks.len)), text.fill_at.load(.monotonic));

    const spec: lanes.readout.Spec = .{ .question_spans = &.{}, .question_types = &.{}, .option_spans = &.{}, .option_question = &.{}, .logits = &.{} };
    var rtoks: [2 * readout.piece_rows]u32 = undefined;
    for (&rtoks, 0..) |*x, i| x.* = @intCast(i % 5);
    var probe: ReadoutProbe = .{};
    const vtable: readout.Readout.VTable = .{ .open = ReadoutProbe.open, .consume = ReadoutProbe.consume, .finish = ReadoutProbe.finish, .close = ReadoutProbe.close };
    const info: lanes.readout.Info = .{ .max_rows = E.grid_rows, .max_questions = 1, .max_options = 1 };
    fake.readout.readout = .{ .ctx = &probe, .info = info, .vtable = &vtable };
    var rd = try lanes.Stream.init(t.allocator, .{ .id = "readout", .prompt = &rtoks, .max_new = 0, .drafts = false, .readout = &spec });
    defer rd.deinit(t.allocator);
    const whole_calls = fake.e.whole_calls;
    try prefill.whole(&fake, &rd);
    try t.expectEqual(whole_calls, fake.e.whole_calls);
    try t.expectEqual(@as(usize, 2), fake.e.piece_calls);
    try t.expectEqual(rtoks.len, fake.e.piece_rows);
    try t.expectEqual(@as(usize, 1), probe.opened);
    try t.expectEqual(@as(usize, 2), probe.pieces);
    try t.expectEqual(rtoks.len, probe.rows);
    try t.expectEqual(@as(usize, 1), hook.calls);
    try t.expectEqual(@as(u32, @intCast(rtoks.len)), rd.fill_at.load(.monotonic));
}

test "Qwen whole prefill: a text stream runs its pass once and a readout only its pieces" {
    try whole();
}

/// A failed restore resets the cache under `try`: a reset error leaves opening instead of filling from row 0.
pub fn reset() !void {
    var eng: Fake.Engine = .{ .reset_fails = true };
    defer eng.rows.deinit(t.allocator);
    var fake: Fake = .{ .e = &eng, .restore_ok = false };
    defer fake.deinit();
    var toks: [3 * E.grid_rows]u32 = @splat(3);
    var saved: u8 = 0;
    var s = try lanes.Stream.init(t.allocator, .{ .id = "reset", .prompt = &toks, .max_new = 1, .reuse = .{ .saved = @ptrCast(&saved), .at = @intCast(E.grid_rows) } });
    defer s.deinit(t.allocator);
    try t.expectError(error.FakeReset, prefill.step(&fake, &s, 0));
    try t.expectEqual(@as(usize, 0), eng.whole_calls + eng.piece_calls);
}

test "Qwen prefill opening: a failed reset propagates out of the opening" {
    try reset();
}
