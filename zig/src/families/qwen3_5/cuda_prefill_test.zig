//! The real CUDA prompt cancellation hook needs no device to name the last completed prefix.
const std = @import("std");
const lanes = @import("lanes");
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
    const prefill = @import("cuda_prefill.zig");
    const reuse = @import("cuda_reuse.zig");
    const Cache = struct { untaught: bool = false, delta: i32 = 0, len: usize = 0, written: usize = 0 };
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
        rows: std.ArrayList(usize) = .empty,
        pub fn forward(e: *Engine, parts: anytype, ids: []const u32, _: anytype) !void {
            try t.expectEqual(parts[0].rows, ids.len);
            try e.rows.append(t.allocator, ids.len);
        }
    };
    gpa: std.mem.Allocator = t.allocator,
    capacity: usize = 1 << 16,
    e: *Engine,
    cache: Cache = .{},
    vision: ?*@import("cuda_vision.zig").Vision = null,
    prefix: ?*Prefix = null,
    pages: ?*Pages = null,
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
    pub fn restoreKept(_: *Fake, _: *Cache, _: *anyopaque, _: []const reuse.Seen) !void {}
    fn deinit(f: *Fake) void {
        var it = f.seen.valueIterator();
        while (it.next()) |v| f.gpa.free(v.*);
        f.seen.deinit(f.gpa);
        f.caches.deinit(f.gpa);
        f.fills.deinit(f.gpa);
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
        const done = try Fake.prefill.step(&f, &s, cap);
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

pub fn capped() !void {
    const m = @import("cuda_engine.zig").max_rows;
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
}
