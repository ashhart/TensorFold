//! One owned GLM prompt pass, yielding after one bounded chunk to the shared lane scheduler.
const std = @import("std");
const Stream = @import("lanes").Stream;

/// Keep decode latency bounded when a prompt shares the GPU; solo passes retain the engine's normal chunk size.
pub const decode_rows: u32 = 256;
pub const State = struct { stream: *Stream, slot: u32, at: u32 = 0, next: usize = 0, reuse_next: usize = 0 };

/// `b` is the real backend or a fake engine at the same prepare/chunk/finish boundary. `_: u32` keeps its 2048 grid.
pub fn step(b: anytype, s: *Stream, _: u32) !bool {
    errdefer {
        if (s.isCancelled()) if (b.filling.get(s)) |p| s.keepInterruptedPrefill(p.at);
        b.abortPrompt(s);
    }
    if (s.isCancelled()) return error.Cancelled;
    if (!b.filling.contains(s)) {
        const begun = try b.beginPrompt(s);
        try b.filling.put(b.gpa, s, begun);
    }
    const p = b.filling.getPtr(s).?;
    const prompt = s.prompt();
    if (p.at < prompt.len) {
        while (p.next < s.chunks.len and s.chunks[p.next] <= p.at) p.next += 1;
        while (p.reuse_next < s.reuse.marks.len and s.reuse.marks[p.reuse_next] <= p.at) p.reuse_next += 1;
        var end: u32 = @intCast(@min(if (p.next < s.chunks.len) s.chunks[p.next] else prompt.len, prompt.len));
        if (p.reuse_next < s.reuse.marks.len) end = @min(end, s.reuse.marks[p.reuse_next]);
        if (b.decoding()) end = @min(end, p.at + @min(decode_rows, @as(u32, @intCast(prompt.len)) - p.at));
        const n = b.promptRows(p.at, end);
        if (n == 0 or n > end - p.at) return error.ChunkOutOfStep;
        try b.chunkPrompt(s, p.slot, p.at, n);
        @import("lanes").prefill_widths.completed(b.gpa, s, n);
        p.at += n;
        s.fill_at.store(p.at, .monotonic);
        if (std.mem.indexOfScalar(u32, s.reuse.marks, p.at) != null) if (s.reuse.hook) |h| h.at(h.ptr, s, p.at);
    }
    if (p.at < prompt.len) return false;
    try b.finishPrompt(s);
    _ = b.filling.remove(s);
    return true;
}

test "glm prefill: cache marks bound otherwise uncut chunks" {
    const Backend = struct {
        gpa: std.mem.Allocator = std.testing.allocator,
        filling: std.AutoHashMapUnmanaged(*Stream, State) = .empty,
        history: std.ArrayList(u32) = .empty,

        fn beginPrompt(_: *@This(), s: *Stream) !State {
            return .{ .stream = s, .slot = 0 };
        }
        fn decoding(_: *const @This()) bool {
            return false;
        }
        fn promptRows(_: *const @This(), at: u32, end: u32) u32 {
            return end - at;
        }
        fn chunkPrompt(b: *@This(), s: *Stream, _: u32, at: u32, n: u32) !void {
            try b.history.appendSlice(std.testing.allocator, s.prompt()[at .. at + n]);
        }
        fn finishPrompt(_: *@This(), _: *Stream) !void {}
        fn abortPrompt(_: *@This(), _: *Stream) void {}
    };
    const Marks = struct {
        at: [2]u32 = undefined,
        n: usize = 0,
        fn hit(p: *anyopaque, _: *Stream, at: u32) void {
            const m: *@This() = @ptrCast(@alignCast(p));
            m.at[m.n] = at;
            m.n += 1;
        }
    };
    var b: Backend = .{};
    defer b.filling.deinit(std.testing.allocator);
    defer b.history.deinit(std.testing.allocator);
    const ids = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    var s = try Stream.init(std.testing.allocator, .{ .id = "marks", .prompt = &ids, .max_new = 1 });
    defer s.deinit(std.testing.allocator);
    var marks: Marks = .{};
    s.reuse = .{ .marks = &.{ 3, 7 }, .hook = .{ .ptr = &marks, .at = Marks.hit } };
    try std.testing.expect(!try step(&b, &s, 0));
    try std.testing.expectEqual(@as(u32, 3), s.fill_at.load(.monotonic));
    try std.testing.expect(!try step(&b, &s, 0));
    try std.testing.expectEqual(@as(u32, 7), s.fill_at.load(.monotonic));
    try std.testing.expect(try step(&b, &s, 0));
    try std.testing.expectEqualSlices(u32, &.{ 3, 7 }, &marks.at);
    try std.testing.expectEqualSlices(u32, &ids, b.history.items);
}
