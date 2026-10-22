//! One completed prompt call per shared fill choice, with cursor and owned resources kept across yields.
const std = @import("std");
const Stream = @import("lanes").Stream;
const segments = @import("../../core/stagger.zig");

pub const decode_rows: u32 = 32;
pub const State = struct {
    stream: *Stream,
    slot: usize,
    at: u32 = 0,
    next: usize = 0,
    reuse_next: usize = 0,
    owner: ?*anyopaque = null,
};

// Reserve first: a successful begin transfers ownership without a later map allocation that could fail.
pub fn start(b: anytype, s: *Stream, slot: ?usize) !void {
    if (b.filling.contains(s)) return error.PromptAlreadyStarted;
    if (s.isCancelled()) return error.Cancelled;
    try b.filling.ensureUnusedCapacity(b.gpa, 1);
    try b.settlePrompt(s);
    const begun = try b.beginPrompt(s, slot);
    b.filling.putAssumeCapacity(s, begun);
}

pub fn step(b: anytype, s: *Stream) !bool {
    errdefer b.abortPrompt(s);
    if (s.isCancelled()) return error.Cancelled;
    if (!b.filling.contains(s)) try start(b, s, null);
    var p = b.filling.get(s).?;
    const prompt = s.prompt();
    if (p.at > prompt.len) return error.PromptOutOfStep;
    if (p.at < prompt.len) {
        while (p.next < s.chunks.len and s.chunks[p.next] <= p.at) p.next += 1;
        while (p.reuse_next < s.reuse.marks.len and s.reuse.marks[p.reuse_next] <= p.at) p.reuse_next += 1;
        var end: u32 = @intCast(prompt.len);
        if (p.next < s.chunks.len) end = @min(end, s.chunks[p.next]);
        if (p.reuse_next < s.reuse.marks.len) end = @min(end, s.reuse.marks[p.reuse_next]);
        if (b.decoding()) end = @min(end, p.at + @min(decode_rows, end - p.at));
        const n = b.promptRows(p.at, end);
        if (n == 0 or n > end - p.at) return error.ChunkOutOfStep;
        try b.settlePrompt(s);
        try b.chunkPrompt(s, p.slot, p.at, n);
        try b.finishChunk(s);
        p.at += n;
        // No map element pointer spans a chunk, image callback or mark hook that could insert another fill.
        const entry = b.filling.getPtr(s) orelse return error.PromptOutOfStep;
        entry.* = p;
        s.fill_at.store(p.at, .monotonic);
        if (std.mem.indexOfScalar(u32, s.reuse.marks, p.at) != null) try b.markPrompt(s, p.at);
    }
    if (p.at < prompt.len) return false;
    try b.finishPrompt(s);
    _ = b.filling.remove(s);
    return true;
}

// True means the prompt completed; false hands its unfinished cursor to the shared host after a completed call.
pub fn lone(b: anytype, s: *Stream, hooks: anytype) !bool {
    try start(b, s, 0);
    while (!try step(b, s)) {
        if (hooks.prefill_yield) |yield_| if (yield_(hooks.ctx)) return false;
    }
    return true;
}

// Both actual vtable entries use this driver; host controls replace only its engine boundary.
pub fn Callbacks(comptime B: type) type {
    return struct {
        pub fn whole(ptr: *anyopaque, s: *Stream) anyerror!void {
            const b = B.self(ptr);
            const pool = b.promptPool();
            defer pool.pop();
            while (!try step(b, s)) {}
        }
        pub fn one(ptr: *anyopaque, s: *Stream) anyerror!bool {
            const b = B.self(ptr);
            const pool = b.promptPool();
            defer pool.pop();
            return step(b, s);
        }
    };
}

pub const Piece = struct { start: usize, rows: usize, head_rows: usize, final: bool };

// A nonfinal segment consumes its next-token lookahead; only the prompt's last row has no head pair.
pub fn piece(at: usize, rows: usize, parts: usize, k: usize, prompt_len: usize) !Piece {
    if (parts == 0 or parts > segments.MAX or k >= parts or rows < parts or at >= prompt_len or rows > prompt_len - at)
        return error.ChunkOutOfStep;
    const begin = at + segments.start(rows, parts, k);
    const n = segments.rows(rows, parts, k);
    return .{ .start = begin, .rows = n, .head_rows = @min(n, prompt_len - 1 - begin), .final = begin + n == prompt_len };
}

test "flashnext prefill: prompt head pairs cover P minus one across odd seams and one-row tails" {
    for ([_]usize{ 1, 2, 3, 17, 31, 32, 33, 65, 257, 1025 }) |p| {
        for ([_]usize{ 1, 2, 7, 16, 32, 256 }) |limit| {
            var at: usize = 0;
            var pairs: usize = 0;
            var final_rows: usize = 0;
            while (at < p) {
                const n = @min(limit, p - at);
                const parts: usize = if (n > 1) 2 else 1;
                for (0..parts) |k| {
                    const span = try piece(at, n, parts, k, p);
                    for (0..span.head_rows) |j| {
                        try std.testing.expectEqual(pairs, span.start + j);
                        try std.testing.expect(span.start + j + 1 < p);
                        pairs += 1;
                    }
                    if (span.final) final_rows += 1;
                }
                at += n;
            }
            try std.testing.expectEqual(p - 1, pairs);
            try std.testing.expectEqual(@as(usize, 1), final_rows);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), (try piece(0, 1, 1, 0, 1)).head_rows);
    try std.testing.expectError(error.ChunkOutOfStep, piece(1, 1, 1, 0, 1));
}
