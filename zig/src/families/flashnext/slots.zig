//! Sized shared sessions (FLASHNEXT_SLOT_CONTEXTS): the list, which session a stream takes, and when it must grow.
const std = @import("std");
const lanes = @import("lanes");
const fz = @import("replay.zig");

/// Rows a window past a stream's last committed token can take.
pub const WINDOW = fz.MAXR + 1;

/// Parses up to most extra sessions; CONTEXTxN repeats a size, each capped at capacity.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, capacity: usize, margin: usize, most: usize) ![]usize {
    var caps: std.ArrayList(usize) = .empty;
    errdefer caps.deinit(gpa);
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |entry| {
        const e = std.mem.trim(u8, entry, " \t");
        const x = std.mem.indexOfScalar(u8, e, 'x');
        const tokens = std.fmt.parseInt(usize, if (x) |i| e[0..i] else e, 10) catch return error.BadSlotContexts;
        const n = if (x) |i| std.fmt.parseInt(usize, e[i + 1 ..], 10) catch return error.BadSlotContexts else 1;
        if (tokens < 1024 or n == 0 or n > most - caps.items.len) return error.BadSlotContexts;
        const rows = std.math.add(usize, tokens, margin) catch return error.BadSlotContexts;
        try caps.appendNTimes(gpa, @min(capacity, rows), n);
    }
    if (caps.items.len == 0) return error.BadSlotContexts;
    return try caps.toOwnedSlice(gpa);
}

/// Chooses first-free when unsized, else the smallest free session holding the prompt and grow tokens.
pub fn pick(caps: []const usize, used: []const bool, sized: bool, prompt: usize, max_new: usize, grow: usize) ?usize {
    const need = prompt + @min(max_new, grow) + WINDOW;
    var best: ?usize = null;
    for (caps, used, 0..) |c, u, j| {
        if (u) continue;
        if (!sized) return j;
        if (c >= need and (best == null or c < caps[best.?])) best = j;
    }
    return best;
}

/// What a stream needs from its session before a round.
pub const Room = enum { ok, move, pause, full };

/// Checks the next window and reply end against this session, the largest session and free larger slots.
pub fn room(cap: usize, pos: usize, rows: usize, end: usize, most: usize, larger: bool) Room {
    if (end <= cap) return .ok; // the reply ends inside this session: it never grows
    if (end > most) return .full; // no session holds it: never wait for one
    if (pos + rows + 2 * WINDOW <= cap) return .ok; // the next round fits after this one
    if (larger) return .move;
    if (pos + rows + WINDOW <= cap) return .pause; // this round fits; then wait for a larger session to free
    return .full;
}

/// Ends a paused stream whose move failed, with an error the host reports as a failed request.
pub fn endPaused(stream: *lanes.Stream) void {
    stream.paused = false;
    stream.finished = true;
    stream.reason = .@"error";
    stream.problem = "the reply outgrew its session and could not move to a larger one";
}

/// Resumes oldest first; LaneMemory waits, other moveOut errors call failed.
pub fn resumePaused(paused: *std.ArrayList(*lanes.Stream), ctx: anytype) void {
    var k: usize = 0;
    while (k < paused.items.len) {
        const stream = paused.items[k];
        ctx.moveOut(stream) catch |err| {
            if (err == error.LaneMemory) {
                k += 1;
                continue;
            }
            _ = paused.orderedRemove(k);
            ctx.failed(stream, err);
            continue;
        };
        stream.paused = false;
        _ = paused.orderedRemove(k);
    }
}

test "the list parses counts and repeats, caps each at the window, and refuses overflow and too many sessions" {
    const gpa = std.testing.allocator;
    const caps = try parse(gpa, "131072, 4096x2", 65536, 32, 15);
    defer gpa.free(caps);
    try std.testing.expectEqualSlices(usize, &.{ 65536, 4128, 4128 }, caps);
    for ([_][]const u8{ "", "512", "4096x0", "4096xa", "18446744073709551615", "4096x16", "4096x8,4096x8", "1024x18446744073709551615" }) |bad|
        try std.testing.expectError(error.BadSlotContexts, parse(gpa, bad, 65536, 32, 15));
    const most = try parse(gpa, "4096x15", 65536, 32, 15);
    defer gpa.free(most);
    try std.testing.expectEqual(@as(usize, 15), most.len);
}

test "unsized sessions go first free, as before sized sessions; sized take the smallest that holds the first grow" {
    const caps = [_]usize{ 262144, 65536, 16384 };
    const free = [_]bool{ false, false, false };
    try std.testing.expectEqual(@as(?usize, 0), pick(&caps, &free, false, 1000, 4000, 8192));
    try std.testing.expectEqual(@as(?usize, 2), pick(&caps, &free, true, 1000, 4000, 8192));
    try std.testing.expectEqual(@as(?usize, 1), pick(&caps, &free, true, 10_000, 100_000, 8192));
    const busy = [_]bool{ true, false, false };
    try std.testing.expectEqual(@as(?usize, 1), pick(&caps, &busy, false, 1000, 4000, 8192));
    try std.testing.expectEqual(@as(?usize, 1), pick(&caps, &busy, false, 1000, 100_000, 8192)); // unsized: first free, its fit checked by the caller
    const all = [_]bool{ true, true, true };
    try std.testing.expectEqual(@as(?usize, null), pick(&caps, &all, false, 1000, 4000, 8192));
    try std.testing.expectEqual(@as(?usize, null), pick(&caps, &busy, true, 100_000, 300_000, 8192)); // sized: busy is the only one that holds it
}

/// Rounds of `rows`-row windows landing `landed` tokens each to the reply's end: the first room that is not ok.
fn run(cap: usize, pos0: usize, rows: usize, landed: usize, prompt: usize, max_new: usize, most: usize, larger: bool) Room {
    const end = prompt + max_new + WINDOW;
    var pos = pos0;
    while (pos < prompt + max_new) : (pos += landed) {
        const r = room(cap, pos, rows, end, most, larger);
        if (r != .ok) return r;
        if (pos + rows > cap) return .full; // a round the session cannot take
    }
    return .ok;
}

test "near the window's end with drafts rejected, a stream in a session that holds its reply never pauses" {
    // the largest session, reply ending at its last rows: every window offers MAXR rows and one lands
    const cap = 1024 + 4096 + WINDOW;
    try std.testing.expectEqual(Room.ok, run(cap, 1024, fz.MAXR, 1, 1024, 4096, cap, false));
    try std.testing.expectEqual(Room.ok, run(cap, 1024, fz.MAXR, fz.MAXR, 1024, 4096, cap, false));
}

test "a stream that outgrows its session moves while a larger one is free, else pauses on a round that still fits" {
    const small = 4096;
    const big = 1024 + 8192 + WINDOW;
    try std.testing.expectEqual(Room.move, run(small, 1024, fz.MAXR, 1, 1024, 8192, big, true));
    try std.testing.expectEqual(Room.pause, run(small, 1024, fz.MAXR, 1, 1024, 8192, big, false));
    try std.testing.expectEqual(Room.pause, run(small, 1024, fz.MAXR, fz.MAXR, 1024, 8192, big, false));
    try std.testing.expectEqual(Room.full, room(small, 1024, fz.MAXR, 1024 + 8192 + WINDOW, small, false)); // no larger session
}

test "the last larger session frees and a paused stream's move fails: it ends with an error, LaneMemory keeps waiting" {
    const gpa = std.testing.allocator;
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{1}, .max_new = 8 });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{1}, .max_new = 8 });
    defer b.deinit(gpa);
    var c = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &.{1}, .max_new = 8 });
    defer c.deinit(gpa);
    var paused: std.ArrayList(*lanes.Stream) = .empty;
    defer paused.deinit(gpa);
    for ([_]*lanes.Stream{ &a, &b, &c }) |s| {
        s.paused = true;
        try paused.append(gpa, s);
    }
    const Mover = struct { // a's copy fails on the GPU, b finds no larger session free, c moves
        a: *lanes.Stream,
        b: *lanes.Stream,
        ended: ?anyerror = null,
        fn moveOut(m: *@This(), s: *lanes.Stream) !void {
            if (s == m.a) return error.GpuFailed;
            if (s == m.b) return error.LaneMemory;
        }
        fn failed(m: *@This(), s: *lanes.Stream, err: anyerror) void {
            m.ended = err;
            endPaused(s);
        }
    };
    var mover: Mover = .{ .a = &a, .b = &b };
    resumePaused(&paused, &mover);
    try std.testing.expectEqual(@as(?anyerror, error.GpuFailed), mover.ended);
    try std.testing.expect(a.finished and !a.paused and a.reason == .@"error" and a.problem.len > 0);
    try std.testing.expect(b.paused and !b.finished);
    try std.testing.expect(!c.paused and !c.finished);
    try std.testing.expectEqual(@as(usize, 1), paused.items.len);
    try std.testing.expectEqual(&b, paused.items[0]);
}
