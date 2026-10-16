//! A thinking budget on Flash Next's GPU-side rounds: once it is spent, the think close lands as a forced span.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");

/// A request's budget: reply tokens inside the think block, the last of them the close's first (0: no budget).
pub const Think = struct { budget: u32 = 0, close: []const u32 = &.{}, end: ?u32 = null };

/// The close's tokens at most.
pub const MAX_CLOSE = 16;
/// Where the span's words start in the rounds' arena, past the history's (verdict.zig reads them there).
pub const AR_SPAN = 808;

/// The budget as the host reads a reply: whether the close may still land, and at which reply token it starts.
pub const Watch = struct {
    close: []const u32 = &.{},
    end: ?u32 = null,
    cut: u64 = 0, // the reply token the close's first replaces (budget - 1)
    seen: u64 = 0, // reply tokens read so far
    armed: bool = false, // while set, every round runs fz_force and fz_span (they catch up on what landed)

    pub fn init(t: Think) !Watch {
        if (t.close.len > MAX_CLOSE) return error.ThinkCloseTooLong;
        return .{ .close = t.close, .end = t.end, .cut = t.budget -| 1, .armed = t.budget > 0 and t.close.len > 0 };
    }

    /// The reply's first token: the close's first in place of the target's when the budget is one token.
    pub fn first(w: *Watch, pick: u32) u32 {
        const t = if (w.armed and w.cut == 0) w.close[0] else pick;
        _ = w.read(&.{t});
        return t;
    }

    /// A round's landed tokens, in order; true when one of them is the close's (no draft rule learns from that round).
    pub fn read(w: *Watch, toks: []const u32) bool {
        var hit = false;
        for (toks) |t| {
            const i = w.seen;
            w.seen += 1;
            if (!w.armed) continue;
            if (i >= w.cut) {
                hit = true;
                w.armed = i + 1 < w.cut + w.close.len;
            } else if (w.end != null and t == w.end.?) w.armed = false;
        }
        return hit;
    }

    /// The span's words (verdict.zig) for the rounds after the first token at `at0`; zero once the close is done.
    pub fn arm(w: *const Watch, ar: []i32, at0: usize) void {
        const f = ar[AR_SPAN..][0 .. 4 + MAX_CLOSE];
        @memset(f, 0);
        if (!w.armed) return;
        f[0] = @intCast(at0 + w.cut);
        f[1] = if (w.end) |e| @intCast(e) else -1;
        f[2] = @intCast(w.close.len);
        f[3] = @intCast(at0); // the tokens it has seen land: the first one, read on the host
        for (w.close, f[4..][0..w.close.len]) |t, *x| x.* = @intCast(t);
    }
};

/// fz_force before a window of `rows`: the last round's tokens end the close (or cancel it), its tokens become drafts.
pub fn force(r: *fz.Run, wids: fz.Buf, hist: mtl.Buffer, rows: usize) void {
    r.enc.setPipeline(r.force_pipe);
    r.enc.setBuffer(wids.b, wids.off, 0);
    r.enc.setBuffer(r.ar.b, r.ar.off, 1);
    r.enc.setBuffer(hist, 0, 2);
    const c = [2]u32{ @intCast(rows), AR_SPAN };
    r.enc.setBytes(std.mem.asBytes(&c), 3);
    r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    if (!r.serial) r.enc.barrier();
}

/// fz_span before the verdict: the target's picks take the close's tokens at its positions.
pub fn span(r: *fz.Run, picks: fz.Buf, rows: usize) void {
    r.enc.setPipeline(r.span_pipe);
    r.enc.setBuffer(picks.b, picks.off, 0);
    r.enc.setBuffer(r.ar.b, r.ar.off, 1);
    const c = [2]u32{ @intCast(rows), AR_SPAN };
    r.enc.setBytes(std.mem.asBytes(&c), 2);
    r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    if (!r.serial) r.enc.barrier();
}

/// The budget's words in rank 0's request to rank 1 (follow.zig): budget, think end + 1 (0: none), close length, close.
pub fn words(t: Think) usize {
    return 3 + t.close.len;
}

pub fn put(t: Think, out: []u32) void {
    out[0] = t.budget;
    out[1] = if (t.end) |e| e + 1 else 0;
    out[2] = @intCast(t.close.len);
    @memcpy(out[3..][0..t.close.len], t.close);
}

/// The budget in a request's words (its close a view of them).
pub fn take(in: []const u32) !Think {
    if (in.len < 3 or in[2] > MAX_CLOSE or in.len < 3 + in[2]) return error.TpProtocol;
    return .{ .budget = in[0], .end = if (in[1] == 0) null else in[1] - 1, .close = in[3..][0..in[2]] };
}

test "the close starts at the budget's last token and lands whole; a round that holds it is marked" {
    var w = try Watch.init(.{ .budget = 4, .close = &.{ 7, 8, 9 }, .end = 8 });
    try std.testing.expectEqual(@as(u32, 100), w.first(100));
    try std.testing.expect(!w.read(&.{ 101, 102 }));
    var ar: [1024]i32 = @splat(0);
    w.arm(&ar, 50);
    try std.testing.expectEqualSlices(i32, &.{ 53, 8, 3, 50, 7, 8, 9, 0 }, ar[AR_SPAN..][0..8]);
    try std.testing.expect(w.read(&.{ 7, 8 }) and w.armed);
    try std.testing.expect(w.read(&.{ 9, 5, 6 }) and !w.armed);
    try std.testing.expect(!w.read(&.{8}));
    w.arm(&ar, 50);
    try std.testing.expectEqual(@as(i32, 0), ar[AR_SPAN]);
}

test "a think end before the budget cancels the close; a one-token budget closes from the first token" {
    var w = try Watch.init(.{ .budget = 10, .close = &.{ 7, 8, 9 }, .end = 8 });
    _ = w.first(5);
    try std.testing.expect(!w.read(&.{ 6, 8, 4 }) and !w.armed);
    var one = try Watch.init(.{ .budget = 1, .close = &.{ 7, 8, 9 }, .end = 8 });
    try std.testing.expectEqual(@as(u32, 7), one.first(8));
    try std.testing.expect(one.armed and one.read(&.{ 8, 9 }) and !one.armed);
    var lone = try Watch.init(.{ .budget = 1, .close = &.{7}, .end = null });
    try std.testing.expectEqual(@as(u32, 7), lone.first(3));
    try std.testing.expect(!lone.armed);
    const none = try Watch.init(.{});
    try std.testing.expect(!none.armed);
    var long: [MAX_CLOSE + 1]u32 = @splat(1);
    try std.testing.expectError(error.ThinkCloseTooLong, Watch.init(.{ .budget = 3, .close = &long }));
}

test "rank 1 reads the budget rank 0 wrote" {
    const t: Think = .{ .budget = 256, .close = &.{ 198, 248069, 271 }, .end = 248069 };
    var buf: [8]u32 = undefined;
    put(t, buf[0..words(t)]);
    const back = try take(buf[0..words(t)]);
    try std.testing.expectEqual(t.budget, back.budget);
    try std.testing.expectEqual(t.end, back.end);
    try std.testing.expectEqualSlices(u32, t.close, back.close);
    put(.{}, buf[0..3]);
    const none = try take(buf[0..3]);
    try std.testing.expect(none.budget == 0 and none.end == null and none.close.len == 0);
    try std.testing.expectError(error.TpProtocol, take(buf[0..2]));
    buf[2] = MAX_CLOSE + 1;
    try std.testing.expectError(error.TpProtocol, take(&buf));
}
