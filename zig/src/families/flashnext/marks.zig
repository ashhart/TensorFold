//! A prompt pass's marks: those a call runs past, the cache's keeps there (the last call's after the first token), and their crossing between the two Macs.
const std = @import("std");
const fz = @import("replay.zig");
const tpm = @import("tp.zig");
const engine = @import("engine.zig");
const segments = @import("../../core/segments.zig");
const Engine = engine.Engine;
const Passed = Engine.Passed;

/// Marks a prompt call can take its DeltaNet states at without ending there.
pub const MARKS = 4;

/// Each DeltaNet layer's conv windows and states at up to MARKS rows of one prompt call ([linear layer][mark]), and each mark's n-gram tail.
pub const Slots = struct {
    cs: fz.Buf,
    so: fz.Buf,
    tails: fz.Buf, // [mark][PLE_TAIL rows]: where speed-up mode's pair chunks leave each mark's n-gram tail on both Macs

    pub fn init(r: *fz.Run, linear: usize) !Slots {
        return .{ .cs = .{ .b = try r.buffer(linear * MARKS * fz.CS_ROW) }, .so = .{ .b = try r.buffer(linear * MARKS * fz.SO_ROW) }, .tails = .{ .b = try r.buffer(MARKS * fz.PLE_TAIL * fz.WIDE * 2) } };
    }
    pub fn tail_at(k: Slots, mark: usize) fz.Buf {
        return .{ .b = k.tails.b, .off = k.tails.off + mark * fz.PLE_TAIL * fz.WIDE * 2 };
    }
    pub fn cs_at(k: Slots, li: usize, mark: usize) fz.Buf {
        return .{ .b = k.cs.b, .off = k.cs.off + (li * MARKS + mark) * fz.CS_ROW };
    }
    pub fn so_at(k: Slots, li: usize, mark: usize) fz.Buf {
        return .{ .b = k.so.b, .off = k.so.off + (li * MARKS + mark) * fz.SO_ROW };
    }
};

/// A layer's index among the DeltaNet layers.
pub fn linearIndex(m: *const fz.Model, i: usize) usize {
    var li: usize = 0;
    for (m.layers[0..i]) |*L| li += @intFromBool(L.linear);
    return li;
}

/// The segment and its prompt buffers' row that a mark `at` rows into a call of `n` rows falls in (its last row before the mark).
pub fn markSeg(ps: []const *fz.Prompt, n: usize, at: usize) struct { p: *fz.Prompt, row: usize } {
    var k: usize = 0;
    while (k + 1 < ps.len and at > segments.start(n, ps.len, k + 1)) k += 1;
    return .{ .p = ps[k], .row = at - segments.start(n, ps.len, k) };
}

pub const Driver = struct {
    marks: []const u32,
    next: usize = 0, // the next mark
    late: [MARKS]Passed = undefined, // the last call's passed marks
    n_late: usize = 0,
    crossing: ?Crossing = null, // the last pair call's marks, crossing after the first token

    pub const Inside = struct { n: usize, cut: ?usize = null };
    const Crossing = struct { rows: [MARKS]u32, n: usize, rows0: usize, start: usize };

    /// Rows from `at` to the next mark (to `len`, the prompt's end, when none).
    pub fn toNext(d: *const Driver, at: usize, len: usize) usize {
        return (if (d.next < d.marks.len) d.marks[d.next] else len) - at;
    }

    /// The marks inside a call of `rows` rows from `at`, as rows of the call (up to MARKS); `cut` ends the call at the next one.
    pub fn inside(d: *const Driver, at: usize, rows: usize, out: *[MARKS]u32) Inside {
        var n: usize = 0;
        while (d.next + n < d.marks.len and d.marks[d.next + n] < at + rows) : (n += 1) {
            if (n == MARKS) return .{ .n = n, .cut = d.marks[d.next + n] - at };
            out[n] = @intCast(d.marks[d.next + n] - at);
        }
        return .{ .n = n };
    }

    /// A mark the call ran past: kept now, or after the first token when the call ends the prompt (its sources stay put till then).
    pub fn passed(d: *Driver, e: *Engine, out: engine.Out, p: Passed, last_call: bool) void {
        if (!last_call) return keep(e, p, out);
        d.late[d.n_late] = p;
        d.n_late += 1;
    }

    /// A pair call's marks cross between the Macs now, or after the first token when the call ends the prompt.
    pub fn cross(d: *Driver, e: *Engine, rows: []const u32, rows0: usize, start: usize, last_call: bool) !void {
        if (rows.len == 0) return;
        if (!last_call) return crossMarks(e.pr, e.m, e.r.tp.?, e.gpa, rows, rows0, start);
        d.crossing = .{ .rows = undefined, .n = rows.len, .rows0 = rows0, .start = start };
        @memcpy(d.crossing.?.rows[0..rows.len], rows);
    }

    /// After a call that ends at `at` with `n` marks inside it: the next mark, and a mark the call ends on kept now.
    pub fn after(d: *Driver, out: engine.Out, at: usize, n: usize) void {
        d.next += n;
        if (d.next < d.marks.len and at == d.marks[d.next]) {
            if (out.marked) |f| f(out.ctx, at);
            d.next += 1;
        }
    }

    /// After the first token: the last call's marks cross, then the cache keeps them.
    pub fn flush(d: *Driver, e: *Engine, out: engine.Out) !void {
        if (d.crossing) |c| try crossMarks(e.pr, e.m, e.r.tp.?, e.gpa, c.rows[0..c.n], c.rows0, c.start);
        for (d.late[0..d.n_late]) |p| keep(e, p, out);
    }
};

/// The n-gram history at row `row` of a call from `at`: `hist0` (the two tokens before the call) moved on by the call's tokens.
pub fn histAt(hist0: [2]i64, prompt: []const u32, at: usize, row: usize) [2]i64 {
    var hist = hist0;
    for (prompt[at + row - @min(row, 2) .. at + row]) |tok| hist = .{ hist[1], tok };
    return hist;
}

/// The prompt cache keeps the state at a mark a call ran past (snapshot.save reads it through e.passed).
fn keep(e: *Engine, p: Passed, out: engine.Out) void {
    e.passed = p;
    defer e.passed = null;
    if (out.marked) |f| f(out.ctx, p.at);
}

fn bytesOf(b: fz.Buf) [*]const u8 {
    return b.b.contents() + b.off;
}

/// Each mark of a pair call (`rows` of it; rank 0 holds `rows0`, this Mac starts at `start`): its states and tail from the Mac holding it into the other's slot.
pub fn crossMarks(p: *fz.Prompt, m: *fz.Model, tp: *fz.Tp2, gpa: std.mem.Allocator, rows: []const u32, rows0: usize, start: usize) !void {
    const mk = m.marks orelse return;
    const tail_bytes = fz.PLE_TAIL * fz.WIDE * 2;
    for (rows, 0..) |row, j| {
        tp.mark_seq += 1;
        const seq = tp.mark_seq;
        const owner: u32 = if (row <= rows0) 0 else 1;
        if (owner == tp.rank) { // this Mac's rows hold it: send it once the peer's MARK is free
            const tail = p.b.cin.b.contents()[p.b.cin.off + (row - start) * fz.WIDE * 2 ..][0..tail_bytes];
            @memcpy(mk.tails.b.contents()[mk.tail_at(j).off..][0..tail_bytes], tail);
            var ws: std.ArrayList(tpm.Write) = .empty;
            defer ws.deinit(gpa);
            for (0..36) |li| {
                try ws.append(gpa, .{ .src = bytesOf(mk.cs_at(li, j)), .len = fz.CS_ROW, .dst = tpm.MARK + li * tpm.DN_SLOT });
                try ws.append(gpa, .{ .src = bytesOf(mk.so_at(li, j)), .len = fz.SO_ROW, .dst = tpm.MARK + li * tpm.DN_SLOT + fz.CS_ROW });
            }
            try ws.append(gpa, .{ .src = tail.ptr, .len = tail_bytes, .dst = tpm.MARK + 36 * tpm.DN_SLOT });
            tp.hostWait(tpm.MARK_READY, seq);
            try tp.sendNow(ws.items, tpm.MARK_FLAG, seq);
        } else { // the peer's: free MARK, wait for the bytes, copy them into this mark's slot
            try tp.markReady(seq);
            tp.hostWait(tpm.MARK_FLAG, seq);
            const w = tp.window();
            const cb = p.r.queue.commandBuffer();
            p.r.enc = cb.compute(if (p.r.serial) .serial else .concurrent);
            for (0..36) |li| {
                p.copyWords(.{ .b = w, .off = tpm.MARK + li * tpm.DN_SLOT }, mk.cs_at(li, j), fz.CS_ROW / 4);
                p.copyWords(.{ .b = w, .off = tpm.MARK + li * tpm.DN_SLOT + fz.CS_ROW }, mk.so_at(li, j), fz.SO_ROW / 4);
            }
            p.copyWords(.{ .b = w, .off = tpm.MARK + 36 * tpm.DN_SLOT }, mk.tail_at(j), tail_bytes / 4);
            try m.finish(cb);
        }
    }
}

test "a call's marks: those inside it, the cut past MARKS, and the history at a mark" {
    const marks = [_]u32{ 10, 20, 30, 40, 50, 60 };
    var d: Driver = .{ .marks = &marks };
    var rows: [MARKS]u32 = undefined;
    const a = d.inside(5, 30, &rows);
    try std.testing.expectEqual(@as(usize, 3), a.n);
    try std.testing.expectEqual(@as(?usize, null), a.cut);
    try std.testing.expectEqualSlices(u32, &.{ 5, 15, 25 }, rows[0..3]);
    const b = d.inside(5, 100, &rows);
    try std.testing.expectEqual(@as(usize, 4), b.n);
    try std.testing.expectEqual(@as(?usize, 45), b.cut);
    try std.testing.expectEqual(@as(usize, 5), d.toNext(5, 999));
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try std.testing.expectEqual([2]i64{ 3, 4 }, histAt(.{ 8, 9 }, &prompt, 2, 2));
    try std.testing.expectEqual([2]i64{ 9, 3 }, histAt(.{ 8, 9 }, &prompt, 2, 1));
}
