//! GPU check: batched MTP drafting leaves each stream as the serial path does (held tokens, ring, state), case by case.
const std = @import("std");
const lanes = @import("lanes");
const mtp = @import("cuda_mtp.zig");

const Mtp = mtp.Mtp;
const Ctx = mtp.Ctx;
const dr = lanes.drafter;

/// Streams of one side: the serial set or the batched one, each its own contexts on the same drafter.
const Side = struct {
    streams: []lanes.Stream,
    ctxs: []*Ctx,
};

/// The two sides' contexts now: held tokens, candidates, ring slots, rings and newest states byte for byte.
fn same(gpa: std.mem.Allocator, d: *Mtp, a: Side, b: Side, n: usize) !bool {
    const ring = d.ringBytes();
    const ra = try gpa.alloc(u8, ring + d.e.g.hidden * 2);
    defer gpa.free(ra);
    const rb = try gpa.alloc(u8, ring + d.e.g.hidden * 2);
    defer gpa.free(rb);
    for (a.ctxs[0..n], b.ctxs[0..n]) |x, y| {
        if (x.held_n != y.held_n or !std.mem.eql(u32, x.held[0..x.held_n], y.held[0..y.held_n])) return false;
        if (!std.mem.eql(u32, &x.cands, &y.cands) or x.begin != y.begin or x.end != y.end) return false;
        try d.e.stream.synchronize();
        try x.ring.download(0, ra[0..ring]);
        try x.last.download(0, ra[ring..]);
        try y.ring.download(0, rb[0..ring]);
        try y.last.download(0, rb[ring..]);
        if (!std.mem.eql(u8, ra, rb)) return false;
    }
    return true;
}

/// `n` streams a side, their contexts opened and zeroed (a context's warm-up leaves scratch-dependent bytes).
fn open(gpa: std.mem.Allocator, d: *Mtp, n: usize, tag: u8) !Side {
    const side: Side = .{ .streams = try gpa.alloc(lanes.Stream, n), .ctxs = try gpa.alloc(*Ctx, n) };
    for (side.streams, side.ctxs, 0..) |*s, *c, i| {
        var id: [16]u8 = undefined;
        s.* = try lanes.Stream.init(gpa, .{ .id = try std.fmt.bufPrint(&id, "{c}{d}", .{ tag, i }), .prompt = &.{1}, .max_new = 8 });
        try d.drafter().open(s);
        c.* = d.streams.get(s).?;
        try c.*.ring.fill8(0, null);
        try c.*.last.fill8(0, null);
        try c.*.step.fill8(0, null);
    }
    return side;
}

fn close(gpa: std.mem.Allocator, d: *Mtp, side: Side) void {
    for (side.streams) |*s| {
        d.drafter().release(s);
        s.deinit(gpa);
    }
    gpa.free(side.streams);
    gpa.free(side.ctxs);
}

/// An absorb of `n` rows at `start` of the target's tap ring (`rows`: picked ones), followed by the ids after them.
fn absorbItem(s: *lanes.Stream, tap: u64, start: u64, n: u32, rows: ?[]const u32, ids: []const u32, prompt: bool) dr.Absorb {
    return .{ .stream = s, .features = .{ .buffer = tap, .offset = start, .rows = n, .row_bytes = 0, .space = .device, .dtype = .bf16 }, .start = start, .follow = ids[start + 1 ..], .rows = rows, .prompt = prompt };
}

/// The same absorbs on both sides, serial then batched, compared; then the same holds, compared again.
fn step(gpa: std.mem.Allocator, d: *Mtp, a: Side, b: Side, n: usize, absorbs: []const dr.Absorb, depths: []const u32) !bool {
    inline for (.{ a, b }, .{ false, true }) |side, batch| {
        d.batch = batch;
        const before = d.batch_passes;
        const items = try gpa.alloc(dr.Absorb, absorbs.len);
        defer gpa.free(items);
        for (items, absorbs) |*x, y| {
            x.* = y;
            x.stream = &side.streams[streamIndex(a, y.stream)];
        }
        try d.drafter().absorb(items);
        if ((d.batch_passes > before) != batch) return error.WrongDraftPath;
    }
    if (!try same(gpa, d, a, b, n)) return false;
    inline for (.{ a, b }, .{ false, true }) |side, batch| {
        d.batch = batch;
        const before = d.batch_passes;
        const holds = try gpa.alloc(dr.Hold, depths.len);
        defer gpa.free(holds);
        for (holds, depths, 0..) |*h, depth, i| h.* = .{ .stream = &side.streams[i], .pending = 0, .position = side.ctxs[i].end + 1, .depth = depth };
        try d.drafter().hold(holds);
        var deep = false;
        for (depths) |x| deep = deep or x > 1;
        if (deep and (d.batch_passes > before) != batch) return error.WrongDraftPath;
    }
    return same(gpa, d, a, b, n);
}

fn streamIndex(a: Side, s: *lanes.Stream) usize {
    for (a.streams, 0..) |*x, i| if (x == s) return i;
    unreachable;
}

/// The cases on the target's tap ring `tap` (filled by a prompt over `ids`); prints a line each.
pub fn run(gpa: std.mem.Allocator, d: *Mtp, tap: u64, ids: []const u32) !bool {
    const trees = d.trees;
    const batch = d.batch;
    defer {
        d.trees = trees;
        d.batch = batch;
    }
    d.trees = false;
    var ok = true;
    {
        const a = try open(gpa, d, 6, 'a');
        defer close(gpa, d, a);
        const b = try open(gpa, d, 6, 'b');
        defer close(gpa, d, b);
        const lens = [_]u32{ 1, 2, 3, 4, 1, 7 };
        var first: [6]dr.Absorb = undefined;
        for (&first, lens, 0..) |*x, n, i| x.* = absorbItem(&a.streams[i], tap, 10 * i + 3, n, null, ids, i == 0);
        const r1 = try step(gpa, d, a, b, 6, &first, &.{ 3, 1, 0, 3, 2, 3 });
        std.debug.print("{s} batched mtp: six streams, absorbs of 1 to 7 rows (one a prompt row), depths 0 to 3\n", .{if (r1) "PASS" else "FAIL"});
        // a second round: picked rows [0, 2] of a 3-row window at each end, and a gap past one stream's end
        const picks = [_]u32{ 0, 2 };
        var second: [6]dr.Absorb = undefined;
        for (&second, 0..) |*x, i| {
            const end = a.ctxs[i].end;
            x.* = if (i == 2) absorbItem(&a.streams[i], tap, end + 5, 2, null, ids, false) else absorbItem(&a.streams[i], tap, end, 2, &picks, ids, false);
        }
        const r2 = try step(gpa, d, a, b, 6, &second, &.{ 3, 3, 3, 3, 3, 3 });
        std.debug.print("{s} batched mtp: partial acceptance (picked rows), a gap past a stream's end, depth 3\n", .{if (r2) "PASS" else "FAIL"});
        ok = ok and r1 and r2;
    }
    {
        // contexts released to the pool and reopened: a reused context drafts as a fresh one
        const a = try open(gpa, d, 3, 'c');
        defer close(gpa, d, a);
        const b = try open(gpa, d, 3, 'd');
        defer close(gpa, d, b);
        var items: [3]dr.Absorb = undefined;
        for (&items, 0..) |*x, i| x.* = absorbItem(&a.streams[i], tap, 40 + 2 * i, 3, null, ids, false);
        const r = try step(gpa, d, a, b, 3, &items, &.{ 3, 2, 3 });
        std.debug.print("{s} batched mtp: reused contexts from the pool\n", .{if (r) "PASS" else "FAIL"});
        ok = ok and r;
    }
    {
        const a = try open(gpa, d, 17, 'e');
        defer close(gpa, d, a);
        const b = try open(gpa, d, 17, 'f');
        defer close(gpa, d, b);
        var items: [17]dr.Absorb = undefined;
        for (&items, 0..) |*x, i| x.* = absorbItem(&a.streams[i], tap, 5 + i, 1, null, ids, false);
        const depths: [17]u32 = @splat(3);
        const r = try step(gpa, d, a, b, 17, &items, &depths);
        std.debug.print("{s} batched mtp: 17 streams, two passes a draft level\n", .{if (r) "PASS" else "FAIL"});
        ok = ok and r;
    }
    return ok;
}
