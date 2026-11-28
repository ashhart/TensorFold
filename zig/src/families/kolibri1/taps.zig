//! Kolibri-1's tapped rows for a drafter: each forward's copies kept per stream in a ring, handed out as Features.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("forward.zig");

/// Positions a stream's ring keeps: the last verify's rows and its prompt's last, with room to spare.
pub const keep = 256;
/// Rows a ring view hands out at most; ring rows below it are mirrored past `keep`, so every view is contiguous.
pub const mirror = 64;
/// Layers a drafter may tap.
pub const most = 8;

/// A copy of `rows` rows: forward row `src` on into ring row `dst` on.
pub const Piece = struct { src: usize, dst: usize, rows: usize };

/// Positions [p0, p0 + n), n <= keep, into ring rows (p % keep), each row below `mirror` again at keep + row.
pub fn pieces(p0: u64, n: usize, out: *[4]Piece) []const Piece {
    std.debug.assert(n <= keep);
    var k: usize = 0;
    const r0: usize = @intCast(p0 % keep);
    const first = @min(n, keep - r0);
    for ([_]Piece{ .{ .src = 0, .dst = r0, .rows = first }, .{ .src = first, .dst = 0, .rows = n - first } }) |seg| {
        if (seg.rows == 0) continue;
        out[k] = seg;
        k += 1;
        if (seg.dst < mirror) {
            out[k] = .{ .src = seg.src, .dst = keep + seg.dst, .rows = @min(seg.rows, mirror - seg.dst) };
            k += 1;
        }
    }
    return out[0..k];
}

/// Rows, each the tapped layers' width, `prepare` allocates: a forward's `max_rows` and a ring for each of `slots`.
pub fn heldRows(max_rows: usize, slots: usize) usize {
    return max_rows + slots * (keep + mirror);
}

pub const Taps = struct {
    ids: [most]u32 = undefined,
    n: usize = 0, // 0: no drafter taps; forwards copy nothing
    rows: ?cuda.DeviceBuffer = null, // the forward's tap_rows
    ring: ?cuda.DeviceBuffer = null, // [slots, keep + mirror, row]
    slots: usize = 0,
    written: []u64 = &.{}, // a slot's ring holds positions [written - keep, written)
    prompt: ?fwd.Chain = null, // the last forward was this stream's prompt chunk: its rows are in tap_rows whole

    pub fn deinit(t: *Taps, gpa: std.mem.Allocator) void {
        if (t.rows) |*b| b.free();
        if (t.ring) |*b| b.free();
        gpa.free(t.written);
        t.* = .{};
    }

    pub fn rowBytes(t: *const Taps, m: *const fwd.Model) usize {
        return t.n * m.c.hidden * 2;
    }

    /// The drafter's layers (0-based, rising, normed outputs): its rows and every slot's ring, before the pool plan.
    pub fn prepare(t: *Taps, gpa: std.mem.Allocator, m: *fwd.Model, taps: []const u32) !void {
        if (t.n != 0) return error.TapsTwice;
        if (taps.len == 0 or taps.len > most) return error.TooManyTaps;
        for (taps, 0..) |id, i| if (id >= m.c.layers or (i > 0 and id <= taps[i - 1])) return error.BadTap;
        @memcpy(t.ids[0..taps.len], taps);
        t.n = taps.len;
        errdefer t.n = 0;
        t.rows = try cuda.DeviceBuffer.alloc(m.d, m.max_rows * t.rowBytes(m));
        errdefer if (t.rows) |*b| {
            b.free();
            t.rows = null;
        };
        try t.reserve(gpa, m, m.slotCap());
        m.taps = t.ids[0..t.n];
        m.tap_rows = t.rows.?.ptr;
    }

    /// A ring for each of `slots` streams (kept rings are dropped: no stream holds one while slots grow).
    fn reserve(t: *Taps, gpa: std.mem.Allocator, m: *const fwd.Model, slots: usize) !void {
        if (slots <= t.slots) return;
        if (t.ring) |*b| b.free();
        t.ring = null;
        t.slots = 0;
        gpa.free(t.written);
        t.written = &.{};
        var ring = try cuda.DeviceBuffer.alloc(m.d, slots * (keep + mirror) * t.rowBytes(m));
        errdefer ring.free(); // published only once the host slice is made too
        const written = try gpa.alloc(u64, slots);
        @memset(written, 0);
        t.ring = ring;
        t.written = written;
        t.slots = slots;
    }

    /// After a forward over `chains` (rows back to back in tap_rows): each chain's last rows into its slot's ring.
    pub fn kept(t: *Taps, gpa: std.mem.Allocator, m: *const fwd.Model, s: cuda.Stream, chains: []const fwd.Chain, prompt: bool) !void {
        if (t.n == 0) return;
        try t.reserve(gpa, m, m.slotCap());
        const row = t.rowBytes(m);
        var at: usize = 0;
        for (chains) |ch| {
            const n = @min(ch.tokens.len, keep);
            const skip = ch.tokens.len - n;
            var buf: [4]Piece = undefined;
            for (pieces(ch.p0 + skip, n, &buf)) |p| {
                const dst = t.ring.?.ptr + (ch.slot * (keep + mirror) + p.dst) * row;
                try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, m.tap_rows + (at + skip + p.src) * row, p.rows * row, s.handle), "cuMemcpyDtoDAsync");
            }
            t.written[ch.slot] = ch.p0 + ch.tokens.len;
            at += ch.tokens.len;
        }
        t.prompt = if (prompt) chains[0] else null;
    }

    /// A keep rolled `slot` back to `len` positions: rows past it were rejected drafts', no longer handed out.
    pub fn keepTo(t: *Taps, slot: usize, len: u64) void {
        if (slot < t.slots) t.written[slot] = @min(t.written[slot], len);
    }

    /// Rows [start, start + count) of `slot`'s taps: the last prompt chunk's whole, else up to `mirror` from the ring.
    pub fn view(t: *const Taps, m: *const fwd.Model, slot: usize, start: u64, count: u32) !lanes.backend.Features {
        if (t.n == 0) return error.NoTaps;
        const row = t.rowBytes(m);
        const f: lanes.backend.Features = .{ .buffer = 0, .rows = count, .row_bytes = @intCast(row), .space = .device, .dtype = .bf16 };
        if (t.prompt) |p| if (p.slot == slot and start >= p.p0 and start + count <= p.p0 + p.tokens.len) {
            var v = f;
            v.buffer = m.tap_rows + (start - p.p0) * row;
            return v;
        };
        if (slot >= t.slots or count > mirror or start + count > t.written[slot] or start < t.written[slot] -| keep) return error.RowsNotHeld;
        var v = f;
        v.buffer = t.ring.?.ptr + (slot * (keep + mirror) + @as(usize, @intCast(start % keep))) * row;
        return v;
    }
};

test "a chunk's rows land at their ring rows, rows below the mirror twice, and every view up to it is contiguous" {
    var buf: [4]Piece = undefined;
    try std.testing.expectEqualSlices(Piece, &.{ .{ .src = 0, .dst = 10, .rows = 5 }, .{ .src = 0, .dst = keep + 10, .rows = 5 } }, pieces(10, 5, &buf));
    try std.testing.expectEqualSlices(Piece, &.{.{ .src = 0, .dst = 100, .rows = 5 }}, pieces(keep + 100, 5, &buf));
    try std.testing.expectEqualSlices(Piece, &.{ .{ .src = 0, .dst = 250, .rows = 6 }, .{ .src = 6, .dst = 0, .rows = 4 }, .{ .src = 6, .dst = keep, .rows = 4 } }, pieces(250, 10, &buf));
    // a whole lap: positions 300..555 -> rows 44..255 then 0..43; rows 44..63 and 0..43 mirrored
    const lap = pieces(300, keep, &buf);
    try std.testing.expectEqual(@as(usize, 4), lap.len);
    try std.testing.expectEqualSlices(Piece, &.{ .{ .src = 0, .dst = 44, .rows = 212 }, .{ .src = 0, .dst = keep + 44, .rows = 20 }, .{ .src = 212, .dst = 0, .rows = 44 }, .{ .src = 212, .dst = keep, .rows = 44 } }, lap);
    // the ring as positions: every view of up to `mirror` rows over the last `keep` reads its positions in order
    var ring: [keep + mirror]u64 = @splat(std.math.maxInt(u64));
    var p0: u64 = 0;
    for ([_]usize{ 37, 256, 1, 17, 200, 3 }) |n| {
        for (pieces(p0, n, &buf)) |p| for (0..p.rows) |i| {
            ring[p.dst + i] = p0 + p.src + i;
        };
        p0 += n;
        var start = p0 -| keep;
        while (start < p0) : (start += 1) {
            const count = @min(mirror, p0 - start);
            const r: usize = @intCast(start % keep);
            for (0..count) |i| try std.testing.expectEqual(start + i, ring[r + i]);
        }
    }
}

const Fake = struct { // a driver that counts live device allocations
    var live: usize = 0;
    var next: u64 = 0x10000;
    fn alloc(out: *u64, len: usize) callconv(.c) c_int {
        out.* = next;
        next += len + 256;
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        live -= 1;
        return 0;
    }
};

test "prepare: a failed host allocation leaves no device allocation behind, the new ring included" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    var m: fwd.Model = undefined;
    m.d = &d;
    m.c.layers = 4;
    m.c.hidden = 8;
    m.max_rows = 16;
    m.paged = false;
    m.slots = 2;
    m.taps = &.{};
    m.tap_rows = 0;
    var fail: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 0 });
    var t: Taps = .{};
    try std.testing.expectError(error.OutOfMemory, t.prepare(fail.allocator(), &m, &.{ 1, 3 }));
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
    try std.testing.expect(t.ring == null and t.rows == null and t.slots == 0 and t.n == 0);
    try t.prepare(std.testing.allocator, &m, &.{ 1, 3 }); // the same call with room: rows and the ring made
    try std.testing.expectEqual(@as(usize, 2), Fake.live);
    t.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
}

test "keepTo: after a verify of four rows keeps two, the rejected rows are not handed out" {
    var t: Taps = .{ .n = 1, .slots = 1 };
    var written = [_]u64{104}; // a verify wrote positions 100..103
    t.written = &written;
    t.ring = .{ .d = undefined, .ptr = 0x10000, .len = 0, .borrowed = true };
    var m: fwd.Model = undefined;
    m.c.hidden = 8;
    _ = try t.view(&m, 0, 100, 4);
    t.keepTo(0, 102);
    _ = try t.view(&m, 0, 100, 2);
    try std.testing.expectError(error.RowsNotHeld, t.view(&m, 0, 100, 3));
    t.keepTo(0, 110); // a keep never extends what the ring holds
    try std.testing.expectEqual(@as(u64, 102), written[0]);
}
