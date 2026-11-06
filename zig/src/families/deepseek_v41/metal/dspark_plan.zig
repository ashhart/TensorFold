//! The drafter's key rings and a launch's records: row positions, ring rows, selected keys and attention rows.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const ctx_ = @import("ctx.zig");
const pipelines = @import("pipelines.zig");

const attn = pipelines.attn;
pub const Ref = ctx_.Ref;
pub const max_rows = pipelines.max_rows;

/// Rows a slot keeps a stage: its context ring (main keys by position mod window), then the block's own.
pub fn poolRows(c: *const Config) u32 {
    return c.window + c.dspark_block;
}

/// The pool row of position `p` of slot `slot`'s ring.
pub fn ringRow(c: *const Config, slot: u32, p: u64) u32 {
    return slot * poolRows(c) + @as(u32, @intCast(p % c.window));
}

/// The pool row of the block's key `i` of slot `slot`.
pub fn blockRow(c: *const Config, slot: u32, i: u32) u32 {
    return slot * poolRows(c) + c.window + i;
}

/// The context positions of a draft block: the first and how many.
pub const Context = struct { first: u64, count: u32 };

/// A slot's context: the ingested positions [first, end) of one run (a gap or a rewind before `first` cut it).
pub const Run = struct {
    first: u64 = 0,
    end: u64 = 0,

    /// Rows landing at [start, start + n): a rewind or a gap restarts the run there; the run then ends after them.
    pub fn land(r: *Run, start: u64, n: u64) void {
        if (start > r.end or start < r.first) r.first = start;
        r.end = start + n;
    }

    /// The context positions of a draft block starting at `p`: at most `window`, none clobbered by a later row.
    pub fn context(r: Run, window: u32, p: u64) Context {
        const lo = @max(r.first, @max(r.end -| window, p -| window));
        const hi = @min(p, r.end);
        return .{ .first = lo, .count = @intCast(if (hi > lo) hi - lo else 0) };
    }
};

/// The records one launch reads, in one shared buffer: all i32.
pub const Plan = struct {
    buf: mtl.Buffer,

    const pos = 0;
    const bdest = max_rows;
    const ipos = 2 * max_rows;
    const idest = 3 * max_rows;
    const meta = 4 * max_rows;
    const sel = meta + 8 * max_rows;
    pub const ints = sel + max_rows * attn.max_selected;

    pub fn init(device: mtl.Device) !Plan {
        const buf = try device.buffer(4 * ints, mtl.ResourceOptions.shared);
        @memset(buf.contents()[0 .. 4 * ints], 0);
        return .{ .buf = buf };
    }

    pub fn deinit(p: *Plan) void {
        p.buf.deinit();
    }

    fn at(p: *const Plan, off: usize) Ref {
        return .{ .buf = p.buf, .off = 4 * off };
    }

    pub fn ints32(p: *const Plan) []i32 {
        return p.buf.slice(i32, ints);
    }

    /// Row positions (RoPE), the pool rows the rows' keys land in, attention rows, and the selected keys of each row.
    pub fn refs(p: *const Plan) struct { pos: Ref, dest: Ref, meta: Ref, sel: Ref } {
        return .{ .pos = p.at(pos), .dest = p.at(bdest), .meta = p.at(meta), .sel = p.at(sel) };
    }

    /// Ingest rows: positions and ring rows (`slot`'s), `n` rows from `start`.
    pub fn writeIngest(p: *const Plan, c: *const Config, slot: u32, start: u64, n: u32) !void {
        if (n == 0 or n > max_rows) return error.BadRows;
        const m = p.ints32();
        for (0..n) |r| {
            m[ipos + r] = @intCast(start + r);
            m[idest + r] = @intCast(ringRow(c, slot, start + r));
        }
    }

    pub fn ingestRefs(p: *const Plan) struct { pos: Ref, dest: Ref } {
        return .{ .pos = p.at(ipos), .dest = p.at(idest) };
    }

    /// One slot of a draft launch: `block` rows at positions p .. p + block - 1 in rows [row0, row0 + block).
    pub fn writeBlock(p: *const Plan, c: *const Config, j: u32, slot: u32, start: u64, ctx: Run) !void {
        const n = c.dspark_block;
        const row0 = j * n;
        if (row0 + n > max_rows) return error.BadRows;
        const m = p.ints32();
        const cx = ctx.context(c.window, start);
        const list = m[sel + row0 * attn.max_selected ..][0..attn.max_selected];
        @memset(list, -1);
        // the selected keys: the context ring rows by position, then the block's keys but the first (the window's)
        for (0..cx.count) |t| list[t] = @intCast((cx.first + t) % c.window);
        if (cx.count + n - 1 > attn.max_selected) return error.ContextWidth;
        for (1..n) |i| list[cx.count + i - 1] = @intCast(c.window + i);
        const base: i32 = @intCast(slot * poolRows(c));
        for (0..n) |i| {
            const r = row0 + i;
            m[pos + r] = @intCast(start + i);
            m[bdest + r] = @intCast(blockRow(c, slot, @intCast(i)));
            // the window holds one key at artificial position 0 (the block's first); the RoPE row is the row's own
            const rec = attn.Meta{ .pos = 0, .wofs = base + @as(i32, @intCast(c.window)), .wbase = 0, .cofs = base, .ncomp = @intCast(cx.count + n - 1), .rope = @intCast(start + i) };
            @memcpy(m[meta + 8 * r ..][0..8], @as(*const [8]i32, @ptrCast(&rec)));
            // every row of the block reads the same selected keys from its own list
            if (i > 0) @memcpy(m[sel + r * attn.max_selected ..][0..attn.max_selected], list);
        }
    }
};

test "a run: contiguous rows extend it, a gap or a rewind before it restarts it" {
    var r: Run = .{};
    r.land(0, 40);
    try std.testing.expectEqual(@as(u64, 0), r.first);
    r.land(40, 5);
    try std.testing.expectEqual(@as(u64, 45), r.end);
    r.land(60, 3); // a gap
    try std.testing.expectEqual(@as(u64, 60), r.first);
    r.land(61, 2); // a rewind inside the run keeps its start
    try std.testing.expectEqual(@as(u64, 60), r.first);
    try std.testing.expectEqual(@as(u64, 63), r.end);
}

test "a context holds at most a window, none of it clobbered or past the ingested rows" {
    var r: Run = .{};
    r.land(0, 300);
    const c = r.context(128, 300);
    try std.testing.expectEqual(@as(u64, 172), c.first);
    try std.testing.expectEqual(@as(u32, 128), c.count);
    try std.testing.expectEqual(@as(u32, 0), r.context(128, 0).count);
    // a block asked after a rewind of two positions sees what is before it, from a ring only 300 deep
    const back = r.context(128, 298);
    try std.testing.expectEqual(@as(u64, 172), back.first);
    try std.testing.expectEqual(@as(u32, 126), back.count);
    r.land(0, 10);
    try std.testing.expectEqual(@as(u32, 10), r.context(128, 12).count);
}
