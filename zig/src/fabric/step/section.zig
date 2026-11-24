//! A step header's optional sections: an 8-byte prefix, `count` entries, zero padding to 8 bytes; each entry's layout.
const std = @import("std");

/// Which section: its id is its bit in the header's `sections` mask, and sections appear in ascending id.
pub const Id = enum(u8) { streams = 1, tokens = 2, cuts = 3, sampling = 4, pages = 5, copies = 6, snapshots = 7, rows = 8, family = 15 };

/// The prefix: `id` u8, `version` u8, reserved u16, `count` u32.
pub const prefix = 8;
pub const version: u8 = 1;

/// Fixed entry sizes; `pages` has variable entries (16 bytes, then 4 a page id) and `family` counts bytes.
pub const stream_size = 28;
pub const sampling_size = 64;
pub const patch_fixed = 16;
pub const copy_size = 8;
pub const snapshot_size = 24;
pub const row_size = 8;

/// What a step does with a stream: start it, extend it, end it, drop it, or resume it from a kept prompt state.
pub const Op = enum(u8) { open = 1, @"continue" = 2, close = 3, cancel = 4, @"resume" = 5 };

/// One stream's part of a step: the ring's plan entry byte for byte (`fabric/stage/plan.zig`).
pub const Stream = struct {
    id: u32,
    op: Op,
    /// The position of the stream's first row this step, and its rows (0 for close and cancel).
    first: u32,
    rows: u32,
    /// The previous verify's accepted count: every rank rolls its state back to it.
    accepted: u32 = 0,
    /// For `resume`, the kept prompt state's position (1 up to `first`); 0 for every other op.
    resume_at: u32 = 0,
    /// The slot's state generation (0 for `open`).
    generation: u32 = 0,
};

/// One sampler call that runs on every rank; a field whose `has` bit is clear is absent (written 0).
pub const Sampling = struct {
    stream: u32,
    top_k: u32 = 0,
    seed: u64 = 0,
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    /// The rows it samples; 0 and 0 mean the stream's rows.
    row0: u32 = 0,
    rows: u32 = 0,
    /// The first draw position.
    pos0: u64 = 0,
    has: Has = .{},

    pub const Has = packed struct(u32) { seed: bool = false, pos0: bool = false, reserved: u30 = 0 };
};

/// A page-table patch: entries `[first, first + ids.len)` of `stream`'s table become `ids`; the first `kept` hold rows.
pub const Patch = struct { stream: u32, first: u32, kept: u32 = 0, ids: []const u32 };

/// Page `to` becomes a copy of page `from`, before the patches apply.
pub const Copy = struct { from: u32, to: u32 };

pub const SnapOp = enum(u8) { drop = 1, restore = 2, save = 3 };

/// A kept state: dropped, restored for a stream, or saved from a stream at the end of the round.
pub const Snapshot = struct { op: SnapOp, stream: u32 = 0, id: u64, at: u32 = 0 };

/// A compact row-map entry: a `continue` of `rows` rows at the follower's own position, nothing else.
pub const RowEntry = struct { id: u32, rows: u32 };

pub fn put32(p: []u8, at: usize, v: u32) void {
    std.mem.writeInt(u32, p[at..][0..4], v, .little);
}

pub fn put64(p: []u8, at: usize, v: u64) void {
    std.mem.writeInt(u64, p[at..][0..8], v, .little);
}

pub fn get32(p: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, p[at..][0..4], .little);
}

pub fn get64(p: []const u8, at: usize) u64 {
    return std.mem.readInt(u64, p[at..][0..8], .little);
}

/// `n` rounded up to a multiple of 8, or null past u64.
pub fn padded(n: u64) ?u64 {
    const up = std.math.add(u64, n, 7) catch return null;
    return up & ~@as(u64, 7);
}

pub fn putPrefix(p: []u8, id: Id, count: u32) void {
    p[0] = @backingInt(id);
    p[1] = version;
    p[2] = 0;
    p[3] = 0;
    put32(p, 4, count);
}

pub fn putStream(p: []u8, s: Stream) void {
    @memset(p[0..stream_size], 0);
    put32(p, 0, s.id);
    p[4] = @backingInt(s.op);
    put32(p, 8, s.first);
    put32(p, 12, s.rows);
    put32(p, 16, s.accepted);
    put32(p, 20, s.resume_at);
    put32(p, 24, s.generation);
}

/// A stream entry whose op byte the decoder has checked.
pub fn getStream(p: []const u8) Stream {
    return .{
        .id = get32(p, 0),
        .op = @fromBackingInt(@intCast(p[4])),
        .first = get32(p, 8),
        .rows = get32(p, 12),
        .accepted = get32(p, 16),
        .resume_at = get32(p, 20),
        .generation = get32(p, 24),
    };
}

pub fn putSampling(p: []u8, s: Sampling) void {
    @memset(p[0..sampling_size], 0);
    put32(p, 0, s.stream);
    put32(p, 4, s.top_k);
    put64(p, 8, s.seed);
    put64(p, 16, @bitCast(s.temperature));
    put64(p, 24, @bitCast(s.top_p));
    put64(p, 32, @bitCast(s.min_p));
    put32(p, 40, s.row0);
    put32(p, 44, s.rows);
    put64(p, 48, s.pos0);
    put32(p, 56, @bitCast(s.has));
}

pub fn getSampling(p: []const u8) Sampling {
    return .{
        .stream = get32(p, 0),
        .top_k = get32(p, 4),
        .seed = get64(p, 8),
        .temperature = @bitCast(get64(p, 16)),
        .top_p = @bitCast(get64(p, 24)),
        .min_p = @bitCast(get64(p, 32)),
        .row0 = get32(p, 40),
        .rows = get32(p, 44),
        .pos0 = get64(p, 48),
        .has = @bitCast(get32(p, 56)),
    };
}

/// A patch's bytes: 16 + 4 a page id.
pub fn patchSize(n: u64) u64 {
    return patch_fixed + 4 * n;
}

pub fn putPatch(p: []u8, x: Patch) void {
    put32(p, 0, x.stream);
    put32(p, 4, x.first);
    put32(p, 8, @intCast(x.ids.len));
    put32(p, 12, x.kept);
    for (x.ids, 0..) |id, i| put32(p, patch_fixed + 4 * i, id);
}

/// A patch read in place: its page ids stay bytes of the payload.
pub const PatchView = struct {
    stream: u32,
    first: u32,
    n: u32,
    kept: u32,
    ids: []const u8,

    pub fn id(v: PatchView, i: usize) u32 {
        return get32(v.ids, 4 * i);
    }
};

/// The patch at the start of `p`, whose length the decoder has checked.
pub fn getPatch(p: []const u8) PatchView {
    const n = get32(p, 8);
    return .{ .stream = get32(p, 0), .first = get32(p, 4), .n = n, .kept = get32(p, 12), .ids = p[patch_fixed..][0 .. 4 * @as(usize, n)] };
}

pub fn putCopy(p: []u8, c: Copy) void {
    put32(p, 0, c.from);
    put32(p, 4, c.to);
}

pub fn getCopy(p: []const u8) Copy {
    return .{ .from = get32(p, 0), .to = get32(p, 4) };
}

pub fn putSnapshot(p: []u8, s: Snapshot) void {
    @memset(p[0..snapshot_size], 0);
    p[0] = @backingInt(s.op);
    put32(p, 4, s.stream);
    put64(p, 8, s.id);
    put32(p, 16, s.at);
}

/// A snapshot entry whose op byte the decoder has checked.
pub fn getSnapshot(p: []const u8) Snapshot {
    return .{ .op = @fromBackingInt(@intCast(p[0])), .stream = get32(p, 4), .id = get64(p, 8), .at = get32(p, 16) };
}

pub fn putRow(p: []u8, r: RowEntry) void {
    put32(p, 0, r.id);
    put32(p, 4, r.rows);
}

pub fn getRow(p: []const u8) RowEntry {
    return .{ .id = get32(p, 0), .rows = get32(p, 4) };
}

/// A checked section in place: its id, its count and its body (entries without padding).
pub const Section = struct {
    id: Id,
    count: u32,
    body: []const u8,

    fn at(s: Section, size: usize, i: usize) []const u8 {
        std.debug.assert(i < s.count);
        return s.body[size * i ..][0..size];
    }

    pub fn stream(s: Section, i: usize) Stream {
        return getStream(s.at(stream_size, i));
    }

    /// A u32 entry: a token id or a cut.
    pub fn word(s: Section, i: usize) u32 {
        return get32(s.at(4, i), 0);
    }

    pub fn sampling(s: Section, i: usize) Sampling {
        return getSampling(s.at(sampling_size, i));
    }

    pub fn copy(s: Section, i: usize) Copy {
        return getCopy(s.at(copy_size, i));
    }

    pub fn snapshot(s: Section, i: usize) Snapshot {
        return getSnapshot(s.at(snapshot_size, i));
    }

    pub fn row(s: Section, i: usize) RowEntry {
        return getRow(s.at(row_size, i));
    }

    /// The patches in order; the decoder has bounded each one by the body.
    pub fn patches(s: Section) Patches {
        return .{ .body = s.body, .left = s.count };
    }
};

pub const Patches = struct {
    body: []const u8,
    left: u32,

    pub fn next(it: *Patches) ?PatchView {
        if (it.left == 0) return null;
        const v = getPatch(it.body);
        it.body = it.body[patchSize(v.n)..];
        it.left -= 1;
        return v;
    }
};

const t = std.testing;

test "step sections: every entry type round-trips through its writer and its reader" {
    var buf: [128]u8 = undefined;
    const s: Stream = .{ .id = 9, .op = .@"resume", .first = 32, .rows = 2, .accepted = 1, .resume_at = 30, .generation = 2 };
    putStream(&buf, s);
    try t.expectEqual(s, getStream(&buf));
    try t.expectEqualSlices(u8, &.{ 0, 0, 0 }, buf[5..8]);
    const nan: f64 = @bitCast(@as(u64, 0x7FF8_0000_0000_0123));
    const g: Sampling = .{ .stream = 3, .top_k = 40, .seed = 1 << 63, .temperature = -0.0, .top_p = nan, .min_p = 0.05, .row0 = 1, .rows = 2, .pos0 = 77, .has = .{ .seed = true, .pos0 = true } };
    putSampling(&buf, g);
    const back = getSampling(&buf);
    try t.expectEqual(@as(u64, @bitCast(g.temperature)), @as(u64, @bitCast(back.temperature))); // -0.0 keeps its sign
    try t.expectEqual(@as(u64, @bitCast(g.top_p)), @as(u64, @bitCast(back.top_p))); // the NaN keeps its payload
    try t.expectEqual(g.seed, back.seed);
    try t.expectEqual(g.has, back.has);
    try t.expectEqual(@as(u32, 3), get32(&buf, 56));
    const ids = [_]u32{ 4, 5, 6 };
    putPatch(&buf, .{ .stream = 1, .first = 2, .kept = 1, .ids = &ids });
    const v = getPatch(buf[0..patchSize(3)]);
    try t.expectEqual(@as(u32, 3), v.n);
    try t.expectEqual(@as(u32, 1), v.kept);
    try t.expectEqual(@as(u32, 6), v.id(2));
    putCopy(&buf, .{ .from = 7, .to = 8 });
    try t.expectEqual(Copy{ .from = 7, .to = 8 }, getCopy(&buf));
    const k: Snapshot = .{ .op = .save, .stream = 3, .id = 0x1_0000_0002, .at = 12 };
    putSnapshot(&buf, k);
    try t.expectEqual(k, getSnapshot(&buf));
    putRow(&buf, .{ .id = 5, .rows = 1 });
    try t.expectEqual(RowEntry{ .id = 5, .rows = 1 }, getRow(&buf));
}

test "step sections: padding rounds up to 8 and refuses past u64" {
    try t.expectEqual(@as(?u64, 0), padded(0));
    try t.expectEqual(@as(?u64, 8), padded(1));
    try t.expectEqual(@as(?u64, 8), padded(8));
    try t.expectEqual(@as(?u64, 16), padded(9));
    try t.expectEqual(@as(?u64, null), padded(std.math.maxInt(u64) - 3));
}
