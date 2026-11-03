//! A stage's boot messages to rank 0 (its description, then ready) and rank 0's checks that refuse a mismatch by name.
const std = @import("std");
const payload = @import("payload.zig");

const Backend = payload.Backend;
const Dtype = payload.Dtype;
const get32 = payload.get32;
const put32 = payload.put32;

/// The longest engine version text, and the most drafter tap layers a model has.
pub const most_version = 64;
pub const most_taps = 16;

/// The description's fixed part; its taps (4 bytes each) and then its version text follow.
pub const describe_fixed = 72;
pub const ready_size = 16;

pub const DecodeError = payload.Error || error{
    BadBackend,
    BadDtype,
    BadRowDtype,
    BadVersionText,
    BadTaps,
    EmptyRange,
    BadBoundary,
    BadShapes,
    BadFlags,
};

pub const CheckError = error{
    OtherRank,
    OtherVersion,
    OtherBackend,
    OtherKvDtype,
    OtherRange,
    OtherWidth,
    OtherDtype,
    OtherPlanes,
    OtherDraftWidth,
    OtherShapes,
    OtherTaps,
    ShortCapacity,
};

pub const LayoutError = error{ BadLayout, EmptyRange, RangeGap, RangeOverlap, RangeShort, RangeBeyond };

/// A layer range, half-open: layers `first` up to but not including `end`.
pub const Range = struct { first: u32, end: u32 };

/// The hidden rows crossing a stage boundary; `draft_width` 0 means no drafter taps ride with them.
pub const Boundary = struct { hidden: u32, row_dtype: Dtype, planes: u8, draft_width: u32 };

/// The execution shapes every stage runs with: one placement's bits assume the same ones (RFC section 5).
pub const Shapes = struct { prompt_chunk: u32, max_rows: u32, parallel: u32, context: u32 };

/// The engine's version, printable ASCII, 1 to 64 bytes.
pub const Text = struct {
    len: u8 = 0,
    bytes: [most_version]u8 = @splat(0),

    pub fn init(s: []const u8) DecodeError!Text {
        if (s.len == 0 or s.len > most_version) return error.BadVersionText;
        for (s) |c| if (c < 0x20 or c > 0x7e) return error.BadVersionText;
        var t: Text = .{ .len = @intCast(s.len) };
        @memcpy(t.bytes[0..s.len], s);
        return t;
    }

    pub fn slice(t: *const Text) []const u8 {
        return t.bytes[0..t.len];
    }
};

/// Drafter tap layer ids, strictly ascending, at most 16.
pub const Taps = struct {
    n: u8 = 0,
    ids: [most_taps]u32 = @splat(0),

    pub fn init(ids: []const u32) DecodeError!Taps {
        if (ids.len > most_taps) return error.BadTaps;
        if (ids.len > 1) for (ids[1..], ids[0 .. ids.len - 1]) |id, before| if (id <= before) return error.BadTaps;
        var t: Taps = .{ .n = @intCast(ids.len) };
        @memcpy(t.ids[0..ids.len], ids);
        return t;
    }

    pub fn slice(t: *const Taps) []const u32 {
        return t.ids[0..t.n];
    }
};

/// What a stage tells rank 0 at boot, before any round runs.
pub const Description = struct {
    rank: u32,
    world: u32,
    version: Text,
    backend: Backend,
    kv_dtype: Dtype,
    range: Range,
    boundary: Boundary,
    shapes: Shapes,
    /// The drafter tap layers inside this stage's range.
    taps: Taps,
    /// Capacity: the KV rows and the recurrent states this stage can hold.
    kv_rows: u64,
    states: u32,

    pub fn size(d: *const Description) usize {
        return describe_fixed + 4 * @as(usize, d.taps.n) + d.version.len;
    }
};

/// The description's bytes at the start of `out`.
pub fn encodeDescription(d: *const Description, out: []u8) payload.Error![]u8 {
    const n = d.size();
    if (out.len < n) return error.NoRoom;
    const p = out[0..n];
    @memset(p, 0);
    payload.put(p, .describe);
    put32(p, 8, d.rank);
    put32(p, 12, d.world);
    p[16] = @backingInt(d.backend);
    p[17] = @backingInt(d.kv_dtype);
    p[18] = @backingInt(d.boundary.row_dtype);
    p[19] = d.boundary.planes;
    p[20] = d.taps.n;
    p[21] = d.version.len;
    put32(p, 24, d.range.first);
    put32(p, 28, d.range.end);
    put32(p, 32, d.boundary.hidden);
    put32(p, 36, d.boundary.draft_width);
    put32(p, 40, d.shapes.prompt_chunk);
    put32(p, 44, d.shapes.max_rows);
    put32(p, 48, d.shapes.parallel);
    put32(p, 52, d.shapes.context);
    payload.put64(p, 56, d.kv_rows);
    put32(p, 64, d.states);
    for (d.taps.slice(), 0..) |id, i| put32(p, describe_fixed + 4 * i, id);
    @memcpy(p[describe_fixed + 4 * @as(usize, d.taps.n) ..], d.version.slice());
    return p;
}

/// A description from `p`, refused by name when malformed: length, enums, reserved bytes, ranges, taps, text.
pub fn decodeDescription(p: []const u8) DecodeError!Description {
    try payload.expect(p, .describe);
    if (p.len < describe_fixed) return error.BadLength;
    const n_taps: usize = p[20];
    if (n_taps > most_taps) return error.BadTaps;
    if (p.len != describe_fixed + 4 * n_taps + p[21]) return error.BadLength;
    if (p[22] != 0 or p[23] != 0 or get32(p, 68) != 0) return error.BadReserved;
    const row_dtype = std.enums.fromInt(Dtype, p[18]) orelse return error.BadDtype;
    if (row_dtype.bytes() == null) return error.BadRowDtype;
    var ids: [most_taps]u32 = undefined;
    for (ids[0..n_taps], 0..) |*id, i| id.* = get32(p, describe_fixed + 4 * i);
    const d: Description = .{
        .rank = get32(p, 8),
        .world = get32(p, 12),
        .version = try Text.init(p[describe_fixed + 4 * n_taps ..]),
        .backend = std.enums.fromInt(Backend, p[16]) orelse return error.BadBackend,
        .kv_dtype = std.enums.fromInt(Dtype, p[17]) orelse return error.BadDtype,
        .range = .{ .first = get32(p, 24), .end = get32(p, 28) },
        .boundary = .{ .hidden = get32(p, 32), .row_dtype = row_dtype, .planes = p[19], .draft_width = get32(p, 36) },
        .shapes = .{
            .prompt_chunk = get32(p, 40),
            .max_rows = get32(p, 44),
            .parallel = get32(p, 48),
            .context = get32(p, 52),
        },
        .taps = try Taps.init(ids[0..n_taps]),
        .kv_rows = payload.get64(p, 56),
        .states = get32(p, 64),
    };
    if (d.range.first >= d.range.end) return error.EmptyRange;
    for (d.taps.slice()) |id| if (id < d.range.first or id >= d.range.end) return error.BadTaps;
    if (d.boundary.hidden == 0 or d.boundary.planes == 0) return error.BadBoundary;
    const s = d.shapes;
    if (s.prompt_chunk == 0 or s.max_rows == 0 or s.parallel == 0 or s.context == 0) return error.BadShapes;
    return d;
}

/// The model and engine every stage must match, as rank 0 sees them.
pub const Model = struct {
    version: []const u8,
    kv_dtype: Dtype,
    layers: u32,
    boundary: Boundary,
    shapes: Shapes,
    /// Every drafter tap layer, strictly ascending; each stage owns those inside its range.
    taps: []const u32,
};

/// Rank 0's plan for one stage: its rank, the ring's size, its backend and its range.
pub const Expect = struct { model: *const Model, rank: u32, world: u32, backend: Backend, range: Range };

/// A stage's description against rank 0's expectation; the first mismatch is refused by its name.
pub fn check(e: Expect, d: *const Description) CheckError!void {
    const m = e.model;
    if (d.rank != e.rank or d.world != e.world) return error.OtherRank;
    if (!std.mem.eql(u8, d.version.slice(), m.version)) return error.OtherVersion;
    if (d.backend != e.backend) return error.OtherBackend;
    if (d.kv_dtype != m.kv_dtype) return error.OtherKvDtype;
    if (d.range.first != e.range.first or d.range.end != e.range.end) return error.OtherRange;
    if (d.boundary.hidden != m.boundary.hidden) return error.OtherWidth;
    if (d.boundary.row_dtype != m.boundary.row_dtype) return error.OtherDtype;
    if (d.boundary.planes != m.boundary.planes) return error.OtherPlanes;
    if (d.boundary.draft_width != m.boundary.draft_width) return error.OtherDraftWidth;
    if (!std.meta.eql(d.shapes, m.shapes)) return error.OtherShapes;
    var owned: usize = 0;
    for (m.taps) |id| {
        if (id < e.range.first or id >= e.range.end) continue;
        if (owned >= d.taps.n or d.taps.ids[owned] != id) return error.OtherTaps;
        owned += 1;
    }
    if (owned != d.taps.n) return error.OtherTaps;
    if (d.kv_rows < m.shapes.context or d.states < m.shapes.parallel) return error.ShortCapacity;
}

/// A layout of `ranges` in rank order: contiguous from layer 0 to the model's `layers`, none empty, none overlapping.
pub fn checkLayout(layers: u32, ranges: []const Range) LayoutError!void {
    if (ranges.len == 0) return error.BadLayout;
    var at: u32 = 0;
    for (ranges) |r| {
        if (r.first >= r.end) return error.EmptyRange;
        if (r.first > at) return error.RangeGap;
        if (r.first < at) return error.RangeOverlap;
        if (r.end > layers) return error.RangeBeyond;
        at = r.end;
    }
    if (at != layers) return error.RangeShort;
}

/// The tightest stage's capacity: rank 0's admission charges by it.
pub fn tightest(descs: []const Description) struct { kv_rows: u64, states: u32 } {
    var kv: u64 = std.math.maxInt(u64);
    var st: u32 = std.math.maxInt(u32);
    for (descs) |*d| {
        kv = @min(kv, d.kv_rows);
        st = @min(st, d.states);
    }
    return .{ .kv_rows = kv, .states = st };
}

/// A stage's ready report: its range loaded, and warm (graphs captured); rank 0 serves once every stage is both.
pub const Ready = struct {
    rank: u32,
    loaded: bool,
    warm: bool,

    pub fn ready(r: Ready) bool {
        return r.loaded and r.warm;
    }
};

pub fn encodeReady(r: Ready, out: []u8) payload.Error![]u8 {
    if (out.len < ready_size) return error.NoRoom;
    const p = out[0..ready_size];
    payload.put(p, .ready);
    put32(p, 8, r.rank);
    put32(p, 12, @as(u32, @intFromBool(r.loaded)) | @as(u32, @intFromBool(r.warm)) << 1);
    return p;
}

/// A ready report from `p`; flags other than bit 0 (loaded) and bit 1 (warm) are refused.
pub fn decodeReady(p: []const u8) DecodeError!Ready {
    try payload.expect(p, .ready);
    if (p.len != ready_size) return error.BadLength;
    const flags = get32(p, 12);
    if (flags & ~@as(u32, 3) != 0) return error.BadFlags;
    return .{ .rank = get32(p, 8), .loaded = flags & 1 != 0, .warm = flags & 2 != 0 };
}
