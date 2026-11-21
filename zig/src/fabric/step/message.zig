//! A whole step message: the header and its sections, written by a leader and checked by every receiver.
const std = @import("std");
const header = @import("header.zig");
const section = @import("section.zig");

const Header = header.Header;
const Id = section.Id;
const Section = section.Section;
const get32 = section.get32;
const max32 = std.math.maxInt(u32);

/// What a leader writes; every slice may be empty, and an empty slice writes no section.
pub const Message = struct {
    /// Its kind, mode, round, group, family and family flags; `sections`, `streams`, `rows` and `bytes` are computed.
    header: Header,
    streams: []const section.Stream = &.{},
    tokens: []const u32 = &.{},
    cuts: []const u32 = &.{},
    sampling: []const section.Sampling = &.{},
    pages: []const section.Patch = &.{},
    copies: []const section.Copy = &.{},
    snapshots: []const section.Snapshot = &.{},
    rows: []const section.RowEntry = &.{},
    family: []const u8 = &.{},
    /// Write section 1 even with no stream (a row map of count 0).
    streams_section: bool = false,
};

/// What a receiver takes in one message: the transport's byte cap and the shapes it serves.
pub const Limits = struct { bytes: u64, streams: u32, rows: u32, context: u32 };

pub const Error = header.HeaderError || error{
    BadLength,
    TooLarge,
    BadSectionVersion,
    UnknownSection,
    UnorderedSections,
    BadSections,
    BothRowMaps,
    BadStreams,
    TooManyStreams,
    BadOp,
    BadOpen,
    BadResume,
    RowsOnClose,
    PastContext,
    TooManyRows,
    BadTotal,
    RowsOffRound,
    SectionsOffRound,
    BadTokens,
    BadCut,
    BadSampling,
    BadPages,
    BadSnapshotOp,
    BadSnapshot,
};

fn count(n: usize) error{TooLarge}!u32 {
    return std.math.cast(u32, n) orelse error.TooLarge;
}

fn add(a: u64, b: u64) error{TooLarge}!u64 {
    return std.math.add(u64, a, b) catch error.TooLarge;
}

fn mul(a: u64, b: u64) error{TooLarge}!u64 {
    return std.math.mul(u64, a, b) catch error.TooLarge;
}

/// One section's bytes: prefix, entries and padding; 0 when absent.
fn sectionSize(present: bool, body: u64) error{TooLarge}!u64 {
    if (!present) return 0;
    return add(section.prefix, section.padded(body) orelse return error.TooLarge);
}

fn pagesBody(pages: []const section.Patch) error{TooLarge}!u64 {
    var n: u64 = 0;
    for (pages) |p| n = try add(n, try add(section.patch_fixed, try mul(4, try count(p.ids.len))));
    return n;
}

/// The message's bytes: the header, then each present section's prefix, entries and padding.
pub fn size(m: Message) error{TooLarge}!u64 {
    var n: u64 = header.size;
    n = try add(n, try sectionSize(m.streams.len > 0 or m.streams_section, try mul(section.stream_size, try count(m.streams.len))));
    n = try add(n, try sectionSize(m.tokens.len > 0, try mul(4, try count(m.tokens.len))));
    n = try add(n, try sectionSize(m.cuts.len > 0, try mul(4, try count(m.cuts.len))));
    n = try add(n, try sectionSize(m.sampling.len > 0, try mul(section.sampling_size, try count(m.sampling.len))));
    _ = try count(m.pages.len);
    n = try add(n, try sectionSize(m.pages.len > 0, try pagesBody(m.pages)));
    n = try add(n, try sectionSize(m.copies.len > 0, try mul(section.copy_size, try count(m.copies.len))));
    n = try add(n, try sectionSize(m.snapshots.len > 0, try mul(section.snapshot_size, try count(m.snapshots.len))));
    n = try add(n, try sectionSize(m.rows.len > 0, try mul(section.row_size, try count(m.rows.len))));
    n = try add(n, try sectionSize(m.family.len > 0, try count(m.family.len)));
    return n;
}

/// Starts one section at `at`: its prefix written, its padded room zeroed; returns where its entries start.
fn open(p: []u8, at: *usize, id: Id, n: u32, body: usize, mask: *u32) usize {
    section.putPrefix(p[at.*..], id, n);
    const start = at.* + section.prefix;
    const end = start + @as(usize, @intCast(section.padded(body).?));
    @memset(p[start..end], 0);
    at.* = end;
    mask.* |= @as(u32, 1) << @intCast(@backingInt(id));
    return start;
}

/// The message's bytes at the start of `out`, unchecked: `encode` checks them, and the fixtures plant bad ones.
pub fn write(m: Message, out: []u8) error{ NoRoom, TooLarge }![]u8 {
    const total = try size(m);
    if (out.len < total) return error.NoRoom;
    const p = out[0..@intCast(total)];
    const streams_map = m.streams.len > 0 or m.streams_section; // the row map: section 1, else section 8
    var rows: u64 = 0;
    if (streams_map) {
        for (m.streams) |s| rows += s.rows;
    } else for (m.rows) |r| rows += r.rows;
    if (rows > max32) return error.TooLarge;
    var mask: u32 = 0;
    var at: usize = header.size;
    if (streams_map) {
        const b = open(p, &at, .streams, @intCast(m.streams.len), section.stream_size * m.streams.len, &mask);
        for (m.streams, 0..) |s, i| section.putStream(p[b + section.stream_size * i ..], s);
    }
    if (m.tokens.len > 0) {
        const b = open(p, &at, .tokens, @intCast(m.tokens.len), 4 * m.tokens.len, &mask);
        for (m.tokens, 0..) |v, i| section.put32(p, b + 4 * i, v);
    }
    if (m.cuts.len > 0) {
        const b = open(p, &at, .cuts, @intCast(m.cuts.len), 4 * m.cuts.len, &mask);
        for (m.cuts, 0..) |v, i| section.put32(p, b + 4 * i, v);
    }
    if (m.sampling.len > 0) {
        const b = open(p, &at, .sampling, @intCast(m.sampling.len), section.sampling_size * m.sampling.len, &mask);
        for (m.sampling, 0..) |s, i| section.putSampling(p[b + section.sampling_size * i ..], s);
    }
    if (m.pages.len > 0) {
        var b = open(p, &at, .pages, @intCast(m.pages.len), @intCast(pagesBody(m.pages) catch unreachable), &mask);
        for (m.pages) |x| {
            section.putPatch(p[b..], x);
            b += @intCast(section.patchSize(x.ids.len));
        }
    }
    if (m.copies.len > 0) {
        const b = open(p, &at, .copies, @intCast(m.copies.len), section.copy_size * m.copies.len, &mask);
        for (m.copies, 0..) |c, i| section.putCopy(p[b + section.copy_size * i ..], c);
    }
    if (m.snapshots.len > 0) {
        const b = open(p, &at, .snapshots, @intCast(m.snapshots.len), section.snapshot_size * m.snapshots.len, &mask);
        for (m.snapshots, 0..) |s, i| section.putSnapshot(p[b + section.snapshot_size * i ..], s);
    }
    if (m.rows.len > 0) {
        const b = open(p, &at, .rows, @intCast(m.rows.len), section.row_size * m.rows.len, &mask);
        for (m.rows, 0..) |r, i| section.putRow(p[b + section.row_size * i ..], r);
    }
    if (m.family.len > 0) {
        const b = open(p, &at, .family, @intCast(m.family.len), m.family.len, &mask);
        @memcpy(p[b..][0..m.family.len], m.family);
    }
    std.debug.assert(at == p.len);
    var h = m.header;
    h.sections = mask;
    h.streams = @intCast(if (streams_map) m.streams.len else m.rows.len);
    h.rows = @intCast(rows);
    h.bytes = total - header.size;
    p[0..header.size].* = header.encodeHeader(h);
    return p;
}

/// The message's bytes at the start of `out`, refused unless a receiver with `limits` would take them.
pub fn encode(m: Message, limits: Limits, out: []u8) (Error || error{NoRoom})![]u8 {
    const p = try write(m, out);
    _ = try decode(p, limits);
    return p;
}

/// A checked message: a view of its bytes, nothing copied or allocated.
pub const Step = struct {
    header: Header,
    bytes: []const u8,
    sections: [16]?Section = @splat(null),

    pub fn get(s: Step, id: Id) ?Section {
        return s.sections[@backingInt(id)];
    }

    /// The row map: section 1, else section 8, else none.
    pub fn rowMap(s: Step) ?Section {
        return s.get(.streams) orelse s.get(.rows);
    }

    /// A `round` with no row map and no section: the ring's keep-alive.
    pub fn keepAlive(s: Step) bool {
        return s.header.kind == .round and s.header.streams == 0 and s.header.sections == 0;
    }

    /// Section 15's bytes, or none.
    pub fn familyBytes(s: Step) []const u8 {
        return if (s.get(.family)) |f| f.body else &.{};
    }
};

/// The bytes one section's entries take, refused when they would not fit in `room`.
fn bodyBytes(id: Id, n: u32, rest: []const u8) Error!usize {
    const fixed: u64 = switch (id) {
        .streams => section.stream_size,
        .tokens, .cuts => 4,
        .sampling => section.sampling_size,
        .copies => section.copy_size,
        .snapshots => section.snapshot_size,
        .rows => section.row_size,
        .family => 1,
        .pages => return pagesBytes(n, rest),
    };
    const want = @as(u64, n) * fixed; // u32 × at most 64 fits u64
    if (want > rest.len) return error.BadLength;
    return @intCast(want);
}

/// The pages body: `n` patches walked one by one, each bounded by what is left before it is read.
fn pagesBytes(n: u32, rest: []const u8) Error!usize {
    var at: u64 = 0;
    for (0..n) |_| {
        if (rest.len - at < section.patch_fixed) return error.BadLength;
        const ids = get32(rest, @intCast(at + 8));
        at += section.patch_fixed;
        if (rest.len - at < 4 * @as(u64, ids)) return error.BadLength;
        at += 4 * @as(u64, ids);
    }
    return @intCast(at);
}

/// The sections from offset 64: each prefix, body and padding bounded by the payload before it is read.
fn walk(p: []const u8, step: *Step) Error!u32 {
    var at: usize = header.size;
    var mask: u32 = 0;
    var last: u8 = 0;
    while (at < p.len) {
        if (p.len - at < section.prefix) return error.BadLength;
        if (p[at + 1] != section.version) return error.BadSectionVersion;
        if (p[at + 2] != 0 or p[at + 3] != 0) return error.BadReserved;
        const id = std.enums.fromInt(Id, p[at]) orelse return error.UnknownSection;
        if (p[at] <= last) return error.UnorderedSections;
        last = p[at];
        const n = get32(p, at + 4);
        const start = at + section.prefix;
        const body = try bodyBytes(id, n, p[start..]);
        const end = start + body;
        const pad: usize = @intCast(section.padded(body).? - body);
        if (p.len - end < pad) return error.BadLength;
        for (p[end..][0..pad]) |b| if (b != 0) return error.BadReserved;
        step.sections[p[at]] = .{ .id = id, .count = n, .body = p[start..end] };
        mask |= @as(u32, 1) << @intCast(p[at]);
        at = end + pad;
    }
    return mask;
}

/// A step from `p`, every count bounded by its length before any read, and every row against `limits`.
pub fn decode(p: []const u8, limits: Limits) Error!Step {
    if (p.len < header.size) return error.BadLength;
    const h = try header.decodeHeader(p[0..header.size]);
    if (h.bytes > limits.bytes) return error.TooLarge;
    if (h.bytes != p.len - header.size) return error.BadLength;
    var s: Step = .{ .header = h, .bytes = p };
    if (try walk(p, &s) != h.sections) return error.BadSections;
    try checkRowMap(&s, limits);
    try kindRules(s);
    try sectionRules(s);
    return s;
}

fn checkRowMap(s: *const Step, limits: Limits) Error!void {
    const h = s.header;
    if (s.get(.streams) != null and s.get(.rows) != null) return error.BothRowMaps;
    const map = s.rowMap();
    if ((if (map) |m| m.count else 0) != h.streams) return error.BadStreams;
    if (h.streams > limits.streams) return error.TooManyStreams;
    var sum: u64 = 0;
    if (s.get(.streams)) |m| for (0..m.count) |i| {
        const e = m.body[section.stream_size * i ..][0..section.stream_size];
        _ = std.enums.fromInt(section.Op, e[4]) orelse return error.BadOp;
        if (e[5] != 0 or e[6] != 0 or e[7] != 0) return error.BadReserved;
        const st = m.stream(i);
        switch (st.op) {
            .open => if (st.first != 0 or st.accepted != 0 or st.generation != 0) return error.BadOpen,
            .close, .cancel => if (st.rows != 0) return error.RowsOnClose,
            .@"continue" => {},
            .@"resume" => if (st.resume_at == 0 or st.resume_at > st.first) return error.BadResume,
        }
        if (st.op != .@"resume" and st.resume_at != 0) return error.BadResume;
        if (@as(u64, st.first) + st.rows > limits.context) return error.PastContext;
        sum += st.rows;
        if (sum > limits.rows) return error.TooManyRows;
    };
    if (s.get(.rows)) |m| for (0..m.count) |i| {
        sum += m.row(i).rows;
        if (sum > limits.rows) return error.TooManyRows;
    };
    if (sum != h.rows) return error.BadTotal;
}

fn kindRules(s: Step) Error!void {
    const h = s.header;
    switch (h.kind) {
        .state, .idle, .stop => if (h.rows != 0) return error.RowsOffRound,
        .round, .family => {},
    }
    if ((h.kind == .idle or h.kind == .stop) and h.sections != 0) return error.SectionsOffRound;
    if (h.kind == .round and h.streams == 0 and h.mode != .window) return error.BadMode; // a keep-alive runs no rows
}

fn sectionRules(s: Step) Error!void {
    const h = s.header;
    if (s.get(.tokens)) |x| if (x.count < h.rows) return error.BadTokens;
    if (s.get(.cuts)) |x| for (0..x.count) |i| {
        const c = x.word(i);
        if (c == 0 or c >= h.rows or (i > 0 and c <= x.word(i - 1))) return error.BadCut;
    };
    if (s.get(.sampling)) |x| for (0..x.count) |i| {
        const g = x.sampling(i);
        if (g.has.reserved != 0) return error.BadReserved;
        if ((!g.has.seed and g.seed != 0) or (!g.has.pos0 and g.pos0 != 0)) return error.BadSampling;
        if (get32(x.body, section.sampling_size * i + 60) != 0) return error.BadReserved;
    };
    if (s.get(.pages)) |x| {
        var it = x.patches();
        while (it.next()) |v| if (v.kept > v.n) return error.BadPages;
    }
    if (s.get(.snapshots)) |x| for (0..x.count) |i| {
        const e = x.body[section.snapshot_size * i ..][0..section.snapshot_size];
        const op = std.enums.fromInt(section.SnapOp, e[0]) orelse return error.BadSnapshotOp;
        if (op == .drop and get32(e, 4) != 0) return error.BadSnapshot;
        if (e[1] != 0 or e[2] != 0 or e[3] != 0 or get32(e, 20) != 0) return error.BadReserved;
    };
}
