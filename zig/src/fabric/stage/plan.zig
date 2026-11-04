//! Rank 0's round plan (its streams in a fixed order and the round's chunk cuts) and the layout of a rows frame.
const std = @import("std");
const payload = @import("payload.zig");
const frame = @import("frame.zig");
const boot = @import("boot.zig");

const get32 = payload.get32;
const put32 = payload.put32;

/// The plan's fixed part, and each stream's entry after it; the cuts (4 bytes each) follow the streams.
pub const plan_fixed = 24;
pub const stream_size = 28;

/// What a round does with a stream: start it, extend it, end it, drop it, or resume it from a kept prompt state.
pub const Op = enum(u8) { open = 1, @"continue" = 2, close = 3, cancel = 4, @"resume" = 5 };

/// One stream's part of a round.
pub const Stream = struct {
    id: u32,
    op: Op,
    /// The position of the stream's first row this round, and its rows (0 for close and cancel).
    first: u32,
    rows: u32,
    /// The previous verify's accepted count: every stage rolls its KV and recurrent state back to it.
    accepted: u32 = 0,
    /// For `resume`, the kept prompt state's position (at most `first`); 0 for every other op.
    resume_at: u32 = 0,
    /// The slot's state generation (0 for `open`); a keep, a release or a restore advances it (see `sameGeneration`).
    generation: u32 = 0,
};

/// A stage refuses a stream whose generation is not its slot's, so a missed rollback fails by name, not as wrong rows.
pub fn sameGeneration(own: u32, s: Stream) error{OtherGeneration}!void {
    if (s.generation != own) return error.OtherGeneration;
}

/// What a stage takes in one plan; from its shapes.
pub const Limits = struct {
    streams: u32,
    rows: u32,
    context: u32,

    pub fn of(s: boot.Shapes) Limits {
        return .{ .streams = s.parallel, .rows = s.max_rows, .context = s.context };
    }
};

pub const Error = payload.Error || error{
    TooManyStreams,
    UnorderedStreams,
    BadOp,
    BadOpen,
    BadResume,
    RowsOnClose,
    PastContext,
    TooManyRows,
    BadTotal,
    BadCut,
};

pub fn planSize(streams: usize, cuts: usize) usize {
    return plan_fixed + stream_size * streams + 4 * cuts;
}

/// The plan's bytes at the start of `out`, unchecked: `encode` checks them, the fixtures plant bad ones with this.
pub fn write(streams: []const Stream, cuts: []const u32, out: []u8) payload.Error![]u8 {
    const n = planSize(streams.len, cuts.len);
    if (out.len < n) return error.NoRoom;
    const p = out[0..n];
    @memset(p, 0);
    payload.put(p, .plan);
    put32(p, 8, @intCast(streams.len));
    put32(p, 12, @intCast(cuts.len));
    var total: u32 = 0;
    for (streams, 0..) |s, i| {
        const at = plan_fixed + stream_size * i;
        put32(p, at, s.id);
        p[at + 4] = @backingInt(s.op);
        put32(p, at + 8, s.first);
        put32(p, at + 12, s.rows);
        put32(p, at + 16, s.accepted);
        put32(p, at + 20, s.resume_at);
        put32(p, at + 24, s.generation);
        total +%= s.rows;
    }
    put32(p, 16, total);
    for (cuts, 0..) |c, i| put32(p, plan_fixed + stream_size * streams.len + 4 * i, c);
    return p;
}

/// The plan's bytes at the start of `out`, refused unless a stage with `limits` would take them.
pub fn encode(streams: []const Stream, cuts: []const u32, limits: Limits, out: []u8) Error![]u8 {
    if (streams.len > limits.streams) return error.TooManyStreams;
    const p = try write(streams, cuts, out);
    _ = try decode(p, limits);
    return p;
}

/// A checked plan: a view of its payload, nothing copied or allocated.
pub const Plan = struct {
    bytes: []const u8,
    streams: u32,
    cuts: u32,
    /// The rows the round's rows frames carry: the sum of every stream's rows.
    rows: u32,

    pub fn stream(p: Plan, i: usize) Stream {
        const at = plan_fixed + stream_size * i;
        const b = p.bytes;
        return .{
            .id = get32(b, at),
            .op = @fromBackingInt(@intCast(b[at + 4])),
            .first = get32(b, at + 8),
            .rows = get32(b, at + 12),
            .accepted = get32(b, at + 16),
            .resume_at = get32(b, at + 20),
            .generation = get32(b, at + 24),
        };
    }

    pub fn cut(p: Plan, i: usize) u32 {
        return get32(p.bytes, plan_fixed + stream_size * @as(usize, p.streams) + 4 * i);
    }

    /// A plan with no streams: the keep-alive.
    pub fn keepAlive(p: Plan) bool {
        return p.streams == 0;
    }
};

/// A plan from `p`, every count checked against its length before any read, and every stream against `limits`.
pub fn decode(p: []const u8, limits: Limits) Error!Plan {
    try payload.expect(p, .plan);
    if (p.len < plan_fixed) return error.BadLength;
    const n: u64 = get32(p, 8);
    const c: u64 = get32(p, 12);
    if (p.len != plan_fixed + stream_size * n + 4 * c) return error.BadLength;
    if (get32(p, 20) != 0) return error.BadReserved;
    if (n > limits.streams) return error.TooManyStreams;
    const plan: Plan = .{ .bytes = p, .streams = @intCast(n), .cuts = @intCast(c), .rows = get32(p, 16) };
    var sum: u64 = 0;
    for (0..plan.streams) |i| {
        const at = plan_fixed + stream_size * i;
        if (p[at + 5] != 0 or p[at + 6] != 0 or p[at + 7] != 0) return error.BadReserved;
        _ = std.enums.fromInt(Op, p[at + 4]) orelse return error.BadOp;
        const s = plan.stream(i);
        if (i > 0 and s.id <= plan.stream(i - 1).id) return error.UnorderedStreams;
        switch (s.op) {
            .open => if (s.first != 0 or s.accepted != 0 or s.generation != 0) return error.BadOpen,
            .close, .cancel => if (s.rows != 0) return error.RowsOnClose,
            .@"continue" => {},
            .@"resume" => if (s.resume_at == 0 or s.resume_at > s.first) return error.BadResume,
        }
        if (s.op != .@"resume" and s.resume_at != 0) return error.BadResume;
        if (@as(u64, s.first) + s.rows > limits.context) return error.PastContext;
        sum += s.rows;
        if (sum > limits.rows) return error.TooManyRows;
    }
    if (sum != plan.rows) return error.BadTotal;
    for (0..plan.cuts) |i| {
        const at = plan.cut(i);
        if (at == 0 or at >= plan.rows or (i > 0 and at <= plan.cut(i - 1))) return error.BadCut;
    }
    return plan;
}

pub const RowsError = error{ BadRowDtype, TooLarge, WrongKind, OtherRowCount, OtherRowBytes };

/// A rows payload: `planes` × `rows` × `hidden` values, plane-major in plan order, then `rows` × `draft_width` fp32.
pub const Rows = struct {
    rows: u32,
    boundary: boot.Boundary,
    row_bytes: u64,
    plane_bytes: u64,
    taps_bytes: u64,

    pub fn of(b: boot.Boundary, rows: u32) RowsError!Rows {
        const el = b.row_dtype.bytes() orelse return error.BadRowDtype;
        const row = @as(u64, b.hidden) * el;
        const plane = std.math.mul(u64, row, rows) catch return error.TooLarge;
        const taps = std.math.mul(u64, @as(u64, b.draft_width) * 4, rows) catch return error.TooLarge;
        const planes = std.math.mul(u64, plane, b.planes) catch return error.TooLarge;
        _ = std.math.add(u64, planes, taps) catch return error.TooLarge;
        return .{ .rows = rows, .boundary = b, .row_bytes = row, .plane_bytes = plane, .taps_bytes = taps };
    }

    /// Where row `row` of plane `plane` starts.
    pub fn rowOffset(r: Rows, plane: u32, row: u32) u64 {
        std.debug.assert(plane < r.boundary.planes and row < r.rows);
        return r.plane_bytes * plane + r.row_bytes * row;
    }

    /// Where the tap partial sums start (the end of the last plane), and where row `row`'s sum starts.
    pub fn tapsOffset(r: Rows) u64 {
        return r.plane_bytes * r.boundary.planes;
    }

    pub fn tapOffset(r: Rows, row: u32) u64 {
        std.debug.assert(row < r.rows and r.boundary.draft_width > 0);
        return r.tapsOffset() + @as(u64, r.boundary.draft_width) * 4 * row;
    }

    pub fn total(r: Rows) u64 {
        return r.tapsOffset() + r.taps_bytes;
    }
};

/// A received rows frame's header against the round's plan and the boundary: its rows and its bytes.
pub fn checkRows(h: frame.Header, p: Plan, b: boot.Boundary) RowsError!Rows {
    if (h.kind != .rows) return error.WrongKind;
    if (h.rows != p.rows) return error.OtherRowCount;
    const r = try Rows.of(b, p.rows);
    if (h.bytes != r.total()) return error.OtherRowBytes;
    return r;
}
