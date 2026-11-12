//! A pool's groups sized by bytes: a kind's bytes a row, the rows one page's bytes hold of it, and what it pads.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// What a group caches: every row, a sliding window's rows, a stream's fixed state, a drafter's rows.
pub const Kind = enum { full, window, state, draft };

/// One kind's bytes: a row in one layer and the layers holding it (a state: a stream's bytes, one layer).
pub const Group = struct {
    kind: Kind,
    row: u64,
    layers: u32 = 1,

    /// Bytes one row adds over every layer of the group (a state: a stream's bytes).
    pub fn bytes(g: Group) u64 {
        return g.row * g.layers;
    }

    /// A group of `layers` layers whose rows add `total` bytes over all of them; null when they don't split evenly.
    pub fn split(kind: Kind, total: u64, layers: u32) ?Group {
        if (layers == 0 or total % layers != 0) return null;
        return .{ .kind = kind, .row = total / layers, .layers = layers };
    }
};

/// A group on pages: whole rows a page holds (0 for a state), the pages it takes, and the bytes they leave unused.
pub const Fit = struct { rows: u64, pages: u64, pad: u64 };

/// Rows of `row` bytes in a page of `page` bytes, and the tail past them.
pub fn rows(row: u64, page: u64) Fit {
    const n = page / row;
    return .{ .rows = n, .pages = 1, .pad = page - n * row };
}

/// Whole pages of `page` bytes that hold `bytes`, and the tail of the last one.
pub fn state(bytes: u64, page: u64) Fit {
    const n = (bytes + page - 1) / page;
    return .{ .rows = 0, .pages = n, .pad = n * page - bytes };
}

/// What the load report reads: the pool's group and page, and the kinds beside it in or outside the pool.
pub const Shape = struct {
    full: Group,
    named: bool = true, // the family names the full group's layers; false: its row is a token's bytes over all of them
    page_rows: u32,
    window: ?Group = null, // rows outside the pool: a sliding window's rings
    run: u64 = 0, // a stream's state as a run of the pool's pages
    snap: u64 = 0, // a kept snapshot's slot outside the pool

    /// One layer's share of a page: the rows of the full group's layer a page holds.
    pub fn layerPage(s: Shape) u64 {
        return s.full.row * s.page_rows;
    }

    pub fn pageBytes(s: Shape) u64 {
        return s.full.bytes() * s.page_rows;
    }
};

/// The load report: a line of each group's bytes a row, rows a page and padding, apart from the startup line.
pub fn line(a: Allocator, s: Shape) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    errdefer w.deinit();
    const o = &w.writer;
    const page = s.pageBytes();
    try o.print("KV page groups: one page {f}, {d} rows of the full group; full: ", .{ size(page), s.page_rows });
    try rowsPart(o, s.full, s);
    if (s.window) |g| {
        try o.writeAll("; window, outside the pool: ");
        try rowsPart(o, g, s);
    }
    if (s.run > 0) try statePart(o, "state run, in the pool", s.run, page);
    if (s.snap > 0) try statePart(o, "snapshot, outside the pool", s.snap, page);
    return w.toOwnedSlice();
}

fn rowsPart(o: *std.Io.Writer, g: Group, s: Shape) !void {
    const f = rows(g.row, s.layerPage());
    if (g.kind == .full and !s.named) {
        return o.print("layers not named, {f} a token, {d} rows a page, {f} padding ({d:.1} %)", .{ size(g.row), f.rows, size(f.pad), pct(f.pad, s.layerPage()) });
    }
    try o.print("{d} layer{s} x {f} a row", .{ g.layers, if (g.layers == 1) "" else "s", size(g.row) });
    if (g.kind != .full) try o.print(" ({d:.2}x a full row)", .{ratio(g.row, s.full.row)});
    try o.print(", {d} rows a layer's page, {f} padding ({d:.1} %)", .{ f.rows, size(f.pad), pct(f.pad, s.layerPage()) });
}

fn statePart(o: *std.Io.Writer, what: []const u8, bytes: u64, page: u64) !void {
    const f = state(bytes, page);
    try o.print("; {s}: {f} on {d} page{s}, {f} padding ({d:.1} %)", .{ what, size(bytes), f.pages, if (f.pages == 1) "" else "s", size(f.pad), pct(f.pad, f.pages * page) });
}

fn ratio(x: u64, y: u64) f64 {
    return @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(y));
}

fn pct(pad: u64, of: u64) f64 {
    return if (of == 0) 0 else 100 * ratio(pad, of);
}

/// Bytes as B, KiB or MiB.
const Size = struct {
    bytes: u64,
    pub fn format(z: Size, o: *std.Io.Writer) !void {
        if (z.bytes < 1 << 10) return o.print("{d} B", .{z.bytes});
        if (z.bytes < 1 << 20) return o.print("{d:.2} KiB", .{ratio(z.bytes, 1 << 10)});
        return o.print("{d:.2} MiB", .{ratio(z.bytes, 1 << 20)});
    }
};

fn size(bytes: u64) Size {
    return .{ .bytes = bytes };
}
