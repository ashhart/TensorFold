//! The hand-off's neutral state: named bf16/f32 parts (swa, carry, ring, comp, ixk) both ends derive from one `Shape`.
const std = @import("std");
const g = @import("geometry.zig");

pub const chunk_bytes: u64 = 2 << 20;
pub const max_world = 8;

pub const Dtype = enum {
    bf16,
    f32,
    pub fn size(d: Dtype) u64 {
        return if (d == .f32) 4 else 2;
    }
};

pub const Kind = enum { swa, carry, ring, comp, ixk };

pub const Spec = struct {
    layer: u32,
    kind: Kind,
    /// a comp / ixk part of a KV_SPLIT shard: its pages are the window's residue class
    split: bool = false,
    first_page: u32 = 0,
    rows: u32,
    cols: u32,
    dtype: Dtype,

    pub fn bytes(s: Spec) u64 {
        return @as(u64, s.rows) * s.cols * s.dtype.size();
    }

    pub fn name(s: Spec, buf: *[40]u8) []const u8 {
        const base = @tagName(s.kind);
        return switch (s.kind) {
            .comp, .ixk => std.fmt.bufPrint(buf, "{s}{s}@{d}", .{ base, if (s.split) ".split" else "", s.first_page }) catch unreachable,
            else => std.fmt.bufPrint(buf, "{s}", .{base}) catch unreachable,
        };
    }
};

pub const Parsed = struct { kind: Kind, split: bool, first_page: u32 };

pub fn parseName(name: []const u8) ?Parsed {
    var it = std.mem.splitScalar(u8, name, '@');
    const head = it.next().?;
    const page = it.next();
    if (it.next() != null) return null;
    var split = false;
    var kind_name = head;
    if (std.mem.endsWith(u8, head, ".split")) {
        split = true;
        kind_name = head[0 .. head.len - ".split".len];
    }
    const kind = std.meta.stringToEnum(Kind, kind_name) orelse return null;
    const paged = kind == .comp or kind == .ixk;
    if (paged != (page != null) or (split and !paged)) return null;
    const first = if (page) |p| (std.fmt.parseInt(u32, p, 10) catch return null) else 0;
    if (page) |p| if (p.len > 1 and p[0] == '0') return null; // one spelling a page
    return .{ .kind = kind, .split = split, .first_page = first };
}

pub const Shape = struct {
    geo: g.Geometry = g.release,
    /// prompt tokens the state covers (the next position to decode)
    upto: u64,
    /// the producer this export is: shard `shard` of `shards` (1: whole)
    shard: u32 = 0,
    shards: u32 = 1,
    /// TF_DSV41_KV_SPLIT's residue sharding of the comp / index pages over the `shards` producers
    split: bool = false,
    /// the DSpark blocks' valid key rows (0: none ride)
    ds_rows: u32 = 0,
    /// the part size both ends agree on (the format's constant; tests shrink it to cut many windows from small states)
    chunk: u64 = chunk_bytes,

    pub fn check(s: Shape) error{ BadGeometry, BadShape }!void {
        try s.geo.check();
        if (s.upto == 0 or s.upto > 1 << 20 or s.chunk == 0) return error.BadShape;
        if (s.shards == 0 or s.shards > max_world or s.shard >= s.shards) return error.BadShape;
        if (s.split != (s.shards > 1)) return error.BadShape;
        if (s.ds_rows > @min(s.upto, s.geo.window)) return error.BadShape;
    }

    pub fn windowRows(s: Shape) u32 {
        return @intCast(@min(s.upto, s.geo.window));
    }

    pub fn owns(s: Shape, page: u64) bool {
        return !s.split or page % s.shards == s.shard;
    }

    pub fn pages(s: Shape, source: u32) u64 {
        const pr = s.geo.pageRows(source);
        return (s.geo.rowsOf(source, s.upto) + pr - 1) / pr;
    }

    pub fn cols(s: Shape, kind: Kind) u32 {
        return switch (kind) {
            .ixk => s.geo.index_dim,
            else => s.geo.head_dim,
        };
    }

    /// Logical pages a comp / ixk part's window spans: `chunk_bytes` of this shard's pages.
    pub fn span(s: Shape, source: u32, kind: Kind) u32 {
        const per = s.chunk / (@as(u64, s.geo.pageRows(source)) * s.cols(kind) * 2);
        return @as(u32, @intCast(@max(1, per))) * if (s.split) s.shards else 1;
    }
};

/// One owned logical page of a part: rows [row, row + n) of the source.
pub const Run = struct { page: u32, row: u64, n: u32 };

pub const Pages = struct {
    shape: Shape,
    source: u32,
    page: u64,
    end: u64,

    pub fn next(p: *Pages) ?Run {
        const pr: u64 = p.shape.geo.pageRows(p.source);
        const rows = p.shape.geo.rowsOf(p.source, p.shape.upto);
        while (p.page < p.end) {
            const pg = p.page;
            p.page += 1;
            if (!p.shape.owns(pg)) continue;
            return .{ .page = @intCast(pg), .row = pg * pr, .n = @intCast(@min(pr, rows - pg * pr)) };
        }
        return null;
    }
};

/// The pages of the part starting at `first_page`; null when that is not a part boundary of this shape.
pub fn partPages(shape: Shape, source: u32, kind: Kind, first_page: u32) ?Pages {
    const span = shape.span(source, kind);
    const total = shape.pages(source);
    if (first_page % span != 0 or first_page >= total) return null;
    return .{ .shape = shape, .source = source, .page = first_page, .end = @min(total, @as(u64, first_page) + span) };
}

pub fn partRows(shape: Shape, source: u32, kind: Kind, first_page: u32) u32 {
    var it = partPages(shape, source, kind, first_page) orelse return 0;
    var n: u32 = 0;
    while (it.next()) |r| n += r.n;
    return n;
}

/// The export's parts in model order (layer, then swa, carry, comp pages, ixk pages), owned by the caller's allocator.
pub fn specs(gpa: std.mem.Allocator, shape: Shape) ![]Spec {
    try shape.check();
    var out: std.ArrayList(Spec) = .empty;
    errdefer out.deinit(gpa);
    const geo = shape.geo;
    const lead = shape.shard == 0;
    var layer: u32 = 0;
    while (layer < geo.layers) : (layer += 1) {
        if (lead) try out.append(gpa, .{ .layer = layer, .kind = .swa, .rows = shape.windowRows(), .cols = geo.head_dim, .dtype = .bf16 });
        const src = geo.sourceIndex(layer) orelse continue;
        if (lead and geo.sources[src].ratio == 2) try out.append(gpa, .{ .layer = layer, .kind = .carry, .rows = 2, .cols = geo.head_dim, .dtype = .f32 });
        inline for (.{ Kind.comp, Kind.ixk }) |kind| {
            const span = shape.span(src, kind);
            var first: u32 = 0;
            while (first < shape.pages(src)) : (first += span) {
                const rows = partRows(shape, src, kind, first);
                if (rows > 0) try out.append(gpa, .{ .layer = layer, .kind = kind, .split = shape.split, .first_page = first, .rows = rows, .cols = shape.cols(kind), .dtype = .bf16 });
            }
        }
    }
    if (lead and shape.ds_rows > 0) {
        for (0..geo.mtp_layers) |i| try out.append(gpa, .{ .layer = geo.layers + @as(u32, @intCast(i)), .kind = .ring, .rows = shape.ds_rows, .cols = geo.head_dim, .dtype = .bf16 });
    }
    return out.toOwnedSlice(gpa);
}

/// The spec of one part a peer names, or null when this shape has no such part (wrong layer, page, shard, rows).
pub fn find(shape: Shape, layer: u32, name: []const u8) ?Spec {
    const p = parseName(name) orelse return null;
    const geo = shape.geo;
    const lead = shape.shard == 0;
    switch (p.kind) {
        .swa => if (lead and layer < geo.layers) return .{ .layer = layer, .kind = .swa, .rows = shape.windowRows(), .cols = geo.head_dim, .dtype = .bf16 },
        .carry => if (lead) if (geo.sourceIndex(layer)) |s| if (geo.sources[s].ratio == 2) return .{ .layer = layer, .kind = .carry, .rows = 2, .cols = geo.head_dim, .dtype = .f32 },
        .ring => if (lead and shape.ds_rows > 0 and layer >= geo.layers and layer < geo.layers + geo.mtp_layers) return .{ .layer = layer, .kind = .ring, .rows = shape.ds_rows, .cols = geo.head_dim, .dtype = .bf16 },
        .comp, .ixk => {
            const s = geo.sourceIndex(layer) orelse return null;
            if (p.split != shape.split) return null;
            const rows = partRows(shape, s, p.kind, p.first_page);
            if (rows == 0) return null;
            return .{ .layer = layer, .kind = p.kind, .split = p.split, .first_page = p.first_page, .rows = rows, .cols = shape.cols(p.kind), .dtype = .bf16 };
        },
    }
    return null;
}
