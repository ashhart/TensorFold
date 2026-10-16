//! The H3 CUDA family's buffers that depend on the packed sequence: allocated again when its rows or tiles change.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Model = h3.Model;
const head_dim = h3.head_dim;
const slot_tile = h3.slot_tile;

/// Rows are padded to whole product tiles of this many.
pub const tile = 256;

/// What depends on the packed sequence; keys, values and means are indexed by block parity like the region's halves.
pub const Geometry = struct {
    rows: usize = 0,
    tiles: usize = 0,
    prefix: usize = 0,
    keep: usize = 0,
    /// This rank's tiles, [own0, own1); all of them without ranks.
    own0: usize = 0,
    own1: usize = 0,
    exchange: bool = false,
    /// In the registered region: bytes a tile takes, and the bytes of one parity's half.
    unit: usize = 0,
    half: usize = 0,
    buffers: [40]cuda.DeviceBuffer = undefined,
    used: usize = 0,
    y: u64 = 0,
    q8: u64 = 0,
    xs: u64 = 0,
    qkvc: u64 = 0,
    tq: u64 = 0,
    tqs: u64 = 0,
    tk: [2]u64 = .{ 0, 0 },
    tvt: [2]u64 = .{ 0, 0 },
    ks: [2]u64 = .{ 0, 0 },
    vs: [2]u64 = .{ 0, 0 },
    qp: [2]u64 = .{ 0, 0 },
    kp: [2]u64 = .{ 0, 0 },
    vp: [2]u64 = .{ 0, 0 },
    scores: u64 = 0,
    chosen: u64 = 0,
    coarse: u64 = 0,
    // the tiles' means as a sequence of their own, in groups of 64 tiles
    pq: u64 = 0,
    pqs: u64 = 0,
    pk: u64 = 0,
    pks: u64 = 0,
    pvt: u64 = 0,
    pvs: u64 = 0,
    group_sizes: u64 = 0,
    group_rows: u64 = 0,
    ya: u64 = 0,
    a8: u64 = 0,
    as: u64 = 0,
    wide: u64 = 0,
    w8: u64 = 0,
    wxs: u64 = 0,
    row_of: u64 = 0,
    all: u64 = 0,
};

pub fn release(self: *Model) void {
    const g = &self.geometry;
    for (g.buffers[0..g.used]) |*b| b.free();
    g.* = .{};
}

fn buffer(self: *Model, bytes: usize, zero: bool) !u64 {
    const g = &self.geometry;
    g.buffers[g.used] = try cuda.DeviceBuffer.alloc(&self.driver, bytes);
    g.used += 1;
    if (zero) try g.buffers[g.used - 1].fill8(0, null);
    return g.buffers[g.used - 1].ptr;
}

/// One buffer for both parities.
fn pair(self: *Model, bytes: usize, zero: bool) ![2]u64 {
    return @splat(try buffer(self, bytes, zero));
}

pub fn padded(rows: usize) usize {
    return (rows + tile - 1) / tile * tile;
}

/// Buffers for `rows` packed rows in `tiles` routing tiles; `slot` is each row's padded slot on the device.
pub fn prepare(self: *Model, rows: usize, tiles: usize, prefix: usize, keep: usize, slot: u64, own0: usize, own1: usize, exchange: bool) !void {
    const g = &self.geometry;
    if (g.rows == rows and g.tiles == tiles and g.prefix == prefix and g.keep == keep and g.own0 == own0 and g.own1 == own1 and g.exchange == exchange) return;
    release(self);
    const hidden = self.hidden;
    const inner = self.inner;
    const heads = self.heads;
    const slots = tiles * slot_tile;
    const mp = padded(rows);
    g.y = try buffer(self, rows * hidden * 2, false);
    g.q8 = try buffer(self, mp * hidden, false);
    g.xs = try buffer(self, mp * 4, false);
    g.qkvc = try buffer(self, rows * 4 * inner * 2, false);
    // slots past a tile's real rows are never written: they stay zero
    g.tq = try buffer(self, heads * slots * head_dim, true);
    g.tqs = try buffer(self, heads * slots * 4, true);
    g.tk = try pair(self, heads * slots * head_dim, true);
    g.tvt = try pair(self, heads * slots * head_dim, true);
    g.ks = try pair(self, heads * tiles * 4, false);
    g.vs = try pair(self, heads * tiles * 4, false);
    g.qp = try pair(self, heads * tiles * head_dim * 4, false);
    g.kp = try pair(self, heads * tiles * head_dim * 4, false);
    g.vp = try pair(self, heads * tiles * head_dim * 4, false);
    if (exchange) {
        // ranks exchange through the registered region, tile after tile, in the half of the block's parity
        const t = &self.tp.?;
        g.unit = heads * (2 * slot_tile * head_dim + 2 * 4 + 3 * head_dim * 4);
        g.half = (tiles * g.unit + 4095) / 4096 * 4096;
        if (2 * g.half > t.region.bytes.len) {
            std.debug.print("tensorfold h3: the exchange region holds {d} MB, this sequence needs {d} MB\n", .{ t.region.bytes.len >> 20, (2 * g.half) >> 20 });
            return error.RegionTooSmall;
        }
    }
    const groups = (tiles + slot_tile - 1) / slot_tile;
    const span = groups * slot_tile;
    g.scores = try buffer(self, heads * span * span * 4, false);
    g.chosen = try buffer(self, heads * @max(tiles - prefix, 1) * (prefix + keep) * 4, false);
    g.coarse = try buffer(self, tiles * inner * 2, false);
    g.pq = try buffer(self, heads * span * head_dim, false);
    g.pqs = try buffer(self, heads * span * 4, false);
    g.pk = try buffer(self, heads * span * head_dim, false);
    g.pks = try buffer(self, heads * groups * 4, false);
    g.pvt = try buffer(self, heads * span * head_dim, false);
    g.pvs = try buffer(self, heads * groups * 4, false);
    {
        // each group's real tiles, and the tile at each of its places (or -1)
        const sizes = try self.gpa.alloc(i32, groups);
        defer self.gpa.free(sizes);
        for (sizes, 0..) |*n, i| n.* = @intCast(@min(slot_tile, tiles - i * slot_tile));
        const places = try self.gpa.alloc(i32, span);
        defer self.gpa.free(places);
        for (places, 0..) |*t, i| t.* = if (i < tiles) @intCast(i) else -1;
        g.group_sizes = try buffer(self, groups * 4, false);
        try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(sizes));
        g.group_rows = try buffer(self, span * 4, false);
        try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(places));
    }
    g.ya = try buffer(self, rows * inner * 2, false);
    g.a8 = try buffer(self, mp * inner, false);
    g.as = try buffer(self, mp * 4, false);
    g.wide = try buffer(self, rows * self.mlp * 2, false);
    g.w8 = try buffer(self, mp * self.mlp, false);
    g.wxs = try buffer(self, mp * (self.mlp / h3.wide_group) * 4, false);
    // the row at each slot (or -1), and the list of every tile
    const row_of = try self.gpa.alloc(i32, slots);
    defer self.gpa.free(row_of);
    @memset(row_of, -1);
    const slot_host = try self.gpa.alloc(i32, rows);
    defer self.gpa.free(slot_host);
    try self.driver.check(self.driver.api.cuMemcpyDtoH_v2(@ptrCast(slot_host.ptr), slot, rows * 4), "cuMemcpyDtoH");
    for (slot_host, 0..) |at, row| {
        if (at < 0 or @as(usize, @intCast(at)) >= slots) return error.BadGeometry;
        // a rank's rows are the rows of its own tiles
        const at_tile = @as(usize, @intCast(at)) / slot_tile;
        if (at_tile < own0 or at_tile >= own1) return error.BadGeometry;
        row_of[@intCast(at)] = @intCast(row);
    }
    const all = try self.gpa.alloc(i32, tiles);
    defer self.gpa.free(all);
    for (all, 0..) |*t, i| t.* = @intCast(i);
    g.row_of = try buffer(self, slots * 4, false);
    try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(row_of));
    g.all = try buffer(self, tiles * 4, false);
    try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(all));
    g.rows = rows;
    g.tiles = tiles;
    g.prefix = prefix;
    g.keep = keep;
    g.own0 = own0;
    g.own1 = own1;
    g.exchange = exchange;
}
