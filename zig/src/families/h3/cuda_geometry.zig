//! The H3 CUDA family's buffers that depend on the packed sequence: allocated again when its rows or tiles change.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const bounds = @import("cuda_bounds.zig");
const Model = h3.Model;
const head_dim = h3.head_dim;
const slot_tile = h3.slot_tile;

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
    buffers: [24]cuda.DeviceBuffer = undefined,
    used: usize = 0,
    y: u64 = 0,
    xn: u64 = 0,
    qkvc: u64 = 0,
    tq: u64 = 0,
    tk: [2]u64 = .{ 0, 0 },
    tv: [2]u64 = .{ 0, 0 },
    qp: u64 = 0,
    kp: [2]u64 = .{ 0, 0 },
    vp: [2]u64 = .{ 0, 0 },
    scores: u64 = 0,
    chosen: u64 = 0,
    coarse: u64 = 0,
    ya: u64 = 0,
    wide: u64 = 0,
    act: u64 = 0,
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

/// Buffers for `rows` packed rows in `tiles` routing tiles; `slot` is each row's padded slot on the device.
pub fn prepare(self: *Model, rows: usize, tiles: usize, prefix: usize, keep: usize, slot: u64, own0: usize, own1: usize, exchange: bool) !void {
    const g = &self.geometry;
    if (g.rows == rows and g.tiles == tiles and g.prefix == prefix and g.keep == keep and g.own0 == own0 and g.own1 == own1 and g.exchange == exchange) return;
    // every size, and the exchange region's, checked before anything is released or allocated
    const sz = try bounds.bytes(.{ .rows = rows, .tiles = tiles, .prefix = prefix, .keep = keep, .hidden = self.hidden, .inner = self.inner, .heads = self.heads, .mlp = self.mlp, .head_dim = head_dim, .slot_tile = slot_tile, .exchange = exchange });
    if (exchange) {
        // ranks exchange through the registered region, tile after tile, in the half of the block's parity
        const t = &self.tp.?;
        if (sz.region > t.region.bytes.len) {
            std.debug.print("tensorfold h3: the exchange region holds {d} MB, this sequence needs {d} MB\n", .{ t.region.bytes.len >> 20, sz.region >> 20 });
            return error.RegionTooSmall;
        }
    }
    release(self);
    const slots = tiles * slot_tile;
    g.y = try buffer(self, sz.stream, false);
    g.xn = try buffer(self, sz.stream, false);
    g.qkvc = try buffer(self, sz.qkvc, false);
    // slots past a tile's real rows are never written: they stay zero
    g.tq = try buffer(self, sz.tiled, true);
    g.tk = try pair(self, sz.tiled, true);
    g.tv = try pair(self, sz.tiled, true);
    g.qp = try buffer(self, sz.pooled, false);
    g.kp = try pair(self, sz.pooled, false);
    g.vp = try pair(self, sz.pooled, false);
    g.unit = sz.unit;
    g.half = sz.half;
    g.scores = try buffer(self, sz.scores, false);
    g.chosen = try buffer(self, sz.chosen, false);
    g.coarse = try buffer(self, sz.pooled, false);
    g.ya = try buffer(self, sz.ya, false);
    g.wide = try buffer(self, sz.wide, false);
    g.act = try buffer(self, sz.act, false);
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
    g.row_of = try buffer(self, sz.row_of, false);
    try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(row_of));
    g.all = try buffer(self, sz.all, false);
    try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(all));
    g.rows = rows;
    g.tiles = tiles;
    g.prefix = prefix;
    g.keep = keep;
    g.own0 = own0;
    g.own1 = own1;
    g.exchange = exchange;
}
