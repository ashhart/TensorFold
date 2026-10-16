//! The H3 CUDA family's buffers that depend on the packed sequence: allocated again when its rows or tiles change.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
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
    release(self);
    const hidden = self.hidden;
    const inner = self.inner;
    const heads = self.heads;
    const slots = tiles * slot_tile;
    g.y = try buffer(self, rows * hidden * 2, false);
    g.xn = try buffer(self, rows * hidden * 2, false);
    g.qkvc = try buffer(self, rows * 4 * inner * 2, false);
    // slots past a tile's real rows are never written: they stay zero
    g.tq = try buffer(self, heads * slots * head_dim * 2, true);
    g.tk = try pair(self, heads * slots * head_dim * 2, true);
    g.tv = try pair(self, heads * slots * head_dim * 2, true);
    g.qp = try buffer(self, heads * tiles * head_dim * 4, false);
    g.kp = try pair(self, heads * tiles * head_dim * 4, false);
    g.vp = try pair(self, heads * tiles * head_dim * 4, false);
    if (exchange) {
        // ranks exchange through the registered region, tile after tile, in the half of the block's parity
        const t = &self.tp.?;
        g.unit = heads * (2 * slot_tile * head_dim * 2 + 2 * head_dim * 4);
        g.half = (tiles * g.unit + 4095) / 4096 * 4096;
        if (2 * g.half > t.region.bytes.len) {
            std.debug.print("tensorfold h3: the exchange region holds {d} MB, this sequence needs {d} MB\n", .{ t.region.bytes.len >> 20, (2 * g.half) >> 20 });
            return error.RegionTooSmall;
        }
    }
    g.scores = try buffer(self, heads * tiles * tiles * 4, false);
    g.chosen = try buffer(self, heads * @max(tiles - prefix, 1) * (prefix + keep) * 4, false);
    g.coarse = try buffer(self, heads * tiles * head_dim * 4, false);
    g.ya = try buffer(self, rows * inner * 2, false);
    g.wide = try buffer(self, rows * 2 * self.mlp * 2, false);
    g.act = try buffer(self, rows * self.mlp * 2, false);
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
