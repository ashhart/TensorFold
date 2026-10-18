//! The H3 CUDA family's buffers that depend on the packed sequence: allocated again when its rows or tiles change.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Model = h3.Model;
const head_dim = h3.head_dim;
const slot_tile = h3.slot_tile;

/// What depends on the packed sequence; keys, values and means reuse the same storage across block parity.
pub const Geometry = struct {
    rows: usize = 0,
    tiles: usize = 0,
    prefix: usize = 0,
    keep: usize = 0,
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
    self.drain() catch |e| {
        std.debug.print("H3 geometry retained: {t}\n", .{e});
        return;
    };
    releaseCompleted(self);
}
pub fn releaseCompleted(self: *Model) void {
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

/// Admission precedes this call. Drain before any replacement, overwrite or device packing read.
pub fn prepare(self: *Model, in: *const Model.Inputs, p: @import("cuda_validation.zig").Plan) !void {
    try self.drain();
    const row_of = try @import("cuda_packing.zig").read(self, in, p);
    defer self.gpa.free(row_of);
    const g = &self.geometry;
    const same = g.rows == in.rows and g.tiles == in.tiles and g.prefix == p.prefix and g.keep == p.keep;
    if (same) {
        const map: cuda.DeviceBuffer = .{ .d = &self.driver, .ptr = g.row_of, .len = row_of.len * 4, .borrowed = true };
        try map.upload(0, std.mem.sliceAsBytes(row_of));
        try self.drain();
        return;
    }
    releaseCompleted(self);
    g.y = try buffer(self, p.buffers[0], false);
    g.xn = try buffer(self, p.buffers[1], false);
    g.qkvc = try buffer(self, p.buffers[2], false);
    g.tq = try buffer(self, p.buffers[3], true);
    g.tk = try pair(self, p.buffers[4], true);
    g.tv = try pair(self, p.buffers[5], true);
    g.qp = try buffer(self, p.buffers[6], false);
    g.kp = try pair(self, p.buffers[7], false);
    g.vp = try pair(self, p.buffers[8], false);
    g.scores = try buffer(self, p.buffers[9], false);
    g.chosen = try buffer(self, p.buffers[10], false);
    g.coarse = try buffer(self, p.buffers[11], false);
    g.ya = try buffer(self, p.buffers[12], false);
    g.wide = try buffer(self, p.buffers[13], false);
    g.act = try buffer(self, p.buffers[14], false);
    g.row_of = try buffer(self, p.buffers[15], false);
    g.all = try buffer(self, p.buffers[16], false);
    try g.buffers[g.used - 2].upload(0, std.mem.sliceAsBytes(row_of));
    const all = try self.gpa.alloc(i32, in.tiles);
    defer self.gpa.free(all);
    for (all, 0..) |*t, i| t.* = @intCast(i);
    try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(all));
    g.rows = in.rows;
    g.tiles = in.tiles;
    g.prefix = p.prefix;
    g.keep = p.keep;
}
