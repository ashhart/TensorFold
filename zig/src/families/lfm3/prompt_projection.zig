//! Explicit qualification candidate using the shared prompt tiles and their same-bit decode twin.
const std = @import("std");
const mtl = @import("metal");
const row = @import("../../core/row_projection.zig");
const tile = @import("../../core/tile_qmm.zig");
const affine = @import("../../core/affine_mm.zig");
const pairs = @import("row_projection.zig");
const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

pub const Candidate = struct {
    tiles: ?tile.Pipelines = null,
    twin: bool = false,
    affine32: ?[tile.bit_widths.len]affine.Pipes = null,

    pub fn init(device: mtl.Device, matrices: []const row.Weights) !Candidate {
        var out: Candidate = .{};
        errdefer out.deinit();
        const has64 = for (matrices) |w| {
            if (tiled(w)) break true;
        } else false;
        const has32 = for (matrices) |w| {
            if (w.group == 32 and w.n % 64 == 0) break true;
        } else false;
        if (has64) {
            out.tiles = try tile.Pipelines.load(device);
            out.twin = true;
            for (matrices, 0..) |w, i| {
                if (!tiled(w)) continue;
                const seen = for (matrices[0..i]) |prior| {
                    if (tiled(prior) and prior.n == w.n and prior.k == w.k and prior.bits == w.bits) break true;
                } else false;
                if (seen) continue;
                if (!try tile.twinExact(&out.tiles.?, device, matrix(w))) out.twin = false;
            }
        }
        if (has32 and device.tensorUnits()) out.affine32 = try loadAffine32(device);
        return out;
    }

    pub fn deinit(self: *Candidate) void {
        if (self.tiles) |*p| p.deinit();
        if (self.affine32) |set| for (set) |pipes| for (pipes) |p| p.deinit();
        self.* = .{};
    }

    pub fn encode(self: *const Candidate, e: mtl.ComputeEncoder, fallback: *const pairs.Pipelines, w: row.Weights, x: mtl.Buffer, x_off: usize, sums: mtl.Buffer, y: mtl.Buffer, y_off: usize, rows: usize) !void {
        try w.validate();
        if (w.group == 32 and w.n % 64 == 0 and w.sum == .f32) {
            if (self.affine32) |set| {
                const index = std.mem.indexOfScalar(u8, &tile.bit_widths, w.bits) orelse return error.UnsupportedLfmTileBits;
                const pipes = set[index];
                const input = Ref{ .buf = x, .off = x_off };
                const temporary = Ref{ .buf = sums };
                affine.rowSums(e, pipes, 32, input, temporary, @intCast(rows), @intCast(w.k));
                affine.dense(e, pipes, input, temporary, .{ .w = Ref{ .buf = w.w, .off = w.w_off }, .s = Ref{ .buf = w.scales, .off = w.s_off }, .b = Ref{ .buf = w.biases, .off = w.b_off }, .n = @as(u32, @intCast(w.n)), .k = @as(u32, @intCast(w.k)) }, Ref{ .buf = y, .off = y_off }, @intCast(rows));
                return;
            }
        }
        if (self.tiles) |*p| {
            if (tiled(w)) {
                if (rows <= tile.twin_rows and self.twin) {
                    try tile.encodeTwin(e, p, matrix(w), x, x_off, y, y_off, rows);
                } else {
                    tile.encodeSums(e, p, w.k, x, x_off, sums, rows);
                    try tile.encode(e, p, matrix(w), x, x_off, sums, 0, y, y_off, rows);
                }
                return;
            }
        }
        try pairs.encode(e, fallback, w, x, x_off, y, y_off, rows);
    }
};

fn tiled(w: row.Weights) bool {
    return w.group == 64 and w.sum == .f32 and w.n % 32 == 0 and w.k % 64 == 0 and w.k <= 12288 and
        std.mem.indexOfScalar(u8, &tile.bit_widths, w.bits) != null;
}

fn matrix(w: row.Weights) tile.Matrix {
    return .{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off, .n = w.n, .k = w.k, .bits = w.bits };
}

fn loadAffine32(device: mtl.Device) ![tile.bit_widths.len]affine.Pipes {
    const a = std.heap.c_allocator;
    var set: [tile.bit_widths.len]affine.Pipes = undefined;
    var made: usize = 0;
    errdefer for (0..made) |i| set[i / affine.names.len][i % affine.names.len].deinit();
    for (tile.bit_widths, 0..) |bits, i| {
        const source = try affine.source(device, a, .{ .bits = bits, .group = 32 });
        defer a.free(source);
        const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        for (affine.names, 0..) |name, j| {
            set[i][j] = try mtl.Pipeline.init(device, library, name, false);
            made += 1;
        }
    }
    return set;
}
