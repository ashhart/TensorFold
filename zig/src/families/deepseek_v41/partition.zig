const exl3 = @import("core").exl3_layout;

pub const Share = struct { lo: u32, len: u32 };

pub fn share(width: u32, rank: usize, world: usize) !Share {
    if (world == 0 or rank >= world or width % 128 != 0 or width / 128 < world) return error.SplitNotOnBlocks;
    const blocks: usize = width / 128;
    const base = blocks / world;
    const extra = blocks % world;
    return .{ .lo = @intCast((rank * base + @min(rank, extra)) * 128), .len = @intCast((base + @intFromBool(rank < extra)) * 128) };
}

pub fn group(g: exl3.Group, split: exl3.Split, rank: u32, world: u32) !exl3.Part {
    if (split == .whole) return g.part(split, rank, world);
    const s = try share(if (split == .row) g.k() else g.n(), rank, world);
    var p = try g.part(.whole, rank, world);
    const tiles = s.len / 16;
    if (split == .row) {
        p.group.k_tiles = tiles;
        const row = @as(u64, g.n_tiles) * g.tileBytes();
        p.trellis = .{ .offset = @as(u64, s.lo / 16) * row, .len = @as(u64, tiles) * row };
        p.suh = .{ .offset = 2 * @as(u64, s.lo), .len = 2 * @as(u64, s.len) };
    } else {
        p.group.n_tiles = tiles;
        p.trellis = .{ .offset = @as(u64, s.lo / 16) * g.tileBytes(), .len = @as(u64, tiles) * g.tileBytes(), .stride = @as(u64, g.n_tiles) * g.tileBytes(), .count = g.k_tiles };
        p.svh = .{ .offset = 2 * @as(u64, s.lo), .len = 2 * @as(u64, s.len) };
    }
    return p;
}
