const canon = @import("canon");
const exl3 = @import("core").exl3_layout;

pub const Share = struct { lo: u32, len: u32 };

pub fn share(width: u32, rank: usize, world: usize) !Share {
    if (world == 0 or rank >= world or width % 128 != 0 or width / 128 < world) return error.SplitNotOnBlocks;
    const blocks: usize = width / 128;
    const base = blocks / world;
    const extra = blocks % world;
    return .{ .lo = @intCast((rank * base + @min(rank, extra)) * 128), .len = @intCast((base + @intFromBool(rank < extra)) * 128) };
}

/// A rank's whole output groups by canon's rule: heads, `wq_b` columns and `wo_b` rows all follow them.
pub fn outputGroups(groups: u32, rank: usize, world: usize) !Share {
    if (world == 0 or world > 256 or rank >= world or groups < world) return error.SplitNotOnGroups;
    const ones: [256]u64 = @splat(1);
    var ranges: [256]canon.Range = undefined;
    canon.partition(groups, ones[0..world], ranges[0..world]);
    return .{ .lo = ranges[rank].begin, .len = ranges[rank].len() };
}

/// A rank's slice of `width` elements laid out by output group.
pub fn outputShare(width: u32, groups: u32, rank: usize, world: usize) !Share {
    if (groups == 0 or width % groups != 0) return error.SplitNotOnGroups;
    const g = try outputGroups(groups, rank, world);
    return .{ .lo = g.lo * (width / groups), .len = g.len * (width / groups) };
}

pub fn group(g: exl3.Group, split: exl3.Split, rank: u32, world: u32) !exl3.Part {
    if (split == .whole) return g.part(split, rank, world);
    return slice(g, split, try share(if (split == .row) g.k() else g.n(), rank, world), rank, world);
}

/// `group`, with the split following whole output groups and still on whole blocks.
pub fn outputGroup(g: exl3.Group, split: exl3.Split, groups: u32, rank: u32, world: u32) !exl3.Part {
    if (split == .whole) return g.part(split, rank, world);
    const s = try outputShare(if (split == .row) g.k() else g.n(), groups, rank, world);
    if (s.lo % 128 != 0 or s.len % 128 != 0) return error.SplitNotOnBlocks;
    return slice(g, split, s, rank, world);
}

fn slice(g: exl3.Group, split: exl3.Split, s: Share, rank: u32, world: u32) !exl3.Part {
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
