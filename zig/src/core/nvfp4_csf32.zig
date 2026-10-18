//! csf_32x2_v1 NVFP4 block scales, decoded and encoded on the host (local-inference-lab's docs/csf-scale-blocks.md).
const std = @import("std");

const Allocator = std.mem.Allocator;

/// Rows of a block; a block is two columns wide, the last of an odd column count one.
const block_rows = 32;
/// A tile is 4 block rows by 2 column pairs; a page indexes 32 tiles.
const tile_rows = 4;
const tile_pairs = 2;
const page_tiles = 32;
/// The highest NVFP4 base: base + 15 is still a byte.
pub const max_base = 240;

pub const Error = error{
    /// The parts' lengths disagree with an R x C matrix (R a positive multiple of 32, C positive).
    CsfShape,
    /// A row base above `max_base`.
    CsfBase,
    /// A page's prefixes not from 0 or decreasing, its origin not after the records before it, or records left over.
    CsfIndex,
    /// An exception outside its block, or not after the block's previous one.
    CsfRecord,
};

/// A scale matrix as stored, little-endian: u8 bases [R] and codes [R*C/2], u32 page_bases [P], u16 offsets [B+P].
pub const Parts = struct {
    bases: []const u8,
    codes: []const u8,
    page_bases: []const u8,
    offsets: []const u8,
    exceptions: []const u8,
};

const Geometry = struct {
    rows: usize,
    cols: usize,

    fn of(rows: usize, cols: usize) Error!Geometry {
        if (rows == 0 or rows % block_rows != 0 or cols == 0) return error.CsfShape;
        _ = std.math.mul(usize, rows, cols) catch return error.CsfShape;
        return .{ .rows = rows, .cols = cols };
    }

    fn blockRows(g: Geometry) usize {
        return g.rows / block_rows;
    }

    fn pairs(g: Geometry) usize {
        return (g.cols + 1) / 2;
    }

    fn blocks(g: Geometry) usize {
        return g.blockRows() * g.pairs();
    }

    fn pages(g: Geometry) usize {
        const tiles = (g.blockRows() + tile_rows - 1) / tile_rows * ((g.pairs() + tile_pairs - 1) / tile_pairs);
        return (tiles + page_tiles - 1) / page_tiles;
    }
};

/// A block: its first row and column, its width, its tile, and whether it opens a page.
const Block = struct { row: usize, col: usize, width: usize, tile: usize, opens: bool };

/// The blocks in stream order: tiles row-major; in a tile, the first column pair's block rows, then the second's.
const Blocks = struct {
    g: Geometry,
    row: usize = 0, // the tile's first block row
    pair: usize = 0, // the tile's first column pair
    at: usize = 0, // the block within the tile
    tile: usize = 0,

    fn next(it: *Blocks) ?Block {
        while (it.row < it.g.blockRows()) {
            const h: usize = @min(tile_rows, it.g.blockRows() - it.row);
            const w: usize = @min(tile_pairs, it.g.pairs() - it.pair);
            if (it.at < h * w) {
                const pair = it.pair + it.at / h;
                const b: Block = .{
                    .row = (it.row + it.at % h) * block_rows,
                    .col = 2 * pair,
                    .width = @min(2, it.g.cols - 2 * pair),
                    .tile = it.tile,
                    .opens = it.at == 0 and it.tile % page_tiles == 0,
                };
                it.at += 1;
                return b;
            }
            it.at = 0;
            it.tile += 1;
            it.pair += tile_pairs;
            if (it.pair >= it.g.pairs()) {
                it.pair = 0;
                it.row += tile_rows;
            }
        }
        return null;
    }
};

fn read16(b: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, b[2 * i ..][0..2], .little);
}

fn read32(b: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, b[4 * i ..][0..4], .little);
}

fn write16(b: []u8, i: usize, v: u16) void {
    std.mem.writeInt(u16, b[2 * i ..][0..2], v, .little);
}

fn write32(b: []u8, i: usize, v: u32) void {
    std.mem.writeInt(u32, b[4 * i ..][0..4], v, .little);
}

/// The R x C scale bytes `p` encodes, row-major into `out`: each offset added to its row's base, then the exceptions.
pub fn decode(p: Parts, rows: usize, cols: usize, out: []u8) Error!void {
    const g = try Geometry.of(rows, cols);
    const pages = g.pages();
    if (out.len != rows * cols or p.bases.len != rows or p.codes.len != rows * cols / 2 or p.page_bases.len != 4 * pages or
        p.offsets.len != 2 * (g.blocks() + pages) or p.exceptions.len % 2 != 0) return error.CsfShape;
    for (p.bases) |b| if (b > max_base) return error.CsfBase;
    const records = p.exceptions.len / 2;
    var field: usize = 0; // offsets read
    var entry: usize = 0; // the block's first prefix
    var origin: usize = 0; // the page's first record
    var end: usize = 0; // the page's records so far
    var it: Blocks = .{ .g = g };
    while (it.next()) |b| {
        if (b.opens) {
            if (b.tile != 0) {
                entry += 1; // past the closed page's terminal prefix
                origin += end;
            }
            if (read32(p.page_bases, b.tile / page_tiles) != 2 * origin or read16(p.offsets, entry) != 0) return error.CsfIndex;
        }
        for (0..block_rows) |r| for (0..b.width) |c| {
            const byte = p.codes[field / 2];
            const offset = if (field % 2 == 0) byte & 0xF else byte >> 4;
            out[(b.row + r) * cols + b.col + c] = p.bases[b.row + r] + offset;
            field += 1;
        };
        const lo = read16(p.offsets, entry);
        const hi = read16(p.offsets, entry + 1);
        if (hi < lo or origin + hi > records) return error.CsfIndex;
        var after: usize = 0; // the lowest position the next record may take
        for (origin + lo..origin + hi) |k| {
            const rec = read16(p.exceptions, k);
            const at = rec & 0xFF; // 2 * row + column within the block
            if (at < after or at / 2 >= block_rows or at % 2 >= b.width) return error.CsfRecord;
            after = at + 1;
            out[(b.row + at / 2) * cols + b.col + at % 2] = @intCast(rec >> 8);
        }
        end = hi;
        entry += 1;
    }
    if (origin + end != records) return error.CsfIndex;
}

/// One matrix's parts, each its own allocation.
pub const Encoded = struct {
    bases: []u8,
    codes: []u8,
    page_bases: []u8,
    offsets: []u8,
    exceptions: []u8,

    pub fn parts(e: Encoded) Parts {
        return .{ .bases = e.bases, .codes = e.codes, .page_bases = e.page_bases, .offsets = e.offsets, .exceptions = e.exceptions };
    }

    pub fn deinit(e: Encoded, a: Allocator) void {
        for ([_][]u8{ e.bases, e.codes, e.page_bases, e.offsets, e.exceptions }) |b| a.free(b);
    }
};

/// `scale` (R x C, row-major) encoded: a row's base starts its most covering 16-wide window, the lowest on a tie.
pub fn encode(a: Allocator, scale: []const u8, rows: usize, cols: usize) (Allocator.Error || Error)!Encoded {
    const g = try Geometry.of(rows, cols);
    if (scale.len != rows * cols) return error.CsfShape;
    const pages = g.pages();
    const bases = try a.alloc(u8, rows);
    errdefer a.free(bases);
    var count: usize = 0; // exceptions
    for (bases, 0..) |*base, r| {
        var hist: [256]usize = @splat(0);
        for (scale[r * cols ..][0..cols]) |v| hist[v] += 1;
        var cover: usize = 0;
        for (hist[0..16]) |n| cover += n;
        var best = cover;
        base.* = 0;
        for (1..max_base + 1) |b| {
            cover = cover + hist[b + 15] - hist[b - 1];
            if (cover > best) {
                best = cover;
                base.* = @intCast(b);
            }
        }
        count += cols - best;
    }
    if (count > std.math.maxInt(u32) / 2) return error.CsfShape; // the arena's byte origins are u32
    const codes = try a.alloc(u8, rows * cols / 2);
    errdefer a.free(codes);
    const page_bases = try a.alloc(u8, 4 * pages);
    errdefer a.free(page_bases);
    const offsets = try a.alloc(u8, 2 * (g.blocks() + pages));
    errdefer a.free(offsets);
    const exceptions = try a.alloc(u8, 2 * count);
    var field: usize = 0;
    var record: usize = 0;
    var entry: usize = 0;
    var origin: usize = 0;
    var it: Blocks = .{ .g = g };
    while (it.next()) |b| {
        if (b.opens) {
            if (b.tile != 0) entry += 1;
            origin = record;
            write32(page_bases, b.tile / page_tiles, @intCast(2 * origin));
            write16(offsets, entry, 0);
        }
        for (0..block_rows) |r| for (0..b.width) |c| {
            const v = scale[(b.row + r) * cols + b.col + c];
            const base = bases[b.row + r];
            const offset: u8 = if (v >= base and v - base < 16) v - base else miss: {
                write16(exceptions, record, @as(u16, v) << 8 | @as(u16, @intCast(2 * r + c)));
                record += 1;
                break :miss 0;
            };
            if (field % 2 == 0) codes[field / 2] = offset else codes[field / 2] |= offset << 4;
            field += 1;
        };
        entry += 1;
        write16(offsets, entry, @intCast(record - origin)); // a page holds at most 16384 scales
    }
    return .{ .bases = bases, .codes = codes, .page_bases = page_bases, .offsets = offsets, .exceptions = exceptions };
}

/// The decode tests' block: R 32 x C 2, base 10, offset (r + c) % 16, exceptions 200 at (0, 1) and 3 at (31, 0).
const Hand = struct {
    bases: [32]u8 = @splat(10),
    codes: [32]u8 = codes: {
        var c: [32]u8 = undefined;
        for (0..32) |r| c[r] = r % 16 | (r + 1) % 16 << 4; // row r's two offsets, column 0 in the low nibble
        break :codes c;
    },
    page_bases: [4]u8 = @splat(0),
    offsets: [4]u8 = .{ 0, 0, 2, 0 },
    exceptions: [4]u8 = .{ 1, 200, 62, 3 },

    fn parts(h: *const Hand) Parts {
        return .{ .bases = &h.bases, .codes = &h.codes, .page_bases = &h.page_bases, .offsets = &h.offsets, .exceptions = &h.exceptions };
    }
};

/// A test matrix: a band per row with stray bytes; the golden's generator computes the same.
fn sample(i: usize, row: usize) u8 {
    var x: u32 = @as(u32, @truncate(i)) *% 0x9E3779B1 +% 0x7F4A7C15;
    x ^= x >> 16;
    x *%= 0x85EBCA6B;
    x ^= x >> 13;
    x *%= 0xC2B2AE35;
    x ^= x >> 16;
    if (x % 11 == 0) return @truncate(x >> 8);
    return @intCast(96 + (row * 37) % 64 + (x >> 4) % 18);
}

test "a block decodes as its row bases plus offsets, then its exceptions" {
    const h: Hand = .{};
    var out: [64]u8 = undefined;
    try decode(h.parts(), 32, 2, &out);
    for (0..32) |r| for (0..2) |c| {
        const want: u8 = if (r == 0 and c == 1) 200 else if (r == 31 and c == 0) 3 else @intCast(10 + (r + c) % 16);
        try std.testing.expectEqual(want, out[r * 2 + c]);
    };
}

test "blocks run tile by tile, the first column pair's block rows first" {
    // R 64 x C 4 is one tile of four blocks; each scale holds its block's number in that order
    var scale: [64 * 4]u8 = undefined;
    for (0..64) |r| for (0..4) |c| {
        scale[r * 4 + c] = @intCast(1 + c / 2 * 2 + r / 32);
    };
    const e = try encode(std.testing.allocator, &scale, 64, 4);
    defer e.deinit(std.testing.allocator);
    const zeros: [64]u8 = @splat(0);
    try std.testing.expectEqualSlices(u8, &zeros, e.bases);
    for (0..4) |n| for (e.codes[32 * n ..][0..32]) |b| try std.testing.expectEqual(@as(u8, @intCast(0x11 * (n + 1))), b);
    try std.testing.expectEqual(@as(usize, 0), e.exceptions.len);
}

test "a row's base is its most covering window, the lowest on a tie, at most 240" {
    // row 0 holds 0 and 16 (every window covers one), row 1 holds 255 twice, the rest 5 (windows 0 to 5 cover both)
    var scale: [64]u8 = @splat(5);
    scale[0..4].* = .{ 0, 16, 255, 255 };
    const e = try encode(std.testing.allocator, &scale, 32, 2);
    defer e.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u8, &.{ 0, 240, 0 }, e.bases[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 16 }, e.exceptions); // 16 at (0, 1)
}

test "parts match the reference encoder's bytes" {
    // each part's sha256 from encode_csf.py's encode_scales of `sample`: one column, odd boundary tiles, 34 tiles
    const golden = [_]struct { rows: usize, cols: usize, sha: [5][]const u8 }{
        .{ .rows = 32, .cols = 1, .sha = .{
            "22398d175284dd029a1d821b2246c4dd8a5f26792f18288d4b40a20b2ac81f93",
            "5ac6a5945f16500911219129984ba8b387a06f24fe383ce4e81a73294065461b",
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        } },
        .{ .rows = 160, .cols = 3, .sha = .{
            "9d619ac307c9069040829a6706abc714a70e211e4cc11a7ef2f8f061711f15db",
            "fa7e29965dd9df08cb889d6d92b0b43c7f7653160a0f49b3900f7e380e354816",
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
            "38cbb6ef752ceaff2bc461ca353b7b1c207de8bddcc50509b54921b0cb572895",
            "b70b64afbf3d1fbcf9024540a08bce3d5d1e387f39604fa3e4c6ffab49b049bb",
        } },
        .{ .rows = 160, .cols = 67, .sha = .{
            "8bfb5a39f0230d3c2010cb2ebd404712c3a599349be267090f1ee3e13fc0443e",
            "17fc8131793cf5760f182531ce1de42e492a7016f957952784f973c01363fb09",
            "724141ec21c423374babec5b754ad8556c45639c0183aee003976853f41919a4",
            "1679e3a939225bcf8ea373a4238cd81696b022eaac1f45e71b66668c63d41534",
            "e318269691664c405d4ddde7868983f3bf85ed73ca0eae3f956c40d55dad76e0",
        } },
    };
    const a = std.testing.allocator;
    for (golden) |g| {
        const scale = try a.alloc(u8, g.rows * g.cols);
        defer a.free(scale);
        for (scale, 0..) |*v, i| v.* = sample(i, i / g.cols);
        const e = try encode(a, scale, g.rows, g.cols);
        defer e.deinit(a);
        for ([_][]const u8{ e.bases, e.codes, e.page_bases, e.offsets, e.exceptions }, g.sha) |bytes, want| {
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
            try std.testing.expectEqualStrings(want, &std.fmt.bytesToHex(digest, .lower));
        }
    }
}

test "matrices of every boundary shape decode to what was encoded" {
    // a short final page (33 tiles), GLM-5.3-Flash's routed-expert shapes (gate/up 2048 x 256, down 4096 x 128)
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(17);
    const rand = prng.random();
    const shapes = [_][2]usize{ .{ 32, 1 }, .{ 32, 3 }, .{ 160, 3 }, .{ 160, 67 }, .{ 128, 132 }, .{ 2048, 256 }, .{ 4096, 128 } };
    for (shapes) |s| {
        const scale = try a.alloc(u8, s[0] * s[1]);
        defer a.free(scale);
        for (scale, 0..) |*v, i| v.* = if (rand.uintLessThan(u8, 8) == 0) rand.int(u8) else @intCast(100 + i / s[1] % 50 + rand.uintLessThan(u8, 18));
        const e = try encode(a, scale, s[0], s[1]);
        defer e.deinit(a);
        const back = try a.alloc(u8, scale.len);
        defer a.free(back);
        try decode(e.parts(), s[0], s[1], back);
        try std.testing.expectEqualSlices(u8, scale, back);
    }
}

test "malformed parts are refused" {
    var out: [64]u8 = undefined;
    var h: Hand = .{};
    h.bases[5] = 241;
    try std.testing.expectError(error.CsfBase, decode(h.parts(), 32, 2, &out));
    h = .{};
    h.exceptions = .{ 62, 3, 1, 200 }; // out of order
    try std.testing.expectError(error.CsfRecord, decode(h.parts(), 32, 2, &out));
    h.exceptions = .{ 1, 200, 1, 3 }; // repeated
    try std.testing.expectError(error.CsfRecord, decode(h.parts(), 32, 2, &out));
    h.exceptions = .{ 1, 200, 64, 3 }; // row 32
    try std.testing.expectError(error.CsfRecord, decode(h.parts(), 32, 2, &out));
    h = .{};
    h.offsets = .{ 1, 0, 2, 0 }; // the page's prefixes start at 1
    try std.testing.expectError(error.CsfIndex, decode(h.parts(), 32, 2, &out));
    h.offsets = .{ 0, 0, 1, 0 }; // a record left over
    try std.testing.expectError(error.CsfIndex, decode(h.parts(), 32, 2, &out));
    h = .{};
    h.page_bases = .{ 2, 0, 0, 0 }; // the arena starts after a record
    try std.testing.expectError(error.CsfIndex, decode(h.parts(), 32, 2, &out));
    h = .{};
    try std.testing.expectError(error.CsfShape, decode(h.parts(), 48, 2, &out));
    try std.testing.expectError(error.CsfShape, decode(h.parts(), 32, 3, &out));

    // C 1: column 1 is outside the block
    const narrow: Parts = .{ .bases = &(@as([32]u8, @splat(0))), .codes = &(@as([16]u8, @splat(0))), .page_bases = &.{ 0, 0, 0, 0 }, .offsets = &.{ 0, 0, 1, 0 }, .exceptions = &.{ 1, 9 } };
    try std.testing.expectError(error.CsfRecord, decode(narrow, 32, 1, out[0..32]));

    // R 64 x C 2: the second block's prefixes decrease
    const zeros: [64]u8 = @splat(0);
    const falling: Parts = .{ .bases = &zeros, .codes = &zeros, .page_bases = &.{ 0, 0, 0, 0 }, .offsets = &.{ 0, 0, 1, 0, 0, 0 }, .exceptions = &.{ 0, 7 } };
    var wide: [128]u8 = undefined;
    try std.testing.expectError(error.CsfIndex, decode(falling, 64, 2, &wide));
}

test "a page's origin continues the arena" {
    const a = std.testing.allocator;
    const scale = try a.alloc(u8, 160 * 67);
    defer a.free(scale);
    for (scale, 0..) |*v, i| v.* = sample(i, i / 67);
    const e = try encode(a, scale, 160, 67);
    defer e.deinit(a);
    const out = try a.alloc(u8, scale.len);
    defer a.free(out);
    try decode(e.parts(), 160, 67, out);
    write32(e.page_bases, 1, read32(e.page_bases, 1) + 2);
    try std.testing.expectError(error.CsfIndex, decode(e.parts(), 160, 67, out));
}

test "overflowing matrix geometry is refused before indexing or allocation" {
    const empty: Parts = .{ .bases = &.{}, .codes = &.{}, .page_bases = &.{}, .offsets = &.{}, .exceptions = &.{} };
    const huge = std.math.maxInt(usize);
    for ([_][2]usize{ .{ 32, huge }, .{ huge & ~@as(usize, 31), 32 } }) |dims| {
        try std.testing.expectError(error.CsfShape, decode(empty, dims[0], dims[1], &.{}));
        try std.testing.expectError(error.CsfShape, encode(std.testing.allocator, &.{}, dims[0], dims[1]));
    }
}

fn allocationCase(a: Allocator) !void {
    const scale: [32 * 3]u8 = @splat(128);
    const encoded = try encode(a, &scale, 32, 3);
    defer encoded.deinit(a);
}

test "each encoding allocation failure releases all earlier parts" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}
