//! DFloat11 groups: checks their tables and decodes their Huffman-coded BF16 exponents block by block.

const std = @import("std");

pub const Error = error{
    LutsShape,
    LutsPointer,
    LutsTree,
    LutsDepth,
    PositionsShape,
    PositionsRange,
    StreamBlocks,
    GapsShape,
    BlockCount,
    GapMismatch,
    CodeLength,
    CodeDepth,
    StreamEnd,
    SplitPositions,
    UnsupportedBytesPerThread,
    UnsupportedThreadsPerBlock,
    OutputSize,
};

/// One group's stored tensors, as the bytes of the safetensors file.
pub const Group = struct {
    luts: []const u8,
    encoded_exponent: []const u8,
    sign_mantissa: []const u8,
    output_positions: []const u8,
    gaps: []const u8,
    bytes_per_thread: u32 = 8,
    threads_per_block: u32 = 512,
};

const pointer = 240; // entries from here to 255 name table 256 - entry
const max_tables = 4; // tables on a path: a code has at most 32 bits
const thread_bits: u64 = 64; // the bits of a thread's chunk, n = 8 bytes

/// Checks the group's shapes and tables (everything that can be checked without walking the stream).
pub fn check(g: Group) Error!void {
    if (g.bytes_per_thread != 8) return error.UnsupportedBytesPerThread;
    if (g.threads_per_block == 0) return error.UnsupportedThreadsPerBlock;
    if (g.luts.len % 256 != 0 or g.luts.len < 2 * 256) return error.LutsShape;
    try checkTree(g.luts);
    const p = g.output_positions;
    if (p.len % 4 != 0 or p.len < 8) return error.PositionsShape;
    const blocks = p.len / 4 - 1;
    if (position(p, 0) != 0 or position(p, blocks) != g.sign_mantissa.len) return error.PositionsRange;
    for (1..blocks + 1) |b| if (position(p, b) < position(p, b - 1)) return error.PositionsRange;
    const t: u64 = g.threads_per_block;
    const threads = (@as(u64, g.encoded_exponent.len) + 7) / 8;
    if ((threads + t - 1) / t != blocks) return error.StreamBlocks;
    if (g.gaps.len != (5 * blocks * t + 7) / 8) return error.GapsShape;
}

/// Walks the tables from table 0: each pointer names a table below k, reached once, at most four tables deep.
fn checkTree(luts: []const u8) Error!void {
    const k = luts.len / 256 - 1;
    var depth: [257 - pointer]u8 = @splat(0);
    var queue: [257 - pointer]usize = undefined;
    depth[0] = 1;
    queue[0] = 0;
    var queued: usize = 1;
    for (0..depth.len) |i| {
        if (i == queued) break;
        const t = queue[i];
        for (luts[256 * t ..][0..256]) |e| {
            if (e < pointer) continue;
            const c = 256 - @as(usize, e);
            if (c >= k) return error.LutsPointer;
            if (depth[c] != 0) return error.LutsTree;
            if (depth[t] == max_tables) return error.LutsDepth;
            depth[c] = depth[t] + 1;
            queue[queued] = c;
            queued += 1;
        }
    }
}

/// Decodes the whole group into out, 2 * N bytes of little-endian BF16; out is unspecified on an error.
pub fn decode(g: Group, out: []u8) Error!void {
    if (out.len != 2 * g.sign_mantissa.len) return error.OutputSize;
    try check(g);
    for (0..g.output_positions.len / 4 - 1) |b| try decodeBlock(g, b, out);
}

/// Decodes the codes that begin in block b's bytes, from its first gap to where the next block's first gap points.
fn decodeBlock(g: Group, b: usize, out: []u8) Error!void {
    const k = g.luts.len / 256 - 1;
    const lengths = g.luts[256 * k ..][0..256];
    const s = g.encoded_exponent;
    const s_bits = 8 * @as(u64, s.len);
    const n = g.sign_mantissa.len;
    const last = b + 2 == g.output_positions.len / 4;
    const first = b * @as(u64, g.threads_per_block);
    const end = (first + g.threads_per_block) * thread_bits;
    const stop: usize = position(g.output_positions, b + 1);
    var i: usize = position(g.output_positions, b);
    if (b == 0 and gap(g.gaps, 0) != 0) return error.GapMismatch;
    var pos = first * thread_bits + gap(g.gaps, first);
    var thread = first; // the thread whose gap the walk has met
    while (pos < end) {
        if (i == stop) {
            if (last) break;
            return error.BlockCount;
        }
        const at = pos / thread_bits;
        if (at != thread) {
            if (pos - at * thread_bits != gap(g.gaps, at)) return error.GapMismatch;
            thread = at;
        }
        var e = g.luts[byteAt(s, pos)];
        var d: u64 = 1;
        while (e >= pointer) : (d += 1) e = g.luts[256 * (256 - @as(usize, e)) + byteAt(s, pos + 8 * d)];
        const len: u64 = lengths[e];
        if (len == 0) return error.CodeLength;
        if ((len + 7) / 8 != d) return error.CodeDepth;
        const sm = g.sign_mantissa[i];
        const v = @as(u16, sm & 0x80) << 8 | @as(u16, e) << 7 | (sm & 0x7f);
        std.mem.writeInt(u16, out[2 * i ..][0..2], v, .little);
        pos += len;
        i += 1;
        if (i == n and (pos > s_bits or pos + 8 <= s_bits)) return error.StreamEnd;
    }
    if (i != stop) return error.BlockCount;
    if (!last and pos != end + gap(g.gaps, first + g.threads_per_block)) return error.GapMismatch;
}

fn position(p: []const u8, b: usize) u32 {
    return std.mem.readInt(u32, p[4 * b ..][0..4], .little);
}

/// Thread j's 5-bit gap, packed most significant bit first.
fn gap(gaps: []const u8, j: u64) u64 {
    const q: usize = @intCast(5 * j / 8);
    const pair = @as(u16, gaps[q]) << 8 | @as(u16, if (q + 1 < gaps.len) gaps[q + 1] else 0);
    return (pair >> @intCast(11 - 5 * j % 8)) & 31;
}

/// The 8 stream bits from bit `bit`, most significant first; bits past the stream are zero.
fn byteAt(s: []const u8, bit: u64) u8 {
    const q = bit / 8;
    const hi: u16 = if (q < s.len) s[@intCast(q)] else 0;
    const lo: u16 = if (q + 1 < s.len) s[@intCast(q + 1)] else 0;
    return @truncate((hi << 8 | lo) >> @intCast(8 - bit % 8));
}

/// Member j's element range [begin, end) from split_positions' bytes (little-endian i64, members - 1 of them).
pub fn member(split_positions: []const u8, members: usize, elements: usize, j: usize) Error!struct { begin: usize, end: usize } {
    if (members == 0 or j >= members) return error.SplitPositions;
    if (split_positions.len % 8 != 0 or split_positions.len / 8 != members - 1) return error.SplitPositions;
    var begin: usize = 0;
    var end: usize = 0;
    var prev: usize = 0;
    for (0..members) |i| {
        const split = if (i + 1 == members) elements else std.math.cast(
            usize,
            std.mem.readInt(i64, split_positions[8 * i ..][0..8], .little),
        ) orelse return error.SplitPositions;
        if (split < prev) return error.SplitPositions;
        if (i == j) {
            begin = prev;
            end = split;
        }
        prev = split;
    }
    return .{ .begin = begin, .end = end };
}

const Buf = struct {
    luts: [6 * 256]u8 = @splat(0),
    enc: [512]u8 = @splat(0),
    pos: [64]u8 = @splat(0),
    gaps: [384]u8 = @splat(0),
    sm: [1536]u8 = @splat(0),
    luts_len: usize,
    enc_len: usize,
    pos_len: usize,
    gaps_len: usize,
    sm_len: usize,
    threads_per_block: u32,
    bytes_per_thread: u32 = 8,

    fn init(f: anytype) Buf {
        var b: Buf = .{
            .luts_len = f.luts.len,
            .enc_len = f.encoded_exponent.len,
            .pos_len = f.output_positions.len,
            .gaps_len = f.gaps.len,
            .sm_len = f.sign_mantissa.len,
            .threads_per_block = f.threads_per_block,
        };
        @memcpy(b.luts[0..f.luts.len], f.luts);
        @memcpy(b.enc[0..f.encoded_exponent.len], f.encoded_exponent);
        @memcpy(b.pos[0..f.output_positions.len], f.output_positions);
        @memcpy(b.gaps[0..f.gaps.len], f.gaps);
        @memcpy(b.sm[0..f.sign_mantissa.len], f.sign_mantissa);
        return b;
    }

    fn group(b: *const Buf) Group {
        return .{
            .luts = b.luts[0..b.luts_len],
            .encoded_exponent = b.enc[0..b.enc_len],
            .sign_mantissa = b.sm[0..b.sm_len],
            .output_positions = b.pos[0..b.pos_len],
            .gaps = b.gaps[0..b.gaps_len],
            .bytes_per_thread = b.bytes_per_thread,
            .threads_per_block = b.threads_per_block,
        };
    }

    fn bytes(b: *Buf, which: usize) []u8 {
        return switch (which) {
            0 => b.luts[0..b.luts_len],
            1 => b.enc[0..b.enc_len],
            2 => b.pos[0..b.pos_len],
            else => b.gaps[0..b.gaps_len],
        };
    }

    fn position(b: *const Buf, i: usize) u32 {
        return std.mem.readInt(u32, b.pos[4 * i ..][0..4], .little);
    }

    fn setPosition(b: *Buf, i: usize, v: u32) void {
        std.mem.writeInt(u32, b.pos[4 * i ..][0..4], v, .little);
    }

    fn setGap(b: *Buf, j: usize, v: u5) void {
        for (0..5) |i| {
            const bit = 5 * j + i;
            const mask = @as(u8, 0x80) >> @intCast(bit % 8);
            if ((v >> @intCast(4 - i)) & 1 == 1) b.gaps[bit / 8] |= mask else b.gaps[bit / 8] &= ~mask;
        }
    }

    fn decoded(b: *const Buf, out: *[2 * 1536]u8) Error![]const u8 {
        try decode(b.group(), out[0 .. 2 * b.sm_len]);
        return out[0 .. 2 * b.sm_len];
    }

    fn refused(b: *const Buf, want: Error) !void {
        var out: [2 * 1536]u8 = undefined;
        try std.testing.expectError(want, b.decoded(&out));
    }
};

fn splitBytes(f: anytype, buf: *[64]u8) []const u8 {
    for (f.split_positions, 0..) |s, i| std.mem.writeInt(i64, buf[8 * i ..][0..8], s, .little);
    return buf[0 .. 8 * f.split_positions.len];
}

test "every fixture group passes the checks, decodes to its bytes and splits into its members" {
    var out: [2 * 1536]u8 = undefined;
    for (@import("dfloat11_fixtures.zig").groups) |f| {
        const b = Buf.init(f);
        try check(b.group());
        try std.testing.expectEqualSlices(u8, f.want, try b.decoded(&out));
        try std.testing.expectEqual(f.tables, f.luts.len / 256 - 1);
        var buf: [64]u8 = undefined;
        const split = splitBytes(f, &buf);
        var at: usize = 0;
        for (0..f.members) |j| {
            const r = try member(split, f.members, f.elements, j);
            try std.testing.expectEqual(at, r.begin);
            if (j + 1 < f.members) try std.testing.expectEqual(f.split_positions[j], @as(i64, @intCast(r.end)));
            at = r.end;
        }
        try std.testing.expectEqual(f.elements, at);
    }
}

test "a stream of the wrong length is refused" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var b = Buf.init(fx[1]);
    b.enc_len -= 33;
    try b.refused(error.StreamBlocks);
    b = .init(fx[1]);
    b.enc_len += 32;
    try b.refused(error.StreamBlocks);
    b = .init(fx[1]);
    b.enc_len = 0;
    try b.refused(error.StreamBlocks);
    b = .init(fx[0]);
    b.enc_len -= 1;
    try b.refused(error.StreamEnd);
    b = .init(fx[0]);
    b.enc_len += 1;
    try b.refused(error.StreamEnd);
    b = .init(fx[0]);
    b.sm_len = 192;
    b.setPosition(1, 192);
    try b.refused(error.StreamEnd);
    b.enc_len = 24;
    var out: [2 * 1536]u8 = undefined;
    try std.testing.expectEqualSlices(u8, fx[0].want[0 .. 2 * 192], try b.decoded(&out));
}

test "sign_mantissa, gaps and output_positions of the wrong shape are refused" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var b = Buf.init(fx[1]);
    b.sm_len -= 1;
    try b.refused(error.PositionsRange);
    b = .init(fx[1]);
    b.sm_len += 1;
    try b.refused(error.PositionsRange);
    b = .init(fx[1]);
    b.gaps_len -= 1;
    try b.refused(error.GapsShape);
    b = .init(fx[1]);
    b.gaps_len += 1;
    try b.refused(error.GapsShape);
    b = .init(fx[1]);
    b.pos_len -= 1;
    try b.refused(error.PositionsShape);
    b = .init(fx[1]);
    b.pos_len -= 4;
    try b.refused(error.PositionsRange);
    b = .init(fx[1]);
    b.pos_len = 4;
    try b.refused(error.PositionsShape);
}

test "output positions that do not count the blocks' elements are refused" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var b = Buf.init(fx[1]);
    b.setPosition(0, 1);
    try b.refused(error.PositionsRange);
    b = .init(fx[1]);
    b.setPosition(5, 241);
    try b.refused(error.PositionsRange);
    b = .init(fx[1]);
    b.setPosition(2, b.position(3) + 1);
    try b.refused(error.PositionsRange);
    b = .init(fx[1]);
    b.setPosition(1, b.position(1) + 1);
    try b.refused(error.BlockCount);
}

test "tables that are not a tree of at most four levels are refused" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var b = Buf.init(fx[0]);
    b.luts_len = 256;
    try b.refused(error.LutsShape);
    b = .init(fx[0]);
    b.luts_len = 513;
    try b.refused(error.LutsShape);
    b = .init(fx[0]);
    b.luts[7] = 256 - 1;
    try b.refused(error.LutsPointer);
    b = .init(fx[2]);
    try std.testing.expectEqual(@as(u8, 255), b.luts[255]);
    b.luts[3] = 255;
    try b.refused(error.LutsTree);
    b = .init(fx[0]);
    b.luts_len = 6 * 256;
    @memset(&b.luts, 0);
    for (0..4) |t| b.luts[256 * t] = @intCast(255 - t);
    try b.refused(error.LutsDepth);
}

test "elements whose code has no length, or a length that does not fit its tables, are refused" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var b = Buf.init(fx[0]);
    b.luts[256 + 0x78] = 9;
    try b.refused(error.CodeDepth);
    b = .init(fx[0]);
    @memset(b.luts[128..256], 0);
    b.enc[3] = 0x10;
    try b.refused(error.CodeLength);
}

test "a gap that names another offset, or an unsupported n or T, is refused" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var b = Buf.init(fx[1]);
    b.setGap(1, 2);
    try b.refused(error.GapMismatch);
    b = .init(fx[0]);
    b.bytes_per_thread = 4;
    try b.refused(error.UnsupportedBytesPerThread);
    b = .init(fx[0]);
    b.threads_per_block = 0;
    try b.refused(error.UnsupportedThreadsPerBlock);
}

test "split positions out of order, past N or of the wrong count are refused" {
    const f = @import("dfloat11_fixtures.zig").groups[1];
    var buf: [64]u8 = undefined;
    const split = splitBytes(f, &buf);
    try std.testing.expectError(error.SplitPositions, member(split, 3, f.elements, 0));
    try std.testing.expectError(error.SplitPositions, member(split[0..16], 4, f.elements, 0));
    try std.testing.expectError(error.SplitPositions, member(split, 4, 210, 0));
    std.mem.writeInt(i64, buf[0..8], 62, .little);
    try std.testing.expectError(error.SplitPositions, member(split, 4, f.elements, 3));
    std.mem.writeInt(i64, buf[0..8], -1, .little);
    try std.testing.expectError(error.SplitPositions, member(split, 4, f.elements, 3));
}

test "bits after the last element are not checked, whatever entries they reach" {
    const f = @import("dfloat11_fixtures.zig").groups[0];
    var out: [2 * 1536]u8 = undefined;
    for ([_]u8{ 0x79, 0x00 }) |v| {
        var b = Buf.init(f);
        @memset(b.luts[128..256], v);
        b.luts[256 + 0x79] = 20;
        b.enc[24] |= 0x1f;
        try std.testing.expectEqualSlices(u8, f.want, try b.decoded(&out));
    }
}

test "random single-byte changes are refused or decoded, never anything else" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var prng = std.Random.DefaultPrng.init(1117);
    const r = prng.random();
    var out: [2 * 1536]u8 = undefined;
    var seen = [2]usize{ 0, 0 };
    for (0..4000) |_| {
        var b = Buf.init(fx[r.uintLessThan(usize, fx.len)]);
        const field = b.bytes(r.uintLessThan(usize, 4));
        field[r.uintLessThan(usize, field.len)] = r.int(u8);
        _ = b.decoded(&out) catch {
            seen[0] += 1;
            continue;
        };
        seen[1] += 1;
    }
    try std.testing.expect(seen[0] > 0 and seen[1] > 0);
}

test "every block's first gap must continue the stream where the block before it ended" {
    const fx = @import("dfloat11_fixtures.zig").groups;
    var b = Buf.init(fx[0]);
    b.setGap(0, 5);
    try b.refused(error.GapMismatch);
    for (fx) |f| {
        const blocks = f.output_positions.len / 4 - 1;
        if (blocks == 1) continue;
        for (0..blocks) |k| {
            const j = k * f.threads_per_block;
            for ([_]i64{ 1, -1 }) |step| {
                const v = @as(i64, @intCast(gap(f.gaps, j))) + step;
                if (v < 0 or v > 31) continue;
                b = .init(f);
                b.setGap(j, @intCast(v));
                try b.refused(error.GapMismatch);
            }
        }
    }
}

test "a table no pointer names is not walked and does not change the values" {
    const f = @import("dfloat11_fixtures.zig").groups[2];
    var b = Buf.init(f);
    @memcpy(b.luts[768..1024], f.luts[512..768]);
    for (b.luts[512..768], 0..) |*e, i| e.* = @truncate(i * 151 + 7);
    b.luts_len = 1024;
    var out: [2 * 1536]u8 = undefined;
    try std.testing.expectEqualSlices(u8, f.want, try b.decoded(&out));
}

test "an output of the wrong size is refused" {
    const b = Buf.init(@import("dfloat11_fixtures.zig").groups[1]);
    var out: [2 * 1536]u8 = undefined;
    try std.testing.expectError(error.OutputSize, decode(b.group(), out[0 .. 2 * b.sm_len + 2]));
    try std.testing.expectError(error.OutputSize, decode(b.group(), out[0 .. 2 * b.sm_len - 1]));
}
