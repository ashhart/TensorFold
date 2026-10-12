//! A session file (format 2): a header, then segments of 4 KiB-aligned chunks, each with its own SHA-256.

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const up = @import("lanes.zig").up;

pub const magic = "TFSESS02".*;
pub const version: u32 = 2;

pub const SegKind = enum(u32) { ids = 1, pool = 2, blob = 3, _ };

pub const Head = extern struct {
    magic: [8]u8 = magic,
    version: u32 = version,
    /// bytes of Head + segments + digests (before the pad)
    head_bytes: u32 = 0,
    /// SHA-256 of those bytes with this field zero
    head_sha: [32]u8 = @splat(0),
    key: [16]u8,
    /// the entry whose files hold logical pages [0, first_page); zero: none
    base: [16]u8 = @splat(0),
    /// what makes these bytes mean the same thing to a build (family, knobs, layout, split)
    compat: [16]u8,
    tag: u32,
    kind: u32,
    page: u32,
    world: u32,
    rank: u32,
    first_page: u32,
    npages: u32,
    nseg: u32,
    ndigest: u32,
    _pad: u32 = 0,
    pos: u64,
    total: u64 = 0,

    pub fn hasBase(h: *const Head) bool {
        return !std.mem.allEqual(u8, &h.base, 0);
    }
};

pub const Seg = extern struct {
    name: [40]u8,
    kind: SegKind,
    /// pool: the family's index in the layout
    family: u32,
    nbytes: u64,
    offset: u64 = 0,
    /// bytes a chunk (the last may be shorter) and file bytes a chunk (4 KiB multiple)
    chunk: u32,
    stride: u32 = 0,
    first_digest: u32 = 0,
    nchunks: u32 = 0,

    pub fn nameOf(s: *const Seg) []const u8 {
        return std.mem.sliceTo(&s.name, 0);
    }

    /// Chunk i's data bytes and file span.
    pub fn chunkAt(s: *const Seg, i: u32) struct { off: u64, data: u32, span: u32 } {
        const data: u32 = @intCast(@min(@as(u64, s.chunk), s.nbytes - @as(u64, i) * s.chunk));
        return .{ .off = s.offset + @as(u64, i) * s.stride, .data = data, .span = @intCast(up(data)) };
    }
};

pub const Digest32 = [32]u8;

pub fn segName(name: []const u8) [40]u8 {
    var n: [40]u8 = @splat(0);
    @memcpy(n[0..@min(name.len, 39)], name[0..@min(name.len, 39)]);
    return n;
}

/// A file's layout: offsets, chunks and the digest slots, before or after its chunks are written.
pub const Layout = struct {
    head: Head,
    segs: []Seg,
    digests: []Digest32,
    /// file bytes of the padded header
    head_span: u64,

    /// Places segments (each with name, kind, family, nbytes, chunk) after the header.
    pub fn build(gpa: std.mem.Allocator, head: Head, segs_in: []const Seg) !Layout {
        const segs = try gpa.dupe(Seg, segs_in);
        errdefer gpa.free(segs);
        var nd: u32 = 0;
        for (segs) |*s| {
            if (s.chunk == 0) return error.BadSegment;
            s.nchunks = @intCast((s.nbytes + s.chunk - 1) / s.chunk);
            s.stride = @intCast(up(s.chunk));
            s.first_digest = nd;
            nd += s.nchunks;
        }
        var h = head;
        h.nseg = @intCast(segs.len);
        h.ndigest = nd;
        h.head_bytes = @intCast(@sizeOf(Head) + segs.len * @sizeOf(Seg) + nd * @sizeOf(Digest32));
        const span = up(h.head_bytes);
        var off = span;
        for (segs) |*s| {
            s.offset = off;
            if (s.nchunks > 0) off += @as(u64, s.nchunks - 1) * s.stride + up(s.nbytes - @as(u64, s.nchunks - 1) * s.chunk);
        }
        h.total = off;
        const d = try gpa.alloc(Digest32, nd);
        @memset(d, @splat(0));
        return .{ .head = h, .segs = segs, .digests = d, .head_span = span };
    }

    pub fn deinit(l: *Layout, gpa: std.mem.Allocator) void {
        gpa.free(l.segs);
        gpa.free(l.digests);
        l.* = undefined;
    }

    /// The header's bytes with its checksum into out (len >= head_span; the pad is zeroed).
    pub fn encode(l: *Layout, out: []u8) void {
        @memset(out[0..l.head_span], 0);
        var h = l.head;
        h.head_sha = @splat(0);
        const hb = std.mem.asBytes(&h);
        const sb = std.mem.sliceAsBytes(l.segs);
        const db = std.mem.sliceAsBytes(l.digests);
        @memcpy(out[0..hb.len], hb);
        @memcpy(out[hb.len..][0..sb.len], sb);
        @memcpy(out[hb.len + sb.len ..][0..db.len], db);
        var sha: [32]u8 = undefined;
        Sha256.hash(out[0..h.head_bytes], &sha, .{});
        @memcpy(out[@offsetOf(Head, "head_sha")..][0..32], &sha);
        l.head.head_sha = sha;
    }

    /// The header read back from bytes (at least head_bytes of them), every field and offset checked.
    pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) !Layout {
        if (bytes.len < @sizeOf(Head)) return error.BadHeader;
        var h: Head = undefined;
        @memcpy(std.mem.asBytes(&h), bytes[0..@sizeOf(Head)]);
        if (!std.mem.eql(u8, &h.magic, &magic) or h.version != version) return error.BadHeader;
        const want = @sizeOf(Head) + @as(u64, h.nseg) * @sizeOf(Seg) + @as(u64, h.ndigest) * @sizeOf(Digest32);
        if (h.head_bytes != want or bytes.len < want) return error.BadHeader;
        var check: [32]u8 = undefined;
        var sha = Sha256.init(.{});
        sha.update(bytes[0..@offsetOf(Head, "head_sha")]);
        sha.update(&@as([32]u8, @splat(0)));
        sha.update(bytes[@offsetOf(Head, "head_sha") + 32 .. want]);
        sha.final(&check);
        if (!std.mem.eql(u8, &check, &h.head_sha)) return error.ChecksumMismatch;
        const segs = try gpa.alloc(Seg, h.nseg);
        errdefer gpa.free(segs);
        @memcpy(std.mem.sliceAsBytes(segs), bytes[@sizeOf(Head)..][0 .. segs.len * @sizeOf(Seg)]);
        const d = try gpa.alloc(Digest32, h.ndigest);
        errdefer gpa.free(d);
        @memcpy(std.mem.sliceAsBytes(d), bytes[@sizeOf(Head) + segs.len * @sizeOf(Seg) ..][0 .. d.len * 32]);
        var nd: u64 = 0;
        for (segs) |s| {
            if (s.chunk == 0 or s.stride != up(s.chunk) or s.first_digest != nd or s.offset % 4096 != 0) return error.BadHeader;
            if (s.nchunks != (s.nbytes + s.chunk - 1) / s.chunk) return error.BadHeader;
            if (s.nchunks > 0 and s.offset + @as(u64, s.nchunks - 1) * s.stride + up(s.nbytes - @as(u64, s.nchunks - 1) * s.chunk) > h.total) return error.BadHeader;
            nd += s.nchunks;
        }
        if (nd != h.ndigest) return error.BadHeader;
        return .{ .head = h, .segs = segs, .digests = d, .head_span = up(h.head_bytes) };
    }

    pub fn find(l: *const Layout, kind: SegKind, name: []const u8) ?*const Seg {
        for (l.segs) |*s| if (s.kind == kind and std.mem.eql(u8, s.nameOf(), name)) return s;
        return null;
    }
};

test "a layout encodes, decodes, and refuses a damaged header" {
    const gpa = std.testing.allocator;
    const head: Head = .{ .key = @splat(1), .compat = @splat(2), .tag = 3, .kind = 0, .page = 256, .world = 2, .rank = 1, .first_page = 4, .npages = 9, .nseg = 0, .ndigest = 0, .pos = 2200 };
    var l = try Layout.build(gpa, head, &.{
        .{ .name = segName("ids"), .kind = .ids, .family = 0, .nbytes = 4 * (2200 - 1024), .chunk = 1 << 20 },
        .{ .name = segName("comp.20"), .kind = .pool, .family = 0, .nbytes = 3 * 149_504, .chunk = 2 * 149_504 },
        .{ .name = segName("taps"), .kind = .blob, .family = 0, .nbytes = 0, .chunk = 4096 },
    });
    defer l.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 1 + 2 + 0), l.head.ndigest);
    try std.testing.expectEqual(@as(u64, 4096), l.segs[0].offset);
    try std.testing.expectEqual(@as(u64, 4096 + up(4 * 1176)), l.segs[1].offset);
    const c1 = l.segs[1].chunkAt(1);
    try std.testing.expectEqual(@as(u32, 149_504), c1.data);
    try std.testing.expectEqual(l.segs[1].offset + up(2 * 149_504), c1.off);
    try std.testing.expectEqual(l.segs[2].offset, l.head.total);
    l.digests[1][5] = 9;
    var buf: [8192]u8 = undefined;
    l.encode(&buf);
    var back = try Layout.decode(gpa, &buf);
    defer back.deinit(gpa);
    try std.testing.expectEqual(l.head.total, back.head.total);
    try std.testing.expectEqual(@as(u8, 9), back.digests[1][5]);
    try std.testing.expectEqualStrings("comp.20", back.find(.pool, "comp.20").?.nameOf());
    buf[200] ^= 0x40;
    try std.testing.expectError(error.ChecksumMismatch, Layout.decode(gpa, &buf));
}
