//! A torch.save archive's zip container: the central directory and its stored members, CRC-checked.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Member = struct { local: u64, size: u64, crc: u32 };

pub fn readU16(b: []const u8, at: usize) u16 {
    return std.mem.readInt(u16, b[at..][0..2], .little);
}

pub fn readU32(b: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, b[at..][0..4], .little);
}

pub fn readU64(b: []const u8, at: usize) u64 {
    return std.mem.readInt(u64, b[at..][0..8], .little);
}

/// A stored member's bytes into `out` (exactly its size), CRC-checked.
pub fn readMember(io: std.Io, file: std.Io.File, m: Member, out: []u8) !void {
    if (out.len != m.size) return error.BadArchive;
    var head: [30]u8 = undefined;
    if (try file.readPositionalAll(io, &head, m.local) != head.len) return error.ShortRead;
    if (readU32(&head, 0) != 0x04034b50) return error.BadArchive;
    const data: u64 = m.local + @as(u64, head.len) + @as(u64, readU16(&head, 26)) + @as(u64, readU16(&head, 28));
    if (try file.readPositionalAll(io, out, data) != out.len) return error.ShortRead;
    if (std.hash.Crc32.hash(out) != m.crc) return error.ChecksumMismatch;
}

/// The central directory: every member stored (method 0), zip64 records honoured.
pub fn readDirectory(a: Allocator, io: std.Io, file: std.Io.File) !std.StringHashMapUnmanaged(Member) {
    const size = try file.length(io);
    if (size < 22) return error.BadArchive;
    const tail_len: usize = @intCast(@min(size, 22 + 0xFFFF));
    const tail = try a.alloc(u8, tail_len);
    if (try file.readPositionalAll(io, tail, size - @as(u64, tail_len)) != tail_len) return error.ShortRead;
    var at: usize = tail_len - 22;
    while (readU32(tail, at) != 0x06054b50) {
        if (at == 0) return error.BadArchive;
        at -= 1;
    }
    var count: u64 = readU16(tail, at + 10);
    var cd_size: u64 = readU32(tail, at + 12);
    var cd_off: u64 = readU32(tail, at + 16);
    if (count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF) {
        if (at < 20 or readU32(tail, at - 20) != 0x07064b50) return error.BadArchive;
        var rec: [56]u8 = undefined;
        if (try file.readPositionalAll(io, &rec, readU64(tail, at - 20 + 8)) != rec.len) return error.ShortRead;
        if (readU32(&rec, 0) != 0x06064b50) return error.BadArchive;
        count = readU64(&rec, 32);
        cd_size = readU64(&rec, 40);
        cd_off = readU64(&rec, 48);
    }
    if (cd_size > 64 << 20 or cd_off > size or cd_size > size - cd_off) return error.BadArchive;
    const cd = try a.alloc(u8, @intCast(cd_size));
    if (try file.readPositionalAll(io, cd, cd_off) != cd.len) return error.ShortRead;
    var members: std.StringHashMapUnmanaged(Member) = .empty;
    var p: usize = 0;
    var n: u64 = 0;
    while (n < count) : (n += 1) {
        if (cd.len - p < 46 or readU32(cd, p) != 0x02014b50) return error.BadArchive;
        const method = readU16(cd, p + 10);
        const crc = readU32(cd, p + 16);
        var csize: u64 = readU32(cd, p + 20);
        var usize_: u64 = readU32(cd, p + 24);
        const nlen: usize = readU16(cd, p + 28);
        const xlen: usize = readU16(cd, p + 30);
        const clen: usize = readU16(cd, p + 32);
        var local: u64 = readU32(cd, p + 42);
        if (cd.len - p - 46 < nlen + xlen + clen) return error.BadArchive;
        const name = cd[p + 46 ..][0..nlen];
        var x = p + 46 + nlen;
        const xend = x + xlen;
        while (xend - x >= 4) {
            const id = readU16(cd, x);
            const len: usize = readU16(cd, x + 2);
            if (xend - x - 4 < len) return error.BadArchive;
            if (id == 0x0001) {
                var f = x + 4;
                const fend = f + len;
                if (usize_ == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    usize_ = readU64(cd, f);
                    f += 8;
                }
                if (csize == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    csize = readU64(cd, f);
                    f += 8;
                }
                if (local == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    local = readU64(cd, f);
                }
            }
            x += 4 + len;
        }
        if (method != 0 or csize != usize_) {
            std.log.err("codec.pth member {s} is compressed; torch.save stores members", .{name});
            return error.CompressedMember;
        }
        if (local > size or usize_ > size - local) return error.BadArchive;
        try members.put(a, name, .{ .local = local, .size = usize_, .crc = crc });
        p += 46 + nlen + xlen + clen;
    }
    return members;
}
