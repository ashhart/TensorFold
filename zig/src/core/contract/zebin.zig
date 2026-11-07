//! The .ze_info section of an Intel zebin: the YAML ocloc writes for each kernel of the image.
const std = @import("std");

pub const Error = error{ NotAnElf, NotZebin, OutOfMemory };

const elf_magic = "\x7fELF";
const sht_zeinfo: u32 = 0xff000011;

/// The `.ze_info` bytes of the zebin in `bytes`, found by its section type (the name is not required).
pub fn zeInfo(a: std.mem.Allocator, bytes: []const u8) Error![]u8 {
    if (bytes.len < 64 or !std.mem.eql(u8, bytes[0..4], elf_magic)) return error.NotAnElf;
    if (bytes[4] != 2 or bytes[5] != 1) return error.NotAnElf; // ELF64, little-endian
    const shoff = int(u64, bytes, 0x28) orelse return error.NotAnElf;
    const shentsize = int(u16, bytes, 0x3a) orelse return error.NotAnElf;
    const shnum = int(u16, bytes, 0x3c) orelse return error.NotAnElf;
    if (shentsize < 64) return error.NotAnElf;
    var i: usize = 0;
    while (i < shnum) : (i += 1) {
        const at = shoff + i * shentsize;
        const kind = int(u32, bytes, at + 4) orelse return error.NotAnElf;
        if (kind != sht_zeinfo) continue;
        const off = int(u64, bytes, at + 24) orelse return error.NotAnElf;
        const size = int(u64, bytes, at + 32) orelse return error.NotAnElf;
        if (off + size > bytes.len) return error.NotZebin;
        return a.dupe(u8, bytes[@intCast(off)..][0..@intCast(size)]);
    }
    return error.NotZebin;
}

fn int(comptime T: type, bytes: []const u8, at: u64) ?T {
    const end = at + @sizeOf(T);
    if (end > bytes.len) return null;
    return std.mem.readInt(T, bytes[@intCast(at)..][0..@sizeOf(T)], .little);
}

/// An ELF64 with one `.ze_info` section, for the tests: `body` goes after the header and the section header table.
pub fn build(a: std.mem.Allocator, body: []const u8) ![]u8 {
    const shoff = 64;
    const body_at = shoff + 64;
    var out = try a.alloc(u8, body_at + body.len);
    @memset(out, 0);
    @memcpy(out[0..4], elf_magic);
    out[4] = 2;
    out[5] = 1;
    std.mem.writeInt(u64, out[0x28..][0..8], shoff, .little);
    std.mem.writeInt(u16, out[0x3a..][0..2], 64, .little);
    std.mem.writeInt(u16, out[0x3c..][0..2], 1, .little);
    std.mem.writeInt(u32, out[shoff + 4 ..][0..4], sht_zeinfo, .little);
    std.mem.writeInt(u64, out[shoff + 24 ..][0..8], body_at, .little);
    std.mem.writeInt(u64, out[shoff + 32 ..][0..8], body.len, .little);
    @memcpy(out[body_at..], body);
    return out;
}

test "the .ze_info section is found by type, and a plain ELF is refused" {
    const a = std.testing.allocator;
    const body = "---\nversion: '1.73'\nkernels:\n";
    const elf = try build(a, body);
    defer a.free(elf);
    const got = try zeInfo(a, elf);
    defer a.free(got);
    try std.testing.expectEqualStrings(body, got);
    try std.testing.expectError(error.NotAnElf, zeInfo(a, "not an elf at all, not even close to one"));
    const empty = try build(a, "");
    defer a.free(empty);
    const none = try zeInfo(a, empty);
    defer a.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
    try std.testing.expectError(error.NotZebin, zeInfo(a, empty[0..110]));
}
