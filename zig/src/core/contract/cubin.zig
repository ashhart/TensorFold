//! What nvcc made of each kernel, read from a fatbin's ELF images: registers, static shared bytes, launch bounds.
const std = @import("std");
const limits = @import("limits.zig");

const fatbin_magic: u32 = 0xBA55ED50;
const elf_kind: u16 = 2;
const sht_nobits = 8;
/// .nv.info records: EIATTR_MAX_THREADS (__launch_bounds__) and EIATTR_REGCOUNT, both sized values.
const max_threads_attr = 0x05;
const regcount_attr = 0x2f;
const sval = 0x04;
/// sm_120 and sm_121 images count a block's 1 KiB of reserved shared memory in .nv.shared; the driver does not.
const reserved_shared = 1024;

/// One kernel in one SM's image.
pub const Kernel = struct { sm: u32, name: []const u8, usage: limits.Usage };

/// One SM's ELF image in a fatbin.
pub const Image = struct { sm: u32, elf: []const u8 };

/// Every ELF image in a fatbin, with its SM; a plain ELF cubin is one image of an unknown SM (0).
pub fn images(a: std.mem.Allocator, bytes: []const u8) ![]Image {
    var out: std.ArrayList(Image) = .empty;
    errdefer out.deinit(a);
    if (bytes.len >= 4 and std.mem.eql(u8, bytes[0..4], "\x7fELF")) {
        try out.append(a, .{ .sm = 0, .elf = bytes });
        return out.toOwnedSlice(a);
    }
    var at: usize = 0;
    while (at < bytes.len) {
        if (bytes.len - at < 16 or read(u32, bytes, at) != fatbin_magic) return error.NotAFatbin;
        const header: usize = read(u16, bytes, at + 6);
        const end = at + header + (std.math.cast(usize, read(u64, bytes, at + 8)) orelse return error.BadFatbin);
        if (end > bytes.len) return error.BadFatbin;
        var entry = at + header;
        while (entry + 32 <= end) {
            const entry_header: usize = read(u32, bytes, entry + 4);
            const payload = std.math.cast(usize, read(u64, bytes, entry + 8)) orelse return error.BadFatbin;
            const next = entry + entry_header + payload;
            if (entry_header < 32 or next > end) return error.BadFatbin;
            const flags = read(u64, bytes, entry + 40);
            if (read(u16, bytes, entry) == elf_kind) {
                if (flags & 0x2000 != 0) return error.CompressedFatbin;
                try out.append(a, .{ .sm = read(u32, bytes, entry + 28), .elf = bytes[entry + entry_header .. next] });
            }
            entry = next;
        }
        at = std.mem.alignForward(usize, end, 8);
    }
    return out.toOwnedSlice(a);
}

fn read(comptime T: type, bytes: []const u8, at: usize) T {
    return std.mem.readInt(T, bytes[at..][0..@sizeOf(T)], .little);
}

const Section = struct { name: []const u8, kind: u32, info: u32, off: usize, size: usize };

fn sections(a: std.mem.Allocator, elf: []const u8) ![]Section {
    if (elf.len < 64 or !std.mem.eql(u8, elf[0..4], "\x7fELF") or elf[4] != 2) return error.NotElf64;
    const shoff: usize = @intCast(read(u64, elf, 0x28));
    const shentsize: usize = read(u16, elf, 0x3a);
    const shnum: usize = read(u16, elf, 0x3c);
    const strndx: usize = read(u16, elf, 0x3e);
    if (shentsize < 64 or shoff + shnum * shentsize > elf.len or strndx >= shnum) return error.BadElf;
    const out = try a.alloc(Section, shnum);
    for (out, 0..) |*s, i| {
        const h = shoff + i * shentsize;
        s.* = .{ .name = "", .kind = read(u32, elf, h + 4), .info = read(u32, elf, h + 44), .off = @intCast(read(u64, elf, h + 24)), .size = @intCast(read(u64, elf, h + 32)) };
    }
    const strtab = out[strndx];
    for (out, 0..) |*s, i| {
        const at = strtab.off + read(u32, elf, shoff + i * shentsize);
        if (at >= elf.len) return error.BadElf;
        s.name = std.mem.sliceTo(elf[at..], 0);
    }
    return out;
}

/// Each kernel's registers, static shared bytes and __launch_bounds__, as the driver will count them.
pub fn kernels(a: std.mem.Allocator, elf: []const u8, sm: u32) ![]Kernel {
    const all = try sections(a, elf);
    defer a.free(all);
    var global: []const u8 = &.{};
    for (all) |s| if (std.mem.eql(u8, s.name, ".nv.info")) {
        global = elf[s.off..][0..s.size];
    };
    var out: std.ArrayList(Kernel) = .empty;
    errdefer out.deinit(a);
    for (all) |s| if (std.mem.startsWith(u8, s.name, ".text.")) {
        const name = s.name[".text.".len..];
        // the .text section's sh_info is the kernel's symbol, its top byte the registers before sm_120
        var u = limits.Usage{ .regs = try record(global, regcount_attr, s.info & 0xffffff) orelse s.info >> 24, .static = 0 };
        for (all) |t| {
            if (std.mem.startsWith(u8, t.name, ".nv.shared.") and std.mem.eql(u8, t.name[".nv.shared.".len..], name))
                u.static = @intCast(if (sm >= 120 and t.size >= reserved_shared) t.size - reserved_shared else t.size);
            if (std.mem.startsWith(u8, t.name, ".nv.info.") and std.mem.eql(u8, t.name[".nv.info.".len..], name) and t.kind != sht_nobits)
                u.bound = try bound(elf[t.off..][0..t.size]);
        }
        try out.append(a, .{ .sm = sm, .name = name, .usage = u });
    };
    return out.toOwnedSlice(a);
}

/// The value a global .nv.info record of `attr` keeps for symbol `sym` (EIATTR_REGCOUNT's symbol, count).
fn record(info: []const u8, attr: u8, sym: u32) !?u32 {
    var at: usize = 0;
    while (at + 4 <= info.len) {
        const size: usize = if (info[at] == sval) read(u16, info, at + 2) else 0;
        if (at + 4 + size > info.len) return error.BadNvInfo;
        if (info[at] == sval and info[at + 1] == attr and size == 8 and read(u32, info, at + 4) == sym) return read(u32, info, at + 8);
        at += 4 + size;
    }
    return null;
}

/// A kernel's .nv.info records: EIATTR_MAX_THREADS gives its __launch_bounds__ (0: none).
fn bound(info: []const u8) !u32 {
    var at: usize = 0;
    while (at + 4 <= info.len) {
        const fmt = info[at];
        const attr = info[at + 1];
        const size: usize = if (fmt == sval) read(u16, info, at + 2) else 0;
        if (at + 4 + size > info.len) return error.BadNvInfo;
        if (fmt == sval and attr == max_threads_attr and size >= 12)
            return read(u32, info, at + 4) * read(u32, info, at + 8) * read(u32, info, at + 12);
        at += 4 + size;
    }
    return 0;
}

/// A tiny ELF64 with one kernel k (symbol 7): its registers in .text.k's sh_info top byte, or in .nv.info alone.
fn synthetic(a: std.mem.Allocator, regs: u32, shared: u64, threads: u32, packed_regs: bool) ![]u8 {
    const names = "\x00.shstrtab\x00.text.k\x00.nv.shared.k\x00.nv.info.k\x00.nv.info\x00";
    const sym = 7;
    var info: [16 + 24]u8 = undefined;
    info[0] = sval;
    info[1] = max_threads_attr;
    std.mem.writeInt(u16, info[2..4], 12, .little);
    for ([_]u32{ threads, 1, 1 }, 0..) |v, i| std.mem.writeInt(u32, info[4 + 4 * i ..][0..4], v, .little);
    for ([_][2]u32{ .{ sym + 1, 7 }, .{ sym, regs } }, 0..) |r, i| { // the next symbol's count first, a decoy
        const at = 16 + 12 * i;
        info[at] = sval;
        info[at + 1] = regcount_attr;
        std.mem.writeInt(u16, info[at + 2 ..][0..2], 8, .little);
        std.mem.writeInt(u32, info[at + 4 ..][0..4], r[0], .little);
        std.mem.writeInt(u32, info[at + 8 ..][0..4], r[1], .little);
    }
    const data_at = 64;
    const info_at = data_at + names.len;
    const shoff = std.mem.alignForward(usize, info_at + info.len, 8);
    const out = try a.alloc(u8, shoff + 6 * 64);
    @memset(out, 0);
    @memcpy(out[0..4], "\x7fELF");
    out[4] = 2;
    std.mem.writeInt(u64, out[0x28..0x30], shoff, .little);
    std.mem.writeInt(u16, out[0x3a..0x3c], 64, .little);
    std.mem.writeInt(u16, out[0x3c..0x3e], 6, .little);
    std.mem.writeInt(u16, out[0x3e..0x40], 1, .little);
    @memcpy(out[data_at..][0..names.len], names);
    @memcpy(out[info_at..][0..info.len], &info);
    const rows = [_]struct { name: u32, kind: u32, info: u32, off: u64, size: u64 }{
        .{ .name = 0, .kind = 0, .info = 0, .off = 0, .size = 0 },
        .{ .name = 1, .kind = 3, .info = 0, .off = data_at, .size = names.len },
        .{ .name = 11, .kind = 1, .info = if (packed_regs) regs << 24 | sym else sym, .off = data_at, .size = 0 },
        .{ .name = 19, .kind = sht_nobits, .info = 0, .off = 0, .size = shared },
        .{ .name = 32, .kind = 0x70000000, .info = 0, .off = info_at, .size = 16 },
        .{ .name = 43, .kind = 0x70000000, .info = 0, .off = info_at + 16, .size = if (packed_regs) 0 else 24 },
    };
    for (rows, 0..) |r, i| {
        const h = shoff + 64 * i;
        std.mem.writeInt(u32, out[h..][0..4], r.name, .little);
        std.mem.writeInt(u32, out[h + 4 ..][0..4], r.kind, .little);
        std.mem.writeInt(u64, out[h + 24 ..][0..8], r.off, .little);
        std.mem.writeInt(u64, out[h + 32 ..][0..8], r.size, .little);
        std.mem.writeInt(u32, out[h + 44 ..][0..4], r.info, .little);
    }
    return out;
}

test "a cubin's registers, static shared bytes and launch bounds" {
    const a = std.testing.allocator;
    for ([_]bool{ true, false }) |packed_regs| {
        const elf = try synthetic(a, 143, 1060, 256, packed_regs);
        defer a.free(elf);
        const imgs = try images(a, elf);
        defer a.free(imgs);
        try std.testing.expectEqual(@as(usize, 1), imgs.len);
        for ([_]u32{ 86, 121 }, [_]u32{ 1060, 36 }) |sm, static| {
            const ks = try kernels(a, imgs[0].elf, sm);
            defer a.free(ks);
            try std.testing.expectEqual(@as(usize, 1), ks.len);
            try std.testing.expectEqualStrings("k", ks[0].name);
            try std.testing.expectEqual(limits.Usage{ .regs = 143, .static = static, .bound = 256 }, ks[0].usage);
        }
    }
    try std.testing.expectError(error.NotAFatbin, images(a, "not a fatbin"));
}
