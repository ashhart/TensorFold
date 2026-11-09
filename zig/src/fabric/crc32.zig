//! zlib's CRC-32 of a frame's payload, with the ARMv8 CRC32 instructions where the CPU has them (same polynomial).
const std = @import("std");
const builtin = @import("builtin");

/// The target promises the instructions (an M-series Mac, a GB10 built for its own CPU): no check at run time.
const promised = builtin.cpu.arch == .aarch64 and std.Target.aarch64.featureSetHas(builtin.cpu.features, .crc);

/// Whether this CPU runs them: promised, or the kernel's HWCAP_CRC32 on aarch64 Linux; else the table code.
pub fn hardware() bool {
    if (promised) return true;
    if (comptime builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux or !builtin.link_libc) return false;
    return switch (probed.load(.acquire)) {
        1 => true,
        2 => false,
        else => blk: {
            const has = std.c.getauxval(16) & (1 << 7) != 0; // AT_HWCAP
            probed.store(if (has) 1 else 2, .release);
            break :blk has;
        },
    };
}

var probed: std.atomic.Value(u8) = .init(0);

/// zlib.crc32(bytes): reflected polynomial 0xEDB88320, initial and final value 0xFFFFFFFF.
pub fn hash(bytes: []const u8) u32 {
    if (comptime builtin.cpu.arch != .aarch64) return std.hash.Crc32.hash(bytes);
    if (!hardware()) return std.hash.Crc32.hash(bytes);
    return ~update(0xFFFF_FFFF, bytes);
}

fn update(start: u32, bytes: []const u8) u32 {
    var crc = start;
    var i: usize = 0;
    while (i + 8 <= bytes.len) : (i += 8) crc = word(crc, std.mem.readInt(u64, bytes[i..][0..8], .little));
    while (i < bytes.len) : (i += 1) crc = byte(crc, bytes[i]);
    return crc;
}

inline fn word(crc: u32, v: u64) u32 {
    return asm (".arch_extension crc\n crc32x %[out:w], %[crc:w], %[v:x]"
        : [out] "=r" (-> u32),
        : [crc] "r" (crc),
          [v] "r" (v),
    );
}

inline fn byte(crc: u32, v: u8) u32 {
    return asm (".arch_extension crc\n crc32b %[out:w], %[crc:w], %[v:w]"
        : [out] "=r" (-> u32),
        : [crc] "r" (crc),
          [v] "r" (@as(u32, v)),
    );
}

test "the CRC equals std.hash.Crc32 (zlib's) at every length and offset, and the check value" {
    try std.testing.expectEqual(@as(u32, 0xCBF43926), hash("123456789"));
    var data: [301]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 11);
    for (0..9) |off| for (0..data.len - off) |n| {
        try std.testing.expectEqual(std.hash.Crc32.hash(data[off..][0..n]), hash(data[off..][0..n]));
    };
}
