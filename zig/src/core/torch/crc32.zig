const std = @import("std");
const builtin = @import("builtin");

pub fn hash(bytes: []const u8) u32 {
    if (comptime builtin.cpu.arch != .aarch64 or
        !std.Target.aarch64.featureSetHas(builtin.cpu.features, .crc)) return std.hash.Crc32.hash(bytes);
    var state: u32 = 0xffffffff;
    var offset: usize = 0;
    while (bytes.len - offset >= 8) : (offset += 8) {
        const word = std.mem.readInt(u64, bytes[offset..][0..8], .little);
        state = asm ("crc32x w0, w0, x1"
            : [result] "={x0}" (-> u32),
            : [state] "{x0}" (state),
              [word] "{x1}" (word),
        );
    }
    for (bytes[offset..]) |byte| {
        state = asm ("crc32b w0, w0, w1"
            : [result] "={x0}" (-> u32),
            : [state] "{x0}" (state),
              [byte] "{x1}" (@as(u32, byte)),
        );
    }
    return ~state;
}

test "archive CRC preserves IEEE checksum across alignment and tails" {
    try std.testing.expectEqual(@as(u32, 0xcbf43926), hash("123456789"));
    var storage: [1040]u8 = undefined;
    var random = std.Random.DefaultPrng.init(0xc32);
    random.random().bytes(&storage);
    for (0..16) |offset| {
        for (0..1025) |size| {
            const bytes = storage[offset..][0..size];
            try std.testing.expectEqual(std.hash.Crc32.hash(bytes), hash(bytes));
        }
    }
    @memset(&storage, 0);
    try std.testing.expectEqual(std.hash.Crc32.hash(&storage), hash(&storage));
    @memset(&storage, 0xff);
    try std.testing.expectEqual(std.hash.Crc32.hash(&storage), hash(&storage));
}
