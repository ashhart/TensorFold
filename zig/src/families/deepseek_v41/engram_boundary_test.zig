const std = @import("std");
const t = std.testing;
const eh = @import("engram_host.zig");
const er = @import("engram_rows.zig");
test "truncated legacy Engram metadata is rejected before reading fields" {
    var b: [24]u8 = @splat(0);
    @memcpy(b[0..8], "DSV41EH1");
    std.mem.writeInt(u32, b[8..12], 2, .little);
    std.mem.writeInt(u32, b[12..16], 4, .little);
    std.mem.writeInt(u32, b[16..20], 100, .little);
    var ids: [4]u32 = undefined;
    try t.expectError(error.BadEngramHost, eh.Host.loadLegacyFixture(&b, 8, 2, 16000000, &ids));
}
test "Engram shard cannot claim rows past the declared full table" {
    var b: [48]u8 = undefined;
    for ([_]u64{ er.magic, 1, 10, 12, 11, 264 }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    try t.expectError(error.BadEngramShard, er.Header.parse(&b, 4096 + 2 * 264));
}
test "UE8M0 scale zero is the reference minimum positive exponent" {
    const lut = try t.allocator.create([65536]u16);
    defer t.allocator.destroy(lut);
    er.fillTable(lut);
    // E4M3FN 448 * 2^-127 remains representable in BF16.
    try t.expectEqual(@as(u16, 0x0460), lut[0x7e]);
}
