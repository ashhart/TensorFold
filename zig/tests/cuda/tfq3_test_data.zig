//! Finite nonzero synthetic TFQ3 records for the runtime's width-invariance contract.
const std = @import("std");
const cuda = @import("cuda");
const fmt = cuda.tfq3_format;

pub const widths = blk: {
    var out: [516]usize = undefined;
    for (0..512) |i| out[i] = i + 1;
    @memcpy(out[512..], &[_]usize{ 513, 1024, 1536, 4218 });
    break :blk out;
};
pub const max_rows = 4218;

pub fn tables() [fmt.tables_file_bytes]u8 {
    var b: [fmt.tables_file_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, b[0..4], fmt.magic_tables, .little);
    std.mem.writeInt(u32, b[4..8], 3, .little);
    for (0..6) |i| std.mem.writeInt(u32, b[8 + i * 4 ..][0..4], @bitCast(@as(f32, 0.125)), .little);
    var off: usize = 64;
    while (off < fmt.tables_min_bytes) : (off += 2) {
        const v: f16 = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((off * 13) % 31)) - 15)) / 32);
        std.mem.writeInt(u16, b[off..][0..2], @bitCast(v), .little);
    }
    return b;
}

pub fn words(gpa: std.mem.Allocator, len: usize) ![]u32 {
    const out = try gpa.alloc(u32, len);
    var rng = std.Random.DefaultPrng.init(0x74667133);
    for (out) |*w| w.* = rng.random().int(u32);
    return out;
}

pub fn inputs(gpa: std.mem.Allocator, k: usize) ![]u16 {
    const out = try gpa.alloc(u16, max_rows * k);
    for (out, 0..) |*x, i| {
        const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + i / k) % 37)) - 18)) / 64;
        x.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }
    return out;
}

pub fn scales(gpa: std.mem.Allocator, len: usize) ![]u16 {
    const out = try gpa.alloc(u16, len);
    @memset(out, @as(u16, @bitCast(@as(f16, 0.125))));
    return out;
}
