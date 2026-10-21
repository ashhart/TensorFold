//! Kimi's zero-copy tensor entries, admitted on the host before creating any device buffer.
const std = @import("std");
const st = @import("safetensors");

pub const Entry = struct { offset: usize, bytes: usize, dtype: st.DType, shape: [3]u32, rank: u8 };
pub const Entries = std.StringHashMapUnmanaged(Entry);

pub fn shortName(name: []const u8) ?[]const u8 {
    for ([_][]const u8{ "language_model.model.", "language_model." }) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) return name[prefix.len..];
    }
    return null;
}

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Entries {
    if (bytes.len < 8) return error.BadSafetensors;
    const header_len = try st.headerSize(std.mem.readInt(u64, bytes[0..8], .little), bytes.len);
    const data = 8 + header_len;
    if (data % 8 != 0) return error.MisalignedShard;
    const header = try st.parseHeaderPrefix(a, bytes[8..data], bytes.len - data, "language_model.");
    var out: Entries = .empty;
    for (header.keys(), header.values()) |key, value| {
        const name = shortName(key) orelse continue;
        if (value.rank > 3) return error.RankTooHigh;
        var entry: Entry = .{
            .offset = data + value.begin,
            .bytes = value.end - value.begin,
            .dtype = value.dtype,
            .shape = .{ 1, 1, 1 },
            .rank = value.rank,
        };
        if (entry.dtype != .bf16 and entry.dtype != .f32 and entry.dtype != .u8) return error.UnsupportedDType;
        for (value.shape[0..value.rank], 0..) |dim, i|
            entry.shape[i] = std.math.cast(u32, dim) orelse return error.BadSafetensors;
        if (entry.offset % 8 != 0) return error.MisalignedTensor;
        const slot = try out.getOrPut(a, name);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = entry;
    }
    return out;
}

test {
    _ = @import("shard_header_test.zig");
}
