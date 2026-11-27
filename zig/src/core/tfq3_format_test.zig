//! The TFQ3 records' bytes pinned field by field: offsets, sizes, alignment and a fixture each converter writes.
const std = @import("std");
const f = @import("tfq3_format.zig");
const t = std.testing;

const Field = struct { []const u8, usize, usize }; // name, byte offset, byte size

fn expectLayout(comptime T: type, comptime fields: []const Field, size: usize, alignment: usize) !void {
    try t.expectEqual(size, @sizeOf(T));
    try t.expectEqual(alignment, @alignOf(T));
    const info = @typeInfo(T).@"struct";
    try t.expectEqual(fields.len, info.field_names.len);
    inline for (fields, info.field_names, info.field_types) |want, name, ty| {
        try t.expectEqualStrings(want[0], name);
        try t.expectEqual(want[1], @offsetOf(T, want[0]));
        try t.expectEqual(want[2], @sizeOf(ty));
    }
}

test "TFQ3 format: every record's fields sit where main's three copies put them" {
    try expectLayout(f.LayerHeader, &.{
        .{ "magic", 0, 4 },          .{ "version", 4, 4 },       .{ "layer_idx", 8, 4 },    .{ "num_experts", 12, 4 },
        .{ "num_tensors", 16, 4 },   .{ "hidden_size", 20, 4 },  .{ "moe_inter", 24, 4 },   .{ "reserved0", 28, 4 },
        .{ "raw_fp8_bytes", 32, 8 }, .{ "stream_words", 40, 8 }, .{ "scale_halfs", 48, 8 }, .{ "off_meta", 56, 8 },
        .{ "off_scales", 64, 8 },    .{ "off_weights", 72, 8 },  .{ "reserved1", 80, 32 },  .{ "meta_tag", 112, 32 },
    }, 144, 8);
    try expectLayout(f.ExpertMeta, &.{
        .{ "b_bits", 0, 4 },         .{ "heat_rank", 4, 4 },    .{ "gate_word_off", 8, 8 }, .{ "up_word_off", 16, 8 },
        .{ "down_word_off", 24, 8 }, .{ "gate_sc_off", 32, 4 }, .{ "up_sc_off", 36, 4 },    .{ "down_sc_off", 40, 4 },
        .{ "gate_snr_db", 44, 4 },   .{ "up_snr_db", 48, 4 },   .{ "down_snr_db", 52, 4 },  .{ "reserved", 56, 8 },
    }, 64, 8);
    try expectLayout(f.NonMoeHeader, &.{
        .{ "magic", 0, 4 },               .{ "version", 4, 4 },     .{ "num_tensors", 8, 4 },  .{ "pad0", 12, 4 },
        .{ "off_meta", 16, 8 },           .{ "off_scales", 24, 8 }, .{ "off_weights", 32, 8 }, .{ "total_scale_halfs", 40, 8 },
        .{ "total_weight_words", 48, 8 }, .{ "pad1", 56, 8 },
    }, 64, 8);
}

test "TFQ3 format: the magics spell their file kinds and the sizes are the kernels' and converter's" {
    try t.expectEqualStrings("TFPW", std.mem.asBytes(&std.mem.nativeToLittle(u32, f.magic_layer)));
    try t.expectEqualStrings("TFPN", std.mem.asBytes(&std.mem.nativeToLittle(u32, f.magic_nonmoe)));
    try t.expectEqualStrings("PWV3", std.mem.asBytes(&std.mem.nativeToLittle(u32, f.magic_tables)));
    try t.expectEqual(@as(u32, 3), f.format_version);
    try t.expectEqual(@as(usize, 16384), f.section_align);
    try t.expectEqual(@as(usize, 10816), f.tables_bytes);
    try t.expectEqual(@as(usize, 16384), f.tables_file_bytes);
}

/// A record filled with each field's own index, so any moved, resized or reordered field changes the bytes.
fn numbered(comptime T: type) T {
    var v = std.mem.zeroes(T);
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types, 1..) |name, ty, i| switch (@typeInfo(ty)) {
        .int => @field(v, name) = @intCast(i),
        .float => @field(v, name) = @floatFromInt(i),
        .array => @field(v, name) = @splat(@intCast(i)),
        else => unreachable,
    };
    return v;
}

fn expectBytes(comptime T: type, want_hex: []const u8) !void {
    const v = numbered(T);
    var want: [@sizeOf(T)]u8 = undefined;
    _ = try std.fmt.hexToBytes(&want, want_hex);
    try t.expectEqualSlices(u8, &want, std.mem.asBytes(&v));
}

test "TFQ3 format: a numbered record of each kind serializes to main's bytes" {
    try expectBytes(f.LayerHeader, "01000000" ++ "02000000" ++ "03000000" ++ "04000000" ++ "05000000" ++ "06000000" ++
        "07000000" ++ "08000000" ++ "0900000000000000" ++ "0a00000000000000" ++ "0b00000000000000" ++
        "0c00000000000000" ++ "0d00000000000000" ++ "0e00000000000000" ++
        "0f000000000000000f000000000000000f000000000000000f00000000000000" ++
        "1010101010101010101010101010101010101010101010101010101010101010");
    try expectBytes(f.ExpertMeta, "01000000" ++ "02000000" ++ "0300000000000000" ++ "0400000000000000" ++
        "0500000000000000" ++ "06000000" ++ "07000000" ++ "08000000" ++ "00001041" ++ "00002041" ++ "00003041" ++
        "0c0000000c000000");
    try expectBytes(f.NonMoeHeader, "01000000" ++ "02000000" ++ "03000000" ++ "04000000" ++ "0500000000000000" ++
        "0600000000000000" ++ "0700000000000000" ++ "0800000000000000" ++ "0900000000000000" ++ "0a00000000000000");
}
