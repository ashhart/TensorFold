//! Host unit tests for Inkling 2 Mini Metal TFQ3/TFQ4 table and hybrid TFPW layer validation.
const std = @import("std");
const tfq3 = @import("tfq3.zig");

fn buildTestTfpw(
    gpa: std.mem.Allocator,
    layer_idx: u32,
    hidden: u32,
    inter: u32,
    bits: []const u32,
    with_checksum: bool,
) ![]u8 {
    const sc_gu = try tfq3.matScales(inter, hidden);
    const sc_dn = try tfq3.matScales(hidden, inter);
    var total_words: u64 = 0;
    var total_scales: u64 = 0;
    for (bits) |b| {
        const gu_w = try tfq3.matWords(inter, hidden, b);
        const dn_w = try tfq3.matWords(hidden, inter, b);
        total_words += gu_w * 2 + dn_w;
        total_scales += sc_gu * 2 + sc_dn;
    }
    const meta_off: u64 = @sizeOf(tfq3.LayerHeader);
    const meta_bytes: u64 = @as(u64, bits.len) * @sizeOf(tfq3.ExpertMeta);
    const sc_off: u64 = std.mem.alignForward(u64, meta_off + meta_bytes, tfq3.section_align);
    const sc_bytes: u64 = total_scales * 2;
    const w_off: u64 = std.mem.alignForward(u64, sc_off + sc_bytes, tfq3.section_align);
    const w_bytes: u64 = total_words * 4;
    const file_len: usize = @intCast(std.mem.alignForward(u64, w_off + w_bytes, tfq3.section_align));

    const buf = try gpa.alloc(u8, file_len);
    @memset(buf, 0);
    var w_cur: u64 = 0;
    var s_cur: u32 = 0;
    for (bits, 0..) |b, e| {
        const gu_w = try tfq3.matWords(inter, hidden, b);
        const dn_w = try tfq3.matWords(hidden, inter, b);
        var em = std.mem.zeroes(tfq3.ExpertMeta);
        em.b_bits = b;
        em.gate_word_off = w_cur;
        w_cur += gu_w;
        em.up_word_off = w_cur;
        w_cur += gu_w;
        em.down_word_off = w_cur;
        w_cur += dn_w;
        em.gate_sc_off = s_cur;
        s_cur += @intCast(sc_gu);
        em.up_sc_off = s_cur;
        s_cur += @intCast(sc_gu);
        em.down_sc_off = s_cur;
        s_cur += @intCast(sc_dn);
        @memcpy(buf[@intCast(meta_off + e * @sizeOf(tfq3.ExpertMeta))..][0..@sizeOf(tfq3.ExpertMeta)], std.mem.asBytes(&em));
    }
    var hdr = tfq3.LayerHeader{
        .magic = tfq3.magic_layer,
        .version = tfq3.format_version,
        .layer_idx = layer_idx,
        .num_experts = @intCast(bits.len),
        .num_tensors = @intCast(bits.len * 3),
        .hidden_size = hidden,
        .moe_inter = inter,
        .reserved0 = if (with_checksum) 1 else 0,
        .raw_fp8_bytes = 0,
        .stream_words = total_words,
        .scale_halfs = total_scales,
        .off_meta = meta_off,
        .off_scales = sc_off,
        .off_weights = w_off,
        .reserved1 = .{ 0, 0, 0, 0 },
        .meta_tag = @splat(0),
    };
    if (with_checksum) {
        hdr.reserved1[0] = std.hash.XxHash64.hash(0, buf[@intCast(meta_off)..][0..@intCast(meta_bytes)]);
        hdr.reserved1[1] = std.hash.XxHash64.hash(0, buf[@intCast(sc_off)..][0..@intCast(sc_bytes)]);
        hdr.reserved1[2] = std.hash.XxHash64.hash(0, buf[@intCast(w_off)..][0..@intCast(w_bytes)]);
    }
    @memcpy(buf[0..@sizeOf(tfq3.LayerHeader)], std.mem.asBytes(&hdr));
    return buf;
}

test "Inkling-2 Metal TFQ3 validates PWV3 and PWV4 tables and rejects malformed inputs" {
    var tbl: [tfq3.tables_file_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], tfq3.magic_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], tfq3.format_version, .little);
    try tfq3.validateTables(&tbl);
    try std.testing.expectError(error.InvalidTfq3Tables, tfq3.validateTables(tbl[0..1024]));
    std.mem.writeInt(u16, tbl[64..66], 0x7c00, .little);
    try std.testing.expectError(error.InvalidTfq3Tables, tfq3.validateTables(&tbl));
    std.mem.writeInt(u16, tbl[64..66], 0, .little);
    std.mem.writeInt(u32, tbl[8..12], 0x7fc00000, .little);
    try std.testing.expectError(error.InvalidTfq3Tables, tfq3.validateTables(&tbl));

    var tbl4_2: [tfq3.tfq4.tables_256_bytes]u8 = undefined;
    try tfq3.tfq4.fillTrueE8pTables(&tbl4_2, 2);
    const h2 = try tfq3.validateTables4(&tbl4_2);
    try std.testing.expectEqual(@as(u32, 2), h2.table_kind);

    var tbl4_3: [tfq3.tfq4.tables_1024_bytes]u8 = undefined;
    try tfq3.tfq4.fillTrueE8pTables(&tbl4_3, 3);
    const h3 = try tfq3.validateTables4(&tbl4_3);
    try std.testing.expectEqual(@as(u32, 3), h3.table_kind);
    try std.testing.expectError(error.InvalidTfq4Tables, tfq3.validateTables4(tbl4_3[0..16384]));
}

test "Inkling-2 Metal TFQ3 validates TFPW v3 header, XXH64 checksums, hybrid 2/225/3/4/5b, and 4b alignment" {
    const gpa = std.testing.allocator;
    const checked = try buildTestTfpw(gpa, 7, 2304, 1152, &.{ 2, 225, 3, 4, 5 }, true);
    defer gpa.free(checked);
    const want: tfq3.LayerWant = .{ .layer = 7, .experts = 5, .hidden = 2304, .inter = 1152 };
    const parsed = try tfq3.validateLayer(checked, want);
    try std.testing.expect(parsed.has_5b);
    try std.testing.expect(parsed.has_tfq3);
    try std.testing.expect(parsed.has_tfq4);
    try std.testing.expect(parsed.needs_e8p1024);
    try std.testing.expectEqual(@as(usize, @sizeOf(tfq3.LayerHeader)), parsed.meta.off);
    try std.testing.expectEqual(@as(usize, 0), parsed.scales.off % tfq3.section_align);
    try std.testing.expectEqual(@as(usize, 0), parsed.weights.off % tfq3.section_align);

    for ([_]usize{ parsed.meta.off, parsed.scales.off, parsed.weights.off }) |flip_at| {
        checked[flip_at] ^= 0x01;
        try std.testing.expectError(error.Tfq3ChecksumMismatch, tfq3.validateLayer(checked, want));
        checked[flip_at] ^= 0x01;
    }

    try std.testing.expectError(error.Tfq3LayerMismatch, tfq3.validateLayer(checked, .{ .layer = 8, .experts = 5, .hidden = 2304, .inter = 1152 }));
    try std.testing.expectError(error.Tfq3LayerMismatch, tfq3.validateLayer(checked, .{ .layer = 7, .experts = 128, .hidden = 2304, .inter = 1152 }));

    const legacy = try buildTestTfpw(gpa, 0, 128, 128, &.{ 3, 4 }, false);
    defer gpa.free(legacy);
    const lwant: tfq3.LayerWant = .{ .layer = 0, .experts = 2, .hidden = 128, .inter = 128 };
    const lparsed = try tfq3.validateLayer(legacy, lwant);
    try std.testing.expect(!lparsed.has_5b);
    try std.testing.expect(!lparsed.has_tfq4);
    try std.testing.expect(!lparsed.needs_e8p1024);

    var em = std.mem.bytesToValue(tfq3.ExpertMeta, legacy[lparsed.meta.off + @sizeOf(tfq3.ExpertMeta) ..][0..@sizeOf(tfq3.ExpertMeta)]);
    em.gate_word_off += 1;
    @memcpy(legacy[lparsed.meta.off + @sizeOf(tfq3.ExpertMeta) ..][0..@sizeOf(tfq3.ExpertMeta)], std.mem.asBytes(&em));
    try std.testing.expectError(error.Tfq3Misaligned, tfq3.validateLayer(legacy, lwant));

    em.gate_word_off -= 1;
    em.b_bits = 6;
    @memcpy(legacy[lparsed.meta.off + @sizeOf(tfq3.ExpertMeta) ..][0..@sizeOf(tfq3.ExpertMeta)], std.mem.asBytes(&em));
    try std.testing.expectError(error.Tfq3Bits, tfq3.validateLayer(legacy, lwant));
}
