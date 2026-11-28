//! Host-only unit tests for Qwen27 TFQ3 binary format validation, pair fusion checks, and shard naming.
const std = @import("std");
const c = @import("tfq3_check.zig");

fn validTables(a: std.mem.Allocator) ![]u8 {
    const buf = try a.alloc(u8, c.section_align);
    @memset(buf, 0);
    std.mem.writeInt(u32, buf[0..4], c.magic_v3_tables, .little);
    std.mem.writeInt(u32, buf[4..8], c.format_version, .little);
    return buf;
}

fn validShard(a: std.mem.Allocator, bits: u32, r: u32, cols: u32) ![]u8 {
    const words = try c.matWords(r, cols, bits);
    const halfs = try c.matScales(r, cols);
    const off_meta: u64 = 64;
    const off_scales: u64 = c.section_align;
    const off_weights: u64 = 2 * c.section_align;
    const total = off_weights + try c.mappedLen(words * 4);
    const buf = try a.alloc(u8, total);
    @memset(buf, 0);
    const h: c.Header = .{
        .magic = c.magic_nonmoe,
        .version = c.format_version,
        .num_tensors = 1,
        .pad0 = 0,
        .off_meta = off_meta,
        .off_scales = off_scales,
        .off_weights = off_weights,
        .total_scale_halfs = halfs,
        .total_weight_words = words,
        .pad1 = 0,
    };
    @memcpy(buf[0..@sizeOf(c.Header)], std.mem.asBytes(&h));
    var m: c.TensorMeta = std.mem.zeroes(c.TensorMeta);
    const name = "model.layers.0.mlp.gate_proj";
    @memcpy(m.name[0..name.len], name);
    m.R = r;
    m.C = cols;
    m.b_bits = bits;
    m.sc_off = 0;
    m.word_off = 0;
    m.word_count = words;
    @memcpy(buf[off_meta..][0..@sizeOf(c.TensorMeta)], std.mem.asBytes(&m));
    return buf;
}

test "qwen27 tfq3: valid tables and 3/4/5-bit shards pass every check" {
    const a = std.testing.allocator;
    const tbl = try validTables(a);
    defer a.free(tbl);
    try c.tables(tbl);
    for ([_]u32{ 3, 4, 5 }) |bits| {
        const sh = try validShard(a, bits, 256, 512);
        defer a.free(sh);
        const layout = try c.shard(sh);
        try std.testing.expectEqual(@as(usize, 1), layout.count);
        try std.testing.expectEqual(@as(usize, c.section_align), layout.scales.off);
        try std.testing.expectEqual(@as(usize, 2 * c.section_align), layout.weights.off);
    }
}

test "qwen27 tfq3: malformed tables refuse with named errors" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.Tfq3Truncated, c.tables(&.{ 0, 1, 2 }));
    const tbl = try validTables(a);
    defer a.free(tbl);
    tbl[0] = 0;
    try std.testing.expectError(error.UnsupportedQuantization, c.tables(tbl));
    std.mem.writeInt(u32, tbl[0..4], c.magic_v3_tables, .little);
    try std.testing.expectError(error.Tfq3Table, c.tables(tbl[0 .. c.tables_bytes - 2]));
    const nan_f32: u32 = 0x7fc00000;
    std.mem.writeInt(u32, tbl[8..12], nan_f32, .little);
    try std.testing.expectError(error.Tfq3Table, c.tables(tbl));
    std.mem.writeInt(u32, tbl[8..12], 0, .little);
    std.mem.writeInt(u16, tbl[64..66], 0x7e00, .little);
    try std.testing.expectError(error.Tfq3Table, c.tables(tbl));
}

test "qwen27 tfq3: malformed shards refuse bad magic, bits, alignment, overlap, geometry and truncation" {
    const a = std.testing.allocator;
    const base = try validShard(a, 3, 256, 512);
    defer a.free(base);
    try std.testing.expectError(error.Tfq3Truncated, c.shard(base[0..32]));
    try std.testing.expectError(error.Tfq3Truncated, c.shard(base[0 .. base.len - 4]));

    const bad_magic = try a.dupe(u8, base);
    defer a.free(bad_magic);
    bad_magic[0] ^= 1;
    try std.testing.expectError(error.UnsupportedQuantization, c.shard(bad_magic));

    const bad_bits = try a.dupe(u8, base);
    defer a.free(bad_bits);
    var m = std.mem.bytesToValue(c.TensorMeta, bad_bits[64..][0..@sizeOf(c.TensorMeta)]);
    m.b_bits = 2;
    @memcpy(bad_bits[64..][0..@sizeOf(c.TensorMeta)], std.mem.asBytes(&m));
    try std.testing.expectError(error.Tfq3Bits, c.shard(bad_bits));
    m.b_bits = 6;
    @memcpy(bad_bits[64..][0..@sizeOf(c.TensorMeta)], std.mem.asBytes(&m));
    try std.testing.expectError(error.Tfq3Bits, c.shard(bad_bits));

    const bad_geom = try a.dupe(u8, base);
    defer a.free(bad_geom);
    m.b_bits = 3;
    m.R = 192;
    @memcpy(bad_geom[64..][0..@sizeOf(c.TensorMeta)], std.mem.asBytes(&m));
    try std.testing.expectError(error.Tfq3Geometry, c.shard(bad_geom));

    const bad_align = try a.dupe(u8, base);
    defer a.free(bad_align);
    var h = std.mem.bytesToValue(c.Header, bad_align[0..@sizeOf(c.Header)]);
    h.off_scales += 64;
    @memcpy(bad_align[0..@sizeOf(c.Header)], std.mem.asBytes(&h));
    try std.testing.expectError(error.Tfq3Misaligned, c.shard(bad_align));

    const bad_overlap = try a.dupe(u8, base);
    defer a.free(bad_overlap);
    h = std.mem.bytesToValue(c.Header, bad_overlap[0..@sizeOf(c.Header)]);
    h.off_weights = h.off_scales;
    @memcpy(bad_overlap[0..@sizeOf(c.Header)], std.mem.asBytes(&h));
    try std.testing.expectError(error.Tfq3Overlap, c.shard(bad_overlap));
}

test "qwen27 tfq3: 4-bit records require uint4 word alignment while 3-bit and 5-bit do not" {
    const a = std.testing.allocator;
    for ([_]u32{ 3, 4, 5 }) |bits| {
        var sh = try validShard(a, bits, 128, 128);
        defer a.free(sh);
        const old_len = sh.len;
        sh = try a.realloc(sh, old_len + c.section_align);
        @memset(sh[old_len..], 0);
        var h = std.mem.bytesToValue(c.Header, sh[0..@sizeOf(c.Header)]);
        h.total_weight_words *= 2;
        @memcpy(sh[0..@sizeOf(c.Header)], std.mem.asBytes(&h));
        var m = std.mem.bytesToValue(c.TensorMeta, sh[64..][0..@sizeOf(c.TensorMeta)]);
        for ([_]u64{ 1, 2, 3, 4 }) |off| {
            m.word_off = off;
            @memcpy(sh[64..][0..@sizeOf(c.TensorMeta)], std.mem.asBytes(&m));
            if (bits == 4 and off != 4) {
                try std.testing.expectError(error.Tfq3Misaligned, c.shard(sh));
            } else {
                _ = try c.shard(sh);
            }
        }
    }
}

test "qwen27 tfq3: shard and rounded no-copy limits refuse before allocation" {
    try std.testing.expectEqual(@as(usize, c.max_shard_bytes), try c.shardLen(c.max_shard_bytes));
    try std.testing.expectError(error.Tfq3ShardTooLarge, c.shardLen(c.max_shard_bytes + 1));
    try std.testing.expectEqual(@as(usize, c.max_shard_bytes), try c.noCopyLen(c.max_shard_bytes, c.max_shard_bytes));
    try std.testing.expectEqual(@as(usize, c.max_shard_bytes), try c.noCopyLen(c.max_shard_bytes - 1, c.max_shard_bytes));
    try std.testing.expectError(error.Tfq3BufferTooLarge, c.noCopyLen(c.max_shard_bytes + 1, c.max_shard_bytes));
    try std.testing.expectError(error.Tfq3BufferTooLarge, c.noCopyLen(c.section_align + 1, c.section_align));
    try std.testing.expectEqual(c.section_align, try c.noCopyLen(0, c.section_align));
}

test "qwen27 tfq3: pair fusion requires same shard, same bits, matching shapes and contiguous offsets" {
    const w = try c.matWords(17408, 5120, 3);
    const s = try c.matScales(17408, 5120);
    const p0: c.PairPart = .{ .shard = 0, .n = 17408, .k = 5120, .bits = 3, .word_off = 0, .word_count = w, .sc_off = 0 };
    const p1: c.PairPart = .{ .shard = 0, .n = 17408, .k = 5120, .bits = 3, .word_off = w, .word_count = w, .sc_off = @intCast(s) };
    try std.testing.expect(c.canFusePair(p0, p1));
    var diff_shard = p1;
    diff_shard.shard = 1;
    try std.testing.expect(!c.canFusePair(p0, diff_shard));
    var diff_bits = p1;
    diff_bits.bits = 4;
    try std.testing.expect(!c.canFusePair(p0, diff_bits));
    var gap = p1;
    gap.word_off += 1;
    try std.testing.expect(!c.canFusePair(p0, gap));
}

test "qwen27 tfq3: numbered shards are one exact sorted set" {
    const one = "qwen27_tfq3-00001-of-00003.tffpw";
    const two = "qwen27_tfq3-00002-of-00003.tffpw";
    const three = "qwen27_tfq3-00003-of-00003.tffpw";
    try c.validateSortedShardSet(&.{ one, two, three });
    try c.validateSortedShardSet(&.{c.single_file});
    try std.testing.expectError(error.Tfq3ShardSet, c.validateSortedShardSet(&.{ one, two }));
    try std.testing.expectError(error.Tfq3ShardSet, c.validateSortedShardSet(&.{ one, one, three }));
    try std.testing.expectError(error.Tfq3ShardSet, c.validateSortedShardSet(&.{"qwen27_tfq3-00002-of-00002.tffpw"}));
    try std.testing.expectError(error.Tfq3ShardSet, c.validateSortedShardSet(&.{ c.single_file, "qwen27_tfq3-00001-of-00001.tffpw" }));
    try std.testing.expectError(error.Tfq3ShardSet, c.validateSortedShardSet(&.{ "qwen27_tfq3-00001-of-00002.tffpw", "qwen27_tfq3-00002-of-00003.tffpw" }));
}

test "qwen27 tfq3: malformed candidates refuse while ordinary files are ignored" {
    const single = (try c.parseShardName(c.single_file)).?;
    try std.testing.expect(single == .single);
    const numbered = (try c.parseShardName("qwen27_tfq3-00004-of-00004.tffpw")).?;
    try std.testing.expectEqual(@as(usize, 4), numbered.numbered.index);
    try std.testing.expectEqual(@as(usize, 4), numbered.numbered.of);
    for ([_][]const u8{
        "qwen27_tfq3-backup.tffpw",
        "qwen27_tfq3-.tffpw",
        "qwen27_tfq3-1-of-00004.tffpw",
        "qwen27_tfq3-000001-of-00004.tffpw",
        "qwen27_tfq3-00001-of-4.tffpw",
        "qwen27_tfq3-00001-of-00004-extra.tffpw",
        "qwen27_tfq3-0000x-of-00004.tffpw",
        "qwen27_tfq3-00000-of-00004.tffpw",
        "qwen27_tfq3-00005-of-00004.tffpw",
        "qwen27_tfq3-00001-of-00000.tffpw",
    }) |name| try std.testing.expectError(error.Tfq3ShardName, c.parseShardName(name));
    try std.testing.expect((try c.parseShardName("layer_00.tffpw")) == null);
    try std.testing.expect((try c.parseShardName("model.safetensors")) == null);
    try std.testing.expectEqualStrings("model.layers.0.mlp.gate_proj", c.canonicalModule("language_model.model.layers.0.mlp.gate_proj"));
}
