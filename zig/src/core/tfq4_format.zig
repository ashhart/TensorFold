//! TFQ4 (`PWV4` v4 + `TFP4` v4, 2.001/2.251 bpw 8D E8P) on-disk records, validators, and reference decoder.
const std = @import("std");

pub const magic_tables: u32 = 0x34565750; // "PWV4": pw_v4_e8p_tables.bin
pub const magic_tfp4: u32 = 0x34504654; // "TFP4": *.tffp4 dense shards
pub const format_version: u32 = 4;
pub const dim: u32 = 8;
pub const section_align: usize = 16384;
pub const lut_offset: usize = 64;
pub const flag_xxh64_present: u32 = 1;

pub const table_kind_legacy_trained: u32 = 0;
pub const table_kind_trained: u32 = 1;
pub const table_kind_true_e8p: u32 = 2;
pub const table_kind_true_e8p1024: u32 = 3;
pub const table_kind_max: u32 = 3;

pub const k_codes_256: u32 = 256;
pub const k_codes_1024: u32 = 1024;
pub const lut_256_bytes: usize = 256 * 8 * 2;
pub const lut_1024_bytes: usize = 1024 * 8 * 2;
pub const tables_256_bytes: usize = 16384;
pub const tables_1024_bytes: usize = 32768;

pub const PwV4Header = extern struct {
    magic: u32,
    version: u32,
    k_codes: u32,
    dim: u32,
    table_kind: u32,
    pad0: [11]u32,
};

pub const Tfp4Header = extern struct {
    magic: u32,
    version: u32,
    num_tensors: u32,
    pad0: u32,
    off_meta: u64,
    off_scales: u64,
    off_weights: u64,
    total_scale_halfs: u64,
    total_weight_words: u64,
    pad1: u64,
};

pub const Tfp4TensorMeta = extern struct {
    name: [80]u8,
    r: u32,
    c: u32,
    b_bits: u32,
    sc_off: u32,
    word_off: u64,
    word_count: u64,
    snr_db: f32,
    pad: [3]u32,

    pub fn nameSlice(self: *align(1) const Tfp4TensorMeta) []const u8 {
        const end = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..end];
    }

    pub fn origR(self: *align(1) const Tfp4TensorMeta) u32 {
        return if (self.pad[0] != 0) self.pad[0] else self.r;
    }

    pub fn origC(self: *align(1) const Tfp4TensorMeta) u32 {
        return if (self.pad[1] != 0) self.pad[1] else self.c;
    }

    pub fn flags(self: *align(1) const Tfp4TensorMeta) u32 {
        return self.pad[2];
    }
};

comptime {
    std.debug.assert(@sizeOf(PwV4Header) == 64);
    std.debug.assert(@sizeOf(Tfp4Header) == 64);
    std.debug.assert(@sizeOf(Tfp4TensorMeta) == 128);
}

pub const Error = error{
    InvalidTfq4Tables,
    InvalidTfq4Shard,
    InvalidTfq4Bits,
    InvalidTfq4Geometry,
    Tfq4ChecksumMismatch,
    Tfq4Overflow,
};

pub const PwV4TablesInfo = struct {
    header: PwV4Header,
    k_codes: u32,
    table_kind: u32,
    lut_bytes: usize,
    file_bytes: usize,
};

pub const Tfp4Shard = struct {
    header: Tfp4Header,
    metas: []align(1) const Tfp4TensorMeta,
    needs_e8p1024: bool,
};

pub fn isTfq4Bits(bits: u32) bool {
    return bits == 2 or bits == 225;
}

pub fn isValidHybridBits(bits: u32) bool {
    return bits == 2 or bits == 225 or bits == 3 or bits == 4 or bits == 5;
}

/// Whether a `PWV4` table of `table_kind` can decode `b_bits` (`table_kind=3` decodes both `2` and `225`).
pub fn tableKindSupportsBits(table_kind: u32, b_bits: u32) bool {
    if (b_bits == 2) return table_kind <= table_kind_max;
    if (b_bits == 225) return table_kind == table_kind_true_e8p1024;
    return false;
}

/// Packed `u32` weight word count for a TFQ4 `[r, c]` matrix (`b_bits == 2` or `225`).
pub fn tfq4MatWords(r: u32, c: u32, b_bits: u32) Error!u64 {
    if (r == 0 or c == 0 or (r % 128) != 0 or (c % 128) != 0) return error.InvalidTfq4Geometry;
    return switch (b_bits) {
        2 => @as(u64, r) * @as(u64, c / 16),
        225 => @as(u64, r) * @as(u64, c / 128) * 9,
        else => error.InvalidTfq4Bits,
    };
}

/// Packed `u32` weight word count for a hybrid `TFPW` `[r, c]` matrix (`b_bits in {2, 225, 3, 4, 5}`).
pub fn hybridMatWords(r: u32, c: u32, b_bits: u32) Error!u64 {
    if (r == 0 or c == 0 or (r % 128) != 0 or (c % 128) != 0) return error.InvalidTfq4Geometry;
    return switch (b_bits) {
        2 => @as(u64, r) * @as(u64, c / 16),
        225 => @as(u64, r) * @as(u64, c / 128) * 9,
        3, 4, 5 => @as(u64, r / 16) * @as(u64, c / 128) * @as(u64, b_bits * 2) * 32,
        else => error.InvalidTfq4Bits,
    };
}

pub fn matScales(r: u32, c: u32) u64 {
    return @as(u64, r / 128) * @as(u64, c / 128);
}

fn finiteHalf(bits: u16) bool {
    return (bits & 0x7c00) != 0x7c00;
}

/// Validates a `pw_v4_e8p_tables.bin` buffer (`table_kind = 0/1/2` at 16 KiB, `table_kind = 3` at 32 KiB).
pub fn validatePwV4Tables(bytes: []const u8) Error!PwV4TablesInfo {
    if (bytes.len < @sizeOf(PwV4Header)) return error.InvalidTfq4Tables;
    const h: PwV4Header = std.mem.bytesToValue(PwV4Header, bytes[0..@sizeOf(PwV4Header)]);
    if (h.magic != magic_tables or h.version != format_version or h.dim != dim) return error.InvalidTfq4Tables;
    if (h.table_kind > table_kind_max) return error.InvalidTfq4Tables;
    const expect_k: u32 = if (h.table_kind == table_kind_true_e8p1024) k_codes_1024 else k_codes_256;
    const expect_lut: usize = if (h.table_kind == table_kind_true_e8p1024) lut_1024_bytes else lut_256_bytes;
    const expect_file: usize = if (h.table_kind == table_kind_true_e8p1024) tables_1024_bytes else tables_256_bytes;
    if (h.k_codes != expect_k or bytes.len != expect_file) return error.InvalidTfq4Tables;

    const is_true_e8p = h.table_kind == table_kind_true_e8p or h.table_kind == table_kind_true_e8p1024;
    var at: usize = lut_offset;
    const end: usize = lut_offset + expect_lut;
    while (at < end) : (at += 2) {
        const raw = std.mem.readInt(u16, bytes[at..][0..2], .little);
        if (!finiteHalf(raw)) return error.InvalidTfq4Tables;
        const v: f32 = @floatCast(@as(f16, @bitCast(raw)));
        if (v < 0.0) return error.InvalidTfq4Tables;
        if (is_true_e8p) {
            if (v < 0.5) return error.InvalidTfq4Tables;
            const two_v = v * 2.0;
            const rounded = @round(two_v);
            if (@abs(two_v - rounded) > 1e-3) return error.InvalidTfq4Tables;
            const iv: u32 = @intFromFloat(rounded);
            if ((iv & 1) == 0) return error.InvalidTfq4Tables;
        }
    }
    return .{
        .header = h,
        .k_codes = expect_k,
        .table_kind = h.table_kind,
        .lut_bytes = expect_lut,
        .file_bytes = expect_file,
    };
}

/// Validates a `TFP4` v4 dense shard header, section alignment/bounds, optional XXH64 hash, and tensor records.
pub fn validateTfp4Shard(bytes: []const u8) Error!Tfp4Shard {
    if (bytes.len < @sizeOf(Tfp4Header)) return error.InvalidTfq4Shard;
    const h: Tfp4Header = std.mem.bytesToValue(Tfp4Header, bytes[0..@sizeOf(Tfp4Header)]);
    if (h.magic != magic_tfp4 or h.version != format_version) return error.InvalidTfq4Shard;
    if ((h.off_scales % section_align) != 0 or (h.off_weights % section_align) != 0) return error.InvalidTfq4Shard;
    const meta_bytes = std.math.mul(u64, h.num_tensors, @sizeOf(Tfp4TensorMeta)) catch return error.Tfq4Overflow;
    const meta_end = std.math.add(u64, h.off_meta, meta_bytes) catch return error.Tfq4Overflow;
    const sc_bytes = std.math.mul(u64, h.total_scale_halfs, 2) catch return error.Tfq4Overflow;
    const sc_end = std.math.add(u64, h.off_scales, sc_bytes) catch return error.Tfq4Overflow;
    const w_bytes = std.math.mul(u64, h.total_weight_words, 4) catch return error.Tfq4Overflow;
    const w_end = std.math.add(u64, h.off_weights, w_bytes) catch return error.Tfq4Overflow;
    if (h.off_meta < @sizeOf(Tfp4Header) or meta_end > h.off_scales or sc_end > h.off_weights or w_end > bytes.len) {
        return error.InvalidTfq4Shard;
    }
    const raw_meta = bytes[@intCast(h.off_meta)..@intCast(meta_end)];
    const metas: []align(1) const Tfp4TensorMeta = std.mem.bytesAsSlice(Tfp4TensorMeta, raw_meta);
    if ((h.pad0 & flag_xxh64_present) != 0) {
        var hasher = std.hash.XxHash64.init(0);
        hasher.update(raw_meta);
        hasher.update(bytes[@intCast(h.off_scales)..@intCast(sc_end)]);
        hasher.update(bytes[@intCast(h.off_weights)..@intCast(w_end)]);
        if (hasher.final() != h.pad1) return error.Tfq4ChecksumMismatch;
    }
    var needs_e8p1024 = false;
    for (metas) |*m| {
        if (!isTfq4Bits(m.b_bits)) return error.InvalidTfq4Bits;
        if (m.r == 0 or m.c == 0 or (m.r % 128) != 0 or (m.c % 128) != 0) return error.InvalidTfq4Geometry;
        if (m.nameSlice().len == 0) return error.InvalidTfq4Shard;
        const ew = try tfq4MatWords(m.r, m.c, m.b_bits);
        const es = matScales(m.r, m.c);
        const w_last = std.math.add(u64, m.word_off, m.word_count) catch return error.Tfq4Overflow;
        const s_last = std.math.add(u64, m.sc_off, es) catch return error.Tfq4Overflow;
        if (m.word_count != ew or w_last > h.total_weight_words or s_last > h.total_scale_halfs) {
            return error.InvalidTfq4Shard;
        }
        needs_e8p1024 = needs_e8p1024 or (m.b_bits == 225);
    }
    return .{ .header = h, .metas = metas, .needs_e8p1024 = needs_e8p1024 };
}

/// Normative 8D TFQ4 codeword decode (`b_bits == 2` or `225`, `table_kind == 0/1/2/3`) into `out[0..8]`.
pub fn decodeCodeword(low16: u16, hi2: u2, b_bits: u32, table_kind: u32, lut_f16: []align(1) const u16, out: *[8]f64) void {
    const abs_idx: usize = if (b_bits == 225)
        (@as(usize, low16 & 0xFF) | (@as(usize, hi2) << 8))
    else
        @as(usize, low16 & 0xFF);
    const row = lut_f16[abs_idx * 8 ..][0..8];
    if (table_kind == table_kind_legacy_trained or table_kind == table_kind_trained) {
        const smask: u32 = low16 >> 8;
        for (0..8) |d| {
            const v: f64 = @floatCast(@as(f16, @bitCast(row[d])));
            const sgn: f64 = if (((smask >> @intCast(d)) & 1) != 0) -1.0 else 1.0;
            out[d] = v * sgn;
        }
        return;
    }
    const s7: u32 = (low16 >> 8) & 0x7F;
    const coset: u32 = (low16 >> 15) & 1;
    var m_par: u32 = 0;
    for (0..8) |d| {
        const v: f32 = @floatCast(@as(f16, @bitCast(row[d])));
        m_par += @intFromFloat(@floor(v));
    }
    const s7_bit: u32 = (@as(u32, @popCount(s7)) ^ (m_par & 1)) & 1;
    const smask: u32 = s7 | (s7_bit << 7);
    for (0..8) |d| {
        const v: f64 = @floatCast(@as(f16, @bitCast(row[d])));
        const sgn: f64 = if (((smask >> @intCast(d)) & 1) != 0) -1.0 else 1.0;
        const q = v * sgn - 0.25;
        out[d] = if (coset != 0) -q else q;
    }
}

/// Decodes one 128-column block of a TFQ4 row (`8` words for `b_bits=2`, `9` words for `b_bits=225`) scaled by `sc`.
pub fn decodeRowBlock128(blk: []const u32, b_bits: u32, table_kind: u32, lut_f16: []align(1) const u16, sc: f64, out: *[128]f64) void {
    const hi_word: u32 = if (b_bits == 225) blk[8] else 0;
    for (0..16) |j| {
        const packed_w = blk[j >> 1];
        const low16: u16 = if ((j & 1) == 0) @truncate(packed_w & 0xFFFF) else @truncate(packed_w >> 16);
        const hi2: u2 = if (b_bits == 225) @truncate((hi_word >> @intCast(2 * j)) & 3) else 0;
        var q: [8]f64 = undefined;
        decodeCodeword(low16, hi2, b_bits, table_kind, lut_f16, &q);
        for (0..8) |d| out[j * 8 + d] = q[d] * sc;
    }
}

/// Builds deterministic `E8 + 1/4` LUT (`table_kind = 2` for 256, `table_kind = 3` for 1024).
pub fn fillTrueE8pTables(buf: []u8, table_kind: u32) Error!void {
    const k_codes: usize = switch (table_kind) {
        table_kind_true_e8p => k_codes_256,
        table_kind_true_e8p1024 => k_codes_1024,
        else => return error.InvalidTfq4Tables,
    };
    const need_bytes: usize = if (k_codes == k_codes_1024) tables_1024_bytes else tables_256_bytes;
    if (buf.len != need_bytes) return error.InvalidTfq4Tables;
    @memset(buf, 0);
    const hdr: PwV4Header = .{
        .magic = magic_tables,
        .version = format_version,
        .k_codes = @intCast(k_codes),
        .dim = dim,
        .table_kind = table_kind,
        .pad0 = @splat(0),
    };
    @memcpy(buf[0..@sizeOf(PwV4Header)], std.mem.asBytes(&hdr));
    const Cand = struct { n2_x4: u32, max_m: u8, m: [8]u8 };
    var cands: [6561]Cand = undefined;
    var count: usize = 0;
    for (0..6561) |code| {
        var tmp = code;
        var m: [8]u8 = undefined;
        var n2: u32 = 0;
        var mx: u8 = 0;
        var d: usize = 8;
        while (d > 0) {
            d -= 1;
            const md: u8 = @intCast(tmp % 3);
            tmp /= 3;
            m[d] = md;
            const v2: u32 = 2 * @as(u32, md) + 1;
            n2 += v2 * v2;
            if (md > mx) mx = md;
        }
        cands[count] = .{ .n2_x4 = n2, .max_m = mx, .m = m };
        count += 1;
    }
    std.mem.sort(Cand, cands[0..count], {}, struct {
        fn lt(_: void, a: Cand, b: Cand) bool {
            if (a.n2_x4 != b.n2_x4) return a.n2_x4 < b.n2_x4;
            if (a.max_m != b.max_m) return a.max_m < b.max_m;
            return std.mem.order(u8, &a.m, &b.m) == .lt;
        }
    }.lt);
    for (0..k_codes) |k| {
        for (0..8) |d| {
            const val: f16 = @floatCast(@as(f32, @floatFromInt(cands[k].m[d])) + 0.5);
            const at = lut_offset + (k * 8 + d) * 2;
            std.mem.writeInt(u16, buf[at..][0..2], @bitCast(val), .little);
        }
    }
}

test "TFQ4 format: record layouts, magics, and word count formulas match spec" {
    const t = std.testing;
    try t.expectEqualStrings("PWV4", std.mem.asBytes(&std.mem.nativeToLittle(u32, magic_tables)));
    try t.expectEqualStrings("TFP4", std.mem.asBytes(&std.mem.nativeToLittle(u32, magic_tfp4)));
    try t.expectEqual(@as(usize, 64), @sizeOf(PwV4Header));
    try t.expectEqual(@as(usize, 16), @offsetOf(PwV4Header, "table_kind"));
    try t.expectEqual(@as(usize, 64), @sizeOf(Tfp4Header));
    try t.expectEqual(@as(usize, 128), @sizeOf(Tfp4TensorMeta));
    try t.expectEqual(@as(u64, 1024), try tfq4MatWords(128, 128, 2));
    try t.expectEqual(@as(u64, 1152), try tfq4MatWords(128, 128, 225));
    try t.expectEqual(@as(u64, 1536), try hybridMatWords(128, 128, 3));
    try t.expectEqual(@as(u64, 2048), try hybridMatWords(128, 128, 4));
    try t.expectEqual(@as(u64, 2560), try hybridMatWords(128, 128, 5));
    try t.expectError(error.InvalidTfq4Bits, tfq4MatWords(128, 128, 3));
    try t.expectError(error.InvalidTfq4Geometry, tfq4MatWords(64, 128, 2));
    try t.expect(tableKindSupportsBits(0, 2) and !tableKindSupportsBits(0, 225));
    try t.expect(tableKindSupportsBits(2, 2) and !tableKindSupportsBits(2, 225));
    try t.expect(tableKindSupportsBits(3, 2) and tableKindSupportsBits(3, 225));
}

test "TFQ4 format: PWV4 v4 validation covers table_kind 0..3 and rejects malformed tables" {
    const t = std.testing;
    var tbl256: [tables_256_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, tbl256[0..4], magic_tables, .little);
    std.mem.writeInt(u32, tbl256[4..8], format_version, .little);
    std.mem.writeInt(u32, tbl256[8..12], k_codes_256, .little);
    std.mem.writeInt(u32, tbl256[12..16], dim, .little);
    std.mem.writeInt(u32, tbl256[16..20], table_kind_legacy_trained, .little);
    const info0 = try validatePwV4Tables(&tbl256);
    try t.expectEqual(@as(u32, 256), info0.k_codes);
    try t.expectEqual(@as(u32, 0), info0.table_kind);

    std.mem.writeInt(u32, tbl256[16..20], table_kind_trained, .little);
    try t.expectEqual(@as(u32, 1), (try validatePwV4Tables(&tbl256)).table_kind);

    std.mem.writeInt(u32, tbl256[16..20], table_kind_true_e8p, .little);
    try t.expectError(error.InvalidTfq4Tables, validatePwV4Tables(&tbl256));

    try fillTrueE8pTables(&tbl256, table_kind_true_e8p);
    try t.expectEqual(@as(u32, 2), (try validatePwV4Tables(&tbl256)).table_kind);

    var tbl1024: [tables_1024_bytes]u8 = @splat(0);
    try fillTrueE8pTables(&tbl1024, table_kind_true_e8p1024);
    const info3 = try validatePwV4Tables(&tbl1024);
    try t.expectEqual(@as(u32, 1024), info3.k_codes);
    try t.expectEqual(@as(u32, 3), info3.table_kind);
    try t.expectEqualSlices(u8, tbl256[lut_offset .. lut_offset + lut_256_bytes], tbl1024[lut_offset .. lut_offset + lut_256_bytes]);

    std.mem.writeInt(u32, tbl1024[16..20], 4, .little);
    try t.expectError(error.InvalidTfq4Tables, validatePwV4Tables(&tbl1024));
    try fillTrueE8pTables(&tbl1024, table_kind_true_e8p1024);
    std.mem.writeInt(u16, tbl1024[lut_offset..][0..2], @bitCast(@as(f16, 1.0)), .little);
    try t.expectError(error.InvalidTfq4Tables, validatePwV4Tables(&tbl1024));
    try t.expectError(error.InvalidTfq4Tables, validatePwV4Tables(tbl1024[0..tables_256_bytes]));
}

test "TFQ4 format: TFP4 shard validation checks bounds, b_bits 2/225, and XXH64 checksum" {
    const t = std.testing;
    const gpa = t.allocator;
    const w2 = try tfq4MatWords(128, 128, 2);
    const w225 = try tfq4MatWords(128, 128, 225);
    const tot_w = w2 + w225;
    const file_len = 3 * section_align;
    const buf = try gpa.alloc(u8, file_len);
    defer gpa.free(buf);
    @memset(buf, 0);

    var m0 = std.mem.zeroes(Tfp4TensorMeta);
    @memcpy(m0.name[0..6], "proj_2");
    m0.r = 128;
    m0.c = 128;
    m0.b_bits = 2;
    m0.sc_off = 0;
    m0.word_off = 0;
    m0.word_count = w2;

    var m1 = std.mem.zeroes(Tfp4TensorMeta);
    @memcpy(m1.name[0..8], "proj_225");
    m1.r = 128;
    m1.c = 128;
    m1.b_bits = 225;
    m1.sc_off = 1;
    m1.word_off = w2;
    m1.word_count = w225;
    m1.pad = .{ 100, 120, 1 };

    @memcpy(buf[64..192], std.mem.asBytes(&m0));
    @memcpy(buf[192..320], std.mem.asBytes(&m1));

    var h: Tfp4Header = .{
        .magic = magic_tfp4,
        .version = format_version,
        .num_tensors = 2,
        .pad0 = flag_xxh64_present,
        .off_meta = 64,
        .off_scales = section_align,
        .off_weights = 2 * section_align,
        .total_scale_halfs = 2,
        .total_weight_words = tot_w,
        .pad1 = 0,
    };
    var hasher = std.hash.XxHash64.init(0);
    hasher.update(buf[64..320]);
    hasher.update(buf[section_align .. section_align + 4]);
    hasher.update(buf[2 * section_align .. 2 * section_align + @as(usize, @intCast(tot_w)) * 4]);
    h.pad1 = hasher.final();
    @memcpy(buf[0..64], std.mem.asBytes(&h));

    const parsed = try validateTfp4Shard(buf);
    try t.expect(parsed.needs_e8p1024);
    try t.expectEqualStrings("proj_2", parsed.metas[0].nameSlice());
    try t.expectEqual(@as(u32, 100), parsed.metas[1].origR());
    try t.expectEqual(@as(u32, 120), parsed.metas[1].origC());

    buf[2 * section_align] ^= 0x1;
    try t.expectError(error.Tfq4ChecksumMismatch, validateTfp4Shard(buf));
}

test "TFQ4 format: True E8P decode produces E8 + 1/4 lattice vectors for b_bits=2 and b_bits=225" {
    const t = std.testing;
    var tbl: [tables_1024_bytes]u8 = undefined;
    try fillTrueE8pTables(&tbl, table_kind_true_e8p1024);
    const lut: []align(1) const u16 = std.mem.bytesAsSlice(u16, tbl[lut_offset .. lut_offset + lut_1024_bytes]);
    for ([_]u32{ 2, 225 }) |bits| {
        for ([_]u16{ 0x0000, 0x7f13, 0x80ff, 0xaa55, 0xffff }) |low16| {
            const hi2: u2 = if (bits == 225) 3 else 0;
            var q: [8]f64 = undefined;
            decodeCodeword(low16, hi2, bits, table_kind_true_e8p1024, lut, &q);
            var sum_d8: i64 = 0;
            const coset = (low16 >> 15) & 1;
            for (q) |x| {
                const y = if (coset != 0) (-x + 0.25) else (x + 0.25);
                const m = @round(y - 0.5);
                try t.expect(@abs((y - 0.5) - m) < 1e-6);
                sum_d8 += @intFromFloat(m);
            }
            try t.expectEqual(@as(i64, 0), @mod(sum_d8, 2));
        }
    }
}
