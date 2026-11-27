//! Shared TFQ3 binary format headers, validators, and directory discovery (PWV3, TFPN, TFPW v3).
const std = @import("std");

const format = @import("tfq3_format");
pub const tfq4 = format.tfq4;

pub const magic_tables = format.magic_tables;
pub const magic_tfpn = format.magic_nonmoe;
pub const magic_tfpw = format.magic_layer;
pub const format_version = format.format_version;
pub const section_align: u64 = format.section_align;
pub const tables_min_bytes = format.tables_bytes;
pub const tables_file_bytes = format.tables_file_bytes;

pub const TfpnHeader = format.NonMoeHeader;

pub const TfpnTensorMeta = extern struct {
    name: [80]u8,
    r: u32,
    c: u32,
    b_bits: u32,
    sc_off: u32,
    word_off: u64,
    word_count: u64,
    snr_db: f32,
    pad: [3]u32,

    pub fn nameSlice(self: *align(1) const TfpnTensorMeta) []const u8 {
        const end = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..end];
    }
};

pub const TfpwHeader = format.LayerHeader;
pub const TfpwExpertMeta = format.ExpertMeta;

comptime {
    if (@sizeOf(TfpnTensorMeta) != 128) @compileError("TfpnTensorMeta must be 128 bytes");
}

pub fn validBits(bits: u32) bool {
    return bits == 3 or bits == 4 or bits == 5;
}

pub fn validTfpwBits(bits: u32) bool {
    return format.validTfpwBits(bits);
}

pub fn tensorWords(r: u32, c: u32, bits: u32) u64 {
    if (bits == 2) return @as(u64, r) * @as(u64, c / 16);
    if (bits == 225) return @as(u64, r) * @as(u64, c / 128) * 9;
    return @as(u64, r / 16) * @as(u64, c / 128) * @as(u64, bits * 2) * 32;
}

pub fn tensorScales(r: u32, c: u32) u64 {
    return @as(u64, r / 128) * @as(u64, c / 128);
}

/// Strips HF/MLX namespace prefixes so tensor lookups match across layouts.
pub fn canonicalName(name: []const u8) []const u8 {
    var s = name;
    if (std.mem.startsWith(u8, s, "language_model.")) s = s["language_model.".len..];
    if (std.mem.startsWith(u8, s, "model.language_model.")) s = s["model.language_model.".len..];
    if (std.mem.startsWith(u8, s, "model.")) s = s["model.".len..];
    return s;
}

/// Validates a `pw_v3_tables.bin` buffer (magic, version, minimum active length, finite header scales and LUTs).
pub fn validateTables(bytes: []const u8) !void {
    if (bytes.len < tables_min_bytes) return error.InvalidTfq3Tables;
    const magic = std.mem.readInt(u32, bytes[0..4], .little);
    const ver = std.mem.readInt(u32, bytes[4..8], .little);
    if (magic != magic_tables or ver != format_version) return error.InvalidTfq3Tables;
    for (0..6) |i| {
        const v: f32 = @bitCast(std.mem.readInt(u32, bytes[8 + i * 4 ..][0..4], .little));
        if (!std.math.isFinite(v)) return error.InvalidTfq3Tables;
    }
    var at: usize = 64;
    while (at < tables_min_bytes) : (at += 2) {
        if ((std.mem.readInt(u16, bytes[at..][0..2], .little) & 0x7c00) == 0x7c00) return error.InvalidTfq3Tables;
    }
}

/// Validates a `TFPN` shard header and all tensor descriptors against `file_len`.
pub const DenseShard = struct { header: TfpnHeader, metas: []align(1) const TfpnTensorMeta };

pub fn validateTfpn(bytes: []const u8) !DenseShard {
    return validateTfpnNamed(bytes, null);
}

/// On alignment failure, bad_name borrows the offending descriptor in bytes; the caller keeps bytes alive.
pub fn validateTfpnNamed(bytes: []const u8, bad_name: ?*[]const u8) !DenseShard {
    if (bytes.len < @sizeOf(TfpnHeader)) return error.InvalidTfq3Shard;
    const h: TfpnHeader = std.mem.bytesToValue(TfpnHeader, bytes[0..@sizeOf(TfpnHeader)]);
    if (h.magic != magic_tfpn or h.version != format_version) return error.InvalidTfq3Shard;
    if ((h.off_scales % section_align) != 0 or (h.off_weights % section_align) != 0) return error.InvalidTfq3Shard;
    const meta_bytes = try std.math.mul(u64, h.num_tensors, @sizeOf(TfpnTensorMeta));
    const meta_end = try std.math.add(u64, h.off_meta, meta_bytes);
    const sc_bytes = try std.math.mul(u64, h.total_scale_halfs, 2);
    const sc_end = try std.math.add(u64, h.off_scales, sc_bytes);
    const w_bytes = try std.math.mul(u64, h.total_weight_words, 4);
    const w_end = try std.math.add(u64, h.off_weights, w_bytes);
    if (h.off_meta < @sizeOf(TfpnHeader) or meta_end > h.off_scales or sc_end > h.off_weights or w_end > bytes.len) {
        return error.InvalidTfq3Shard;
    }
    const raw_meta = bytes[@intCast(h.off_meta)..@intCast(meta_end)];
    const metas: []align(1) const TfpnTensorMeta = std.mem.bytesAsSlice(TfpnTensorMeta, raw_meta);
    if ((h.pad0 & 0x1) != 0) {
        var hasher = std.hash.XxHash64.init(0);
        hasher.update(raw_meta);
        hasher.update(bytes[@intCast(h.off_scales)..@intCast(sc_end)]);
        hasher.update(bytes[@intCast(h.off_weights)..@intCast(w_end)]);
        if (hasher.final() != h.pad1) return error.Tfq3ChecksumMismatch;
    }
    for (metas) |*m| {
        if (!validBits(m.b_bits) or m.r == 0 or m.c == 0 or (m.r % 128) != 0 or (m.c % 128) != 0) return error.InvalidTfq3Shard;
        if (m.b_bits == 4 and m.word_off % 4 != 0) {
            if (bad_name) |name| name.* = m.nameSlice();
            return error.MisalignedTfq3Tensor;
        }
        const ew = tensorWords(m.r, m.c, m.b_bits);
        const es = tensorScales(m.r, m.c);
        const w_last = std.math.add(u64, m.word_off, m.word_count) catch return error.InvalidTfq3Shard;
        const s_last = std.math.add(u64, m.sc_off, es) catch return error.InvalidTfq3Shard;
        if (m.word_count != ew or w_last > h.total_weight_words or s_last > h.total_scale_halfs) return error.InvalidTfq3Shard;
    }
    return .{ .header = h, .metas = metas };
}

/// Validated `TFPW` MoE layer shard with hybrid TFQ3 (`3/4/5b`) and TFQ4 (`2/225`) bit-width flags.
pub const TfpwShard = struct {
    header: TfpwHeader,
    metas: []align(1) const TfpwExpertMeta,
    has_tfq3: bool,
    has_tfq4: bool,
    needs_e8p1024: bool,
};

/// Validates a `TFPW` MoE layer shard header and all expert descriptors against `file_len` and expected shape.
pub fn validateTfpw(bytes: []const u8, expect_experts: ?u32, expect_hidden: ?u32, expect_inter: ?u32, expect_layer: ?u32) !TfpwShard {
    if (bytes.len < @sizeOf(TfpwHeader)) return error.InvalidTfq3Shard;
    const h: TfpwHeader = std.mem.bytesToValue(TfpwHeader, bytes[0..@sizeOf(TfpwHeader)]);
    if (h.magic != magic_tfpw or h.version != format_version) return error.InvalidTfq3Shard;
    if (expect_layer) |layer| if (h.layer_idx != layer) return error.Tfq3LayerMismatch;
    if (h.num_experts == 0 or h.hidden_size == 0 or h.moe_inter == 0 or (h.hidden_size % 128) != 0 or (h.moe_inter % 128) != 0) return error.InvalidTfq3Shard;
    if (expect_experts) |e| if (h.num_experts != e) return error.InvalidTfq3Shard;
    if (expect_hidden) |d| if (h.hidden_size != d) return error.InvalidTfq3Shard;
    if (expect_inter) |i| if (h.moe_inter != i) return error.InvalidTfq3Shard;
    if ((h.off_scales % section_align) != 0 or (h.off_weights % section_align) != 0) return error.InvalidTfq3Shard;
    const meta_bytes = try std.math.mul(u64, h.num_experts, @sizeOf(TfpwExpertMeta));
    const meta_end = try std.math.add(u64, h.off_meta, meta_bytes);
    const sc_bytes = try std.math.mul(u64, h.scale_halfs, 2);
    const sc_end = try std.math.add(u64, h.off_scales, sc_bytes);
    const w_bytes = try std.math.mul(u64, h.stream_words, 4);
    const w_end = try std.math.add(u64, h.off_weights, w_bytes);
    if (h.off_meta < @sizeOf(TfpwHeader) or meta_end > h.off_scales or sc_end > h.off_weights or w_end > bytes.len) {
        return error.InvalidTfq3Shard;
    }
    const raw_meta = bytes[@intCast(h.off_meta)..@intCast(meta_end)];
    const metas: []align(1) const TfpwExpertMeta = std.mem.bytesAsSlice(TfpwExpertMeta, raw_meta);
    if ((h.reserved0 & 0x1) != 0) {
        if (std.hash.XxHash64.hash(0, raw_meta) != h.reserved1[0]) return error.Tfq3ChecksumMismatch;
        if (std.hash.XxHash64.hash(0, bytes[@intCast(h.off_scales)..@intCast(sc_end)]) != h.reserved1[1]) return error.Tfq3ChecksumMismatch;
        if (std.hash.XxHash64.hash(0, bytes[@intCast(h.off_weights)..@intCast(w_end)]) != h.reserved1[2]) return error.Tfq3ChecksumMismatch;
    }
    const sc_gu = tensorScales(h.moe_inter, h.hidden_size);
    const sc_dn = tensorScales(h.hidden_size, h.moe_inter);
    var has_tfq3 = false;
    var has_tfq4 = false;
    var needs_e8p1024 = false;
    for (metas) |m| {
        if (!validTfpwBits(m.b_bits)) return error.InvalidTfq3Shard;
        if (m.b_bits == 4 and (m.gate_word_off % 4 != 0 or m.up_word_off % 4 != 0 or m.down_word_off % 4 != 0)) return error.MisalignedTfq3Tensor;
        if (tfq4.isTfq4Bits(m.b_bits)) {
            has_tfq4 = true;
            if (m.b_bits == 225) needs_e8p1024 = true;
        } else {
            has_tfq3 = true;
        }
        const gu_w = tensorWords(h.moe_inter, h.hidden_size, m.b_bits);
        const dn_w = tensorWords(h.hidden_size, h.moe_inter, m.b_bits);
        if ((std.math.add(u64, m.gate_word_off, gu_w) catch return error.InvalidTfq3Shard) > h.stream_words) return error.InvalidTfq3Shard;
        if ((std.math.add(u64, m.up_word_off, gu_w) catch return error.InvalidTfq3Shard) > h.stream_words) return error.InvalidTfq3Shard;
        if ((std.math.add(u64, m.down_word_off, dn_w) catch return error.InvalidTfq3Shard) > h.stream_words) return error.InvalidTfq3Shard;
        if ((std.math.add(u64, m.gate_sc_off, sc_gu) catch return error.InvalidTfq3Shard) > h.scale_halfs) return error.InvalidTfq3Shard;
        if ((std.math.add(u64, m.up_sc_off, sc_gu) catch return error.InvalidTfq3Shard) > h.scale_halfs) return error.InvalidTfq3Shard;
        if ((std.math.add(u64, m.down_sc_off, sc_dn) catch return error.InvalidTfq3Shard) > h.scale_halfs) return error.InvalidTfq3Shard;
    }
    return .{
        .header = h,
        .metas = metas,
        .has_tfq3 = has_tfq3,
        .has_tfq4 = has_tfq4,
        .needs_e8p1024 = needs_e8p1024,
    };
}

/// Returns true if `dir/pw_v4_e8p_tables.bin` exists.
pub fn hasTfq4Tables(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    return hasFile(gpa, io, dir, "pw_v4_e8p_tables.bin");
}

/// Returns true if `dir/pw_v3_tables.bin` exists.
pub fn hasTables(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "pw_v3_tables.bin" }) catch return false;
    defer gpa.free(path);
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// Returns true if `dir/<rel>` exists.
pub fn hasFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rel: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, rel }) catch return false;
    defer gpa.free(path);
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// A format header must advertise records; a tables sidecar alone never selects TFQ3.
pub fn hasRecords(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, file: []const u8, magic: u32) bool {
    const path = std.fs.path.join(gpa, &.{ dir, file }) catch return false;
    defer gpa.free(path);
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    defer f.close(io);
    var h: [16]u8 = undefined;
    if ((f.readPositional(io, &.{&h}, 0) catch return false) != h.len) return false;
    const count_at: usize = if (magic == magic_tfpw) 12 else 8;
    return std.mem.readInt(u32, h[0..4], .little) == magic and
        std.mem.readInt(u32, h[4..8], .little) == format_version and
        std.mem.readInt(u32, h[count_at..][0..4], .little) != 0;
}

pub fn hasDense(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    if (!hasTables(gpa, io, dir)) return false;
    const paths = listShards(gpa, io, dir, "") catch return false;
    defer freePaths(gpa, paths);
    for (paths) |path| if (hasRecords(gpa, io, dir, std.fs.path.basename(path), magic_tfpn)) return true;
    return false;
}

/// Returns true if `dir` contains at least one valid `layer_*.tffpw` shard with `num_experts > 0`.
pub fn hasAnyTfpw(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const paths = listShards(gpa, io, dir, "layer_") catch return false;
    defer freePaths(gpa, paths);
    for (paths) |path| {
        if (hasRecords(gpa, io, dir, std.fs.path.basename(path), magic_tfpw)) return true;
    }
    return false;
}

/// Returns true if `dir/layer_{layer_idx:0>2}.tffpw` is a valid `TFPW` v3 shard with `num_experts > 0`.
pub fn hasLayerTfpw(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, layer_idx: u32) bool {
    var name_buf: [32]u8 = undefined;
    const fname = std.fmt.bufPrint(&name_buf, "layer_{d:0>2}.tffpw", .{layer_idx}) catch return false;
    return hasRecords(gpa, io, dir, fname, magic_tfpw);
}

/// Lists sorted files in `dir` matching optional `prefix` and `suffix`.
fn listShardsExt(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, prefix: []const u8, suffix: []const u8) ![][]u8 {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |p| gpa.free(p);
        out.deinit(gpa);
    }
    var it = d.iterate();
    while (try it.next(io)) |ent| {
        if (ent.kind != .file and ent.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, ent.name, suffix)) continue;
        if (prefix.len > 0 and !std.mem.startsWith(u8, ent.name, prefix)) continue;
        const path = try std.fs.path.join(gpa, &.{ dir, ent.name });
        errdefer gpa.free(path);
        try out.append(gpa, path);
    }
    std.mem.sort([]u8, out.items, {}, struct {
        fn less(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.less);
    return out.toOwnedSlice(gpa);
}

/// Lists sorted `.tffpw` files in `dir` matching optional `prefix` (e.g., `"qwen27_tfq3-"`).
pub fn listShards(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, prefix: []const u8) ![][]u8 {
    return listShardsExt(gpa, io, dir, prefix, ".tffpw");
}

/// Lists sorted `.tffp4` files in `dir` matching optional `prefix`.
pub fn listTfp4Shards(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, prefix: []const u8) ![][]u8 {
    return listShardsExt(gpa, io, dir, prefix, ".tffp4");
}

pub fn freePaths(gpa: std.mem.Allocator, paths: [][]u8) void {
    for (paths) |p| gpa.free(p);
    gpa.free(paths);
}

test "tfq3 format validates PWV3, TFPN, and TFPW headers and rejects corruption" {
    var tbl: [tables_file_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], magic_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], format_version, .little);
    try validateTables(&tbl);
    try std.testing.expectError(error.InvalidTfq3Tables, validateTables(tbl[0..1024]));
    std.mem.writeInt(u32, tbl[4..8], 2, .little);
    try std.testing.expectError(error.InvalidTfq3Tables, validateTables(&tbl));

    // Valid minimal TFPN (1 tensor: 128x128 @ 4b -> 2048 words, 1 scale)
    const tfpn_size: usize = 3 * @as(usize, @intCast(section_align));
    const tfpn = try std.testing.allocator.alloc(u8, tfpn_size);
    defer std.testing.allocator.free(tfpn);
    @memset(tfpn, 0);
    const hdr = TfpnHeader{
        .magic = magic_tfpn,
        .version = format_version,
        .num_tensors = 1,
        .pad0 = 0,
        .off_meta = @sizeOf(TfpnHeader),
        .off_scales = section_align,
        .off_weights = 2 * section_align,
        .total_scale_halfs = 1,
        .total_weight_words = tensorWords(128, 128, 4),
        .pad1 = 0,
    };
    @memcpy(tfpn[0..@sizeOf(TfpnHeader)], std.mem.asBytes(&hdr));
    var meta = std.mem.zeroes(TfpnTensorMeta);
    @memcpy(meta.name[0..7], "lm_head");
    meta.r = 128;
    meta.c = 128;
    meta.b_bits = 4;
    meta.word_count = tensorWords(128, 128, 4);
    @memcpy(tfpn[@sizeOf(TfpnHeader)..][0..@sizeOf(TfpnTensorMeta)], std.mem.asBytes(&meta));
    const parsed = try validateTfpn(tfpn);
    try std.testing.expectEqualStrings("lm_head", parsed.metas[0].nameSlice());
    // Out-of-bounds word_off rejected
    meta.word_off = 1;
    @memcpy(tfpn[@sizeOf(TfpnHeader)..][0..@sizeOf(TfpnTensorMeta)], std.mem.asBytes(&meta));
    try std.testing.expectError(error.MisalignedTfq3Tensor, validateTfpn(tfpn));

    // Valid minimal TFPW (1 expert: hidden=128, inter=128 @ 3b -> 1536 words each x 3 = 4608 words = 18432B)
    const tfpw_size: usize = 4 * @as(usize, @intCast(section_align));
    const tfpw = try std.testing.allocator.alloc(u8, tfpw_size);
    defer std.testing.allocator.free(tfpw);
    @memset(tfpw, 0);
    const gw = tensorWords(128, 128, 3);
    const wh = TfpwHeader{
        .magic = magic_tfpw,
        .version = format_version,
        .layer_idx = 3,
        .num_experts = 1,
        .num_tensors = 3,
        .hidden_size = 128,
        .moe_inter = 128,
        .reserved0 = 0,
        .raw_fp8_bytes = 0,
        .stream_words = gw * 3,
        .scale_halfs = 3,
        .off_meta = @sizeOf(TfpwHeader),
        .off_scales = section_align,
        .off_weights = 2 * section_align,
        .reserved1 = .{ 0, 0, 0, 0 },
        .meta_tag = @splat(0),
    };
    @memcpy(tfpw[0..@sizeOf(TfpwHeader)], std.mem.asBytes(&wh));
    var em = std.mem.zeroes(TfpwExpertMeta);
    em.b_bits = 3;
    em.gate_word_off = 0;
    em.up_word_off = gw;
    em.down_word_off = gw * 2;
    em.gate_sc_off = 0;
    em.up_sc_off = 1;
    em.down_sc_off = 2;
    @memcpy(tfpw[@sizeOf(TfpwHeader)..][0..@sizeOf(TfpwExpertMeta)], std.mem.asBytes(&em));
    const pw = try validateTfpw(tfpw, 1, 128, 128, null);
    try std.testing.expectEqual(@as(u32, 3), pw.metas[0].b_bits);
    try std.testing.expect(pw.has_tfq3 and !pw.has_tfq4 and !pw.needs_e8p1024);
    try std.testing.expectError(error.InvalidTfq3Shard, validateTfpw(tfpw, 2, 128, 128, null));
}

test "CUDA TFQ3: validateTfpw accepts hybrid 2/225/3/4/5-bit experts and sets TFQ4 flags" {
    const gpa = std.testing.allocator;
    const bits_all = [_]u32{ 4, 3, 2, 225, 5 };
    var total_words: u64 = 0;
    for (bits_all) |b| total_words += tensorWords(128, 128, b) * 3;
    const w_bytes = total_words * 4;
    const file_len: usize = @intCast(std.mem.alignForward(u64, 2 * section_align + w_bytes, section_align));
    const buf = try gpa.alloc(u8, file_len);
    defer gpa.free(buf);
    @memset(buf, 0);

    var w_cur: u64 = 0;
    var s_cur: u32 = 0;
    for (bits_all, 0..) |b, i| {
        const mw = tensorWords(128, 128, b);
        var em = std.mem.zeroes(TfpwExpertMeta);
        em.b_bits = b;
        em.gate_word_off = w_cur;
        w_cur += mw;
        em.up_word_off = w_cur;
        w_cur += mw;
        em.down_word_off = w_cur;
        w_cur += mw;
        em.gate_sc_off = s_cur;
        s_cur += 1;
        em.up_sc_off = s_cur;
        s_cur += 1;
        em.down_sc_off = s_cur;
        s_cur += 1;
        @memcpy(buf[@sizeOf(TfpwHeader) + i * @sizeOf(TfpwExpertMeta) ..][0..@sizeOf(TfpwExpertMeta)], std.mem.asBytes(&em));
    }
    const wh = TfpwHeader{
        .magic = magic_tfpw,
        .version = format_version,
        .layer_idx = 1,
        .num_experts = bits_all.len,
        .num_tensors = bits_all.len * 3,
        .hidden_size = 128,
        .moe_inter = 128,
        .reserved0 = 0,
        .raw_fp8_bytes = 0,
        .stream_words = total_words,
        .scale_halfs = bits_all.len * 3,
        .off_meta = @sizeOf(TfpwHeader),
        .off_scales = section_align,
        .off_weights = 2 * section_align,
        .reserved1 = .{ 0, 0, 0, 0 },
        .meta_tag = @splat(0),
    };
    @memcpy(buf[0..@sizeOf(TfpwHeader)], std.mem.asBytes(&wh));
    const parsed = try validateTfpw(buf, bits_all.len, 128, 128, 1);
    try std.testing.expect(parsed.has_tfq3 and parsed.has_tfq4 and parsed.needs_e8p1024);

    // Reject unknown bit-width (e.g. 226)
    var bad = std.mem.bytesToValue(TfpwExpertMeta, buf[@sizeOf(TfpwHeader)..][0..@sizeOf(TfpwExpertMeta)]);
    bad.b_bits = 226;
    @memcpy(buf[@sizeOf(TfpwHeader)..][0..@sizeOf(TfpwExpertMeta)], std.mem.asBytes(&bad));
    try std.testing.expectError(error.InvalidTfq3Shard, validateTfpw(buf, bits_all.len, 128, 128, 1));
}
