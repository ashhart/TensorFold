//! Host-validated Qwen27 TFQ3 binary format checks before any mmap slice or Metal buffer view is created.
const std = @import("std");

pub const magic_nonmoe: u32 = 0x4E504654;
pub const magic_v3_tables: u32 = 0x33565750;
pub const format_version: u32 = 3;
pub const section_align: usize = 16384;
/// Bytes of Q27PwV3Tables (common.metal): magic, version, six affine floats, padding, three LUTs, three c64 tables.
pub const tables_bytes: usize = 10816;
pub const tables_file = "pw_v3_tables.bin";
pub const single_file = "qwen27_tfq3.tffpw";
pub const shard_prefix = "qwen27_tfq3-";
pub const shard_suffix = ".tffpw";

pub const Header = extern struct {
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

pub const TensorMeta = extern struct {
    name: [80]u8,
    R: u32,
    C: u32,
    b_bits: u32,
    sc_off: u32,
    word_off: u64,
    word_count: u64,
    snr_db: f32,
    pad: [3]u32,

    pub fn nameSlice(self: *const TensorMeta) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..len];
    }
};

comptime {
    std.debug.assert(@sizeOf(Header) == 64);
    std.debug.assert(@sizeOf(TensorMeta) == 128);
}

pub const Error = error{
    UnsupportedQuantization,
    Tfq3Truncated,
    Tfq3Overflow,
    Tfq3Misaligned,
    Tfq3Overlap,
    Tfq3Geometry,
    Tfq3Bits,
    Tfq3Table,
    Tfq3RecordShape,
    Tfq3ShardName,
    Tfq3ShardSet,
    Tfq3ShardTooLarge,
    Tfq3BufferTooLarge,
};

/// A validated byte range inside a file: [off, off + len).
pub const Span = struct { off: usize, len: usize };

/// Checked section layout of one TFQ3 dense shard.
pub const ShardLayout = struct { metas: Span, count: usize, scales: Span, weights: Span };

/// Address and geometry of one TFQ3 tensor, used to check contiguous pair fusion.
pub const PairPart = struct { shard: usize, n: usize, k: usize, bits: u8, word_off: u64, word_count: u64, sc_off: u32 };

fn cast(v: u64) Error!usize {
    return std.math.cast(usize, v) orelse error.Tfq3Overflow;
}

fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.Tfq3Overflow;
}

fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.Tfq3Overflow;
}

/// File length rounded up to the 16 KiB no-copy view alignment.
pub fn mappedLen(file_len: usize) Error!usize {
    const rounded = try add(file_len, section_align - 1);
    return rounded / section_align * section_align;
}

pub const max_shard_bytes: u64 = 4 << 30;

/// A shard is capped before its length reaches mmap or file-backed view arithmetic.
pub fn shardLen(len: u64) Error!usize {
    if (len > max_shard_bytes) return error.Tfq3ShardTooLarge;
    return cast(len);
}

/// A rounded no-copy view must fit the device's actual buffer ceiling.
pub fn noCopyLen(section_len: usize, device_limit: u64) Error!usize {
    const len = try mappedLen(@max(section_len, 16));
    if (len > device_limit) return error.Tfq3BufferTooLarge;
    return len;
}
/// Validates a span [off, off + len) inside a file; `gpu` enforces 16 KiB alignment and rounded mapping bounds.
pub fn span(file_len: usize, off: u64, len: usize, gpu: bool) Error!Span {
    const o = try cast(off);
    const end = try add(o, len);
    if (end > file_len) return error.Tfq3Truncated;
    if (gpu) {
        if (o % section_align != 0) return error.Tfq3Misaligned;
        const view = try add(o, try mappedLen(@max(len, 16)));
        if (view > try mappedLen(file_len)) return error.Tfq3Truncated;
    }
    return .{ .off = o, .len = len };
}

/// Refuses spans that overlap each other or intrude into the fixed header.
pub fn disjoint(head: usize, spans: []const Span) Error!void {
    for (spans, 0..) |a, i| {
        if (a.len > 0 and a.off < head) return error.Tfq3Overlap;
        for (spans[i + 1 ..]) |b| {
            if (a.len == 0 or b.len == 0) continue;
            if (a.off < b.off + b.len and b.off < a.off + a.len) return error.Tfq3Overlap;
        }
    }
}

fn read(comptime T: type, bytes: []const u8, off: usize) Error!T {
    const end = try add(off, @sizeOf(T));
    if (end > bytes.len) return error.Tfq3Truncated;
    return std.mem.bytesToValue(T, bytes[off..end]);
}

fn finiteHalf(bits: u16) bool {
    return bits & 0x7c00 != 0x7c00;
}

/// Validates `pw_v3_tables.bin`: magic, version, full table payload, finite affine floats and finite half LUTs.
pub fn tables(bytes: []const u8) Error!void {
    if (bytes.len < 8) return error.Tfq3Truncated;
    if (std.mem.readInt(u32, bytes[0..4], .little) != magic_v3_tables or
        std.mem.readInt(u32, bytes[4..8], .little) != format_version) return error.UnsupportedQuantization;
    if (bytes.len < tables_bytes) return error.Tfq3Table;
    for (0..6) |i| if (!std.math.isFinite(try read(f32, bytes, 8 + 4 * i))) return error.Tfq3Table;
    var at: usize = 64;
    while (at < tables_bytes) : (at += 2) {
        if (!finiteHalf(std.mem.readInt(u16, bytes[at..][0..2], .little))) return error.Tfq3Table;
    }
}

/// Packed u32 words of an R x C TFQ3 matrix at `bits` (3, 4, or 5).
pub fn matWords(r: usize, c: usize, bits: usize) Error!usize {
    return try mul(try mul(r, c), bits) / 32;
}

/// Scale half count of an R x C TFQ3 matrix (one half per 128 x 128 block).
pub fn matScales(r: usize, c: usize) Error!usize {
    return try mul(r / 128, c / 128);
}

/// Validates one dense tensor record: 3/4/5-bit width, 128-aligned shape, and bounds within shard sections.
pub fn tensor(m: TensorMeta, words: usize, halfs: usize) Error!void {
    if (m.b_bits < 3 or m.b_bits > 5) return error.Tfq3Bits;
    if (m.R == 0 or m.C == 0 or m.R % 128 != 0 or m.C % 128 != 0) return error.Tfq3Geometry;
    if (m.nameSlice().len == 0) return error.Tfq3Geometry;
    if (try cast(m.word_count) != try matWords(m.R, m.C, m.b_bits)) return error.Tfq3Geometry;
    if (m.b_bits == 4 and m.word_off % 4 != 0) return error.Tfq3Misaligned;
    if (try add(try cast(m.word_off), try cast(m.word_count)) > words) return error.Tfq3Truncated;
    if (try add(m.sc_off, try matScales(m.R, m.C)) > halfs) return error.Tfq3Truncated;
}

/// Validates one TFQ3 shard header, records, and 16 KiB-aligned GPU sections before forming any buffer views.
pub fn shard(bytes: []const u8) Error!ShardLayout {
    const h = try read(Header, bytes, 0);
    if (h.magic != magic_nonmoe or h.version != format_version) return error.UnsupportedQuantization;
    if (h.num_tensors == 0) return error.Tfq3Geometry;
    const metas = try span(bytes.len, h.off_meta, try mul(h.num_tensors, @sizeOf(TensorMeta)), false);
    if (metas.off % @alignOf(TensorMeta) != 0) return error.Tfq3Misaligned;
    const halfs = try cast(h.total_scale_halfs);
    const words = try cast(h.total_weight_words);
    const out: ShardLayout = .{
        .metas = metas,
        .count = h.num_tensors,
        .scales = try span(bytes.len, h.off_scales, try mul(halfs, 2), true),
        .weights = try span(bytes.len, h.off_weights, try mul(words, 4), true),
    };
    try disjoint(@sizeOf(Header), &.{ out.metas, out.scales, out.weights });
    for (0..out.count) |i| try tensor(try read(TensorMeta, bytes, metas.off + i * @sizeOf(TensorMeta)), words, halfs);
    return out;
}

/// Whether two equal-shape TFQ3 tensors (e.g. gate+up or k+v) are contiguous in one shard and safe to fuse.
pub fn canFusePair(a: PairPart, b: PairPart) bool {
    if (a.shard != b.shard or a.bits != b.bits or a.bits < 3 or a.bits > 5) return false;
    if (a.n == 0 or a.k == 0 or a.n != b.n or a.k != b.k or a.n % 128 != 0 or a.k % 128 != 0) return false;
    const want_w = matWords(a.n, a.k, a.bits) catch return false;
    const want_s = matScales(a.n, a.k) catch return false;
    if (a.word_count != want_w or b.word_count != want_w) return false;
    const next_w = std.math.add(u64, a.word_off, a.word_count) catch return false;
    const next_s = std.math.add(usize, a.sc_off, want_s) catch return false;
    return b.word_off == next_w and b.sc_off == next_s;
}
pub const ShardNumber = struct { index: usize, of: usize };
pub const ShardKind = union(enum) { single, numbered: ShardNumber };

/// Parses one standalone file or the fixed-width numbered grammar; malformed TFQ3 candidates are refusals.
pub fn parseShardName(name: []const u8) Error!?ShardKind {
    if (std.mem.eql(u8, name, single_file)) return .single;
    if (!std.mem.startsWith(u8, name, shard_prefix) or !std.mem.endsWith(u8, name, shard_suffix)) return null;
    if (name.len != shard_prefix.len + 14 + shard_suffix.len) return error.Tfq3ShardName;
    const body = name[shard_prefix.len .. name.len - shard_suffix.len];
    if (!std.mem.eql(u8, body[5..9], "-of-")) return error.Tfq3ShardName;
    for (body[0..5]) |ch| if (!std.ascii.isDigit(ch)) return error.Tfq3ShardName;
    for (body[9..14]) |ch| if (!std.ascii.isDigit(ch)) return error.Tfq3ShardName;
    const index = std.fmt.parseInt(usize, body[0..5], 10) catch return error.Tfq3ShardName;
    const of = std.fmt.parseInt(usize, body[9..14], 10) catch return error.Tfq3ShardName;
    if (index == 0 or of == 0 or index > of) return error.Tfq3ShardName;
    return .{ .numbered = .{ .index = index, .of = of } };
}

/// The lexical names must be either one standalone file or exactly indexes 1...N of one N-shard set.
pub fn validateSortedShardSet(names: []const []const u8) Error!void {
    if (names.len == 0) return;
    for (names, 0..) |name, i| {
        const kind = (try parseShardName(name)) orelse return error.Tfq3ShardSet;
        switch (kind) {
            .single => if (names.len != 1) return error.Tfq3ShardSet,
            .numbered => |part| if (part.index != i + 1 or part.of != names.len) return error.Tfq3ShardSet,
        }
    }
}

/// Strips optional `language_model.` prefix so tensor lookup works uniformly across namespace layouts.
pub fn canonicalModule(name: []const u8) []const u8 {
    if (std.mem.startsWith(u8, name, "language_model.")) return name["language_model.".len..];
    return name;
}

test {
    _ = @import("tfq3_check_test.zig");
}
