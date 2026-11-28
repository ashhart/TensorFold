//! Flash Next TFQ3 shard records and the checks a shard passes before admission, budgeting or a pointer into it.
const std = @import("std");

const format = @import("tfq3_format");

pub const magic_layer = format.magic_layer;
pub const magic_v3_tables = format.magic_tables;
pub const format_version = format.format_version;
/// The routed kernels' fixed shape (moe_decode.metal): 512 experts of 640 x 2560 gate/up and 2560 x 640 down.
pub const num_experts: u32 = 512;
pub const hidden_size: u32 = 2560;
pub const moe_inter: u32 = 640;
/// The kernels' FnPwV3Tables (flashnext_tfq3/common.metal:40): a file holds at least it (published ones pad to 16 KiB).
pub const tables_bytes = format.tables_bytes;

pub const LayerHeader = format.LayerHeader;
pub const ExpertMeta = format.ExpertMeta;

/// Each refusal names the boundary it guards.
pub const Error = error{
    UnsupportedQuantization, // magic or version
    Tfq3Truncated, // a record or section past the file's end
    Tfq3Overflow, // an offset, count or extent that does not fit the arithmetic
    Tfq3Overlap, // two sections, or a section and the header, share bytes
    Tfq3Misaligned, // a 4-bit projection whose words the kernels' uint4 loads would read off 16 bytes
    Tfq3Geometry, // experts, tensors or widths the kernels do not take
    Tfq3Bits, // an expert width the kernels do not decode
    Tfq3Layer, // a layer file holding another layer
    Tfq3Table, // TFQ3 shards without the shared table, or a table too short or not version 3
};

/// A byte range of a file: [off, off + len).
pub const Span = struct { off: usize, len: usize };

/// A checked shard: its three sections, their extents in words and halfs, and the 5-bit routed specialization.
pub const Layout = struct { meta: Span, scales: Span, weights: Span, words: usize, halfs: usize, has_5b: bool = false };

fn cast(v: u64) Error!usize {
    return std.math.cast(usize, v) orelse error.Tfq3Overflow;
}

fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.Tfq3Overflow;
}

fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.Tfq3Overflow;
}

fn span(file_len: usize, off: u64, len: usize) Error!Span {
    const o = try cast(off);
    if (try add(o, len) > file_len) return error.Tfq3Truncated;
    return .{ .off = o, .len = len };
}

fn read(comptime T: type, bytes: []const u8, off: usize) Error!T {
    const end = try add(off, @sizeOf(T));
    if (end > bytes.len) return error.Tfq3Truncated;
    return std.mem.bytesToValue(T, bytes[off..end]);
}

/// A shard's header (its first 144 bytes) against the file's length and, when given, the layer it must hold.
pub fn header(head: []const u8, file_len: usize, want_layer: ?u32) Error!Layout {
    const h = try read(LayerHeader, head, 0);
    if (h.magic != magic_layer or h.version != format_version) return error.UnsupportedQuantization;
    if (want_layer) |want| if (h.layer_idx != want) return error.Tfq3Layer;
    if (h.num_experts != num_experts or h.num_tensors != 3 * num_experts) return error.Tfq3Geometry;
    if (h.hidden_size != hidden_size or h.moe_inter != moe_inter) return error.Tfq3Geometry;
    const words = try cast(h.stream_words);
    const halfs = try cast(h.scale_halfs);
    const out: Layout = .{
        .meta = try span(file_len, h.off_meta, @as(usize, num_experts) * @sizeOf(ExpertMeta)),
        .scales = try span(file_len, h.off_scales, try mul(halfs, 2)),
        .weights = try span(file_len, h.off_weights, try mul(words, 4)),
        .words = words,
        .halfs = halfs,
    };
    const all = [_]Span{ out.meta, out.scales, out.weights };
    for (all, 0..) |a, i| {
        if (a.len > 0 and a.off < @sizeOf(LayerHeader)) return error.Tfq3Overlap;
        for (all[i + 1 ..]) |b| if (a.len > 0 and b.len > 0 and a.off < b.off + b.len and b.off < a.off + a.len) return error.Tfq3Overlap;
    }
    return out;
}

/// Words of an R x C matrix at `bits` (16-row strips of 128-column blocks, bits * 2 words a lane) and its halfs.
pub fn matWords(r: usize, c: usize, bits: usize) Error!usize {
    return try mul(try mul(r, c), bits) / 32;
}

pub fn matScales(r: usize, c: usize) usize {
    return (r / 128) * (c / 128);
}

/// Every expert's width, its gate, up and down words (4-bit: uint4-aligned) and scales in the sections; the 5-bit flag.
pub fn experts(meta: []const u8, l: *Layout) Error!void {
    if (meta.len < l.meta.len) return error.Tfq3Truncated;
    for (0..num_experts) |e| {
        const m = try read(ExpertMeta, meta, e * @sizeOf(ExpertMeta));
        if (m.b_bits < 3 or m.b_bits > 5) return error.Tfq3Bits;
        l.has_5b = l.has_5b or m.b_bits == 5;
        const io = [3][2]usize{ .{ moe_inter, hidden_size }, .{ moe_inter, hidden_size }, .{ hidden_size, moe_inter } };
        const w_offs = [3]u64{ m.gate_word_off, m.up_word_off, m.down_word_off };
        const s_offs = [3]u32{ m.gate_sc_off, m.up_sc_off, m.down_sc_off };
        for (io, w_offs, s_offs) |rc, wo, so| {
            // 4-bit words are read four at a time as uint4 (moe_mv, prefill); 3/5-bit words and scales one at a time
            if (m.b_bits == 4 and wo % 4 != 0) return error.Tfq3Misaligned;
            if (try add(try cast(wo), try matWords(rc[0], rc[1], m.b_bits)) > l.words) return error.Tfq3Truncated;
            if (try add(so, matScales(rc[0], rc[1])) > l.halfs) return error.Tfq3Truncated;
        }
    }
}

/// A whole shard held in memory (the loader's mapping): header, sections and every expert's matrices.
pub fn layer(bytes: []const u8, want: ?u32) Error!Layout {
    var l = try header(bytes, bytes.len, want);
    try experts(bytes[l.meta.off..][0..l.meta.len], &l);
    return l;
}

fn finiteHalf(bits: u16) bool {
    return bits & 0x7c00 != 0x7c00;
}

/// The table file's first `tables_bytes` and length: version 3, the whole table, its floats and halves finite (as GLM).
pub fn tables(head: []const u8, file_len: usize) Error!void {
    if (head.len < 8 or file_len < tables_bytes) return error.Tfq3Table;
    if (std.mem.readInt(u32, head[0..4], .little) != magic_v3_tables or std.mem.readInt(u32, head[4..8], .little) != format_version) return error.UnsupportedQuantization;
    if (head.len < tables_bytes) return error.Tfq3Table;
    for (0..6) |k| if (!std.math.isFinite(try read(f32, head, 8 + 4 * k))) return error.Tfq3Table;
    var at: usize = 64; // lut3, lut4, lut5 and the three c64 tables, all halves
    while (at < tables_bytes) : (at += 2) if (!finiteHalf(std.mem.readInt(u16, head[at..][0..2], .little))) return error.Tfq3Table;
}

test {
    _ = @import("tfq3_check_test.zig");
}
