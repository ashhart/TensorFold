//! TFQ3's file records and the checks every file passes before a pointer is formed or a byte is read through it.
const std = @import("std");

pub const magic_layer: u32 = 0x57504654;
pub const magic_nonmoe: u32 = 0x4E504654;
pub const magic_v3_tables: u32 = 0x33565750;
pub const format_version: u32 = 3;
pub const section_align: usize = 16384;
/// Bytes of the kernels' PwV3Tables (common.metal): magic, version, six affine floats, padding, three LUTs, three c64s.
pub const tables_bytes: usize = 10816;
/// The routed kernels' fixed shape (moe_decode.metal: 4096-wide rotated rows, 2048 outputs a projection).
pub const expert_hidden: u32 = 4096;
pub const expert_inter: u32 = 2048;

pub const LayerHeader = extern struct {
    magic: u32,
    version: u32,
    layer_idx: u32,
    num_experts: u32,
    num_tensors: u32,
    hidden_size: u32,
    moe_inter: u32,
    reserved0: u32,
    raw_fp8_bytes: u64,
    stream_words: u64,
    scale_halfs: u64,
    off_meta: u64,
    off_scales: u64,
    off_weights: u64,
    reserved1: [4]u64,
    meta_tag: [32]u8,
};

pub const ExpertMeta = extern struct {
    b_bits: u32,
    heat_rank: u32,
    gate_word_off: u64,
    up_word_off: u64,
    down_word_off: u64,
    gate_sc_off: u32,
    up_sc_off: u32,
    down_sc_off: u32,
    gate_snr_db: f32,
    up_snr_db: f32,
    down_snr_db: f32,
    reserved: [2]u32,
};

pub const NonMoeHeader = extern struct {
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

pub const NonMoeTensorMeta = extern struct {
    name: [80]u8,
    R: u32,
    C: u32,
    b_bits: u32,
    sc_off: u32,
    word_off: u64,
    word_count: u64,
    snr_db: f32,
    pad: [3]u32,

    pub fn nameSlice(self: *const NonMoeTensorMeta) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..len];
    }
};

comptime {
    std.debug.assert(@sizeOf(LayerHeader) == 144);
    std.debug.assert(@sizeOf(ExpertMeta) == 64);
    std.debug.assert(@sizeOf(NonMoeHeader) == 64);
    std.debug.assert(@sizeOf(NonMoeTensorMeta) == 128);
}

/// Each refusal names the boundary it guards.
pub const Error = error{
    UnsupportedQuantization, // magic or version
    Tfq3Truncated, // a record or section past the file's end
    Tfq3Overflow, // an offset, count or extent that does not fit the arithmetic
    Tfq3Misaligned, // a GPU section not on a 16 KiB boundary, or host records off their alignment
    Tfq3Overlap, // two sections, or a section and the header, share bytes
    Tfq3Geometry, // dimensions, counts or extents the kernels do not take
    Tfq3Bits, // a width the kernels do not decode
    Tfq3Layer, // a layer file's index, experts or widths differ from the checkpoint's
    Tfq3Table, // the shared table's payload is short or not finite
    Tfq3RecordShape, // a nonmoe.tffpw record whose rows or columns are not the model's for its name
    Tfq3RecordSlice, // a nonmoe.tffpw record the model would read only part of
    Tfq3TensorParallel, // TFQ3 files with --tp: their records are whole tensors, never a rank's share
};

/// A byte range of a file: [off, off + len).
pub const Span = struct { off: usize, len: usize };

fn cast(v: u64) Error!usize {
    return std.math.cast(usize, v) orelse error.Tfq3Overflow;
}

fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.Tfq3Overflow;
}

fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.Tfq3Overflow;
}

/// The bytes a mapping of `file_len` holds once rounded to whole sections (what no-copy views may cover).
pub fn mappedLen(file_len: usize) Error!usize {
    const rounded = try add(file_len, section_align - 1);
    return rounded / section_align * section_align;
}

/// A span `off`..`off + len` inside the file; `gpu`: a no-copy view, 16 KiB aligned, its rounded length in the mapping.
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

/// Refuses spans that share a byte; `head` is the fixed header's length, which no section may cover.
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

/// The shared table file: magic, version, the kernels' whole table present, its floats and halves finite.
pub fn tables(bytes: []const u8) Error!void {
    if (bytes.len < 8) return error.Tfq3Truncated;
    if (std.mem.readInt(u32, bytes[0..4], .little) != magic_v3_tables or std.mem.readInt(u32, bytes[4..8], .little) != format_version) return error.UnsupportedQuantization;
    if (bytes.len < tables_bytes) return error.Tfq3Table;
    for (0..6) |i| if (!std.math.isFinite(try read(f32, bytes, 8 + 4 * i))) return error.Tfq3Table;
    var at: usize = 64; // lut3, lut4, lut5 and the three c64 tables, all halves
    while (at < tables_bytes) : (at += 2) if (!finiteHalf(std.mem.readInt(u16, bytes[at..][0..2], .little))) return error.Tfq3Table;
}

/// nonmoe.tffpw's checked layout: where its records, scales and weights are.
pub const NonMoeLayout = struct { metas: Span, count: usize, scales: Span, weights: Span };

/// Words of an R x C TFQ3 matrix at `bits`, and its half scales (one a 128 x 128 block).
pub fn matWords(r: usize, c: usize, bits: usize) Error!usize {
    return try mul(try mul(r, c), bits) / 32;
}

pub fn matScales(r: usize, c: usize) Error!usize {
    return try mul(r / 128, c / 128);
}

/// One dense tensor's record: a 4 or 5 bit width, whole 128 blocks, and its extent inside the stream and scales.
pub fn tensor(m: NonMoeTensorMeta, words: usize, halfs: usize) Error!void {
    if (m.b_bits != 4 and m.b_bits != 5) return error.Tfq3Bits;
    if (m.R == 0 or m.C == 0 or m.R % 128 != 0 or m.C % 128 != 0) return error.Tfq3Geometry;
    if (m.nameSlice().len == 0) return error.Tfq3Geometry;
    if (try cast(m.word_count) != try matWords(m.R, m.C, m.b_bits)) return error.Tfq3Geometry;
    if (try add(try cast(m.word_off), try cast(m.word_count)) > words) return error.Tfq3Truncated;
    if (try add(m.sc_off, try matScales(m.R, m.C)) > halfs) return error.Tfq3Truncated;
}

/// nonmoe.tffpw: its header, records, and both GPU sections, before any of them is read through a pointer.
pub fn nonMoe(bytes: []const u8) Error!NonMoeLayout {
    const h = try read(NonMoeHeader, bytes, 0);
    if (h.magic != magic_nonmoe or h.version != format_version) return error.UnsupportedQuantization;
    if (h.num_tensors == 0) return error.Tfq3Geometry;
    const metas = try span(bytes.len, h.off_meta, try mul(h.num_tensors, @sizeOf(NonMoeTensorMeta)), false);
    if (metas.off % @alignOf(NonMoeTensorMeta) != 0) return error.Tfq3Misaligned;
    const halfs = try cast(h.total_scale_halfs);
    const words = try cast(h.total_weight_words);
    const out: NonMoeLayout = .{
        .metas = metas,
        .count = h.num_tensors,
        .scales = try span(bytes.len, h.off_scales, try mul(halfs, 2), true),
        .weights = try span(bytes.len, h.off_weights, try mul(words, 4), true),
    };
    try disjoint(@sizeOf(NonMoeHeader), &.{ out.metas, out.scales, out.weights });
    for (0..out.count) |i| try tensor(try read(NonMoeTensorMeta, bytes, metas.off + i * @sizeOf(NonMoeTensorMeta)), words, halfs);
    return out;
}

/// What the checkpoint expects of layer `layer`'s file.
pub const LayerWant = struct { layer: usize, experts: usize, hidden: usize, inter: usize };

/// layer_NN.tffpw's checked layout, and whether any expert is 5-bit (the routed kernels' specialization).
pub const LayerLayout = struct { meta: Span, scales: Span, weights: Span, has_5b: bool };

/// layer_NN.tffpw: index, experts and widths against `want`; every expert's three projections inside both sections.
pub fn layer(bytes: []const u8, want: LayerWant) Error!LayerLayout {
    const h = try read(LayerHeader, bytes, 0);
    if (h.magic != magic_layer or h.version != format_version) return error.UnsupportedQuantization;
    if (h.layer_idx != want.layer or h.num_experts != want.experts or h.hidden_size != want.hidden or h.moe_inter != want.inter) return error.Tfq3Layer;
    if (h.num_experts == 0 or h.num_tensors != try mul(h.num_experts, 3)) return error.Tfq3Geometry;
    if (h.hidden_size != expert_hidden or h.moe_inter != expert_inter) return error.Tfq3Geometry;
    const words = try cast(h.stream_words);
    const halfs = try cast(h.scale_halfs);
    const out: LayerLayout = .{
        .meta = try span(bytes.len, h.off_meta, try mul(h.num_experts, @sizeOf(ExpertMeta)), true),
        .scales = try span(bytes.len, h.off_scales, try mul(halfs, 2), true),
        .weights = try span(bytes.len, h.off_weights, try mul(words, 4), true),
        .has_5b = false,
    };
    try disjoint(@sizeOf(LayerHeader), &.{ out.meta, out.scales, out.weights });
    var has_5b = false;
    for (0..h.num_experts) |e| {
        const m = try read(ExpertMeta, bytes, out.meta.off + e * @sizeOf(ExpertMeta));
        if (m.b_bits < 3 or m.b_bits > 5) return error.Tfq3Bits;
        has_5b = has_5b or m.b_bits == 5;
        const io = [3][2]usize{ .{ h.moe_inter, h.hidden_size }, .{ h.moe_inter, h.hidden_size }, .{ h.hidden_size, h.moe_inter } };
        const w_offs = [3]u64{ m.gate_word_off, m.up_word_off, m.down_word_off };
        const s_offs = [3]u32{ m.gate_sc_off, m.up_sc_off, m.down_sc_off };
        for (io, w_offs, s_offs) |rc, wo, so| {
            if (try add(try cast(wo), try matWords(rc[0], rc[1], m.b_bits)) > words) return error.Tfq3Truncated;
            if (try add(so, try matScales(rc[0], rc[1])) > halfs) return error.Tfq3Truncated;
        }
    }
    var with = out;
    with.has_5b = has_5b;
    return with;
}

/// Whether a named record serves the model's [all, k] tensor read whole from row lo, x rows: else a named refusal.
pub fn recordFits(r: u32, c: u32, lo: usize, x: usize, all: usize, k: usize) Error!void {
    if (lo != 0 or x != all) return error.Tfq3RecordSlice;
    if (r != all or c != k) return error.Tfq3RecordShape;
}

/// A projection's widths as the fusion sees them.
pub const Proj = struct { runs: usize, bits: u8, group: u16, n: usize, k: usize };

/// The one shared-expert fusion decision (moe.zig makes it, the dispatcher obeys): routed width, shapes, stages.
pub fn sharedFusion(gu: Proj, down: Proj, has_5b: bool, hidden: usize, inter: usize, stages: bool, combine: bool) bool {
    if (!stages or !combine) return false;
    if (gu.runs != 1 or down.runs != 1 or gu.group != 512 or down.group != 512) return false;
    const bits: u8 = if (has_5b) 5 else 4;
    if (gu.bits != bits or down.bits != bits) return false;
    if (hidden != expert_hidden or inter != expert_inter) return false;
    return gu.n == 2 * inter and gu.k == hidden and down.n == hidden and down.k == inter;
}

/// A run's addresses: buffer identity and byte offsets of its words, scales and table, and its rows.
pub const RunAt = struct { w_buf: usize, w_off: usize, s_buf: usize, s_off: usize, t_buf: usize, t_off: usize, row: usize, n: usize, bits: u8, group: u16 };

/// Whether [gate; up] can go through the fused 2n-row launch: up's words, scales and table right where it reads them.
pub fn gluContiguous(gate: RunAt, up: RunAt, n: usize, k: usize) bool {
    if (gate.group != 512 or up.group != 512 or gate.bits != up.bits) return false;
    if (gate.n != n or up.n != n or gate.row != 0 or up.row != n or n % 128 != 0 or k % 128 != 0) return false;
    if (gate.w_buf != up.w_buf or gate.s_buf != up.s_buf or gate.t_buf != up.t_buf or gate.t_off != up.t_off) return false;
    const w_bytes = (matWords(n, k, gate.bits) catch return false) * 4;
    const s_bytes = (matScales(n, k) catch return false) * 2;
    return up.w_off == gate.w_off + w_bytes and up.s_off == gate.s_off + s_bytes;
}

/// The q8 KDA kernels a TFQ3 checkpoint's 8-bit/64 gates may take; `.none` keeps main's selection exactly.
pub const KdaQ8 = enum { none, parts, window };

/// Only a layer whose stacked input holds TFQ3 runs, with 8-bit/64 gates and the q8 pipelines built, leaves main's.
pub fn kdaQ8(tfq3_layer: bool, q8_gates: bool, built: bool, split: bool, tp: u32, rows: u32) KdaQ8 {
    if (!tfq3_layer or !q8_gates or !built or tp != 1) return .none;
    if (split) return .parts;
    return if (rows > 1) .window else .none;
}

test {
    _ = @import("tfq3_check_test.zig");
}
