//! Inkling 2 Mini TFQ3/TFQ4 (2/225/3/4/5-bit 2D FWHT-128) zero-copy loader and Metal pipelines.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const mtl = @import("metal");
const format = @import("tfq3_format");
const state = @import("state.zig");

pub const Pipelines = @import("tfq3_pipelines.zig").Pipelines;

pub const Ref = @import("inkling2_metal_ref").Ref;
pub const tfq4 = format.tfq4;

pub const magic_layer = format.magic_layer;
pub const magic_tables = format.magic_tables;
pub const format_version = format.format_version;
pub const section_align = format.section_align;
pub const tables_bytes = format.tables_bytes;
pub const tables_file_bytes = format.tables_file_bytes;

pub const LayerHeader = format.LayerHeader;
pub const ExpertMeta = format.ExpertMeta;

/// Scratch bytes for FP16 forward-rotated activations `[max_rows, hidden]` (`256 * 2304 * 2 = 1.13 MiB`).
pub const xrot_bytes: usize = state.max_rows * state.hidden * 2;

pub const Args = extern struct {
    rows: u32,
    hidden: u32,
    inter: u32,
    experts: u32,
    topk: u32,
    gate_cap: f32,
    up_cap: f32,
    table4_kind: u32 = 0,
};

pub const Dense4Args = extern struct {
    rows: u32,
    n: u32,
    k: u32,
    ldo: u32,
    bits: u32,
    table_kind: u32,
    row_off: u32 = 0,
    sr_split: u32 = 0,
    out_f32: u32 = 0,
    pad: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Args) == 32);
    std.debug.assert(@sizeOf(Dense4Args) == 40);
}

pub const Error = error{
    InvalidTfq3Tables,
    InvalidTfq4Tables,
    InvalidTfq3Shard,
    MissingTfq3Tables,
    MissingTfq4Tables,
    Tfq3Truncated,
    Tfq3Overflow,
    Tfq3Misaligned,
    Tfq3Overlap,
    Tfq3Geometry,
    Tfq3Bits,
    Tfq3LayerMismatch,
    Tfq3ChecksumMismatch,
};

pub const Span = struct { off: usize, len: usize };

pub const LayerWant = struct {
    layer: u32,
    experts: u32,
    hidden: u32,
    inter: u32,
};

pub const LayerLayout = struct {
    header: LayerHeader,
    meta: Span,
    scales: Span,
    weights: Span,
    has_5b: bool,
    has_tfq3: bool,
    has_tfq4: bool,
    needs_e8p1024: bool,
};

pub const LoadedTables4 = struct {
    buf: mtl.Buffer,
    table_kind: u32,
    k_codes: u32,
};

pub const MappedShard = struct {
    bytes: []align(std.heap.page_size_min) u8,

    pub fn deinit(self: MappedShard) void {
        unmapBytes(self.bytes);
    }
};

fn unmapBytes(bytes: []align(std.heap.page_size_min) u8) void {
    if (comptime builtin.os.tag == .windows) return;
    std.posix.munmap(bytes);
}

pub const Layer = struct {
    layer_idx: u32,
    num_experts: u32,
    hidden: u32,
    inter: u32,
    tables: ?mtl.Buffer = null,
    tables4: ?mtl.Buffer = null,
    table4_kind: u32 = 0,
    has_tfq4: bool = false,
    meta: mtl.Buffer,
    scales: mtl.Buffer,
    weights: mtl.Buffer,
    has_5b: bool,
    bytes: usize,
};

fn cast(v: u64) Error!usize {
    return std.math.cast(usize, v) orelse error.Tfq3Overflow;
}

fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.Tfq3Overflow;
}

fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.Tfq3Overflow;
}

pub fn mappedLen(file_len: usize) Error!usize {
    const rounded = try add(file_len, section_align - 1);
    return rounded / section_align * section_align;
}

pub fn matWords(r: usize, c: usize, bits: usize) Error!usize {
    if (bits == 2) return try mul(r, c / 16);
    if (bits == 225) return try mul(try mul(r, c / 128), 9);
    if (bits >= 3 and bits <= 5) return try mul(try mul(r, c), bits) / 32;
    return error.Tfq3Bits;
}

pub fn matScales(r: usize, c: usize) Error!usize {
    return try mul(r / 128, c / 128);
}

pub fn span(file_len: usize, off: u64, len: usize, gpu: bool) Error!Span {
    const o = try cast(off);
    const end = try add(o, len);
    if (end > file_len) return error.Tfq3Truncated;
    if (gpu) {
        if (o % section_align != 0) return error.Tfq3Misaligned;
        const view_end = try add(o, try mappedLen(@max(len, 16)));
        if (view_end > try mappedLen(file_len)) return error.Tfq3Truncated;
    }
    return .{ .off = o, .len = len };
}

fn finiteHalf(bits: u16) bool {
    return (bits & 0x7c00) != 0x7c00;
}

/// Validates a `pw_v3_tables.bin` payload (magic, version, length, finite affine floats and half tables).
pub fn validateTables(bytes: []const u8) Error!void {
    if (bytes.len < tables_bytes) return error.InvalidTfq3Tables;
    const magic = std.mem.readInt(u32, bytes[0..4], .little);
    const version = std.mem.readInt(u32, bytes[4..8], .little);
    if (magic != magic_tables or version != format_version) return error.InvalidTfq3Tables;
    for (0..6) |i| {
        const v: f32 = @bitCast(std.mem.readInt(u32, bytes[8 + i * 4 ..][0..4], .little));
        if (!std.math.isFinite(v)) return error.InvalidTfq3Tables;
    }
    var at: usize = 64;
    while (at < tables_bytes) : (at += 2) {
        if (!finiteHalf(std.mem.readInt(u16, bytes[at..][0..2], .little))) return error.InvalidTfq3Tables;
    }
}

/// Validates a `pw_v4_e8p_tables.bin` (`PWV4` v4, table_kind 0..3) payload and returns its header info.
pub fn validateTables4(bytes: []const u8) Error!tfq4.PwV4TablesInfo {
    return tfq4.validatePwV4Tables(bytes) catch error.InvalidTfq4Tables;
}

/// Validates a `TFPW` v3 MoE layer file (bits in {2, 225, 3, 4, 5}) against `want`, alignment, bounds, and XXH64.
pub fn validateLayer(bytes: []const u8, want: LayerWant) Error!LayerLayout {
    if (bytes.len < @sizeOf(LayerHeader)) return error.Tfq3Truncated;
    const h: LayerHeader = std.mem.bytesToValue(LayerHeader, bytes[0..@sizeOf(LayerHeader)]);
    if (h.magic != magic_layer or h.version != format_version) return error.InvalidTfq3Shard;
    if (h.layer_idx != want.layer or h.num_experts != want.experts or h.hidden_size != want.hidden or h.moe_inter != want.inter) {
        return error.Tfq3LayerMismatch;
    }
    if (h.num_experts == 0 or h.hidden_size == 0 or h.moe_inter == 0 or
        (h.hidden_size % 128) != 0 or (h.moe_inter % 128) != 0 or
        h.num_tensors != try mul(h.num_experts, 3))
    {
        return error.Tfq3Geometry;
    }
    const words = try cast(h.stream_words);
    const halfs = try cast(h.scale_halfs);
    const meta = try span(bytes.len, h.off_meta, try mul(h.num_experts, @sizeOf(ExpertMeta)), false);
    if (meta.off % @alignOf(ExpertMeta) != 0) return error.Tfq3Misaligned;
    const scales = try span(bytes.len, h.off_scales, try mul(halfs, 2), true);
    const weights = try span(bytes.len, h.off_weights, try mul(words, 4), true);

    const meta_end = try add(meta.off, meta.len);
    const scales_end = try add(scales.off, scales.len);
    if (meta.off < @sizeOf(LayerHeader) or meta_end > scales.off or scales_end > weights.off) {
        return error.Tfq3Overlap;
    }

    if ((h.reserved0 & 0x1) != 0) {
        if (std.hash.XxHash64.hash(0, bytes[meta.off..][0..meta.len]) != h.reserved1[0]) return error.Tfq3ChecksumMismatch;
        if (std.hash.XxHash64.hash(0, bytes[scales.off..][0..scales.len]) != h.reserved1[1]) return error.Tfq3ChecksumMismatch;
        if (std.hash.XxHash64.hash(0, bytes[weights.off..][0..weights.len]) != h.reserved1[2]) return error.Tfq3ChecksumMismatch;
    }

    const sc_gu = try matScales(h.moe_inter, h.hidden_size);
    const sc_dn = try matScales(h.hidden_size, h.moe_inter);
    var has_5b = false;
    var has_tfq3 = false;
    var has_tfq4 = false;
    var needs_e8p1024 = false;
    for (0..h.num_experts) |e| {
        const m_off = meta.off + e * @sizeOf(ExpertMeta);
        const m: ExpertMeta = std.mem.bytesToValue(ExpertMeta, bytes[m_off..][0..@sizeOf(ExpertMeta)]);
        if (!format.validTfpwBits(m.b_bits)) return error.Tfq3Bits;
        has_5b = has_5b or (m.b_bits == 5);
        has_tfq3 = has_tfq3 or (m.b_bits >= 3 and m.b_bits <= 5);
        has_tfq4 = has_tfq4 or (m.b_bits == 2 or m.b_bits == 225);
        needs_e8p1024 = needs_e8p1024 or (m.b_bits == 225);
        const gu_w = try matWords(h.moe_inter, h.hidden_size, m.b_bits);
        const dn_w = try matWords(h.hidden_size, h.moe_inter, m.b_bits);
        const word_offs = [3]u64{ m.gate_word_off, m.up_word_off, m.down_word_off };
        const word_lens = [3]usize{ gu_w, gu_w, dn_w };
        const sc_offs = [3]u32{ m.gate_sc_off, m.up_sc_off, m.down_sc_off };
        const sc_lens = [3]usize{ sc_gu, sc_gu, sc_dn };
        for (word_offs, word_lens, sc_offs, sc_lens) |wo, wl, so, sl| {
            if (m.b_bits == 4 and wo % 4 != 0) return error.Tfq3Misaligned;
            if (try add(try cast(wo), wl) > words) return error.Tfq3Truncated;
            if (try add(so, sl) > halfs) return error.Tfq3Truncated;
        }
    }
    return .{
        .header = h,
        .meta = meta,
        .scales = scales,
        .weights = weights,
        .has_5b = has_5b,
        .has_tfq3 = has_tfq3,
        .has_tfq4 = has_tfq4,
        .needs_e8p1024 = needs_e8p1024,
    };
}

fn readPrefix(path: [:0]const u8, out: []u8) bool {
    if (comptime builtin.os.tag == .windows) return false;
    const fd = platform.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return false;
    defer _ = platform.c.close(fd);
    var done: usize = 0;
    while (done < out.len) {
        const n = platform.c.pread(fd, out.ptr + done, out.len - done, @intCast(done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

/// Returns true if `dir/pw_v3_tables.bin` exists and starts with a `PWV3` v3 header.
pub fn hasTables(dir: []const u8) bool {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/pw_v3_tables.bin", .{dir}) catch return false;
    buf[s.len] = 0;
    var hdr: [8]u8 = undefined;
    if (!readPrefix(buf[0..s.len :0], &hdr)) return false;
    return std.mem.readInt(u32, hdr[0..4], .little) == magic_tables and
        std.mem.readInt(u32, hdr[4..8], .little) == format_version;
}

/// Returns true if `dir/pw_v4_e8p_tables.bin` exists and starts with a valid `PWV4` v4 header.
pub fn hasTables4(dir: []const u8) bool {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/pw_v4_e8p_tables.bin", .{dir}) catch return false;
    buf[s.len] = 0;
    var hdr: [20]u8 = undefined;
    if (!readPrefix(buf[0..s.len :0], &hdr)) return false;
    return std.mem.readInt(u32, hdr[0..4], .little) == tfq4.magic_tables and
        std.mem.readInt(u32, hdr[4..8], .little) == tfq4.format_version and
        std.mem.readInt(u32, hdr[12..16], .little) == tfq4.dim and
        std.mem.readInt(u32, hdr[16..20], .little) <= tfq4.table_kind_max;
}

/// Returns true if `dir/layer_{layer:0>2}.tffpw` exists with a `TFPW` v3 header and `num_experts != 0`.
pub fn hasLayer(dir: []const u8, layer: usize) bool {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/layer_{d:0>2}.tffpw", .{ dir, layer }) catch return false;
    buf[s.len] = 0;
    var hdr: [16]u8 = undefined;
    if (!readPrefix(buf[0..s.len :0], &hdr)) return false;
    return std.mem.readInt(u32, hdr[0..4], .little) == magic_layer and
        std.mem.readInt(u32, hdr[4..8], .little) == format_version and
        std.mem.readInt(u32, hdr[12..16], .little) != 0;
}

const Mapped = struct {
    bytes: []align(std.heap.page_size_min) u8,
    len: usize,

    fn file(m: Mapped) []const u8 {
        return m.bytes[0..m.len];
    }
};

fn mapFile(dir: []const u8, name: []const u8, least: usize) !?Mapped {
    if (comptime builtin.os.tag == .windows) return error.UnsupportedPlatform;
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/{s}", .{ dir, name }) catch return error.NameTooLong;
    buf[s.len] = 0;
    const fd = platform.c.open(buf[0..s.len :0].ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 0 or end < least) return error.Tfq3Truncated;
    const len = std.math.cast(usize, end) orelse return error.Tfq3Overflow;
    const room = try mappedLen(len);
    const bytes = try std.posix.mmap(
        null,
        std.mem.alignForward(usize, room, std.heap.pageSize()),
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE },
        fd,
        0,
    );
    return .{ .bytes = bytes, .len = len };
}

fn view(device: mtl.Device, m: Mapped, at: Span) !mtl.Buffer {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    return device.bufferNoCopy(@ptrCast(m.bytes.ptr + at.off), try mappedLen(@max(at.len, 16)), opts);
}

fn keep(buffers: *std.ArrayList(mtl.Buffer), maps: *std.ArrayList(MappedShard), m: Mapped, views: []const mtl.Buffer) void {
    for (views) |b| buffers.appendAssumeCapacity(b);
    maps.appendAssumeCapacity(.{ .bytes = m.bytes });
}

/// Zero-copy maps and validates `dir/pw_v3_tables.bin`.
pub fn loadTables(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
) !mtl.Buffer {
    const m = (try mapFile(dir, "pw_v3_tables.bin", tables_bytes)) orelse return error.MissingTfq3Tables;
    errdefer unmapBytes(m.bytes);
    try validateTables(m.file());
    const all = try span(m.len, 0, m.len, true);
    try buffers.ensureUnusedCapacity(gpa, 1);
    try maps.ensureUnusedCapacity(gpa, 1);
    const b = try view(device, m, all);
    keep(buffers, maps, m, &.{b});
    total_bytes.* += m.len;
    return b;
}

/// Zero-copy maps and validates `dir/pw_v4_e8p_tables.bin` if present, or returns null when absent.
pub fn loadTables4Optional(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
) !?LoadedTables4 {
    if (!hasTables4(dir)) return null;
    const m = (try mapFile(dir, "pw_v4_e8p_tables.bin", tfq4.tables_256_bytes)) orelse return null;
    errdefer unmapBytes(m.bytes);
    const h = try validateTables4(m.file());
    const all = try span(m.len, 0, m.len, true);
    try buffers.ensureUnusedCapacity(gpa, 1);
    try maps.ensureUnusedCapacity(gpa, 1);
    const b = try view(device, m, all);
    keep(buffers, maps, m, &.{b});
    total_bytes.* += m.len;
    return .{ .buf = b, .table_kind = h.table_kind, .k_codes = h.k_codes };
}

/// Zero-copy maps `dir/layer_{want.layer:0>2}.tffpw` (`bufferNoCopy` scales/weights; unaligned meta copied).
pub fn loadLayer(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    want: LayerWant,
    tables: ?mtl.Buffer,
    tables4: ?LoadedTables4,
) !Layer {
    var name: [32]u8 = undefined;
    const file = try std.fmt.bufPrint(&name, "layer_{d:0>2}.tffpw", .{want.layer});
    const m = (try mapFile(dir, file, @sizeOf(LayerHeader))) orelse return error.OpenFailed;
    errdefer unmapBytes(m.bytes);
    const at = try validateLayer(m.file(), want);
    if (at.has_tfq3 and tables == null) return error.MissingTfq3Tables;
    if (at.has_tfq4) {
        const t4 = tables4 orelse return error.MissingTfq4Tables;
        if (at.needs_e8p1024 and !tfq4.tableKindSupportsBits(t4.table_kind, 225)) return error.InvalidTfq4Tables;
    }
    try buffers.ensureUnusedCapacity(gpa, 3);
    try maps.ensureUnusedCapacity(gpa, 1);
    const meta_copied = at.meta.off % section_align != 0;
    const b_meta = if (!meta_copied) try view(device, m, at.meta) else blk: {
        const buf = try device.buffer(@max(at.meta.len, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memcpy(buf.contents()[0..at.meta.len], m.file()[at.meta.off..][0..at.meta.len]);
        break :blk buf;
    };
    errdefer b_meta.deinit();
    const b_sc = try view(device, m, at.scales);
    errdefer b_sc.deinit();
    const b_wt = try view(device, m, at.weights);
    keep(buffers, maps, m, &.{ b_meta, b_sc, b_wt });
    const extra_meta: usize = if (meta_copied) @max(at.meta.len, 16) else 0;
    total_bytes.* += m.len + extra_meta;
    return .{
        .layer_idx = want.layer,
        .num_experts = at.header.num_experts,
        .hidden = at.header.hidden_size,
        .inter = at.header.moe_inter,
        .tables = tables,
        .tables4 = if (tables4) |t4| t4.buf else null,
        .table4_kind = if (tables4) |t4| t4.table_kind else 0,
        .has_tfq4 = at.has_tfq4,
        .meta = b_meta,
        .scales = b_sc,
        .weights = b_wt,
        .has_5b = at.has_5b,
        .bytes = m.len + extra_meta,
    };
}
