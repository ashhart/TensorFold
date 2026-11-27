//! Inkling 2 Mini TFQ3 (3/4/5-bit 2D-AffineCoset/half4 + 2D FWHT-128) zero-copy loader and Metal pipelines.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const format = @import("tfq3_format");
const state = @import("state.zig");

pub const Ref = @import("inkling2_metal_ref").Ref;

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
    pad: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Args) == 32);
}

pub const Error = error{
    InvalidTfq3Tables,
    InvalidTfq3Shard,
    MissingTfq3Tables,
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
    tables: mtl.Buffer,
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
    return try mul(try mul(r, c), bits) / 32;
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

/// Validates a `TFPW` v3 MoE layer file against `want`, alignment, bounds, expert descriptors, and XXH64.
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
    for (0..h.num_experts) |e| {
        const m_off = meta.off + e * @sizeOf(ExpertMeta);
        const m: ExpertMeta = std.mem.bytesToValue(ExpertMeta, bytes[m_off..][0..@sizeOf(ExpertMeta)]);
        if (m.b_bits < 3 or m.b_bits > 5) return error.Tfq3Bits;
        has_5b = has_5b or (m.b_bits == 5);
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

/// Zero-copy maps `dir/layer_{want.layer:0>2}.tffpw` (`bufferNoCopy` scales/weights; unaligned meta copied).
pub fn loadLayer(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    want: LayerWant,
    tables: mtl.Buffer,
) !Layer {
    var name: [32]u8 = undefined;
    const file = try std.fmt.bufPrint(&name, "layer_{d:0>2}.tffpw", .{want.layer});
    const m = (try mapFile(dir, file, @sizeOf(LayerHeader))) orelse return error.OpenFailed;
    errdefer unmapBytes(m.bytes);
    const at = try validateLayer(m.file(), want);
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
        .meta = b_meta,
        .scales = b_sc,
        .weights = b_wt,
        .has_5b = at.has_5b,
        .bytes = m.len + extra_meta,
    };
}

pub const Pipelines = struct {
    fwht_x: mtl.Pipeline,
    moe_gu: mtl.Pipeline,
    moe_down: mtl.Pipeline,
    xrot: mtl.Buffer,

    pub fn load(device: mtl.Device) !Pipelines {
        const lib = try mtl.Library.fromSource(device, sources.inkling2_tfq3, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const fwht_x = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq3_fwht_x", false);
        errdefer fwht_x.deinit();
        if (fwht_x.maxThreads() < 32) return error.Inkling2Threads;
        const moe_gu = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq3_moe_gu_softcap", false);
        errdefer moe_gu.deinit();
        if (moe_gu.maxThreads() < 256) return error.Inkling2Threads;
        const moe_down = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq3_moe_down", false);
        errdefer moe_down.deinit();
        if (moe_down.maxThreads() < 256) return error.Inkling2Threads;
        const xrot = try device.buffer(xrot_bytes, mtl.ResourceOptions.private);
        return .{
            .fwht_x = fwht_x,
            .moe_gu = moe_gu,
            .moe_down = moe_down,
            .xrot = xrot,
        };
    }

    pub fn deinit(p: *Pipelines) void {
        p.xrot.deinit();
        p.moe_down.deinit();
        p.moe_gu.deinit();
        p.fwht_x.deinit();
        p.* = undefined;
    }

    /// Dispatches forward FWHT-128, softcapped SwiGLU gate+up, and down into BF16 `down_out[rows * 8, 2304]`.
    pub fn dispatch(
        p: *const Pipelines,
        e: mtl.ComputeEncoder,
        layer: Layer,
        mlp_normed: Ref,
        route_ids: Ref,
        act_rot_gu: Ref,
        down_out: Ref,
        rows: usize,
    ) !void {
        if (rows == 0 or rows > state.max_rows) return error.Inkling2RowsExceeded;
        if (layer.hidden != state.hidden or layer.inter != state.intermediate) return error.Tfq3Geometry;
        const a: Args = .{
            .rows = @intCast(rows),
            .hidden = layer.hidden,
            .inter = layer.inter,
            .experts = layer.num_experts,
            .topk = state.experts_per_token,
            .gate_cap = 4.0,
            .up_cap = 25.0,
        };
        const slots: usize = rows * state.experts_per_token;

        e.setPipeline(p.fwht_x);
        e.setBuffer(mlp_normed.buf, mlp_normed.off, 0);
        e.setBuffer(p.xrot, 0, 1);
        e.setValue(a, 2);
        e.dispatchGroups(mtl.Size.of(layer.hidden / 128, rows, 1), mtl.Size.of(32, 1, 1));

        e.setPipeline(p.moe_gu);
        e.setBuffer(layer.weights, 0, 0);
        e.setBuffer(layer.scales, 0, 1);
        e.setBuffer(layer.meta, 0, 2);
        e.setBuffer(layer.tables, 0, 3);
        e.setBuffer(p.xrot, 0, 4);
        e.setBuffer(route_ids.buf, route_ids.off, 5);
        e.setBuffer(act_rot_gu.buf, act_rot_gu.off, 6);
        e.setValue(a, 7);
        e.dispatchGroups(mtl.Size.of(layer.inter / 128, slots, 1), mtl.Size.of(256, 1, 1));

        e.setPipeline(p.moe_down);
        e.setBuffer(layer.weights, 0, 0);
        e.setBuffer(layer.scales, 0, 1);
        e.setBuffer(layer.meta, 0, 2);
        e.setBuffer(layer.tables, 0, 3);
        e.setBuffer(act_rot_gu.buf, act_rot_gu.off, 4);
        e.setBuffer(route_ids.buf, route_ids.off, 5);
        e.setBuffer(down_out.buf, down_out.off, 6);
        e.setValue(a, 7);
        e.dispatchGroups(mtl.Size.of(layer.hidden / 128, slots, 1), mtl.Size.of(256, 1, 1));
    }
};

fn buildTestTfpw(
    gpa: std.mem.Allocator,
    layer_idx: u32,
    hidden: u32,
    inter: u32,
    bits: []const u32,
    with_checksum: bool,
) ![]u8 {
    const sc_gu = try matScales(inter, hidden);
    const sc_dn = try matScales(hidden, inter);
    var total_words: u64 = 0;
    var total_scales: u64 = 0;
    for (bits) |b| {
        const gu_w = try matWords(inter, hidden, b);
        const dn_w = try matWords(hidden, inter, b);
        total_words += gu_w * 2 + dn_w;
        total_scales += sc_gu * 2 + sc_dn;
    }
    const meta_off: u64 = @sizeOf(LayerHeader);
    const meta_bytes: u64 = @as(u64, bits.len) * @sizeOf(ExpertMeta);
    const sc_off: u64 = std.mem.alignForward(u64, meta_off + meta_bytes, section_align);
    const sc_bytes: u64 = total_scales * 2;
    const w_off: u64 = std.mem.alignForward(u64, sc_off + sc_bytes, section_align);
    const w_bytes: u64 = total_words * 4;
    const file_len: usize = @intCast(std.mem.alignForward(u64, w_off + w_bytes, section_align));

    const buf = try gpa.alloc(u8, file_len);
    @memset(buf, 0);
    var w_cur: u64 = 0;
    var s_cur: u32 = 0;
    for (bits, 0..) |b, e| {
        const gu_w = try matWords(inter, hidden, b);
        const dn_w = try matWords(hidden, inter, b);
        var em = std.mem.zeroes(ExpertMeta);
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
        @memcpy(buf[@intCast(meta_off + e * @sizeOf(ExpertMeta))..][0..@sizeOf(ExpertMeta)], std.mem.asBytes(&em));
    }
    var hdr = LayerHeader{
        .magic = magic_layer,
        .version = format_version,
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
    @memcpy(buf[0..@sizeOf(LayerHeader)], std.mem.asBytes(&hdr));
    return buf;
}

test "Inkling-2 Metal TFQ3 validates PWV3 tables and rejects truncated or non-finite data" {
    var tbl: [tables_file_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], magic_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], format_version, .little);
    try validateTables(&tbl);
    try std.testing.expectError(error.InvalidTfq3Tables, validateTables(tbl[0..1024]));
    std.mem.writeInt(u16, tbl[64..66], 0x7c00, .little); // +inf half
    try std.testing.expectError(error.InvalidTfq3Tables, validateTables(&tbl));
    std.mem.writeInt(u16, tbl[64..66], 0, .little);
    std.mem.writeInt(u32, tbl[8..12], 0x7fc00000, .little); // NaN float
    try std.testing.expectError(error.InvalidTfq3Tables, validateTables(&tbl));
}

test "Inkling-2 Metal TFQ3 validates TFPW v3 header, XXH64 checksums, mixed bits, and 4-bit alignment" {
    const gpa = std.testing.allocator;
    // Mixed 3b/4b/5b with Inkling-2 Mini dimensions (H=2304, I=1152) and XXH64 enabled.
    const checked = try buildTestTfpw(gpa, 7, 2304, 1152, &.{ 3, 4, 5 }, true);
    defer gpa.free(checked);
    const want: LayerWant = .{ .layer = 7, .experts = 3, .hidden = 2304, .inter = 1152 };
    const parsed = try validateLayer(checked, want);
    try std.testing.expect(parsed.has_5b);
    try std.testing.expectEqual(@as(usize, @sizeOf(LayerHeader)), parsed.meta.off);
    try std.testing.expectEqual(@as(usize, 0), parsed.scales.off % section_align);
    try std.testing.expectEqual(@as(usize, 0), parsed.weights.off % section_align);

    // 1-bit flip in meta, scales, and weights each fail with Tfq3ChecksumMismatch.
    for ([_]usize{ parsed.meta.off, parsed.scales.off, parsed.weights.off }) |flip_at| {
        checked[flip_at] ^= 0x01;
        try std.testing.expectError(error.Tfq3ChecksumMismatch, validateLayer(checked, want));
        checked[flip_at] ^= 0x01;
    }

    // Wrong layer index or shape fails with Tfq3LayerMismatch.
    try std.testing.expectError(error.Tfq3LayerMismatch, validateLayer(checked, .{ .layer = 8, .experts = 3, .hidden = 2304, .inter = 1152 }));
    try std.testing.expectError(error.Tfq3LayerMismatch, validateLayer(checked, .{ .layer = 7, .experts = 128, .hidden = 2304, .inter = 1152 }));

    // Legacy reserved0 == 0 accepted, has_5b == false when only 3b/4b present, misaligned 4b word_off refused.
    const legacy = try buildTestTfpw(gpa, 0, 128, 128, &.{ 3, 4 }, false);
    defer gpa.free(legacy);
    const lwant: LayerWant = .{ .layer = 0, .experts = 2, .hidden = 128, .inter = 128 };
    const lparsed = try validateLayer(legacy, lwant);
    try std.testing.expect(!lparsed.has_5b);

    var em = std.mem.bytesToValue(ExpertMeta, legacy[lparsed.meta.off + @sizeOf(ExpertMeta) ..][0..@sizeOf(ExpertMeta)]);
    em.gate_word_off += 1; // breaks 4-bit uint4 alignment
    @memcpy(legacy[lparsed.meta.off + @sizeOf(ExpertMeta) ..][0..@sizeOf(ExpertMeta)], std.mem.asBytes(&em));
    try std.testing.expectError(error.Tfq3Misaligned, validateLayer(legacy, lwant));

    em.gate_word_off -= 1;
    em.b_bits = 6;
    @memcpy(legacy[lparsed.meta.off + @sizeOf(ExpertMeta) ..][0..@sizeOf(ExpertMeta)], std.mem.asBytes(&em));
    try std.testing.expectError(error.Tfq3Bits, validateLayer(legacy, lwant));
}
