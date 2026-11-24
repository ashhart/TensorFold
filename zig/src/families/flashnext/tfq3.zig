//! Flash Next TFQ3 (2D-AffineCoset/half4) binary format, zero-copy loader, and Metal pipelines.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

pub const files = @import("flashnext_memory").tfq3_files;
pub const dispatch = @import("tfq3_dispatch.zig");

pub const magic_layer: u32 = 0x57504654;
pub const magic_v3_tables: u32 = 0x33565750;
pub const format_version: u32 = 3;
pub const section_align: usize = 16384;
pub const hidden_size: u32 = 2560;
pub const moe_inter: u32 = 640;
pub const num_experts: u32 = 512;
pub const topk: u32 = 10;
pub const max_decode_rows: u32 = 16;
pub const max_decode_slots: u32 = max_decode_rows * topk;
pub const max_prefill_tiles: u32 = 4096;

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

pub const Dims = extern struct {
    rows: u32,
    slots: u32,
    topk: u32,
    experts: u32,
    sh_gu_bits: u32 = 6,
    sh_gu_group: u32 = 32,
    sh_dn_bits: u32 = 6,
    sh_dn_group: u32 = 32,
    flags: u32 = 0,
    pad: [3]u32 = .{ 0, 0, 0 },
};

comptime {
    std.debug.assert(@sizeOf(LayerHeader) == 144);
    std.debug.assert(@sizeOf(ExpertMeta) == 64);
    std.debug.assert(@sizeOf(Dims) == 48);
}

pub const MappedShard = struct {
    bytes: []align(std.heap.page_size_min) u8,

    pub fn deinit(self: MappedShard) void {
        std.posix.munmap(self.bytes);
    }
};

pub const Layer = struct {
    layer_idx: u32,
    bytes: usize,
    tables: mtl.Buffer,
    meta: mtl.Buffer,
    scales: mtl.Buffer,
    weights: mtl.Buffer,
    has_5b: bool = true,
};

fn align16k(n: usize) usize {
    return std.mem.alignForward(usize, @max(n, 16), section_align);
}

fn copySection(
    device: mtl.Device,
    held: *std.ArrayList(mtl.Buffer),
    arena: std.mem.Allocator,
    src: []const u8,
) !mtl.Buffer {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const b = try device.buffer(@max(src.len, 16), opts);
    errdefer b.deinit();
    @memcpy(b.contents()[0..src.len], src);
    try held.append(arena, b);
    return b;
}

fn wrapSection(
    device: mtl.Device,
    held: *std.ArrayList(mtl.Buffer),
    arena: std.mem.Allocator,
    mapped: []u8,
    off: u64,
    byte_len: usize,
) !mtl.Buffer {
    const start: usize = @intCast(off);
    const page = std.heap.pageSize();
    const aligned = align16k(byte_len);
    if (start % page != 0 or aligned % page != 0 or start + aligned > mapped.len) {
        return copySection(device, held, arena, mapped[start .. start + byte_len]);
    }
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const b = try device.bufferNoCopy(@ptrCast(mapped.ptr + start), aligned, opts);
    errdefer b.deinit();
    try held.append(arena, b);
    return b;
}

pub fn loadV3Tables(
    arena: std.mem.Allocator,
    device: mtl.Device,
    held: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
) !?mtl.Buffer {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/pw_v3_tables.bin", .{dir}) catch return null;
    buf[s.len] = 0;
    const fd = platform.c.open(buf[0..s.len :0].ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 16384) return error.EmptyFile;
    const fsize: usize = @intCast(end);
    total_bytes.* += fsize;
    const page = std.heap.pageSize();
    const mapped = try std.posix.mmap(null, std.mem.alignForward(usize, fsize, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    errdefer std.posix.munmap(mapped);
    const hdr: [*]const u32 = @ptrCast(@alignCast(mapped.ptr));
    if (hdr[0] != magic_v3_tables or hdr[1] != format_version) return error.UnsupportedQuantization;
    try maps.append(arena, .{ .bytes = mapped });
    return try wrapSection(device, held, arena, mapped, 0, fsize);
}

fn loadShard(
    arena: std.mem.Allocator,
    device: mtl.Device,
    held: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    path: [:0]const u8,
    layer_idx: u32,
    v3_tables: ?mtl.Buffer,
) !Layer {
    const fd = platform.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.OpenFailed;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < @sizeOf(LayerHeader)) return error.EmptyFile;
    const fsize: usize = @intCast(end);
    total_bytes.* += fsize;
    const page = std.heap.pageSize();
    const mapped = try std.posix.mmap(null, std.mem.alignForward(usize, fsize, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    errdefer std.posix.munmap(mapped);
    const ph: *const LayerHeader = @ptrCast(@alignCast(mapped.ptr));
    if (ph.magic != magic_layer or ph.version != format_version) return error.UnsupportedQuantization;
    if (ph.num_experts != num_experts or ph.hidden_size != hidden_size or ph.moe_inter != moe_inter) {
        return error.UnsupportedQuantization;
    }
    const meta_bytes = @as(usize, ph.num_experts) * @sizeOf(ExpertMeta);
    const sc_bytes = @as(usize, @intCast(ph.scale_halfs)) * 2;
    const wt_bytes = @as(usize, @intCast(ph.stream_words)) * 4;
    if (ph.off_meta + meta_bytes > fsize or ph.off_scales + sc_bytes > fsize or ph.off_weights + wt_bytes > fsize) {
        return error.UnsupportedQuantization;
    }
    const metas_ptr: [*]const ExpertMeta = @ptrCast(@alignCast(mapped.ptr + @as(usize, @intCast(ph.off_meta))));
    var has_5b = false;
    for (0..ph.num_experts) |ei| {
        if (metas_ptr[ei].b_bits == 5) {
            has_5b = true;
            break;
        }
    }
    const tb = v3_tables orelse return error.UnsupportedQuantization;
    try maps.append(arena, .{ .bytes = mapped });
    const b_meta = try wrapSection(device, held, arena, mapped, ph.off_meta, meta_bytes);
    const b_sc = try wrapSection(device, held, arena, mapped, ph.off_scales, sc_bytes);
    const b_wt = try wrapSection(device, held, arena, mapped, ph.off_weights, wt_bytes);
    return .{ .layer_idx = layer_idx, .bytes = fsize, .tables = tb, .meta = b_meta, .scales = b_sc, .weights = b_wt, .has_5b = has_5b };
}

pub fn loadLayer(
    arena: std.mem.Allocator,
    device: mtl.Device,
    held: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    layer: usize,
    v3_tables: ?mtl.Buffer,
) !Layer {
    var buf: [512]u8 = undefined;
    const s = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/layer_{d:0>2}.tffpw", .{ dir, layer });
    buf[s.len] = 0;
    return loadShard(arena, device, held, maps, total_bytes, buf[0..s.len :0], @intCast(layer), v3_tables);
}

pub fn loadMtp(
    arena: std.mem.Allocator,
    device: mtl.Device,
    held: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    v3_tables: ?mtl.Buffer,
) !Layer {
    var buf: [512]u8 = undefined;
    const s = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/mtp_00.tffpw", .{dir});
    buf[s.len] = 0;
    return loadShard(arena, device, held, maps, total_bytes, buf[0..s.len :0], 48, v3_tables);
}

pub const Pipe = enum(usize) {
    route_dedup,
    fwht_x_fwd,
    fwht_glu_fwht,
    fwht_y_inv,
    dec_gu_b1_543,
    dec_down_b1_543,
    dec_gu_dedup_543,
    dec_down_dedup_543,
    dec_gu_b1_43,
    dec_down_b1_43,
    dec_gu_dedup_43,
    dec_down_dedup_43,
    prefill_fwht_x,
    tile_scan,
    prefill_gu_543,
    prefill_down_543,
    prefill_gu_43,
    prefill_down_43,
};

const pipe_names = blk: {
    const names = std.meta.fieldNames(Pipe);
    var out: [names.len][]const u8 = undefined;
    for (names, 0..) |name, i| out[i] = "fn_pw_" ++ name;
    break :blk out;
};

pub fn buildSource(gpa: std.mem.Allocator, units: bool) ![]u8 {
    const raw = try std.mem.concat(gpa, u8, &.{
        ks.flashnext_tfq3.common,
        ks.flashnext_tfq3.moe_mv,
        ks.flashnext_tfq3.moe_decode,
        ks.flashnext_tfq3.prefill,
    });
    defer gpa.free(raw);
    return frags.precision.source(gpa, units, raw, ks.nax, false);
}

pub const Pipes = struct {
    list: [pipe_names.len]mtl.Pipeline,
    xrot_buf: mtl.Buffer,
    gu_buf: mtl.Buffer,
    act_rot_buf: mtl.Buffer,
    yr_buf: mtl.Buffer,
    uids_buf: mtl.Buffer,
    umem_buf: mtl.Buffer,
    ucount_buf: mtl.Buffer,
    owns_bufs: bool,
    hash: u64,

    pub inline fn get(p: *const Pipes, k: Pipe) mtl.Pipeline {
        return p.list[@backingInt(k)];
    }

    fn allocPriv(device: mtl.Device, size: usize, arena: ?std.mem.Allocator, held: ?*std.ArrayList(mtl.Buffer)) !mtl.Buffer {
        const b = try device.buffer(size, mtl.ResourceOptions.private);
        errdefer b.deinit();
        if (held) |h| try h.append(arena.?, b);
        return b;
    }

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, held: ?*std.ArrayList(mtl.Buffer)) !Pipes {
        const src = try buildSource(gpa, device.tensorUnits());
        defer gpa.free(src);
        var h = std.hash.Wyhash.init(0);
        h.update(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var list: [pipe_names.len]mtl.Pipeline = undefined;
        var built: usize = 0;
        errdefer for (list[0..built]) |pl| pl.deinit();
        for (pipe_names, 0..) |name, i| {
            list[i] = try mtl.Pipeline.init(device, lib, name, false);
            built += 1;
        }
        const a: ?std.mem.Allocator = if (held != null) gpa else null;
        const b_xrot = try allocPriv(device, @as(usize, max_decode_rows) * hidden_size * 2, a, held);
        errdefer if (held == null) b_xrot.deinit();
        const b_gu = try allocPriv(device, @as(usize, max_decode_slots) * 2 * moe_inter * 4, a, held);
        errdefer if (held == null) b_gu.deinit();
        const b_act_rot = try allocPriv(device, @as(usize, max_decode_slots) * moe_inter * 2, a, held);
        errdefer if (held == null) b_act_rot.deinit();
        const b_yr = try allocPriv(device, @as(usize, max_decode_slots) * hidden_size * 4, a, held);
        errdefer if (held == null) b_yr.deinit();
        const b_uids = try allocPriv(device, @as(usize, max_decode_slots) * 4, a, held);
        errdefer if (held == null) b_uids.deinit();
        const b_umem = try allocPriv(device, @as(usize, max_decode_slots) * 16 * 4, a, held);
        errdefer if (held == null) b_umem.deinit();
        const b_ucount = try allocPriv(device, 16, a, held);
        errdefer if (held == null) b_ucount.deinit();
        return .{
            .list = list,
            .xrot_buf = b_xrot,
            .gu_buf = b_gu,
            .act_rot_buf = b_act_rot,
            .yr_buf = b_yr,
            .uids_buf = b_uids,
            .umem_buf = b_umem,
            .ucount_buf = b_ucount,
            .owns_bufs = held == null,
            .hash = h.final(),
        };
    }

    pub fn deinit(p: *Pipes) void {
        for (p.list) |pl| pl.deinit();
        if (p.owns_bufs) {
            p.xrot_buf.deinit();
            p.gu_buf.deinit();
            p.act_rot_buf.deinit();
            p.yr_buf.deinit();
            p.uids_buf.deinit();
            p.umem_buf.deinit();
            p.ucount_buf.deinit();
        }
    }
};
