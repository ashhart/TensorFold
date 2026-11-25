//! Flash Next TFQ3 (2D-AffineCoset/half4) binary format, zero-copy loader, and Metal pipelines.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

pub const files = @import("flashnext_memory").tfq3_files;
pub const dispatch = @import("tfq3_dispatch.zig");

pub const check = files.check;
pub const magic_layer = check.magic_layer;
pub const magic_v3_tables = check.magic_v3_tables;
pub const format_version = check.format_version;
pub const section_align: usize = 16384;
pub const hidden_size = check.hidden_size;
pub const moe_inter = check.moe_inter;
pub const num_experts = check.num_experts;
pub const topk: u32 = 10;
pub const max_decode_rows: u32 = 16;
pub const max_decode_slots: u32 = max_decode_rows * topk;
pub const max_prefill_tiles: u32 = 4096;

pub const LayerHeader = check.LayerHeader;
pub const ExpertMeta = check.ExpertMeta;

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

/// A section copied into its own buffer (a page-unaligned section, or one whose rounded view leaves the mapping).
fn copySection(device: mtl.Device, src: []const u8) !mtl.Buffer {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const b = try device.buffer(@max(src.len, 16), opts);
    @memcpy(b.contents()[0..src.len], src);
    return b;
}

/// A checked section's view: no-copy when page-aligned inside the mapping, else a copy. The caller owns it.
fn view(device: mtl.Device, mapped: []u8, at: check.Span) !mtl.Buffer {
    const page = std.heap.pageSize();
    const aligned = align16k(at.len);
    if (at.off % page != 0 or aligned % page != 0 or aligned > mapped.len - at.off) return copySection(device, mapped[at.off..][0..at.len]);
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    return device.bufferNoCopy(@ptrCast(mapped.ptr + at.off), aligned, opts);
}

/// Maps a file whole (null: absent); the caller owns the mapping until it hands it to `maps`.
fn mapFile(path: [:0]const u8, least: usize) !?struct { bytes: []align(std.heap.page_size_min) u8, len: usize } {
    const fd = platform.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 0 or end < least) return error.Tfq3Truncated;
    const len = std.math.cast(usize, end) orelse return error.Tfq3Overflow;
    const bytes = try std.posix.mmap(null, std.mem.alignForward(usize, len, std.heap.pageSize()), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    return .{ .bytes = bytes, .len = len };
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
    const m = (try mapFile(buf[0..s.len :0], 8)) orelse return null;
    errdefer std.posix.munmap(m.bytes);
    try check.tables(m.bytes[0..m.len], m.len);
    try held.ensureUnusedCapacity(arena, 1);
    try maps.ensureUnusedCapacity(arena, 1);
    const b = try view(device, m.bytes, .{ .off = 0, .len = m.len });
    // nothing fails past here: the mapping and its view pass to their owners together
    held.appendAssumeCapacity(b);
    maps.appendAssumeCapacity(.{ .bytes = m.bytes });
    total_bytes.* += m.len;
    return b;
}

/// A shard checked whole (check.layer, as admission checked it) before any view; `layer_idx` null for the MTP.
fn loadShard(
    arena: std.mem.Allocator,
    device: mtl.Device,
    held: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    path: [:0]const u8,
    layer_idx: ?u32,
    v3_tables: ?mtl.Buffer,
) !Layer {
    const m = (try mapFile(path, @sizeOf(LayerHeader))) orelse return error.OpenFailed;
    errdefer std.posix.munmap(m.bytes);
    const at = try check.layer(m.bytes[0..m.len], layer_idx);
    const tb = v3_tables orelse return error.Tfq3Table;
    try held.ensureUnusedCapacity(arena, 3);
    try maps.ensureUnusedCapacity(arena, 1);
    const b_meta = try view(device, m.bytes, at.meta);
    errdefer b_meta.deinit();
    const b_sc = try view(device, m.bytes, at.scales);
    errdefer b_sc.deinit();
    const b_wt = try view(device, m.bytes, at.weights);
    // nothing fails past here: the mapping and its views pass to their owners together
    for ([_]mtl.Buffer{ b_meta, b_sc, b_wt }) |b| held.appendAssumeCapacity(b);
    maps.appendAssumeCapacity(.{ .bytes = m.bytes });
    total_bytes.* += m.len;
    return .{ .layer_idx = layer_idx orelse 48, .bytes = m.len, .tables = tb, .meta = b_meta, .scales = b_sc, .weights = b_wt, .has_5b = at.has_5b };
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
    return loadShard(arena, device, held, maps, total_bytes, buf[0..s.len :0], null, v3_tables);
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
        return initUnits(gpa, device, held, device.tensorUnits());
    }

    /// `units`: the M5 tensor-unit source, else the pre-M5 simdgroup one; prefill is right only on its own chip.
    pub fn initUnits(gpa: std.mem.Allocator, device: mtl.Device, held: ?*std.ArrayList(mtl.Buffer), units: bool) !Pipes {
        const src = try buildSource(gpa, units);
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
