//! GLM-5.3-Flash TFQ3 (2D-AffineCoset/half4) binary format, zero-copy loader, and Metal pipelines.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const affine = @import("affine.zig");
const quant = @import("quant.zig");

pub const dispatch = @import("tfq3_dispatch.zig");

pub const magic_layer: u32 = 0x57504654;
pub const magic_nonmoe: u32 = 0x4E504654;
pub const magic_v3_tables: u32 = 0x33565750;
pub const format_version: u32 = 3;
pub const section_align: usize = 16384;
pub const max_decode_slots: u32 = 128;
pub const max_xrot_elems: u32 = 16384 * 4096;
pub const max_yr_elems: u32 = 8 * 154880;
pub const max_prefill_tiles: u32 = 16384;

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

pub const Dims = extern struct {
    rows: u32,
    slots: u32,
    topk: u32,
    experts: u32,
    lim: f32,
    routed_scale: f32 = 1.0,
    flags: u32 = 0,
    pad: u32 = 0,
};

pub const DenseDims = extern struct {
    rows: u32,
    n: u32,
    k: u32,
    ldx: u32,
    ldo: u32,
    lim: f32 = 0.0,
    ksplit: u32 = 1,
    pad1: u32 = 0,
};

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

pub const NonMoe = struct {
    tables: mtl.Buffer,
    scales: mtl.Buffer,
    weights: mtl.Buffer,
    metas: []const NonMoeTensorMeta,
    bytes: usize,

    pub fn find(self: *const NonMoe, name: []const u8) ?*const NonMoeTensorMeta {
        for (self.metas) |*m| {
            if (std.mem.eql(u8, m.nameSlice(), name)) return m;
        }
        return null;
    }

    pub fn runOf(self: *const NonMoe, pm: *const NonMoeTensorMeta, row: usize, n: usize) affine.Run {
        return .{
            .w = .{ .buf = self.weights, .off = @intCast(pm.word_off * 4) },
            .s = .{ .buf = self.scales, .off = @as(usize, pm.sc_off) * 2 },
            .b = .{ .buf = self.tables, .off = 0 },
            .row = @intCast(row),
            .n = @intCast(n),
            .bits = @intCast(pm.b_bits),
            .group = 512,
        };
    }
};

pub const LoadCtx = struct {
    pw_tables: ?mtl.Buffer = null,
    nonmoe: ?NonMoe = null,
};

pub fn matchPart(nm: ?*const NonMoe, name: []const u8, lo: usize, x: usize, all: usize, k: usize, slot: *?*const NonMoeTensorMeta, w_out: *quant.Width, any_pw: *bool) bool {
    const store = nm orelse return false;
    if (lo != 0 or x != all) return false;
    const pm = store.find(name) orelse return false;
    if (pm.R != all or pm.C != k) return false;
    slot.* = pm;
    w_out.* = .{ .bits = @intCast(pm.b_bits), .group = 512 };
    any_pw.* = true;
    return true;
}

pub fn emitRun(nm: *const NonMoe, pm: *const NonMoeTensorMeta, out: *affine.Affine, r: usize, row: *usize, n: usize, k: usize) void {
    const run = nm.runOf(pm, row.*, n);
    if (r == 0) out.* = .{ .w = run.w, .s = run.s, .b = run.b, .n = 0, .k = @intCast(k), .bits = run.bits, .group = 512 } else out.runs[r - 1] = run;
    row.* += n;
}

pub fn initLoader(l: anytype) !void {
    l.tfq3_ctx.pw_tables = try loadV3Tables(l.gpa, l.device, &l.w.buffers, &l.w.tfq3_maps, &l.w.bytes, l.dir, l.dry);
    l.tfq3_ctx.nonmoe = try loadNonMoe(l.gpa, l.arena, l.device, &l.w.buffers, &l.w.tfq3_maps, &l.w.bytes, l.dir, l.tfq3_ctx.pw_tables, l.dry);
    if (l.tfq3_ctx.pw_tables != null or l.tfq3_ctx.nonmoe != null) l.w.has_tfq3 = true;
}

pub fn loadMoe(moe: anytype, l: anytype, i: usize, inter: usize, hidden: usize) !bool {
    if (!hasLayer(l.dir, i)) return false;
    moe.tfq3 = try loadLayer(l.gpa, l.device, &l.w.buffers, &l.w.tfq3_maps, &l.w.bytes, l.dir, i, l.tfq3_ctx.pw_tables, l.dry);
    l.w.has_tfq3 = true;
    const dummy: affine.Ref = .{ .buf = .{ .id = undefined }, .off = 0 };
    moe.gate = .{ .w = dummy, .s = dummy, .b = dummy, .n = @intCast(inter), .k = @intCast(hidden), .bits = 8, .group = 64 };
    moe.up = moe.gate;
    moe.down = .{ .w = dummy, .s = dummy, .b = dummy, .n = @intCast(hidden), .k = @intCast(inter), .bits = 8, .group = 64 };
    return true;
}

/// Partition a stacked projection into runs, merging contiguous TFQ3 tensors (KDA q/k/v, Shared gate/up, MLA q_b+wq_b).
pub fn planRuns(
    n_parts: usize,
    pws: []const ?*const NonMoeTensorMeta,
    ws: []const quant.Width,
    any_pw: bool,
    mixed_widths: bool,
    starts: *[affine.max_runs + 1]usize,
) usize {
    if (!any_pw) {
        const runs = if (mixed_widths) n_parts else 1;
        for (0..runs) |r| starts[r] = r;
        starts[runs] = n_parts;
        return runs;
    }
    var runs: usize = 0;
    var idx: usize = 0;
    while (idx < n_parts) {
        starts[runs] = idx;
        runs += 1;
        if (pws[idx]) |pm0| {
            var cur = pm0;
            var j = idx + 1;
            while (j < n_parts and (pm0.R == 8192 or pm0.R == 2048 or pm0.R == 16384)) : (j += 1) {
                const nxt = pws[j] orelse break;
                const cur_sc_cnt: u32 = (cur.R / 128) * (cur.C / 128);
                const ok_r = (nxt.R == pm0.R) or (pm0.R == 16384 and nxt.R == 4096 and cur.C == 1536);
                if (ok_r and nxt.b_bits == cur.b_bits and nxt.word_off == cur.word_off + cur.word_count and
                    nxt.sc_off == cur.sc_off + cur_sc_cnt)
                {
                    cur = nxt;
                } else break;
            }
            idx = j;
        } else {
            var j = idx + 1;
            while (j < n_parts and pws[j] == null and std.meta.eql(ws[j], ws[idx])) : (j += 1) {}
            idx = j;
        }
    }
    starts[runs] = n_parts;
    return runs;
}

fn layerPath(buf: *[512]u8, dir: []const u8, layer: usize) ?[:0]const u8 {
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/layer_{d:0>2}.tffpw", .{ dir, layer }) catch return null;
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

pub fn hasLayer(dir: []const u8, layer: usize) bool {
    var buf: [512]u8 = undefined;
    const path = layerPath(&buf, dir, layer) orelse return false;
    const fd = platform.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return false;
    _ = platform.c.close(fd);
    return true;
}

fn align16k(n: usize) usize {
    return std.mem.alignForward(usize, @max(n, 16), section_align);
}

fn wrapSection(
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    gpa: std.mem.Allocator,
    base: [*]u8,
    off: u64,
    byte_len: usize,
) !mtl.Buffer {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const b = try device.bufferNoCopy(@ptrCast(base + @as(usize, @intCast(off))), align16k(byte_len), opts);
    errdefer b.deinit();
    try buffers.append(gpa, b);
    return b;
}

pub fn loadV3Tables(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    dry: bool,
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
    if (dry) {
        std.posix.munmap(mapped);
        return .{ .id = undefined };
    }
    try maps.append(gpa, .{ .bytes = mapped });
    return try wrapSection(device, buffers, gpa, mapped.ptr, 0, fsize);
}

pub fn loadNonMoe(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    tables_buf: ?mtl.Buffer,
    dry: bool,
) !?NonMoe {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/nonmoe.tffpw", .{dir}) catch return null;
    buf[s.len] = 0;
    const fd = platform.c.open(buf[0..s.len :0].ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < @sizeOf(NonMoeHeader)) return error.EmptyFile;
    const fsize: usize = @intCast(end);
    total_bytes.* += fsize;
    const page = std.heap.pageSize();
    const mapped = try std.posix.mmap(null, std.mem.alignForward(usize, fsize, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    errdefer std.posix.munmap(mapped);
    const hdr: *const NonMoeHeader = @ptrCast(@alignCast(mapped.ptr));
    if (hdr.magic != magic_nonmoe or hdr.version != format_version) return error.UnsupportedQuantization;
    const n_t: usize = hdr.num_tensors;
    const m_src: [*]const NonMoeTensorMeta = @ptrCast(@alignCast(mapped.ptr + @as(usize, @intCast(hdr.off_meta))));
    const metas = try arena.dupe(NonMoeTensorMeta, m_src[0..n_t]);
    if (dry) {
        std.posix.munmap(mapped);
        return .{ .tables = .{ .id = undefined }, .scales = .{ .id = undefined }, .weights = .{ .id = undefined }, .metas = metas, .bytes = fsize };
    }
    const tb = tables_buf orelse return error.UnsupportedQuantization;
    try maps.append(gpa, .{ .bytes = mapped });
    const b_sc = try wrapSection(device, buffers, gpa, mapped.ptr, hdr.off_scales, @as(usize, @intCast(hdr.total_scale_halfs)) * 2);
    const b_wt = try wrapSection(device, buffers, gpa, mapped.ptr, hdr.off_weights, @as(usize, @intCast(hdr.total_weight_words)) * 4);
    return .{ .tables = tb, .scales = b_sc, .weights = b_wt, .metas = metas, .bytes = fsize };
}

pub fn loadLayer(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    layer: usize,
    v3_tables: ?mtl.Buffer,
    dry: bool,
) !Layer {
    var buf: [512]u8 = undefined;
    const path = layerPath(&buf, dir, layer) orelse return error.MissingTensor;
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
    const metas_ptr: [*]const ExpertMeta = @ptrCast(@alignCast(mapped.ptr + @as(usize, @intCast(ph.off_meta))));
    var has_5b = false;
    for (0..ph.num_experts) |ei| {
        if (metas_ptr[ei].b_bits == 5) {
            has_5b = true;
            break;
        }
    }
    if (dry) {
        std.posix.munmap(mapped);
        return .{ .layer_idx = @intCast(layer), .bytes = fsize, .tables = .{ .id = undefined }, .meta = .{ .id = undefined }, .scales = .{ .id = undefined }, .weights = .{ .id = undefined }, .has_5b = has_5b };
    }
    const tb = v3_tables orelse return error.UnsupportedQuantization;
    try maps.append(gpa, .{ .bytes = mapped });
    const b_meta = try wrapSection(device, buffers, gpa, mapped.ptr, ph.off_meta, @as(usize, ph.num_experts) * @sizeOf(ExpertMeta));
    const b_sc = try wrapSection(device, buffers, gpa, mapped.ptr, ph.off_scales, @as(usize, @intCast(ph.scale_halfs)) * 2);
    const b_wt = try wrapSection(device, buffers, gpa, mapped.ptr, ph.off_weights, @as(usize, @intCast(ph.stream_words)) * 4);
    return .{ .layer_idx = @intCast(layer), .bytes = fsize, .tables = tb, .meta = b_meta, .scales = b_sc, .weights = b_wt, .has_5b = has_5b };
}

pub const Pipe = enum(usize) {
    fwht_x_fwd,
    fwht_glu_fwht,
    fwht_y_inv,
    fwht_y_combine,
    dec_gu_b1_543,
    dec_down_b1_543,
    dec_gu_dedup_543,
    dec_down_dedup_543,
    prefill_nax_glu_543,
    prefill_nax_down_543,
    dec_gu_b1_43,
    dec_down_b1_43,
    dec_gu_dedup_43,
    dec_down_dedup_43,
    prefill_nax_glu_43,
    prefill_nax_down_43,
    tile_scan,
    fwht_x_gen,
    fwht_x_plus_q8,
    fwht_y_gen,
    fwht_glu_unrot,
    dense_dec_fused1_4b,
    dense_dec_fused1_5b,
    dense_dec_fused2_4b,
    dense_dec_fused2_5b,
    dense_dec_fused4_4b,
    dense_dec_fused4_5b,
    dense_dec_nosplit1_4b,
    dense_dec_nosplit1_5b,
    dense_dec_nosplit2_4b,
    dense_dec_nosplit2_5b,
    dense_dec_nosplit4_4b,
    dense_dec_nosplit4_5b,
    dense_dec_2dsk8_1_4b,
    dense_dec_2dsk8_1_5b,
    dense_dec_2dsk8_2_4b,
    dense_dec_2dsk8_2_5b,
    dense_dec_2dsk8_4_4b,
    dense_dec_2dsk8_4_5b,
    dense_prefill_nax_4b,
    dense_prefill_nax_5b,
};

const pipe_names = blk: {
    const names = std.meta.fieldNames(Pipe);
    var out: [names.len][]const u8 = undefined;
    for (names, 0..) |name, i| out[i] = "pw_" ++ name;
    break :blk out;
};

/// Build the full TFQ3 Metal source with nax.h expanded for this GPU (M5 NAX or pre-M5 simdgroup_matrix).
pub fn buildSource(gpa: std.mem.Allocator, units: bool, glue_math: []const u8) ![]u8 {
    const raw = try std.mem.concat(gpa, u8, &.{
        ks.glm_tfq3.header,
        glue_math,
        ks.glm_tfq3.common,
        ks.glm_tfq3.moe_mv,
        ks.glm_tfq3.moe_decode,
        ks.glm_tfq3.dense_decode,
        ks.glm_tfq3.prefill,
    });
    defer gpa.free(raw);
    return frags.precision.source(gpa, units, raw, ks.nax, false);
}

pub const Pipes = struct {
    list: [pipe_names.len]mtl.Pipeline,
    xrot_buf: mtl.Buffer,
    gu_buf: mtl.Buffer,
    sact_buf: mtl.Buffer,
    yr_buf: mtl.Buffer,
    tile_meta_buf: mtl.Buffer,
    tile_cnt_buf: mtl.Buffer,
    hash: u64,

    pub inline fn get(p: *const Pipes, k: Pipe) mtl.Pipeline {
        return p.list[@backingInt(k)];
    }

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, glue_math: []const u8) !Pipes {
        const src = try buildSource(gpa, device.tensorUnits(), glue_math);
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
        const b_xrot = try device.buffer(@as(usize, max_xrot_elems) * 2, mtl.ResourceOptions.private);
        errdefer b_xrot.deinit();
        const b_gu = try device.buffer(@as(usize, max_decode_slots + 16) * 2 * 2048 * 4, mtl.ResourceOptions.private);
        errdefer b_gu.deinit();
        const b_sact = try device.buffer(16 * 2048 * 2, mtl.ResourceOptions.private);
        errdefer b_sact.deinit();
        const b_yr = try device.buffer(@as(usize, max_yr_elems) * 4, mtl.ResourceOptions.private);
        errdefer b_yr.deinit();
        const b_tmeta = try device.buffer(@as(usize, max_prefill_tiles) * 4, mtl.ResourceOptions.private);
        errdefer b_tmeta.deinit();
        const b_tcnt = try device.buffer(16, mtl.ResourceOptions.private);
        errdefer b_tcnt.deinit();
        return .{ .list = list, .xrot_buf = b_xrot, .gu_buf = b_gu, .sact_buf = b_sact, .yr_buf = b_yr, .tile_meta_buf = b_tmeta, .tile_cnt_buf = b_tcnt, .hash = h.final() };
    }

    pub fn deinit(p: *Pipes) void {
        for (p.list) |pl| pl.deinit();
        p.xrot_buf.deinit();
        p.gu_buf.deinit();
        p.sact_buf.deinit();
        p.yr_buf.deinit();
        p.tile_meta_buf.deinit();
        p.tile_cnt_buf.deinit();
    }
};
