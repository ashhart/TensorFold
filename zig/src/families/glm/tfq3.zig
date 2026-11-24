//! GLM-5.3-Flash TFQ3 (2D-AffineCoset/half4) binary format, zero-copy loader, and Metal pipelines.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const affine = @import("affine.zig");
const quant = @import("quant.zig");

pub const dispatch = @import("tfq3_dispatch.zig");

pub const check = @import("tfq3_check.zig");
pub const magic_layer = check.magic_layer;
pub const magic_nonmoe = check.magic_nonmoe;
pub const magic_v3_tables = check.magic_v3_tables;
pub const format_version = check.format_version;
pub const section_align = check.section_align;
pub const max_decode_slots: u32 = 128;
pub const max_xrot_elems: u32 = 16384 * 4096;
pub const max_yr_elems: u32 = 8 * 154880;
pub const max_prefill_tiles: u32 = 16384;

pub const LayerHeader = check.LayerHeader;
pub const ExpertMeta = check.ExpertMeta;
pub const NonMoeHeader = check.NonMoeHeader;
pub const NonMoeTensorMeta = check.NonMoeTensorMeta;

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

/// A named record, asked alike by planner and `stack`: null, the ordinary tensor; one it cannot read whole, refused.
pub fn record(ctx: *const LoadCtx, name: []const u8, lo: usize, x: usize, all: usize, k: usize) !?*const NonMoeTensorMeta {
    const store = if (ctx.nonmoe) |*nm| nm else return null;
    const pm = store.find(name) orelse return null;
    check.recordFits(pm.R, pm.C, lo, x, all, k) catch |err| {
        std.log.err("nonmoe.tffpw: {s} is {d}x{d}; the model reads rows [{d}, {d}) of {d}x{d} ({t})", .{ name, pm.R, pm.C, lo, lo + x, all, k, err });
        return err;
    };
    return pm;
}

pub fn matchPart(ctx: *const LoadCtx, name: []const u8, lo: usize, x: usize, all: usize, k: usize, slot: *?*const NonMoeTensorMeta, w_out: *quant.Width, any_pw: *bool) !bool {
    const pm = (try record(ctx, name, lo, x, all, k)) orelse return false;
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

/// A KDA layer whose stacked input holds TFQ3 runs and whose f_b and g_b gates are 8-bit/64: the q8 kernels' case.
pub fn q8Kda(in: affine.Affine, f: affine.Affine, g: affine.Affine) bool {
    var tfq3_in = false;
    for (0..@as(usize, in.more) + 1) |i| tfq3_in = tfq3_in or in.run(i).group == 512;
    return tfq3_in and f.more == 0 and g.more == 0 and f.group == 64 and g.group == 64 and f.bits == 8 and g.bits == 8;
}

/// moe.zig's one shared-fusion decision over the stored shared projections (check.sharedFusion's rule).
pub fn sharedFusion(gu: affine.Affine, down: affine.Affine, has_5b: bool, hidden: usize, inter: usize, stages: bool, combine: bool) bool {
    const g: check.Proj = .{ .runs = @as(usize, gu.more) + 1, .bits = gu.bits, .group = gu.group, .n = gu.n, .k = gu.k };
    const d: check.Proj = .{ .runs = @as(usize, down.more) + 1, .bits = down.bits, .group = down.group, .n = down.n, .k = down.k };
    return check.sharedFusion(g, d, has_5b, hidden, inter, stages, combine);
}

pub fn initLoader(l: anytype) !void {
    l.tfq3_ctx.pw_tables = try loadV3Tables(l.gpa, l.device, &l.w.buffers, &l.w.tfq3_maps, &l.w.bytes, l.dir, l.dry);
    l.tfq3_ctx.nonmoe = try loadNonMoe(l.gpa, l.arena, l.device, &l.w.buffers, &l.w.tfq3_maps, &l.w.bytes, l.dir, l.tfq3_ctx.pw_tables, l.dry);
    if (l.tfq3_ctx.pw_tables != null or l.tfq3_ctx.nonmoe != null) l.w.has_tfq3 = true;
    if (l.w.has_tfq3 and l.c.tp != 1) { // before any allocation: a rank's share of a record is never read
        std.log.err("TFQ3 files load on one Mac: nonmoe.tffpw's records are whole tensors, not a rank's share (--tp {d})", .{l.c.tp});
        return error.Tfq3TensorParallel;
    }
}

pub fn loadMoe(moe: anytype, l: anytype, i: usize, inter: usize, hidden: usize) !bool {
    if (!hasLayer(l.dir, i)) return false;
    const want: check.LayerWant = .{ .layer = i, .experts = l.c.experts, .hidden = hidden, .inter = inter };
    moe.tfq3 = try loadLayer(l.gpa, l.device, &l.w.buffers, &l.w.tfq3_maps, &l.w.bytes, l.dir, want, l.tfq3_ctx.pw_tables, l.dry);
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

/// A file mapped whole (rounded to sections) and read only through `check`'s spans.
const Mapped = struct {
    bytes: []align(std.heap.page_size_min) u8,
    len: usize, // the file's own bytes; the mapping past them is zero

    fn file(m: Mapped) []const u8 {
        return m.bytes[0..m.len];
    }
};

/// Maps `dir/name` (null: no such file); the caller owns the mapping until it hands it to `maps`.
fn mapFile(dir: []const u8, name: []const u8, least: usize) !?Mapped {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/{s}", .{ dir, name }) catch return error.NameTooLong;
    buf[s.len] = 0;
    const fd = platform.c.open(buf[0..s.len :0].ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 0 or end < least) return error.EmptyFile;
    const len = std.math.cast(usize, end) orelse return error.Tfq3Overflow;
    const room = try check.mappedLen(len);
    const bytes = try std.posix.mmap(null, std.mem.alignForward(usize, room, std.heap.pageSize()), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    return .{ .bytes = bytes, .len = len };
}

/// A no-copy view of a checked span (16 KiB aligned, its rounded length inside the mapping).
fn view(device: mtl.Device, m: Mapped, at: check.Span) !mtl.Buffer {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    return device.bufferNoCopy(@ptrCast(m.bytes.ptr + at.off), try check.mappedLen(@max(at.len, 16)), opts);
}

/// Hands `m` and `views` to the weights in one step that cannot fail: room is reserved first.
fn keep(buffers: *std.ArrayList(mtl.Buffer), maps: *std.ArrayList(MappedShard), m: Mapped, views: []const mtl.Buffer) void {
    for (views) |b| buffers.appendAssumeCapacity(b);
    maps.appendAssumeCapacity(.{ .bytes = m.bytes });
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
    const m = (try mapFile(dir, "pw_v3_tables.bin", 16384)) orelse return null;
    errdefer std.posix.munmap(m.bytes);
    try check.tables(m.file());
    const all = try check.span(m.len, 0, m.len, true);
    if (dry) {
        std.posix.munmap(m.bytes);
        total_bytes.* += m.len;
        return .{ .id = undefined };
    }
    try buffers.ensureUnusedCapacity(gpa, 1);
    try maps.ensureUnusedCapacity(gpa, 1);
    const b = try view(device, m, all);
    keep(buffers, maps, m, &.{b});
    total_bytes.* += m.len;
    return b;
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
    const m = (try mapFile(dir, "nonmoe.tffpw", @sizeOf(NonMoeHeader))) orelse return null;
    errdefer std.posix.munmap(m.bytes);
    const at = try check.nonMoe(m.file());
    const tb = tables_buf orelse return error.UnsupportedQuantization; // its rows decode through the shared table
    const metas = try arena.alloc(NonMoeTensorMeta, at.count);
    @memcpy(std.mem.sliceAsBytes(metas), m.file()[at.metas.off..][0..at.metas.len]);
    if (dry) {
        std.posix.munmap(m.bytes);
        total_bytes.* += m.len;
        return .{ .tables = .{ .id = undefined }, .scales = .{ .id = undefined }, .weights = .{ .id = undefined }, .metas = metas, .bytes = m.len };
    }
    try buffers.ensureUnusedCapacity(gpa, 2);
    try maps.ensureUnusedCapacity(gpa, 1);
    const b_sc = try view(device, m, at.scales);
    errdefer b_sc.deinit();
    const b_wt = try view(device, m, at.weights);
    keep(buffers, maps, m, &.{ b_sc, b_wt });
    total_bytes.* += m.len;
    return .{ .tables = tb, .scales = b_sc, .weights = b_wt, .metas = metas, .bytes = m.len };
}

pub fn loadLayer(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: *std.ArrayList(mtl.Buffer),
    maps: *std.ArrayList(MappedShard),
    total_bytes: *usize,
    dir: []const u8,
    want: check.LayerWant,
    v3_tables: ?mtl.Buffer,
    dry: bool,
) !Layer {
    var name: [32]u8 = undefined;
    const file = try std.fmt.bufPrint(&name, "layer_{d:0>2}.tffpw", .{want.layer});
    const m = (try mapFile(dir, file, @sizeOf(LayerHeader))) orelse return error.OpenFailed;
    errdefer std.posix.munmap(m.bytes);
    const at = try check.layer(m.file(), want);
    const tb = v3_tables orelse return error.UnsupportedQuantization;
    const idx: u32 = @intCast(want.layer);
    if (dry) {
        std.posix.munmap(m.bytes);
        total_bytes.* += m.len;
        return .{ .layer_idx = idx, .bytes = m.len, .tables = .{ .id = undefined }, .meta = .{ .id = undefined }, .scales = .{ .id = undefined }, .weights = .{ .id = undefined }, .has_5b = at.has_5b };
    }
    try buffers.ensureUnusedCapacity(gpa, 3);
    try maps.ensureUnusedCapacity(gpa, 1);
    const b_meta = try view(device, m, at.meta);
    errdefer b_meta.deinit();
    const b_sc = try view(device, m, at.scales);
    errdefer b_sc.deinit();
    const b_wt = try view(device, m, at.weights);
    keep(buffers, maps, m, &.{ b_meta, b_sc, b_wt });
    total_bytes.* += m.len;
    return .{ .layer_idx = idx, .bytes = m.len, .tables = tb, .meta = b_meta, .scales = b_sc, .weights = b_wt, .has_5b = at.has_5b };
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
    kda_q8: ?[2]mtl.Pipeline = null, // the KDA parts and window at 8-bit/64 gates (kernels.zig enableTfq3)
    hash: u64,

    pub inline fn get(p: *const Pipes, k: Pipe) mtl.Pipeline {
        return p.list[@backingInt(k)];
    }

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, glue_math: []const u8) !Pipes {
        return initUnits(gpa, device, device.tensorUnits(), glue_math);
    }

    /// `units`: the M5 tensor-unit source, else the simdgroup one (a test runs both where the GPU can).
    pub fn initUnits(gpa: std.mem.Allocator, device: mtl.Device, units: bool, glue_math: []const u8) !Pipes {
        const src = try buildSource(gpa, units, glue_math);
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
        if (p.kda_q8) |q| for (q) |x| x.deinit();
    }
};
