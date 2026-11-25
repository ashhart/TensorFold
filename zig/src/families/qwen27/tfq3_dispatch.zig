//! Qwen27 TFQ3 (3/4/5-bit 2D-AffineCoset/half4) and BF16 linear Metal pipelines and dispatch.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("core").lane_projection.frags;
const projection = @import("projection.zig");
const Ref = projection.Ref;

pub const max_prefill_chunk: u32 = 512;
pub const max_xrot_elems: usize = 512 * 17408;
pub const max_yr_elems: usize = 8 * 4 * 34816;

pub const DenseDims = extern struct {
    rows: u32,
    n: u32,
    k: u32,
    ldx: u32,
    ldo: u32,
    sr_split: u32 = 0,
    ksplit: u32 = 1,
    pad: u32 = 0,
};

pub const Pipe = enum(usize) {
    pw_fwht_x,
    pw_fwht_y,
    bf16_mv1,
    bf16_mm,
    pw_dec_fused1_3b,
    pw_dec_fused1_4b,
    pw_dec_fused1_5b,
    pw_dec_fused2_3b,
    pw_dec_fused2_4b,
    pw_dec_fused2_5b,
    pw_dec_fused4_3b,
    pw_dec_fused4_4b,
    pw_dec_fused4_5b,
    pw_dec_2dsk8_1_3b,
    pw_dec_2dsk8_1_4b,
    pw_dec_2dsk8_1_5b,
    pw_dec_2dsk8_2_3b,
    pw_dec_2dsk8_2_4b,
    pw_dec_2dsk8_2_5b,
    pw_dec_2dsk8_4_3b,
    pw_dec_2dsk8_4_4b,
    pw_dec_2dsk8_4_5b,
    pw_dense_prefill_3b,
    pw_dense_prefill_4b,
    pw_dense_prefill_5b,
};

const pipe_names = blk: {
    const names = std.meta.fieldNames(Pipe);
    var out: [names.len][]const u8 = undefined;
    for (names, 0..) |name, i| out[i] = "q27_" ++ name;
    break :blk out;
};

/// Builds Qwen27 TFQ3 + BF16 Metal source with nax.h expanded for M5 NAX or pre-M5 simdgroup_matrix.
pub fn buildSource(gpa: std.mem.Allocator, units: bool) ![]u8 {
    const raw = try std.mem.concat(gpa, u8, &.{
        ks.qwen27_tfq3.header,
        ks.qwen27_tfq3.common,
        ks.qwen27_tfq3.dense_decode,
        ks.qwen27_tfq3.dense_prefill,
    });
    defer gpa.free(raw);
    return frags.precision.source(gpa, units, raw, ks.nax, false);
}

pub const Pipes = struct {
    list: [pipe_names.len]mtl.Pipeline,
    xrot_buf: mtl.Buffer,
    yr_buf: mtl.Buffer,

    pub inline fn get(p: *const Pipes, k: Pipe) mtl.Pipeline {
        return p.list[@backingInt(k)];
    }

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device) !Pipes {
        return initUnits(gpa, device, device.tensorUnits());
    }

    pub fn initUnits(gpa: std.mem.Allocator, device: mtl.Device, units: bool) !Pipes {
        const src = try buildSource(gpa, units);
        defer gpa.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var list: [pipe_names.len]mtl.Pipeline = undefined;
        var built: usize = 0;
        errdefer for (list[0..built]) |pl| pl.deinit();
        for (pipe_names, 0..) |name, i| {
            list[i] = try mtl.Pipeline.init(device, lib, name, false);
            built += 1;
        }
        const xrot_buf = try device.buffer(max_xrot_elems * 2, mtl.ResourceOptions.private);
        errdefer xrot_buf.deinit();
        const yr_buf = try device.buffer(max_yr_elems * 4, mtl.ResourceOptions.private);
        return .{ .list = list, .xrot_buf = xrot_buf, .yr_buf = yr_buf };
    }

    pub fn setResident(p: *const Pipes, hook: mtl.ResidencySet.Follow, joining: bool) void {
        mtl.resident.fields.one(p.xrot_buf, hook, joining);
        mtl.resident.fields.one(p.yr_buf, hook, joining);
    }

    pub fn deinit(p: *Pipes) void {
        for (p.list) |pl| pl.deinit();
        p.xrot_buf.deinit();
        p.yr_buf.deinit();
    }
};

fn bitIndex(bits: u8) !usize {
    return switch (bits) {
        3 => 0,
        4 => 1,
        5 => 2,
        else => error.Tfq3Bits,
    };
}

fn fusedPipe(rows: u32, bi: usize) Pipe {
    const base: usize = if (rows == 1)
        @backingInt(Pipe.pw_dec_fused1_3b)
    else if (rows == 2)
        @backingInt(Pipe.pw_dec_fused2_3b)
    else
        @backingInt(Pipe.pw_dec_fused4_3b);
    return @fromBackingInt(@intCast(base + bi));
}

fn splitPipe(rows: u32, bi: usize) Pipe {
    const base: usize = if (rows == 1)
        @backingInt(Pipe.pw_dec_2dsk8_1_3b)
    else if (rows == 2)
        @backingInt(Pipe.pw_dec_2dsk8_2_3b)
    else
        @backingInt(Pipe.pw_dec_2dsk8_4_3b);
    return @fromBackingInt(@intCast(base + bi));
}

fn prefillPipe(bi: usize) Pipe {
    return @fromBackingInt(@intCast(@backingInt(Pipe.pw_dense_prefill_3b) + bi));
}

/// Encodes one TFQ3 (3/4/5-bit) dense projection for `rows` with input stride `ldx` and output stride `ldo`.
pub fn encodeTfq3(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    t: projection.Tfq3,
    bits: u8,
    n: u32,
    k: u32,
    x: Ref,
    ldx: u32,
    y: Ref,
    ldo: u32,
    rows: u32,
) !void {
    if (rows == 0 or n == 0 or k == 0 or n % 128 != 0 or k % 128 != 0 or ldx < k or ldo < n) return error.BadCoreProjectionRows;
    const bi = try bitIndex(bits);
    if (rows <= 16) {
        const n_bc = k >> 7;
        const use_sk8 = n <= 34816 and n_bc >= 8 and n_bc % 8 == 0 and @as(usize, rows) * n * 8 <= max_yr_elems;
        const dims: DenseDims = .{ .rows = rows, .n = n, .k = k, .ldx = ldx, .ldo = ldo, .sr_split = t.sr_split, .ksplit = if (use_sk8) 8 else 1 };
        e.setPipeline(p.get(.pw_fwht_x));
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(p.xrot_buf, 0, 1);
        e.setValue(dims, 2);
        e.dispatchGroups(mtl.Size.of(n_bc, rows, 1), mtl.Size.of(32, 1, 1));
        e.barrier();

        if (use_sk8) {
            e.setPipeline(p.get(splitPipe(rows, bi)));
            e.setBuffer(t.words.buffer, t.words.offset, 0);
            e.setBuffer(t.scales.buffer, t.scales.offset, 1);
            e.setBuffer(t.tables.buffer, t.tables.offset, 2);
            e.setBuffer(p.xrot_buf, 0, 3);
            e.setBuffer(p.yr_buf, 0, 4);
            e.setValue(dims, 5);
            e.dispatchGroups(mtl.Size.of(n / 32, 8, 1), mtl.Size.of(64, 1, 1));
            e.barrier();

            e.setPipeline(p.get(.pw_fwht_y));
            e.setBuffer(p.yr_buf, 0, 0);
            e.setBuffer(y.buffer, y.offset, 1);
            e.setValue(dims, 2);
            e.dispatchGroups(mtl.Size.of(n / 128, rows, 1), mtl.Size.of(32, 1, 1));
        } else {
            e.setPipeline(p.get(fusedPipe(rows, bi)));
            e.setBuffer(t.words.buffer, t.words.offset, 0);
            e.setBuffer(t.scales.buffer, t.scales.offset, 1);
            e.setBuffer(t.tables.buffer, t.tables.offset, 2);
            e.setBuffer(p.xrot_buf, 0, 3);
            e.setBuffer(y.buffer, y.offset, 4);
            e.setValue(dims, 5);
            e.dispatchGroups(mtl.Size.of(n / 128, 1, 1), mtl.Size.of(256, 1, 1));
        }
        return;
    }

    const cap_rows: u32 = @intCast(@min(@as(usize, max_prefill_chunk), max_xrot_elems / k));
    if (cap_rows == 0) return error.BadCoreProjectionRows;
    var done: u32 = 0;
    while (done < rows) {
        const take = @min(rows - done, cap_rows);
        const dims: DenseDims = .{ .rows = take, .n = n, .k = k, .ldx = ldx, .ldo = ldo, .sr_split = t.sr_split, .ksplit = 1 };
        const x_off = x.offset + @as(usize, done) * ldx * 2;
        const y_off = y.offset + @as(usize, done) * ldo * 2;
        e.setPipeline(p.get(.pw_fwht_x));
        e.setBuffer(x.buffer, x_off, 0);
        e.setBuffer(p.xrot_buf, 0, 1);
        e.setValue(dims, 2);
        e.dispatchGroups(mtl.Size.of(k >> 7, take, 1), mtl.Size.of(32, 1, 1));
        e.barrier();

        e.setPipeline(p.get(prefillPipe(bi)));
        e.setBuffer(t.words.buffer, t.words.offset, 0);
        e.setBuffer(t.scales.buffer, t.scales.offset, 1);
        e.setBuffer(t.tables.buffer, t.tables.offset, 2);
        e.setBuffer(p.xrot_buf, 0, 3);
        e.setBuffer(y.buffer, y_off, 4);
        e.setValue(dims, 5);
        e.dispatchGroups(mtl.Size.of(n / 128, (take + 31) / 32, 1), mtl.Size.of(128, 1, 1));
        done += take;
        if (done < rows) e.barrier();
    }
}

/// Encodes one BF16 dense linear projection for residual modules (`in_proj_a/b`, optional BF16 `lm_head`).
pub fn encodeBf16(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    w: Ref,
    n: u32,
    k: u32,
    x: Ref,
    ldx: u32,
    y: Ref,
    ldo: u32,
    rows: u32,
) !void {
    if (rows == 0 or n == 0 or k == 0 or k % 4 != 0 or ldx < k or ldo < n) return error.BadCoreProjectionRows;
    const dims: DenseDims = .{ .rows = rows, .n = n, .k = k, .ldx = ldx, .ldo = ldo };
    if (rows == 1) {
        e.setPipeline(p.get(.bf16_mv1));
        e.setBuffer(w.buffer, w.offset, 0);
        e.setBuffer(x.buffer, x.offset, 1);
        e.setBuffer(y.buffer, y.offset, 2);
        e.setValue(dims, 3);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
    } else {
        e.setPipeline(p.get(.bf16_mm));
        e.setBuffer(w.buffer, w.offset, 0);
        e.setBuffer(x.buffer, x.offset, 1);
        e.setBuffer(y.buffer, y.offset, 2);
        e.setValue(dims, 3);
        e.dispatchGroups(mtl.Size.of((n + 3) / 4, (rows + 3) / 4, 1), mtl.Size.of(128, 1, 1));
    }
}
