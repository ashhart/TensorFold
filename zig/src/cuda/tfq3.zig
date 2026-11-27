//! Shared CUDA TFQ3 (3/4/5-bit 2D-AffineCoset/half4 + 2D FWHT-128) dense & MoE runtime.
const std = @import("std");
const fmt = @import("tfq3_format.zig");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const memory = @import("memory.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const DeviceBuffer = memory.DeviceBuffer;

/// One dense / non-MoE TFQ3 matrix [n, k] on the device (from a `TFPN` shard).
pub const DenseWeight = struct {
    words: u64,
    scales: u64,
    tables: u64,
    n: u32,
    k: u32,
    bits: u8,
    sr_split: u32 = 0,
    row_off: u32 = 0,

    /// Output rows [from, from + count) as a 128-row-aligned view keeping exact row sign seeds.
    pub fn sliceRows(w: DenseWeight, from: u32, count: u32) !DenseWeight {
        if (from % 128 != 0 or count % 128 != 0 or count == 0 or @as(u64, from) + count > w.n) return error.Tfq3Shape;
        const n_bc: u64 = w.k >> 7;
        const w_off: u64 = @as(u64, from / 16) * n_bc * @as(u64, w.bits) * 2 * 32 * 4;
        const s_off: u64 = @as(u64, from / 128) * n_bc * 2;
        return .{
            .words = w.words + w_off,
            .scales = w.scales + s_off,
            .tables = w.tables,
            .n = count,
            .k = w.k,
            .bits = w.bits,
            .sr_split = w.sr_split,
            .row_off = w.row_off + from,
        };
    }
};

/// One routed MoE layer on the device (from a `TFPW` shard).
pub const MoeLayer = struct {
    weights: u64,
    scales: u64,
    metas: u64,
    tables: u64,
    experts: u32,
    dims: u32,
    inter: u32,
};

pub const OwnedMoeLayer = struct {
    layer: MoeLayer,
    w_buf: DeviceBuffer,
    s_buf: DeviceBuffer,
    m_buf: DeviceBuffer,

    pub fn deinit(self: *OwnedMoeLayer) void {
        self.m_buf.free();
        self.s_buf.free();
        self.w_buf.free();
    }
};

/// Compiled TFQ3 dense + MoE CUDA modules and preallocated scratch (CUDA-graph safe).
pub const Kernels = struct {
    d: *const Driver,
    mod_dense: Module,
    mod_moe: Module,
    fwht_x: Function,
    fwht_y: Function,
    wmma_sk8: [3]Function, // 3b, 4b, 5b: split-K=8 BM=16 (all widths, fixed reduction order)
    wmma_fused: [3]Function, // 3b, 4b, 5b: fused BM=16 (wide N, small K, rows < 48)
    wmma_prefill64: [3]Function, // 3b, 4b, 5b: fused BM=64, BN=128 Tensor Core prefill (rows >= 48)
    moe_gu: Function,
    moe_gu_softcap: Function,
    moe_down: Function,
    xrot: ?DeviceBuffer = null,
    xrot_elems: usize = 0,
    yr: ?DeviceBuffer = null,
    yr_elems: usize = 0,
    act_rot: ?DeviceBuffer = null,
    act_rot_elems: usize = 0,

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod_dense = try Module.load(d, kernels.tfq3);
        errdefer mod_dense.unload();
        var mod_moe = try Module.load(d, kernels.tfq3_moe);
        errdefer mod_moe.unload();
        var k: Kernels = .{
            .d = d,
            .mod_dense = mod_dense,
            .mod_moe = mod_moe,
            .fwht_x = try mod_dense.function("tf_tfq3_fwht_x"),
            .fwht_y = try mod_dense.function("tf_tfq3_fwht_y"),
            .wmma_sk8 = undefined,
            .wmma_fused = undefined,
            .wmma_prefill64 = undefined,
            .moe_gu = try mod_moe.function("tf_tfq3_moe_gu_swiglu"),
            .moe_gu_softcap = try mod_moe.function("tf_tfq3_moe_gu_softcap"),
            .moe_down = try mod_moe.function("tf_tfq3_moe_down"),
        };
        inline for ([_]u8{ 3, 4, 5 }, 0..) |b, i| {
            const sb = std.fmt.comptimePrint("{d}b", .{b});
            k.wmma_sk8[i] = try mod_dense.function("tf_tfq3_wmma_sk8_" ++ sb);
            k.wmma_fused[i] = try mod_dense.function("tf_tfq3_wmma_fused_" ++ sb);
            k.wmma_prefill64[i] = try mod_dense.function("tf_tfq3_wmma_prefill64_" ++ sb);
        }
        return k;
    }

    /// Preallocates rotation and split-K scratch before any CUDA graph capture.
    pub fn reserve(k: *Kernels, max_xrot_elems: usize, max_yr_elems: usize, max_act_rot_elems: usize) !void {
        var x = if (max_xrot_elems > k.xrot_elems) try DeviceBuffer.alloc(k.d, try std.math.mul(usize, max_xrot_elems, 2)) else null;
        errdefer if (x) |*v| v.free();
        var y = if (max_yr_elems > k.yr_elems) try DeviceBuffer.alloc(k.d, try std.math.mul(usize, max_yr_elems, 4)) else null;
        errdefer if (y) |*v| v.free();
        var a = if (max_act_rot_elems > k.act_rot_elems) try DeviceBuffer.alloc(k.d, try std.math.mul(usize, max_act_rot_elems, 2)) else null;
        errdefer if (a) |*v| v.free();
        if (x) |v| {
            if (k.xrot) |*old| old.free();
            k.xrot = v;
            k.xrot_elems = max_xrot_elems;
        }
        if (y) |v| {
            if (k.yr) |*old| old.free();
            k.yr = v;
            k.yr_elems = max_yr_elems;
        }
        if (a) |v| {
            if (k.act_rot) |*old| old.free();
            k.act_rot = v;
            k.act_rot_elems = max_act_rot_elems;
        }
    }

    pub fn unload(k: *Kernels) void {
        if (k.act_rot) |*b| b.free();
        if (k.yr) |*b| b.free();
        if (k.xrot) |*b| b.free();
        k.mod_moe.unload();
        k.mod_dense.unload();
        k.* = undefined;
    }

    fn bitIndex(bits: u8) !usize {
        return switch (bits) {
            3 => 0,
            4 => 1,
            5 => 2,
            else => error.Tfq3Bits,
        };
    }

    /// A weight's reduction order is independent of decode window and prompt chunk width.
    pub fn densePlan(k: *const Kernels, w: DenseWeight) !struct { split: bool, cap: usize } {
        const n_bc = w.k >> 7;
        const split = k.yr != null and w.n <= 34816 and n_bc >= 8 and n_bc % 8 == 0;
        const yr_rows = if (k.yr != null) k.yr_elems / (@as(usize, w.n) * 8) else 0;
        const cap = @min(if (split) @min(yr_rows, @as(usize, 64)) else @as(usize, 512), k.xrot_elems / w.k);
        if (cap == 0) return error.Tfq3ScratchNotReserved;
        return .{ .split = split, .cap = cap };
    }

    /// Dense projection: `out[rows, ldo] = x[rows, ldx] @ W^T` (bf16 or fp32 output).
    pub fn dense(k: *const Kernels, stream: Stream, w: DenseWeight, x: u64, ldx: usize, out: u64, ldo: usize, rows: usize, out_f32: bool) !void {
        if (rows == 0 or w.n == 0 or w.k == 0 or w.n % 128 != 0 or w.k % 128 != 0 or ldx < w.k or ldo < w.n) return error.Tfq3Shape;
        const xrot = k.xrot orelse return error.Tfq3ScratchNotReserved;
        const bi = try bitIndex(w.bits);
        const out_elem: usize = if (out_f32) 4 else 2;
        const f32_flag: u32 = @intFromBool(out_f32);
        const n_bc: u32 = w.k >> 7;
        const plan = try k.densePlan(w);
        const use_sk8 = plan.split;
        const cap_rows = plan.cap;

        var done: usize = 0;
        while (done < rows) {
            const take: usize = @min(rows - done, cap_rows);
            const x_off = x + done * ldx * 2;
            const y_off = out + done * ldo * out_elem;
            {
                var a: launch_.Args = .{};
                a.add(x_off);
                a.add(xrot.ptr);
                a.add(w.k);
                a.add(@as(u32, @intCast(ldx)));
                try launch_.launch(k.fwht_x, .{ .grid = .{ .x = n_bc, .y = @intCast(take) }, .block = .{ .x = 32 } }, stream, &a);
            }
            if (use_sk8) {
                const yr = k.yr.?.ptr;
                var a: launch_.Args = .{};
                a.add(w.words);
                a.add(w.scales);
                a.add(w.tables);
                a.add(xrot.ptr);
                a.add(yr);
                a.add(@as(u32, @intCast(take)));
                a.add(w.n);
                a.add(w.k);
                try launch_.launch(k.wmma_sk8[bi], .{ .grid = .{ .x = w.n / 64, .y = 8, .z = @intCast((take + 15) / 16) }, .block = .{ .x = 128 } }, stream, &a);

                var b: launch_.Args = .{};
                b.add(yr);
                b.add(y_off);
                b.add(@as(u32, @intCast(take)));
                b.add(w.n);
                b.add(@as(u32, @intCast(ldo)));
                b.add(w.sr_split);
                b.add(w.row_off);
                b.add(@as(u32, 8));
                b.add(f32_flag);
                try launch_.launch(k.fwht_y, .{ .grid = .{ .x = w.n / 128, .y = @intCast(take) }, .block = .{ .x = 32 } }, stream, &b);
            } else {
                var a: launch_.Args = .{};
                a.add(w.words);
                a.add(w.scales);
                a.add(w.tables);
                a.add(xrot.ptr);
                a.add(y_off);
                a.add(@as(u32, @intCast(take)));
                a.add(w.k);
                a.add(@as(u32, @intCast(ldo)));
                a.add(w.sr_split);
                a.add(w.row_off);
                a.add(f32_flag);
                if (take >= 48) {
                    try launch_.launch(k.wmma_prefill64[bi], .{ .grid = .{ .x = w.n / 128, .y = @intCast((take + 63) / 64) }, .block = .{ .x = 256 } }, stream, &a);
                } else {
                    try launch_.launch(k.wmma_fused[bi], .{ .grid = .{ .x = w.n / 128, .y = @intCast((take + 15) / 16) }, .block = .{ .x = 256 } }, stream, &a);
                }
            }
            done += take;
        }
    }

    /// Tiled Tensor Core prompt prefill (`BM=64, BN=128`, 4x weight reuse, zero split-K `yr` global traffic).
    pub fn densePrompt(k: *const Kernels, stream: Stream, w: DenseWeight, x: u64, ldx: usize, out: u64, ldo: usize, rows: usize, out_f32: bool) !void {
        if (rows == 0 or w.n == 0 or w.k == 0 or w.n % 128 != 0 or w.k % 128 != 0 or ldx < w.k or ldo < w.n) return error.Tfq3Shape;
        const xrot = k.xrot orelse return error.Tfq3ScratchNotReserved;
        const bi = try bitIndex(w.bits);
        const out_elem: usize = if (out_f32) 4 else 2;
        const f32_flag: u32 = @intFromBool(out_f32);
        const n_bc: u32 = w.k >> 7;
        const cap_rows: usize = @min(@as(usize, 512), k.xrot_elems / w.k);
        if (cap_rows == 0) return error.Tfq3ScratchNotReserved;

        var done: usize = 0;
        while (done < rows) {
            const take: usize = @min(rows - done, cap_rows);
            const x_off = x + done * ldx * 2;
            const y_off = out + done * ldo * out_elem;
            {
                var a: launch_.Args = .{};
                a.add(x_off);
                a.add(xrot.ptr);
                a.add(w.k);
                a.add(@as(u32, @intCast(ldx)));
                try launch_.launch(k.fwht_x, .{ .grid = .{ .x = n_bc, .y = @intCast(take) }, .block = .{ .x = 32 } }, stream, &a);
            }
            {
                var a: launch_.Args = .{};
                a.add(w.words);
                a.add(w.scales);
                a.add(w.tables);
                a.add(xrot.ptr);
                a.add(y_off);
                a.add(@as(u32, @intCast(take)));
                a.add(w.k);
                a.add(@as(u32, @intCast(ldo)));
                a.add(w.sr_split);
                a.add(w.row_off);
                a.add(f32_flag);
                if (take > 16) {
                    try launch_.launch(k.wmma_prefill64[bi], .{ .grid = .{ .x = w.n / 128, .y = @intCast((take + 63) / 64) }, .block = .{ .x = 256 } }, stream, &a);
                } else {
                    try launch_.launch(k.wmma_fused[bi], .{ .grid = .{ .x = w.n / 128, .y = @intCast((take + 15) / 16) }, .block = .{ .x = 256 } }, stream, &a);
                }
            }
            done += take;
        }
    }

    /// Routed MoE projection for `rows` tokens × `top_k` picks (supports mixed 3b/4b/5b per expert).
    pub fn moe(
        k: *const Kernels,
        stream: Stream,
        l: MoeLayer,
        x: u64,
        x_stride: usize,
        picks: u64,
        pick_stride: u32,
        ey: u64,
        out_slot_stride: u32,
        rows: usize,
        top_k: u32,
        lim: f32,
        act_mode: u32,
        out_f32: bool,
    ) !void {
        if (rows == 0 or top_k == 0 or l.dims % 128 != 0 or l.inter % 128 != 0) return error.Tfq3Shape;
        const xrot = k.xrot orelse return error.Tfq3ScratchNotReserved;
        const act_rot = k.act_rot orelse return error.Tfq3ScratchNotReserved;
        const cap_x: usize = k.xrot_elems / l.dims;
        const cap_a: usize = k.act_rot_elems / (@as(usize, top_k) * l.inter);
        const cap_rows: usize = @min(cap_x, cap_a);
        if (cap_rows == 0) return error.Tfq3ScratchNotReserved;
        const ey_elem: usize = if (out_f32) 4 else 2;

        var done: usize = 0;
        while (done < rows) {
            const take: usize = @min(rows - done, cap_rows);
            const x_off = x + done * x_stride * 2;
            const p_off = picks + done * @as(usize, pick_stride) * 4;
            const ey_off = ey + done * @as(usize, out_slot_stride) * l.dims * ey_elem;
            {
                var a: launch_.Args = .{};
                a.add(x_off);
                a.add(xrot.ptr);
                a.add(l.dims);
                a.add(@as(u32, @intCast(x_stride)));
                try launch_.launch(k.fwht_x, .{ .grid = .{ .x = l.dims >> 7, .y = @intCast(take) }, .block = .{ .x = 32 } }, stream, &a);
            }
            {
                var a: launch_.Args = .{};
                a.add(l.weights);
                a.add(l.scales);
                a.add(l.metas);
                a.add(l.tables);
                a.add(xrot.ptr);
                a.add(p_off);
                a.add(act_rot.ptr);
                a.add(l.dims);
                a.add(l.inter);
                a.add(l.experts);
                a.add(top_k);
                a.add(pick_stride);
                a.add(lim);
                a.add(act_mode);
                try launch_.launch(k.moe_gu, .{ .grid = .{ .x = l.inter / 128, .y = @intCast(take * top_k) }, .block = .{ .x = 256 } }, stream, &a);
            }
            {
                var a: launch_.Args = .{};
                a.add(l.weights);
                a.add(l.scales);
                a.add(l.metas);
                a.add(l.tables);
                a.add(act_rot.ptr);
                a.add(p_off);
                a.add(ey_off);
                a.add(l.dims);
                a.add(l.inter);
                a.add(l.experts);
                a.add(top_k);
                a.add(pick_stride);
                a.add(out_slot_stride);
                a.add(@as(u32, @intFromBool(out_f32)));
                try launch_.launch(k.moe_down, .{ .grid = .{ .x = l.dims / 128, .y = @intCast(take * top_k) }, .block = .{ .x = 256 } }, stream, &a);
            }
            done += take;
        }
    }

    /// Routed MoE projection using Inkling-2 softcapped SwiGLU (`gate_cap`, `up_cap`).
    pub fn moeSoftcap(
        k: *const Kernels,
        stream: Stream,
        l: MoeLayer,
        x: u64,
        x_stride: usize,
        picks: u64,
        pick_stride: u32,
        ey: u64,
        out_slot_stride: u32,
        rows: usize,
        top_k: u32,
        gate_cap: f32,
        up_cap: f32,
        out_f32: bool,
    ) !void {
        if (rows == 0 or top_k == 0 or l.dims % 128 != 0 or l.inter % 128 != 0) return error.Tfq3Shape;
        const xrot = k.xrot orelse return error.Tfq3ScratchNotReserved;
        const act_rot = k.act_rot orelse return error.Tfq3ScratchNotReserved;
        const cap_x: usize = k.xrot_elems / l.dims;
        const cap_a: usize = k.act_rot_elems / (@as(usize, top_k) * l.inter);
        const cap_rows: usize = @min(cap_x, cap_a);
        if (cap_rows == 0) return error.Tfq3ScratchNotReserved;
        const ey_elem: usize = if (out_f32) 4 else 2;

        var done: usize = 0;
        while (done < rows) {
            const take: usize = @min(rows - done, cap_rows);
            const x_off = x + done * x_stride * 2;
            const p_off = picks + done * @as(usize, pick_stride) * 4;
            const ey_off = ey + done * @as(usize, out_slot_stride) * l.dims * ey_elem;
            {
                var a: launch_.Args = .{};
                a.add(x_off);
                a.add(xrot.ptr);
                a.add(l.dims);
                a.add(@as(u32, @intCast(x_stride)));
                try launch_.launch(k.fwht_x, .{ .grid = .{ .x = l.dims >> 7, .y = @intCast(take) }, .block = .{ .x = 32 } }, stream, &a);
            }
            {
                var a: launch_.Args = .{};
                a.add(l.weights);
                a.add(l.scales);
                a.add(l.metas);
                a.add(l.tables);
                a.add(xrot.ptr);
                a.add(p_off);
                a.add(act_rot.ptr);
                a.add(l.dims);
                a.add(l.inter);
                a.add(l.experts);
                a.add(top_k);
                a.add(pick_stride);
                a.add(gate_cap);
                a.add(up_cap);
                try launch_.launch(k.moe_gu_softcap, .{ .grid = .{ .x = l.inter / 128, .y = @intCast(take * top_k) }, .block = .{ .x = 256 } }, stream, &a);
            }
            {
                var a: launch_.Args = .{};
                a.add(l.weights);
                a.add(l.scales);
                a.add(l.metas);
                a.add(l.tables);
                a.add(act_rot.ptr);
                a.add(p_off);
                a.add(ey_off);
                a.add(l.dims);
                a.add(l.inter);
                a.add(l.experts);
                a.add(top_k);
                a.add(pick_stride);
                a.add(out_slot_stride);
                a.add(@as(u32, @intFromBool(out_f32)));
                try launch_.launch(k.moe_down, .{ .grid = .{ .x = l.dims / 128, .y = @intCast(take * top_k) }, .block = .{ .x = 256 } }, stream, &a);
            }
            done += take;
        }
    }
};

/// Uploads validated `dir/pw_v3_tables.bin` to device memory.
pub fn uploadTables(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8) !DeviceBuffer {
    const path = try std.fs.path.join(gpa, &.{ dir, "pw_v3_tables.bin" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 10));
    defer gpa.free(bytes);
    try fmt.validateTables(bytes);
    return DeviceBuffer.fromHost(d, bytes);
}

/// Uploads one validated `TFPW` MoE layer file to device memory.
pub fn uploadTfpwLayer(
    io: std.Io,
    d: *const Driver,
    path: []const u8,
    tables_ptr: u64,
    expect_experts: ?u32,
    expect_hidden: ?u32,
    expect_inter: ?u32,
    expect_layer: ?u32,
) !OwnedMoeLayer {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const len: usize = std.math.cast(usize, stat.size) orelse return error.InvalidTfq3Shard;
    var map = try std.Io.File.MemoryMap.create(io, f, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
    defer map.destroy(io);
    const mapped = map.memory;
    const parsed = try fmt.validateTfpw(mapped, expect_experts, expect_hidden, expect_inter, expect_layer);
    const h = parsed.header;
    const meta_slice = mapped[@intCast(h.off_meta)..][0 .. @as(usize, h.num_experts) * @sizeOf(fmt.TfpwExpertMeta)];
    const sc_slice = mapped[@intCast(h.off_scales)..][0 .. @as(usize, @intCast(h.scale_halfs)) * 2];
    const w_slice = mapped[@intCast(h.off_weights)..][0 .. @as(usize, @intCast(h.stream_words)) * 4];
    var m_buf = try DeviceBuffer.fromHost(d, meta_slice);
    errdefer m_buf.free();
    var s_buf = try DeviceBuffer.fromHost(d, sc_slice);
    errdefer s_buf.free();
    const w_buf = try DeviceBuffer.fromHost(d, w_slice);
    return .{
        .layer = .{
            .weights = w_buf.ptr,
            .scales = s_buf.ptr,
            .metas = m_buf.ptr,
            .tables = tables_ptr,
            .experts = h.num_experts,
            .dims = h.hidden_size,
            .inter = h.moe_inter,
        },
        .w_buf = w_buf,
        .s_buf = s_buf,
        .m_buf = m_buf,
    };
}

/// Collection of `TFPN` dense shards uploaded to device memory with O(1) tensor lookup by name.
pub const TfpnStore = struct {
    gpa: std.mem.Allocator,
    tables: DeviceBuffer,
    owns_tables: bool = true,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    tensors: std.StringHashMapUnmanaged(DenseWeight) = .empty,
    bytes: usize = 0,

    pub fn loadDir(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, prefix: []const u8) !TfpnStore {
        const paths = try fmt.listShards(gpa, io, dir, prefix);
        defer fmt.freePaths(gpa, paths);
        if (paths.len == 0) return error.InvalidTfq3Shard;

        const tables = try uploadTables(gpa, io, d, dir);
        var store: TfpnStore = .{ .gpa = gpa, .tables = tables, .owns_tables = true, .bytes = tables.len };
        errdefer store.deinit();
        for (paths) |path| try store.addFile(io, d, path);
        return store;
    }

    pub fn loadFile(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, file: []const u8) !TfpnStore {
        const path = try std.fs.path.join(gpa, &.{ dir, file });
        defer gpa.free(path);
        const tables = try uploadTables(gpa, io, d, dir);
        var store: TfpnStore = .{ .gpa = gpa, .tables = tables, .owns_tables = true, .bytes = tables.len };
        errdefer store.deinit();
        try store.addFile(io, d, path);
        return store;
    }

    /// Loads one `TFPN` shard sharing a caller-owned `pw_v3_tables.bin` device buffer.
    pub fn loadFileWithTables(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, file: []const u8, tables_ptr: u64) !TfpnStore {
        const path = try std.fs.path.join(gpa, &.{ dir, file });
        defer gpa.free(path);
        var store: TfpnStore = .{
            .gpa = gpa,
            .tables = .{ .d = d, .ptr = tables_ptr, .len = fmt.tables_file_bytes },
            .owns_tables = false,
            .bytes = 0,
        };
        errdefer store.deinit();
        try store.addFile(io, d, path);
        return store;
    }

    pub fn addFile(store: *TfpnStore, io: std.Io, d: *const Driver, path: []const u8) !void {
        var f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        const stat = try f.stat(io);
        const len: usize = std.math.cast(usize, stat.size) orelse return error.InvalidTfq3Shard;
        var map = try std.Io.File.MemoryMap.create(io, f, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        defer map.destroy(io);
        const mapped = map.memory;
        var bad_name: []const u8 = "";
        const parsed = fmt.validateTfpnNamed(mapped, &bad_name) catch |err| {
            if (err == error.MisalignedTfq3Tensor) std.log.err("TFQ3 tensor {s}: word_off must be a multiple of 4", .{bad_name});
            return err;
        };
        const h = parsed.header;
        const sc_slice = mapped[@intCast(h.off_scales)..][0 .. @as(usize, @intCast(h.total_scale_halfs)) * 2];
        const w_slice = mapped[@intCast(h.off_weights)..][0 .. @as(usize, @intCast(h.total_weight_words)) * 4];
        var incoming: std.StringHashMapUnmanaged(DenseWeight) = .empty;
        defer {
            var it = incoming.keyIterator();
            while (it.next()) |key| store.gpa.free(key.*);
            incoming.deinit(store.gpa);
        }
        for (parsed.metas) |m| {
            const name = fmt.canonicalName(m.nameSlice());
            if (store.tensors.contains(name) or incoming.contains(name)) return error.DuplicateTfq3Tensor;
            const owned = try store.gpa.dupe(u8, name);
            errdefer store.gpa.free(owned);
            try incoming.put(store.gpa, owned, .{
                .words = m.word_off * 4,
                .scales = @as(u64, m.sc_off) * 2,
                .tables = store.tables.ptr,
                .n = m.r,
                .k = m.c,
                .bits = @intCast(m.b_bits),
            });
        }
        try store.buffers.ensureUnusedCapacity(store.gpa, 2);
        try store.tensors.ensureUnusedCapacity(store.gpa, incoming.count());
        const added_bytes = try std.math.add(usize, sc_slice.len, w_slice.len);
        const total_bytes = try std.math.add(usize, store.bytes, added_bytes);
        var s_buf = try DeviceBuffer.fromHost(d, sc_slice);
        errdefer s_buf.free();
        var w_buf = try DeviceBuffer.fromHost(d, w_slice);
        errdefer w_buf.free();
        // No fallible operation follows the ownership transfer.
        store.buffers.appendAssumeCapacity(s_buf);
        store.buffers.appendAssumeCapacity(w_buf);
        var it = incoming.iterator();
        while (it.next()) |entry| {
            var w = entry.value_ptr.*;
            w.words += w_buf.ptr;
            w.scales += s_buf.ptr;
            store.tensors.putAssumeCapacityNoClobber(entry.key_ptr.*, w);
        }
        incoming.clearRetainingCapacity();
        store.bytes = total_bytes;
    }

    pub fn find(store: *const TfpnStore, name: []const u8) ?DenseWeight {
        return store.tensors.get(fmt.canonicalName(name));
    }

    pub fn findChecked(store: *const TfpnStore, name: []const u8, n: usize, k: usize) !DenseWeight {
        const w = store.find(name) orelse return error.MissingTfq3Tensor;
        if (w.n != n or w.k != k) return error.Tfq3Shape;
        return w;
    }

    pub fn deinit(store: *TfpnStore) void {
        var it = store.tensors.keyIterator();
        while (it.next()) |k| store.gpa.free(k.*);
        store.tensors.deinit(store.gpa);
        for (store.buffers.items) |*b| b.free();
        store.buffers.deinit(store.gpa);
        if (store.owns_tables) store.tables.free();
    }
};

test "DenseWeight.sliceRows preserves alignment and computes exact word and scale offsets" {
    const base: DenseWeight = .{ .words = 0x10000, .scales = 0x20000, .tables = 0x30000, .n = 248320, .k = 5120, .bits = 4 };
    const first = try base.sliceRows(0, 98304);
    try std.testing.expectEqual(@as(u32, 98304), first.n);
    try std.testing.expectEqual(@as(u32, 0), first.row_off);
    try std.testing.expectEqual(base.words, first.words);
    const tail = try base.sliceRows(247936, 384);
    try std.testing.expectEqual(@as(u32, 384), tail.n);
    try std.testing.expectEqual(@as(u32, 247936), tail.row_off);
    try std.testing.expectEqual(base.scales + @as(u64, 247936 / 128) * (5120 / 128) * 2, tail.scales);
    try std.testing.expectError(error.Tfq3Shape, base.sliceRows(64, 128));
    try std.testing.expectError(error.Tfq3Shape, base.sliceRows(248320, 128));
}
