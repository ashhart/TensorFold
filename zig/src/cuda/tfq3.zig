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
    wmma_sk8: [3]Function, // 3b, 4b, 5b: split-K=8 BM=16 (rows 1..16, bit-exact)
    wmma_fused: [3]Function, // 3b, 4b, 5b: fused BM=16 (wide N, small K, and prefill)
    moe_gu: Function,
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
            .moe_gu = try mod_moe.function("tf_tfq3_moe_gu_swiglu"),
            .moe_down = try mod_moe.function("tf_tfq3_moe_down"),
        };
        inline for ([_]u8{ 3, 4, 5 }, 0..) |b, i| {
            const sb = std.fmt.comptimePrint("{d}b", .{b});
            k.wmma_sk8[i] = try mod_dense.function("tf_tfq3_wmma_sk8_" ++ sb);
            k.wmma_fused[i] = try mod_dense.function("tf_tfq3_wmma_fused_" ++ sb);
        }
        return k;
    }

    /// Preallocates rotation and split-K scratch before any CUDA graph capture.
    pub fn reserve(k: *Kernels, max_xrot_elems: usize, max_yr_elems: usize, max_act_rot_elems: usize) !void {
        if (max_xrot_elems > k.xrot_elems) {
            if (k.xrot) |*b| b.free();
            k.xrot = try DeviceBuffer.alloc(k.d, max_xrot_elems * 2);
            k.xrot_elems = max_xrot_elems;
        }
        if (max_yr_elems > k.yr_elems) {
            if (k.yr) |*b| b.free();
            k.yr = try DeviceBuffer.alloc(k.d, max_yr_elems * 4);
            k.yr_elems = max_yr_elems;
        }
        if (max_act_rot_elems > k.act_rot_elems) {
            if (k.act_rot) |*b| b.free();
            k.act_rot = try DeviceBuffer.alloc(k.d, max_act_rot_elems * 2);
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

    /// Dense projection: `out[rows, ldo] = x[rows, ldx] @ W^T` (bf16 or fp32 output).
    pub fn dense(k: *const Kernels, stream: Stream, w: DenseWeight, x: u64, ldx: usize, out: u64, ldo: usize, rows: usize, out_f32: bool) !void {
        if (rows == 0 or w.n == 0 or w.k == 0 or w.n % 128 != 0 or w.k % 128 != 0 or ldx < w.k or ldo < w.n) return error.Tfq3Shape;
        const xrot = k.xrot orelse return error.Tfq3ScratchNotReserved;
        const bi = try bitIndex(w.bits);
        const out_elem: usize = if (out_f32) 4 else 2;
        const f32_flag: u32 = @intFromBool(out_f32);
        const n_bc: u32 = w.k >> 7;
        const yr_rows: usize = if (k.yr != null) k.yr_elems / (@as(usize, w.n) * 8) else 0;
        const use_sk8 = rows <= 64 and w.n <= 34816 and n_bc >= 8 and n_bc % 8 == 0 and yr_rows >= @min(rows, 16);
        const cap_rows: usize = @min(if (use_sk8) @min(yr_rows, @as(usize, 64)) else @as(usize, 512), k.xrot_elems / w.k);
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
                try launch_.launch(k.wmma_fused[bi], .{ .grid = .{ .x = w.n / 128, .y = @intCast((take + 15) / 16) }, .block = .{ .x = 256 } }, stream, &a);
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
) !OwnedMoeLayer {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const len: usize = std.math.cast(usize, stat.size) orelse return error.InvalidTfq3Shard;
    const mapped = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, f.handle, 0);
    defer std.posix.munmap(mapped);
    const parsed = try fmt.validateTfpw(mapped, expect_experts, expect_hidden, expect_inter);
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
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    tensors: std.StringHashMapUnmanaged(DenseWeight) = .empty,
    bytes: usize = 0,

    pub fn loadDir(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, prefix: []const u8) !TfpnStore {
        var tables = try uploadTables(gpa, io, d, dir);
        errdefer tables.free();
        const paths = try fmt.listShards(gpa, io, dir, prefix);
        defer fmt.freePaths(gpa, paths);
        if (paths.len == 0) return error.InvalidTfq3Shard;

        var store: TfpnStore = .{ .gpa = gpa, .tables = tables, .bytes = tables.len };
        errdefer store.deinit();
        for (paths) |path| try store.addFile(io, d, path);
        return store;
    }

    pub fn loadFile(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, file: []const u8) !TfpnStore {
        var tables = try uploadTables(gpa, io, d, dir);
        errdefer tables.free();
        const path = try std.fs.path.join(gpa, &.{ dir, file });
        defer gpa.free(path);
        var store: TfpnStore = .{ .gpa = gpa, .tables = tables, .bytes = tables.len };
        errdefer store.deinit();
        try store.addFile(io, d, path);
        return store;
    }

    pub fn addFile(store: *TfpnStore, io: std.Io, d: *const Driver, path: []const u8) !void {
        var f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        const stat = try f.stat(io);
        const len: usize = std.math.cast(usize, stat.size) orelse return error.InvalidTfq3Shard;
        const mapped = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, f.handle, 0);
        defer std.posix.munmap(mapped);
        const parsed = try fmt.validateTfpn(mapped);
        const h = parsed.header;
        const sc_slice = mapped[@intCast(h.off_scales)..][0 .. @as(usize, @intCast(h.total_scale_halfs)) * 2];
        const w_slice = mapped[@intCast(h.off_weights)..][0 .. @as(usize, @intCast(h.total_weight_words)) * 4];
        var s_buf = try DeviceBuffer.fromHost(d, sc_slice);
        errdefer s_buf.free();
        var w_buf = try DeviceBuffer.fromHost(d, w_slice);
        errdefer w_buf.free();
        try store.buffers.append(store.gpa, s_buf);
        try store.buffers.append(store.gpa, w_buf);
        store.bytes += s_buf.len + w_buf.len;
        for (parsed.metas) |m| {
            const name_dup = try store.gpa.dupe(u8, fmt.canonicalName(m.nameSlice()));
            errdefer store.gpa.free(name_dup);
            const gop = try store.tensors.getOrPut(store.gpa, name_dup);
            if (gop.found_existing) {
                store.gpa.free(name_dup);
                return error.DuplicateTfq3Tensor;
            }
            gop.value_ptr.* = .{
                .words = w_buf.ptr + m.word_off * 4,
                .scales = s_buf.ptr + @as(u64, m.sc_off) * 2,
                .tables = store.tables.ptr,
                .n = m.r,
                .k = m.c,
                .bits = @intCast(m.b_bits),
            };
        }
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
        store.tables.free();
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
