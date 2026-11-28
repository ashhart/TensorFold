//! Shared CUDA TFQ4 (`PWV4` v4 + `TFP4` v4, 2.001/2.251 bpw 8D E8P) dense runtime and table/shard loaders.
const std = @import("std");
const fmt = @import("tfq3_format.zig");
const tfq4 = fmt.tfq4;
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

/// Uploaded `pw_v4_e8p_tables.bin` device buffer plus validated codebook metadata.
pub const UploadedTables = struct {
    buf: DeviceBuffer,
    table_kind: u32,
    k_codes: u32,

    /// Device pointer to `e8p_lut[0]` at byte offset 64 (64-byte aligned for `uint4` loads).
    pub fn lutPtr(self: UploadedTables) u64 {
        return self.buf.ptr + tfq4.lut_offset;
    }

    pub fn deinit(self: *UploadedTables) void {
        self.buf.free();
        self.* = undefined;
    }

    pub fn free(self: *UploadedTables) void {
        self.deinit();
    }
};

/// One dense TFQ4 matrix `[n, k]` on the device (`b_bits == 2` or `225`, `table_kind in 0..3`).
pub const DenseWeight = struct {
    words: u64,
    scales: u64,
    lut: u64,
    table_kind: u32,
    n: u32,
    k: u32,
    bits: u16,
    sr_split: u32 = 0,
    row_off: u32 = 0,
    k_sign_off: u32 = 0,

    /// Output rows `[from, from + count)` as a 128-row-aligned view keeping exact row sign seeds.
    pub fn sliceRows(w: DenseWeight, from: u32, count: u32) !DenseWeight {
        if (from % 128 != 0 or count % 128 != 0 or count == 0 or @as(u64, from) + count > w.n) return error.Tfq4Shape;
        const w_words: u64 = if (from == 0) 0 else try tfq4.tfq4MatWords(from, w.k, w.bits);
        const s_halfs: u64 = tfq4.matScales(from, w.k);
        return .{
            .words = w.words + w_words * 4,
            .scales = w.scales + s_halfs * 2,
            .lut = w.lut,
            .table_kind = w.table_kind,
            .n = count,
            .k = w.k,
            .bits = w.bits,
            .sr_split = w.sr_split,
            .row_off = w.row_off + from,
            .k_sign_off = w.k_sign_off,
        };
    }
};

/// Compiled TFQ4 dense + hybrid MoE CUDA module and preallocated rotation scratch.
pub const Kernels = struct {
    d: *const Driver,
    mod: Module,
    fwht_x: Function,
    dense_trained: Function,
    dense_e8p: Function,
    dense_e8p1024: Function,
    moe_gu: Function,
    moe_gu_softcap: Function,
    moe_down: Function,
    xrot: ?DeviceBuffer = null,
    xrot_elems: usize = 0,

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.tfq4);
        errdefer mod.unload();
        return .{
            .d = d,
            .mod = mod,
            .fwht_x = try mod.function("tf_tfq4_fwht_x"),
            .dense_trained = try mod.function("tf_tfq4_e8p_dense_trained"),
            .dense_e8p = try mod.function("tf_tfq4_e8p_dense_e8p"),
            .dense_e8p1024 = try mod.function("tf_tfq4_e8p_dense_e8p1024"),
            .moe_gu = try mod.function("tf_tfq4_moe_gu_swiglu"),
            .moe_gu_softcap = try mod.function("tf_tfq4_moe_gu_softcap"),
            .moe_down = try mod.function("tf_tfq4_moe_down"),
        };
    }

    pub fn reserve(k: *Kernels, max_xrot_elems: usize) !void {
        if (max_xrot_elems <= k.xrot_elems) return;
        const buf = try DeviceBuffer.alloc(k.d, try std.math.mul(usize, max_xrot_elems, 2));
        if (k.xrot) |*old| old.free();
        k.xrot = buf;
        k.xrot_elems = max_xrot_elems;
    }

    pub fn unload(k: *Kernels) void {
        if (k.xrot) |*b| b.free();
        k.mod.unload();
        k.* = undefined;
    }

    /// Dense projection: `out[rows, ldo] = x[rows, ldx] @ W^T` (bf16 or fp32 output, bit-exact across widths).
    pub fn dense(k: *const Kernels, stream: Stream, w: DenseWeight, x: u64, ldx: usize, out: u64, ldo: usize, rows: usize, out_f32: bool) !void {
        if (rows == 0 or w.n == 0 or w.k == 0 or w.n % 128 != 0 or w.k % 128 != 0 or ldx < w.k or ldo < w.n) return error.Tfq4Shape;
        if (!tfq4.tableKindSupportsBits(w.table_kind, w.bits)) return error.InvalidTfq4Tables;
        const xrot = k.xrot orelse return error.Tfq4ScratchNotReserved;
        const cap_rows: usize = @min(@as(usize, 512), k.xrot_elems / w.k);
        if (cap_rows == 0) return error.Tfq4ScratchNotReserved;
        const out_elem: usize = if (out_f32) 4 else 2;
        const f32_flag: u32 = @intFromBool(out_f32);
        const n_bc: u32 = w.k >> 7;
        const kernel = if (w.bits == 225)
            k.dense_e8p1024
        else if (w.table_kind >= tfq4.table_kind_true_e8p)
            k.dense_e8p
        else
            k.dense_trained;

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
                a.add(w.k_sign_off);
                try launch_.launch(k.fwht_x, .{ .grid = .{ .x = n_bc, .y = @intCast(take) }, .block = .{ .x = 32 } }, stream, &a);
            }
            {
                var a: launch_.Args = .{};
                a.add(w.words);
                a.add(w.scales);
                a.add(w.lut);
                a.add(xrot.ptr);
                a.add(y_off);
                a.add(@as(u32, @intCast(take)));
                a.add(w.n);
                a.add(w.k);
                a.add(@as(u32, @intCast(ldo)));
                a.add(w.sr_split);
                a.add(w.row_off);
                a.add(f32_flag);
                try launch_.launch(kernel, .{ .grid = .{ .x = w.n / 128, .y = @intCast((take + 15) / 16) }, .block = .{ .x = 256 } }, stream, &a);
            }
            done += take;
        }
    }
};

/// Uploads validated `dir/pw_v4_e8p_tables.bin` (`table_kind = 0/1/2/3`, 16 KiB or 32 KiB) to device memory.
pub fn uploadTables(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8) !UploadedTables {
    const path = try std.fs.path.join(gpa, &.{ dir, "pw_v4_e8p_tables.bin" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 10));
    defer gpa.free(bytes);
    const info = try tfq4.validatePwV4Tables(bytes);
    const buf = try DeviceBuffer.fromHost(d, bytes);
    return .{ .buf = buf, .table_kind = info.table_kind, .k_codes = info.k_codes };
}

/// Uploads `dir/pw_v4_e8p_tables.bin` when present, or returns `null` if the file does not exist.
pub fn uploadTablesOptional(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8) !?UploadedTables {
    if (!fmt.hasTfq4Tables(gpa, io, dir)) return null;
    return try uploadTables(gpa, io, d, dir);
}

/// Collection of `TFP4` v4 dense shards uploaded to device memory with O(1) tensor lookup by name.
pub const Tfp4Store = struct {
    gpa: std.mem.Allocator,
    tables: UploadedTables,
    owns_tables: bool = true,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    tensors: std.StringHashMapUnmanaged(DenseWeight) = .empty,
    bytes: usize = 0,

    pub fn loadDir(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, prefix: []const u8) !Tfp4Store {
        const paths = try fmt.listTfp4Shards(gpa, io, dir, prefix);
        defer fmt.freePaths(gpa, paths);
        if (paths.len == 0) return error.InvalidTfq4Shard;
        const tables = try uploadTables(gpa, io, d, dir);
        var store: Tfp4Store = .{ .gpa = gpa, .tables = tables, .owns_tables = true, .bytes = tables.buf.len };
        errdefer store.deinit();
        for (paths) |path| try store.addFile(io, d, path);
        return store;
    }

    pub fn loadFile(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, file: []const u8) !Tfp4Store {
        const path = try std.fs.path.join(gpa, &.{ dir, file });
        defer gpa.free(path);
        const tables = try uploadTables(gpa, io, d, dir);
        var store: Tfp4Store = .{ .gpa = gpa, .tables = tables, .owns_tables = true, .bytes = tables.buf.len };
        errdefer store.deinit();
        try store.addFile(io, d, path);
        return store;
    }

    pub fn addFile(store: *Tfp4Store, io: std.Io, d: *const Driver, path: []const u8) !void {
        var f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        const stat = try f.stat(io);
        const len: usize = std.math.cast(usize, stat.size) orelse return error.InvalidTfq4Shard;
        var map = try std.Io.File.MemoryMap.create(io, f, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        defer map.destroy(io);
        const mapped = map.memory;
        const parsed = try tfq4.validateTfp4Shard(mapped);
        if (parsed.needs_e8p1024 and store.tables.table_kind != tfq4.table_kind_true_e8p1024) return error.InvalidTfq4Tables;
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
            if (!tfq4.tableKindSupportsBits(store.tables.table_kind, m.b_bits)) return error.InvalidTfq4Tables;
            const name = fmt.canonicalName(m.nameSlice());
            if (store.tensors.contains(name) or incoming.contains(name)) return error.DuplicateTfq4Tensor;
            const owned = try store.gpa.dupe(u8, name);
            errdefer store.gpa.free(owned);
            try incoming.put(store.gpa, owned, .{
                .words = m.word_off * 4,
                .scales = @as(u64, m.sc_off) * 2,
                .lut = store.tables.lutPtr(),
                .table_kind = store.tables.table_kind,
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

    pub fn find(store: *const Tfp4Store, name: []const u8) ?DenseWeight {
        return store.tensors.get(fmt.canonicalName(name));
    }

    pub fn findChecked(store: *const Tfp4Store, name: []const u8, n: usize, k: usize) !DenseWeight {
        const w = store.find(name) orelse return error.MissingTfq4Tensor;
        if (w.n != n or w.k != k) return error.Tfq4Shape;
        return w;
    }

    pub fn deinit(store: *Tfp4Store) void {
        var it = store.tensors.keyIterator();
        while (it.next()) |k| store.gpa.free(k.*);
        store.tensors.deinit(store.gpa);
        for (store.buffers.items) |*b| b.free();
        store.buffers.deinit(store.gpa);
        if (store.owns_tables) store.tables.deinit();
    }
};

test "CUDA TFQ3: TFQ4 DenseWeight.sliceRows computes exact 2b and 225b row offsets" {
    const t = std.testing;
    const base2: DenseWeight = .{ .words = 0x10000, .scales = 0x20000, .lut = 0x30040, .table_kind = 2, .n = 512, .k = 256, .bits = 2 };
    const s0 = try base2.sliceRows(0, 128);
    try t.expectEqual(base2.words, s0.words);
    const s2 = try base2.sliceRows(128, 256);
    try t.expectEqual(@as(u64, 0x10000 + 128 * 16 * 4), s2.words);
    try t.expectEqual(@as(u64, 0x20000 + 2 * 2), s2.scales);
    try t.expectEqual(@as(u32, 128), s2.row_off);

    const base225: DenseWeight = .{ .words = 0x10000, .scales = 0x20000, .lut = 0x30040, .table_kind = 3, .n = 512, .k = 256, .bits = 225 };
    const s225 = try base225.sliceRows(256, 128);
    try t.expectEqual(@as(u64, 0x10000 + 256 * 2 * 9 * 4), s225.words);
    try t.expectEqual(@as(u64, 0x20000 + 2 * 2 * 2), s225.scales);
    try t.expectError(error.Tfq4Shape, base225.sliceRows(64, 128));
}
