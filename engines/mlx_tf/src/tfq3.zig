//! Zero-copy mmap + `newBufferWithBytesNoCopy` loader for TensorFold TFQ3 checkpoints.
//!
//! Validates and maps cross-platform TFQ3 v3 artifacts per FORMAT_SPEC.md:
//!   - `pw_v3_tables.bin` (`PWV3`, 16 KiB universal 2D-AffineCoset/half4 codebook)
//!   - Dense / non-MoE shards `*.tffpw` (`TFPN`, 16 KiB-aligned fp16 scales & u32 packed streams)
//!   - MoE layer shards `layer_XX.tffpw` (`TFPW`, 16 KiB-aligned per-expert mixed 3/4/5-bit streams)
//!   - Contiguous pair fusion (`canFusePair` -> `sr_split = R0`) for `gate+up` and `k+v`.
const std = @import("std");
const mtl = @import("metal4.zig");
const w_mod = @import("weights.zig");

const Buffer = mtl.Buffer;
const BufferRef = mtl.BufferRef;
const Tfq3Matrix = w_mod.Tfq3Matrix;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;

const O_RDONLY: c_int = 0;
const SEEK_END: c_int = 2;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_PRIVATE: c_int = 2;

pub const page_size: usize = 16384;
pub const magic_pwv3: u32 = 0x33565750; // "PWV3"
pub const magic_tfpn: u32 = 0x4E504654; // "TFPN"
pub const magic_tfpw: u32 = 0x57504654; // "TFPW"
pub const format_version: u32 = 3;
pub const tables_filename = "pw_v3_tables.bin";
pub const tables_bytes: usize = 16384;

pub const PwV3TablesFile = extern struct {
    magic: u32,
    version: u32,
    a3: f32,
    b3: f32,
    a4: f32,
    b4: f32,
    a5: f32,
    b5: f32,
    pad0: [8]u32,
    lut3: [256]u32,
    lut4: [1024]u32,
    lut5: [1024]u32,
    c64_3: [64][4]u16,
    c64_4: [64][4]u16,
    c64_5: [64][4]u16,
    pad1: [16384 - 10816]u8,
};

pub const TfpnHeader = extern struct {
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

pub const TfpnTensorMeta = extern struct {
    name: [80]u8,
    R: u32,
    C: u32,
    b_bits: u32,
    sc_off: u32,
    word_off: u64,
    word_count: u64,
    snr_db: f32,
    pad: [3]u32,

    pub fn nameSlice(self: *const TfpnTensorMeta) []const u8 {
        const end = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..end];
    }
};

pub const TfpwHeader = extern struct {
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

pub const TfpwExpertMeta = extern struct {
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

comptime {
    if (@sizeOf(PwV3TablesFile) != 16384) @compileError("PwV3TablesFile size mismatch");
    if (@sizeOf(TfpnHeader) != 64) @compileError("TfpnHeader size mismatch");
    if (@sizeOf(TfpnTensorMeta) != 128) @compileError("TfpnTensorMeta size mismatch");
    if (@sizeOf(TfpwHeader) != 144) @compileError("TfpwHeader size mismatch");
    if (@sizeOf(TfpwExpertMeta) != 64) @compileError("TfpwExpertMeta size mismatch");
}

pub fn align16K(n: usize) usize {
    return (n + page_size - 1) & ~(page_size - 1);
}

/// Normalize module names across `language_model.model.`, `model.language_model.`, `language_model.`, `model.`, `backbone.`.
pub fn canonicalModule(raw: []const u8) []const u8 {
    var s = raw;
    if (std.mem.endsWith(u8, s, ".weight")) s = s[0 .. s.len - ".weight".len];
    const prefixes = [_][]const u8{
        "language_model.model.",
        "model.language_model.",
        "language_model.",
        "model.",
        "backbone.",
    };
    for (prefixes) |p| {
        if (std.mem.startsWith(u8, s, p)) return s[p.len..];
    }
    return s;
}

const MappedFile = struct {
    bytes: []u8,

    fn map(a: std.mem.Allocator, path: []const u8) !?MappedFile {
        const zpath = try a.dupeSentinel(u8, path, 0);
        defer a.free(zpath);
        const fd = open(zpath.ptr, O_RDONLY);
        if (fd < 0) return null;
        const sz_i64 = lseek(fd, 0, SEEK_END);
        if (sz_i64 <= 0) {
            _ = close(fd);
            return error.Tfq3Truncated;
        }
        const len: usize = @intCast(sz_i64);
        const raw = mmap(null, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        _ = close(fd);
        if (raw == null or @intFromPtr(raw) == std.math.maxInt(usize)) return error.Tfq3MapFailed;
        const ptr: [*]u8 = @ptrCast(raw.?);
        return .{ .bytes = ptr[0..len] };
    }

    fn deinit(self: MappedFile) void {
        _ = munmap(@ptrCast(self.bytes.ptr), self.bytes.len);
    }
};

const TfpnShard = struct {
    map: MappedFile,
    scales_buf: Buffer,
    weights_buf: Buffer,
    metas: []TfpnTensorMeta,
};

const Entry = struct {
    shard_idx: usize,
    meta_idx: usize,
};

pub const Found = struct {
    tfq3: Tfq3Matrix,
    shard_idx: usize,
    word_off: u64,
    word_count: u64,
    sc_off: u32,
};

pub fn canFusePair(a: Found, b: Found) bool {
    if (a.shard_idx != b.shard_idx) return false;
    if (a.tfq3.k != b.tfq3.k or a.tfq3.bits != b.tfq3.bits) return false;
    const expected_sc: u32 = a.sc_off + (a.tfq3.n / 128) * (a.tfq3.k / 128);
    if (b.sc_off != expected_sc) return false;
    return b.word_off == a.word_off + a.word_count;
}

pub fn fusePair(a: Found, b: Found) Tfq3Matrix {
    return .{
        .words = a.tfq3.words,
        .scales = a.tfq3.scales,
        .tables = a.tfq3.tables,
        .n = a.tfq3.n + b.tfq3.n,
        .k = a.tfq3.k,
        .bits = a.tfq3.bits,
        .sr_split = a.tfq3.n,
    };
}

fn validateTables(bytes: []const u8) !void {
    if (bytes.len < 10816) return error.Tfq3InvalidTables;
    const tbl: *const PwV3TablesFile = @ptrCast(@alignCast(bytes.ptr));
    if (tbl.magic != magic_pwv3 or tbl.version != format_version) return error.Tfq3InvalidTables;
    const scalars = [_]f32{ tbl.a3, tbl.b3, tbl.a4, tbl.b4, tbl.a5, tbl.b5 };
    for (scalars) |s| if (!std.math.isFinite(s)) return error.Tfq3InvalidTables;
}

fn listTffpwFiles(a: std.mem.Allocator, dir: []const u8) ![][]u8 {
    const zdir = try a.dupeSentinel(u8, dir, 0);
    defer a.free(zdir);
    const dp = std.c.opendir(zdir.ptr) orelse return &.{};
    defer _ = std.c.closedir(dp);

    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |n| a.free(n);
        out.deinit(a);
    }

    while (std.c.readdir(dp)) |ent| {
        const name = ent.name[0..ent.namlen];
        if (std.mem.endsWith(u8, name, ".tffpw")) {
            try out.append(a, try a.dupe(u8, name));
        }
    }
    std.mem.sort([]u8, out.items, {}, struct {
        fn lt(_: void, lhs: []u8, rhs: []u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.lt);
    return out.toOwnedSlice(a);
}

pub const Store = struct {
    a: std.mem.Allocator,
    tables_map: MappedFile,
    tables_buf: Buffer,
    shards: std.ArrayList(TfpnShard) = .empty,
    index: std.StringHashMapUnmanaged(Entry) = .empty,
    total_bytes: usize = 0,

    pub fn open(a: std.mem.Allocator, ctx: *mtl.Context, dir: []const u8) !?Store {
        const tbl_path = try std.fs.path.join(a, &.{ dir, tables_filename });
        defer a.free(tbl_path);
        const tbl_opt = try MappedFile.map(a, tbl_path);
        if (tbl_opt == null) return null;

        const names = try listTffpwFiles(a, dir);
        defer {
            for (names) |n| a.free(n);
            a.free(names);
        }
        if (names.len == 0) {
            tbl_opt.?.deinit();
            return null;
        }

        const tbl_map = tbl_opt.?;
        errdefer tbl_map.deinit();
        try validateTables(tbl_map.bytes);
        if (tbl_map.bytes.len % page_size != 0) return error.Tfq3UnalignedTables;
        const tbl_buf = try ctx.allocBufferNoCopy(tbl_map.bytes.ptr, tbl_map.bytes.len);
        errdefer tbl_buf.deinit();

        var store = Store{
            .a = a,
            .tables_map = tbl_map,
            .tables_buf = tbl_buf,
            .total_bytes = tbl_map.bytes.len,
        };
        errdefer store.deinitShards();

        for (names) |fname| {
            const spath = try std.fs.path.join(a, &.{ dir, fname });
            defer a.free(spath);
            const mf = (try MappedFile.map(a, spath)) orelse return error.Tfq3Truncated;
            errdefer mf.deinit();
            if (mf.bytes.len < @sizeOf(TfpnHeader)) return error.Tfq3Truncated;
            const magic = std.mem.readInt(u32, mf.bytes[0..4], .little);
            if (magic == magic_tfpw) {
                // TFPW MoE shard (handled by MoE loader; do not treat as TFPN error)
                mf.deinit();
                continue;
            }
            if (magic != magic_tfpn) return error.Tfq3BadMagic;

            const hdr: *const TfpnHeader = @ptrCast(@alignCast(mf.bytes.ptr));
            if (hdr.version != format_version) return error.Tfq3BadVersion;
            if (hdr.off_scales % page_size != 0 or hdr.off_weights % page_size != 0) return error.Tfq3Unaligned;
            if (hdr.off_meta < @sizeOf(TfpnHeader)) return error.Tfq3BadBounds;

            const meta_bytes = @as(usize, hdr.num_tensors) * @sizeOf(TfpnTensorMeta);
            const meta_end = @as(usize, @intCast(hdr.off_meta)) + meta_bytes;
            const sc_off_file: usize = @intCast(hdr.off_scales);
            const sc_bytes = @as(usize, @intCast(hdr.total_scale_halfs)) * 2;
            const sc_map_len = align16K(@max(sc_bytes, page_size));
            const wt_off_file: usize = @intCast(hdr.off_weights);
            const wt_bytes = @as(usize, @intCast(hdr.total_weight_words)) * 4;
            const wt_map_len = align16K(@max(wt_bytes, page_size));
            if (meta_end > sc_off_file or sc_off_file + sc_map_len > wt_off_file or wt_off_file + wt_map_len > mf.bytes.len) {
                return error.Tfq3BadBounds;
            }

            const metas = try a.alloc(TfpnTensorMeta, hdr.num_tensors);
            errdefer a.free(metas);
            @memcpy(std.mem.sliceAsBytes(metas), mf.bytes[@intCast(hdr.off_meta)..][0..meta_bytes]);

            for (metas) |*m| {
                if ((m.b_bits != 3 and m.b_bits != 4 and m.b_bits != 5) or
                    m.R == 0 or m.C == 0 or m.R % 128 != 0 or m.C % 128 != 0)
                {
                    return error.Tfq3BadTensorMeta;
                }
                const expected_sc: u64 = @as(u64, m.R / 128) * @as(u64, m.C / 128);
                const expected_wd: u64 = @as(u64, m.R / 16) * @as(u64, m.C / 128) * @as(u64, m.b_bits * 2) * 32;
                if (m.word_count != expected_wd) return error.Tfq3BadTensorMeta;
                if (@as(u64, m.sc_off) + expected_sc > hdr.total_scale_halfs) return error.Tfq3BadBounds;
                if (m.word_off + m.word_count > hdr.total_weight_words) return error.Tfq3BadBounds;
            }

            const sc_buf = try ctx.allocBufferNoCopy(mf.bytes.ptr + sc_off_file, sc_map_len);
            errdefer sc_buf.deinit();
            const wt_buf = try ctx.allocBufferNoCopy(mf.bytes.ptr + wt_off_file, wt_map_len);
            errdefer wt_buf.deinit();

            const si = store.shards.items.len;
            try store.shards.append(a, .{
                .map = mf,
                .scales_buf = sc_buf,
                .weights_buf = wt_buf,
                .metas = metas,
            });
            store.total_bytes += mf.bytes.len;

            for (store.shards.items[si].metas, 0..) |*m, mi| {
                const key = canonicalModule(m.nameSlice());
                const gop = try store.index.getOrPut(a, key);
                if (gop.found_existing) return error.Tfq3DuplicateTensor;
                gop.value_ptr.* = .{ .shard_idx = si, .meta_idx = mi };
            }
        }

        if (store.shards.items.len == 0) {
            store.deinit();
            return null;
        }
        return store;
    }

    pub fn find(self: *const Store, module: []const u8, n: usize, k: usize) ?Found {
        const key = canonicalModule(module);
        const e = self.index.get(key) orelse return null;
        const sh = &self.shards.items[e.shard_idx];
        const m = &sh.metas[e.meta_idx];
        if (m.R != n or m.C != k) return null;
        return .{
            .tfq3 = .{
                .words = sh.weights_buf.view(@intCast(m.word_off * 4)),
                .scales = sh.scales_buf.view(@as(usize, m.sc_off) * 2),
                .tables = self.tables_buf.view(0),
                .n = m.R,
                .k = m.C,
                .bits = @intCast(m.b_bits),
                .sr_split = 0,
            },
            .shard_idx = e.shard_idx,
            .word_off = m.word_off,
            .word_count = m.word_count,
            .sc_off = m.sc_off,
        };
    }

    fn deinitShards(self: *Store) void {
        self.index.deinit(self.a);
        for (self.shards.items) |sh| {
            sh.weights_buf.deinit();
            sh.scales_buf.deinit();
            self.a.free(sh.metas);
            sh.map.deinit();
        }
        self.shards.deinit(self.a);
    }

    pub fn deinit(self: *Store) void {
        self.deinitShards();
        self.tables_buf.deinit();
        self.tables_map.deinit();
    }
};
