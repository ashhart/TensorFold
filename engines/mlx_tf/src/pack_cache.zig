//! Zero-copy page-aligned `.mlxtf` pre-packed tile cache for Apple Silicon unified memory.
//!
//! Eliminates load-time safetensors repacking by persisting pre-packed 1152-byte `qk` Metal buffers
//! at 16 KiB page boundaries and mapping them directly via `newBufferWithBytesNoCopy` (zero memcpy).
//! Cached transparently in `~/.cache/tensorfold/pack_cache/<hash>.mlxtf` (override with
//! `MLX_TF_CACHE_DIR` or disable with `MLX_TF_NO_CACHE=1`).
const std = @import("std");
const mtl = @import("metal4.zig");
const cfg_mod = @import("config.zig");
const w_mod = @import("weights.zig");
const st = @import("safetensors.zig");

const ModelConfig = cfg_mod.ModelConfig;
const ModelWeights = w_mod.ModelWeights;
const LayerWeights = w_mod.LayerWeights;
const QkMatrix = w_mod.QkMatrix;
const LinearWeight = w_mod.LinearWeight;
const BufferRef = mtl.BufferRef;

extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern "c" fn getpid() c_int;
extern "c" fn mkdir(path: [*:0]const u8, mode: u16) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn rename(old_path: [*:0]const u8, new_path: [*:0]const u8) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;

const O_RDONLY: c_int = 0;
const O_WRONLY: c_int = 0x0001;
const O_CREAT: c_int = 0x0200;
const O_TRUNC: c_int = 0x0400;
const SEEK_END: c_int = 2;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_PRIVATE: c_int = 2;

pub const page_size: usize = 16384; // Apple Silicon arm64 native VM page size
const cache_magic: [8]u8 = "MLXTF001".*;
const cache_version: u32 = 1;

const RefDesc = extern struct {
    buf_idx: u32 = 0,
    _pad: u32 = 0,
    off: u64 = 0,
};

const QkDesc = extern struct {
    ref: RefDesc = .{},
    n: u32 = 0,
    k: u32 = 0,
};

const BufDesc = extern struct {
    file_off: u64 = 0,
    byte_len: u64 = 0,
};

const LayerDesc = extern struct {
    kind: u32 = 0,
    has_qkv_bias: u32 = 0,
    input_norm: RefDesc = .{},
    post_attn_norm: RefDesc = .{},
    gate_up: QkDesc = .{},
    down_proj: QkDesc = .{},
    // linear (GDN)
    lin_in_proj: QkDesc = .{},
    lin_out_proj: QkDesc = .{},
    lin_conv_w: RefDesc = .{},
    lin_a_log: RefDesc = .{},
    lin_dt_bias: RefDesc = .{},
    lin_gdn_norm: RefDesc = .{},
    // full (GQA)
    full_qkv: QkDesc = .{},
    full_o: QkDesc = .{},
    full_q_norm: RefDesc = .{},
    full_k_norm: RefDesc = .{},
    full_qkv_bias: RefDesc = .{},
    // mamba (Mamba-2)
    mamba_in_proj: QkDesc = .{},
    mamba_out_proj: QkDesc = .{},
    mamba_conv_w: RefDesc = .{},
    mamba_conv_b: RefDesc = .{},
    mamba_a_log: RefDesc = .{},
    mamba_dt_bias: RefDesc = .{},
    mamba_d_skip: RefDesc = .{},
    mamba_norm: RefDesc = .{},
    // moe (Sparse MoE)
    moe_gate_w: RefDesc = .{},
    moe_gate_b: RefDesc = .{},
    moe_shared_up: QkDesc = .{},
    moe_shared_down: QkDesc = .{},
    moe_fc1_experts: RefDesc = .{},
    moe_fc2_experts: RefDesc = .{},
    moe_fc1_stride: u64 = 0,
    moe_fc2_stride: u64 = 0,
};

const FileHeader = extern struct {
    magic: [8]u8 = cache_magic,
    version: u32 = cache_version,
    arch: u32 = 0,
    num_buffers: u32 = 0,
    num_layers: u32 = 0,
    config_hash: u64 = 0,
    total_bytes: u64 = 0,
    embed_codes: RefDesc = .{},
    embed_scales: RefDesc = .{},
    embed_biases: RefDesc = .{},
    final_norm: RefDesc = .{},
    lm_head: QkDesc = .{},
};

fn align16K(n: usize) usize {
    return (n + page_size - 1) & ~(page_size - 1);
}

fn fnvUpdate(h: *u64, bytes: []const u8) void {
    var x = h.*;
    for (bytes) |b| {
        x ^= @as(u64, b);
        x *%= 0x100000001b3;
    }
    h.* = x;
}

pub fn isDisabled() bool {
    const v = getenv("MLX_TF_NO_CACHE") orelse return false;
    const s = std.mem.span(v);
    return std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true");
}

pub fn computeCacheKey(a: std.mem.Allocator, model_dir: []const u8, c: ModelConfig) u64 {
    var h: u64 = 0xcbf29ce484222325;
    fnvUpdate(&h, &cache_magic);
    fnvUpdate(&h, model_dir);
    fnvUpdate(&h, std.mem.asBytes(&c.num_layers));
    fnvUpdate(&h, std.mem.asBytes(&c.hidden_size));
    fnvUpdate(&h, std.mem.asBytes(&c.vocab_size));
    const arch_u32: u32 = @intFromEnum(c.arch);
    fnvUpdate(&h, std.mem.asBytes(&arch_u32));

    // Include config.json bytes + index/single safetensors file size for automatic invalidation
    if (std.fs.path.join(a, &.{ model_dir, "config.json" })) |p| {
        defer a.free(p);
        if (st.mapFile(a, p)) |raw| {
            defer st.unmapFile(raw);
            fnvUpdate(&h, raw);
        } else |_| {}
    } else |_| {}

    const probes = [_][]const u8{ "model.safetensors.index.json", "model.safetensors", "model-00001-of-00004.safetensors" };
    for (probes) |name| {
        if (std.fs.path.join(a, &.{ model_dir, name })) |p| {
            defer a.free(p);
            if (a.dupeSentinel(u8, p, 0)) |zp| {
                defer a.free(zp);
                const fd = open(zp.ptr, O_RDONLY);
                if (fd >= 0) {
                    const sz = lseek(fd, 0, SEEK_END);
                    _ = close(fd);
                    fnvUpdate(&h, name);
                    fnvUpdate(&h, std.mem.asBytes(&sz));
                }
            } else |_| {}
        } else |_| {}
    }
    return h;
}

fn cacheFilePath(a: std.mem.Allocator, hash: u64) ![]u8 {
    var dir_owned: ?[]u8 = null;
    defer if (dir_owned) |d| a.free(d);
    const cache_dir: []const u8 = if (getenv("MLX_TF_CACHE_DIR")) |ev|
        std.mem.span(ev)
    else blk: {
        const home = getenv("HOME") orelse return error.NoHomeDir;
        dir_owned = try std.fmt.allocPrint(a, "{s}/.cache/tensorfold/pack_cache", .{std.mem.span(home)});
        break :blk dir_owned.?;
    };
    return std.fmt.allocPrint(a, "{s}/{x:0>16}.mlxtf", .{ cache_dir, hash });
}

fn makeDirP(a: std.mem.Allocator, path: []const u8) !void {
    var cur: std.ArrayList(u8) = .empty;
    defer cur.deinit(a);
    for (path, 0..) |ch, i| {
        try cur.append(a, ch);
        if (ch == '/' and i > 0) {
            try cur.append(a, 0);
            _ = mkdir(@ptrCast(cur.items.ptr), 0o755);
            cur.items.len -= 1;
        }
    }
    try cur.append(a, 0);
    _ = mkdir(@ptrCast(cur.items.ptr), 0o755);
}

fn writeAllFd(fd: c_int, bytes: []const u8) !void {
    var off: usize = 0;
    while (off < bytes.len) {
        const chunk = @min(bytes.len - off, 64 * 1024 * 1024);
        const n = write(fd, bytes.ptr + off, chunk);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
}

fn refToDesc(bufs: []const mtl.Buffer, ref: BufferRef) RefDesc {
    for (bufs, 0..) |b, i| {
        if (b.id == ref.buf_id) {
            return .{ .buf_idx = @intCast(i), .off = @intCast(ref.offset) };
        }
    }
    return .{};
}

fn qkToDesc(bufs: []const mtl.Buffer, lw: LinearWeight) QkDesc {
    const q = lw.qk4;
    return .{ .ref = refToDesc(bufs, q.ref), .n = q.n, .k = q.k };
}

fn descToRef(bufs: []const mtl.Buffer, d: RefDesc) BufferRef {
    return bufs[d.buf_idx].view(@intCast(d.off));
}

fn descToQk(bufs: []const mtl.Buffer, d: QkDesc) LinearWeight {
    return .{ .qk4 = .{ .ref = descToRef(bufs, d.ref), .n = d.n, .k = d.k } };
}

pub fn unmap(bytes: []u8) void {
    _ = munmap(@ptrCast(bytes.ptr), bytes.len);
}

/// Attempt instant zero-copy mmap load from the `.mlxtf` pre-packed tile cache.
/// Returns `null` on cache miss or mismatch.
pub fn tryLoad(a: std.mem.Allocator, ctx: *mtl.Context, c: ModelConfig, model_dir: []const u8) ?ModelWeights {
    if (isDisabled()) return null;
    const hash = computeCacheKey(a, model_dir, c);
    const path = cacheFilePath(a, hash) catch return null;
    defer a.free(path);
    const zpath = a.dupeSentinel(u8, path, 0) catch return null;
    defer a.free(zpath);

    const fd = open(zpath.ptr, O_RDONLY);
    if (fd < 0) return null;
    const sz_i64 = lseek(fd, 0, SEEK_END);
    if (sz_i64 < @as(i64, @intCast(page_size))) {
        _ = close(fd);
        return null;
    }
    const file_size: usize = @intCast(sz_i64);
    const raw_ptr = mmap(null, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    _ = close(fd);
    if (raw_ptr == null or @intFromPtr(raw_ptr) == std.math.maxInt(usize)) return null;

    const base: [*]u8 = @ptrCast(raw_ptr.?);
    const mmap_slice = base[0..file_size];
    errdefer unmap(mmap_slice);

    const hdr: *const FileHeader = @ptrCast(@alignCast(base));
    if (!std.mem.eql(u8, &hdr.magic, &cache_magic) or
        hdr.version != cache_version or
        hdr.config_hash != hash or
        hdr.arch != @intFromEnum(c.arch) or
        hdr.num_layers != c.num_layers or
        hdr.num_buffers == 0)
    {
        unmap(mmap_slice);
        return null;
    }

    const buf_descs_off = @sizeOf(FileHeader);
    const buf_descs_bytes = @as(usize, hdr.num_buffers) * @sizeOf(BufDesc);
    const layer_descs_off = buf_descs_off + buf_descs_bytes;
    const layer_descs_bytes = @as(usize, hdr.num_layers) * @sizeOf(LayerDesc);
    if (layer_descs_off + layer_descs_bytes > file_size) {
        unmap(mmap_slice);
        return null;
    }

    const buf_descs: [*]const BufDesc = @ptrCast(@alignCast(base + buf_descs_off));
    const layer_descs: [*]const LayerDesc = @ptrCast(@alignCast(base + layer_descs_off));

    var w = ModelWeights{
        .a = a,
        .embed = undefined,
        .final_norm = undefined,
        .lm_head = undefined,
        .layers = a.alloc(LayerWeights, c.num_layers) catch {
            unmap(mmap_slice);
            return null;
        },
        .total_bytes = @intCast(hdr.total_bytes),
        .mmap_backing = mmap_slice,
    };

    for (0..hdr.num_buffers) |bi| {
        const bd = buf_descs[bi];
        const foff: usize = @intCast(bd.file_off);
        const blen: usize = @intCast(bd.byte_len);
        if (foff % page_size != 0 or blen % page_size != 0 or foff + blen > file_size) {
            w.deinit();
            return null;
        }
        const mbuf = ctx.allocBufferNoCopy(base + foff, blen) catch {
            w.deinit();
            return null;
        };
        w.buffers.append(a, mbuf) catch {
            mbuf.deinit();
            w.deinit();
            return null;
        };
    }

    const bufs = w.buffers.items;
    w.embed = .{
        .codes = descToRef(bufs, hdr.embed_codes),
        .scales = descToRef(bufs, hdr.embed_scales),
        .biases = descToRef(bufs, hdr.embed_biases),
    };
    w.final_norm = descToRef(bufs, hdr.final_norm);
    w.lm_head = descToQk(bufs, hdr.lm_head);

    for (0..c.num_layers) |li| {
        const ld = layer_descs[li];
        const kind: cfg_mod.LayerKind = @enumFromInt(ld.kind);
        var lw = LayerWeights{
            .kind = kind,
            .input_norm = descToRef(bufs, ld.input_norm),
            .post_attn_norm = descToRef(bufs, ld.post_attn_norm),
            .gate_up = if (c.arch != .nemotron_h) descToQk(bufs, ld.gate_up) else undefined,
            .down_proj = if (c.arch != .nemotron_h) descToQk(bufs, ld.down_proj) else undefined,
        };
        switch (kind) {
            .linear => {
                lw.linear = .{
                    .in_proj_all = descToQk(bufs, ld.lin_in_proj),
                    .out_proj = descToQk(bufs, ld.lin_out_proj),
                    .conv_weight = descToRef(bufs, ld.lin_conv_w),
                    .a_log = descToRef(bufs, ld.lin_a_log),
                    .dt_bias = descToRef(bufs, ld.lin_dt_bias),
                    .gdn_norm = descToRef(bufs, ld.lin_gdn_norm),
                };
            },
            .full => {
                lw.full = .{
                    .qkv_proj = descToQk(bufs, ld.full_qkv),
                    .o_proj = descToQk(bufs, ld.full_o),
                    .q_norm = descToRef(bufs, ld.full_q_norm),
                    .k_norm = descToRef(bufs, ld.full_k_norm),
                    .qkv_bias = if (ld.has_qkv_bias != 0) descToRef(bufs, ld.full_qkv_bias) else null,
                };
            },
            .mamba => {
                lw.mamba = .{
                    .in_proj = descToQk(bufs, ld.mamba_in_proj),
                    .out_proj = descToQk(bufs, ld.mamba_out_proj),
                    .conv_weight = descToRef(bufs, ld.mamba_conv_w),
                    .conv_bias = descToRef(bufs, ld.mamba_conv_b),
                    .a_log = descToRef(bufs, ld.mamba_a_log),
                    .dt_bias = descToRef(bufs, ld.mamba_dt_bias),
                    .d_skip = descToRef(bufs, ld.mamba_d_skip),
                    .mamba_norm = descToRef(bufs, ld.mamba_norm),
                };
            },
            .moe => {
                lw.moe = .{
                    .gate_weight = descToRef(bufs, ld.moe_gate_w),
                    .gate_bias = descToRef(bufs, ld.moe_gate_b),
                    .shared_up = descToQk(bufs, ld.moe_shared_up),
                    .shared_down = descToQk(bufs, ld.moe_shared_down),
                    .fc1_experts = descToRef(bufs, ld.moe_fc1_experts),
                    .fc2_experts = descToRef(bufs, ld.moe_fc2_experts),
                    .fc1_expert_stride = @intCast(ld.moe_fc1_stride),
                    .fc2_expert_stride = @intCast(ld.moe_fc2_stride),
                };
            },
            .short_conv => unreachable,
        }
        w.layers[li] = lw;
    }

    return w;
}

/// Atomically persist packed Metal weight buffers into a 16 KiB page-aligned `.mlxtf` file.
pub fn save(a: std.mem.Allocator, c: ModelConfig, model_dir: []const u8, w: *const ModelWeights) !void {
    if (isDisabled()) return;
    const hash = computeCacheKey(a, model_dir, c);
    const dst_path = try cacheFilePath(a, hash);
    defer a.free(dst_path);

    if (std.fs.path.dirname(dst_path)) |parent| {
        try makeDirP(a, parent);
    }

    const tmp_path = try std.fmt.allocPrint(a, "{s}.tmp.{d}", .{ dst_path, getpid() });
    defer a.free(tmp_path);
    const ztmp = try a.dupeSentinel(u8, tmp_path, 0);
    defer a.free(ztmp);
    const zdst = try a.dupeSentinel(u8, dst_path, 0);
    defer a.free(zdst);

    const bufs = w.buffers.items;
    const n_bufs = bufs.len;
    const raw_hdr_bytes = @sizeOf(FileHeader) + n_bufs * @sizeOf(BufDesc) + c.num_layers * @sizeOf(LayerDesc);
    const hdr_aligned = align16K(raw_hdr_bytes);

    const hdr_buf = try a.alloc(u8, hdr_aligned);
    defer a.free(hdr_buf);
    @memset(hdr_buf, 0);

    var total_aligned: usize = 0;
    const buf_descs: [*]BufDesc = @ptrCast(@alignCast(hdr_buf.ptr + @sizeOf(FileHeader)));
    var cur_off: usize = hdr_aligned;
    for (bufs, 0..) |b, i| {
        const blen = align16K(b.len);
        buf_descs[i] = .{ .file_off = @intCast(cur_off), .byte_len = @intCast(blen) };
        cur_off += blen;
        total_aligned += blen;
    }

    const hdr: *FileHeader = @ptrCast(@alignCast(hdr_buf.ptr));
    hdr.* = .{
        .arch = @intFromEnum(c.arch),
        .num_buffers = @intCast(n_bufs),
        .num_layers = @intCast(c.num_layers),
        .config_hash = hash,
        .total_bytes = @intCast(total_aligned),
        .embed_codes = refToDesc(bufs, w.embed.codes),
        .embed_scales = refToDesc(bufs, w.embed.scales),
        .embed_biases = refToDesc(bufs, w.embed.biases),
        .final_norm = refToDesc(bufs, w.final_norm),
        .lm_head = qkToDesc(bufs, w.lm_head),
    };

    const layer_descs: [*]LayerDesc = @ptrCast(@alignCast(hdr_buf.ptr + @sizeOf(FileHeader) + n_bufs * @sizeOf(BufDesc)));
    for (0..c.num_layers) |li| {
        const lw = w.layers[li];
        var ld = LayerDesc{
            .kind = @intFromEnum(lw.kind),
            .input_norm = refToDesc(bufs, lw.input_norm),
            .post_attn_norm = refToDesc(bufs, lw.post_attn_norm),
        };
        if (c.arch != .nemotron_h) {
            ld.gate_up = qkToDesc(bufs, lw.gate_up);
            ld.down_proj = qkToDesc(bufs, lw.down_proj);
        }
        switch (lw.kind) {
            .linear => {
                ld.lin_in_proj = qkToDesc(bufs, lw.linear.in_proj_all);
                ld.lin_out_proj = qkToDesc(bufs, lw.linear.out_proj);
                ld.lin_conv_w = refToDesc(bufs, lw.linear.conv_weight);
                ld.lin_a_log = refToDesc(bufs, lw.linear.a_log);
                ld.lin_dt_bias = refToDesc(bufs, lw.linear.dt_bias);
                ld.lin_gdn_norm = refToDesc(bufs, lw.linear.gdn_norm);
            },
            .full => {
                ld.full_qkv = qkToDesc(bufs, lw.full.qkv_proj);
                ld.full_o = qkToDesc(bufs, lw.full.o_proj);
                ld.full_q_norm = refToDesc(bufs, lw.full.q_norm);
                ld.full_k_norm = refToDesc(bufs, lw.full.k_norm);
                if (lw.full.qkv_bias) |qb| {
                    ld.has_qkv_bias = 1;
                    ld.full_qkv_bias = refToDesc(bufs, qb);
                }
            },
            .mamba => {
                ld.mamba_in_proj = qkToDesc(bufs, lw.mamba.in_proj);
                ld.mamba_out_proj = qkToDesc(bufs, lw.mamba.out_proj);
                ld.mamba_conv_w = refToDesc(bufs, lw.mamba.conv_weight);
                ld.mamba_conv_b = refToDesc(bufs, lw.mamba.conv_bias);
                ld.mamba_a_log = refToDesc(bufs, lw.mamba.a_log);
                ld.mamba_dt_bias = refToDesc(bufs, lw.mamba.dt_bias);
                ld.mamba_d_skip = refToDesc(bufs, lw.mamba.d_skip);
                ld.mamba_norm = refToDesc(bufs, lw.mamba.mamba_norm);
            },
            .moe => {
                ld.moe_gate_w = refToDesc(bufs, lw.moe.gate_weight);
                ld.moe_gate_b = refToDesc(bufs, lw.moe.gate_bias);
                ld.moe_shared_up = qkToDesc(bufs, lw.moe.shared_up);
                ld.moe_shared_down = qkToDesc(bufs, lw.moe.shared_down);
                ld.moe_fc1_experts = refToDesc(bufs, lw.moe.fc1_experts);
                ld.moe_fc2_experts = refToDesc(bufs, lw.moe.fc2_experts);
                ld.moe_fc1_stride = @intCast(lw.moe.fc1_expert_stride);
                ld.moe_fc2_stride = @intCast(lw.moe.fc2_expert_stride);
            },
            .short_conv => unreachable,
        }
        layer_descs[li] = ld;
    }

    const fd = open(ztmp.ptr, O_WRONLY | O_CREAT | O_TRUNC, @as(c_int, 0o644));
    if (fd < 0) return error.CacheWriteFailed;
    errdefer {
        _ = close(fd);
        _ = unlink(ztmp.ptr);
    }

    try writeAllFd(fd, hdr_buf);
    var pad_zeros: [page_size]u8 = @splat(0);
    for (bufs) |b| {
        try writeAllFd(fd, b.ptr[0..b.len]);
        const rem = align16K(b.len) - b.len;
        if (rem > 0) try writeAllFd(fd, pad_zeros[0..rem]);
    }
    _ = close(fd);
    if (rename(ztmp.ptr, zdst.ptr) != 0) {
        _ = unlink(ztmp.ptr);
        return error.CacheRenameFailed;
    }
}
