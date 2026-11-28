//! Built-in Apple Silicon Metal GPU TFQ3 quantizer (`mlx_tf convert SRC DST --format tfq3`).
//!
//! Zero Apple MLX, zero external C++ tools, 100% data-free (zero calibration data):
//!   - Generates (or loads cached) deterministic 16 KiB `pw_v3_tables.bin` (`PWV3`) codebook
//!   - Runs 2D FWHT-128 + 2D-AffineCoset/half4 beam encoding on Metal GPU (`tfq3_encode.metal`)
//!   - Supports 3-bit, 4-bit, 5-bit, and 3.5-bit mixed profile (`--profile 3b|3.5b|4b|5b`)
//!   - Emits 16 KiB-aligned cross-platform `.tffpw` (`TFPN`) shards with pair-contiguous `gate+up` & `k+v`
//!   - Emits lossless BF16 residual `.safetensors` (norms, conv1d, embeddings, non-128-multiple linears)
//!   - Output runs byte-for-byte identically on `mlx_tf` (Metal), `tensorfold-native` (Metal), & CUDA!
const std = @import("std");
const shaders = @import("shaders");
const mtl = @import("metal4.zig");
const st = @import("safetensors.zig");
const tfq3 = @import("tfq3.zig");
const model_mod = @import("model.zig");

const nowMs = model_mod.nowMs;

extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern "c" fn mkdir(path: [*:0]const u8, mode: u16) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;

const O_WRONLY: c_int = 0x0001;
const O_CREAT: c_int = 0x0200;
const O_TRUNC: c_int = 0x0400;

pub const Tfq3ConvertOptions = struct {
    bits: usize = 4, // 3, 4, or 5
    profile: []const u8 = "", // "", "3b", "3.5b", "4b", "5b"
    head_bits: usize = 0, // 0 = auto (max(bits, 4))
    quantize_head: bool = true,
};

pub const Tfq3ConvertStats = struct {
    tensors_tfq3: usize = 0,
    tensors_residual: usize = 0,
    shards_written: usize = 0,
    bytes_in: usize = 0,
    bytes_out: usize = 0,
    mean_snr_db: f64 = 0,
    elapsed_ms: f64 = 0,
};

const EncMatParams = extern struct {
    R: u32,
    C: u32,
    b_bits: u32,
    shrink: f32,
    r_off: u32,
    src_dtype: u32,
};

const EncStreamParams = extern struct {
    R: u32,
    C: u32,
    b_bits: u32,
    stream_off: u32,
    n_streams: u32,
};

const P2 = struct { x: f32, y: f32 };

inline fn f32ToF16Bits(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

inline fn f16BitsToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

inline fn packH2(x: f32, y: f32) u32 {
    return @as(u32, f32ToF16Bits(x)) | (@as(u32, f32ToF16Bits(y)) << 16);
}

inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7FFF +% lsb) >> 16);
}

fn invNormCdf(p_in: f64) f64 {
    const p = std.math.clamp(p_in, 1e-12, 1.0 - 1e-12);
    const a = [_]f64{ -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
    const b = [_]f64{ -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
    const c = [_]f64{ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
    const d = [_]f64{ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
    if (p < 0.02425) {
        const q = @sqrt(-2.0 * @log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 0.97575) {
        const q = p - 0.5;
        const r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const q = @sqrt(-2.0 * @log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}

/// Deterministic 2D unit-Gaussian Lloyd-Max k-means codebook trainer (zero calibration data).
fn train2D(a: std.mem.Allocator, k_cnt: usize, seed: u64, n_iter: usize) ![]P2 {
    const ns: usize = 65536;
    const data = try a.alloc(P2, ns);
    defer a.free(data);

    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (data) |*pt| {
        const q1 = @max(1e-12, rand.float(f64));
        const q2 = rand.float(f64);
        const rad = @sqrt(-2.0 * @log(q1));
        const ang = 2.0 * std.math.pi * q2;
        pt.* = .{ .x = @floatCast(rad * @cos(ang)), .y = @floatCast(rad * @sin(ang)) };
    }

    const cen = try a.alloc(P2, k_cnt);
    const phi2: f64 = 1.3247179572447460;
    const a1: f64 = 1.0 / phi2;
    const a2: f64 = 1.0 / (phi2 * phi2);
    for (0..k_cnt) |i| {
        const q1 = @mod(0.5 + (@as(f64, @floatFromInt(i)) + 0.5) * a1 + 0.031 * @as(f64, @floatFromInt(seed)), 1.0);
        const q2 = @mod(0.5 + (@as(f64, @floatFromInt(i)) + 0.5) * a2 + 0.067 * @as(f64, @floatFromInt(seed)), 1.0);
        cen[i] = .{ .x = @floatCast(invNormCdf(q1)), .y = @floatCast(invNormCdf(q2)) };
    }

    const sx = try a.alloc(f64, k_cnt);
    defer a.free(sx);
    const sy = try a.alloc(f64, k_cnt);
    defer a.free(sy);
    const cnt = try a.alloc(u32, k_cnt);
    defer a.free(cnt);

    for (0..n_iter) |_| {
        @memset(sx, 0);
        @memset(sy, 0);
        @memset(cnt, 0);
        for (data) |pt| {
            var bi: usize = 0;
            var bd: f32 = 1e30;
            for (cen, 0..) |c, k| {
                const dx = pt.x - c.x;
                const dy = pt.y - c.y;
                const dist = dx * dx + dy * dy;
                if (dist < bd) {
                    bd = dist;
                    bi = k;
                }
            }
            sx[bi] += pt.x;
            sy[bi] += pt.y;
            cnt[bi] += 1;
        }
        for (0..k_cnt) |k| {
            if (cnt[k] > 0) {
                cen[k].x = @floatCast(sx[k] / @as(f64, @floatFromInt(cnt[k])));
                cen[k].y = @floatCast(sy[k] / @as(f64, @floatFromInt(cnt[k])));
            }
        }
    }
    return cen;
}

fn fillC64(out: *[64][4]u16, alpha: f32, beta: f32) void {
    for (0..64) |i| {
        const fi: f64 = @as(f64, @floatFromInt(i)) + 0.5;
        const sx: f32 = if ((i & 1) != 0) -1.0 else 1.0;
        const sy: f32 = if ((i & 2) != 0) -1.0 else 1.0;
        const v0: f32 = @floatCast(@mod(fi * 0.6180339887498949, 1.0) * 2.0 - 1.0);
        const v1: f32 = @floatCast(@mod(fi * 0.3819660112501051, 1.0) * 2.0 - 1.0);
        const v2: f32 = @floatCast(@mod(fi * 0.7548776662466927, 1.0) * 2.0 - 1.0);
        const v3: f32 = @floatCast(@mod(fi * 0.5698402909980532, 1.0) * 2.0 - 1.0);
        out[i][0] = f32ToF16Bits(sx * (1.0 + alpha * v0));
        out[i][1] = f32ToF16Bits(sy * (1.0 + alpha * v1));
        out[i][2] = f32ToF16Bits(beta * v2);
        out[i][3] = f32ToF16Bits(beta * v3);
    }
}

fn loadOrGenerateTables(a: std.mem.Allocator, out_dir: []const u8) !tfq3.PwV3TablesFile {
    var tbl: tfq3.PwV3TablesFile = std.mem.zeroes(tfq3.PwV3TablesFile);

    // 1. Check cached universal codebook (~/.cache/tensorfold/pw_v3_tables.bin)
    var cache_path_opt: ?[]u8 = null;
    defer if (cache_path_opt) |p| a.free(p);
    if (getenv("HOME")) |home_z| {
        const cp = try std.fmt.allocPrint(a, "{s}/.cache/tensorfold/pw_v3_tables.bin", .{std.mem.span(home_z)});
        cache_path_opt = cp;
        if (st.mapFile(a, cp)) |raw| {
            defer st.unmapFile(raw);
            if (raw.len == @sizeOf(tfq3.PwV3TablesFile)) {
                @memcpy(std.mem.asBytes(&tbl), raw);
                if (tbl.magic == tfq3.magic_pwv3 and tbl.version == tfq3.format_version) {
                    const dst_p = try std.fs.path.join(a, &.{ out_dir, tfq3.tables_filename });
                    defer a.free(dst_p);
                    try writeFileBytes(a, dst_p, std.mem.asBytes(&tbl));
                    return tbl;
                }
            }
        } else |_| {}
    }

    // 2. Generate deterministic 16 KiB PWV3 codebook (synthetic unit Gaussian, zero data)
    tbl = std.mem.zeroes(tfq3.PwV3TablesFile);
    tbl.magic = tfq3.magic_pwv3;
    tbl.version = tfq3.format_version;
    tbl.a3 = 0.22;
    tbl.b3 = 0.38;
    tbl.a4 = 0.18;
    tbl.b4 = 0.30;
    tbl.a5 = 0.10;
    tbl.b5 = 0.18;

    for (0..4) |c| {
        const t3 = try train2D(a, 64, 10 + c, 24);
        defer a.free(t3);
        for (0..64) |i| tbl.lut3[i * 4 + c] = packH2(t3[i].x, t3[i].y);
    }
    for (0..4) |c| {
        const t4 = try train2D(a, 256, 100 + c, 20);
        defer a.free(t4);
        for (0..256) |i| tbl.lut4[i * 4 + c] = packH2(t4[i].x, t4[i].y);
    }
    const t5 = try train2D(a, 1024, 500, 16);
    defer a.free(t5);
    for (0..1024) |i| tbl.lut5[i] = packH2(t5[i].x, t5[i].y);

    fillC64(&tbl.c64_3, tbl.a3, tbl.b3);
    fillC64(&tbl.c64_4, tbl.a4, tbl.b4);
    fillC64(&tbl.c64_5, tbl.a5, tbl.b5);

    if (cache_path_opt) |cp| {
        if (std.fs.path.dirname(cp)) |parent| makeDirP(a, parent) catch {};
        writeFileBytes(a, cp, std.mem.asBytes(&tbl)) catch {};
    }
    const dst_p = try std.fs.path.join(a, &.{ out_dir, tfq3.tables_filename });
    defer a.free(dst_p);
    try writeFileBytes(a, dst_p, std.mem.asBytes(&tbl));
    return tbl;
}

fn buildGrid(a: std.mem.Allocator, lut_u32: []const u32, n_sym: usize, c_shift: u5, g_dim: usize, kc: usize) ![]u16 {
    const n_cb = lut_u32.len / n_sym;
    const rmax: f32 = 3.2;
    const grid = try a.alloc(u16, n_cb * g_dim * g_dim * kc);
    const pts = try a.alloc(P2, lut_u32.len);
    defer a.free(pts);
    for (lut_u32, 0..) |u, i| {
        pts[i] = .{ .x = f16BitsToF32(@intCast(u & 0xFFFF)), .y = f16BitsToF32(@intCast(u >> 16)) };
    }

    const PairDist = struct { d: f32, idx: u16 };
    const dl = try a.alloc(PairDist, n_sym);
    defer a.free(dl);

    for (0..n_cb) |c| {
        for (0..g_dim) |gy| {
            const cy = -rmax + (@as(f32, @floatFromInt(gy)) + 0.5) * (2.0 * rmax / @as(f32, @floatFromInt(g_dim)));
            for (0..g_dim) |gx| {
                const cx = -rmax + (@as(f32, @floatFromInt(gx)) + 0.5) * (2.0 * rmax / @as(f32, @floatFromInt(g_dim)));
                for (0..n_sym) |i| {
                    const idx = (i << c_shift) | c;
                    const dx = pts[idx].x - cx;
                    const dy = pts[idx].y - cy;
                    dl[i] = .{ .d = dx * dx + dy * dy, .idx = @intCast(i) };
                }
                std.mem.sort(PairDist, dl, {}, struct {
                    fn lt(_: void, lhs: PairDist, rhs: PairDist) bool {
                        return lhs.d < rhs.d;
                    }
                }.lt);
                const base = ((c * g_dim + gy) * g_dim + gx) * kc;
                for (0..kc) |k| grid[base + k] = dl[k].idx;
            }
        }
    }
    return grid;
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

fn writeFileBytes(a: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const zpath = try a.dupeSentinel(u8, path, 0);
    defer a.free(zpath);
    const fd = open(zpath.ptr, O_WRONLY | O_CREAT | O_TRUNC, @as(c_int, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = close(fd);
    try writeAllFd(fd, bytes);
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

fn copyFileIfExists(a: std.mem.Allocator, src_dir: []const u8, dst_dir: []const u8, fname: []const u8) !void {
    const src_p = try std.fs.path.join(a, &.{ src_dir, fname });
    defer a.free(src_p);
    const raw = st.mapFile(a, src_p) catch return;
    defer st.unmapFile(raw);
    const dst_p = try std.fs.path.join(a, &.{ dst_dir, fname });
    defer a.free(dst_p);
    try writeFileBytes(a, dst_p, raw);
}

fn isZeroCenteredQwenNorm(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".input_layernorm.weight") or
        std.mem.endsWith(u8, name, ".post_attention_layernorm.weight") or
        std.mem.endsWith(u8, name, "model.norm.weight") or
        std.mem.endsWith(u8, name, ".q_norm.weight") or
        std.mem.endsWith(u8, name, ".k_norm.weight");
}

fn shouldTfq3Quantize(name: []const u8, t: st.Tensor, quantize_head: bool) bool {
    if (t.dtype != .bf16 and t.dtype != .f16 and t.dtype != .f32) return false;
    if (t.rank != 2) return false;
    if (!std.mem.endsWith(u8, name, ".weight")) return false;
    if (std.mem.indexOf(u8, name, "embed_tokens") != null or std.mem.indexOf(u8, name, "embeddings") != null) return false;
    if (!quantize_head and std.mem.endsWith(u8, name, "lm_head.weight")) return false;
    if (std.mem.indexOf(u8, name, "conv1d") != null or std.mem.indexOf(u8, name, "norm") != null) return false;
    if (std.mem.endsWith(u8, name, ".gate.weight")) return false;
    const r = t.shape[0];
    const c = t.shape[1];
    return r > 0 and c > 0 and (r % 128 == 0) and (c % 128 == 0);
}

fn readElemF32(t: st.Tensor, idx: usize) f32 {
    return switch (t.dtype) {
        .bf16 => bf16ToF32(t.asSlice(u16)[idx]),
        .f16 => f16BitsToF32(t.asSlice(u16)[idx]),
        .f32 => t.asSlice(f32)[idx],
        else => 0.0,
    };
}

fn dtypeName(dt: st.DType) []const u8 {
    return switch (dt) {
        .u32 => "U32",
        .bf16 => "BF16",
        .f16 => "F16",
        .f32 => "F32",
        .other => "U8",
    };
}

const ResidualOut = struct {
    name: []const u8,
    dtype_str: []const u8,
    shape: [6]usize = @splat(0),
    rank: usize = 0,
    data: []u8,
};

fn writeSafetensors(a: std.mem.Allocator, path: []const u8, tensors: []const ResidualOut) !void {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(a);
    try hdr.appendSlice(a, "{\"__metadata__\":{\"format\":\"mlx_tf_tfq3\"}");
    var offset: usize = 0;
    for (tensors) |ot| {
        const end_off = offset + ot.data.len;
        const pre = try std.fmt.allocPrint(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ ot.name, ot.dtype_str });
        defer a.free(pre);
        try hdr.appendSlice(a, pre);
        for (0..ot.rank) |di| {
            if (di > 0) try hdr.append(a, ',');
            const ds = try std.fmt.allocPrint(a, "{d}", .{ot.shape[di]});
            defer a.free(ds);
            try hdr.appendSlice(a, ds);
        }
        const post = try std.fmt.allocPrint(a, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end_off });
        defer a.free(post);
        try hdr.appendSlice(a, post);
        offset = end_off;
    }
    try hdr.append(a, '}');
    while (hdr.items.len % 8 != 0) try hdr.append(a, ' ');

    const zpath = try a.dupeSentinel(u8, path, 0);
    defer a.free(zpath);
    const fd = open(zpath.ptr, O_WRONLY | O_CREAT | O_TRUNC, @as(c_int, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = close(fd);
    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, @intCast(hdr.items.len), .little);
    try writeAllFd(fd, &len_le);
    try writeAllFd(fd, hdr.items);
    for (tensors) |ot| {
        if (ot.data.len > 0) try writeAllFd(fd, ot.data);
    }
}

fn writeTfpnShard(
    a: std.mem.Allocator,
    path: []const u8,
    metas: []const tfq3.TfpnTensorMeta,
    scales: []const u16,
    weights: []const u32,
) !usize {
    const off_meta: usize = @sizeOf(tfq3.TfpnHeader);
    const meta_bytes = metas.len * @sizeOf(tfq3.TfpnTensorMeta);
    const off_scales: usize = tfq3.align16K(off_meta + meta_bytes);
    const sc_bytes = scales.len * 2;
    const off_weights: usize = tfq3.align16K(off_scales + @max(sc_bytes, tfq3.page_size));
    const wt_bytes = weights.len * 4;
    const end_pos = off_weights + @max(wt_bytes, tfq3.page_size);
    const file_len = tfq3.align16K(end_pos);

    const hdr = tfq3.TfpnHeader{
        .magic = tfq3.magic_tfpn,
        .version = tfq3.format_version,
        .num_tensors = @intCast(metas.len),
        .pad0 = 0,
        .off_meta = @intCast(off_meta),
        .off_scales = @intCast(off_scales),
        .off_weights = @intCast(off_weights),
        .total_scale_halfs = @intCast(scales.len),
        .total_weight_words = @intCast(weights.len),
        .pad1 = 0,
    };

    const zpath = try a.dupeSentinel(u8, path, 0);
    defer a.free(zpath);
    const fd = open(zpath.ptr, O_WRONLY | O_CREAT | O_TRUNC, @as(c_int, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = close(fd);

    var zeros: [tfq3.page_size]u8 = @splat(0);
    try writeAllFd(fd, std.mem.asBytes(&hdr));
    try writeAllFd(fd, std.mem.sliceAsBytes(metas));
    var cur = off_meta + meta_bytes;
    while (cur < off_scales) {
        const step = @min(off_scales - cur, zeros.len);
        try writeAllFd(fd, zeros[0..step]);
        cur += step;
    }
    if (sc_bytes > 0) try writeAllFd(fd, std.mem.sliceAsBytes(scales));
    cur = off_scales + sc_bytes;
    while (cur < off_weights) {
        const step = @min(off_weights - cur, zeros.len);
        try writeAllFd(fd, zeros[0..step]);
        cur += step;
    }
    if (wt_bytes > 0) try writeAllFd(fd, std.mem.sliceAsBytes(weights));
    cur = off_weights + wt_bytes;
    while (cur < file_len) {
        const step = @min(file_len - cur, zeros.len);
        try writeAllFd(fd, zeros[0..step]);
        cur += step;
    }
    return file_len;
}

const EncodeTask = struct {
    raw_key: []const u8,
    canon_name: []const u8,
    t: st.Tensor,
    bits: u32,
};

fn pickBits(canon: []const u8, opts: Tfq3ConvertOptions) u32 {
    if (std.mem.eql(u8, canon, "lm_head")) {
        const hb = if (opts.head_bits != 0) opts.head_bits else @max(opts.bits, 4);
        return @intCast(hb);
    }
    if (std.mem.eql(u8, opts.profile, "3b")) return 3;
    if (std.mem.eql(u8, opts.profile, "4b")) return 4;
    if (std.mem.eql(u8, opts.profile, "5b")) return 5;
    if (std.mem.eql(u8, opts.profile, "3.5b")) {
        if (std.mem.indexOf(u8, canon, ".mlp.") != null) return 3;
        return 4;
    }
    return @intCast(opts.bits);
}

/// Pair-friendly sort: keeps `k_proj` right before `v_proj` and `gate_proj` right before `up_proj`!
fn taskOrderKey(name: []const u8) u8 {
    if (std.mem.endsWith(u8, name, ".gate_proj")) return 10;
    if (std.mem.endsWith(u8, name, ".up_proj")) return 11;
    if (std.mem.endsWith(u8, name, ".down_proj")) return 12;
    if (std.mem.endsWith(u8, name, ".q_proj")) return 20;
    if (std.mem.endsWith(u8, name, ".k_proj")) return 21;
    if (std.mem.endsWith(u8, name, ".v_proj")) return 22;
    if (std.mem.endsWith(u8, name, ".o_proj")) return 23;
    return 50;
}

fn layerPrefixOf(name: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot| return name[0..dot];
    return name;
}

pub fn convertModelTfq3(a: std.mem.Allocator, src_dir: []const u8, dst_dir: []const u8, opts: Tfq3ConvertOptions) !Tfq3ConvertStats {
    if (opts.bits != 3 and opts.bits != 4 and opts.bits != 5) return error.UnsupportedQuantConfig;
    const t0 = nowMs();
    try makeDirP(a, dst_dir);

    var ckpt = try st.Checkpoint.openDir(a, src_dir);
    defer ckpt.close();

    // Detect raw HF Qwen3.5/3.8 zero-centered norms & conv1d layout
    var shift_qwen_norms = false;
    {
        var it = ckpt.tensors.iterator();
        while (it.next()) |e| {
            if (std.mem.indexOf(u8, e.key_ptr.*, "conv1d.weight") != null and e.value_ptr.*.rank == 3 and e.value_ptr.*.shape[2] != 1) {
                shift_qwen_norms = true;
                break;
            }
        }
    }

    // Generate or load universal 16 KiB PWV3 codebook + nearest-neighbor grids
    const tbl = try loadOrGenerateTables(a, dst_dir);
    const grid3 = try buildGrid(a, &tbl.lut3, 64, 2, 16, 8);
    defer a.free(grid3);
    const grid4 = try buildGrid(a, &tbl.lut4, 256, 2, 16, 10);
    defer a.free(grid4);
    const grid5 = try buildGrid(a, &tbl.lut5, 1024, 0, 32, 12);
    defer a.free(grid5);

    // Collect sorted keys -> split into TFQ3 encode tasks + BF16 residual tensors
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(a);
    {
        var it = ckpt.tensors.keyIterator();
        while (it.next()) |kp| try keys.append(a, kp.*);
    }
    std.mem.sort([]const u8, keys.items, {}, struct {
        fn lt(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.lt);

    var tasks: std.ArrayList(EncodeTask) = .empty;
    defer {
        for (tasks.items) |tk| a.free(tk.canon_name);
        tasks.deinit(a);
    }
    var residuals: std.ArrayList(ResidualOut) = .empty;
    defer {
        for (residuals.items) |r| {
            a.free(r.name);
            a.free(r.data);
        }
        residuals.deinit(a);
    }

    var stats = Tfq3ConvertStats{};
    for (keys.items) |raw_key| {
        if (std.mem.startsWith(u8, raw_key, "model.visual.") or
            std.mem.startsWith(u8, raw_key, "vision_tower.") or
            std.mem.indexOf(u8, raw_key, "mtp.") != null)
        {
            continue;
        }
        const t = ckpt.tensors.get(raw_key).?;
        stats.bytes_in += t.bytes.len;

        const norm_key = if (std.mem.startsWith(u8, raw_key, "model.language_model."))
            try std.fmt.allocPrint(a, "language_model.model.{s}", .{raw_key["model.language_model.".len..]})
        else
            try a.dupe(u8, raw_key);

        if (shouldTfq3Quantize(norm_key, t, opts.quantize_head)) {
            const canon = tfq3.canonicalModule(norm_key);
            const canon_owned = try a.dupe(u8, canon);
            a.free(norm_key);
            try tasks.append(a, .{
                .raw_key = raw_key,
                .canon_name = canon_owned,
                .t = t,
                .bits = pickBits(canon_owned, opts),
            });
        } else {
            var shape = t.shape;
            if (std.mem.indexOf(u8, norm_key, "conv1d.weight") != null and t.rank == 3 and t.shape[2] != 1) {
                shape[1] = t.shape[2];
                shape[2] = t.shape[1];
            }
            if (shift_qwen_norms and isZeroCenteredQwenNorm(norm_key) and t.rank == 1) {
                const n = t.elemCount();
                const buf = try a.alloc(u8, n * 2);
                const dst = @as([*]u16, @ptrCast(@alignCast(buf.ptr)))[0..n];
                for (0..n) |i| dst[i] = f32ToBf16(readElemF32(t, i) + 1.0);
                try residuals.append(a, .{ .name = norm_key, .dtype_str = "BF16", .shape = shape, .rank = t.rank, .data = buf });
            } else {
                const buf = try a.dupe(u8, t.bytes);
                try residuals.append(a, .{ .name = norm_key, .dtype_str = dtypeName(t.dtype), .shape = shape, .rank = t.rank, .data = buf });
            }
        }
    }

    // Sort tasks so (gate_proj, up_proj) and (k_proj, v_proj) are ALWAYS adjacent for pair-fusion!
    std.mem.sort(EncodeTask, tasks.items, {}, struct {
        fn lt(_: void, lhs: EncodeTask, rhs: EncodeTask) bool {
            const lp = layerPrefixOf(lhs.canon_name);
            const rp = layerPrefixOf(rhs.canon_name);
            const ord = std.mem.order(u8, lp, rp);
            if (ord != .eq) return ord == .lt;
            const lk = taskOrderKey(lhs.canon_name);
            const rk = taskOrderKey(rhs.canon_name);
            if (lk != rk) return lk < rk;
            return std.mem.order(u8, lhs.canon_name, rhs.canon_name) == .lt;
        }
    }.lt);

    // Initialize Metal GPU encoder context
    var ctx = try mtl.Context.init(256);
    defer ctx.deinit();
    const lib = try ctx.compileSource(shaders.tfq3_encode, .fast);
    defer lib.deinit();

    const pso_row = try ctx.makePipeline(lib, "tfq3_raw_row_fwht");
    defer pso_row.deinit();
    const pso_col = try ctx.makePipeline(lib, "tfq3_col_fwht_and_ss");
    defer pso_col.deinit();
    const pso_pack = try ctx.makePipeline(lib, "tfq3_finalize_scales_pack");
    defer pso_pack.deinit();
    const pso_enc3 = try ctx.makePipeline(lib, "tfq3_beam_encode_3b");
    defer pso_enc3.deinit();
    const pso_enc4 = try ctx.makePipeline(lib, "tfq3_beam_encode_4b");
    defer pso_enc4.deinit();
    const pso_enc5 = try ctx.makePipeline(lib, "tfq3_beam_encode_5b");
    defer pso_enc5.deinit();
    const pso_snr = try ctx.makePipeline(lib, "tfq3_verify_snr");
    defer pso_snr.deinit();

    // Upload tables & nearest-neighbor grids
    const b_lut3 = try ctx.allocBuffer(@sizeOf(@TypeOf(tbl.lut3)));
    defer b_lut3.deinit();
    @memcpy(b_lut3.ptr[0..@sizeOf(@TypeOf(tbl.lut3))], std.mem.asBytes(&tbl.lut3));
    const b_lut4 = try ctx.allocBuffer(@sizeOf(@TypeOf(tbl.lut4)));
    defer b_lut4.deinit();
    @memcpy(b_lut4.ptr[0..@sizeOf(@TypeOf(tbl.lut4))], std.mem.asBytes(&tbl.lut4));
    const b_lut5 = try ctx.allocBuffer(@sizeOf(@TypeOf(tbl.lut5)));
    defer b_lut5.deinit();
    @memcpy(b_lut5.ptr[0..@sizeOf(@TypeOf(tbl.lut5))], std.mem.asBytes(&tbl.lut5));

    const b_c64_3 = try ctx.allocBuffer(@sizeOf(@TypeOf(tbl.c64_3)));
    defer b_c64_3.deinit();
    @memcpy(b_c64_3.ptr[0..@sizeOf(@TypeOf(tbl.c64_3))], std.mem.asBytes(&tbl.c64_3));
    const b_c64_4 = try ctx.allocBuffer(@sizeOf(@TypeOf(tbl.c64_4)));
    defer b_c64_4.deinit();
    @memcpy(b_c64_4.ptr[0..@sizeOf(@TypeOf(tbl.c64_4))], std.mem.asBytes(&tbl.c64_4));
    const b_c64_5 = try ctx.allocBuffer(@sizeOf(@TypeOf(tbl.c64_5)));
    defer b_c64_5.deinit();
    @memcpy(b_c64_5.ptr[0..@sizeOf(@TypeOf(tbl.c64_5))], std.mem.asBytes(&tbl.c64_5));

    const b_grid3 = try ctx.allocBuffer(grid3.len * 2);
    defer b_grid3.deinit();
    @memcpy(b_grid3.ptr[0 .. grid3.len * 2], std.mem.sliceAsBytes(grid3));
    const b_grid4 = try ctx.allocBuffer(grid4.len * 2);
    defer b_grid4.deinit();
    @memcpy(b_grid4.ptr[0 .. grid4.len * 2], std.mem.sliceAsBytes(grid4));
    const b_grid5 = try ctx.allocBuffer(grid5.len * 2);
    defer b_grid5.deinit();
    @memcpy(b_grid5.ptr[0 .. grid5.len * 2], std.mem.sliceAsBytes(grid5));

    // Size reusable GPU scratch buffers dynamically from the largest task chunk (max 8192 rows/chunk!)
    var max_elems: usize = 128 * 128;
    var max_raw_bytes: usize = max_elems * 4;
    var max_blks: usize = 64;
    for (tasks.items) |tk| {
        const sr: usize = @min(tk.t.shape[0], 8192);
        const c = tk.t.shape[1];
        const elems = sr * c;
        if (elems > max_elems) max_elems = elems;
        const elem_b: usize = if (tk.t.dtype == .f32) 4 else 2;
        if (elems * elem_b > max_raw_bytes) max_raw_bytes = elems * elem_b;
        const nb = (sr / 128) * (c / 128);
        if (nb > max_blks) max_blks = nb;
    }

    const hist_slots: usize = 524288; // 512K pair-slots * 256 * 4B = 512 MiB
    const b_in = try ctx.allocBuffer(max_raw_bytes);
    defer b_in.deinit();
    const b_tmp = try ctx.allocBuffer(max_elems * 4);
    defer b_tmp.deinit();
    const b_rot = try ctx.allocBuffer(max_elems * 4);
    defer b_rot.deinit();
    const b_ss = try ctx.allocBuffer(@max(max_blks, 64) * 4);
    defer b_ss.deinit();
    const b_str = try ctx.allocBuffer(max_elems * 4);
    defer b_str.deinit();
    const b_hist = try ctx.allocBuffer(hist_slots * 256 * 4);
    defer b_hist.deinit();
    const b_pack = try ctx.allocBuffer(max_elems); // max 5 bits -> 5/8 byte/elem < 1 byte/elem!
    defer b_pack.deinit();
    const b_sc = try ctx.allocBuffer(@max(max_blks, 64) * 2);
    defer b_sc.deinit();
    const b_se = try ctx.allocBuffer(8);
    defer b_se.deinit();
    ctx.commitResidency();

    // Encode all tasks into <= 3.4 GiB TFPN shards
    const max_shard_bytes: usize = 3400 * 1024 * 1024;
    var cur_metas: std.ArrayList(tfq3.TfpnTensorMeta) = .empty;
    defer cur_metas.deinit(a);
    var cur_scales: std.ArrayList(u16) = .empty;
    defer cur_scales.deinit(a);
    var cur_words: std.ArrayList(u32) = .empty;
    defer cur_words.deinit(a);

    var snr_sum: f64 = 0;
    var snr_cnt: usize = 0;
    stats.bytes_out = tfq3.tables_bytes;

    for (tasks.items, 0..) |tk, ti| {
        const R: u32 = @intCast(tk.t.shape[0]);
        const C: u32 = @intCast(tk.t.shape[1]);
        const bits = tk.bits;
        const sc_cnt: usize = @as(usize, R / 128) * @as(usize, C / 128);
        const wd_cnt: usize = @as(usize, R / 16) * @as(usize, C / 128) * @as(usize, bits * 2) * 32;

        // Flush shard if adding this tensor exceeds 3.4 GiB (avoid splitting gate_proj/up_proj or k_proj/v_proj pairs!)
        const is_second_of_pair = std.mem.endsWith(u8, tk.canon_name, ".up_proj") or std.mem.endsWith(u8, tk.canon_name, ".v_proj");
        if (cur_metas.items.len > 0 and !is_second_of_pair and
            (cur_words.items.len + wd_cnt) * 4 + (cur_scales.items.len + sc_cnt) * 2 > max_shard_bytes)
        {
            stats.shards_written += 1;
            const sname = try std.fmt.allocPrint(a, "model_tfq3-{d:0>5}.tffpw", .{stats.shards_written});
            defer a.free(sname);
            const spath = try std.fs.path.join(a, &.{ dst_dir, sname });
            defer a.free(spath);
            stats.bytes_out += try writeTfpnShard(a, spath, cur_metas.items, cur_scales.items, cur_words.items);
            cur_metas.clearRetainingCapacity();
            cur_scales.clearRetainingCapacity();
            cur_words.clearRetainingCapacity();
        }

        const sc_off_base: u32 = @intCast(cur_scales.items.len);
        const wd_off_base: u64 = @intCast(cur_words.items.len);
        try cur_scales.resize(a, cur_scales.items.len + sc_cnt);
        try cur_words.resize(a, cur_words.items.len + wd_cnt);

        const src_dtype: u32 = switch (tk.t.dtype) {
            .bf16 => 0,
            .f16 => 1,
            .f32 => 2,
            else => 0,
        };
        const elem_bytes: usize = if (src_dtype == 2) 4 else 2;
        const pso_enc = if (bits == 3) pso_enc3 else if (bits == 4) pso_enc4 else pso_enc5;
        const b_lut = if (bits == 3) b_lut3 else if (bits == 4) b_lut4 else b_lut5;
        const b_c64 = if (bits == 3) b_c64_3 else if (bits == 4) b_c64_4 else b_c64_5;
        const b_grid = if (bits == 3) b_grid3 else if (bits == 4) b_grid4 else b_grid5;

        const chunk_r: u32 = if (R > 8192) 8192 else R;
        var tot_sig: f64 = 0;
        var tot_err: f64 = 0;
        var r0: u32 = 0;
        while (r0 < R) : (r0 += chunk_r) {
            const sr: u32 = @min(chunk_r, R - r0);
            const raw_off = @as(usize, r0) * C * elem_bytes;
            const raw_len = @as(usize, sr) * C * elem_bytes;
            @memcpy(b_in.ptr[0..raw_len], tk.t.bytes[raw_off..][0..raw_len]);
            const n_blks: usize = (sr / 128) * (C / 128);
            @memset(b_ss.ptr[0 .. n_blks * 4], 0);
            @memset(b_se.ptr[0..8], 0);

            const mp = EncMatParams{ .R = sr, .C = C, .b_bits = bits, .shrink = 1.0, .r_off = r0, .src_dtype = src_dtype };
            const total_streams: u32 = (sr / 16) * 32;
            const n_pairs: usize = C / 4;
            const batch_sz: u32 = @intCast(@max(@as(usize, 1), hist_slots / n_pairs));

            var cb = ctx.beginCommandBuffer();
            var enc = cb.beginCompute();

            enc.setPipeline(pso_row);
            enc.setBuffer(b_in, 0, 0);
            enc.setBuffer(b_tmp, 0, 1);
            enc.setValue(mp, 2);
            enc.dispatchGroups(.{ .width = C / 128, .height = sr, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            enc.setPipeline(pso_col);
            enc.setBuffer(b_tmp, 0, 0);
            enc.setBuffer(b_rot, 0, 1);
            enc.setBuffer(b_ss, 0, 2);
            enc.setValue(mp, 3);
            enc.dispatchGroups(.{ .width = C, .height = sr / 128, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            enc.setPipeline(pso_pack);
            enc.setBuffer(b_rot, 0, 0);
            enc.setBuffer(b_ss, 0, 1);
            enc.setBuffer(b_sc, 0, 2);
            enc.setBuffer(b_str, 0, 3);
            enc.setValue(mp, 4);
            enc.dispatchGroups(.{ .width = sr / 16, .height = C / 128, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            enc.setPipeline(pso_enc);
            var s_off: u32 = 0;
            while (s_off < total_streams) : (s_off += batch_sz) {
                const cnt: u32 = @min(batch_sz, total_streams - s_off);
                const ep = EncStreamParams{ .R = sr, .C = C, .b_bits = bits, .stream_off = s_off, .n_streams = cnt };
                enc.setBuffer(b_str, 0, 0);
                enc.setBuffer(b_lut, 0, 1);
                enc.setBuffer(b_c64, 0, 2);
                enc.setBuffer(b_grid, 0, 3);
                enc.setBuffer(b_hist, 0, 4);
                enc.setBuffer(b_pack, 0, 5);
                enc.setValue(ep, 6);
                enc.dispatchGroups(.{ .width = cnt, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();
            }

            enc.setPipeline(pso_snr);
            enc.setBuffer(b_pack, 0, 0);
            enc.setBuffer(b_sc, 0, 1);
            enc.setBuffer(b_lut, 0, 2);
            enc.setBuffer(b_c64, 0, 3);
            enc.setBuffer(b_rot, 0, 4);
            enc.setBuffer(b_se, 0, 5);
            enc.setValue(mp, 6);
            enc.dispatchGroups(.{ .width = sr / 16, .height = 1, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.end();
            cb.commit();
            try ctx.waitIdle();

            const se = b_se.slice(f32, 2);
            tot_sig += se[0];
            tot_err += se[1];

            const sc_off_chunk = (r0 / 128) * (C / 128);
            const wd_off_chunk = @as(usize, r0 / 16) * (C / 128) * (bits * 2) * 32;
            const wd_cnt_chunk = @as(usize, sr / 16) * (C / 128) * (bits * 2) * 32;
            @memcpy(cur_scales.items[sc_off_base + sc_off_chunk ..][0..n_blks], b_sc.slice(u16, n_blks));
            @memcpy(cur_words.items[@as(usize, @intCast(wd_off_base)) + wd_off_chunk ..][0..wd_cnt_chunk], b_pack.slice(u32, wd_cnt_chunk));
        }

        const snr_db: f32 = if (tot_err > 1e-18) @floatCast(10.0 * @log10(tot_sig / tot_err)) else 99.0;
        snr_sum += snr_db;
        snr_cnt += 1;

        var meta = std.mem.zeroes(tfq3.TfpnTensorMeta);
        const copy_len = @min(tk.canon_name.len, meta.name.len - 1);
        @memcpy(meta.name[0..copy_len], tk.canon_name[0..copy_len]);
        meta.R = R;
        meta.C = C;
        meta.b_bits = bits;
        meta.sc_off = sc_off_base;
        meta.word_off = wd_off_base;
        meta.word_count = @intCast(wd_cnt);
        meta.snr_db = snr_db;
        try cur_metas.append(a, meta);
        stats.tensors_tfq3 += 1;
        _ = ti;
    }

    if (cur_metas.items.len > 0) {
        stats.shards_written += 1;
        const sname = if (stats.shards_written == 1)
            try a.dupe(u8, "model_tfq3.tffpw")
        else
            try std.fmt.allocPrint(a, "model_tfq3-{d:0>5}.tffpw", .{stats.shards_written});
        defer a.free(sname);
        const spath = try std.fs.path.join(a, &.{ dst_dir, sname });
        defer a.free(spath);
        stats.bytes_out += try writeTfpnShard(a, spath, cur_metas.items, cur_scales.items, cur_words.items);
    }

    // Write BF16 residual tensors to model.safetensors
    stats.tensors_residual = residuals.items.len;
    if (residuals.items.len > 0) {
        const res_path = try std.fs.path.join(a, &.{ dst_dir, "model.safetensors" });
        defer a.free(res_path);
        try writeSafetensors(a, res_path, residuals.items);
        for (residuals.items) |r| stats.bytes_out += r.data.len;
    }

    // Write config.json with TFQ3 metadata
    {
        const src_cfg_path = try std.fs.path.join(a, &.{ src_dir, "config.json" });
        defer a.free(src_cfg_path);
        const cfg_raw = try st.mapFile(a, src_cfg_path);
        defer st.unmapFile(cfg_raw);
        const dst_cfg_path = try std.fs.path.join(a, &.{ dst_dir, "config.json" });
        defer a.free(dst_cfg_path);

        if (std.mem.indexOf(u8, cfg_raw, "\"quantization\"") == null) {
            if (std.mem.lastIndexOfScalar(u8, cfg_raw, '}')) |close_pos| {
                var new_cfg: std.ArrayList(u8) = .empty;
                defer new_cfg.deinit(a);
                try new_cfg.appendSlice(a, std.mem.trimEnd(u8, cfg_raw[0..close_pos], " \t\r\n"));
                const q_block = try std.fmt.allocPrint(a, ",\n  \"quantization\": {{\n    \"mode\": \"tfq3\",\n    \"bits\": {d}\n  }}\n}}\n", .{opts.bits});
                defer a.free(q_block);
                try new_cfg.appendSlice(a, q_block);
                try writeFileBytes(a, dst_cfg_path, new_cfg.items);
            } else {
                try writeFileBytes(a, dst_cfg_path, cfg_raw);
            }
        } else {
            try writeFileBytes(a, dst_cfg_path, cfg_raw);
        }
    }

    const aux_files = [_][]const u8{
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    };
    for (aux_files) |f| try copyFileIfExists(a, src_dir, dst_dir, f);

    stats.mean_snr_db = if (snr_cnt > 0) snr_sum / @as(f64, @floatFromInt(snr_cnt)) else 0.0;
    stats.elapsed_ms = nowMs() - t0;
    return stats;
}
