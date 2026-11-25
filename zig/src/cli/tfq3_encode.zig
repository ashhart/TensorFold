//! Pure-Zig zero-install TFQ3 encoder: 2D FWHT-128, PWV3 v3 tables, TFPN v3 shards, and residuals.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const magic_pwv3: u32 = 0x33565750;
pub const magic_tfpn: u32 = 0x4E504654;
pub const format_ver: u32 = 3;
pub const align_16k: usize = 16384;

pub const flag_padded: u32 = 1 << 0;
pub const flag_vision: u32 = 1 << 1;
pub const flag_drafter: u32 = 1 << 2;
pub const flag_embed: u32 = 1 << 3;

pub const EncodeOptions = struct {
    src_dir: []const u8,
    out_dir: []const u8,
    bits: u32 = 4,
    head_bits: u32 = 4,
    target_gib: f64 = 0.0,
    quantize_vision: bool = false,
    quantize_drafter: bool = false,
    quantize_embed: bool = false,
};

pub const PwV3Tables = extern struct {
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
    r: u32,
    c: u32,
    b_bits: u32,
    sc_off: u32,
    word_off: u64,
    word_count: u64,
    snr_db: f32,
    pad: [3]u32,
};

comptime {
    if (@sizeOf(PwV3Tables) != 16384) @compileError("PwV3Tables must be 16384 bytes");
    if (@sizeOf(TfpnHeader) != 64) @compileError("TfpnHeader must be 64 bytes");
    if (@sizeOf(TfpnTensorMeta) != 128) @compileError("TfpnTensorMeta must be 128 bytes");
}

fn align128(x: u32) u32 {
    return (x + 127) & ~@as(u32, 127);
}
fn align16k(x: usize) usize {
    return (x + align_16k - 1) & ~(align_16k - 1);
}
fn f32ToF16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}
fn f16BitsToF32(u: u16) f32 {
    return @floatCast(@as(f16, @bitCast(u)));
}
fn packH2(x: f32, y: f32) u32 {
    return @as(u32, f32ToF16Bits(x)) | (@as(u32, f32ToF16Bits(y)) << 16);
}
fn bf16ToF32(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}
fn sgnBit(seed: u32, i: u32) f32 {
    return if (((((seed ^ i) *% 2654435761) >> 31) & 1) != 0) -1.0 else 1.0;
}
fn fragHome(lane: u32) struct { x: u32, y: u32 } {
    return .{ .x = (lane & 8) + ((lane & 1) << 2), .y = ((lane & 16) >> 2) | ((lane >> 1) & 3) };
}

fn invNormCdf(p: f64) f64 {
    const a = [_]f64{ -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
    const b = [_]f64{ -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
    const c = [_]f64{ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
    const d = [_]f64{ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
    const plow = 0.02425;
    if (p < plow) {
        const q = @sqrt(-2.0 * @log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p <= 1.0 - plow) {
        const q = p - 0.5;
        const r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    const q = @sqrt(-2.0 * @log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
}

fn lloydMax1d(g: usize, out: *[32]f32) void {
    var samples: [4096]f64 = undefined;
    for (0..4096) |i| samples[i] = invNormCdf((@as(f64, @floatFromInt(i)) + 0.5) / 4096.0);
    var c: [32]f64 = undefined;
    for (0..g) |i| c[i] = invNormCdf((@as(f64, @floatFromInt(i)) + 0.5) / @as(f64, @floatFromInt(g)));
    for (0..24) |_| {
        var sum: [32]f64 = @splat(0.0);
        var cnt: [32]usize = @splat(0);
        var j: usize = 0;
        for (samples) |s| {
            while (j + 1 < g and s > 0.5 * (c[j] + c[j + 1])) j += 1;
            sum[j] += s;
            cnt[j] += 1;
        }
        for (0..g) |i| if (cnt[i] > 0) {
            c[i] = sum[i] / @as(f64, @floatFromInt(cnt[i]));
        };
    }
    for (0..g) |i| out[i] = @floatCast(c[i]);
}

fn nearest1d(mids: *const [31]f32, g: usize, v: f32) u32 {
    var lo: usize = 0;
    var hi: usize = g - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (v > mids[mid]) lo = mid + 1 else hi = mid;
    }
    return @intCast(lo);
}

fn fillGrid2d(dst: []u32, stride: usize, offset: usize, g: usize, cent: *const [32]f32) void {
    for (0..(g * g)) |i| dst[i * stride + offset] = packH2(cent[i % g], cent[i / g]);
}

/// Generates a valid deterministic 16 KiB PWV3 v3 Max-Lloyd Gaussian codebook table.
pub fn generateTables() PwV3Tables {
    var t: PwV3Tables = std.mem.zeroes(PwV3Tables);
    t.magic = magic_pwv3;
    t.version = format_ver;
    var c8: [32]f32 = undefined;
    var c16: [32]f32 = undefined;
    var c32: [32]f32 = undefined;
    lloydMax1d(8, &c8);
    lloydMax1d(16, &c16);
    lloydMax1d(32, &c32);
    for (0..4) |c| fillGrid2d(&t.lut3, 4, c, 8, &c8);
    for (0..4) |c| fillGrid2d(&t.lut4, 4, c, 16, &c16);
    fillGrid2d(&t.lut5, 1, 0, 32, &c32);
    const id_c = [4]u16{ f32ToF16Bits(1.0), f32ToF16Bits(1.0), 0, 0 };
    for (0..64) |c| {
        t.c64_3[c] = id_c;
        t.c64_4[c] = id_c;
        t.c64_5[c] = id_c;
    }
    return t;
}

/// In-place normalized FWHT-128.
pub fn fwht128(v: *[128]f32) void {
    var len: usize = 1;
    while (len < 128) : (len <<= 1) {
        var i: usize = 0;
        while (i < 128) : (i += len * 2) {
            for (0..len) |j| {
                const a = v[i + j];
                const b = v[i + j + len];
                v[i + j] = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
    for (v) |*x| x.* *= 0.08838834764831845;
}

fn rotate2d(w: []f32, r: u32, c: u32) void {
    var buf: [128]f32 = undefined;
    for (0..r) |ri| for (0..(c / 128)) |bc| {
        const c0 = bc * 128;
        for (0..128) |j| buf[j] = w[ri * c + c0 + j] * sgnBit(0x12345678, @intCast(c0 + j));
        fwht128(&buf);
        @memcpy(w[ri * c + c0 .. ri * c + c0 + 128], &buf);
    };
    for (0..(r / 128)) |br| {
        const r0 = br * 128;
        for (0..c) |ci| {
            for (0..128) |j| buf[j] = w[(r0 + j) * c + ci] * sgnBit(0x9E3779B9, @intCast(r0 + j));
            fwht128(&buf);
            for (0..128) |j| w[(r0 + j) * c + ci] = buf[j];
        }
    }
}

pub const EncodedMatrix = struct { scales: []u16, words: []u32, snr_db: f32 };

/// Encodes a padded `[r, c]` FP32 weight matrix (multiples of 128) into canonical TFPN v3 layout.
pub fn encodeMatrix(a: Allocator, tbl: *const PwV3Tables, w_in: []const f32, r: u32, c: u32, bits: u32) !EncodedMatrix {
    const w_rot = try a.dupe(f32, w_in);
    defer a.free(w_rot);
    rotate2d(w_rot, r, c);

    const n_br = r / 128;
    const n_bc = c / 128;
    const wpb = bits * 2;
    const scales = try a.alloc(u16, n_br * n_bc);
    const words = try a.alloc(u32, @as(usize, r / 16) * n_bc * wpb * 32);
    @memset(words, 0);

    for (0..n_br) |br| for (0..n_bc) |bc| {
        var ss: f64 = 0.0;
        for (0..128) |dr| {
            for (w_rot[(br * 128 + dr) * c + bc * 128 ..][0..128]) |v| ss += @as(f64, v) * @as(f64, v);
        }
        scales[br * n_bc + bc] = f32ToF16Bits(@floatCast(@sqrt(ss / 16384.0 + 1e-12)));
    };

    const lut_slice: []const u32 = if (bits == 3) &tbl.lut3 else if (bits == 4) &tbl.lut4 else &tbl.lut5;
    const c64_slice: *const [64][4]u16 = if (bits == 3) &tbl.c64_3 else if (bits == 4) &tbl.c64_4 else &tbl.c64_5;
    const g: usize = @as(usize, 1) << @intCast(bits);
    const n_sym: u32 = @as(u32, 1) << @intCast(2 * bits);
    const c_shift: u5 = if (bits == 5) 0 else 2;
    const h_shift: u5 = if (bits == 3) 24 else 22;
    const sym_mask: u32 = n_sym - 1;

    var cent: [32]f32 = @splat(0.0);
    var mids: [31]f32 = @splat(0.0);
    for (0..g) |i| cent[i] = f16BitsToF32(@truncate(lut_slice[i << c_shift]));
    for (0..(g - 1)) |i| mids[i] = 0.5 * (cent[i] + cent[i + 1]);

    var tot_sig: f64 = 0.0;
    var tot_err: f64 = 0.0;
    for (0..(r / 16)) |strip| {
        const br = strip >> 3;
        for (0..32) |lane_u| {
            const lane: u32 = @intCast(lane_u);
            const home = fragHome(lane);
            var st: u32 = 0x13579BDF ^ (lane *% 0x9E3779B1);
            for (0..n_bc) |bc| {
                const sc = f16BitsToF32(scales[br * n_bc + bc]);
                const inv_sc = if (sc > 0.0) 1.0 / sc else 0.0;
                var wbuf: [10]u32 = @splat(0);
                var bit_pos: u32 = 0;
                for (0..8) |kt| {
                    const cc0 = bc * 128 + kt * 16 + home.x;
                    const r0 = strip * 16 + home.y;
                    const p0 = w_rot[r0 * c + cc0 ..][0..4];
                    const p1 = w_rot[(r0 + 8) * c + cc0 ..][0..4];
                    const pairs = [4][2]f32{ .{ p0[0], p0[1] }, .{ p0[2], p0[3] }, .{ p1[0], p1[1] }, .{ p1[2], p1[3] } };
                    var h_even: u32 = 0;
                    for (pairs, 0..) |pr, pi| {
                        const x0 = pr[0] * inv_sc;
                        const y0 = pr[1] * inv_sc;
                        const h = st *% 0x9E3779B1;
                        if ((pi & 1) == 0) h_even = h;
                        const cf = c64_slice[((if ((pi & 1) != 0) h_even else h) >> 16) & 63];
                        const sx = f16BitsToF32(cf[0]);
                        const sy = f16BitsToF32(cf[1]);
                        const tx = f16BitsToF32(cf[2]);
                        const ty = f16BitsToF32(cf[3]);
                        const sym_xor = ((h >> h_shift) >> c_shift) & sym_mask;

                        const ix = nearest1d(&mids, g, (x0 - tx) / sx);
                        const iy = nearest1d(&mids, g, (y0 - ty) / sy);
                        const best_sub: u32 = (iy << @intCast(bits)) | ix;
                        const best_qx = cent[ix] * sx + tx;
                        const best_qy = cent[iy] * sy + ty;
                        const sym = best_sub ^ sym_xor;
                        st = (st << @intCast(2 * bits)) | sym;
                        const ex = pr[0] - best_qx * sc;
                        const ey = pr[1] - best_qy * sc;
                        tot_sig += @as(f64, pr[0]) * @as(f64, pr[0]) + @as(f64, pr[1]) * @as(f64, pr[1]);
                        tot_err += @as(f64, ex) * @as(f64, ex) + @as(f64, ey) * @as(f64, ey);

                        var k: i32 = @intCast(2 * bits - 1);
                        while (k >= 0) : (k -= 1) {
                            if (((sym >> @intCast(k)) & 1) != 0) {
                                wbuf[bit_pos >> 5] |= (@as(u32, 1) << @intCast(31 - (bit_pos & 31)));
                            }
                            bit_pos += 1;
                        }
                    }
                }
                const base = (strip * n_bc + bc) * wpb * 32;
                if (bits == 4) {
                    for (0..4) |k| words[base + lane * 4 + k] = wbuf[k];
                    for (0..4) |k| words[base + 128 + lane * 4 + k] = wbuf[k + 4];
                } else {
                    for (0..wpb) |wi| words[base + wi * 32 + lane] = wbuf[wi];
                }
            }
        }
    }
    const snr: f32 = if (tot_sig > 0.0) @floatCast(10.0 * std.math.log10(tot_sig / @max(tot_err, 1e-18))) else 0.0;
    return .{ .scales = scales, .words = words, .snr_db = snr };
}

const StTensor = struct { name: []const u8, file: []const u8, dtype: []const u8, shape: []u64, off: u64, len: u64 };

fn parseStFile(a: Allocator, io: std.Io, dir: []const u8, fn_name: []const u8, list: *std.ArrayList(StTensor)) !void {
    const f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, fn_name }), .{});
    defer f.close(io);
    var lb: [8]u8 = undefined;
    if ((try f.readPositionalAll(io, &lb, 0)) < 8) return error.InvalidSafetensors;
    const hlen = std.mem.readInt(u64, &lb, .little);
    if (hlen == 0 or hlen > (64 << 20)) return error.InvalidSafetensors;
    const hbytes = try a.alloc(u8, @intCast(hlen));
    if ((try f.readPositionalAll(io, hbytes, 8)) < hlen) return error.InvalidSafetensors;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, hbytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensors;
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__") or kv.value_ptr.* != .object) continue;
        const obj = kv.value_ptr.object;
        const dt = obj.get("dtype") orelse continue;
        const sh = obj.get("shape") orelse continue;
        const dof = obj.get("data_offsets") orelse continue;
        if (dt != .string or sh != .array or dof != .array or dof.array.items.len != 2) continue;
        if (dof.array.items[0] != .integer or dof.array.items[1] != .integer) continue;
        const o0: u64 = @intCast(dof.array.items[0].integer);
        const o1: u64 = @intCast(dof.array.items[1].integer);
        const shape = try a.alloc(u64, sh.array.items.len);
        for (sh.array.items, 0..) |sv, i| {
            if (sv != .integer or sv.integer < 0) return error.InvalidSafetensors;
            shape[i] = @intCast(sv.integer);
        }
        try list.append(a, .{
            .name = try a.dupe(u8, kv.key_ptr.*),
            .file = try a.dupe(u8, fn_name),
            .dtype = try a.dupe(u8, dt.string),
            .shape = shape,
            .off = 8 + hlen + o0,
            .len = o1 - o0,
        });
    }
}

fn readTensorBytes(a: Allocator, io: std.Io, dir: []const u8, t: StTensor) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, t.file }), .{});
    defer f.close(io);
    const buf = try a.alloc(u8, @intCast(t.len));
    if ((try f.readPositionalAll(io, buf, t.off)) < t.len) return error.ShortRead;
    return buf;
}

fn decodeToPaddedF32(a: Allocator, raw: []const u8, dtype: []const u8, orig_r: u32, orig_c: u32, pad_r: u32, pad_c: u32) ![]f32 {
    const out = try a.alloc(f32, @as(usize, pad_r) * pad_c);
    @memset(out, 0.0);
    if (std.mem.eql(u8, dtype, "BF16")) {
        for (0..orig_r) |ri| for (0..orig_c) |ci| {
            out[ri * pad_c + ci] = bf16ToF32(std.mem.readInt(u16, raw[(ri * orig_c + ci) * 2 ..][0..2], .little));
        };
    } else if (std.mem.eql(u8, dtype, "F16")) {
        for (0..orig_r) |ri| for (0..orig_c) |ci| {
            out[ri * pad_c + ci] = f16BitsToF32(std.mem.readInt(u16, raw[(ri * orig_c + ci) * 2 ..][0..2], .little));
        };
    } else if (std.mem.eql(u8, dtype, "F32")) {
        for (0..orig_r) |ri| for (0..orig_c) |ci| {
            out[ri * pad_c + ci] = @bitCast(std.mem.readInt(u32, raw[(ri * orig_c + ci) * 4 ..][0..4], .little));
        };
    } else return error.UnsupportedDtype;
    return out;
}

fn tfpnBytes(pad_r: u32, pad_c: u32, bits: u32) u64 {
    return @as(u64, pad_r / 16) * (pad_c / 128) * (bits * 2) * 32 * 4 + @as(u64, pad_r / 128) * (pad_c / 128) * 2;
}

fn hasSub(s: []const u8, sub: []const u8) bool {
    return std.mem.indexOf(u8, s, sub) != null;
}

/// Converts a Safetensors directory into a complete TFQ3 checkpoint using pure Zig (no external tool).
pub fn convertDir(a: Allocator, io: std.Io, out: *std.Io.Writer, err: *std.Io.Writer, opt: EncodeOptions) !u8 {
    var d = std.Io.Dir.cwd().openDir(io, opt.src_dir, .{ .iterate = true }) catch {
        try err.print("tensorfold tfq3: cannot open {s}\n", .{opt.src_dir});
        return 1;
    };
    defer d.close(io);

    var st_files: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if ((entry.kind == .file or entry.kind == .sym_link) and std.mem.endsWith(u8, entry.name, ".safetensors")) {
            try st_files.append(a, try a.dupe(u8, entry.name));
        }
    }
    if (st_files.items.len == 0) {
        try err.print("tensorfold tfq3: no .safetensors files in {s}\n", .{opt.src_dir});
        return 1;
    }
    std.mem.sort([]const u8, st_files.items, {}, struct {
        fn less(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.less);

    var all_tensors: std.ArrayList(StTensor) = .empty;
    for (st_files.items) |fn_name| parseStFile(a, io, opt.src_dir, fn_name, &all_tensors) catch |e| {
        try err.print("tensorfold tfq3: failed reading {s} ({t})\n", .{ fn_name, e });
        return 1;
    };

    const Job = struct { t: StTensor, tfpn_name: []const u8, orig_r: u32, orig_c: u32, pad_r: u32, pad_c: u32, bits: u32, flags: u32, prio: u32 };
    var jobs: std.ArrayList(Job) = .empty;
    var residuals: std.ArrayList(StTensor) = .empty;
    var res_bytes: u64 = 0;

    for (all_tensors.items) |t| {
        const is_float = std.mem.eql(u8, t.dtype, "BF16") or std.mem.eql(u8, t.dtype, "F16") or std.mem.eql(u8, t.dtype, "F32");
        const is_w = std.mem.endsWith(u8, t.name, ".weight") and t.shape.len == 2 and is_float;
        const is_embed = hasSub(t.name, "embed_tokens") or hasSub(t.name, "word_embeddings");
        const is_vis = hasSub(t.name, "visual.") or hasSub(t.name, "vision_tower.");
        const is_head = std.mem.eql(u8, t.name, "lm_head.weight") or hasSub(t.name, ".lm_head.weight");
        const quantize = is_w and t.shape[0] >= 128 and t.shape[1] >= 128 and
            (!is_embed or opt.quantize_embed) and (!is_vis or opt.quantize_vision) and (!is_head or opt.head_bits > 0);
        if (!quantize) {
            try residuals.append(a, t);
            res_bytes += t.len;
            continue;
        }
        const orig_r: u32 = @intCast(t.shape[0]);
        const orig_c: u32 = @intCast(t.shape[1]);
        const pad_r = align128(orig_r);
        const pad_c = align128(orig_c);
        var flags: u32 = 0;
        if (pad_r != orig_r or pad_c != orig_c) flags |= flag_padded;
        if (is_vis) flags |= flag_vision;
        if (opt.quantize_drafter) flags |= flag_drafter;
        if (is_embed) flags |= flag_embed;
        const prio: u32 = if (is_head) 100 else if (hasSub(t.name, "attn") or hasSub(t.name, "_proj")) 70 else 40;
        const stem = t.name[0 .. t.name.len - 7];
        const short = if (stem.len >= 80) stem[stem.len - 79 ..] else stem;
        try jobs.append(a, .{
            .t = t,
            .tfpn_name = short,
            .orig_r = orig_r,
            .orig_c = orig_c,
            .pad_r = pad_r,
            .pad_c = pad_c,
            .bits = if (is_head) opt.head_bits else opt.bits,
            .flags = flags,
            .prio = prio,
        });
    }
    if (jobs.items.len == 0) {
        try err.print("tensorfold tfq3: no quantizable 2D >=128x128 float tensors in {s}\n", .{opt.src_dir});
        return 1;
    }

    if (opt.target_gib > 0.0) {
        const target_bytes: u64 = @intFromFloat(opt.target_gib * @as(f64, @floatFromInt(@as(u64, 1) << 30)));
        const budget = if (target_bytes > res_bytes + 65536) target_bytes - res_bytes - 65536 else 0;
        var cur: u64 = 0;
        for (jobs.items) |*j| {
            j.bits = 3;
            cur += tfpnBytes(j.pad_r, j.pad_c, 3);
        }
        for ([_]u32{ 4, 5 }) |next_b| {
            for ([_]u32{ 100, 70, 40 }) |tier| {
                for (jobs.items) |*j| {
                    if (j.prio != tier or j.bits != next_b - 1) continue;
                    const delta = tfpnBytes(j.pad_r, j.pad_c, next_b) - tfpnBytes(j.pad_r, j.pad_c, next_b - 1);
                    if (cur + delta <= budget) {
                        j.bits = next_b;
                        cur += delta;
                    }
                }
            }
        }
    }

    try std.Io.Dir.cwd().createDirPath(io, opt.out_dir);
    const tbl = generateTables();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ opt.out_dir, "pw_v3_tables.bin" }), .data = std.mem.asBytes(&tbl) });

    var metas: std.ArrayList(TfpnTensorMeta) = .empty;
    var all_scales: std.ArrayList(u16) = .empty;
    var all_words: std.ArrayList(u32) = .empty;

    for (jobs.items) |j| {
        const raw = try readTensorBytes(a, io, opt.src_dir, j.t);
        defer a.free(raw);
        const f32_w = try decodeToPaddedF32(a, raw, j.t.dtype, j.orig_r, j.orig_c, j.pad_r, j.pad_c);
        defer a.free(f32_w);
        const enc = try encodeMatrix(a, &tbl, f32_w, j.pad_r, j.pad_c, j.bits);
        defer a.free(enc.scales);
        defer a.free(enc.words);

        var m: TfpnTensorMeta = std.mem.zeroes(TfpnTensorMeta);
        @memcpy(m.name[0..j.tfpn_name.len], j.tfpn_name);
        m.r = j.pad_r;
        m.c = j.pad_c;
        m.b_bits = j.bits;
        m.sc_off = @intCast(all_scales.items.len);
        m.word_off = all_words.items.len;
        m.word_count = enc.words.len;
        m.snr_db = enc.snr_db;
        m.pad = .{ j.orig_r, j.orig_c, j.flags };
        try metas.append(a, m);
        try all_scales.appendSlice(a, enc.scales);
        try all_words.appendSlice(a, enc.words);
    }

    try writeTfpnShard(a, io, try std.fs.path.join(a, &.{ opt.out_dir, "tfq3-00001-of-00001.tffpw" }), metas.items, all_scales.items, all_words.items);
    if (residuals.items.len > 0) try writeResidualSafetensors(a, io, opt.src_dir, opt.out_dir, residuals.items);
    for ([_][]const u8{ "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json" }) |cf| {
        if (std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ opt.src_dir, cf }), a, .limited(64 << 20))) |bytes| {
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ opt.out_dir, cf }), .data = bytes });
        } else |_| {}
    }
    try out.print("native TFQ3 encoded {d} tensor(s) + {d} residual(s) -> {s}\n", .{ metas.items.len, residuals.items.len, opt.out_dir });
    return 0;
}

fn writeTfpnShard(a: Allocator, io: std.Io, path: []const u8, metas: []const TfpnTensorMeta, scales: []const u16, words: []const u32) !void {
    const off_meta: usize = @sizeOf(TfpnHeader);
    const off_scales: usize = align16k(off_meta + metas.len * @sizeOf(TfpnTensorMeta));
    const off_weights: usize = align16k(off_scales + scales.len * 2);
    const total_size: usize = align16k(off_weights + words.len * 4);
    const buf = try a.alloc(u8, total_size);
    defer a.free(buf);
    @memset(buf, 0);
    const hdr: TfpnHeader = .{
        .magic = magic_tfpn,
        .version = format_ver,
        .num_tensors = @intCast(metas.len),
        .pad0 = 0,
        .off_meta = off_meta,
        .off_scales = off_scales,
        .off_weights = off_weights,
        .total_scale_halfs = scales.len,
        .total_weight_words = words.len,
        .pad1 = 0,
    };
    @memcpy(buf[0..@sizeOf(TfpnHeader)], std.mem.asBytes(&hdr));
    @memcpy(buf[off_meta .. off_meta + metas.len * @sizeOf(TfpnTensorMeta)], std.mem.sliceAsBytes(metas));
    @memcpy(buf[off_scales .. off_scales + scales.len * 2], std.mem.sliceAsBytes(scales));
    @memcpy(buf[off_weights .. off_weights + words.len * 4], std.mem.sliceAsBytes(words));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf });
}

fn writeResidualSafetensors(a: Allocator, io: std.Io, src_dir: []const u8, out_dir: []const u8, list: []const StTensor) !void {
    var hdr: std.ArrayList(u8) = .empty;
    try hdr.appendSlice(a, "{\"__metadata__\":{\"format\":\"tfq3\"}");
    var off: u64 = 0;
    for (list) |t| {
        try hdr.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        for (t.shape, 0..) |dim, i| {
            if (i > 0) try hdr.append(a, ',');
            try hdr.print(a, "{d}", .{dim});
        }
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.len });
        off += t.len;
    }
    try hdr.append(a, '}');
    while ((hdr.items.len & 7) != 0) try hdr.append(a, ' ');

    const f = try std.Io.Dir.cwd().createFile(io, try std.fs.path.join(a, &.{ out_dir, "residual.safetensors" }), .{});
    defer f.close(io);
    var hb: [8]u8 = undefined;
    std.mem.writeInt(u64, &hb, hdr.items.len, .little);
    try f.writeStreamingAll(io, &hb);
    try f.writeStreamingAll(io, hdr.items);
    for (list) |t| {
        const bytes = try readTensorBytes(a, io, src_dir, t);
        defer a.free(bytes);
        try f.writeStreamingAll(io, bytes);
    }
}
