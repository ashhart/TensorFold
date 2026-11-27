//! BF16, Block-FP8 (`f8_e4m3` + 128x128 scale_inv), and MLX Q4/Q8 dequantization helpers for GLM CUDA TFQ3.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const weights = @import("cuda_weights.zig");
const quant = @import("cuda_quant.zig");
const projection = @import("cuda_proj.zig");
const glue = @import("cuda_glue.zig");

const X = cuda.experts.Nvfp4Experts;
const bf16Of = core.bf16_round.bits;

pub fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn uploadBuf(w: *weights.Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
    var buf = try cuda.DeviceBuffer.fromHost(d, bytes);
    errdefer buf.free();
    try w.buffers.append(w.gpa, buf);
    return buf.ptr;
}

pub fn uploadF32(w: *weights.Weights, d: *const cuda.Driver, t: core.checkpoint.Tensor) !u64 {
    if (t.dtype == .f32) return uploadBuf(w, d, t.bytes);
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try w.gpa.alloc(f32, t.numel());
    defer w.gpa.free(out);
    for (out, 0..) |*o, i| o.* = bf16f(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
    return uploadBuf(w, d, std.mem.sliceAsBytes(out));
}

pub fn uploadNorm(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, key: []const u8, hidden: usize) !glue.Norm {
    const t = try ck.get(key);
    if (t.numel() != hidden) return error.UnexpectedTensor;
    if (t.dtype == .f32) return .{ .f32 = try uploadBuf(w, d, t.bytes) };
    if (t.dtype == .bf16) return .{ .bf16 = try uploadBuf(w, d, t.bytes) };
    return error.UnexpectedTensor;
}

/// Exact IEEE FP8 E4M3FN -> f32 conversion (normal, subnormal, signed zero, NaN; max finite 448.0).
pub fn e4m3ToF32(code: u8) f32 {
    const sign: u32 = @as(u32, code >> 7) << 31;
    const exp: u32 = (code >> 3) & 0xf;
    const man: u32 = code & 0x7;
    if (exp == 0) {
        if (man == 0) return @bitCast(sign);
        const mag: f32 = @as(f32, @floatFromInt(man)) * 0x1p-9;
        return if (sign != 0) -mag else mag;
    }
    if (exp == 15 and man == 7) return std.math.nan(f32);
    return @bitCast(sign | ((exp + 120) << 23) | (man << 20));
}

pub const e4m3_lut: [256]f32 = blk: {
    @setEvalBranchQuota(4096);
    var lut: [256]f32 = undefined;
    for (0..256) |i| lut[i] = e4m3ToF32(@intCast(i));
    break :blk lut;
};

/// Reads 128x128 block-FP8 scale_inv (`<mod>.weight_scale_inv` or `<mod>.weight_scale`, F32/BF16/F16) as `[]f32`.
pub fn readBlockFp8Scales(gpa: std.mem.Allocator, ck: *core.Checkpoint, mod_path: []const u8, rows: usize, cols: usize) !?[]f32 {
    var buf: [240]u8 = undefined;
    const inv_name = try std.fmt.bufPrint(&buf, "{s}.weight_scale_inv", .{mod_path});
    const ts: core.checkpoint.Tensor = if (ck.has(inv_name))
        try ck.get(inv_name)
    else blk: {
        const sc_name = try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{mod_path});
        if (!ck.has(sc_name)) return null;
        break :blk try ck.get(sc_name);
    };
    if (ts.dtype != .f32 and ts.dtype != .bf16 and ts.dtype != .f16) return null;
    const rb = (rows + 127) / 128;
    const cb = (cols + 127) / 128;
    const shape_ok = switch (ts.rank) {
        2 => ts.shape[0] == rb and ts.shape[1] == cb,
        4 => ts.shape[0] == rb and ts.shape[1] == 1 and ts.shape[2] == cb and ts.shape[3] == 1,
        else => false,
    };
    if (!shape_ok or ts.numel() != rb * cb) return error.UnexpectedTensor;
    const scales = try gpa.alloc(f32, rb * cb);
    errdefer gpa.free(scales);
    switch (ts.dtype) {
        .f32 => {
            if (ts.bytes.len != scales.len * 4) return error.UnexpectedTensor;
            for (scales, 0..) |*sv, idx| sv.* = @bitCast(std.mem.readInt(u32, ts.bytes[4 * idx ..][0..4], .little));
        },
        .bf16 => {
            if (ts.bytes.len != scales.len * 2) return error.UnexpectedTensor;
            for (scales, 0..) |*sv, idx| sv.* = bf16f(std.mem.readInt(u16, ts.bytes[2 * idx ..][0..2], .little));
        },
        .f16 => {
            if (ts.bytes.len != scales.len * 2) return error.UnexpectedTensor;
            for (scales, 0..) |*sv, idx| {
                const bits = std.mem.readInt(u16, ts.bytes[2 * idx ..][0..2], .little);
                sv.* = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        else => unreachable,
    }
    return scales;
}

/// Dequantizes 128x128 block-FP8 (`codes [rows, cols]`, `scale_inv [ceil(rows/128), ceil(cols/128)]`) to BF16.
pub fn dequantBlockFp8ToBf16(codes: []const u8, scale_inv: []const f32, rows: usize, cols: usize, out: []u16) void {
    const cb = (cols + 127) / 128;
    for (0..rows) |r| {
        const srow = scale_inv[(r / 128) * cb ..][0..cb];
        const wrow = codes[r * cols ..][0..cols];
        const orow = out[r * cols ..][0..cols];
        for (0..cols) |c| {
            orow[c] = bf16Of(e4m3_lut[wrow[c]] * srow[c / 128]);
        }
    }
}

/// Dequantizes a BF16, Block-FP8, or MLX affine (4/8-bit, g32/g64) tensor into contiguous BF16 `[rows, cols]`.
pub fn dequantToBf16(gpa: std.mem.Allocator, ck: *core.Checkpoint, mod_path: []const u8, rows: usize, cols: usize) ![]u16 {
    var buf: [240]u8 = undefined;
    const tw = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{mod_path}));
    const out = try gpa.alloc(u16, rows * cols);
    errdefer gpa.free(out);
    if (tw.dtype == .bf16) {
        if (tw.numel() != rows * cols) return error.UnexpectedTensor;
        @memcpy(std.mem.sliceAsBytes(out), tw.bytes);
        return out;
    }
    if (tw.dtype == .f8_e4m3) {
        if (tw.numel() != rows * cols or tw.bytes.len != rows * cols) return error.UnexpectedTensor;
        const scales = (try readBlockFp8Scales(gpa, ck, mod_path, rows, cols)) orelse return error.UnexpectedTensor;
        defer gpa.free(scales);
        dequantBlockFp8ToBf16(tw.bytes, scales, rows, cols, out);
        return out;
    }
    if (tw.dtype != .u32) return error.UnexpectedTensor;
    const ts = try ck.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{mod_path}));
    const tb = try ck.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{mod_path}));
    if (ts.dtype != .bf16 or tb.dtype != .bf16 or ts.numel() != tb.numel()) return error.UnexpectedTensor;
    const w_cols = tw.numel() / rows;
    const groups = ts.numel() / rows;
    if (w_cols * rows != tw.numel() or groups * rows != ts.numel() or groups == 0 or cols % groups != 0) return error.UnexpectedTensor;
    const group_size = cols / groups;
    const bits = w_cols * 32 / cols;
    if ((bits != 4 and bits != 8) or w_cols != cols * bits / 32) return error.UnexpectedTensor;
    const per_word: usize = 32 / bits;
    const mask: u32 = if (bits == 8) 0xff else 0xf;
    for (0..rows) |r| {
        const wrow = tw.bytes[r * w_cols * 4 ..][0 .. w_cols * 4];
        const srow = ts.bytes[r * groups * 2 ..][0 .. groups * 2];
        const brow = tb.bytes[r * groups * 2 ..][0 .. groups * 2];
        for (0..cols) |c| {
            const word = std.mem.readInt(u32, wrow[(c / per_word) * 4 ..][0..4], .little);
            const code: f32 = @floatFromInt((word >> @intCast(bits * (c % per_word))) & mask);
            const g = c / group_size;
            const sv = bf16f(std.mem.readInt(u16, srow[2 * g ..][0..2], .little));
            const bv = bf16f(std.mem.readInt(u16, brow[2 * g ..][0..2], .little));
            out[r * cols + c] = bf16Of(code * sv + bv);
        }
    }
    return out;
}

/// Stacks dequantized BF16 tensors vertically and zero-pads trailing rows up to `padded_rows`.
pub fn stackDequantPadded(
    w: *weights.Weights,
    d: *const cuda.Driver,
    ck: *core.Checkpoint,
    mods: []const []const u8,
    row_counts: []const usize,
    padded_rows: usize,
    cols: usize,
) !u64 {
    var sum_rows: usize = 0;
    for (row_counts) |rn| sum_rows += rn;
    const total_rows = @max(sum_rows, padded_rows);
    const out = try w.gpa.alloc(u16, total_rows * cols);
    defer w.gpa.free(out);
    var at: usize = 0;
    for (mods, row_counts) |m, rn| {
        const part = try dequantToBf16(w.gpa, ck, m, rn, cols);
        defer w.gpa.free(part);
        @memcpy(out[at * cols ..][0 .. rn * cols], part);
        at += rn;
    }
    if (total_rows > sum_rows) @memset(out[sum_rows * cols ..], 0);
    return uploadBuf(w, d, std.mem.sliceAsBytes(out));
}

pub fn stackDequant(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, mods: []const []const u8, row_counts: []const usize, cols: usize) !u64 {
    return stackDequantPadded(w, d, ck, mods, row_counts, 0, cols);
}

/// Uploads a block-FP8 projection onto the FP8G lane when source is `f8_e4m3` + block scales and `tools.lane` fits.
pub fn tryLoadFp8g(
    w: *weights.Weights,
    d: *const cuda.Driver,
    ck: *core.Checkpoint,
    tools: weights.Tools,
    ck_mod: []const u8,
    rows: usize,
    cols: usize,
) !?cuda.qmmf.Weight {
    if (rows == 0 or cols == 0 or cols % cuda.fp8.group != 0) return null;
    const npad = cuda.fp8.padded(rows);
    if (rows > std.math.maxInt(c_int) or cols > std.math.maxInt(c_int) or npad > std.math.maxInt(c_int)) return null;
    var buf: [240]u8 = undefined;
    const w_name = try std.fmt.bufPrint(&buf, "{s}.weight", .{ck_mod});
    if (!ck.has(w_name)) return null;
    const tw = try ck.get(w_name);
    if (tw.dtype != .f8_e4m3) return null;
    if (tw.numel() != rows * cols or tw.bytes.len != rows * cols) return error.UnexpectedTensor;
    const stub = cuda.fp8.weight(0, 0, rows, cols);
    projection.fits(tools.lane, stub) catch return null;
    const scales = (try readBlockFp8Scales(w.gpa, ck, ck_mod, rows, cols)) orelse return null;
    defer w.gpa.free(scales);
    const packed_codes = try w.gpa.alloc(u8, npad * cols);
    defer w.gpa.free(packed_codes);
    cuda.fp8.packCodes(packed_codes, tw.bytes, rows, cols);
    const packed_scales = try w.gpa.alloc(f32, npad * (cols / cuda.fp8.group));
    defer w.gpa.free(packed_scales);
    cuda.fp8.packScales(packed_scales, scales, rows, cols);
    const c_ptr = try uploadBuf(w, d, packed_codes);
    const s_ptr = try uploadBuf(w, d, std.mem.sliceAsBytes(packed_scales));
    return cuda.fp8.weight(c_ptr, s_ptr, rows, cols);
}

/// Dequantizes MTP routed experts and repacks them to NVFP4 on GPU for CUDA speculative drafting.
pub fn loadMtpExperts(w: *weights.Weights, d: *const cuda.Driver, stream: cuda.Stream, tools: weights.Tools, ck: *core.Checkpoint, pfx: []const u8, i: u32) !cuda.experts.Layer {
    const qk = tools.quant orelse return error.NoQuantizer;
    const c = &w.config;
    const dim: usize = c.hidden;
    const ni: usize = c.moe_inter;
    const count: usize = c.experts;
    const up_len = X.bytes(1, ni, dim, 2) / 4;
    const down_len = X.bytes(1, dim, ni, 1) / 4;
    const up_words = try w.gpa.alloc(u32, count * up_len);
    defer w.gpa.free(up_words);
    const down_words = try w.gpa.alloc(u32, count * down_len);
    defer w.gpa.free(down_words);
    const up_scale = try w.gpa.alloc(f32, count * 2);
    defer w.gpa.free(up_scale);
    const down_scale = try w.gpa.alloc(f32, count);
    defer w.gpa.free(down_scale);
    const chunk: usize = 8;
    var wb = try cuda.DeviceBuffer.alloc(d, chunk * ni * dim * 2);
    defer wb.free();
    var cb = try cuda.DeviceBuffer.alloc(d, chunk * ni * dim / 2);
    defer cb.free();
    var sb = try cuda.DeviceBuffer.alloc(d, chunk * ni * dim / 16);
    defer sb.free();
    var gb = try cuda.DeviceBuffer.alloc(d, chunk * 4);
    defer gb.free();
    const codes = try w.gpa.alloc(u8, 3 * chunk * ni * dim / 2);
    defer w.gpa.free(codes);
    const scales = try w.gpa.alloc(u8, 3 * chunk * ni * dim / 16);
    defer w.gpa.free(scales);
    const stage = try w.gpa.alloc(u16, chunk * ni * dim);
    defer w.gpa.free(stage);
    var g: [3][chunk]f32 = undefined;
    var nb: [192]u8 = undefined;
    var e0: usize = 0;
    while (e0 < count) : (e0 += chunk) {
        const ec = @min(chunk, count - e0);
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
            const n = if (m < 2) ni else dim;
            const k = if (m < 2) dim else ni;
            for (0..ec) |j| {
                const mp = try std.fmt.bufPrint(&nb, "{s}layers.{d}.mlp.experts.{d}." ++ proj, .{ pfx, i, e0 + j });
                const rows = try dequantToBf16(w.gpa, ck, mp, n, k);
                defer w.gpa.free(rows);
                @memcpy(stage[j * n * k ..][0 .. n * k], rows);
            }
            try wb.upload(0, std.mem.sliceAsBytes(stage[0 .. ec * n * k]));
            try quant.nvfp4(qk, stream, wb.ptr, ec, n, k, cb.ptr, sb.ptr, gb.ptr);
            try stream.synchronize();
            try cb.download(0, codes[m * chunk * n * k / 2 ..][0 .. ec * n * k / 2]);
            try sb.download(0, scales[m * chunk * n * k / 16 ..][0 .. ec * n * k / 16]);
            try gb.download(0, std.mem.sliceAsBytes(g[m][0..ec]));
        }
        const mc = chunk * ni * dim / 2;
        for (0..ec) |j| {
            const ei = e0 + j;
            const one = ni * dim;
            var cs: [3][]const u8 = undefined;
            var ss: [3][]const u8 = undefined;
            for (0..3) |m| {
                cs[m] = codes[m * mc + j * one / 2 ..][0 .. one / 2];
                ss[m] = scales[m * mc / 8 + j * one / 16 ..][0 .. one / 16];
            }
            X.packGateUp(up_words[ei * up_len ..][0..up_len], .{ cs[0], ss[0] }, .{ cs[1], ss[1] }, ni, dim);
            X.packOne(down_words[ei * down_len ..][0..down_len], cs[2], ss[2], dim, ni);
            up_scale[ei * 2] = g[0][j];
            up_scale[ei * 2 + 1] = g[1][j];
            down_scale[ei] = g[2][j];
        }
    }
    return .{
        .format = .nvfp4,
        .up = try uploadBuf(w, d, std.mem.sliceAsBytes(up_words)),
        .down = try uploadBuf(w, d, std.mem.sliceAsBytes(down_words)),
        .up_scale = try uploadBuf(w, d, std.mem.sliceAsBytes(up_scale)),
        .down_scale = try uploadBuf(w, d, std.mem.sliceAsBytes(down_scale)),
        .width = ni,
        .dims = dim,
        .experts = count,
        .limit = c.swiglu_limit,
    };
}
