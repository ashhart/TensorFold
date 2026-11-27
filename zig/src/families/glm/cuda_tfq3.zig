//! GLM TFQ3 (3/4/5-bit 2D-FWHT) CUDA loader: dense, MoE, BF16/FP8/MLX residuals, MTP, and DFlash2.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const b16 = @import("cuda_b16.zig");
const projection = @import("cuda_proj.zig");
const glue = @import("cuda_glue.zig");

const X = cuda.experts.Nvfp4Experts;
const bf16Of = core.bf16_round.bits;

/// Draft vocabulary projection shared by MTP and DFlash2: Q4 (NVFP4/EXL3), TFQ3, or BF16 lm_head (zero extra VRAM).
pub const DraftHead = union(enum) {
    q4: q4.Q4,
    tfq3: cuda.tfq3.DenseWeight,
    bf16: struct { w: u64, n: usize, k: usize },

    pub fn outputs(h: DraftHead) usize {
        return switch (h) {
            .q4 => |q| q.n,
            .tfq3 => |w| w.n,
            .bf16 => |b| b.n,
        };
    }

    pub fn project(h: DraftHead, set: glue.Set, q4k: *const q4.Kernels, tfq3k: ?*const cuda.tfq3.Kernels, stream: cuda.Stream, x: u64, xs: u64, out: u64, rows: usize) !void {
        switch (h) {
            .q4 => |q| {
                var r0: usize = 0;
                while (r0 < rows) : (r0 += 16) {
                    const take = @min(16, rows - r0);
                    try q4.matmul(q4k, stream, x + r0 * q.k * 2, q.k, xs + r0 * (q.k / 64) * 4, q, out + r0 * q.n * 2, take);
                }
            },
            .tfq3 => |w| {
                const tk = tfq3k orelse return error.FormatNotLoaded;
                try tk.dense(stream, w, x, w.k, out, w.n, rows, false);
            },
            .bf16 => |b| {
                try b16.matmul(set, stream, x, b.k, b.w, out, 0, rows, b.n, b.k, false);
            },
        }
    }
};

/// Returns true if `dir` has `pw_v3_tables.bin` and a valid `nonmoe.tffpw` (`TFPN` v3) shard with records.
pub fn hasNonMoe(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    return cuda.tfq3_format.hasTables(gpa, io, dir) and
        cuda.tfq3_format.hasRecords(gpa, io, dir, "nonmoe.tffpw", cuda.tfq3_format.magic_tfpn);
}

/// Returns true if `dir` is a GLM TFQ3 checkpoint (`pw_v3_tables.bin` plus `nonmoe.tffpw` or any `layer_*.tffpw`).
pub fn isTfq3(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    return cuda.tfq3_format.hasTables(gpa, io, dir) and
        (cuda.tfq3_format.hasRecords(gpa, io, dir, "nonmoe.tffpw", cuda.tfq3_format.magic_tfpn) or
            cuda.tfq3_format.hasAnyTfpw(gpa, io, dir));
}

/// Device buffers owning `pw_v3_tables.bin`, optional `nonmoe.tffpw`, and `layer_XX.tffpw` shards.
pub const Store = struct {
    tables: cuda.DeviceBuffer,
    nonmoe: ?cuda.tfq3.TfpnStore = null,
    layers: std.ArrayList(cuda.tfq3.OwnedMoeLayer) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Store {
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        var tables = try cuda.tfq3.uploadTables(gpa, io, d, dir);
        errdefer tables.free();
        var nonmoe: ?cuda.tfq3.TfpnStore = null;
        if (hasNonMoe(gpa, io, dir)) {
            nonmoe = try cuda.tfq3.TfpnStore.loadFileWithTables(gpa, io, d, dir, "nonmoe.tffpw", tables.ptr);
        }
        s.* = .{ .tables = tables, .nonmoe = nonmoe };
        return s;
    }

    pub fn findOptional(s: *const Store, name: []const u8, n: usize, k: usize) !?cuda.tfq3.DenseWeight {
        const nm = if (s.nonmoe) |*store| store else return null;
        const w = nm.find(name) orelse return null;
        if (w.n != n or w.k != k) return error.Tfq3Shape;
        return w;
    }

    pub fn loadMoeLayer(s: *Store, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, i: u32, experts: u32, hidden: u32, inter: u32) !cuda.tfq3.MoeLayer {
        var name_buf: [32]u8 = undefined;
        const fname = try std.fmt.bufPrint(&name_buf, "layer_{d:0>2}.tffpw", .{i});
        const path = try std.fs.path.join(gpa, &.{ dir, fname });
        defer gpa.free(path);
        var owned = try cuda.tfq3.uploadTfpwLayer(io, d, path, s.tables.ptr, experts, hidden, inter, i);
        errdefer owned.deinit();
        try s.layers.append(gpa, owned);
        return owned.layer;
    }

    pub fn deinit(s: *Store, gpa: std.mem.Allocator) void {
        for (s.layers.items) |*l| l.deinit();
        s.layers.deinit(gpa);
        if (s.nonmoe) |*nm| nm.deinit();
        s.tables.free();
        gpa.destroy(s);
    }
};

fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn uploadBuf(w: *weights.Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
    var buf = try cuda.DeviceBuffer.fromHost(d, bytes);
    errdefer buf.free();
    try w.buffers.append(w.gpa, buf);
    return buf.ptr;
}

fn uploadF32(w: *weights.Weights, d: *const cuda.Driver, t: core.checkpoint.Tensor) !u64 {
    if (t.dtype == .f32) return uploadBuf(w, d, t.bytes);
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try w.gpa.alloc(f32, t.numel());
    defer w.gpa.free(out);
    for (out, 0..) |*o, i| o.* = bf16f(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
    return uploadBuf(w, d, std.mem.sliceAsBytes(out));
}

fn checkedRouterWeight(t: core.checkpoint.Tensor, experts: usize, hidden: usize) !core.checkpoint.Tensor {
    if (!t.is(.bf16, &.{ experts, hidden })) return error.UnexpectedTensor;
    return t;
}

fn checkedRouterBias(t: core.checkpoint.Tensor, experts: usize) !core.checkpoint.Tensor {
    if ((t.dtype != .f32 and t.dtype != .bf16) or t.numel() != experts) return error.UnexpectedTensor;
    return t;
}

fn uploadNorm(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, key: []const u8, hidden: usize) !glue.Norm {
    const t = try ck.get(key);
    if (t.numel() != hidden) return error.UnexpectedTensor;
    if (t.dtype == .f32) return .{ .f32 = try uploadBuf(w, d, t.bytes) };
    if (t.dtype == .bf16) return .{ .bf16 = try uploadBuf(w, d, t.bytes) };
    return error.UnexpectedTensor;
}

/// Exact IEEE FP8 E4M3FN -> f32 conversion (normal, subnormal, signed zero, NaN; max finite 448.0).
fn e4m3ToF32(code: u8) f32 {
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

const e4m3_lut: [256]f32 = blk: {
    @setEvalBranchQuota(4096);
    var lut: [256]f32 = undefined;
    for (0..256) |i| lut[i] = e4m3ToF32(@intCast(i));
    break :blk lut;
};

/// Reads 128x128 block-FP8 scale_inv (`<mod>.weight_scale_inv` or `<mod>.weight_scale`, F32/BF16/F16) as `[]f32`.
fn readBlockFp8Scales(gpa: std.mem.Allocator, ck: *core.Checkpoint, mod_path: []const u8, rows: usize, cols: usize) !?[]f32 {
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
fn dequantBlockFp8ToBf16(codes: []const u8, scale_inv: []const f32, rows: usize, cols: usize, out: []u16) void {
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

/// Dequantizes a BF16, Block-FP8 (`f8_e4m3` + 128x128 `weight_scale_inv`), or MLX affine (4/8-bit, group 32/64) tensor into contiguous BF16 rows `[rows, cols]`.
fn dequantToBf16(gpa: std.mem.Allocator, ck: *core.Checkpoint, mod_path: []const u8, rows: usize, cols: usize) ![]u16 {
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
fn stackDequantPadded(
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

fn stackDequant(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, mods: []const []const u8, row_counts: []const usize, cols: usize) !u64 {
    return stackDequantPadded(w, d, ck, mods, row_counts, 0, cols);
}

/// Uploads a block-FP8 projection directly onto the FP8G lane when source is `f8_e4m3` + block scales and `tools.lane` accepts it.
fn tryLoadFp8g(
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

/// Loads one projection `[rows, cols]`: TFQ3 from `nonmoe.tffpw` when present, else direct `.fp8g` if supported, else `.bf16`.
fn loadProj(
    w: *weights.Weights,
    d: *const cuda.Driver,
    ck: *core.Checkpoint,
    st: *const Store,
    tools: weights.Tools,
    tfq3_name: ?[]const u8,
    ck_mod: []const u8,
    rows: usize,
    cols: usize,
) !projection.Proj {
    if (tfq3_name) |name| {
        if (try st.findOptional(name, rows, cols)) |dw| return .{ .tfq3 = dw };
    }
    if (try tryLoadFp8g(w, d, ck, tools, ck_mod, rows, cols)) |fp8_w| {
        return .{ .fp8g = fp8_w };
    }
    return .{ .bf16 = try stackDequant(w, d, ck, &.{ck_mod}, &.{rows}, cols) };
}

fn ropeInv(w: *weights.Weights, d: *const cuda.Driver) !u64 {
    const c = &w.config;
    if (c.rope == 0) return 0;
    if (w.rope_inv != 0) return w.rope_inv;
    if (c.rope != cuda.mla.rope_width) return error.UnexpectedTensor;
    const f = cuda.mla.ropeFreqs(c.rope_theta);
    w.rope_inv = try uploadBuf(w, d, std.mem.sliceAsBytes(&f));
    return w.rope_inv;
}

fn loadHc(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, pfx: []const u8, i: u32, comptime site: []const u8) !weights.Hc {
    var b: [192]u8 = undefined;
    const f = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b, "{s}layers.{d}.hc_" ++ site ++ "_fn", .{ pfx, i }))).bytes);
    const base = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b, "{s}layers.{d}.hc_" ++ site ++ "_base", .{ pfx, i })));
    const scale = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b, "{s}layers.{d}.hc_" ++ site ++ "_scale", .{ pfx, i })));
    return .{ .fn_w = f, .base = base, .scale = scale };
}

fn loadKda(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, st: *const Store, tools: weights.Tools, pfx: []const u8, i: u32) !kda_block.Layer {
    const c = &w.config;
    const dk: usize = c.kda_dim;
    const all: usize = c.kda_heads * dk;
    var b: [6][192]u8 = undefined;
    const tq = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.q_proj", .{i}), all, c.hidden);
    const tk = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.k_proj", .{i}), all, c.hidden);
    const tv = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.v_proj", .{i}), all, c.hidden);
    const proj: kda_block.In = if (tq != null and tk != null and tv != null) blk: {
        const m_fa = try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.f_a_proj", .{ pfx, i });
        const m_ga = try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.g_a_proj", .{ pfx, i });
        const m_bp = try std.fmt.bufPrint(&b[2], "{s}layers.{d}.self_attn.b_proj", .{ pfx, i });
        const rest = try stackDequant(w, d, ck, &.{ m_fa, m_ga, m_bp }, &.{ dk, dk, c.kda_heads }, c.hidden);
        break :blk .{ .tfq3_split = .{ .q = tq.?, .k = tk.?, .v = tv.?, .rest = rest } };
    } else blk: {
        const m_q = try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.q_proj", .{ pfx, i });
        const m_k = try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.k_proj", .{ pfx, i });
        const m_v = try std.fmt.bufPrint(&b[2], "{s}layers.{d}.self_attn.v_proj", .{ pfx, i });
        const m_fa = try std.fmt.bufPrint(&b[3], "{s}layers.{d}.self_attn.f_a_proj", .{ pfx, i });
        const m_ga = try std.fmt.bufPrint(&b[4], "{s}layers.{d}.self_attn.g_a_proj", .{ pfx, i });
        const m_bp = try std.fmt.bufPrint(&b[5], "{s}layers.{d}.self_attn.b_proj", .{ pfx, i });
        break :blk .{ .bf16 = try stackDequantPadded(
            w,
            d,
            ck,
            &.{ m_q, m_k, m_v, m_fa, m_ga, m_bp },
            &.{ all, all, all, dk, dk, c.kda_heads },
            kda_block.rowWidth(c.kda_heads),
            c.hidden,
        ) };
    };
    const o = try loadProj(
        w,
        d,
        ck,
        st,
        tools,
        try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.o_proj", .{i}),
        try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.o_proj", .{ pfx, i }),
        c.hidden,
        all,
    );
    const fb = try stackDequant(w, d, ck, &.{try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.f_b_proj", .{ pfx, i })}, &.{all}, dk);
    const gb = try stackDequant(w, d, ck, &.{try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.g_b_proj", .{ pfx, i })}, &.{all}, dk);
    var conv_host = try w.gpa.alloc(f32, 3 * all * c.conv);
    defer w.gpa.free(conv_host);
    inline for (.{ "q", "k", "v" }, 0..) |ch, ci| {
        const t = try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn." ++ ch ++ "_conv1d.weight", .{ pfx, i }));
        if (t.numel() != all * c.conv) return error.UnexpectedTensor;
        const dst = conv_host[ci * all * c.conv ..][0 .. all * c.conv];
        if (t.dtype == .f32) {
            @memcpy(std.mem.sliceAsBytes(dst), t.bytes);
        } else if (t.dtype == .bf16) {
            for (dst, 0..) |*o_, idx| o_.* = bf16f(std.mem.readInt(u16, t.bytes[2 * idx ..][0..2], .little));
        } else return error.UnexpectedTensor;
    }
    const conv_w = try uploadBuf(w, d, std.mem.sliceAsBytes(conv_host));
    const a_log = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.A_log", .{ pfx, i })));
    const dt_bias = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.dt_bias", .{ pfx, i })));
    const o_norm = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.o_norm.weight", .{ pfx, i }))).bytes);
    return .{
        .proj = proj,
        .fb = fb,
        .gb = gb,
        .o = o,
        .conv_w = conv_w,
        .a_log = a_log,
        .dt_bias = dt_bias,
        .norm_w = o_norm,
        .eps = c.eps,
        .lower = c.lower_bound,
        .heads = c.kda_heads,
        .dims = c.hidden,
    };
}

fn loadDsa(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, st: *const Store, tools: weights.Tools, pfx: []const u8, i: u32, use_tfq3: bool) !dsa_block.Layer {
    const c = &w.config;
    const h: usize = c.mla_heads;
    const qk: usize = c.nope;
    const qh: usize = qk + c.rope;
    const lw: usize = c.kv_lora;
    const dv: usize = c.v_dim;
    var b: [2][192]u8 = undefined;
    const m_qa = try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.q_a_proj", .{ pfx, i });
    const m_kva = try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.kv_a_proj_with_mqa", .{ pfx, i });
    const proj_ptr = try stackDequant(w, d, ck, &.{ m_qa, m_kva }, &.{ c.q_lora, c.latentRow() }, c.hidden);
    const q_norm = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.q_a_layernorm.weight", .{ pfx, i }))).bytes);
    const kv_norm = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.kv_a_layernorm.weight", .{ pfx, i }))).bytes);
    const q_b = try loadProj(
        w,
        d,
        ck,
        st,
        tools,
        if (use_tfq3) try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.q_b_proj", .{i}) else null,
        try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.q_b_proj", .{ pfx, i }),
        h * qh,
        c.q_lora,
    );
    const o = try loadProj(
        w,
        d,
        ck,
        st,
        tools,
        if (use_tfq3) try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.o_proj", .{i}) else null,
        try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.o_proj", .{ pfx, i }),
        c.hidden,
        h * dv,
    );
    var wk: u64 = 0;
    var wv: u64 = 0;
    if (ck.has(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.embed_q.weight", .{ pfx, i }))) {
        const eq = try dequantToBf16(w.gpa, ck, try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.embed_q", .{ pfx, i }), h * lw, qk);
        defer w.gpa.free(eq);
        const wk_h = try w.gpa.alloc(u16, h * qk * lw);
        defer w.gpa.free(wk_h);
        for (0..h) |hi| for (0..qk) |r| for (0..lw) |col| {
            wk_h[(hi * qk + r) * lw + col] = eq[(hi * lw + col) * qk + r];
        };
        wk = try uploadBuf(w, d, std.mem.sliceAsBytes(wk_h));
        const uo = try dequantToBf16(w.gpa, ck, try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.unembed_out", .{ pfx, i }), h * dv, lw);
        defer w.gpa.free(uo);
        wv = try uploadBuf(w, d, std.mem.sliceAsBytes(uo));
    } else {
        const kvb = try dequantToBf16(w.gpa, ck, try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.kv_b_proj", .{ pfx, i }), h * (qk + dv), lw);
        defer w.gpa.free(kvb);
        const wk_h = try w.gpa.alloc(u16, h * qk * lw);
        defer w.gpa.free(wk_h);
        const wv_h = try w.gpa.alloc(u16, h * dv * lw);
        defer w.gpa.free(wv_h);
        for (0..h) |hi| {
            @memcpy(wk_h[hi * qk * lw ..][0 .. qk * lw], kvb[hi * (qk + dv) * lw ..][0 .. qk * lw]);
            @memcpy(wv_h[hi * dv * lw ..][0 .. dv * lw], kvb[(hi * (qk + dv) + qk) * lw ..][0 .. dv * lw]);
        }
        wk = try uploadBuf(w, d, std.mem.sliceAsBytes(wk_h));
        wv = try uploadBuf(w, d, std.mem.sliceAsBytes(wv_h));
    }
    const inv = try ropeInv(w, d);
    const index: dsa_block.Index = if (c.shared.isSet(i)) blk: {
        const g: sparse.Geometry = .{ .pool = c.kpool, .topk = c.i_topk / c.kpool };
        break :blk .{ .kw = .{ .bf16 = 0 }, .gate = 0, .ln_w = 0, .ln_b = 0, .ape = 0, .qb = .{ .bf16 = 0 }, .heads = c.i_heads, .d = c.i_dim, .geom = g, .own = false, .apart = true };
    } else blk: {
        const m_wk = try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.indexer.wk", .{ pfx, i });
        const m_wp = try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.indexer.weights_proj", .{ pfx, i });
        const kw = try stackDequant(w, d, ck, &.{ m_wk, m_wp }, &.{ c.i_dim, c.i_heads }, c.hidden);
        const qb = try loadProj(
            w,
            d,
            ck,
            st,
            tools,
            if (use_tfq3) try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.indexer.wq_b", .{i}) else null,
            try std.fmt.bufPrint(&b[1], "{s}layers.{d}.self_attn.indexer.wq_b", .{ pfx, i }),
            c.i_heads * c.i_dim,
            c.q_lora,
        );
        const ln_w = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.indexer.k_norm.weight", .{ pfx, i }))).bytes);
        const ln_b = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.indexer.k_norm.bias", .{ pfx, i }))).bytes);
        var gate: u64 = 0;
        var ape: u64 = 0;
        if (c.kpool != 1) {
            gate = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{ pfx, i }))).bytes);
            ape = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{ pfx, i }))).bytes);
        }
        const g: sparse.Geometry = .{ .pool = c.kpool, .topk = c.i_topk / c.kpool, .rope = if (c.dsa) c.rope else 0, .inv = if (c.dsa) inv else 0 };
        break :blk .{ .kw = .{ .bf16 = kw }, .gate = gate, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .qb = qb, .heads = c.i_heads, .d = c.i_dim, .geom = g, .apart = c.dsa };
    };
    return .{
        .proj = .{ .bf16 = proj_ptr },
        .q_norm = q_norm,
        .kv_norm = kv_norm,
        .q_b = q_b,
        .wk = wk,
        .wv = wv,
        .o = o,
        .index = index,
        .heads = h,
        .qk = qk,
        .dv = dv,
        .lw = lw,
        .rope = c.rope,
        .rope_inv = inv,
        .q_lora = c.q_lora,
        .dims = c.hidden,
        .eps = c.eps,
        .dense_limit = c.keyWidth(),
    };
}

fn loadMtpExperts(w: *weights.Weights, d: *const cuda.Driver, stream: cuda.Stream, tools: weights.Tools, ck: *core.Checkpoint, pfx: []const u8, i: u32) !cuda.experts.Layer {
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

/// Full GLM TFQ3 checkpoint load on CUDA (TP=1): supports both full TFQ3 (`nonmoe.tffpw` + `layer_*.tffpw`) and `--keep-nonmoe-bf16` (BF16/FP8/MLX non-MoE + `layer_*.tffpw`).
pub fn loadWeights(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, tools: weights.Tools, dir: []const u8, c: cfg.Config, share: shard.Share, layer_ids: []const u32, with_mtp: bool) !weights.Weights {
    if (share.world != 1) {
        std.log.err("glm tfq3: CUDA TFQ3 runs on --tp 1 (got world={d})", .{share.world});
        return error.UnsupportedQuant;
    }
    var w: weights.Weights = .{ .gpa = gpa, .config = c, .share = share };
    errdefer w.deinit();
    const st = try Store.open(gpa, io, d, dir);
    w.tfq3 = st;
    var ck = try core.Checkpoint.openModelPrefixes(gpa, io, dir, &.{
        "model.language_model.",
        "model.layers.",
        "model.embed_tokens.",
        "model.norm.",
        "model.lm_head.",
        "lm_head.",
    });
    defer core.checkpoint.cache.close(&ck);
    const pfx: []const u8 = if (ck.has("model.language_model.layers.0.input_layernorm.weight"))
        "model.language_model."
    else if (ck.has("model.layers.0.input_layernorm.weight"))
        "model."
    else
        "model.language_model.";
    const use_nonmoe = st.nonmoe != null;
    w.layers = try gpa.alloc(weights.Layer, layer_ids.len);
    var b: [2][192]u8 = undefined;
    for (layer_ids, w.layers) |i, *out_l| {
        const in_norm = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], "{s}layers.{d}.input_layernorm.weight", .{ pfx, i }), c.hidden);
        const post_norm = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], "{s}layers.{d}.post_attention_layernorm.weight", .{ pfx, i }), c.hidden);
        const mixer: weights.Mixer = switch (c.kind(i)) {
            .kda => .{ .kda = try loadKda(&w, d, &ck, st, tools, pfx, i) },
            .mla => .{ .mla = try loadDsa(&w, d, &ck, st, tools, pfx, i, use_nonmoe) },
        };
        const ffn: weights.Ffn = if (!c.isMoe(i)) blk: {
            const tg = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.gate_proj", .{i}), c.dense_inter, c.hidden);
            const tu = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.up_proj", .{i}), c.dense_inter, c.hidden);
            const td = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.down_proj", .{i}), c.hidden, c.dense_inter);
            const mw: mlp.Weights = if (tg != null and tu != null and td != null) .{ .tfq3 = .{
                .gate = tg.?,
                .up = tu.?,
                .down = td.?,
            } } else .{ .bf16 = .{
                .gu = try stackDequant(
                    &w,
                    d,
                    &ck,
                    &.{
                        try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.gate_proj", .{ pfx, i }),
                        try std.fmt.bufPrint(&b[1], "{s}layers.{d}.mlp.up_proj", .{ pfx, i }),
                    },
                    &.{ c.dense_inter, c.dense_inter },
                    c.hidden,
                ),
                .down = try stackDequant(
                    &w,
                    d,
                    &ck,
                    &.{try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.down_proj", .{ pfx, i })},
                    &.{c.hidden},
                    c.dense_inter,
                ),
            } };
            break :blk .{ .dense = .{
                .w = mw,
                .width = c.dense_inter,
                .dims = c.hidden,
                .limit = c.swiglu_limit,
            } };
        } else blk: {
            const sg = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.shared_experts.gate_proj", .{i}), c.shared_inter, c.hidden);
            const su = try st.findOptional(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.shared_experts.up_proj", .{i}), c.shared_inter, c.hidden);
            const shared_gu: projection.Proj = if (sg != null and su != null) .{ .tfq3_pair = .{ sg.?, su.? } } else .{
                .bf16 = try stackDequant(
                    &w,
                    d,
                    &ck,
                    &.{
                        try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.shared_experts.gate_proj", .{ pfx, i }),
                        try std.fmt.bufPrint(&b[1], "{s}layers.{d}.mlp.shared_experts.up_proj", .{ pfx, i }),
                    },
                    &.{ c.shared_inter, c.shared_inter },
                    c.hidden,
                ),
            };
            const shared_down = try loadProj(
                &w,
                d,
                &ck,
                st,
                tools,
                try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.shared_experts.down_proj", .{i}),
                try std.fmt.bufPrint(&b[1], "{s}layers.{d}.mlp.shared_experts.down_proj", .{ pfx, i }),
                c.hidden,
                c.shared_inter,
            );
            break :blk .{ .moe = .{
                .router = try uploadBuf(&w, d, (try checkedRouterWeight(try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.gate.weight", .{ pfx, i })), c.experts, c.hidden)).bytes),
                .bias = try uploadF32(&w, d, try checkedRouterBias(try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.gate.e_score_correction_bias", .{ pfx, i })), c.experts)),
                .routed = .{ .format = .nvfp4, .up = 0, .down = 0, .width = c.moe_inter, .dims = c.hidden, .experts = c.experts, .limit = c.swiglu_limit },
                .tfq3 = try st.loadMoeLayer(gpa, io, d, dir, i, c.experts, c.hidden, c.moe_inter),
                .count = c.experts,
                .shared_gu = shared_gu,
                .shared_down = shared_down,
                .shared = c.shared_inter,
                .top_k = c.topk,
                .scale = c.routed_scale,
                .norm = c.norm_topk,
            } };
        };
        out_l.* = .{
            .index = i,
            .attn_hc = if (c.dsa) weights.none else try loadHc(&w, d, &ck, pfx, i, "attn"),
            .ffn_hc = if (c.dsa) weights.none else try loadHc(&w, d, &ck, pfx, i, "ffn"),
            .in_norm = in_norm,
            .post_norm = post_norm,
            .mixer = mixer,
            .ffn = ffn,
        };
        ck.release();
    }
    w.embed = try stackDequant(&w, d, &ck, &.{try std.fmt.bufPrint(&b[0], "{s}embed_tokens", .{pfx})}, &.{c.vocab}, c.hidden);
    w.norm = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], "{s}norm.weight", .{pfx}), c.hidden);
    var draft_head: DraftHead = undefined;
    if (try st.findOptional("lm_head", c.vocab, c.hidden)) |lm_head| {
        w.head = .{ .tfq3 = lm_head };
        draft_head = .{ .tfq3 = lm_head };
    } else {
        const lm_mod = if (ck.has("lm_head.weight")) "lm_head" else try std.fmt.bufPrint(&b[0], "{s}lm_head", .{pfx});
        const head_ptr = try stackDequant(&w, d, &ck, &.{lm_mod}, &.{c.vocab}, c.hidden);
        w.head = .{ .bf16 = head_ptr };
        draft_head = .{ .bf16 = .{ .w = head_ptr, .n = c.vocab, .k = c.hidden } };
    }
    if (with_mtp and c.mtp > 0) {
        const mi = c.layers;
        const m_in = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], "{s}layers.{d}.input_layernorm.weight", .{ pfx, mi }), c.hidden);
        const m_post = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], "{s}layers.{d}.post_attention_layernorm.weight", .{ pfx, mi }), c.hidden);
        const m_s1 = try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.shared_experts.gate_proj", .{ pfx, mi });
        const m_s2 = try std.fmt.bufPrint(&b[1], "{s}layers.{d}.mlp.shared_experts.up_proj", .{ pfx, mi });
        const m_sgu = try stackDequant(&w, d, &ck, &.{ m_s1, m_s2 }, &.{ c.shared_inter, c.shared_inter }, c.hidden);
        const m_sdn = try stackDequant(&w, d, &ck, &.{try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.shared_experts.down_proj", .{ pfx, mi })}, &.{c.hidden}, c.shared_inter);
        const m_moe: moe.Layer = .{
            .router = try uploadBuf(&w, d, (try checkedRouterWeight(try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.gate.weight", .{ pfx, mi })), c.experts, c.hidden)).bytes),
            .bias = try uploadF32(&w, d, try checkedRouterBias(try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.mlp.gate.e_score_correction_bias", .{ pfx, mi })), c.experts)),
            .routed = try loadMtpExperts(&w, d, stream, tools, &ck, pfx, mi),
            .count = c.experts,
            .shared_gu = .{ .bf16 = m_sgu },
            .shared_down = .{ .bf16 = m_sdn },
            .shared = c.shared_inter,
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        };
        const m_layer: weights.Layer = .{
            .index = mi,
            .attn_hc = weights.none,
            .ffn_hc = weights.none,
            .in_norm = m_in,
            .post_norm = m_post,
            .mixer = .{ .mla = try loadDsa(&w, d, &ck, st, tools, pfx, mi, false) },
            .ffn = .{ .moe = m_moe },
        };
        const enorm = try uploadBuf(&w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.enorm.weight", .{ pfx, mi }))).bytes);
        const hnorm = try uploadBuf(&w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.hnorm.weight", .{ pfx, mi }))).bytes);
        const eh = try stackDequant(&w, d, &ck, &.{try std.fmt.bufPrint(&b[0], "{s}layers.{d}.eh_proj", .{ pfx, mi })}, &.{c.hidden}, 2 * c.hidden);
        const mnorm = try uploadBuf(&w, d, (try ck.get(try std.fmt.bufPrint(&b[0], "{s}layers.{d}.shared_head.norm.weight", .{ pfx, mi }))).bytes);
        w.mtp = .{ .enorm = enorm, .hnorm = hnorm, .eh = eh, .norm = mnorm, .layer = m_layer, .draft = draft_head };
    }
    try stream.synchronize();
    return w;
}

test "CUDA TFQ3: router tensors refuse the wrong dtype, shape and bias length" {
    const weight_bytes: [4 * 64 * 2]u8 = @splat(0);
    const weight: core.checkpoint.Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 4, 64, 1, 1, 1 }, .bytes = &weight_bytes };
    try std.testing.expectEqualDeep(weight, try checkedRouterWeight(weight, 4, 64));

    var wrong = weight;
    wrong.dtype = .u8;
    try std.testing.expectError(error.UnexpectedTensor, checkedRouterWeight(wrong, 4, 64));
    wrong = weight;
    wrong.shape = .{ 8, 32, 1, 1, 1 };
    try std.testing.expectError(error.UnexpectedTensor, checkedRouterWeight(wrong, 4, 64));

    const bias_f32_bytes: [4 * 4]u8 = @splat(0);
    const bias_f32: core.checkpoint.Tensor = .{ .dtype = .f32, .rank = 1, .shape = .{ 4, 1, 1, 1, 1 }, .bytes = &bias_f32_bytes };
    try std.testing.expectEqualDeep(bias_f32, try checkedRouterBias(bias_f32, 4));
    const bias_bf16_bytes: [4 * 2]u8 = @splat(0);
    const bias_bf16: core.checkpoint.Tensor = .{ .dtype = .bf16, .rank = 1, .shape = .{ 4, 1, 1, 1, 1 }, .bytes = &bias_bf16_bytes };
    try std.testing.expectEqualDeep(bias_bf16, try checkedRouterBias(bias_bf16, 4));

    var short = bias_f32;
    short.shape[0] = 3;
    try std.testing.expectError(error.UnexpectedTensor, checkedRouterBias(short, 4));
    var bad_bias = bias_f32;
    bad_bias.dtype = .u8;
    try std.testing.expectError(error.UnexpectedTensor, checkedRouterBias(bad_bias, 4));
}

test "CUDA TFQ3: isTfq3 and hasNonMoe detect nonmoe-only, layer-tfpw-only, and refuse orphan tables" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const fmt = cuda.tfq3_format;

    var tfpn_hdr: [16]u8 = @splat(0);
    std.mem.writeInt(u32, tfpn_hdr[0..4], fmt.magic_tfpn, .little);
    std.mem.writeInt(u32, tfpn_hdr[4..8], fmt.format_version, .little);
    std.mem.writeInt(u32, tfpn_hdr[8..12], 1, .little);

    var tfpw_hdr: [16]u8 = @splat(0);
    std.mem.writeInt(u32, tfpw_hdr[0..4], fmt.magic_tfpw, .little);
    std.mem.writeInt(u32, tfpw_hdr[4..8], fmt.format_version, .little);
    std.mem.writeInt(u32, tfpw_hdr[8..12], 3, .little);
    std.mem.writeInt(u32, tfpw_hdr[12..16], 1, .little);

    var tfpw_empty: [16]u8 = tfpw_hdr;
    std.mem.writeInt(u32, tfpw_empty[12..16], 0, .little);

    // 1. Orphan pw_v3_tables.bin alone is refused.
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);
        try std.testing.expect(!isTfq3(gpa, io, dir));
        try std.testing.expect(!hasNonMoe(gpa, io, dir));

        try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = "PWV3" });
        try std.testing.expect(!isTfq3(gpa, io, dir));
        try std.testing.expect(!hasNonMoe(gpa, io, dir));

        // Zero-record layer_03.tffpw does not trigger TFQ3 either.
        try tmp.dir.writeFile(io, .{ .sub_path = "layer_03.tffpw", .data = &tfpw_empty });
        try std.testing.expect(!isTfq3(gpa, io, dir));
        try std.testing.expect(!hasNonMoe(gpa, io, dir));
    }

    // 2. nonmoe.tffpw + pw_v3_tables.bin (no layer_*.tffpw).
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);
        try tmp.dir.writeFile(io, .{ .sub_path = "nonmoe.tffpw", .data = &tfpn_hdr });
        try std.testing.expect(!isTfq3(gpa, io, dir));
        try std.testing.expect(!hasNonMoe(gpa, io, dir));

        try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = "PWV3" });
        try std.testing.expect(isTfq3(gpa, io, dir));
        try std.testing.expect(hasNonMoe(gpa, io, dir));
    }

    // 3. --keep-nonmoe-bf16: layer_03.tffpw + pw_v3_tables.bin (no nonmoe.tffpw).
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);
        try tmp.dir.writeFile(io, .{ .sub_path = "layer_03.tffpw", .data = &tfpw_hdr });
        try std.testing.expect(!isTfq3(gpa, io, dir));
        try std.testing.expect(!hasNonMoe(gpa, io, dir));

        try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = "PWV3" });
        try std.testing.expect(isTfq3(gpa, io, dir));
        try std.testing.expect(!hasNonMoe(gpa, io, dir));
    }
}

test "CUDA TFQ3: e4m3ToF32 and 128x128 block-FP8 dequantization accuracy" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    // Exact E4M3FN special, subnormal, normal, and max-finite values.
    try std.testing.expectEqual(@as(u32, 0x00000000), @as(u32, @bitCast(e4m3ToF32(0x00))));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(e4m3ToF32(0x80))));
    try std.testing.expectEqual(@as(f32, 0x1p-9), e4m3ToF32(0x01));
    try std.testing.expectEqual(@as(f32, 7.0 * 0x1p-9), e4m3ToF32(0x07));
    try std.testing.expectEqual(@as(f32, -0x1p-9), e4m3ToF32(0x81));
    try std.testing.expectEqual(@as(f32, 0x1p-6), e4m3ToF32(0x08));
    try std.testing.expectEqual(@as(f32, 1.0), e4m3ToF32(0x38));
    try std.testing.expectEqual(@as(f32, -2.0), e4m3ToF32(0xc0));
    try std.testing.expectEqual(@as(f32, 448.0), e4m3ToF32(0x7e));
    try std.testing.expectEqual(@as(f32, -448.0), e4m3ToF32(0xfe));
    try std.testing.expect(std.math.isNan(e4m3ToF32(0x7f)));
    try std.testing.expect(std.math.isNan(e4m3ToF32(0xff)));

    // Exhaustive agreement across all 256 E4M3FN codes against core.quant.fp8.e4m3.
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const got = e4m3ToF32(b);
        const ref = core.quant.fp8.e4m3(b);
        if (b & 0x7f == 0x7f) {
            try std.testing.expect(std.math.isNan(got));
        } else {
            try std.testing.expectEqual(@as(f32, @floatCast(ref)), got);
            try std.testing.expectEqual(got, e4m3_lut[i]);
        }
    }

    // Multi-block non-aligned 128x128 block-FP8 dequantization (130 x 258 -> 2 x 3 blocks).
    const rows: usize = 130;
    const cols: usize = 258;
    const codes = try gpa.alloc(u8, rows * cols);
    defer gpa.free(codes);
    @memset(codes, 0x38); // 1.0
    codes[0 * cols + 127] = 0x3c; // 1.5 in block (0, 0)
    codes[0 * cols + 128] = 0xc0; // -2.0 in block (0, 1)
    codes[0 * cols + 256] = 0x01; // 2^-9 subnormal in block (0, 2)
    codes[128 * cols + 0] = 0x7e; // 448.0 in block (1, 0)
    codes[129 * cols + 257] = 0xbc; // -1.5 in block (1, 2)

    const scales_f32 = [_]f32{ 0.5, 2.0, 4.0, 0.25, 1.5, 0.125 };
    const out = try gpa.alloc(u16, rows * cols);
    defer gpa.free(out);
    dequantBlockFp8ToBf16(codes, &scales_f32, rows, cols, out);

    try std.testing.expectEqual(bf16Of(1.0 * 0.5), out[0 * cols + 0]);
    try std.testing.expectEqual(bf16Of(1.5 * 0.5), out[0 * cols + 127]);
    try std.testing.expectEqual(bf16Of(-2.0 * 2.0), out[0 * cols + 128]);
    try std.testing.expectEqual(bf16Of(0x1p-9 * 4.0), out[0 * cols + 256]);
    try std.testing.expectEqual(bf16Of(448.0 * 0.25), out[128 * cols + 0]);
    try std.testing.expectEqual(bf16Of(-1.5 * 0.125), out[129 * cols + 257]);

    // End-to-end dequantToBf16 with F32, BF16, and F16 weight_scale_inv + 4D weight_scale.
    var scales_bf16: [6]u16 = undefined;
    var scales_f16: [6]u16 = undefined;
    for (scales_f32, 0..) |sv, idx| {
        scales_bf16[idx] = bf16Of(sv);
        scales_f16[idx] = @bitCast(@as(f16, @floatCast(sv)));
    }
    const tensors = [_]core.shard_edit.New{
        .{ .name = "m_f32.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_f32.weight_scale_inv", .dtype = .f32, .shape = &.{ 2, 3 }, .bytes = std.mem.sliceAsBytes(&scales_f32) },
        .{ .name = "m_bf16.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_bf16.weight_scale_inv", .dtype = .bf16, .shape = &.{ 2, 3 }, .bytes = std.mem.sliceAsBytes(&scales_bf16) },
        .{ .name = "m_f16.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_f16.weight_scale_inv", .dtype = .f16, .shape = &.{ 2, 3 }, .bytes = std.mem.sliceAsBytes(&scales_f16) },
        .{ .name = "m_4d.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_4d.weight_scale", .dtype = .f32, .shape = &.{ 2, 1, 3, 1 }, .bytes = std.mem.sliceAsBytes(&scales_f32) },
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const st_path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(st_path);
    try core.shard_edit.create(gpa, st_path, &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    inline for (.{ "m_f32", "m_bf16", "m_f16", "m_4d" }) |name| {
        const got = try dequantToBf16(gpa, &ck, name, rows, cols);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u16, out, got);
    }

    // Store.findOptional returns null when nonmoe is absent, and DraftHead.bf16 reports vocab outputs.
    const st: Store = .{ .tables = .{ .d = undefined, .ptr = 0, .len = 0 }, .nonmoe = null };
    try std.testing.expectEqual(@as(?cuda.tfq3.DenseWeight, null), try st.findOptional("lm_head", 154880, 4096));
    const dh: DraftHead = .{ .bf16 = .{ .w = 0x1000, .n = 154880, .k = 4096 } };
    try std.testing.expectEqual(@as(usize, 154880), dh.outputs());
}
