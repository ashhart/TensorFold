//! GLM-5.3-Flash TFQ3 (3/4/5-bit 2D-FWHT) CUDA loader: dense, MoE, MLX/BF16 residuals, MTP, and DFlash2.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const projection = @import("cuda_proj.zig");
const glue = @import("cuda_glue.zig");

const X = cuda.experts.Nvfp4Experts;
const prefix = "model.language_model.";
const bf16Of = core.bf16_round.bits;

/// Draft vocabulary projection shared by MTP and DFlash2: Q4 (NVFP4/EXL3) or TFQ3 lm_head (zero extra VRAM).
pub const DraftHead = union(enum) {
    q4: q4.Q4,
    tfq3: cuda.tfq3.DenseWeight,

    pub fn outputs(h: DraftHead) usize {
        return switch (h) {
            .q4 => |q| q.n,
            .tfq3 => |w| w.n,
        };
    }

    pub fn project(h: DraftHead, q4k: *const q4.Kernels, tfq3k: ?*const cuda.tfq3.Kernels, stream: cuda.Stream, x: u64, xs: u64, out: u64, rows: usize) !void {
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
        }
    }
};

/// Returns true if `dir` is a GLM TFQ3 checkpoint (`pw_v3_tables.bin` + `nonmoe.tffpw`).
pub fn isTfq3(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    return cuda.tfq3_format.hasTables(gpa, io, dir) and cuda.tfq3_format.hasRecords(gpa, io, dir, "nonmoe.tffpw", cuda.tfq3_format.magic_tfpn);
}

/// Device buffers owning `pw_v3_tables.bin`, `nonmoe.tffpw`, and `layer_03..44.tffpw`.
pub const Store = struct {
    nonmoe: cuda.tfq3.TfpnStore,
    layers: std.ArrayList(cuda.tfq3.OwnedMoeLayer) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Store {
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        s.* = .{ .nonmoe = try cuda.tfq3.TfpnStore.loadFile(gpa, io, d, dir, "nonmoe.tffpw") };
        return s;
    }

    pub fn loadMoeLayer(s: *Store, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, i: u32, experts: u32, hidden: u32, inter: u32) !cuda.tfq3.MoeLayer {
        var name_buf: [32]u8 = undefined;
        const fname = try std.fmt.bufPrint(&name_buf, "layer_{d:0>2}.tffpw", .{i});
        const path = try std.fs.path.join(gpa, &.{ dir, fname });
        defer gpa.free(path);
        var owned = try cuda.tfq3.uploadTfpwLayer(io, d, path, s.nonmoe.tables.ptr, experts, hidden, inter, i);
        errdefer owned.deinit();
        try s.layers.append(gpa, owned);
        return owned.layer;
    }

    pub fn deinit(s: *Store, gpa: std.mem.Allocator) void {
        for (s.layers.items) |*l| l.deinit();
        s.layers.deinit(gpa);
        s.nonmoe.deinit();
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

/// Dequantizes a BF16 or MLX affine (4/8-bit, group 32/64) 2D/3D tensor into contiguous BF16 rows `[rows, cols]`.
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

fn stackDequant(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, mods: []const []const u8, row_counts: []const usize, cols: usize) !u64 {
    var total_rows: usize = 0;
    for (row_counts) |rn| total_rows += rn;
    const out = try w.gpa.alloc(u16, total_rows * cols);
    defer w.gpa.free(out);
    var at: usize = 0;
    for (mods, row_counts) |m, rn| {
        const part = try dequantToBf16(w.gpa, ck, m, rn, cols);
        defer w.gpa.free(part);
        @memcpy(out[at * cols ..][0 .. rn * cols], part);
        at += rn;
    }
    return uploadBuf(w, d, std.mem.sliceAsBytes(out));
}

fn loadHc(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, i: u32, comptime site: []const u8) !weights.Hc {
    var b: [192]u8 = undefined;
    const f = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b, prefix ++ "layers.{d}.hc_" ++ site ++ "_fn", .{i}))).bytes);
    const base = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b, prefix ++ "layers.{d}.hc_" ++ site ++ "_base", .{i})));
    const scale = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b, prefix ++ "layers.{d}.hc_" ++ site ++ "_scale", .{i})));
    return .{ .fn_w = f, .base = base, .scale = scale };
}

fn loadKda(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, st: *Store, i: u32) !kda_block.Layer {
    const c = &w.config;
    const dk: usize = c.kda_dim;
    const all: usize = c.kda_heads * dk;
    var b: [3][192]u8 = undefined;
    const q = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.q_proj", .{i}), all, c.hidden);
    const k = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.k_proj", .{i}), all, c.hidden);
    const v = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.v_proj", .{i}), all, c.hidden);
    const o = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.o_proj", .{i}), c.hidden, all);
    const m_fa = try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.f_a_proj", .{i});
    const m_ga = try std.fmt.bufPrint(&b[1], prefix ++ "layers.{d}.self_attn.g_a_proj", .{i});
    const m_bp = try std.fmt.bufPrint(&b[2], prefix ++ "layers.{d}.self_attn.b_proj", .{i});
    const rest = try stackDequant(w, d, ck, &.{ m_fa, m_ga, m_bp }, &.{ dk, dk, c.kda_heads }, c.hidden);
    const fb = try stackDequant(w, d, ck, &.{try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.f_b_proj", .{i})}, &.{all}, dk);
    const gb = try stackDequant(w, d, ck, &.{try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.g_b_proj", .{i})}, &.{all}, dk);
    var conv_host = try w.gpa.alloc(f32, 3 * all * c.conv);
    defer w.gpa.free(conv_host);
    inline for (.{ "q", "k", "v" }, 0..) |ch, ci| {
        const t = try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn." ++ ch ++ "_conv1d.weight", .{i}));
        if (t.numel() != all * c.conv) return error.UnexpectedTensor;
        const dst = conv_host[ci * all * c.conv ..][0 .. all * c.conv];
        if (t.dtype == .f32) {
            @memcpy(std.mem.sliceAsBytes(dst), t.bytes);
        } else if (t.dtype == .bf16) {
            for (dst, 0..) |*o_, idx| o_.* = bf16f(std.mem.readInt(u16, t.bytes[2 * idx ..][0..2], .little));
        } else return error.UnexpectedTensor;
    }
    const conv_w = try uploadBuf(w, d, std.mem.sliceAsBytes(conv_host));
    const a_log = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.A_log", .{i})));
    const dt_bias = try uploadF32(w, d, try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.dt_bias", .{i})));
    const o_norm = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.o_norm.weight", .{i}))).bytes);
    return .{
        .proj = .{ .tfq3_split = .{ .q = q, .k = k, .v = v, .rest = rest } },
        .fb = fb,
        .gb = gb,
        .o = .{ .tfq3 = o },
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

fn loadDsa(w: *weights.Weights, d: *const cuda.Driver, ck: *core.Checkpoint, st: *Store, i: u32, use_tfq3: bool) !dsa_block.Layer {
    const c = &w.config;
    const h: usize = c.mla_heads;
    const qk: usize = c.nope;
    const qh: usize = qk + c.rope;
    const lw: usize = c.kv_lora;
    const dv: usize = c.v_dim;
    var b: [2][192]u8 = undefined;
    const m_qa = try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.q_a_proj", .{i});
    const m_kva = try std.fmt.bufPrint(&b[1], prefix ++ "layers.{d}.self_attn.kv_a_proj_with_mqa", .{i});
    const proj_ptr = try stackDequant(w, d, ck, &.{ m_qa, m_kva }, &.{ c.q_lora, c.latentRow() }, c.hidden);
    const q_norm = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.q_a_layernorm.weight", .{i}))).bytes);
    const kv_norm = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.kv_a_layernorm.weight", .{i}))).bytes);
    const q_b: projection.Proj = if (use_tfq3)
        .{ .tfq3 = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.q_b_proj", .{i}), h * qh, c.q_lora) }
    else
        .{ .bf16 = try stackDequant(w, d, ck, &.{try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.q_b_proj", .{i})}, &.{h * qh}, c.q_lora) };
    const o: projection.Proj = if (use_tfq3)
        .{ .tfq3 = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.o_proj", .{i}), c.hidden, h * dv) }
    else
        .{ .bf16 = try stackDequant(w, d, ck, &.{try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.o_proj", .{i})}, &.{c.hidden}, h * dv) };
    var wk: u64 = 0;
    var wv: u64 = 0;
    if (ck.has(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.embed_q.weight", .{i}))) {
        const eq = try dequantToBf16(w.gpa, ck, try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.embed_q", .{i}), h * lw, qk);
        defer w.gpa.free(eq);
        const wk_h = try w.gpa.alloc(u16, h * qk * lw);
        defer w.gpa.free(wk_h);
        for (0..h) |hi| for (0..qk) |r| for (0..lw) |col| {
            wk_h[(hi * qk + r) * lw + col] = eq[(hi * lw + col) * qk + r];
        };
        wk = try uploadBuf(w, d, std.mem.sliceAsBytes(wk_h));
        const uo = try dequantToBf16(w.gpa, ck, try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.unembed_out", .{i}), h * dv, lw);
        defer w.gpa.free(uo);
        wv = try uploadBuf(w, d, std.mem.sliceAsBytes(uo));
    } else {
        const kvb = try dequantToBf16(w.gpa, ck, try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.kv_b_proj", .{i}), h * (qk + dv), lw);
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
    const m_wk = try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.indexer.wk", .{i});
    const m_wp = try std.fmt.bufPrint(&b[1], prefix ++ "layers.{d}.self_attn.indexer.weights_proj", .{i});
    const kw = try stackDequant(w, d, ck, &.{ m_wk, m_wp }, &.{ c.i_dim, c.i_heads }, c.hidden);
    const qb: projection.Proj = if (use_tfq3)
        .{ .tfq3 = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.self_attn.indexer.wq_b", .{i}), c.i_heads * c.i_dim, c.q_lora) }
    else
        .{ .bf16 = try stackDequant(w, d, ck, &.{try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.indexer.wq_b", .{i})}, &.{c.i_heads * c.i_dim}, c.q_lora) };
    const ln_w = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.indexer.k_norm.weight", .{i}))).bytes);
    const ln_b = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.indexer.k_norm.bias", .{i}))).bytes);
    const gate = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}))).bytes);
    const ape = try uploadBuf(w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}))).bytes);
    return .{
        .proj = .{ .bf16 = proj_ptr },
        .q_norm = q_norm,
        .kv_norm = kv_norm,
        .q_b = q_b,
        .wk = wk,
        .wv = wv,
        .o = o,
        .index = .{ .kw = .{ .bf16 = kw }, .gate = gate, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .qb = qb, .heads = c.i_heads, .d = c.i_dim, .geom = .{ .pool = c.kpool, .topk = c.i_topk / c.kpool } },
        .heads = h,
        .qk = qk,
        .dv = dv,
        .lw = lw,
        .rope = c.rope,
        .q_lora = c.q_lora,
        .dims = c.hidden,
        .eps = c.eps,
        .dense_limit = c.keyWidth(),
    };
}

fn loadMtpExperts(w: *weights.Weights, d: *const cuda.Driver, stream: cuda.Stream, tools: weights.Tools, ck: *core.Checkpoint, i: u32) !cuda.experts.Layer {
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
                const mp = try std.fmt.bufPrint(&nb, prefix ++ "layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e0 + j });
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

/// Full GLM-5.3-Flash TFQ3 checkpoint load on CUDA (TP=1): backbone + MTP layer + TFQ3 draft head (for MTP & DFlash2).
pub fn loadWeights(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, tools: weights.Tools, dir: []const u8, c: cfg.Config, share: shard.Share, layer_ids: []const u32, with_mtp: bool) !weights.Weights {
    if (share.world != 1) {
        std.log.err("glm tfq3: CUDA TFQ3 runs on --tp 1 (got world={d})", .{share.world});
        return error.UnsupportedQuant;
    }
    var w: weights.Weights = .{ .gpa = gpa, .config = c, .share = share };
    errdefer w.deinit();
    const st = try Store.open(gpa, io, d, dir);
    w.tfq3 = st;
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer core.checkpoint.cache.close(&ck);
    w.layers = try gpa.alloc(weights.Layer, layer_ids.len);
    var b: [2][192]u8 = undefined;
    for (layer_ids, w.layers) |i, *out_l| {
        const in_norm = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.input_layernorm.weight", .{i}), c.hidden);
        const post_norm = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.post_attention_layernorm.weight", .{i}), c.hidden);
        const mixer: weights.Mixer = switch (c.kind(i)) {
            .kda => .{ .kda = try loadKda(&w, d, &ck, st, i) },
            .mla => .{ .mla = try loadDsa(&w, d, &ck, st, i, true) },
        };
        const ffn: weights.Ffn = if (!c.isMoe(i)) .{ .dense = .{
            .w = .{ .tfq3 = .{
                .gate = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.gate_proj", .{i}), c.dense_inter, c.hidden),
                .up = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.up_proj", .{i}), c.dense_inter, c.hidden),
                .down = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.down_proj", .{i}), c.hidden, c.dense_inter),
            } },
            .width = c.dense_inter,
            .dims = c.hidden,
            .limit = c.swiglu_limit,
        } } else .{ .moe = .{
            .router = try uploadBuf(&w, d, (try checkedRouterWeight(try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.mlp.gate.weight", .{i})), c.experts, c.hidden)).bytes),
            .bias = try uploadF32(&w, d, try checkedRouterBias(try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.mlp.gate.e_score_correction_bias", .{i})), c.experts)),
            .routed = .{ .format = .nvfp4, .up = 0, .down = 0, .width = c.moe_inter, .dims = c.hidden, .experts = c.experts, .limit = c.swiglu_limit },
            .tfq3 = try st.loadMoeLayer(gpa, io, d, dir, i, c.experts, c.hidden, c.moe_inter),
            .count = c.experts,
            .shared_gu = .{ .tfq3_pair = .{
                try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.shared_experts.gate_proj", .{i}), c.shared_inter, c.hidden),
                try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.shared_experts.up_proj", .{i}), c.shared_inter, c.hidden),
            } },
            .shared_down = .{ .tfq3 = try st.nonmoe.findChecked(try std.fmt.bufPrint(&b[0], "layers.{d}.mlp.shared_experts.down_proj", .{i}), c.hidden, c.shared_inter) },
            .shared = c.shared_inter,
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        } };
        out_l.* = .{ .index = i, .attn_hc = try loadHc(&w, d, &ck, i, "attn"), .ffn_hc = try loadHc(&w, d, &ck, i, "ffn"), .in_norm = in_norm, .post_norm = post_norm, .mixer = mixer, .ffn = ffn };
        ck.release();
    }
    w.embed = try stackDequant(&w, d, &ck, &.{prefix ++ "embed_tokens"}, &.{c.vocab}, c.hidden);
    w.norm = try uploadNorm(&w, d, &ck, prefix ++ "norm.weight", c.hidden);
    const lm_head = try st.nonmoe.findChecked("lm_head", c.vocab, c.hidden);
    w.head = .{ .tfq3 = lm_head };
    if (with_mtp and c.mtp > 0) {
        const mi = c.layers;
        const m_in = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.input_layernorm.weight", .{mi}), c.hidden);
        const m_post = try uploadNorm(&w, d, &ck, try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.post_attention_layernorm.weight", .{mi}), c.hidden);
        const m_s1 = try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.mlp.shared_experts.gate_proj", .{mi});
        const m_s2 = try std.fmt.bufPrint(&b[1], prefix ++ "layers.{d}.mlp.shared_experts.up_proj", .{mi});
        const m_sgu = try stackDequant(&w, d, &ck, &.{ m_s1, m_s2 }, &.{ c.shared_inter, c.shared_inter }, c.hidden);
        const m_sdn = try stackDequant(&w, d, &ck, &.{try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.mlp.shared_experts.down_proj", .{mi})}, &.{c.hidden}, c.shared_inter);
        const m_moe: moe.Layer = .{
            .router = try uploadBuf(&w, d, (try checkedRouterWeight(try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.mlp.gate.weight", .{mi})), c.experts, c.hidden)).bytes),
            .bias = try uploadF32(&w, d, try checkedRouterBias(try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.mlp.gate.e_score_correction_bias", .{mi})), c.experts)),
            .routed = try loadMtpExperts(&w, d, stream, tools, &ck, mi),
            .count = c.experts,
            .shared_gu = .{ .bf16 = m_sgu },
            .shared_down = .{ .bf16 = m_sdn },
            .shared = c.shared_inter,
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        };
        const none: weights.Hc = .{ .fn_w = 0, .base = 0, .scale = 0 };
        const m_layer: weights.Layer = .{ .index = mi, .attn_hc = none, .ffn_hc = none, .in_norm = m_in, .post_norm = m_post, .mixer = .{ .mla = try loadDsa(&w, d, &ck, st, mi, false) }, .ffn = .{ .moe = m_moe } };
        const enorm = try uploadBuf(&w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.enorm.weight", .{mi}))).bytes);
        const hnorm = try uploadBuf(&w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.hnorm.weight", .{mi}))).bytes);
        const eh = try stackDequant(&w, d, &ck, &.{try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.eh_proj", .{mi})}, &.{c.hidden}, 2 * c.hidden);
        const mnorm = try uploadBuf(&w, d, (try ck.get(try std.fmt.bufPrint(&b[0], prefix ++ "layers.{d}.shared_head.norm.weight", .{mi}))).bytes);
        w.mtp = .{ .enorm = enorm, .hnorm = hnorm, .eh = eh, .norm = mnorm, .layer = m_layer, .draft = .{ .tfq3 = lm_head } };
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
