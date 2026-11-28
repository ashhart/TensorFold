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
const q4 = @import("cuda_q4.zig");
const b16 = @import("cuda_b16.zig");
const projection = @import("cuda_proj.zig");
const glue = @import("cuda_glue.zig");
const deq = @import("cuda_tfq3_dequant.zig");

const bf16f = deq.bf16f;
const uploadBuf = deq.uploadBuf;
const uploadF32 = deq.uploadF32;
const uploadNorm = deq.uploadNorm;
const dequantToBf16 = deq.dequantToBf16;
const stackDequantPadded = deq.stackDequantPadded;
const stackDequant = deq.stackDequant;
const tryLoadFp8g = deq.tryLoadFp8g;

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

/// Returns true if `dir` is a GLM TFQ3/TFQ4 checkpoint (`pw_v3_tables.bin` or `pw_v4_e8p_tables.bin` plus shards).
pub fn isTfq3(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const has_tbl = cuda.tfq3_format.hasTables(gpa, io, dir) or cuda.tfq3_format.hasTfq4Tables(gpa, io, dir);
    return has_tbl and
        (cuda.tfq3_format.hasRecords(gpa, io, dir, "nonmoe.tffpw", cuda.tfq3_format.magic_tfpn) or
            cuda.tfq3_format.hasAnyTfpw(gpa, io, dir));
}

/// Device buffers owning optional `pw_v3_tables.bin`/`pw_v4_e8p_tables.bin`, `nonmoe.tffpw`, and `layer_XX.tffpw`.
pub const Store = struct {
    tables: cuda.DeviceBuffer,
    tables4: ?cuda.tfq4.UploadedTables = null,
    nonmoe: ?cuda.tfq3.TfpnStore = null,
    layers: std.ArrayList(cuda.tfq3.OwnedMoeLayer) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Store {
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        var tables: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
        if (cuda.tfq3_format.hasTables(gpa, io, dir)) {
            tables = try cuda.tfq3.uploadTables(gpa, io, d, dir);
        }
        errdefer tables.free();
        var tables4 = try cuda.tfq4.uploadTablesOptional(gpa, io, d, dir);
        errdefer if (tables4) |*t4| t4.free();
        if (tables.ptr == 0 and tables4 == null) return error.MissingTfq3Tables;
        var nonmoe: ?cuda.tfq3.TfpnStore = null;
        if (tables.ptr != 0 and hasNonMoe(gpa, io, dir)) {
            nonmoe = try cuda.tfq3.TfpnStore.loadFileWithTables(gpa, io, d, dir, "nonmoe.tffpw", tables.ptr);
        }
        s.* = .{ .tables = tables, .tables4 = tables4, .nonmoe = nonmoe };
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
        const t4_ptr: u64 = if (s.tables4) |t4| t4.lutPtr() else 0;
        const t4_kind: u32 = if (s.tables4) |t4| t4.table_kind else 0;
        var owned = try cuda.tfq3.uploadTfpwLayerHybrid(io, d, path, s.tables.ptr, t4_ptr, t4_kind, experts, hidden, inter, i);
        errdefer owned.deinit();
        try s.layers.append(gpa, owned);
        return owned.layer;
    }

    pub fn deinit(s: *Store, gpa: std.mem.Allocator) void {
        for (s.layers.items) |*l| l.deinit();
        s.layers.deinit(gpa);
        if (s.nonmoe) |*nm| nm.deinit();
        if (s.tables4) |*t4| t4.free();
        s.tables.free();
        gpa.destroy(s);
    }
};

pub fn checkedRouterWeight(t: core.checkpoint.Tensor, experts: usize, hidden: usize) !core.checkpoint.Tensor {
    if (!t.is(.bf16, &.{ experts, hidden })) return error.UnexpectedTensor;
    return t;
}

pub fn checkedRouterBias(t: core.checkpoint.Tensor, experts: usize) !core.checkpoint.Tensor {
    if ((t.dtype != .f32 and t.dtype != .bf16) or t.numel() != experts) return error.UnexpectedTensor;
    return t;
}

/// Loads `[rows, cols]`: TFQ3 from `nonmoe.tffpw` when present, else direct `.fp8g`, else `.bf16`.
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

/// Loads GLM TFQ3 on CUDA (TP=1): full (`nonmoe.tffpw`) or `--keep-nonmoe-bf16` (BF16/FP8/MLX non-MoE).
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
            .routed = try deq.loadMtpExperts(&w, d, stream, tools, &ck, pfx, mi),
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

test {
    _ = @import("cuda_tfq3_test.zig");
}
