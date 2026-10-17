//! A TP rank's GLM-5.3-Flash weights on CUDA (the Python family's weights.load, NVFP4), named by its paths for digests.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const X = cuda.experts.Nvfp4Experts;
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const exl3 = @import("cuda_exl3.zig");
const mm_mod = @import("cuda_mm.zig");

const Tensor = core.checkpoint.Tensor;
const prefix = "model.language_model.";

/// A hyper-connection site: fn bf16 [s (s + 2)][s d], base and scale widened to fp32.
pub const Hc = struct { fn_w: u64, base: u64, scale: u64 };

pub const Mixer = union(cfg.Kind) { kda: kda_block.Layer, mla: dsa_block.Layer };
pub const Ffn = union(enum) { dense: mlp.Layer, moe: moe.Layer };

pub const Layer = struct { index: u32, attn_hc: Hc, ffn_hc: Hc, in_norm: u64, post_norm: u64, mixer: Mixer, ffn: Ffn };

/// The MTP head (weights.MTPW): norms, eh_proj, a DSA + MoE layer (NVFP4 experts), shared_head.norm, 4-bit draft head.
pub const Mtp = struct { enorm: u64, hnorm: u64, eh: u64, norm: u64, layer: Layer, draft: q4.Q4 };

/// The kernels a load runs: affine4_pack's tf_pack_dense, and for the MTP head the quantizers and the 4-bit tiling.
pub const Tools = struct { pack_dense: cuda.Function, quant: ?*const quant.Kernels = null, q4: ?*const q4.Kernels = null };

/// A loaded tensor by the Python family's attribute path, for digests against its weights.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    share: shard.Share,
    embed: u64 = 0, // bf16 [vocab][dims]
    layers: []Layer = &.{},
    norm: u64 = 0,
    head: u64 = 0, // bf16 [share.vocab][dims]
    mtp: ?Mtp = null,
    exl3_shx: bool = true, // EXL3: every layer's gate and up share one input rotation (a prompt rotates each row once)
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    /// The upload named `name` (the Python family's attribute path), or null.
    pub fn find(w: *const Weights, name: []const u8) ?Named {
        for (w.named.items) |n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

/// Host bf16 from fp32: round to nearest even (torch's float -> bfloat16), NaN kept quiet.
fn toBf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    pack_dense: cuda.Function,
    tools: Tools,
    w: *Weights,
    ck: *core.Checkpoint,
    host: std.ArrayList(u8) = .empty,
    name_buf: [192]u8 = undefined,

    pub fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ fmt, args);
        return L.ck.get(name);
    }

    /// The host bytes gathered so far uploaded as one tensor named `fmt`.
    pub fn put(L: *Loader, comptime fmt: []const u8, args: anytype) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, fmt, args), .ptr = b.ptr, .len = b.len });
        L.host.clearRetainingCapacity();
        return b.ptr;
    }

    /// Elements [lo, hi) of `t` flat as `to` (bf16 or fp32 from a bf16 or fp32 tensor; other dtypes only as stored).
    fn add(L: *Loader, t: Tensor, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        if (t.dtype == to) {
            const size = t.bytes.len / t.numel();
            return L.host.appendSlice(L.gpa, t.bytes[lo * size .. hi * size]);
        }
        for (lo..hi) |i| {
            const v: f32 = switch (t.dtype) {
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
                .f32 => std.mem.bytesToValue(f32, t.bytes[4 * i ..][0..4]),
                else => return error.UnexpectedTensor,
            };
            switch (to) {
                .f32 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&v)),
                .bf16 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&toBf16(v))),
                else => return error.UnexpectedTensor,
            }
        }
    }

    /// Rows [lo, hi) of `t`'s outermost axis (`total` long; a vector's elements): split.py's "row" cut.
    fn rows(L: *Loader, t: Tensor, lo: usize, hi: usize, total: usize, to: core.checkpoint.DType) !void {
        const per = t.numel() / total;
        if (per * total != t.numel()) return error.UnexpectedTensor;
        try L.add(t, lo * per, hi * per, to);
    }

    fn whole(L: *Loader, t: Tensor, to: core.checkpoint.DType) !void {
        try L.add(t, 0, t.numel(), to);
    }

    /// Columns [lo, hi) of every row of a matrix: split.py's "col" cut.
    fn cols(L: *Loader, t: Tensor, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        if (t.rank != 2) return error.UnexpectedTensor;
        const k = t.dim(1);
        for (0..t.dim(0)) |r| try L.add(t, r * k + lo, r * k + hi, to);
    }

    fn hc(L: *Loader, i: u32, comptime site: []const u8) !Hc {
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_fn", .{i}), .bf16);
        const f = try L.put("layers.{d}.{s}_hc.fn", .{ i, site });
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_base", .{i}), .f32);
        const base = try L.put("layers.{d}.{s}_hc.base", .{ i, site });
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_scale", .{i}), .f32);
        return .{ .fn_w = f, .base = base, .scale = try L.put("layers.{d}.{s}_hc.scale", .{ i, site }) };
    }

    /// weights.load's kda(i): [q | k | v | f_a | g_a | b], f_b/g_b by heads, o by columns, conv taps, A_log, dt_bias.
    fn kda(L: *Loader, i: u32) !kda_block.Layer {
        const c = &L.w.config;
        const h = L.w.share.kda;
        const dk = c.kda_dim;
        const all = c.kda_heads * dk;
        inline for (.{ "q", "k", "v" }) |x| try L.rows(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        inline for (.{ "f_a", "g_a" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), .bf16);
        try L.rows(try L.get("layers.{d}.self_attn.b_proj.weight", .{i}), h.lo, h.hi, c.kda_heads, .bf16);
        const proj = try L.put("layers.{d}.kda.proj", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.f_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        const fb = try L.put("layers.{d}.kda.fb", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.g_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        const gb = try L.put("layers.{d}.kda.gb", .{i});
        inline for (.{ "q", "k", "v" }) |x| try L.rows(try L.get("layers.{d}.self_attn." ++ x ++ "_conv1d.weight", .{i}), h.lo * dk, h.hi * dk, all, .f32);
        const conv = try L.put("layers.{d}.kda.conv", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.A_log", .{i}), h.lo, h.hi, c.kda_heads, .f32);
        const a_log = try L.put("layers.{d}.kda.a_log", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.dt_bias", .{i}), h.lo * dk, h.hi * dk, all, .f32);
        const dt = try L.put("layers.{d}.kda.dt_bias", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.o_norm.weight", .{i}), .bf16);
        const norm = try L.put("layers.{d}.kda.norm", .{i});
        try L.cols(try L.get("layers.{d}.self_attn.o_proj.weight", .{i}), h.lo * dk, h.hi * dk, .bf16);
        const o = try L.put("layers.{d}.kda.o", .{i});
        return .{ .proj = proj, .fb = fb, .gb = gb, .o = o, .conv_w = conv, .a_log = a_log, .dt_bias = dt, .norm_w = norm, .eps = c.eps, .lower = c.lower_bound, .heads = h.len(), .dims = c.hidden };
    }

    /// weights.load's dsa(i) (latent): [q_a | kv_a], norms, q_b by heads, kv_b as wk/wv (AbsorbW), o by cols, indexer.
    fn dsa(L: *Loader, i: u32) !dsa_block.Layer {
        const c = &L.w.config;
        const h = L.w.share.mla;
        const qk = c.nope;
        inline for (.{ "q_a_proj", "kv_a_proj_with_mqa" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ ".weight", .{i}), .bf16);
        const proj = try L.put("layers.{d}.dsa.proj", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.q_a_layernorm.weight", .{i}), .bf16);
        const q_norm = try L.put("layers.{d}.dsa.q_norm", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}), .bf16);
        const kv_norm = try L.put("layers.{d}.dsa.kv_norm", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.q_b_proj.weight", .{i}), h.lo * qk, h.hi * qk, c.mla_heads * qk, .bf16);
        const q_b = try L.put("layers.{d}.dsa.q_b", .{i});
        const kv_b = try L.get("layers.{d}.self_attn.kv_b_proj.weight", .{i});
        const per = qk + c.v_dim; // a head's kv_b rows: its keys, then its values
        const total = c.mla_heads * per;
        for (h.lo..h.hi) |head| try L.rows(kv_b, head * per, head * per + qk, total, .bf16);
        const wk = try L.put("layers.{d}.dsa.absorb.wk", .{i});
        for (h.lo..h.hi) |head| try L.rows(kv_b, head * per + qk, (head + 1) * per, total, .bf16);
        const wv = try L.put("layers.{d}.dsa.absorb.wv", .{i});
        try L.cols(try L.get("layers.{d}.self_attn.o_proj.weight", .{i}), h.lo * c.v_dim, h.hi * c.v_dim, .bf16);
        const o = try L.put("layers.{d}.dsa.o", .{i});
        inline for (.{ "wk", "weights_proj" }) |x| try L.whole(try L.get("layers.{d}.self_attn.indexer." ++ x ++ ".weight", .{i}), .bf16);
        const kw = try L.put("layers.{d}.dsa.index.kw", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.wq_b.weight", .{i}), .bf16);
        const qb = try L.put("layers.{d}.dsa.index.qb", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.weight", .{i}), .bf16);
        const ln_w = try L.put("layers.{d}.dsa.index.ln_w", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.bias", .{i}), .bf16);
        const ln_b = try L.put("layers.{d}.dsa.index.ln_b", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}), .bf16);
        const gate = try L.put("layers.{d}.dsa.index.gate", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}), .bf16);
        const ape = try L.put("layers.{d}.dsa.index.ape", .{i});
        return .{
            .proj = proj,
            .q_norm = q_norm,
            .kv_norm = kv_norm,
            .q_b = q_b,
            .wk = wk,
            .wv = wv,
            .o = o,
            .index = .{ .kw = kw, .gate = gate, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .qb = qb, .heads = c.i_heads, .d = c.i_dim },
            .heads = h.len(),
            .qk = qk,
            .dv = c.v_dim,
            .lw = c.kv_lora,
            .q_lora = c.q_lora,
            .dims = c.hidden,
            .eps = c.eps,
            .dense_limit = c.keyWidth(),
        };
    }

    /// An NVFP4 linear's rows [lo, hi), inputs [klo, khi) as Fp4Linear.from_checkpoint (tf_pack_dense, packScales).
    fn fp4(L: *Loader, comptime fmt: []const u8, i: u32, lo: usize, hi: usize, klo: usize, khi: usize) !cuda.qmmf.Weight {
        const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ fmt, .{i});
        const l = try core.nvfp4.Linear.read(L.ck, name);
        const n = hi - lo;
        const k = khi - klo;
        for (0..n) |r| try L.host.appendSlice(L.gpa, l.codes[(lo + r) * l.k / 2 + klo / 2 ..][0 .. k / 2]);
        var src = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        defer src.free();
        L.host.clearRetainingCapacity();
        const npad = cuda.nvfp4.padded(n);
        const kg = k / 64;
        const total: u64 = @as(u64, npad / 64) * kg * 512;
        const words = try cuda.DeviceBuffer.alloc(L.d, total * 4);
        try L.w.buffers.append(L.gpa, words);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, fmt ++ ".words", .{i}), .ptr = words.ptr, .len = words.len });
        var args: cuda.Args = .{};
        args.add(src.ptr);
        for ([_]usize{ n, k / 8, kg }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(words.ptr);
        args.add(@as(c_longlong, @intCast(total)));
        try cuda.launch.launch(L.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, L.stream, &args);
        try L.stream.synchronize();
        const scales = try L.gpa.alloc(u8, n * k / 16);
        defer L.gpa.free(scales);
        for (0..n) |r| @memcpy(scales[r * k / 16 ..][0 .. k / 16], l.scales[(lo + r) * l.k / 16 + klo / 16 ..][0 .. k / 16]);
        try L.host.resize(L.gpa, npad * k / 16);
        cuda.nvfp4.packScales(L.host.items, scales, n, k);
        const bs = try L.put(fmt ++ ".scales", .{i});
        return cuda.nvfp4.weight(words.ptr, bs, l.global, n, k);
    }

    /// weights.load's mlp for a dense layer: gate and up by the rank's rows, down by its columns (NVFP4).
    fn dense(L: *Loader, i: u32) !mlp.Layer {
        const c = &L.w.config;
        const s = L.w.share.dense;
        const d = c.hidden;
        const w: mlp.Weights = if (c.exl3) bf16: { // an EXL3 checkpoint keeps its dense layers bf16
            inline for (.{ "gate_proj", "up_proj" }) |x| try L.rows(try L.get("layers.{d}.mlp." ++ x ++ ".weight", .{i}), s.lo, s.hi, c.dense_inter, .bf16);
            const gu = try L.put("layers.{d}.mlp.gu", .{i});
            try L.cols(try L.get("layers.{d}.mlp.down_proj.weight", .{i}), s.lo, s.hi, .bf16);
            break :bf16 .{ .bf16 = .{ .gu = gu, .down = try L.put("layers.{d}.mlp.down", .{i}) } };
        } else .{ .nvfp4 = .{
            .gate = try L.fp4("layers.{d}.mlp.gate_proj", i, s.lo, s.hi, 0, d),
            .up = try L.fp4("layers.{d}.mlp.up_proj", i, s.lo, s.hi, 0, d),
            .down = try L.fp4("layers.{d}.mlp.down_proj", i, 0, d, s.lo, s.hi),
        } };
        return .{
            .w = w,
            .width = s.len(),
            .dims = d,
            .limit = c.swiglu_limit,
        };
    }

    /// One device allocation the loader frees itself (staging for the GPU's quantizers).
    fn scratch(L: *Loader, n: usize) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(L.d, n);
    }

    /// moe_nvfp4 of the MTP layer: BF16 experts cut to the rank, GPU-quantized 8 at a time, scale as weight_scale_2.
    fn quantizedExperts(L: *Loader, i: u32, up: []u32, down: []u32, up_scale: []f32, down_scale: []f32) !void {
        const qk = L.tools.quant orelse return error.NoQuantizer;
        const c = &L.w.config;
        const d: usize = c.hidden;
        const cut = L.w.share.moe;
        const ni = cut.len();
        const up_len = X.bytes(1, ni, d, 2) / 4;
        const down_len = X.bytes(1, d, ni, 1) / 4;
        const chunk = 8;
        var wb = try L.scratch(chunk * ni * d * 2);
        defer wb.free();
        var cb = try L.scratch(chunk * ni * d / 2);
        defer cb.free();
        var sb = try L.scratch(chunk * ni * d / 16);
        defer sb.free();
        var gb = try L.scratch(chunk * 4);
        defer gb.free();
        const codes = try L.gpa.alloc(u8, 3 * chunk * ni * d / 2); // gate, up and down apart: packGateUp takes both
        defer L.gpa.free(codes);
        const scales = try L.gpa.alloc(u8, 3 * chunk * ni * d / 16);
        defer L.gpa.free(scales);
        var g: [3][chunk]f32 = undefined;
        var e0: usize = 0;
        while (e0 < c.experts) : (e0 += chunk) {
            const ec = @min(chunk, c.experts - e0);
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const n = if (m < 2) ni else d; // gate and up: the rank's rows; down: every row, the rank's columns
                const k = if (m < 2) d else ni;
                for (e0..e0 + ec) |e| {
                    const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".weight", .{ i, e });
                    if (m < 2) try L.rows(t, cut.lo, cut.hi, c.moe_inter, .bf16) else try L.cols(t, cut.lo, cut.hi, .bf16);
                }
                try wb.upload(0, L.host.items);
                L.host.clearRetainingCapacity();
                try quant.nvfp4(qk, L.stream, wb.ptr, ec, n, k, cb.ptr, sb.ptr, gb.ptr);
                try L.stream.synchronize();
                try cb.download(0, codes[m * chunk * n * k / 2 ..][0 .. ec * n * k / 2]);
                try sb.download(0, scales[m * chunk * n * k / 16 ..][0 .. ec * n * k / 16]);
                try gb.download(0, std.mem.sliceAsBytes(g[m][0..ec]));
            }
            const mc = chunk * ni * d / 2; // one matrix's codes for the chunk; scales an eighth of that
            for (0..ec) |j| {
                const e = e0 + j;
                const one = ni * d;
                var cs: [3][]const u8 = undefined;
                var ss: [3][]const u8 = undefined;
                for (0..3) |m| {
                    cs[m] = codes[m * mc + j * one / 2 ..][0 .. one / 2];
                    ss[m] = scales[m * mc / 8 + j * one / 16 ..][0 .. one / 16];
                }
                X.packGateUp(up[e * up_len ..][0..up_len], .{ cs[0], ss[0] }, .{ cs[1], ss[1] }, ni, d);
                X.packOne(down[e * down_len ..][0..down_len], cs[2], ss[2], d, ni);
                up_scale[e * 2] = g[0][j];
                up_scale[e * 2 + 1] = g[1][j];
                down_scale[e] = g[2][j];
            }
        }
    }

    /// weights.load's moe(i) on NVFP4: router and bias, routed experts as Experts4, the BF16 shared expert's share.
    fn moeLayer(L: *Loader, i: u32) !moe.Layer {
        const c = &L.w.config;
        const d: usize = c.hidden;
        const cut = L.w.share.moe;
        const ni = cut.len();
        try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
        const router = try L.put("layers.{d}.moe.router", .{i});
        try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
        const bias = try L.put("layers.{d}.moe.bias", .{i});
        const count: usize = c.experts;
        var routed: cuda.experts.Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit };
        var x3: ?exl3.Experts = null;
        if (c.exl3) { // the MTP layer's experts too
            x3 = try exl3.load(L, i, count, d, c.moe_inter, cut);
            if (!x3.?.shx) L.w.exl3_shx = false;
        } else routed = try L.nvfp4Experts(i);
        const sh = L.w.share.shared;
        inline for (.{ "gate_proj", "up_proj" }) |x| try L.rows(try L.get("layers.{d}.mlp.shared_experts." ++ x ++ ".weight", .{i}), sh.lo, sh.hi, c.shared_inter, .bf16);
        const gu = try L.put("layers.{d}.moe.shared.gu", .{i});
        try L.cols(try L.get("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}), sh.lo, sh.hi, .bf16);
        const sd = try L.put("layers.{d}.moe.shared.down", .{i});
        return .{
            .router = router,
            .bias = bias,
            .routed = routed,
            .exl3 = x3,
            .count = count,
            .shared_gu = gu,
            .shared_down = sd,
            .shared = sh.len(),
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        };
    }

    /// Layer `i`'s routed NVFP4 experts packed as Experts4 (the MTP layer's quantized from its BF16 ones).
    fn nvfp4Experts(L: *Loader, i: u32) !cuda.experts.Layer {
        const c = &L.w.config;
        const d: usize = c.hidden;
        const cut = L.w.share.moe;
        const ni = cut.len();
        const count: usize = c.experts;
        const up_len = X.bytes(1, ni, d, 2) / 4;
        const down_len = X.bytes(1, d, ni, 1) / 4;
        const up = try L.gpa.alloc(u32, count * up_len);
        defer L.gpa.free(up);
        const down = try L.gpa.alloc(u32, count * down_len);
        defer L.gpa.free(down);
        const up_scale = try L.gpa.alloc(f32, count * 2);
        defer L.gpa.free(up_scale);
        const down_scale = try L.gpa.alloc(f32, count);
        defer L.gpa.free(down_scale);
        const codes = try L.gpa.alloc(u8, d * ni / 2);
        defer L.gpa.free(codes);
        const scales = try L.gpa.alloc(u8, d * ni / 16);
        defer L.gpa.free(scales);
        if (i == c.layers) try L.quantizedExperts(i, up, down, up_scale, down_scale) else for (0..count) |e| {
            var ls: [3]core.nvfp4.Linear = undefined;
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e });
                ls[m] = try core.nvfp4.Linear.read(L.ck, name);
            }
            // gate and up: the rank's rows; down: every row, the rank's columns
            const gate = .{ ls[0].codes[cut.lo * d / 2 ..][0 .. ni * d / 2], ls[0].scales[cut.lo * d / 16 ..][0 .. ni * d / 16] };
            const upm = .{ ls[1].codes[cut.lo * d / 2 ..][0 .. ni * d / 2], ls[1].scales[cut.lo * d / 16 ..][0 .. ni * d / 16] };
            X.packGateUp(up[e * up_len ..][0..up_len], gate, upm, ni, d);
            const l = ls[2];
            for (0..d) |r| {
                @memcpy(codes[r * ni / 2 ..][0 .. ni / 2], l.codes[r * l.k / 2 + cut.lo / 2 ..][0 .. ni / 2]);
                @memcpy(scales[r * ni / 16 ..][0 .. ni / 16], l.scales[r * l.k / 16 + cut.lo / 16 ..][0 .. ni / 16]);
            }
            X.packOne(down[e * down_len ..][0..down_len], codes, scales, d, ni);
            up_scale[e * 2] = ls[0].global;
            up_scale[e * 2 + 1] = ls[1].global;
            down_scale[e] = l.global;
        }
        var ptrs: [4]u64 = undefined;
        for (&ptrs, [_][]const u8{ std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(up_scale), std.mem.sliceAsBytes(down_scale) }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
            try L.host.appendSlice(L.gpa, bytes);
            p.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
        }
        return .{ .format = .nvfp4, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit };
    }

    /// weights.load's MTPW: layer `layers` as plain DSA + MoE, norms, eh_proj, draft head from the rank's head rows.
    fn mtp(L: *Loader) !Mtp {
        const c = &L.w.config;
        const i = c.layers;
        try L.whole(try L.get("layers.{d}.input_layernorm.weight", .{i}), .bf16);
        const in_norm = try L.put("layers.{d}.in_norm", .{i});
        try L.whole(try L.get("layers.{d}.post_attention_layernorm.weight", .{i}), .bf16);
        const post_norm = try L.put("layers.{d}.post_norm", .{i});
        const none: Hc = .{ .fn_w = 0, .base = 0, .scale = 0 };
        const layer_w: Layer = .{ .index = i, .attn_hc = none, .ffn_hc = none, .in_norm = in_norm, .post_norm = post_norm, .mixer = .{ .mla = try L.dsa(i) }, .ffn = .{ .moe = try L.moeLayer(i) } };
        L.ck.release();
        var out: [4]u64 = undefined;
        for (&out, [_][]const u8{ "enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight" }, [_][]const u8{ "enorm", "hnorm", "eh", "norm" }) |*p, name, what| {
            try L.whole(try L.ck.get(try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.{s}", .{ i, name })), .bf16);
            p.* = try L.put("mtp.{s}", .{what});
        }
        // the draft head: quantize4 of the BF16 head's rows, then make_q4's tiling
        const qk = L.tools.quant orelse return error.NoQuantizer;
        const tk = L.tools.q4 orelse return error.NoQuantizer;
        const n = L.w.share.vocab.len();
        const k: usize = c.hidden;
        var raw: [3]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        defer for (raw[0..made]) |*b| b.free();
        for (&raw, [_]usize{ n * k / 2, n * k / 32, n * k / 32 }) |*b, size| {
            b.* = try L.scratch(size);
            made += 1;
        }
        try quant.q4(qk, L.stream, L.w.head, n, k, raw[0].ptr, raw[1].ptr, raw[2].ptr);
        var tiled: [3]u64 = undefined;
        for (&tiled, q4.bytes(n, k), [_][]const u8{ "words", "scales", "biases" }) |*p, size, what| {
            const b = try cuda.DeviceBuffer.alloc(L.d, size);
            try L.w.buffers.append(L.gpa, b);
            try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "draft_head.{s}", .{what}), .ptr = b.ptr, .len = b.len });
            p.* = b.ptr;
        }
        const draft: q4.Q4 = .{ .w = tiled[0], .s = tiled[1], .b = tiled[2], .n = n, .k = k, .npad = q4.npadOf(n) };
        try q4.pack(tk, L.stream, raw[0].ptr, raw[1].ptr, raw[2].ptr, draft);
        try L.stream.synchronize();
        return .{ .enorm = out[0], .hnorm = out[1], .eh = out[2], .norm = out[3], .layer = layer_w, .draft = draft };
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = &L.w.config;
        try L.whole(try L.get("layers.{d}.input_layernorm.weight", .{i}), .bf16);
        const in_norm = try L.put("layers.{d}.in_norm", .{i});
        try L.whole(try L.get("layers.{d}.post_attention_layernorm.weight", .{i}), .bf16);
        const post_norm = try L.put("layers.{d}.post_norm", .{i});
        return .{
            .index = i,
            .attn_hc = try L.hc(i, "attn"),
            .ffn_hc = try L.hc(i, "ffn"),
            .in_norm = in_norm,
            .post_norm = post_norm,
            .mixer = switch (c.kind(i)) {
                .kda => .{ .kda = try L.kda(i) },
                .mla => .{ .mla = try L.dsa(i) },
            },
            .ffn = if (c.isMoe(i)) .{ .moe = try L.moeLayer(i) } else .{ .dense = try L.dense(i) },
        };
    }
};

/// weights.load without the MTP head: embedding, layers, final norm, the rank's BF16 head rows; `pack_dense`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, tools: Tools, dir: []const u8, c: cfg.Config, share: shard.Share, layers: []const u32, with_mtp: bool) !Weights {
    var w: Weights = .{ .gpa = gpa, .config = c, .share = share };
    errdefer w.deinit();
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    var head_ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.");
    defer head_ck.close();
    var L: Loader = .{ .gpa = gpa, .d = d, .stream = stream, .pack_dense = tools.pack_dense, .tools = tools, .w = &w, .ck = &ck };
    defer L.host.deinit(gpa);
    w.layers = try gpa.alloc(Layer, layers.len);
    for (layers, w.layers) |i, *l| {
        if (i >= c.layers) return error.BadLayer;
        l.* = try L.layer(i);
        ck.release(); // a rank reads about half the checkpoint: its pages would crowd the GPU's allocations out
    }
    try L.whole(try L.get("embed_tokens.weight", .{}), .bf16);
    w.embed = try L.put("embed", .{});
    try L.whole(try L.get("norm.weight", .{}), .bf16);
    w.norm = try L.put("norm", .{});
    try L.rows(try head_ck.get("lm_head.weight"), share.vocab.lo, share.vocab.hi, c.vocab, .bf16);
    w.head = try L.put("head", .{});
    if (with_mtp and c.mtp > 0) w.mtp = try L.mtp();
    return w;
}

test "bf16 rounding of fp32 is to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(@bitCast(@as(u32, 0x3f808000)))); // a tie to even (down)
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf16(@bitCast(@as(u32, 0x3f818000)))); // a tie to even (up)
    try std.testing.expectEqual(@as(u16, 0x3f81), toBf16(@bitCast(@as(u32, 0x3f808001))));
}

/// Every bf16 linear b16.matmul reads, retiled in place for glm_mm (the q4 draft head was made from the head already).
pub fn tileLinears(w: *const Weights, mk: *const mm_mod.Kernels, s: cuda.Stream, d: *const cuda.Driver) !void {
    var list: std.ArrayList([3]usize) = .empty; // (pointer, n, k)
    defer list.deinit(w.gpa);
    const add = struct {
        fn f(l: *std.ArrayList([3]usize), gpa: std.mem.Allocator, ptr: u64, n: usize, k: usize) !void {
            if (ptr != 0) try l.append(gpa, .{ ptr, n, k });
        }
    }.f;
    const kd = @import("cuda_kda.zig").dk;
    var layers: std.ArrayList(Layer) = .empty;
    defer layers.deinit(w.gpa);
    try layers.appendSlice(w.gpa, w.layers);
    if (w.mtp) |m| {
        try layers.append(w.gpa, m.layer);
        try add(&list, w.gpa, m.eh, w.config.hidden, 2 * w.config.hidden);
    }
    for (layers.items) |l| {
        switch (l.mixer) {
            .kda => |x| {
                const n = x.heads * kd;
                try add(&list, w.gpa, x.proj, x.width(), x.dims);
                try add(&list, w.gpa, x.fb, n, kd);
                try add(&list, w.gpa, x.gb, n, kd);
                try add(&list, w.gpa, x.o, x.dims, n);
            },
            .mla => |x| {
                try add(&list, w.gpa, x.proj, x.q_lora + x.lw, x.dims);
                try add(&list, w.gpa, x.q_b, x.heads * x.qk, x.q_lora);
                try add(&list, w.gpa, x.index.kw, x.index.d + x.index.heads, x.dims);
                try add(&list, w.gpa, x.index.qb, x.index.heads * x.index.d, x.q_lora);
                try add(&list, w.gpa, x.o, x.dims, x.heads * x.dv);
            },
        }
        switch (l.ffn) {
            .dense => |x| switch (x.w) {
                .bf16 => |b| {
                    try add(&list, w.gpa, b.gu, 2 * x.width, x.dims);
                    try add(&list, w.gpa, b.down, x.dims, x.width);
                },
                .nvfp4 => {},
            },
            .moe => |x| {
                try add(&list, w.gpa, x.shared_gu, 2 * x.shared, x.routed.dims);
                try add(&list, w.gpa, x.shared_down, x.routed.dims, x.shared);
            },
        }
    }
    try add(&list, w.gpa, w.head, w.share.vocab.len(), w.config.hidden);
    var most: usize = 0;
    for (list.items) |e| most = @max(most, e[1] * e[2] * 2);
    var tmp = try cuda.DeviceBuffer.alloc(d, most);
    defer tmp.free();
    errdefer @import("cuda_load_owner.zig").drain(s, d);
    for (list.items) |e| {
        try mm_mod.tile(mk, s, e[0], tmp.ptr, e[1], e[2]);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(e[0], tmp.ptr, e[1] * e[2] * 2, s.handle), "cuMemcpyDtoDAsync");
    }
    try s.synchronize();
}
