//! Inkling 2 backbone ops on Metal (modeling_inkling2 formulas): caller owns encoder, buffers, state; rows independent.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const exl3 = @import("../../core/exl3_projection.zig");
const mv = @import("../deepseek_v41/metal/mv.zig");
const mv_mma = @import("../deepseek_v41/metal/mv_mma.zig");
const dense = @import("metal_dense.zig");

/// A GPU buffer and the byte offset of a tensor in it (the type ExpertStorage and the EXL3 projections use).
pub const View = exl3.View;

pub const conv_taps = 4;
/// Rows of a conv history ring (positions modulo the ring); any ring of at least conv_taps works.
pub const conv_ring = 32;
pub const head_dim = 128;
pub const rel_dim = 16;
pub const max_sources = 17;
pub const max_queries = 3;
pub const max_hidden = 6144;
pub const max_group = 8;
pub const max_experts = 512;
pub const max_topk = 16;
/// Rows up to which attnRes runs on many groups (statistics, mix, norm), bit-equal to the one-group row kernel.
pub const small_rows = 16;
/// Scratch bytes ahead of the fp32 mixes: per-source statistics, sources * (1 + queries) floats a row.
const stats_bytes = 8192;
/// Keys a row's attention chunk covers (kernels/metal/inkling2/attention.metal's I2_CHUNK) and floats of its partial.
pub const attn_chunk = 2048;
pub const attn_partial_floats = 132;
/// Rows of a tile of attention_rows.metal's kernel, which full-attention calls of `attn_rows_min` rows or more use.
pub const attn_rows_tile = 8;
pub const attn_rows_min = 8;
/// Rows from which a projection runs on MMA tiles (bit-equal to the matvec): the matvec re-reads its weights per row.
pub const mma_min_rows = 3;

pub const Options = struct {
    /// Scratch for attention partials and router logits; one buffer, reused by consecutive dispatches.
    scratch_bytes: usize = 64 << 20,
    /// Bytes the dense tile permutes X into (rows * k * 2 a slice; more rows run in slices).
    dense_bytes: usize = 8 << 20,
};

const Kernel = enum {
    embed,
    rms_norm,
    copy_rows,
    attn_res,
    conv_static,
    conv_dynamic,
    conv_store,
    scaled_l2_clip,
    kv_prep,
    attn_merge,
    attn_rows_partial,
    attn_rows_final,
    route_select,
    softcap,
    residual_attn,
    residual_mlp,
    mv_fuse,
    attn_res_stats,
    attn_res_mix,
    attn_res_norm,
    dense_perm,
    dense_mma_bf16,
    dense_mma_f32,
};
const kernel_count = @typeInfo(Kernel).@"enum".field_names.len;

fn attnSymbol(comptime final: bool, comptime group: usize) [:0]const u8 {
    return std.fmt.comptimePrint("inkling2_attn_{s}_g{d}", .{ if (final) "final" else "partial", group });
}

/// Every symbol the library defines, in pipeline order (the registration list the contract's launch records name).
pub const symbols = blk: {
    var out: [kernel_count + 2 * max_group][:0]const u8 = undefined;
    for (@typeInfo(Kernel).@"enum".field_names, 0..) |name, i| out[i] = "inkling2_" ++ name;
    for (0..max_group) |g| {
        out[kernel_count + g] = attnSymbol(false, g + 1);
        out[kernel_count + max_group + g] = attnSymbol(true, g + 1);
    }
    break :blk out;
};

// Constants of the kernels, field for field (sizes are asserted below).
const NormArgs = extern struct { hidden: u32, rows: u32, vocab: u32, x_pitch: u32, eps: f32 };
const CopyArgs = extern struct { rows: u32, cols: u32, src_ld: u32, dst_ld: u32 };
const AttnResKernelArgs = extern struct {
    hidden: u32,
    blocks: u32,
    block_stride: u32,
    partial_f32: u32,
    queries: u32,
    out_stride: u32,
    partial_pitch: u32,
    pad: u32 = 0,
    eps_unweighted: f32,
    eps_norm: f32,
    scale: f32,
};
const ConvKernelArgs = extern struct {
    rows: u32,
    channels: u32,
    pos0: u32,
    ring: u32,
    kv_heads: u32,
    pad: [3]u32 = .{ 0, 0, 0 },
    divisor: f32,
};
const ClipKernelArgs = extern struct {
    rows: u32,
    heads: u32,
    dim: u32,
    pos0: u32,
    log_scaled: u32,
    floor_positions: u32,
    pad: [2]u32 = .{ 0, 0 },
    threshold: f32,
    alpha: f32,
};
const KvKernelArgs = extern struct { rows: u32, kv_heads: u32, dim: u32, pos0: u32, ring: u32, threshold: f32, eps: f32, pad: u32 = 0 };
const AttnKernelArgs = extern struct {
    rows: u32,
    pos0: u32,
    heads: u32,
    kv_heads: u32,
    extent: u32,
    window: u32,
    ring: u32,
    chunks: u32,
    scale: f32,
    pad: [3]u32 = .{ 0, 0, 0 },
};
const RouteKernelArgs = extern struct { rows: u32, experts: u32, topk: u32, pad: u32 = 0, scale: f32 };
const ElemKernelArgs = extern struct {
    rows: u32,
    cols: u32,
    in_ld: u32,
    out_ld: u32,
    has_block: u32,
    pad: [3]u32 = .{ 0, 0, 0 },
    cap: f32,
};

comptime {
    std.debug.assert(@sizeOf(NormArgs) == 20 and @sizeOf(CopyArgs) == 16 and @sizeOf(AttnResKernelArgs) == 44);
    std.debug.assert(@sizeOf(ConvKernelArgs) == 36 and @sizeOf(ClipKernelArgs) == 40 and @sizeOf(KvKernelArgs) == 32);
    std.debug.assert(@sizeOf(AttnKernelArgs) == 48 and @sizeOf(RouteKernelArgs) == 20 and @sizeOf(ElemKernelArgs) == 36);
}

pub const Error = error{ InvalidInkling2Op, Inkling2ScratchTooSmall };

pub const EmbedArgs = struct { ids: View, table: View, norm: View, out: View, rows: u32, vocab: u32, hidden: u32, eps: f32 };
pub const RmsNormArgs = struct { x: View, w: View, out: View, rows: u32, hidden: u32, eps: f32 };
/// A rows x cols block of bf16, pitches in elements; source and destination must not overlap.
pub const CopyRowsArgs = struct { src: View, src_ld: u32, dst: View, dst_ld: u32, rows: u32, cols: u32 };
/// Sources: `blocks` bf16 [blocks][rows][hidden], then `partial`; out q = norm_q(mix of query q) with `norms[q]`.
pub const AttnResArgs = struct {
    blocks: View,
    block_stride: u32,
    nb: u32,
    partial: View,
    partial_f32: bool,
    partial_ld: u32 = 0,
    queries: View,
    nq: u32,
    norms: [max_queries]View,
    out: View,
    out_stride: u32,
    rows: u32,
    hidden: u32,
    eps_unweighted: f32 = 1e-6,
    eps: f32,
};
/// y [rows][n] = bf16(x [rows][k] w[n][k]^T) (or fp32 when `f32_out`), fp32 chains; a row's bits do not depend on rows.
pub const LinearArgs = struct { x: View, w: View, y: View, rows: u32, n: u32, k: u32, f32_out: bool = false };
/// One existing weight and its existing output. The fused launch writes `y` and allocates nothing.
pub const Fused = struct { w: View, y: View, n: u32 };
const FuseArgs = extern struct { k: u32, count: u32, n0: u32, n1: u32, n2: u32, n3: u32, n4: u32, n5: u32, n6: u32, n7: u32 };
/// Router: logits = x w^T in fp32 (x bf16 [rows][hidden], w bf16 [experts][hidden]); ids u32, weights f32 per row.
pub const RouteArgs = struct {
    x: View,
    w: View,
    bias: View,
    ids: View,
    weights: View,
    rows: u32,
    hidden: u32,
    experts: u32,
    topk: u32,
    scale: f32,
};
/// x [rows][channels] bf16 conv input of positions pos0.., hist [ring][channels] bf16 history, w bf16 [channels][4].
pub const ConvStaticArgs = struct { x: View, hist: View, w: View, out: View, rows: u32, channels: u32, pos0: u32, ring: u32 = conv_ring };
/// static_w bf16 [4][channels]; dyn bf16 [rows][kv_heads * 4] (the weight projection's output).
pub const ConvDynamicArgs = struct {
    x: View,
    dyn: View,
    static_w: View,
    hist: View,
    out: View,
    rows: u32,
    channels: u32,
    kv_heads: u32,
    pos0: u32,
    ring: u32 = conv_ring,
};
pub const ConvStoreArgs = struct { x: View, hist: View, rows: u32, channels: u32, pos0: u32, ring: u32 = conv_ring };
/// Log scaling of a full-attention layer: tau = 1 + alpha * log(max((pos + 1) / floor, 1)) at position pos0 + row.
pub const Tau = struct { pos0: u32, alpha: f32, floor: u32 };
/// Scaled L2 clip of x [rows][heads][dim] bf16 by scale bf16 [heads][dim]; `out` may equal `x` (in place).
pub const ClipArgs = struct { x: View, scale: View, out: View, rows: u32, heads: u32, dim: u32, threshold: f32, tau: ?Tau = null };
pub const KvPrepArgs = struct {
    k: View,
    v: View,
    kcache: View,
    vcache: View,
    rows: u32,
    kv_heads: u32,
    dim: u32 = head_dim,
    pos0: u32,
    /// 0: slot = position; else slot = position % ring.
    ring: u32,
    threshold: f32,
    eps: f32,
};
/// proj bf16 [rel_dim][extent]; caches [slot][kv_heads][128]; q/out [rows][heads][128]; rel, sink, gate; window 0: all.
pub const AttentionArgs = struct {
    q: View,
    rel: View,
    sink: View,
    gate: View,
    kcache: View,
    vcache: View,
    proj: View,
    out: View,
    rows: u32,
    pos0: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32 = head_dim,
    d_rel: u32 = rel_dim,
    extent: u32,
    window: u32,
    ring: u32,
    scale: f32,
};
/// out = (has block ? block : 0) + conv, fp32 [rows][cols].
pub const ResidualAttnArgs = struct { conv: View, block: ?View, out: View, rows: u32, cols: u32 };
/// out = bf16(h + conv).
pub const ResidualMlpArgs = struct { h: View, conv: View, out: View, rows: u32, cols: u32 };
/// out f32 [rows][n] = tanh(y / cap) * cap, y bf16 [rows][n] rows `ldy` apart.
pub const SoftcapArgs = struct { y: View, out: View, rows: u32, n: u32, ldy: u32, cap: f32 };

pub const Ops = struct {
    pipes: [symbols.len]mtl.Pipeline,
    dense: mv.Pipelines,
    mma: mv_mma.Pipelines,
    scratch: mtl.Buffer,
    scratch_bytes: usize,
    /// X in the dense tile's fragment order (metal_dense.zig).
    stage: mtl.Buffer,
    stage_bytes: usize,
    /// Rows from which `linear` takes an MMA tile; the contract's controls raise it to keep faulted matvecs in reach.
    mma_min_rows: u32,

    pub fn load(device: mtl.Device, options: Options) !Ops {
        return loadSource(device, options, sources.inkling2_ops);
    }

    /// The same operators from other text (the contract's planted controls edit the source).
    pub fn loadSource(device: mtl.Device, options: Options, text: []const u8) !Ops {
        const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer library.deinit();
        var o: Ops = undefined;
        var loaded: usize = 0;
        errdefer for (o.pipes[0..loaded]) |p| p.deinit();
        for (symbols, 0..) |name, i| {
            o.pipes[i] = try mtl.Pipeline.init(device, library, name, false);
            loaded += 1;
        }
        o.dense = try mv.Pipelines.load(device);
        errdefer o.dense.deinit();
        o.mma = try mv_mma.Pipelines.load(device);
        errdefer o.mma.deinit();
        o.scratch_bytes = options.scratch_bytes;
        o.mma_min_rows = mma_min_rows;
        o.scratch = try device.buffer(options.scratch_bytes, mtl.ResourceOptions.private);
        errdefer o.scratch.deinit();
        o.stage_bytes = options.dense_bytes;
        o.stage = try device.buffer(options.dense_bytes, mtl.ResourceOptions.private);
        return o;
    }

    pub fn deinit(o: *Ops) void {
        for (o.pipes) |p| p.deinit();
        o.dense.deinit();
        o.mma.deinit();
        o.scratch.deinit();
        o.stage.deinit();
    }

    fn pipe(o: *const Ops, k: Kernel) mtl.Pipeline {
        return o.pipes[@backingInt(k)];
    }

    fn attnPipe(o: *const Ops, final: bool, group: u32) mtl.Pipeline {
        return o.pipes[kernel_count + (if (final) @as(usize, max_group) else 0) + group - 1];
    }

    pub fn embed(o: *const Ops, e: mtl.ComputeEncoder, a: EmbedArgs) !void {
        if (a.rows == 0 or a.hidden == 0 or a.hidden > max_hidden) return error.InvalidInkling2Op;
        e.setPipeline(o.pipe(.embed));
        bind(e, &.{ a.ids, a.table, a.norm, a.out });
        e.setValue(NormArgs{ .hidden = a.hidden, .rows = a.rows, .vocab = a.vocab, .x_pitch = a.hidden, .eps = a.eps }, 4);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn rmsNorm(o: *const Ops, e: mtl.ComputeEncoder, a: RmsNormArgs) !void {
        if (a.rows == 0 or a.hidden == 0 or a.hidden > max_hidden) return error.InvalidInkling2Op;
        e.setPipeline(o.pipe(.rms_norm));
        bind(e, &.{ a.x, a.w, a.out });
        e.setValue(NormArgs{ .hidden = a.hidden, .rows = a.rows, .vocab = 0, .x_pitch = a.hidden, .eps = a.eps }, 3);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn copyRows(o: *const Ops, e: mtl.ComputeEncoder, a: CopyRowsArgs) !void {
        if (a.rows == 0 or a.cols == 0 or a.src_ld < a.cols or a.dst_ld < a.cols) return error.InvalidInkling2Op;
        e.setPipeline(o.pipe(.copy_rows));
        bind(e, &.{ a.src, a.dst });
        e.setValue(CopyArgs{ .rows = a.rows, .cols = a.cols, .src_ld = a.src_ld, .dst_ld = a.dst_ld }, 2);
        e.dispatchThreads(mtl.Size.of(a.cols, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn attnRes(o: *const Ops, e: mtl.ComputeEncoder, a: AttnResArgs) !void {
        if (a.rows == 0 or a.nq == 0 or a.nq > max_queries or a.nb + 1 > max_sources or a.hidden == 0 or a.hidden > max_hidden)
            return error.InvalidInkling2Op;
        const args = AttnResKernelArgs{
            .hidden = a.hidden,
            .blocks = a.nb,
            .block_stride = a.block_stride,
            .partial_f32 = @intFromBool(a.partial_f32),
            .queries = a.nq,
            .out_stride = a.out_stride,
            .partial_pitch = if (a.partial_ld == 0) a.hidden else a.partial_ld,
            .eps_unweighted = a.eps_unweighted,
            .eps_norm = a.eps,
            .scale = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(a.hidden)))),
        };
        const norms = [3]View{ a.norms[0], a.norms[if (a.nq > 1) 1 else 0], a.norms[if (a.nq > 2) 2 else 0] };
        const mixed_floats = (a.nq - 1) * @as(usize, a.out_stride) + @as(usize, a.rows) * a.hidden;
        if (a.rows <= small_rows and stats_bytes + mixed_floats * 4 <= o.scratch_bytes) {
            const stats = View{ .buffer = o.scratch };
            const mixed = View{ .buffer = o.scratch, .offset = stats_bytes };
            const groups = std.math.divCeil(u32, a.hidden, 256) catch unreachable;
            e.setPipeline(o.pipe(.attn_res_stats));
            bind(e, &.{ a.blocks, a.partial, a.queries, stats });
            e.setValue(args, 4);
            e.dispatchGroups(mtl.Size.of(a.nb + 1, a.rows, 1), mtl.Size.of(256, 1, 1));
            e.setPipeline(o.pipe(.attn_res_mix));
            bind(e, &.{ a.blocks, a.partial, stats, mixed });
            e.setValue(args, 4);
            e.dispatchGroups(mtl.Size.of(groups, a.rows, 1), mtl.Size.of(256, 1, 1));
            e.setPipeline(o.pipe(.attn_res_norm));
            bind(e, &.{ mixed, norms[0], norms[1], norms[2], a.out });
            e.setValue(args, 5);
            e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        e.setPipeline(o.pipe(.attn_res));
        bind(e, &.{ a.blocks, a.partial, a.queries, norms[0], norms[1], norms[2], a.out });
        e.setValue(args, 7);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// x [rows][k] bf16 times w [n][k] bf16: the shared bf16 matvec (rows < 16) or a bit-equal MMA tile.
    pub fn linear(o: *const Ops, e: mtl.ComputeEncoder, a: LinearArgs) !void {
        try o.denseRows(e, a.x, a.w, a.y, a.rows, a.n, a.k, a.f32_out);
    }

    /// One-row GEMVs that share `x`. Same chains as the split matvec, one launch, the caller's buffers.
    pub fn linearFused(o: *const Ops, e: mtl.ComputeEncoder, x: View, k: u32, ins: []const Fused) !void {
        if (ins.len == 0 or ins.len > 8 or k == 0) return error.InvalidInkling2Op;
        var n = [8]u32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        var total: u32 = 0;
        for (ins, 0..) |in, i| {
            if (in.n == 0) return error.InvalidInkling2Op;
            n[i] = in.n;
            total += in.n;
        }
        e.setPipeline(o.pipe(.mv_fuse));
        e.setBuffer(x.buffer, x.offset, 0);
        for (0..8) |i| {
            const src = if (i < ins.len) ins[i].w else x;
            const dst = if (i < ins.len) ins[i].y else x;
            e.setBuffer(src.buffer, src.offset, @intCast(1 + i));
            e.setBuffer(dst.buffer, dst.offset, @intCast(9 + i));
        }
        e.setValue(FuseArgs{
            .k = k,
            .count = @intCast(ins.len),
            .n0 = n[0],
            .n1 = n[1],
            .n2 = n[2],
            .n3 = n[3],
            .n4 = n[4],
            .n5 = n[5],
            .n6 = n[6],
            .n7 = n[7],
        }, 17);
        e.dispatchGroups(mtl.Size.of(total, 1, 1), mtl.Size.of(32, 1, 1));
    }

    fn denseRows(o: *const Ops, e: mtl.ComputeEncoder, x: View, w: View, y: View, rows: u32, n: u32, k: u32, f32_out: bool) !void {
        if (rows == 0 or n == 0 or k == 0) return error.InvalidInkling2Op;
        if (rows >= o.mma_min_rows and k % dense.chunk == 0 and (x.offset | w.offset) % 16 == 0) {
            const call = dense.Call{ .x = x, .w = w, .y = y, .rows = rows, .n = n, .k = k, .f32_out = f32_out };
            try dense.encode(e, o.pipe(.dense_perm), o.pipe(if (f32_out) .dense_mma_f32 else .dense_mma_bf16), o.stage, o.stage_bytes, call);
        } else if (rows >= o.mma_min_rows and k >= 32) {
            const xr = mv_mma.Ref{ .buf = x.buffer, .off = x.offset };
            const wr = mv_mma.Ref{ .buf = w.buffer, .off = w.offset };
            const yr = mv_mma.Ref{ .buf = y.buffer, .off = y.offset };
            try mv_mma.encode(&o.mma, e, xr, wr, yr, .{ .k = k, .n = n, .rows = rows, .out = if (f32_out) .f32 else .bf16 });
        } else {
            const xr = mv.Ref{ .buf = x.buffer, .off = x.offset };
            const wr = mv.Ref{ .buf = w.buffer, .off = w.offset };
            const yr = mv.Ref{ .buf = y.buffer, .off = y.offset };
            try mv.encode(&o.dense, e, xr, wr, yr, .{ .k = k, .n = n, .rows = rows, .w = .bf16, .out = if (f32_out) .f32 else .bf16 });
        }
    }

    pub fn route(o: *const Ops, e: mtl.ComputeEncoder, a: RouteArgs) !void {
        if (a.rows == 0 or a.topk == 0 or a.topk > max_topk or a.topk > a.experts or a.experts > max_experts) return error.InvalidInkling2Op;
        if (@as(usize, a.rows) * a.experts * 4 > o.scratch_bytes) return error.Inkling2ScratchTooSmall;
        const logits = View{ .buffer = o.scratch };
        try o.denseRows(e, a.x, a.w, logits, a.rows, a.experts, a.hidden, true);
        e.setPipeline(o.pipe(.route_select));
        bind(e, &.{ logits, a.bias, a.ids, a.weights });
        e.setValue(RouteKernelArgs{ .rows = a.rows, .experts = a.experts, .topk = a.topk, .scale = a.scale }, 4);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }

    fn convArgs(rows: u32, channels: u32, pos0: u32, ring: u32, kv_heads: u32, divisor: f64) !ConvKernelArgs {
        if (rows == 0 or channels == 0 or ring < conv_taps) return error.InvalidInkling2Op;
        return .{ .rows = rows, .channels = channels, .pos0 = pos0, .ring = ring, .kv_heads = kv_heads, .divisor = @floatCast(divisor) };
    }

    pub fn convStatic(o: *const Ops, e: mtl.ComputeEncoder, a: ConvStaticArgs) !void {
        const args = try convArgs(a.rows, a.channels, a.pos0, a.ring, 1, @sqrt(@as(f64, conv_taps + 1)));
        e.setPipeline(o.pipe(.conv_static));
        bind(e, &.{ a.x, a.hist, a.w, a.out });
        e.setValue(args, 4);
        e.dispatchThreads(mtl.Size.of(a.channels, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn convDynamic(o: *const Ops, e: mtl.ComputeEncoder, a: ConvDynamicArgs) !void {
        if (a.kv_heads == 0 or a.channels % a.kv_heads != 0) return error.InvalidInkling2Op;
        const args = try convArgs(a.rows, a.channels, a.pos0, a.ring, a.kv_heads, @sqrt(@as(f64, 2 * conv_taps + 1)));
        e.setPipeline(o.pipe(.conv_dynamic));
        bind(e, &.{ a.x, a.hist, a.static_w, a.dyn, a.out });
        e.setValue(args, 5);
        e.dispatchThreads(mtl.Size.of(a.channels, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// Writes the newest min(rows, ring) input rows into the history; encode it after the conv that reads the old one.
    pub fn convStore(o: *const Ops, e: mtl.ComputeEncoder, a: ConvStoreArgs) !void {
        const args = try convArgs(a.rows, a.channels, a.pos0, a.ring, 1, 1);
        e.setPipeline(o.pipe(.conv_store));
        bind(e, &.{ a.x, a.hist });
        e.setValue(args, 2);
        e.dispatchThreads(mtl.Size.of(a.channels, @min(a.rows, a.ring), 1), mtl.Size.of(256, 1, 1));
    }

    pub fn scaledClip(o: *const Ops, e: mtl.ComputeEncoder, a: ClipArgs) !void {
        if (a.rows == 0 or a.heads == 0 or a.dim == 0 or a.dim > 256) return error.InvalidInkling2Op;
        const t = a.tau orelse Tau{ .pos0 = 0, .alpha = 0, .floor = 1 };
        e.setPipeline(o.pipe(.scaled_l2_clip));
        bind(e, &.{ a.x, a.scale, a.out });
        e.setValue(ClipKernelArgs{
            .rows = a.rows,
            .heads = a.heads,
            .dim = a.dim,
            .pos0 = t.pos0,
            .log_scaled = @intFromBool(a.tau != null),
            .floor_positions = t.floor,
            .threshold = a.threshold,
            .alpha = t.alpha,
        }, 3);
        e.dispatchGroups(mtl.Size.of(a.heads, a.rows, 1), mtl.Size.of(32, 1, 1));
    }

    /// Queries: x is the q projection, scale q_l2_clip.scale, threshold qk_clip, tau on full-attention layers.
    pub fn qPrep(o: *const Ops, e: mtl.ComputeEncoder, a: ClipArgs) !void {
        return o.scaledClip(e, a);
    }

    /// Relative states: x is the r projection, scale rel_l2_clip.scale, threshold rel_clip, dim d_rel.
    pub fn relPrep(o: *const Ops, e: mtl.ComputeEncoder, a: ClipArgs) !void {
        return o.scaledClip(e, a);
    }

    pub fn kvPrep(o: *const Ops, e: mtl.ComputeEncoder, a: KvPrepArgs) !void {
        if (a.rows == 0 or a.kv_heads == 0 or a.dim == 0 or a.dim > 256) return error.InvalidInkling2Op;
        e.setPipeline(o.pipe(.kv_prep));
        bind(e, &.{ a.k, a.v, a.kcache, a.vcache });
        e.setValue(KvKernelArgs{ .rows = a.rows, .kv_heads = a.kv_heads, .dim = a.dim, .pos0 = a.pos0, .ring = a.ring, .threshold = a.threshold, .eps = a.eps }, 4);
        e.dispatchGroups(mtl.Size.of(a.kv_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn attention(o: *const Ops, e: mtl.ComputeEncoder, a: AttentionArgs) !void {
        if (a.dim != head_dim or a.d_rel != rel_dim or a.rows == 0 or a.kv_heads == 0 or a.heads % a.kv_heads != 0 or
            a.heads / a.kv_heads > max_group or a.extent == 0) return error.InvalidInkling2Op;
        const group = a.heads / a.kv_heads;
        const last: u64 = @as(u64, a.pos0) + a.rows;
        const keys: u64 = if (a.window != 0) @min(last, a.window) else last;
        const chunks: u32 = @intCast((keys + attn_chunk - 1) / attn_chunk);
        const per_row: usize = @as(usize, a.heads) * chunks * attn_partial_floats * 4;
        const block: u32 = if (chunks == 1) a.rows else @intCast(@min(@as(usize, a.rows), o.scratch_bytes / per_row));
        if (block == 0) return error.Inkling2ScratchTooSmall;
        var r0: u32 = 0;
        while (r0 < a.rows) : (r0 += block) {
            const n = @min(block, a.rows - r0);
            const hd: usize = a.heads;
            const q = View{ .buffer = a.q.buffer, .offset = a.q.offset + r0 * hd * head_dim * 2 };
            const rel = View{ .buffer = a.rel.buffer, .offset = a.rel.offset + r0 * hd * rel_dim * 2 };
            const sink = View{ .buffer = a.sink.buffer, .offset = a.sink.offset + r0 * hd * 2 };
            const gate = View{ .buffer = a.gate.buffer, .offset = a.gate.offset + r0 * hd * head_dim * 2 };
            const out = View{ .buffer = a.out.buffer, .offset = a.out.offset + r0 * hd * head_dim * 2 };
            const args = AttnKernelArgs{
                .rows = n,
                .pos0 = a.pos0 + r0,
                .heads = a.heads,
                .kv_heads = a.kv_heads,
                .extent = a.extent,
                .window = a.window,
                .ring = a.ring,
                .chunks = chunks,
                .scale = a.scale,
            };
            const scratch = View{ .buffer = o.scratch };
            const tiled = a.window == 0 and a.ring == 0 and n >= attn_rows_min;
            e.setPipeline(if (tiled) o.pipe(if (chunks == 1) .attn_rows_final else .attn_rows_partial) else o.attnPipe(chunks == 1, group));
            bind(e, &.{ q, rel, sink, gate, a.kcache, a.vcache, a.proj, out, scratch });
            e.setValue(args, 9);
            if (tiled) {
                e.dispatchGroups(mtl.Size.of(chunks, a.heads, (n + attn_rows_tile - 1) / attn_rows_tile), mtl.Size.of(128, 1, 1));
            } else {
                e.dispatchGroups(mtl.Size.of(chunks, a.kv_heads, n), mtl.Size.of(128, 1, 1));
            }
            if (chunks > 1) {
                e.setPipeline(o.pipe(.attn_merge));
                bind(e, &.{ gate, out, scratch });
                e.setValue(args, 3);
                e.dispatchGroups(mtl.Size.of(a.heads, n, 1), mtl.Size.of(128, 1, 1));
            }
        }
    }

    pub fn residualAttn(o: *const Ops, e: mtl.ComputeEncoder, a: ResidualAttnArgs) !void {
        if (a.rows == 0 or a.cols == 0) return error.InvalidInkling2Op;
        e.setPipeline(o.pipe(.residual_attn));
        bind(e, &.{ a.conv, a.block orelse a.conv, a.out });
        e.setValue(ElemKernelArgs{ .rows = a.rows, .cols = a.cols, .in_ld = a.cols, .out_ld = a.cols, .has_block = @intFromBool(a.block != null), .cap = 0 }, 3);
        e.dispatchThreads(mtl.Size.of(a.cols, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn residualMlp(o: *const Ops, e: mtl.ComputeEncoder, a: ResidualMlpArgs) !void {
        if (a.rows == 0 or a.cols == 0) return error.InvalidInkling2Op;
        e.setPipeline(o.pipe(.residual_mlp));
        bind(e, &.{ a.h, a.conv, a.out });
        e.setValue(ElemKernelArgs{ .rows = a.rows, .cols = a.cols, .in_ld = a.cols, .out_ld = a.cols, .has_block = 0, .cap = 0 }, 3);
        e.dispatchThreads(mtl.Size.of(a.cols, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn softcap(o: *const Ops, e: mtl.ComputeEncoder, a: SoftcapArgs) !void {
        if (a.rows == 0 or a.n == 0 or a.ldy < a.n or !(a.cap > 0)) return error.InvalidInkling2Op;
        e.setPipeline(o.pipe(.softcap));
        bind(e, &.{ a.y, a.out });
        e.setValue(ElemKernelArgs{ .rows = a.rows, .cols = a.n, .in_ld = a.ldy, .out_ld = a.n, .has_block = 0, .cap = a.cap }, 2);
        e.dispatchThreads(mtl.Size.of(a.n, a.rows, 1), mtl.Size.of(256, 1, 1));
    }
};

fn bind(e: mtl.ComputeEncoder, views: []const View) void {
    for (views, 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
}
