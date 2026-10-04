//! Kimi K3's Metal pipelines and how each is dispatched; argument structs mirror zig/kernels/metal/kimi.
const std = @import("std");
const mtl = @import("metal");

/// A buffer binding: the buffer and a byte offset into it.
pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, bytes: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }
};

/// w_stride 0 means K (whole rows); a row-split shard sets the full row length.
pub const RowsArgs = extern struct { K: u32, N: u32, rows: u32, x_stride: u32, y_stride: u32, w_stride: u32 = 0, ls: u32 = 8, slice0: u32 = 0, beta: f32 = 0, lin: f32 = 0 };
pub const ResArgs = extern struct { D: u32, rows: u32, blocks: u32, flags: u32, eps: f32 };
pub const NormArgs = extern struct { D: u32, x_stride: u32, y_stride: u32, eps: f32 };
pub const RouteArgs = extern struct { experts: u32, topk: u32 };
pub const PlanArgs = extern struct { pairs: u32, experts: u32, first: u32, last: u32 };
pub const XpArgs = extern struct { K: u32, N: u32, topk: u32, rows_max: u32, beta: f32, lin: f32 };
pub const KdaArgs = extern struct { heads: u32, head0: u32, proj_stride: u32, gate_stride: u32, out_stride: u32, log_rows: u32, lower_bound: f32, eps: f32 };
pub const KdaSeg = extern struct { first: u32, rows: u32, kept: u32, flags: u32, slot: u32 };
pub const KdaState = extern struct { S: u64, conv: u64, log_a: u64, log_k: u64, log_u: u64, log_x: u64 };
pub const KdaWeights = extern struct { conv_q: u64, conv_k: u64, conv_v: u64, A_log: u64, dt_bias: u64, o_norm: u64 };
pub const MlaRow = extern struct { slot: u32, pos: u32 };
pub const MlaArgs = extern struct { cache: u64, slot_keys: u32, heads: u32, head0: u32, rows: u32, q_stride: u32, kv_stride: u32, gate_stride: u32, out_stride: u32, eps: f32, scale: f32 };
pub const FillArgs = extern struct { seed: u32, count: u32, kind: u32, e0: u32 };
pub const ExpertPtrs = extern struct { w1p: u64, w1s: u64, w3p: u64, w3s: u64, w2p: u64, w2s: u64 };

pub const res_delta: u32 = 1;
pub const res_prefix: u32 = 2;
pub const res_append: u32 = 4;
pub const seg_commit: u32 = 1;
pub const mla_splits: u32 = 16;
pub const fill_w: u32 = 0;
pub const fill_one: u32 = 1;
pub const fill_u8: u32 = 2;
pub const fill_e8: u32 = 3;

/// Rows a pass of the scalar row kernels; from `mma_rows` rows the MMA kernels give the same bits faster.
const row_blocks = [_]u32{ 1, 2, 4, 8 };
pub const mma_rows: u32 = 16;

/// Floats in the MMA path's slice scratch for `rows` rows.
pub fn partFloats(rows: u32) usize {
    return 8 * @as(usize, (rows + 31) / 32 * 32) * 16384;
}

fn rowBlock(rows: u32) usize {
    for (row_blocks, 0..) |rb, i| if (rows <= rb) return i;
    return row_blocks.len - 1;
}

pub const Kernels = struct {
    rows_bf16: [4]mtl.Pipeline,
    rows_f32: [4]mtl.Pipeline,
    glus: [4]mtl.Pipeline,
    slice_mma: mtl.Pipeline,
    slice_out_bf16: mtl.Pipeline,
    slice_out_f32: mtl.Pipeline,
    interleave: mtl.Pipeline,
    res_norm: mtl.Pipeline,
    rms_norm: mtl.Pipeline,
    add2: mtl.Pipeline,
    embed: mtl.Pipeline,
    router: mtl.Pipeline,
    argmax: mtl.Pipeline,
    route_plan: mtl.Pipeline,
    xp_up: [2]mtl.Pipeline,
    xp_down: [2]mtl.Pipeline,
    xp_combine: mtl.Pipeline,
    xp_combine_f32: mtl.Pipeline,
    kda: mtl.Pipeline,
    mla_cache: mtl.Pipeline,
    mla_qlat: mtl.Pipeline,
    mla_attend: mtl.Pipeline,
    mla_merge: mtl.Pipeline,
    mla_uv: mtl.Pipeline,
    fill_bf16: mtl.Pipeline,
    fill_f32: mtl.Pipeline,
    fill_u8: mtl.Pipeline,
    fp4_dequant: mtl.Pipeline,
    mma_peak: mtl.Pipeline,

    pub fn init(device: mtl.Device, lib: mtl.Library) !Kernels {
        var k: Kernels = undefined;
        const info = @typeInfo(Kernels).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Pipeline) @field(k, name) = try mtl.Pipeline.init(device, lib, "k3_" ++ name, false);
        }
        for (row_blocks, 0..) |rb, i| {
            var buf: [32]u8 = undefined;
            k.rows_bf16[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&buf, "k3_rows_bf16_r{d}", .{rb}), false);
            k.rows_f32[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&buf, "k3_rows_f32_r{d}", .{rb}), false);
            k.glus[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&buf, "k3_glu_r{d}", .{rb}), false);
        }
        for ([_]u32{ 1, 4 }, 0..) |rp, i| {
            var buf: [32]u8 = undefined;
            k.xp_up[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&buf, "k3_xp_up_r{d}", .{rp}), false);
            k.xp_down[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&buf, "k3_xp_down_r{d}", .{rp}), false);
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        const info = @typeInfo(Kernels).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Pipeline) @field(k, name).deinit() else for (&@field(k, name)) |*p| p.deinit();
        }
    }

    /// y[r, n] = x[r] . w[n] for rows r (bf16 rounded, or fp32 when `f32_out`); `w` slice-interleaved, `part` scratch.
    pub fn rows(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, part: Ref, f32_out: bool, args: RowsArgs) void {
        k.rowsBy(e, x, w, y, part, f32_out, args, args.rows >= mma_rows);
    }

    /// `rows` on the kernel chosen by `mma` (both give the same bits); shapes the MMA path cannot tile stay scalar.
    pub fn rowsBy(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, part: Ref, f32_out: bool, args: RowsArgs, mma: bool) void {
        var a = args;
        if (a.w_stride == 0) a.w_stride = a.K;
        if (mma and a.K % 256 == 0 and a.N % 64 == 0) return k.sliced(e, x, w, null, y, part, f32_out, a);
        bind(e, .{ x, w, y });
        e.setPipeline(if (f32_out) k.rows_f32[rowBlock(a.rows)] else k.rows_bf16[rowBlock(a.rows)]);
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of((a.N + 15) / 16, 1, 1), mtl.Size.of(128, 1, 1));
    }

    /// act[r, n] = situ(bf16(x . g[n]), bf16(x . u[n])), bf16; `g` and `u` slice-interleaved, `part` scratch.
    pub fn glu(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, g: Ref, u: Ref, y: Ref, part: Ref, args: RowsArgs) void {
        k.gluBy(e, x, g, u, y, part, args, args.rows >= mma_rows);
    }

    /// `glu` on the kernel chosen by `mma` (both give the same bits).
    pub fn gluBy(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, g: Ref, u: Ref, y: Ref, part: Ref, args: RowsArgs, mma: bool) void {
        var a = args;
        if (a.w_stride == 0) a.w_stride = a.K;
        if (mma and a.K % 256 == 0 and a.N % 32 == 0) return k.sliced(e, x, g, u, y, part, false, a);
        bind(e, .{ x, g, u, y });
        e.setValue(a, 4);
        e.setPipeline(k.glus[rowBlock(a.rows)]);
        e.dispatchGroups(mtl.Size.of((a.N + 15) / 16, 1, 1), mtl.Size.of(128, 1, 1));
    }

    /// Slices [slice0, slice0 + n) of a row-split projection as fp32 partials P[s][r][n] (a rank's share of the canon).
    pub fn slicePartials(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, part: Ref, args: RowsArgs, slice0: u32, n: u32) void {
        var a = args;
        if (a.w_stride == 0) a.w_stride = a.K;
        a.slice0 = slice0;
        const zero: u32 = 0;
        e.setPipeline(k.slice_mma);
        bind(e, .{ x, w, w, part });
        e.setValue(a, 4);
        e.setValue(zero, 5);
        e.dispatchGroups(mtl.Size.of(n, a.N / 64, (a.rows + 31) / 32), mtl.Size.of(128, 1, 1));
    }

    /// The MMA path in output chunks that fit `part` (8 slices x rows x 16384 floats): slice chains, then their tree.
    fn sliced(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w0: Ref, w1: ?Ref, y: Ref, part: Ref, f32_out: bool, args: RowsArgs) void {
        const glu_on: u32 = @intFromBool(w1 != null);
        const chunk: u32 = if (w1 != null) 8192 else 16384;
        const tile: u32 = if (w1 != null) 32 else 64;
        const ysize: usize = if (f32_out) 4 else 2;
        var c0: u32 = 0;
        while (c0 < args.N) : (c0 += chunk) {
            var a = args;
            a.N = @min(chunk, args.N - c0);
            const woff = @as(usize, c0) * a.w_stride * 2;
            e.setPipeline(k.slice_mma);
            bind(e, .{ x, w0.at(woff), (w1 orelse w0).at(woff), part });
            e.setValue(a, 4);
            e.setValue(glu_on, 5);
            e.dispatchGroups(mtl.Size.of(8, a.N / tile, (a.rows + 31) / 32), mtl.Size.of(128, 1, 1));
            e.setPipeline(if (f32_out) k.slice_out_f32 else k.slice_out_bf16);
            bind(e, .{ part, y.at(c0 * ysize) });
            e.setValue(a, 2);
            e.setValue(glu_on, 3);
            e.dispatchThreads(mtl.Size.of(a.N, a.rows, 1), mtl.Size.of(64, 4, 1));
        }
    }

    /// dst = `src` [N, K] with row blocks interleaved by slice, the layout the row kernels read whole cache lines from.
    pub fn interleaved(k: *const Kernels, e: mtl.ComputeEncoder, src: Ref, dst: Ref, N: u32, K: u32) void {
        e.setPipeline(k.interleave);
        bind(e, .{ src, dst });
        e.setValue([2]u32{ K, N }, 2);
        e.dispatchThreads(mtl.Size.of(K / 8, N, 1), mtl.Size.of(64, 1, 1));
    }

    /// Residual add, attention residual over the round's block entries, then a KimiRMSNorm: one row a threadgroup.
    pub fn resNorm(k: *const Kernels, e: mtl.ComputeEncoder, p: Ref, delta: Ref, blocks: Ref, w_norm: Ref, w_proj: Ref, w_out: Ref, out: Ref, a: ResArgs) void {
        e.setPipeline(k.res_norm);
        bind(e, .{ p, delta, blocks, w_norm, w_proj, w_out, out });
        e.setValue(a, 7);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn rmsNorm(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows_n: u32, a: NormArgs) void {
        e.setPipeline(k.rms_norm);
        bind(e, .{ x, w, y });
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(rows_n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// out = bf16(a + b) over `count` values (a multiple of 4).
    pub fn add(k: *const Kernels, e: mtl.ComputeEncoder, a_ref: Ref, b_ref: Ref, out: Ref, count: u32) void {
        e.setPipeline(k.add2);
        bind(e, .{ a_ref, b_ref, out });
        e.setValue(count / 4, 3);
        e.dispatchThreads(mtl.Size.of(count / 4, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn embedRows(k: *const Kernels, e: mtl.ComputeEncoder, ids: Ref, table: Ref, out: Ref, rows_n: u32, D: u32) void {
        e.setPipeline(k.embed);
        bind(e, .{ ids, table, out });
        e.setValue(D, 3);
        e.dispatchThreads(mtl.Size.of(D / 4, rows_n, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn routeTopk(k: *const Kernels, e: mtl.ComputeEncoder, logits: Ref, bias: Ref, ids: Ref, weights: Ref, rows_n: u32, a: RouteArgs) void {
        e.setPipeline(k.router);
        bind(e, .{ logits, bias, ids, weights });
        e.setValue(a, 4);
        e.dispatchGroups(mtl.Size.of(rows_n, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    pub fn greedy(k: *const Kernels, e: mtl.ComputeEncoder, logits: Ref, out: Ref, rows_n: u32, V: u32) void {
        e.setPipeline(k.argmax);
        bind(e, .{ logits, out });
        e.setValue(V, 2);
        e.dispatchGroups(mtl.Size.of(rows_n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// Group a round's (row, rank) pairs by local expert; slots and pair positions stay on the GPU.
    pub fn plan(k: *const Kernels, e: mtl.ComputeEncoder, ids: Ref, slots: Ref, nslots: Ref, pairs: Ref, where: Ref, a: PlanArgs) void {
        e.setPipeline(k.route_plan);
        bind(e, .{ ids, slots, nslots, pairs, where });
        e.setValue(a, 5);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    /// Every slot's expert on its pairs: gate|up with SiTU (`down` false) or down; `max_slots` bounds the grid.
    pub fn experts(k: *const Kernels, e: mtl.ComputeEncoder, down: bool, x: Ref, table: Ref, slots: Ref, nslots: Ref, pairs: Ref, out: Ref, many: bool, max_slots: u32, a: XpArgs) void {
        e.setPipeline(if (down) k.xp_down[@intFromBool(many)] else k.xp_up[@intFromBool(many)]);
        bind(e, .{ x, table, slots, nslots, pairs, out });
        e.setValue(a, 6);
        e.dispatchGroups(mtl.Size.of(a.N / 32, max_slots, 1), mtl.Size.of(256, 1, 1));
    }

    /// The routed sum in the cluster's order: a node's aligned groups [group0, group0 + ngroups) as their subtree.
    pub fn combine(k: *const Kernels, e: mtl.ComputeEncoder, out: Ref, where: Ref, weights: Ref, y: Ref, ids: Ref, rows_n: u32, f32_out: bool, per_group: u32, group0: u32, ngroups: u32, a: XpArgs) void {
        e.setPipeline(if (f32_out) k.xp_combine_f32 else k.xp_combine);
        bind(e, .{ out, where, weights, y, ids });
        e.setValue(a, 5);
        e.setValue([3]u32{ per_group, group0, ngroups }, 6);
        e.dispatchThreads(mtl.Size.of(a.N, rows_n, 1), mtl.Size.of(256, 1, 1));
    }

    /// KDA over a round: a threadgroup per (local head, segment), rows in order inside.
    pub fn kdaRound(k: *const Kernels, e: mtl.ComputeEncoder, proj: Ref, f: Ref, braw: Ref, g2: Ref, y: Ref, segs: Ref, nseg: u32, s: KdaState, w: KdaWeights, a: KdaArgs) void {
        e.setPipeline(k.kda);
        bind(e, .{ proj, f, braw, g2, y, segs });
        e.setValue(s, 6);
        e.setValue(w, 7);
        e.setValue(a, 8);
        e.dispatchGroups(mtl.Size.of(a.heads, nseg, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn mlaCache(k: *const Kernels, e: mtl.ComputeEncoder, kv: Ref, w: Ref, rows_ref: Ref, a: MlaArgs) void {
        e.setPipeline(k.mla_cache);
        bind(e, .{ kv, w, rows_ref });
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn mlaQlat(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, kvb: Ref, qlat: Ref, a: MlaArgs) void {
        e.setPipeline(k.mla_qlat);
        bind(e, .{ q, kvb, qlat });
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(2, a.heads, 1), mtl.Size.of(256, 1, 1));
    }

    /// Up to 32 heads a threadgroup (a simdgroup a head) share each staged key tile: a TP-4 node reads a row's keys once.
    pub fn mlaAttend(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, qlat: Ref, rows_ref: Ref, partial: Ref, a: MlaArgs) void {
        e.setPipeline(k.mla_attend);
        bind(e, .{ q, qlat, rows_ref, partial });
        e.setValue(a, 4);
        const fit: u32 = @intCast(@min(32, k.mla_attend.maxThreads() / 32));
        const groups = (a.heads + fit - 1) / fit;
        const per = (a.heads + groups - 1) / groups;
        e.dispatchGroups(mtl.Size.of(groups, mla_splits, a.rows), mtl.Size.of(32 * per, 1, 1));
    }

    pub fn mlaMerge(k: *const Kernels, e: mtl.ComputeEncoder, partial: Ref, rows_ref: Ref, olat: Ref, a: MlaArgs) void {
        e.setPipeline(k.mla_merge);
        bind(e, .{ partial, rows_ref, olat });
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(a.heads, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn mlaUv(k: *const Kernels, e: mtl.ComputeEncoder, olat: Ref, kvb: Ref, gate: Ref, y: Ref, a: MlaArgs) void {
        e.setPipeline(k.mla_uv);
        bind(e, .{ olat, kvb, gate, y });
        e.setValue(a, 4);
        e.dispatchGroups(mtl.Size.of(128 / 4, a.heads, 1), mtl.Size.of(128, 1, 1));
    }

    /// Synthetic values (test data): kind fill_w/fill_one into bf16 or f32, fill_u8/fill_e8 into bytes.
    pub fn fill(k: *const Kernels, e: mtl.ComputeEncoder, out: Ref, dtype: enum { bf16, f32, u8 }, a: FillArgs) void {
        e.setPipeline(switch (dtype) {
            .bf16 => k.fill_bf16,
            .f32 => k.fill_f32,
            .u8 => k.fill_u8,
        });
        e.setBuffer(out.buf, out.off, 0);
        e.setValue(a, 1);
        e.dispatchThreads(mtl.Size.of(a.count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn dequant(k: *const Kernels, e: mtl.ComputeEncoder, packed_ref: Ref, scales: Ref, out: Ref, N: u32, K: u32) void {
        e.setPipeline(k.fp4_dequant);
        bind(e, .{ packed_ref, scales, out });
        e.setValue(K, 3);
        e.dispatchThreads(mtl.Size.of(K / 2, N, 1), mtl.Size.of(256, 1, 1));
    }
};

fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// A buffer's GPU address plus a byte offset, for the argument structs that carry pointers.
pub fn addr(r: Ref) u64 {
    return r.buf.gpuAddress() + r.off;
}
