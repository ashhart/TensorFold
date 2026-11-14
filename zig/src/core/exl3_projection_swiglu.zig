//! The EXL3 projection's dispatch plan, stages, and SwiGLU-plus-rotate entry.
const std = @import("std");
const mtl = @import("metal");
const core = @import("exl3_projection_core.zig");
const pipe = @import("exl3_projection_pipelines.zig");
const Pipelines = pipe.Pipelines;
const Matrix = core.Matrix;
const View = core.View;
const Rows = core.Rows;
const Args = core.Args;
const ExpertDescriptor = core.ExpertDescriptor;
const format = core.format;
const decoder = @import("exl3_projection_decode.zig");
const fmaStaticOn = core.fmaStaticOn;
const fmaLutOn = core.fmaLutOn;
const tableDecodeOn = core.tableDecodeOn;
const knobs = &core.knobs;
const splitOverride = core.splitOverride;
const fmaDenseOn = core.fmaDenseOn;
const denseDb = core.denseDb;
const skipPadOn = core.skipPadOn;
const funnelOnlyOn = core.funnelOnlyOn;
const warp32On = core.warp32On;
const funnelPrefillOn = core.funnelPrefillOn;
const scratchBytes = core.scratchBytes;
const one_row_splits = core.one_row_splits;
const moe_split_rows = core.moe_split_rows;
const moeSplitCount = core.moeSplitCount;
const bind = core.bind;
const shape = core.shape;
const rotatedBytes = core.rotatedBytes;

pub const ReduceArgs = extern struct { splits: u32, n: u32 };

/// Which body a projection call takes: the lane's own plan over the pack's shapes.
pub const Plan = struct {
    pipeline: mtl.Pipeline,
    splits: usize,
    group_cols: usize,
    height: usize,
    threads: usize,
    depth: usize,
};

pub fn planBody(p: *const Pipelines, args: Args, r: Rows) Plan {
    var a = args;
    const decode_nax = p.decode_nax != null and a.indexed == 1 and !knobs.scalar_experts;
    // The compact-decode knob: the one-row, 1 KB-threadgroup decode tile.
    const compact = p.decode_nax_c != null and decode_nax and (knobs.decode_compact or a.rows > 1);
    // The routed-table knob (default on): the routed decode body reads the codebook value from the resident table.
    const routed_table = knobs.routed_table;
    // The two-word funnel read is bit-identical at halves = 6.
    const funnel = knobs.funnel and a.halves == 6;
    const decode_pipe: ?mtl.Pipeline = if (decode_nax) blk: {
        if (compact) {
            if (routed_table and funnel and p.decode_nax_ctab_f != null) break :blk p.decode_nax_ctab_f;
            if (routed_table and p.decode_nax_ctab != null) break :blk p.decode_nax_ctab;
            break :blk p.decode_nax_c;
        }
        if (routed_table and p.decode_nax_tab != null) break :blk p.decode_nax_tab;
        break :blk p.decode_nax;
    } else null;
    const grouped_nax = a.indexed == 2 and a.rows % 64 == 0;
    const moe_splits: usize = if (decode_nax and a.rows <= moe_split_rows and p.reduce != null) moeSplitCount() else 1;
    // The register-only knob, off, restores the tensor-unit body for a same-binary A/B.
    const fma_off = knobs.fma_off;
    const fma = p.decode_fma != null and decode_nax and a.rows <= moe_split_rows and !fma_off;
    // The static state selection in the register-only body.
    const fma_static = p.decode_fma_static != null and decode_nax and a.rows <= moe_split_rows and !fma_off and fmaStaticOn();
    // The same body with the codebook read from the 1024-entry threadgroup LUT.
    const fma_static_lut = p.decode_fma_static_lut != null and decode_nax and a.rows <= moe_split_rows and !fma_off and
        fmaStaticOn() and fmaLutOn();
    // A very wide one-row matrix already has thousands of column tiles.
    const wide = a.n >= 16384;
    const wide_splits: usize = knobs.split_wide;
    // The one-row split order for every decode-width call, rows <= 8 included.
    const row_invariant = knobs.row_invariant;
    // The register-only body on the dense one-row path.
    const fma_dense_h8 = knobs.fma_dense_h8;
    const fma_dense_ok = p.decode_fma_dense != null and a.indexed == 0 and a.routed == 0 and a.rows <= 8 and
        a.n % 16 == 0 and a.k % 16 == 0 and (a.halves == 6 or a.halves == 10 or a.halves == 12 or
        (a.halves == 8 and fma_dense_h8));
    // The MULTI-ROW dense body -- R rows a threadgroup, so the weight decode is shared by the rows.
    const multi_rows: u32 = knobs.fma_dense_multi;
    // (the arm): THE CEIL HEIGHT: the multi body uses ceil(rows/R), so a five-row window still shares one decode.
    const multi_ceil = knobs.fma_dense_multi_ceil;
    const fma_dense_multi4 = fma_dense_ok and p.decode_fma_dense_multi4 != null and multi_rows == 4 and
        a.rows >= 4 and (multi_ceil or a.rows % 4 == 0);
    const fma_dense_multi2 = fma_dense_ok and p.decode_fma_dense_multi2 != null and multi_rows >= 2 and
        a.rows >= 2 and (multi_ceil or a.rows % 2 == 0) and (multi_rows == 2 or !fma_dense_multi4);
    const fma_dense = fma_dense_ok and fmaDenseOn() and !fma_dense_multi4 and !fma_dense_multi2;
    // T-tile exact half-FMA bodies (routed, one-row dense): the widest T up to the knob that keeps the group count.
    const tiles: u32 = if (knobs.fma_tiles == 0) 0 else blk: {
        var t: u32 = if (knobs.fma_tiles >= 4) 4 else if (knobs.fma_tiles >= 2) 2 else 1;
        while (t > 1 and ((@as(usize, a.n) + 16 * t - 1) / (16 * t)) * a.rows < knobs.fma_tile_groups) t /= 2;
        break :blk t;
    };
    const tile_pick: ?usize = switch (tiles) {
        1 => 0,
        2 => 1,
        4 => 2,
        else => null,
    };
    const fma_t = if (tile_pick) |ti| p.decode_fma_t[ti] != null and decode_nax and a.rows <= moe_split_rows and !fma_off else false;
    const fma_t_dense = if (tile_pick) |ti| p.decode_fma_t_dense[ti] != null and fma_dense and a.codebook == 2 else false;
    const splits: usize = if (fma or fma_t or fma_dense or fma_dense_multi2 or fma_dense_multi4) 1 else if (a.indexed == 0 and a.routed == 0 and (a.rows == 1 or (row_invariant and a.rows <= 8)) and p.reduce != null)
        @as(usize, splitOverride() orelse @as(u32, @intCast(if (wide) wide_splits else one_row_splits)))
    else
        moe_splits;
    a.splits = @intCast(splits);
    // Tile shape: rows a threadgroup covers.
    const nax_tile: usize = if (a.indexed != 0) (if (r.tile_rows != 0) r.tile_rows else 64) else knobs.nax_tile_rows;
    // A 256-row tile wants the narrower column tile so the threadgroup stays at 512 threads.
    const moe_cols = if (a.indexed != 0) knobs.moe_cols else 0;
    const nax_cols: usize = if (a.indexed != 0 and nax_tile >= 256) 64 else if (moe_cols != 0) moe_cols else (knobs.nax_cols orelse 64);
    const tile_id: usize = ((nax_tile / 64) - 1) * 2 + (nax_cols / 64 - 1);
    const nax_pick: ?mtl.Pipeline = p.prefill_nax_tiles[tile_id];
    // The funnel read in the routed prompt tile: halves = 6 is the packed width; another width takes the shipped path.
    const nax_funnel_pick: ?mtl.Pipeline = if (funnelPrefillOn() and a.indexed != 0 and tile_id == 0)
        p.prefill_nax_funnel
    else
        null;
    const nax_funnel = nax_funnel_pick != null;
    // The same tile with the grouping's padded rows skipped.
    const nax_skip_pad_pick: ?mtl.Pipeline = if (skipPadOn() and a.indexed != 0)
        (if (tile_id == 0)
            p.prefill_nax_skip_pad
        else if (tile_id == 2)
            (if (warp32On()) p.prefill_nax_warp32 else p.prefill_nax_skip_pad128)
        else
            null)
    else
        null;
    const nax_skip_pad = nax_skip_pad_pick != null;
    // The routed tile with the funnel path as its only path: only halves = 6 selects it.
    const nax_funnel_only_pick: ?mtl.Pipeline = if (funnelOnlyOn() and a.indexed != 0 and tile_id == 0 and
        nax_skip_pad_pick != null)
        p.prefill_nax_funnel_only
    else
        null;
    const nax_funnel_only = nax_funnel_only_pick != null;
    // (THE LEVER, default off): THE DENSE TILE AT EIGHT ROWS, so a wrapped conditioning piece stays on the tile.
    const dense8 = knobs.nax_dense8;
    // The gate is the CALLER's hint and the flag together: the hint alone moves nothing.
    const dense8_on = dense8 and r.dense8;
    const nax_min_rows: u32 = knobs.nax_min_rows;
    const nax = nax_pick != null and a.k % 64 == 0 and
        a.rows >= (if (dense8_on and a.indexed == 0 and a.routed == 0) @min(nax_min_rows, 8) else nax_min_rows) and
        ((a.indexed == 0 and a.routed == 0) or grouped_nax);
    // The dense 128x64 tile's double-buffered twin: one barrier a block.
    const nax_dense_db = p.prefill_nax_dense_db != null and nax and a.indexed == 0 and tile_id == 2 and denseDb();
    if (nax_dense_db and !pipe.nax_db_logged) {
        pipe.nax_db_logged = true;
        std.log.info("exl3: nax31 dense double-buffered body selected rows={d} tile={d}", .{ a.rows, nax_tile });
    }
    const shared = !nax and !decode_nax and p.prefill != null and a.rows >= knobs.shared_min_rows and a.indexed == 0 and a.routed == 0;
    // The table-decode knob: the one-row matmul body reads the codebook from the resident table, bit-identically.
    const matmul_tab = p.matmul_tab != null and a.indexed == 0 and a.routed == 0 and a.rows <= 8 and tableDecodeOn();
    // The same static body with the codebook read from the 1024-entry threadgroup LUT.
    const matmul_static_lut = p.matmul_static_lut != null and a.indexed == 0 and a.routed == 0 and a.rows == 1 and
        a.n <= 32768 and a.n % 128 == 0 and a.k % 16 == 0 and a.codebook == 2 and
        (a.halves == 6 or a.halves == 10 or a.halves == 12) and tableDecodeOn();
    if ((matmul_static_lut or matmul_tab) and !pipe.matmul_tab_logged) {
        pipe.matmul_tab_logged = true;
        std.log.info("exl3: one-row table body {s} rows={d} n={d} k={d} splits={d}", .{ if (matmul_static_lut) "static-lut" else "matmul", a.rows, a.n, a.k, splits });
    }
    const pipeline = (if (fma_t) p.decode_fma_t[tile_pick.?].? else if (fma_t_dense) p.decode_fma_t_dense[tile_pick.?].? else if (fma_static_lut) p.decode_fma_static_lut.? else if (fma_static) p.decode_fma_static.? else if (fma) p.decode_fma.? else if (fma_dense_multi4) p.decode_fma_dense_multi4.? else if (fma_dense_multi2) p.decode_fma_dense_multi2.? else if (fma_dense) p.decode_fma_dense.? else if (decode_pipe) |dp| dp else if (matmul_static_lut) p.matmul_static_lut.? else if (matmul_tab) p.matmul_tab.? else if (nax_dense_db) p.prefill_nax_dense_db.? else if (nax_funnel_only) nax_funnel_only_pick.? else if (nax_skip_pad) nax_skip_pad_pick.? else if (nax_funnel) nax_funnel_pick.? else if (nax) nax_pick.? else if (shared) p.prefill.? else p.matmul);
    // Rows a decode threadgroup takes: one a SIMDgroup (the row knob selects one a threadgroup).
    const moe_rows: usize = if (decode_nax) (if (compact) 1 else knobs.moe_rows) else 4;
    const group_rows: usize = if (fma or fma_static or fma_static_lut or fma_dense_multi2 or fma_dense_multi4) 1 else if (decode_nax) moe_rows else if (nax) nax_tile else if (shared) 32 else 8;
    // The geometry must follow the pipeline: the static one-row body is 128 columns and 256 threads.
    const group_cols: usize = if (fma_t or fma_t_dense) 16 * @as(usize, tiles) else if (fma or fma_static or fma_static_lut or fma_dense) 16 else if (matmul_static_lut) 128 else if (nax) nax_cols else if (decode_nax) 32 else 16;
    const height: usize = if (fma_dense_multi4) (a.rows + 3) / 4 else if (fma_dense_multi2) (a.rows + 1) / 2 else if (fma or fma_static or fma_static_lut or fma_dense) a.rows else if (splits > 1 and !decode_nax) splits else if (a.indexed == 1 and !decode_nax) a.rows else (@as(usize, a.rows) + group_rows - 1) / group_rows;
    // The warp32 tile runs four warps (a warp owns two sixteen-row groups).
    const nax_warp32: bool = nax and a.indexed != 0 and tile_id == 2 and warp32On() and nax_skip_pad_pick != null;
    const threads: usize = if (fma or fma_static or fma_static_lut or fma_dense or fma_dense_multi2 or fma_dense_multi4) 32 * p.fma_warps else if (nax) (if (nax_warp32) nax_tile / 32 * 32 * (nax_cols / 64) else nax_tile / 16 * 32 * (nax_cols / 64)) else if (decode_nax) 32 * moe_rows else if (matmul_static_lut) 256 else if (shared) 128 else 32;
    // One split dispatch over a window: a row group a threadgroup (the tile spans eight rows), the splits in y.
    const split_groups: usize = if (splits > 1 and !decode_nax) (a.rows + 7) / 8 else 1;
    if ((fma or fma_static or fma_static_lut) and core.fma_log_count < 8) {
        core.fma_log_count += 1;
        std.log.info("exl3: routed decode body rows={d} n={d} k={d} halves={d} cols={d} grid={d}x{d}x{d} threads={d} static={} fma={} experts={d}", .{ a.rows, a.n, a.k, a.halves, group_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, if (fma or fma_static) 1 else if (decode_nax) moe_splits else split_groups, threads, fma_static, fma, a.experts });
    }

    if (a.indexed == 0 and a.rows >= 64 and pipe.dense_log_count < 14) {
        pipe.dense_log_count += 1;
        const dpick: []const u8 = if (nax_dense_db) "nax31" else if (nax) "nax_tile" else if (shared) "shared" else "matmul";
        std.log.info("exl3: dense dispatch rows={d} n={d} k={d} tile={d} cols={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, nax_tile, nax_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, dpick });
    }
    // (the instrument): the dense calls the eight-row gate moves, named.
    if (dense8_on and a.indexed == 0 and a.routed == 0 and a.rows >= 8 and a.rows < 64 and pipe.dense8_log_count < 14) {
        pipe.dense8_log_count += 1;
        std.log.info("exl3: dense8 rows={d} n={d} k={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, if (nax) "nax_tile" else if (shared) "shared" else "matmul" });
    }
    if (a.indexed != 0 and !pipe.routed_logged) {
        pipe.routed_logged = true;
        const pick: []const u8 = if (nax_dense_db) "nax31" else if (nax_funnel_only) "nax59" else if (nax_skip_pad) (if (nax_warp32) "nax54" else "nax48") else if (nax_funnel) "nax42" else if (nax) "nax_tile" else if (shared) "shared" else if (decode_nax) "decode" else "matmul";
        std.log.info("exl3: indexed dispatch rows={d} n={d} k={d} tile={d} cols={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, nax_tile, nax_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, pick });
    }
    return .{
        .pipeline = pipeline,
        .splits = splits,
        .group_cols = group_cols,
        .height = height,
        .threads = threads,
        .depth = if (fma or fma_static or fma_static_lut or fma_dense) 1 else if (decode_nax) moe_splits else split_groups,
    };
}

/// Where a call leaves its f32 sums (`Rows.no_epilogue` stops there): past the call's rotated input rows.
pub fn sumsView(scratch: View, rows: usize, k: usize) View {
    return .{ .buffer = scratch.buffer, .offset = scratch.offset + ((rows * k * 2 + 15) & ~@as(usize, 15)) };
}

pub fn stages(p: *const Pipelines, e: mtl.ComputeEncoder, args: Args, trellis: View, su: View, sv: View, bias: View, descriptors: View, x: View, y: View, scratch: View, r: Rows) void {
    var a = args;
    const plan = planBody(p, a, r);
    a.splits = @intCast(plan.splits);
    const sums = sumsView(scratch, a.rows, a.k);
    const partials = View{ .buffer = scratch.buffer, .offset = sums.offset + ((@as(usize, a.rows) * a.n * 4 + 15) & ~@as(usize, 15)) };
    const routes = if (r.routes) |v| v.ids else scratch;
    const rotate_passes: usize = if (knobs.double_rotate) 2 else 1;
    if (!r.prerotated) {
        e.setPipeline(p.rotate_in);
        bind(e, x, 0);
        bind(e, su, 1);
        bind(e, scratch, 2);
        e.setValue(a, 3);
        bind(e, r.input_ids orelse scratch, 4);
        bind(e, routes, 5);
        bind(e, descriptors, 6);
        // Four 128-element blocks a threadgroup (512 of the row), so the pass needs a quarter of the threadgroups.
        const rotate_width: usize = (a.k / 128 + 3) / 4;
        for (0..rotate_passes) |_| e.dispatchGroups(.{ .width = rotate_width, .height = a.rows }, .{ .width = 128 });
        e.barrier();
    }
    e.setPipeline(plan.pipeline);
    bind(e, scratch, 0);
    bind(e, trellis, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, routes, 4);
    bind(e, descriptors, 5);
    bind(e, partials, 6);
    if (p.codebook_table) |t| e.setBuffer(t, 0, 7);
    // The double-gemm measurement repeats the dense GEMM dispatch (rotate passes outside the loop).
    const gemm_passes: usize = if (a.indexed == 0) blk: {
        if (knobs.double_gemm) break :blk 2;
        if (knobs.double_gemm_n) |want| {
            if (want != 0 and a.n == want) break :blk 2;
        }
        break :blk 1;
    } else (if (knobs.double_gemm_indexed) 2 else 1);
    for (0..gemm_passes) |_| e.dispatchGroups(.{ .width = (@as(usize, a.n) + plan.group_cols - 1) / plan.group_cols, .height = plan.height, .depth = plan.depth }, .{ .width = plan.threads });
    e.barrier();
    if (plan.splits > 1) {
        e.setPipeline(p.reduce.?);
        bind(e, partials, 0);
        bind(e, sums, 1);
        e.setValue(ReduceArgs{ .splits = @intCast(plan.splits), .n = @as(u32, @intCast(a.rows)) * a.n }, 2);
        e.dispatchThreads(.{ .width = @as(usize, a.rows) * a.n }, .{ .width = 256 });
        e.barrier();
    }
    if (r.no_epilogue) return;
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, sv, 1);
    bind(e, bias, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, r.output_ids orelse scratch, 5);
    bind(e, routes, 6);
    bind(e, descriptors, 7);
    const rotate_out_width: usize = (a.n / 128 + 3) / 4;
    // Instrument: the double-epilogue measurement repeats the scale+bias pass.
    const rotate_out_passes: usize = if (knobs.double_rotate_out) 2 else 1;
    for (0..rotate_out_passes) |_| e.dispatchGroups(.{ .width = rotate_out_width, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}

/// The routed SwiGLU plus the down projection's rotate_in in one pass.
pub const SwiGLURot = struct {
    rows: usize,
    k: usize,
    input_stride: usize,
    output_stride: usize,
    experts: usize,
    weighted: bool = true,
};

pub fn encodeSwiGLUPreRotate(p: *const Pipelines, e: mtl.ComputeEncoder, gate: View, up: View, weights: View, routes: View, destination: ?View, descriptors: View, rotated: View, d: SwiGLURot) void {
    const A = extern struct { rows: u32, k: u32, input_stride: u32, output_stride: u32, weighted: u32, scatter: u32, experts: u32, reserved: u32 };
    const args = A{
        .rows = @intCast(d.rows),
        .k = @intCast(d.k),
        .input_stride = @intCast(d.input_stride),
        .output_stride = @intCast(d.output_stride),
        .weighted = @intFromBool(d.weighted),
        .scatter = @intFromBool(destination != null),
        .experts = @intCast(d.experts),
        .reserved = 0,
    };
    e.setPipeline(p.swiglu_prerotate);
    bind(e, gate, 0);
    bind(e, up, 1);
    bind(e, weights, 2);
    bind(e, routes, 3);
    bind(e, destination orelse routes, 4);
    bind(e, descriptors, 5);
    bind(e, rotated, 6);
    e.setValue(args, 7);
    e.dispatchGroups(.{ .width = (d.k / 128 + 3) / 4, .height = d.rows }, .{ .width = 128 });
}

/// The gate+up call's rotate_out and the routed and shared SwiGLU plus the down rotate_in in one pass.
pub const GateUpRot = struct {
    /// Routed rows: tokens times picks; the call's gate rows lead, its up rows follow, the shared pair comes last.
    rows: usize,
    shared_rows: usize,
    k: usize,
    experts: usize,
    fused_experts: usize,
    shared_dst: usize,
    weighted: bool = true,
    /// Written to the id row after the routed picks for each shared row, for the down call's table: none leaves them.
    shared_id: ?u32 = null,
};

pub fn encodeGateUpSwiGLU(p: *const Pipelines, e: mtl.ComputeEncoder, sums: View, fused_ids: View, weights: View, routes: View, fused: View, down: View, shared_down: View, rotated: View, d: GateUpRot) void {
    const A = extern struct { rows: u32, shared_rows: u32, k: u32, sums_stride: u32, experts: u32, fused_experts: u32, shared_dst: u32, weighted: u32, shared_id: u32 };
    const args = A{
        .rows = @intCast(d.rows),
        .shared_rows = @intCast(d.shared_rows),
        .k = @intCast(d.k),
        .sums_stride = @intCast(d.k),
        .experts = @intCast(d.experts),
        .fused_experts = @intCast(d.fused_experts),
        .shared_dst = @intCast(d.shared_dst),
        .weighted = @intFromBool(d.weighted),
        .shared_id = d.shared_id orelse std.math.maxInt(u32),
    };
    e.setPipeline(p.gateup_swiglu_prerotate);
    bind(e, sums, 0);
    bind(e, fused_ids, 1);
    bind(e, weights, 2);
    bind(e, routes, 3);
    bind(e, fused, 4);
    bind(e, down, 5);
    bind(e, shared_down, 6);
    bind(e, rotated, 7);
    e.setValue(args, 8);
    e.dispatchGroups(.{ .width = (d.k / 128 + 3) / 4, .height = d.rows + d.shared_rows }, .{ .width = 128 });
}

/// The routed picks' and the shared expert's down rows combined: the call's rotate_out and the weighted-sum's end.
pub const DownCombine = struct {
    tokens: usize,
    picks: usize,
    width: usize,
    /// The table entry of the shared row (the routed count): its descriptor carries that row's scales and bias.
    experts: usize,
    y_stride: usize,
    /// The sums row of token 0's shared output; the routed rows are token * picks + slot from row 0.
    shared_row: usize,
};

pub fn encodeDownCombine(p: *const Pipelines, e: mtl.ComputeEncoder, sums: View, ids: View, descriptors: View, y: View, d: DownCombine) void {
    const A = extern struct { rows: u32, picks: u32, width: u32, sums_stride: u32, experts: u32, y_stride: u32, shared_row: u32, reserved: u32 };
    const args = A{
        .rows = @intCast(d.tokens),
        .picks = @intCast(d.picks),
        .width = @intCast(d.width),
        .sums_stride = @intCast(d.width),
        .experts = @intCast(d.experts),
        .y_stride = @intCast(d.y_stride),
        .shared_row = @intCast(d.shared_row),
        .reserved = 0,
    };
    e.setPipeline(p.down_combine);
    bind(e, sums, 0);
    bind(e, ids, 1);
    bind(e, descriptors, 2);
    bind(e, y, 3);
    e.setValue(args, 4);
    e.dispatchGroups(.{ .width = d.width / 128, .height = d.tokens }, .{ .width = 256 });
}

test "EXL3 scratch is row compact and bounded independently of full matrices" {
    // 214,016 was the pre-split-K bound; the formula now adds the split-K partials and the fused gate+up split room.
    try std.testing.expectEqual(@as(usize, 1025024), try scratchBytes(5120, 2304, 11));
    try std.testing.expectError(error.InvalidExl3Rows, scratchBytes(128, 128, 0));
    try std.testing.expectError(error.UnsupportedExl3Shape, scratchBytes(129, 128, 1));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(ExpertDescriptor));
    // The Args record counts the defaulted fields.
    try std.testing.expectEqual(@as(usize, 104), @sizeOf(Args));
}

test "stored per-matrix width and rectangular strides include public target shapes" {
    var m = Matrix{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = 5120, .n = 2304, .width = .{ .halves = 6 } };
    try m.validate();
    try std.testing.expectEqual(@as(usize, 4423680), try m.storageBytes());
    m.width.halves = 5;
    try m.validate();
    m.codebook = .mcg;
    try std.testing.expectError(error.UnsupportedExl3Width, m.validate());
    m.width.halves = 16;
    m.k = 1280;
    m.n = 4096;
    try m.validate();
    m.tile_n_stride = 258;
    try m.validate();
    m.tile_k_stride = 256;
    try std.testing.expectError(error.InvalidExl3Stride, m.validate());
}

test "invalid and overflowing extents fail before storage arithmetic can wrap" {
    var m = Matrix{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = 0, .n = 128, .width = .{ .halves = 6 } };
    try std.testing.expectError(error.UnsupportedExl3Shape, m.storageBytes());
    m.k = 128;
    m.n = 256;
    m.tile_n_stride = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, m.validate());
    m.tile_n_stride = m.tileBytes();
    m.tile_k_stride = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, m.storageBytes());
    try std.testing.expectError(error.InvalidExl3Rows, scratchBytes(128, 128, std.math.maxInt(u32)));
    try std.testing.expectError(error.Overflow, scratchBytes(0xffffff80, 0xffffff80, 0xffffff80));
}

test {
    _ = decoder;
    _ = @import("exl3_rect_test.zig");
}
