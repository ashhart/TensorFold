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
const envRows = core.envRows;
const fmaStaticOn = core.fmaStaticOn;
const fmaLutOn = core.fmaLutOn;
const tableDecodeOn = core.tableDecodeOn;
const wideDecodeOn = core.wideDecodeOn;
const directDecodeOn = core.directDecodeOn;
const splitOverride = core.splitOverride;
const fmaDenseOn = core.fmaDenseOn;
const staticDecodeOn = core.staticDecodeOn;
const lutDecodeOn = core.lutDecodeOn;
const denseDb = core.denseDb;
const denseRpg2 = core.denseRpg2;
const skipPadOn = core.skipPadOn;
const wideWordsOn = core.wideWordsOn;
const funnelOnlyOn = core.funnelOnlyOn;
const denseFunnelOn = core.denseFunnelOn;
const denseFunnel8On = core.denseFunnel8On;
const coopPrefillOn = core.coopPrefillOn;
const warp32On = core.warp32On;
const funnelPrefillOn = core.funnelPrefillOn;
const moeDb = core.moeDb;
const scratchBytes = core.scratchBytes;
const splitCount = core.splitCount;
const one_row_splits = core.one_row_splits;
const moe_split_rows = core.moe_split_rows;
const moeSplitCount = core.moeSplitCount;
const bind = core.bind;
const shape = core.shape;
const rotatedBytes = core.rotatedBytes;

pub const ReduceArgs = extern struct { splits: u32, n: u32 };

/// Which body a projection call takes: the lane's own plan over the pack's shapes, the resident
/// pipelines and the call's hints. The plan names the pipeline, the split count and the dispatch
/// geometry; `stages` only encodes it.
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
    const matvec_nax = p.matvec_nax != null and a.rows == 1 and a.indexed == 0 and a.routed == 0 and
        std.c.getenv("TF_DSV41_EXL3_MATVEC") != null;
    const decode_nax = p.decode_nax != null and a.indexed == 1 and std.c.getenv("TF_DSV41_EXL3_SCALAR_EXPERTS") == null;
    // TF_DSV41_EXL3_DECODE_COMPACT: the one-row, 1 KB-threadgroup decode tile .
    const compact = p.decode_nax_c != null and decode_nax and (std.c.getenv("TF_DSV41_EXL3_DECODE_COMPACT") != null or a.rows > 1);
    // TF_DSV41_EXL3_ROUTED_TABLE (default on): the routed decode body reads the codebook value from the resident
    // table instead of deriving it in the ALU. The table holds the format's own decoder's values, entry for entry
    // (the codebook check), so the decoded tile and the product are bit-identical: only the ALU work moves to a load.
    // TF_DSV41_EXL3_ROUTED_TABLE=0 restores the ALU body for a same-binary A/B.
    const routed_table = std.c.getenv("TF_DSV41_EXL3_ROUTED_TABLE") == null;
    // The two-word funnel read is bit-identical at halves = 6 (the routed pack's width) and cuts the stage's stream
    // traffic in half. TF_DSV41_EXL3_FUNNEL=0 restores the four-word body for a same-binary A/B.
    const funnel = std.c.getenv("TF_DSV41_EXL3_FUNNEL") == null and a.halves == 6;
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
    // TF_DSV41_EXL3_FMA=0 restores the tensor-unit body for a same-binary A/B.
    var fma_off = false;
    if (std.c.getenv("TF_DSV41_EXL3_FMA")) |v| {
        if (v[0] == '0') fma_off = true;
    }
    const fma = p.decode_fma != null and decode_nax and a.rows <= moe_split_rows and !fma_off;
    // The static state selection in the register-only body. TF_DSV41_EXL3_FMA_STATIC=0 restores the shipped cursor
    // for a same-binary A/B; the outputs are bit-identical (the hash is equal to the shipped body's at the
    // pack's routed shapes).
    const fma_static = p.decode_fma_static != null and decode_nax and a.rows <= moe_split_rows and !fma_off and fmaStaticOn();
    // The same body with the codebook read from the 1024-entry threadgroup LUT (the mul1 value depends only on the
    // state's byte sum, so 1,024 entries hold every value; the LUT's values are the table's, entry for entry, so the
    // output is bit-identical).
    const fma_static_lut = p.decode_fma_static_lut != null and decode_nax and a.rows <= moe_split_rows and !fma_off and
        fmaStaticOn() and fmaLutOn();
    // A very wide one-row matrix already has thousands of column tiles, so splitting it multiplies threadgroups it
    // does not need and adds a partials round trip; TF_DSV41_EXL3_SPLITS_WIDE selects that.
    const wide = a.n >= 16384;
    const wide_splits: usize = envRows("TF_DSV41_EXL3_SPLITS_WIDE", 8, &core.wide_splits_cache);
    // The one-row split order for every decode-width call, rows <= 8 included, so a verify window's row gets the same
    // bits as a solo step (row_invariance.zig covers it; TF_DSV41_NO_ROW_INVARIANT restores the old window
    // arithmetic). Rows == 1 takes this path either way, so no shipped stream's bits move.
    const row_invariant = std.c.getenv("TF_DSV41_NO_ROW_INVARIANT") == null;
    // The register-only body on the dense one-row path. The dense shapes (the attention's four projections and the
    // shared expert's three) run the 128-column threadgroup-tile body today, at half the rate the routed stage
    // reaches on the same shape class; this body is the routed decode's own structure (the wide form's: one warp a
    // sixteen-column tile, the decoded weights in registers, no threadgroup weight tile, no barrier in the k walk, no
    // split-K and no partials round trip). Its reduction order is the register-only body's, so the flag's arm owes
    // the FP64 reference for the dense one-row class. The drafter's own blocks are packed at halves = 8 (the
    // backbone's are 10/12), and the register-only dense body has a halves == 8 branch. The flag is now a VALUE test:
    // unset or any value but "0" is the body; TF_DSV41_EXL3_FMA_DENSE_H8=0 restores the shipped body for a
    // same-binary A/B. (The old presence test armed the flag on "0" -- the record's watch-out; that footgun is fixed
    // here.)
    const fma_dense_h8 = blk: {
        const raw = std.c.getenv("TF_DSV41_EXL3_FMA_DENSE_H8") orelse break :blk true;
        const v = std.mem.span(raw);
        break :blk !(v.len == 1 and v[0] == '0');
    };
    const fma_dense_ok = p.decode_fma_dense != null and a.indexed == 0 and a.routed == 0 and a.rows <= 8 and
        a.n % 16 == 0 and a.k % 16 == 0 and (a.halves == 6 or a.halves == 10 or a.halves == 12 or
        (a.halves == 8 and fma_dense_h8));
    // The MULTI-ROW dense body -- R rows a threadgroup, so the weight decode is shared by the rows. The shipped dense
    // body is one row a threadgroup (`row = tg.y`, the dispatch's height is the row count), so a four-row verify
    // decodes the attention's dense weights four times over the same bytes. Every row keeps the shipped per-row
    // arithmetic, so the rows' sums do not move. TF_DSV41_EXL3_FMA_DENSE_MULTI=<rows a threadgroup> arms it (2 or 4;
    // 0 restores the one-row body).
    const multi_rows: u32 = if (std.c.getenv("TF_DSV41_EXL3_FMA_DENSE_MULTI")) |v|
        (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0)
    else
        4;
    // (the arm): THE CEIL HEIGHT. The multi body's partial group is already guarded in the kernel (`if (row >=
    // a.rows) break;`), so a row count that is not a multiple of the body's R can still share ONE weight decode per
    // group: the height is ceil(rows / R) instead of rows / R. The even row counts are unchanged (the height is the
    // same), so the shipped paths stay bit-identical. Without it a FIVE-ROW window (the depth-4 drafter) falls back
    // to the one-row body and decodes the dense weights once a row. THE CEIL HEIGHT IS THE DEFAULT. The multi body-s
    // height is ceil(rows/R) instead of rows/R, so a window whose row count is not a multiple of R still shares ONE
    // weight decode per group instead of falling back to the one-row body (which decodes the dense weights once a row
    // -- the ALU-bound part of that body). The kernel-s own guard (`if (row >= a.rows) break;`) skips the padded
    // rows, so every row keeps the shipped per-row arithmetic and the output is BIT-IDENTICAL (the band's agreement
    // is the control's to the token at two, three and four rows, and the sea canon holds in every arm).
    // TF_DSV41_EXL3_FMA_DENSE_MULTI_CEIL=0 restores the old height.
    const multi_ceil = blk: {
        const v = std.c.getenv("TF_DSV41_EXL3_FMA_DENSE_MULTI_CEIL") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.span(v), "0");
    };
    const fma_dense_multi4 = fma_dense_ok and p.decode_fma_dense_multi4 != null and multi_rows == 4 and
        a.rows >= 4 and (multi_ceil or a.rows % 4 == 0);
    const fma_dense_multi2 = fma_dense_ok and p.decode_fma_dense_multi2 != null and multi_rows >= 2 and
        a.rows >= 2 and (multi_ceil or a.rows % 2 == 0) and (multi_rows == 2 or !fma_dense_multi4);
    const fma_dense = fma_dense_ok and fmaDenseOn() and !fma_dense_multi4 and !fma_dense_multi2;
    const splits: usize = if (fma or fma_dense or fma_dense_multi2 or fma_dense_multi4) 1 else if (a.indexed == 0 and a.routed == 0 and (a.rows == 1 or (row_invariant and a.rows <= 8)) and p.reduce != null)
        @as(usize, splitOverride() orelse @as(u32, @intCast(if (wide) wide_splits else one_row_splits)))
    else
        moe_splits;
    a.splits = @intCast(splits);
    // Tile shape: rows a threadgroup covers (a dense tile can span more rows because its weights are shared) and
    // columns it covers (a wider column tile re-reads the activation tile fewer times).
    const nax_tile: usize = if (a.indexed != 0) (if (r.tile_rows != 0) r.tile_rows else 64) else envRows("TF_DSV41_EXL3_NAX_TILE_ROWS", 128, &pipe.nax_tile_cache);
    // A 256-row tile wants the narrower column tile so the threadgroup stays at 512 threads. A routed chunk's column
    // tile is the inherited dense constant (64) unless TF_DSV41_EXL3_MOE_COLS says otherwise: a 128-column tile
    // halves the number of column tiles and therefore the activation fragment re-reads, at twice the decoded weight
    // tile and half the threadgroups. Exactness is unchanged (same 16-k block order, hash-identical).
    const moe_cols = if (a.indexed != 0) envRows("TF_DSV41_EXL3_MOE_COLS", 0, &core.moe_cols_cache) else 0;
    const nax_cols: usize = if (a.indexed != 0 and nax_tile >= 256) 64 else if (moe_cols != 0) moe_cols else envRows("TF_DSV41_EXL3_NAX_COLS", if (a.indexed != 0) 128 else 64, &pipe.nax_cols_cache);
    const tile_id: usize = ((nax_tile / 64) - 1) * 2 + (nax_cols / 64 - 1);
    const nax_pick: ?mtl.Pipeline = p.prefill_nax_tiles[tile_id];
    // The funnel read in the routed prompt tile . The routed pack's width is halves = 6, so the kernel's own guard
    // takes it; at another width the kernel reads the shipped path.
    const nax_funnel_pick: ?mtl.Pipeline = if (funnelPrefillOn() and a.indexed != 0 and tile_id == 0)
        p.prefill_nax_funnel
    else
        null;
    const nax_funnel = nax_funnel_pick != null;
    // The same tile with the grouping's padded rows skipped. The 128-row tile (tile_id 2) is the same body with eight
    // row groups; the grouped plan pads each expert's block to the tile's own width, so the two travel together.
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
    // The routed tile with the stream word read as one 4-byte load: the wide form's read. The kernel checks the
    // tile's alignment itself and falls back to the shipped read.
    const nax_wide_pick: ?mtl.Pipeline = if (wideWordsOn() and a.indexed != 0 and tile_id == 0 and
        nax_skip_pad_pick != null)
        p.prefill_nax_wide
    else
        null;
    const nax_wide = nax_wide_pick != null;
    // The routed tile with the funnel path as its only path . The pack's routed width is halves = 6, the funnel
    // path's width; the kernel is only selected there.
    const nax_funnel_only_pick: ?mtl.Pipeline = if (funnelOnlyOn() and a.indexed != 0 and tile_id == 0 and
        nax_skip_pad_pick != null)
        p.prefill_nax_funnel_only
    else
        null;
    const nax_funnel_only = nax_funnel_only_pick != null;
    // The cooperative read of the block's words (the funnel's read, the words by a shuffle). It rides the funnel (the
    // read is inside the funnel's branch) and the padded rows' skip.
    const nax_coop_pick: ?mtl.Pipeline = if (funnelPrefillOn() and coopPrefillOn() and a.indexed != 0 and tile_id == 0)
        (if (skipPadOn()) p.prefill_nax_coop_skip else p.prefill_nax_coop)
    else
        null;
    const nax_coop = nax_coop_pick != null;
    // TF_DSV41_EXL3_TABLE: the 64x128 grouped tile on the double-buffered, table-decoded body.
    const nax_db_pick: ?mtl.Pipeline = if (p.codebook_table == null) null else if (tile_id == 0) p.prefill_nax_db64 else if (tile_id == 1) p.prefill_nax_db else null;
    const nax_db = nax_db_pick != null and grouped_nax and
        std.c.getenv("TF_DSV41_EXL3_TABLE") != null and a.k % 64 == 0 and a.rows >= envRows("TF_DSV41_EXL3_NAX_MIN_ROWS", 64, &pipe.nax_min_rows_cache);
    if (nax_db and !pipe.nax_db_logged) {
        pipe.nax_db_logged = true;
        std.log.info("exl3: nax10 table body selected rows={d} tile={d} cols={d}", .{ a.rows, nax_tile, nax_cols });
    }
    // (THE LEVER, default off): THE DENSE TILE AT EIGHT ROWS. The drafter's conditioning projects the taps ring's
    // 128-row window through main_proj, and the window WRAPS (the ring is circular): it is TWO calls whose row counts
    // are (128-first) and (first), so half the rounds run the smaller piece on 1..63 rows -- and the tensor-unit
    // dense tile's gate is 64 rows, so that piece ran on the general/shared body instead of the tile. The flag lets
    // the DENSE (non-indexed, non-routed) calls take the tile from EIGHT rows up. The only dense calls in that range
    // in the whole engine are the conditioning's wrapped piece: the trunk's own dense calls are 1 row a token, 2-4 a
    // verify window and 3,619 a prompt, and the drafter's blocks are 2-5. The grouped routed calls are multiples of
    // 64 and the indexed ones are not on this path, so nothing else moves. TF_DSV41_EXL3_NAX_DENSE8=1 arms it; unset
    // or 0 is the shipped path.
    const dense8 = blk: {
        const v = std.c.getenv("TF_DSV41_EXL3_NAX_DENSE8") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.span(v), "0");
    };
    // The gate is the CALLER's hint and the flag together: the hint alone moves nothing.
    const dense8_on = dense8 and r.dense8;
    const nax_min_rows: u32 = envRows("TF_DSV41_EXL3_NAX_MIN_ROWS", 64, &pipe.nax_min_rows_cache);
    const nax = nax_pick != null and a.k % 64 == 0 and
        a.rows >= (if (dense8_on and a.indexed == 0 and a.routed == 0) @min(nax_min_rows, 8) else nax_min_rows) and
        ((a.indexed == 0 and a.routed == 0) or grouped_nax);
    // The dense 128x64 tile's double-buffered twin: one barrier a block, the next block's decode in flight with the
    // multiply. Only the dense shapes (the three attention projections and the shared expert) take it.
    const nax_dense_db = p.prefill_nax_dense_db != null and nax and a.indexed == 0 and tile_id == 2 and denseDb();
    // The dense tile with a simdgroup that owns thirty-two rows . It runs four warps, not eight.
    const nax_dense_rpg2 = p.prefill_nax_dense_rpg2 != null and nax and a.indexed == 0 and tile_id == 2 and denseRpg2();
    // The dense tile's funnel read: the same tile and the same order, two words a lane.
    const nax_dense_funnel = p.prefill_nax_dense_funnel != null and nax and a.indexed == 0 and tile_id == 2 and
        ((a.halves == 8 and denseFunnel8On()) or denseFunnelOn());
    if (std.c.getenv("TF_DSV41_EXL3_FUNNEL_LOG") != null and a.indexed == 0 and tile_id == 2 and a.rows >= 4) {
        std.log.info("exl3 funnel pick: rows={d} n={d} k={d} halves={d} tile={d} p51={} f8={} fd={} dense8={} pick={s}", .{
            a.rows,                                                                    a.n, a.k, a.halves, nax_tile, p.prefill_nax_dense_funnel != null, denseFunnel8On(), denseFunnelOn(), dense8_on,
            if (nax_dense_funnel) "nax51" else if (nax_dense_db) "nax31" else "other",
        });
    }
    // TF_DSV41_EXL3_MOE_DB: the routed 64x64 tile on the double-buffered body, one barrier a k block.
    const nax_moe_db = p.prefill_nax_moe_db != null and nax and a.indexed != 0 and tile_id == 0 and moeDb();
    if (nax_dense_db and !pipe.nax_db_logged) {
        pipe.nax_db_logged = true;
        std.log.info("exl3: nax31 dense double-buffered body selected rows={d} tile={d}", .{ a.rows, nax_tile });
    }
    const shared = !nax and !decode_nax and p.prefill != null and a.rows >= envRows("TF_DSV41_EXL3_SHARED_MIN_ROWS", 32, &pipe.shared_min_rows_cache) and a.indexed == 0 and a.routed == 0;
    // TF_DSV41_EXL3_TABLE_DECODE: the one-row / narrow-window matmul body reads the codebook value from the 3 x
    // 65536-half table instead of deriving it (same values, bit-identical output).
    const matmul_tab = p.matmul_tab != null and a.indexed == 0 and a.routed == 0 and a.rows <= 8 and tableDecodeOn();
    // TF_DSV41_EXL3_WIDE_DECODE: the 128-column one-row table body. Bit-identical to matmul_tab (identical at
    // all six split counts); it only takes the plain token's three attention shapes, not the head.
    const matmul_wide = p.matmul_wide != null and a.indexed == 0 and a.routed == 0 and a.rows == 1 and
        a.n <= 32768 and a.n % 128 == 0 and a.k % 16 == 0 and tableDecodeOn() and wideDecodeOn();
    // TF_DSV41_EXL3_DIRECT_DECODE: a lane owns an output column and decodes its 16 states into registers, so the
    // one-row body has no threadgroup weight tile and no barrier. Same table value and ascending-k sum as
    // matmul_wide, hence bit-identical; same plain-token attention shapes.
    const matmul_direct = p.matmul_direct != null and a.indexed == 0 and a.routed == 0 and a.rows == 1 and
        a.n >= 4096 and a.n % 256 == 0 and a.k % 16 == 0 and a.halves <= 14 and
        tableDecodeOn() and wideDecodeOn() and directDecodeOn();
    // The static state selection. The shipped body picks each state's word pair and bit offset at run time (the
    // `rel[k]` cursor); here one funnel shift a block puts the lane's window at a known place and the eight windows
    // then sit at compile-time shifts, so the cursor leaves the inner loop. The same tile, the same activation load,
    // the same 16-product FP32 sums and the same k order, so the output is bit-identical (identical at every
    // shape and split count).
    const matmul_static = p.matmul_static != null and a.indexed == 0 and a.routed == 0 and a.rows == 1 and
        a.n <= 32768 and a.n % 128 == 0 and a.k % 16 == 0 and (a.halves == 6 or a.halves == 10 or a.halves == 12) and
        tableDecodeOn() and wideDecodeOn() and staticDecodeOn();
    // The same static body with the codebook read from the 1024-entry threadgroup LUT.
    const matmul_static_lut = p.matmul_static_lut != null and a.indexed == 0 and a.routed == 0 and a.rows == 1 and
        a.n <= 32768 and a.n % 128 == 0 and a.k % 16 == 0 and a.codebook == 2 and
        (a.halves == 6 or a.halves == 10 or a.halves == 12) and
        tableDecodeOn() and wideDecodeOn() and staticDecodeOn() and lutDecodeOn();
    if (a.rows == 1 and a.indexed == 0 and a.routed == 0 and pipe.direct_log_count < 10) {
        pipe.direct_log_count += 1;
        std.log.info("exl3: one-row body n={d} k={d} splits={d} direct={} wide={} static={} lut={} tab={} pd={} halves={d} rows={d} idx={d} rt={d} nmod={d} kmod={d} env={}", .{ a.n, a.k, splits, matmul_direct, matmul_wide, matmul_static, matmul_static_lut, matmul_tab, p.matmul_direct != null, a.halves, a.rows, a.indexed, a.routed, a.n % 128, a.k % 16, directDecodeOn() });
    }
    if (matmul_wide and !pipe.matmul_tab_logged) {
        pipe.matmul_tab_logged = true;
        std.log.info("exl3: wide table-decode body selected rows={d} n={d} k={d} splits={d}", .{ a.rows, a.n, a.k, splits });
    }
    if (matmul_tab and !pipe.matmul_tab_logged) {
        pipe.matmul_tab_logged = true;
        std.log.info("exl3: table-decode matmul body selected rows={d} n={d} k={d} splits={d}", .{ a.rows, a.n, a.k, splits });
    }
    const pipeline = (if (fma_static_lut) p.decode_fma_static_lut.? else if (fma_static) p.decode_fma_static.? else if (fma) p.decode_fma.? else if (fma_dense_multi4) p.decode_fma_dense_multi4.? else if (fma_dense_multi2) p.decode_fma_dense_multi2.? else if (fma_dense) p.decode_fma_dense.? else if (matvec_nax) p.matvec_nax.? else if (decode_pipe) |dp| dp else if (matmul_static_lut) p.matmul_static_lut.? else if (matmul_static) p.matmul_static.? else if (matmul_direct) p.matmul_direct.? else if (matmul_wide) p.matmul_wide.? else if (matmul_tab) p.matmul_tab.? else if (nax_db) nax_db_pick.? else if (nax_dense_rpg2) p.prefill_nax_dense_rpg2.? else if (nax_dense_funnel) p.prefill_nax_dense_funnel.? else if (nax_dense_db) p.prefill_nax_dense_db.? else if (nax_moe_db) p.prefill_nax_moe_db.? else if (nax_coop) nax_coop_pick.? else if (nax_wide) nax_wide_pick.? else if (nax_funnel_only) nax_funnel_only_pick.? else if (nax_skip_pad) nax_skip_pad_pick.? else if (nax_funnel) nax_funnel_pick.? else if (nax) nax_pick.? else if (shared) p.prefill.? else p.matmul);
    // Rows a decode threadgroup takes: one a SIMDgroup (TF_DSV41_EXL3_MOE_ROWS selects one a threadgroup).
    const moe_rows: usize = if (decode_nax) (if (compact) 1 else envRows("TF_DSV41_EXL3_MOE_ROWS", 2, &core.moe_rows_cache)) else 4;
    const group_rows: usize = if (fma or fma_static or fma_static_lut or fma_dense_multi2 or fma_dense_multi4) 1 else if (decode_nax) moe_rows else if (nax) nax_tile else if (shared) 32 else 8;
    // The geometry must follow the PIPELINE that was chosen: the static body is the wide body's (128 columns, 256
    // threads), so it is named before the direct body's 256-column geometry.
    const group_cols: usize = if (fma or fma_static or fma_static_lut or fma_dense) 16 else if (matmul_static or matmul_static_lut) 128 else if (matmul_direct) 256 else if (matvec_nax or matmul_wide) 128 else if (nax) nax_cols else if (decode_nax) 32 else 16;
    const height: usize = if (fma_dense_multi4) (a.rows + 3) / 4 else if (fma_dense_multi2) (a.rows + 1) / 2 else if (fma or fma_static or fma_static_lut or fma_dense) a.rows else if (matvec_nax) 1 else if (splits > 1 and !decode_nax) splits else if (a.indexed == 1 and !decode_nax) a.rows else (@as(usize, a.rows) + group_rows - 1) / group_rows;
    // The warp32 tile runs four warps (a warp owns two sixteen-row groups), the other nax tiles one warp a
    // sixteen-row group.
    const nax_warp32: bool = nax and a.indexed != 0 and tile_id == 2 and warp32On() and nax_skip_pad_pick != null;
    const threads: usize = if (fma or fma_static or fma_static_lut or fma_dense or fma_dense_multi2 or fma_dense_multi4) 32 * p.fma_warps else if (nax) (if (nax_dense_rpg2 or nax_warp32) nax_tile / 32 * 32 * (nax_cols / 64) else nax_tile / 16 * 32 * (nax_cols / 64)) else if (decode_nax) 32 * moe_rows else if (matmul_static or matmul_static_lut) 256 else if (matmul_direct) 256 else if (matmul_wide) 256 else if (shared or matvec_nax) 128 else 32;
    // One split dispatch over a window: a row group a threadgroup (the tile spans eight rows), the splits in y.
    const split_groups: usize = if (splits > 1 and !decode_nax) (a.rows + 7) / 8 else 1;
    if ((fma or fma_static or fma_static_lut) and core.fma_log_count < 8) {
        core.fma_log_count += 1;
        std.log.info("exl3: routed decode body rows={d} n={d} k={d} halves={d} cols={d} grid={d}x{d}x{d} threads={d} static={} fma={} experts={d}", .{ a.rows, a.n, a.k, a.halves, group_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, if (fma or fma_static) 1 else if (decode_nax) moe_splits else split_groups, threads, fma_static, fma, a.experts });
    }

    if (a.indexed == 0 and a.rows >= 64 and pipe.dense_log_count < 14) {
        pipe.dense_log_count += 1;
        const dpick: []const u8 = if (nax_dense_rpg2) "nax56" else if (nax_dense_funnel) "nax51" else if (nax_dense_db) "nax31" else if (nax) "nax_tile" else if (shared) "shared" else "matmul";
        std.log.info("exl3: dense dispatch rows={d} n={d} k={d} tile={d} cols={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, nax_tile, nax_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, dpick });
    }
    // (the instrument): the dense calls the eight-row gate moves, named.
    if (dense8_on and a.indexed == 0 and a.routed == 0 and a.rows >= 8 and a.rows < 64 and pipe.dense8_log_count < 14) {
        pipe.dense8_log_count += 1;
        std.log.info("exl3: dense8 rows={d} n={d} k={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, if (nax) "nax_tile" else if (shared) "shared" else "matmul" });
    }
    if (a.indexed != 0 and !pipe.routed_logged) {
        pipe.routed_logged = true;
        const pick: []const u8 = if (nax_db) "nax10" else if (nax_dense_db) "nax31" else if (nax_moe_db) "nax19" else if (nax_coop) (if (nax_skip_pad) "nax53" else "nax52") else if (nax_wide) "nax57" else if (nax_funnel_only) "nax59" else if (nax_skip_pad) (if (nax_warp32) "nax54" else "nax48") else if (nax_funnel) "nax42" else if (nax) "nax_tile" else if (shared) "shared" else if (decode_nax) "decode" else "matmul";
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

pub fn stages(p: *const Pipelines, e: mtl.ComputeEncoder, args: Args, trellis: View, su: View, sv: View, bias: View, descriptors: View, x: View, y: View, scratch: View, r: Rows) void {
    var a = args;
    const plan = planBody(p, a, r);
    a.splits = @intCast(plan.splits);
    const sums_off = (@as(usize, a.rows) * a.k * 2 + 15) & ~@as(usize, 15);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + sums_off };
    const partials = View{ .buffer = scratch.buffer, .offset = scratch.offset + sums_off + ((@as(usize, a.rows) * a.n * 4 + 15) & ~@as(usize, 15)) };
    const routes = if (r.routes) |v| v.ids else scratch;
    const rotate_passes: usize = if (std.c.getenv("TF_DSV41_DOUBLE_ROTATE") != null) 2 else 1;
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
    // TF_DSV41_DOUBLE_GEMM: a second identical GEMM dispatch (dense calls only, rotate passes outside the loop), so
    // the prefill delta is the in-situ dense GEMM time itself, separated from the rotate passes that
    // TF_DSV41_DOUBLE_ATTN_DENSE also doubles. TF_DSV41_DOUBLE_GEMM_N=<n> duplicates only the dense calls whose n is
    // <n>; unset keeps TF_DSV41_DOUBLE_GEMM's whole-set form.
    const gemm_passes: usize = if (a.indexed == 0) blk: {
        if (std.c.getenv("TF_DSV41_DOUBLE_GEMM") != null) break :blk 2;
        if (std.c.getenv("TF_DSV41_DOUBLE_GEMM_N")) |v| {
            const want = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
            if (want != 0 and a.n == want) break :blk 2;
        }
        break :blk 1;
    } else (if (std.c.getenv("TF_DSV41_DOUBLE_GEMM_INDEXED") != null) 2 else 1);
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
    // Instrument: TF_DSV41_DOUBLE_ROTATE_OUT duplicates the scale+bias epilogue. It reads the sums and writes the
    // same values again, so the reply does not move; the delta isolates the pass.
    const rotate_out_passes: usize = if (std.c.getenv("TF_DSV41_DOUBLE_ROTATE_OUT") != null) 2 else 1;
    for (0..rotate_out_passes) |_| e.dispatchGroups(.{ .width = rotate_out_width, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}

/// The routed SwiGLU plus the down projection's rotate_in in one pass. `rotated` must be the scratch buffer's base
/// (the same view the down projection's call takes); rows are written at each row's grouped destination or, with
/// `destination == null`, at the row itself. The caller then sends the down projection with `prerotated`.
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

test "EXL3 scratch is row compact and bounded independently of full matrices" {
    // 214,016 was the pre-split-K bound; the formula now adds the split-K partials and the fused gate+up split room
    // (one_row_splits, moeSplitCount). The property is unchanged: the scratch is bounded by rows, never by the full
    // matrices.
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
