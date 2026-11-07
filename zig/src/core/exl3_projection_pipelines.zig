//! The EXL3 projection's resident pipeline set and its loader.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags.zig");
const core = @import("exl3_projection_core.zig");
const format = core.format;
const codebookTable = core.codebookTable;
const fmaStaticOn = core.fmaStaticOn;
const fmaLutOn = core.fmaLutOn;
const tableDecodeOn = core.tableDecodeOn;
const knobs = &core.knobs;

pub const Pipelines = struct {
    rotate_in: mtl.Pipeline,
    matmul: mtl.Pipeline,
    rotate_out: mtl.Pipeline,
    /// The routed SwiGLU with the down projection's rotate_in fused.
    swiglu_prerotate: mtl.Pipeline,
    dense: mtl.Pipeline,
    prefill: ?mtl.Pipeline = null,
    /// The int8 prompt-row body (f32 sums, for a pack that keeps the runtime rotation).
    prefill_fp8: ?mtl.Pipeline = null,
    /// The pre-rotated pack's cast epilogue: f32 sums to the caller's bf16 output, no transform.
    prefill_cast: ?mtl.Pipeline = null,
    /// The same prefill reduction on the tensor units; built only when a caller supplies an allocator.
    reduce: ?mtl.Pipeline = null,
    /// The codebook as a table: 3 x 65536 halves, state to value. Replaces six ALU ops a weight with one load.
    codebook_table: ?mtl.Buffer = null,
    /// The same prefill entry with RG row groups by CG column groups of tile.
    prefill_nax_tiles: [8]?mtl.Pipeline = @splat(null),
    /// The routed 64x64 prompt tile with the funnel read. Bit-identical to the shipped tile.
    prefill_nax_funnel: ?mtl.Pipeline = null,
    /// The same tile with the grouping's padded rows skipped. Bit-identical on the real rows.
    prefill_nax_skip_pad: ?mtl.Pipeline = null,
    /// The routed tile with the funnel path as the only path: the cursor arrays the Fast path needs are dead there.
    prefill_nax_funnel_only: ?mtl.Pipeline = null,
    /// The 128-row routed tile: eight row groups.
    prefill_nax_skip_pad128: ?mtl.Pipeline = null,
    /// The 128-row routed tile with the thirty-two-row warp: a warp owns two sixteen-row groups.
    prefill_nax_warp32: ?mtl.Pipeline = null,
    /// The dense prompt body double-buffered: block b+1 decodes into the tile block b is not multiplying.
    prefill_nax_dense_db: ?mtl.Pipeline = null,
    /// One row an expert, four rows a threadgroup: the decode regime on the tensor units, weights decoded once a tile.
    decode_nax: ?mtl.Pipeline = null,
    /// The compact decode tile (one row a threadgroup, 1 KB of threadgroup memory): the occupancy fix.
    decode_nax_c: ?mtl.Pipeline = null,
    /// The decode bodies with the codebook value read from the resident table.
    decode_nax_tab: ?mtl.Pipeline = null,
    decode_nax_ctab: ?mtl.Pipeline = null,
    /// The compact table body with the two-word funnel read (TF_DECODE_FUNNEL).
    decode_nax_ctab_f: ?mtl.Pipeline = null,
    /// The register-only decode body (`tf_exl3_matmul_decode_fma`): one warp a sixteen-column tile, one k walk.
    decode_fma: ?mtl.Pipeline = null,
    /// The register-only decode body with the static state selection (`tf_exl3_matmul_decode_fma_static`).
    decode_fma_static: ?mtl.Pipeline = null,
    /// The register-only decode body with the static state selection and a 1024-entry LUT codebook read.
    decode_fma_static_lut: ?mtl.Pipeline = null,
    /// The register-only body on the DENSE one-row path (`tf_exl3_matmul_decode_fma_dense`).
    decode_fma_dense: ?mtl.Pipeline = null,
    /// The MULTI-ROW dense body (`tf_exl3_matmul_decode_fma_dense_multi2` / `_multi4`).
    decode_fma_dense_multi2: ?mtl.Pipeline = null,
    decode_fma_dense_multi4: ?mtl.Pipeline = null,
    fma_warps: usize = 16,
    /// The one-row matmul body with the codebook value read from `codebook_table` (bit-identical values).
    matmul_tab: ?mtl.Pipeline = null,
    /// The static state selection with the codebook read from a 1024-entry threadgroup LUT.
    matmul_static_lut: ?mtl.Pipeline = null,
    /// The gated int8 expert path's numerical core: a 16x16 trellis tile as int8 with its own scale.
    int8_tile: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadWith(device, null);
    }

    pub fn loadWith(device: mtl.Device, gpa: ?std.mem.Allocator) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        // Precise transcendental math so the fused SwiGLU's exp is the ops library's exp bit for bit.
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.core_exl3_projection, options);
        defer lib.deinit();
        const ri = try mtl.Pipeline.init(device, lib, "tf_exl3_rotate_in", false);
        errdefer ri.deinit();
        const mm = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_hoisted", false);
        errdefer mm.deinit();
        const mmt = if (gpa != null) (mtl.Pipeline.init(device, lib, "tf_exl3_matmul_tab", false) catch null) else null;
        errdefer if (mmt) |p| p.deinit();
        var mmsl = if (gpa != null) (mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_static_lut", false) catch null) else null;
        errdefer if (mmsl) |p| p.deinit();
        if (mmsl) |p| {
            if (p.simdWidth() != 32 or p.maxThreads() < 256) {
                p.deinit();
                mmsl = null;
            }
        } else std.log.err("exl3: static+LUT one-row pipeline did not load", .{});
        const ro = try mtl.Pipeline.init(device, lib, "tf_exl3_rotate_out", false);
        errdefer ro.deinit();
        const sr = try mtl.Pipeline.init(device, lib, "tf_exl3_swiglu_prerotate", false);
        errdefer sr.deinit();
        const dn = try mtl.Pipeline.init(device, lib, "tf_exl3_dense", false);
        errdefer dn.deinit();
        for ([_]mtl.Pipeline{ ri, mm, ro }) |p|
            if (p.simdWidth() != 32 or p.maxThreads() < 32) return error.UnsupportedExl3Simd;
        var prefill_cast: ?mtl.Pipeline = null;
        const reduce = mtl.Pipeline.init(device, lib, "tf_exl3_splitk_reduce", false) catch null;
        if (knobs.i8_prefill) {
            if (mtl.Pipeline.init(device, lib, "tf_exl3_epilogue", false)) |pf| {
                if (pf.simdWidth() == 32 and pf.maxThreads() >= 64) prefill_cast = pf else pf.deinit();
            } else |err| std.log.err("exl3: epilogue pipeline failed: {s}", .{@errorName(err)});
        }
        var prefill = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill", false) catch null;
        if (prefill) |pf| {
            if (pf.simdWidth() != 32 or pf.maxThreads() < 128) {
                pf.deinit();
                prefill = null;
            }
        }
        var prefill_nax_tiles: [8]?mtl.Pipeline = @splat(null);
        var prefill_nax_dense_db: ?mtl.Pipeline = null;
        var prefill_nax_funnel: ?mtl.Pipeline = null;
        var prefill_nax_skip_pad: ?mtl.Pipeline = null;
        var prefill_nax_funnel_only: ?mtl.Pipeline = null;
        var prefill_nax_skip_pad128: ?mtl.Pipeline = null;
        var prefill_nax_warp32: ?mtl.Pipeline = null;
        var prefill_fp8: ?mtl.Pipeline = null;
        var decode_nax: ?mtl.Pipeline = null;
        var decode_nax_c: ?mtl.Pipeline = null;
        var decode_nax_tab: ?mtl.Pipeline = null;
        var decode_nax_ctab: ?mtl.Pipeline = null;
        var decode_nax_ctab_f: ?mtl.Pipeline = null;
        var decode_fma: ?mtl.Pipeline = null;
        var decode_fma_static: ?mtl.Pipeline = null;
        var decode_fma_static_lut: ?mtl.Pipeline = null;
        // The dense one-row path's register-only body.
        var decode_fma_dense: ?mtl.Pipeline = null;
        var decode_fma_dense_multi2: ?mtl.Pipeline = null;
        var decode_fma_dense_multi4: ?mtl.Pipeline = null;
        var fma_warps: usize = 16;
        var int8_tile: ?mtl.Pipeline = null;
        // The codebook table: the routed decode body reads its values from here.
        const table = if (gpa != null and device.tensorUnits()) codebookTable(device) else null;
        if (gpa) |g| {
            if (device.tensorUnits()) {
                if (frags.source(device, g, sources.core_exl3_projection_nax)) |src| {
                    defer g.free(src);
                    std.log.info("exl3: nax source inlined, {d} bytes", .{src.len});
                    if (mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx())) |nax_lib| {
                        defer nax_lib.deinit();
                        // Indexed by (rows/64-1)*2+(columns/64-1): 64, 128 or 256 rows by 64 or 128 columns.
                        const tile_names = [_]?[]const u8{ "tf_exl3_matmul_prefill_nax5", "tf_exl3_matmul_prefill_nax3", "tf_exl3_matmul_prefill_nax2", "tf_exl3_matmul_prefill_nax4", null, null, "tf_exl3_matmul_prefill_nax6", null };
                        for (tile_names, 0..) |maybe, t| {
                            const name = maybe orelse continue;
                            if (mtl.Pipeline.init(device, nax_lib, name, false)) |pf| {
                                if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_tiles[t] = pf else pf.deinit();
                            } else |_| {}
                        }
                        // The same tile with the grouping's padded rows skipped.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax48", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_skip_pad = pf else pf.deinit();
                        } else |_| {}
                        // The routed tile with the funnel path as its only path.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax59", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_funnel_only = pf else pf.deinit();
                        } else |_| {}
                        // The 128-row routed tile: funnel read on a four-window run, padding rows skipped.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax49", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 256) prefill_nax_skip_pad128 = pf else pf.deinit();
                        } else |_| {}
                        // The 128-row routed tile with the thirty-two-row warp.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax54", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_warp32 = pf else pf.deinit();
                        } else |_| {}
                        // The funnel read in the routed prompt tile.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax42", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_funnel = pf else pf.deinit();
                        } else |_| {}
                        // The dense 128x64 prompt tile on the double-buffered body.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax31", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_dense_db = pf else pf.deinit();
                        } else |err| std.log.err("exl3: nax31 pipeline init failed: {s}", .{@errorName(err)});
                        // The prompt-row int8 body.
                        if (knobs.i8_prefill) {
                            if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_fp8", false)) |pf| {
                                if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) {
                                    prefill_fp8 = pf;
                                    std.log.info("exl3: prompt-row int8 body active (tf_exl3_matmul_prefill_fp8)", .{});
                                } else pf.deinit();
                            } else |err| std.log.err("exl3: fp8 prefill pipeline failed: {s}", .{@errorName(err)});
                        }
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_decode_nax", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) {
                                decode_nax = pf;
                                std.log.info("exl3: tensor-unit decode pipeline active (tf_exl3_matmul_decode_nax)", .{});
                            } else pf.deinit();
                        } else |err| std.log.err("exl3: decode nax pipeline init failed: {s}", .{@errorName(err)});
                        if (std.fmt.allocPrint(g, "#define TF_DECODE_COMPACT 1\n{s}", .{src})) |src_c| {
                            defer g.free(src_c);
                            if (mtl.Library.fromSource(device, src_c, mtl.CompileOptions.mlx()) catch null) |libc| {
                                defer libc.deinit();
                                if (mtl.Pipeline.init(device, libc, "tf_exl3_matmul_decode_nax", false) catch null) |pf| {
                                    if (pf.simdWidth() == 32 and pf.maxThreads() >= 32) {
                                        decode_nax_c = pf;
                                        std.log.info("exl3: compact decode pipeline active (1 KB threadgroup tile)", .{});
                                    } else pf.deinit();
                                }
                            } else std.log.err("exl3: compact decode library compile failed", .{});
                        } else |_| {}
                        // The routed-table knob: the same decode bodies with the codebook read from the resident table.
                        if (table != null) {
                            if (std.fmt.allocPrint(g, "#define TF_DECODE_TABLE 1\n{s}", .{src})) |src_t| {
                                defer g.free(src_t);
                                if (mtl.Library.fromSource(device, src_t, mtl.CompileOptions.mlx()) catch null) |libt| {
                                    defer libt.deinit();
                                    if (mtl.Pipeline.init(device, libt, "tf_exl3_matmul_decode_nax", false) catch null) |pf| {
                                        if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) decode_nax_tab = pf else pf.deinit();
                                    }
                                } else std.log.err("exl3: table decode library compile failed", .{});
                            } else |_| {}
                            if (std.fmt.allocPrint(g, "#define TF_DECODE_TABLE 1\n#define TF_DECODE_COMPACT 1\n{s}", .{src})) |src_tc| {
                                defer g.free(src_tc);
                                if (mtl.Library.fromSource(device, src_tc, mtl.CompileOptions.mlx()) catch null) |libtc| {
                                    defer libtc.deinit();
                                    if (mtl.Pipeline.init(device, libtc, "tf_exl3_matmul_decode_nax", false) catch null) |pf| {
                                        if (pf.simdWidth() == 32 and pf.maxThreads() >= 32) decode_nax_ctab = pf else pf.deinit();
                                    }
                                } else std.log.err("exl3: compact table decode library compile failed", .{});
                            } else |_| {}
                            // The compact table body with the two-word funnel read (routed pack, halves = 6).
                            if (std.fmt.allocPrint(g, "#define TF_DECODE_TABLE 1\n#define TF_DECODE_COMPACT 1\n#define TF_DECODE_FUNNEL 1\n{s}", .{src})) |src_tcf| {
                                defer g.free(src_tcf);
                                if (mtl.Library.fromSource(device, src_tcf, mtl.CompileOptions.mlx()) catch null) |libtcf| {
                                    defer libtcf.deinit();
                                    if (mtl.Pipeline.init(device, libtcf, "tf_exl3_matmul_decode_nax", false) catch null) |pf| {
                                        if (pf.simdWidth() == 32 and pf.maxThreads() >= 32) decode_nax_ctab_f = pf else pf.deinit();
                                    }
                                } else std.log.err("exl3: compact table funnel decode library compile failed", .{});
                            } else |_| {}
                            // The register-only body.
                            const warps: usize = knobs.fma_warps;
                            if (std.fmt.allocPrint(g, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n{s}", .{ warps, src })) |src_fma| {
                                defer g.free(src_fma);
                                if (mtl.Library.fromSource(device, src_fma, mtl.CompileOptions.mlx()) catch null) |libfma| {
                                    defer libfma.deinit();
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma", false) catch null) |pf| {
                                        if (pf.simdWidth() == 32 and pf.maxThreads() >= 32 * warps) {
                                            decode_fma = pf;
                                            fma_warps = warps;
                                            std.log.info("exl3: register-only decode pipeline active (tf_exl3_matmul_decode_fma, {d} warps)", .{warps});
                                        } else pf.deinit();
                                    }
                                    // The same body with the static state selection.
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_static", false) catch null) |ps| {
                                        if (ps.simdWidth() == 32 and ps.maxThreads() >= 32 * warps) {
                                            decode_fma_static = ps;
                                            std.log.info("exl3: register-only decode pipeline active (tf_exl3_matmul_decode_fma_static, {d} warps)", .{warps});
                                        } else ps.deinit();
                                    }
                                    // The same body with the static state selection and a LUT codebook read.
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) |pl| {
                                        if (pl.simdWidth() == 32 and pl.maxThreads() >= 32 * warps) {
                                            decode_fma_static_lut = pl;
                                            std.log.info("exl3: register-only decode pipeline active (tf_exl3_matmul_decode_fma_static_lut, {d} warps)", .{warps});
                                        } else pl.deinit();
                                    }
                                    // The same body on the dense one-row path: a dense matrix has no expert descriptor.
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_dense", false) catch null) |pd| {
                                        if (pd.simdWidth() == 32 and pd.maxThreads() >= 32 * warps) {
                                            decode_fma_dense = pd;
                                            std.log.info("exl3: dense register-only decode pipeline active (tf_exl3_matmul_decode_fma_dense, {d} warps)", .{warps});
                                        } else pd.deinit();
                                    }
                                    // The same body with two and four rows a threadgroup.
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_dense_multi2", false) catch null) |pd| {
                                        if (pd.simdWidth() == 32 and pd.maxThreads() >= 32 * warps) {
                                            decode_fma_dense_multi2 = pd;
                                            std.log.info("exl3: multi-row dense register-only decode pipeline active (tf_exl3_matmul_decode_fma_dense_multi2, {d} warps)", .{warps});
                                        } else pd.deinit();
                                    }
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_dense_multi4", false) catch null) |pd| {
                                        if (pd.simdWidth() == 32 and pd.maxThreads() >= 32 * warps) {
                                            decode_fma_dense_multi4 = pd;
                                            std.log.info("exl3: multi-row dense register-only decode pipeline active (tf_exl3_matmul_decode_fma_dense_multi4, {d} warps)", .{warps});
                                        } else pd.deinit();
                                    }
                                } else std.log.err("exl3: fma decode library compile failed", .{});
                            } else |_| {}
                        }
                    } else |err| std.log.err("exl3: nax library compile failed: {s}", .{@errorName(err)});
                } else |err| std.log.err("exl3: nax source inline failed: {s}", .{@errorName(err)});
            }
        }
        // The gated int8 expert path's numerical core: the trellis tile as int8 with its own scale.
        if (gpa) |g| if (device.tensorUnits()) {
            if (frags.source(device, g, sources.core_exl3_projection_i8) catch null) |src8| {
                defer g.free(src8);
                if (mtl.Library.fromSource(device, src8, mtl.CompileOptions.mlx()) catch null) |lib8| {
                    defer lib8.deinit();
                    if (mtl.Pipeline.init(device, lib8, "tf_exl3_int8_tile", false) catch null) |pf| {
                        if (pf.simdWidth() == 32 and pf.maxThreads() >= 32) int8_tile = pf else pf.deinit();
                    }
                } else std.log.err("exl3: int8 tile library compile failed", .{});
            } else std.log.err("exl3: int8 tile source inline failed", .{});
        };
        // The three codebooks as a table: the tensor-unit kernels read a value instead of deriving it.

        if (table == null) std.log.err("exl3: codebook table not created; the tensor-unit path will read none", .{});
        return .{
            .rotate_in = ri,
            .matmul = mm,
            .rotate_out = ro,
            .swiglu_prerotate = sr,
            .dense = dn,
            .prefill = prefill,
            .prefill_fp8 = prefill_fp8,
            .prefill_cast = prefill_cast,
            .reduce = reduce,
            .prefill_nax_tiles = prefill_nax_tiles,
            .prefill_nax_funnel = prefill_nax_funnel,
            .prefill_nax_skip_pad = prefill_nax_skip_pad,
            .prefill_nax_funnel_only = prefill_nax_funnel_only,
            .prefill_nax_skip_pad128 = prefill_nax_skip_pad128,
            .prefill_nax_warp32 = prefill_nax_warp32,
            .prefill_nax_dense_db = prefill_nax_dense_db,
            .decode_nax = decode_nax,
            .decode_nax_c = decode_nax_c,
            .decode_nax_tab = decode_nax_tab,
            .decode_nax_ctab = decode_nax_ctab,
            .decode_nax_ctab_f = decode_nax_ctab_f,
            .decode_fma = decode_fma,
            .decode_fma_static = decode_fma_static,
            .decode_fma_static_lut = decode_fma_static_lut,
            .decode_fma_dense = decode_fma_dense,
            .decode_fma_dense_multi2 = decode_fma_dense_multi2,
            .decode_fma_dense_multi4 = decode_fma_dense_multi4,
            .fma_warps = fma_warps,
            .int8_tile = int8_tile,
            .matmul_tab = if (table != null) mmt else null,
            .matmul_static_lut = if (table != null) mmsl else null,
            .codebook_table = table,
        };
    }
    pub fn deinit(p: *Pipelines) void {
        p.rotate_in.deinit();
        p.matmul.deinit();
        p.rotate_out.deinit();
        if (p.prefill_fp8) |v| v.deinit();
        if (p.prefill_cast) |v| v.deinit();
        p.swiglu_prerotate.deinit();
        p.dense.deinit();
        if (p.reduce) |pf| pf.deinit();
        if (p.codebook_table) |b| b.deinit();
        if (p.prefill) |pf| pf.deinit();
        for (p.prefill_nax_tiles) |maybe| if (maybe) |pf| pf.deinit();
        if (p.prefill_nax_skip_pad128) |pf| pf.deinit();
        if (p.prefill_nax_warp32) |pf| pf.deinit();
        if (p.prefill_nax_dense_db) |pf| pf.deinit();
        if (p.prefill_nax_funnel_only) |pf| pf.deinit();
        if (p.decode_nax) |pf| pf.deinit();
        if (p.decode_nax_c) |pf| pf.deinit();
        if (p.decode_nax_tab) |pf| pf.deinit();
        if (p.decode_nax_ctab) |pf| pf.deinit();
        if (p.decode_nax_ctab_f) |pf| pf.deinit();
        if (p.decode_fma) |pf| pf.deinit();
        if (p.decode_fma_static) |pf| pf.deinit();
        if (p.decode_fma_static_lut) |pf| pf.deinit();
        if (p.decode_fma_dense) |pf| pf.deinit();
        if (p.decode_fma_dense_multi2) |pf| pf.deinit();
        if (p.decode_fma_dense_multi4) |pf| pf.deinit();
        if (p.matmul_tab) |pf| pf.deinit();
        if (p.matmul_static_lut) |pf| pf.deinit();
        if (p.int8_tile) |pf| pf.deinit();
    }
};
pub var nax_db_logged: bool = false;
pub var routed_logged: bool = false;
pub var dense_log_count: u32 = 0;
/// The log counter for the calls the eight-row dense gate moves.
pub var dense8_log_count: u32 = 0;
pub var matmul_tab_logged: bool = false;

/// The table-decode knob (default on) restores the ALU codebook for a same-binary A/B.
pub fn indexedWidthBody(p: *const Pipelines) bool {
    if (p.decode_nax == null or p.decode_fma == null) return false;
    if (knobs.scalar_experts) return false;
    if (knobs.fma_off) return false;
    return true;
}
