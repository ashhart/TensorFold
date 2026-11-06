// ! Borrowed resident EXL3 projections; fixed tiles keep each row independent of its surrounding window. The !
// model-facing order follows the release's inference/model.py, revision 314113f8; the trellis arithmetic and the !
// kernel shapes are the lane's own pack format.
// Status: the Metal bodies named in the lane notes are re-expressed from the
// release's inference/model.py (revision 314113f8). The host glue for the paired
// decode and the drafter's conditioning still follows the earlier ported shape
// (mlx-serve's dsv41_exl3_guone_k3 / dsv41_exl3_mul1h_k3, MIT/Apache-2.0), and is
// not claimed as rewritten; its rewrite is scheduled.
const std = @import("std");
pub const checkpoint = @import("exl3_load.zig");
pub const load = @import("exl3_metal_load.zig");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags.zig");
pub const format = @import("exl3_format.zig");
const decoder = @import("exl3_projection_decode.zig");

pub const View = struct {
    buffer: mtl.Buffer,
    offset: usize = 0,
};
pub const ScaleFormat = enum { f16, packed_signs };
pub const Output = enum { bf16, f32 };

/// A prompt-row int8 sidecar: the pre-quantized dense weight in the fp8 body's 16-k x 64-n block order (k*n bytes)
/// and one f32 scale an output row. `direct` packs bake the rotation in; the others keep it.
pub const Int8 = struct {
    weight: View,
    scales: View,
    k: usize,
    n: usize,
    pub fn deinit(i: *const Int8) void {
        i.weight.buffer.deinit();
        i.scales.buffer.deinit();
    }
};

/// W[K,N] is stored as trellis[K/16,N/16,8*width.halves] int16, not a conventional [N,K] matrix.
pub const Matrix = struct {
    trellis: View,
    suh: View,
    svh: View,
    bias: ?View = null,
    k: usize,
    n: usize,
    width: format.Width,
    codebook: format.Codebook = .mul1,
    suh_format: ScaleFormat = .f16,
    svh_format: ScaleFormat = .f16,
    /// Byte strides between tiles; zero derives the checkpoint's stored K-major layout.
    tile_k_stride: usize = 0,
    tile_n_stride: usize = 0,
    /// The prompt-row int8 twin, when a sidecar is loaded (TF_DSV41_I8_PREFILL). Decode stays the trellis; the prompt
    /// paths take this only at rows >= 1024.
    int8: ?Int8 = null,
    /// Whether that twin was packed pre-rotated (both Hadamards baked in). The matrix carries the fact, so a caller
    /// that only has the Matrix can pick the one-dispatch form without a model-level flag.
    int8_direct: bool = false,

    pub fn validate(m: Matrix) !void {
        try shape(m.k, m.n);
        if (m.width.halves < 2 or m.width.halves > 16 or
            (m.width.isHalf() and (m.codebook != .mul1 or m.width.halves > 7))) return error.UnsupportedExl3Width;
        const row_bytes = try std.math.mul(usize, m.n / 16, m.nStride());
        const k_stride = if (m.tile_k_stride == 0) row_bytes else m.tile_k_stride;
        if (m.nStride() < m.tileBytes() or m.nStride() % 2 != 0 or
            k_stride < row_bytes or k_stride % 2 != 0)
            return error.InvalidExl3Stride;
    }

    pub fn tileBytes(m: Matrix) usize {
        return m.width.tileWords() * 2;
    }
    pub fn nStride(m: Matrix) usize {
        return if (m.tile_n_stride == 0) m.tileBytes() else m.tile_n_stride;
    }
    pub fn kStride(m: Matrix) !usize {
        return if (m.tile_k_stride == 0) std.math.mul(usize, m.n / 16, m.nStride()) else m.tile_k_stride;
    }
    pub fn storageBytes(m: Matrix) !usize {
        try m.validate();
        const last_k = try std.math.mul(usize, m.k / 16 - 1, try m.kStride());
        const last_n = try std.math.mul(usize, m.n / 16 - 1, m.nStride());
        return std.math.add(usize, try std.math.add(usize, last_k, last_n), m.tileBytes());
    }
    fn checkStorage(m: Matrix) !usize {
        const bytes = try m.storageBytes();
        try check(m.trellis, bytes, 2);
        try check(m.suh, m.k * 2 / (if (m.suh_format == .packed_signs) @as(usize, 16) else 1), 2);
        try check(m.svh, m.n * 2 / (if (m.svh_format == .packed_signs) @as(usize, 16) else 1), 2);
        if (m.bias) |b| try check(b, m.n * 2, 2);
        return bytes;
    }
};

/// Individual expert filtering reads u32 IDs on GPU; repeat top-k slots share one input row.
pub const Routes = struct { ids: View, expert_id: u32 = 0, repeat: usize = 1 };

/// Strides are elements, X is BF16, IDs are u32[count]. Gather/scatter IDs must be in bounds and outputs unique.
pub const Rows = struct {
    count: usize,
    x_stride: usize,
    y_stride: usize,
    output: Output = .bf16,
    input_ids: ?View = null,
    output_ids: ?View = null,
    /// Required with a corresponding ID map; callers guarantee IDs below these capacities and unique output IDs.
    input_rows: usize = 0,
    output_rows: usize = 0,
    routes: ?Routes = null,
    /// Caller-prepared eight-row tiles, same expert per tile; maxInt(u32) padding must follow valid rows.
    grouped: bool = false,
    /// Row tile the routed groups were padded for (0: the 64-row default); the prompt tile follows it so that one
    /// tile is one expert.
    tile_rows: u32 = 0,
    /// The caller wrote this dispatch's rotated input (row-major, k halfs a row) into the scratch base -- the fused
    /// SwiGLU+rotate pass does -- so rotate_in is skipped and the GEMM reads those rows. `x` is that scratch view.
    prerotated: bool = false,
    /// Leave the projection's f32 sums in the scratch for a caller that runs the epilogue itself (the one-row
    /// transform fold reads them, normalises and preps in one kernel). The dispatch's output is not written.
    no_epilogue: bool = false,
    /// This dispatch's routes carry duplicate entries that name the same descriptor (the fused gate/up call's
    /// shared-expert tail), so the paired body can share one weight decode between them.
    pair: bool = false,
    /// The caller asks for the tensor-unit dense tile from EIGHT rows up (the shipped gate is 64). The hint is the
    /// CALLER's, so nothing else in the engine moves.
    dense8: bool = false,
};

/// Metal 3 / tier-2 argument-buffer ABI; borrowed pointers remain resident until the encoded work completes.
pub const ExpertDescriptor = extern struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    bias: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
    halves: u32,
    codebook: u32,
    suh_signs: u32,
    svh_signs: u32,
    has_bias: u32,
    reserved: u32 = 0,
};

/// Populate a caller-owned shared descriptor table once; no weight copying, packing or dequantization.
pub fn descriptor(m: Matrix) !ExpertDescriptor {
    _ = try m.checkStorage();
    return .{
        .trellis = try address(m.trellis),
        .suh = try address(m.suh),
        .svh = try address(m.svh),
        .bias = if (m.bias) |v| try address(v) else 0,
        .tile_k_stride = try m.kStride(),
        .tile_n_stride = m.nStride(),
        .halves = m.width.halves,
        .codebook = @intCast(@backingInt(m.codebook)),
        .suh_signs = @intFromBool(m.suh_format == .packed_signs),
        .svh_signs = @intFromBool(m.svh_format == .packed_signs),
        .has_bias = @intFromBool(m.bias != null),
        // The entry's own output width. An indexed batch may carry matrices of different n (the attention's wq_a 1280
        // and wkv 512 read the same input row), so the decode body bounds each row's column tiles by this field; 0
        // means the call's own n. Every single-width table keeps n == the call's n, so nothing moves for them.
        .reserved = @intCast(m.n),
    };
}

fn address(v: View) !u64 {
    return std.math.add(u64, v.buffer.gpuAddress(), v.offset);
}

/// Call once per encoder for indirectly addressed matrices, unless the caller declares them via residency sets.
pub fn useMatrix(e: mtl.ComputeEncoder, m: Matrix) void {
    e.useResource(m.trellis.buffer, mtl.ResourceUsage.read);
    e.useResource(m.suh.buffer, mtl.ResourceUsage.read);
    e.useResource(m.svh.buffer, mtl.ResourceUsage.read);
    if (m.bias) |b| e.useResource(b.buffer, mtl.ResourceUsage.read);
}

/// Common K,N; descriptor weights must remain resident and disjoint from writable views through completion.
pub const Experts = struct { descriptors: View, count: usize, k: usize, n: usize };

/// The three codebooks as one table, filled from the format's own reference decoder so the values are the ones the
/// tests already pin: state to half, three codebooks of 65536.
fn codebookTable(device: mtl.Device) ?mtl.Buffer {
    const entries = 3 * 65536;
    const buffer = device.buffer(entries * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked) catch return null;
    const out: [*]u16 = @ptrCast(@alignCast(buffer.contents()));
    for (0..3) |cb| {
        const codebook: format.Codebook = @fromBackingInt(@intCast(cb));
        for (0..65536) |state| {
            out[cb * 65536 + state] = @bitCast(format.codebookValue(codebook, @intCast(state)));
        }
    }
    std.log.info("exl3: codebook table {d} entries, first {d} {d} {d}", .{ entries, out[0], out[1], out[65536] });
    return buffer;
}

pub const Pipelines = struct {
    rotate_in: mtl.Pipeline,
    matmul: mtl.Pipeline,
    rotate_out: mtl.Pipeline,
    /// The routed SwiGLU with the down projection's rotate_in fused (TF_DSV41_SWIGLU_ROT): one pass writes the
    /// rotated activation into the scratch base at each row's grouped destination slot.
    swiglu_prerotate: mtl.Pipeline,
    dense: mtl.Pipeline,
    prefill: ?mtl.Pipeline = null,
    /// The int8 prompt-row body (f32 sums, for a pack that keeps the runtime rotation) and its bf16 destination twin
    /// (for a pre-rotated pack, no epilogue transform). Loaded only under TF_DSV41_I8_PREFILL.
    prefill_fp8: ?mtl.Pipeline = null,
    prefill_fp8b: ?mtl.Pipeline = null,
    /// The pre-rotated pack's cast epilogue: f32 sums to the caller's bf16 output, no transform.
    prefill_cast: ?mtl.Pipeline = null,
    /// The same prefill reduction on the tensor units; built only when a caller supplies an allocator (its source
    /// needs nax.h inlined) and the GPU has tensor units.
    reduce: ?mtl.Pipeline = null,
    /// The codebook as a table: 3 x 65536 halves, state to value. Replaces six ALU ops a weight with one load.
    codebook_table: ?mtl.Buffer = null,
    prefill_nax: ?mtl.Pipeline = null,
    /// The same prefill entry with RG row groups by CG column groups of tile: a wider column tile re-reads the
    /// activation tile for fewer output tiles.
    prefill_nax_tiles: [8]?mtl.Pipeline = @splat(null),
    /// The routed 64x64 prompt tile with the funnel read . Bit-identical to .
    prefill_nax_funnel: ?mtl.Pipeline = null,
    /// The same tile with the grouping's padded rows skipped . Bit-identical on the real rows.
    prefill_nax_skip_pad: ?mtl.Pipeline = null,
    /// The routed tile with the stream word read as one 4-byte load . Bit-identical.
    prefill_nax_wide: ?mtl.Pipeline = null,
    /// The routed tile with the funnel path as the only path: the cursor arrays the Fast path needs are dead there.
    /// Bit-identical.
    prefill_nax_funnel_only: ?mtl.Pipeline = null,
    /// The routed 64x64 prompt tile with the funnel read and the block's words by a cooperative load and a shuffle,
    /// and its twin with the grouping's padded rows skipped . Bit-identical.
    prefill_nax_coop: ?mtl.Pipeline = null,
    prefill_nax_coop_skip: ?mtl.Pipeline = null,
    /// The dense 128x64 prompt tile with the funnel read . The dense lane owns four states, so the funnel reads two
    /// words where the cursor path reads four. Bit-identical to .
    prefill_nax_dense_funnel: ?mtl.Pipeline = null,
    /// The 128-row routed tile: eight row groups, the funnel read on a four-window run and the grouping's padded rows
    /// skipped. The weight decode is per tile, so the taller tile decodes each expert's weights once for twice the
    /// rows.
    prefill_nax_skip_pad128: ?mtl.Pipeline = null,
    /// The 128-row routed tile with the thirty-two-row warp: a warp owns two sixteen-row groups, so the block is
    /// decoded by one warp (eight states a lane, eight bits a weight) and the tile covers twice the rows.
    /// Bit-identical to the shipped tile.
    prefill_nax_warp32: ?mtl.Pipeline = null,
    /// The routed prompt tile double-buffered with the codebook table, for the 64x128 grouped tile.
    prefill_nax_db: ?mtl.Pipeline = null,
    /// The same on the engine's 64x64 grouped tile, which is what the shared NAX_COLS cache selects.
    prefill_nax_db64: ?mtl.Pipeline = null,
    /// The dense prompt body double-buffered: block b+1 decodes into the tile block b is not multiplying, one barrier
    /// a block instead of two. Same k order and MMA order as, so its output is bit-identical.
    prefill_nax_dense_db: ?mtl.Pipeline = null,
    prefill_nax_dense_rpg2: ?mtl.Pipeline = null,
    /// The routed 64x64 tile on the double-buffered ALU body: one barrier a k block. Hash-identical.
    prefill_nax_moe_db: ?mtl.Pipeline = null,
    /// One row an expert, four rows a threadgroup: the decode regime on the tensor units, weights decoded once a
    /// tile.
    decode_nax: ?mtl.Pipeline = null,
    /// The compact decode tile (one row a threadgroup, 1 KB of threadgroup memory): the occupancy fix.
    decode_nax_c: ?mtl.Pipeline = null,
    /// The decode bodies with the codebook value read from the resident table (TF_DSV41_EXL3_ROUTED_TABLE, default
    /// on). The table holds the format's own decoder's values, entry for entry, so the decoded tile and the product
    /// are bit-identical; only six ALU ops a weight become one load.
    decode_nax_tab: ?mtl.Pipeline = null,
    decode_nax_ctab: ?mtl.Pipeline = null,
    /// The compact table body with the two-word funnel read (TF_DECODE_FUNNEL).
    decode_nax_ctab_f: ?mtl.Pipeline = null,
    /// The register-only decode body (`tf_exl3_matmul_decode_fma`): one warp a sixteen-column tile, one k walk, no
    /// split-K, no partials and no reduce dispatch, and the decoded weights stay in registers (no threadgroup weight
    /// tile, no barrier in the k loop). The reduction order is the FP32 walk's, not the tensor unit's, so its sums
    /// are FP32 but not bit-identical to the tensor-unit body's.
    decode_fma: ?mtl.Pipeline = null,
    /// The register-only decode body with the static state selection (`tf_exl3_matmul_decode_fma_static`): the lane's
    /// eight windows at compile-time shifts, so the per-state word selection (the `rel[k]` cursor) and the per-state
    /// bit offset leave the inner loop. The same states, the same codebook values, the same products and the same
    /// reduction tree, so the output is bit-identical (identical to the shipped body at the pack's routed
    /// shapes).
    decode_fma_static: ?mtl.Pipeline = null,
    /// The register-only decode body with the static state selection and the codebook read from a 1024-entry
    /// threadgroup LUT (`tf_exl3_matmul_decode_fma_static_lut`).
    decode_fma_static_lut: ?mtl.Pipeline = null,
    /// The one-row transform fold (`tf_exl3_norm_transform`): the output transform, the RMS norm and the next
    /// projection's input transform in one dispatch. Optional so an older metallib still loads.
    norm_transform: ?mtl.Pipeline = null,
    /// The register-only body on the DENSE one-row path (`tf_exl3_matmul_decode_fma_dense`): the routed decode's own
    /// structure (one warp a sixteen-column tile, the decoded weights in registers, no threadgroup weight tile, no
    /// barrier in the k walk, no split-K and no partials round trip), reading a dense matrix's trellis from the
    /// buffer and its strides, width and codebook from the args.
    decode_fma_dense: ?mtl.Pipeline = null,
    /// The MULTI-ROW dense body (`tf_exl3_matmul_decode_fma_dense_multi2` / `_multi4`): the register-only dense body
    /// with R rows a threadgroup, so the weight decode -- the ALU-bound part -- is shared by the rows instead of paid
    /// once a row (the shipped body is one row a threadgroup, and the dispatch's height is the row count, so a
    /// four-row window decodes the attention's dense weights four times over). Every row keeps the shipped per-row
    /// arithmetic (the same k blocks, the same warp split, the same accumulator tree and the same cross-warp sum), so
    /// each row's sum is bit-identical to the one-row body's. The shape the paired decode uses.
    decode_fma_dense_multi2: ?mtl.Pipeline = null,
    decode_fma_dense_multi4: ?mtl.Pipeline = null,
    /// The PAIRED routed decode (`tf_exl3_matmul_decode_fma_static_pair` / `_pair_lut`): the entries that name the
    /// same expert -- the same weight matrix -- are computed in one threadgroup, so the weight decode and the weight
    /// traffic are paid once for the pair and the odd rank exits. The shape one decode for the rows that share a
    /// weight. Each row keeps the shipped per-row arithmetic, so the pair's rows are bit-identical to the single-row
    /// body's.
    decode_fma_static_pair: ?mtl.Pipeline = null,
    decode_fma_static_pair_lut: ?mtl.Pipeline = null,
    fma_warps: usize = 16,
    /// One row, four column tiles a threadgroup: dense projections at decode width on the tensor units.
    matvec_nax: ?mtl.Pipeline = null,
    /// The one-row matmul body with the codebook value read from `codebook_table` (bit-identical values).
    matmul_tab: ?mtl.Pipeline = null,
    /// The 128-column one-row table body (`tf_exl3_matmul_wide_tab`): eight warps on eight 16-column tiles of one
    /// activation block, double buffered so a k block costs one barrier. Same arithmetic as `matmul_tab`.
    matmul_wide: ?mtl.Pipeline = null,
    /// The direct-register one-row table body (`tf_exl3_matmul_direct_tab`): a lane owns an output column and decodes
    /// its 16 states into registers, so there is no threadgroup weight tile and no barrier. Same k order and table
    /// value as `matmul_wide`, hence bit-identical.
    matmul_direct: ?mtl.Pipeline = null,
    /// The static state selection in the one-row body (`tf_exl3_matmul_wide_static_tab`): the lane's eight windows
    /// are extracted at compile-time shifts from one funnel shift a block, so the per-state word selection (the
    /// `rel[k]` cursor) and the per-state bit offset leave the inner loop. The same tile, the same activation load,
    /// the same 16-product FP32 sums and the same k order as `matmul_wide`, so the output is bit-identical (identical
    /// at every shape and split count).
    matmul_static: ?mtl.Pipeline = null,
    /// The static state selection with the codebook read from a 1024-entry threadgroup LUT
    /// (`tf_exl3_matmul_wide_static_lut`): the mul1 value depends only on the state's byte sum, so the 65,536-entry
    /// table holds 1,024 distinct values and a threadgroup can hold them all. The LUT's values are the table's, entry
    /// for entry (identical over the whole output), so the output is bit-identical; the table's L2 traffic is
    /// what the arm removes.
    matmul_static_lut: ?mtl.Pipeline = null,
    /// The gated int8 expert path's numerical core: a 16x16 trellis tile as int8 with its own scale.
    int8_tile: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadWith(device, null);
    }

    pub fn loadWith(device: mtl.Device, gpa: ?std.mem.Allocator) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        // Precise transcendental math so the fused SwiGLU's exp is the ops library's exp bit for bit; the rest of
        // this library uses no math functions, so its code is unchanged.
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
        var mmw = if (gpa != null) (mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_tab", false) catch null) else null;
        errdefer if (mmw) |p| p.deinit();
        if (mmw) |p| {
            // Eight SIMDgroups a threadgroup: the 128-column body needs a 256-thread group.
            if (p.simdWidth() != 32 or p.maxThreads() < 256) {
                p.deinit();
                mmw = null;
            }
        }
        var mmd = if (gpa != null) (mtl.Pipeline.init(device, lib, "tf_exl3_matmul_direct_tab", false) catch null) else null;
        errdefer if (mmd) |p| p.deinit();
        if (mmd) |p| {
            // Four SIMDgroups a threadgroup: 128 columns of the one-row direct body.
            if (p.simdWidth() != 32 or p.maxThreads() < 128) {
                std.log.err("exl3: direct pipeline rejected simd={d} maxThreads={d}", .{ p.simdWidth(), p.maxThreads() });
                p.deinit();
                mmd = null;
            }
        } else std.log.err("exl3: direct pipeline did not load", .{});
        // The static state selection in the one-row body (the same 128-column geometry as the wide body: eight
        // SIMDgroups a threadgroup).
        var mms = if (gpa != null) (mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_static_tab", false) catch null) else null;
        var mmsl = if (gpa != null) (mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_static_lut", false) catch null) else null;
        errdefer if (mms) |p| p.deinit();
        if (mms) |p| {
            if (p.simdWidth() != 32 or p.maxThreads() < 256) {
                p.deinit();
                mms = null;
            }
        } else std.log.err("exl3: static one-row pipeline did not load", .{});
        if (mmsl) |p| {
            if (p.simdWidth() != 32 or p.maxThreads() < 256) {
                p.deinit();
                mmsl = null;
            }
        } else std.log.err("exl3: static+LUT one-row pipeline did not load", .{});
        const ro = try mtl.Pipeline.init(device, lib, "tf_exl3_rotate_out", false);
        errdefer ro.deinit();
        const nt = mtl.Pipeline.init(device, lib, "tf_exl3_norm_transform", false) catch null;
        if (nt) |v| {
            if (v.simdWidth() != 32 or v.maxThreads() < 256) {
                v.deinit();
            } else {
                std.log.info("exl3: one-row transform fold active (tf_exl3_norm_transform)", .{});
            }
        }
        const sr = try mtl.Pipeline.init(device, lib, "tf_exl3_swiglu_prerotate", false);
        errdefer sr.deinit();
        const dn = try mtl.Pipeline.init(device, lib, "tf_exl3_dense", false);
        errdefer dn.deinit();
        for ([_]mtl.Pipeline{ ri, mm, ro }) |p|
            if (p.simdWidth() != 32 or p.maxThreads() < 32) return error.UnsupportedExl3Simd;
        var prefill_cast: ?mtl.Pipeline = null;
        const reduce = mtl.Pipeline.init(device, lib, "tf_exl3_splitk_reduce", false) catch null;
        if (std.c.getenv("TF_DSV41_I8_PREFILL") != null) {
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
        var prefill_nax: ?mtl.Pipeline = null;
        var prefill_nax_tiles: [8]?mtl.Pipeline = @splat(null);
        var prefill_nax_db: ?mtl.Pipeline = null;
        var prefill_nax_db64: ?mtl.Pipeline = null;
        var prefill_nax_dense_db: ?mtl.Pipeline = null;
        var prefill_nax_dense_rpg2: ?mtl.Pipeline = null;
        var prefill_nax_moe_db: ?mtl.Pipeline = null;
        var prefill_nax_funnel: ?mtl.Pipeline = null;
        var prefill_nax_skip_pad: ?mtl.Pipeline = null;
        var prefill_nax_wide: ?mtl.Pipeline = null;
        var prefill_nax_funnel_only: ?mtl.Pipeline = null;
        var prefill_nax_skip_pad128: ?mtl.Pipeline = null;
        var prefill_nax_warp32: ?mtl.Pipeline = null;
        var prefill_nax_coop: ?mtl.Pipeline = null;
        var prefill_nax_coop_skip: ?mtl.Pipeline = null;
        var prefill_nax_dense_funnel: ?mtl.Pipeline = null;
        var prefill_fp8: ?mtl.Pipeline = null;
        var prefill_fp8b: ?mtl.Pipeline = null;
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
        var decode_fma_static_pair: ?mtl.Pipeline = null;
        var decode_fma_static_pair_lut: ?mtl.Pipeline = null;
        var fma_warps: usize = 16;
        var matvec_nax: ?mtl.Pipeline = null;
        var int8_tile: ?mtl.Pipeline = null;
        // The codebook table: the routed decode body reads its values from here (TF_DSV41_EXL3_ROUTED_TABLE), and the
        // dense one-row bodies bind it too. Built before the pipelines that need it.
        const table = if (gpa != null and device.tensorUnits()) codebookTable(device) else null;
        if (gpa) |g| {
            if (device.tensorUnits()) {
                if (frags.source(device, g, sources.core_exl3_projection_nax)) |src| {
                    defer g.free(src);
                    std.log.info("exl3: nax source inlined, {d} bytes", .{src.len});
                    if (mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx())) |nax_lib| {
                        defer nax_lib.deinit();
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) {
                                prefill_nax = pf;
                                std.log.info("exl3: tensor-unit prefill pipeline active (tf_exl3_matmul_prefill_nax)", .{});
                            } else {
                                std.log.err("exl3: nax pipeline rejected: simdWidth={d} maxThreads={d}", .{ pf.simdWidth(), pf.maxThreads() });
                                pf.deinit();
                            }
                        } else |err| std.log.err("exl3: nax pipeline init failed: {s}", .{@errorName(err)});
                        // Indexed by (rows / 64 - 1) * 2 + (columns / 64 - 1): 64, 128 and 256 rows by 64 or 128
                        // columns.
                        const tile_names = [_]?[]const u8{ "tf_exl3_matmul_prefill_nax5", "tf_exl3_matmul_prefill_nax3", "tf_exl3_matmul_prefill_nax2", "tf_exl3_matmul_prefill_nax4", null, null, "tf_exl3_matmul_prefill_nax6", null };
                        for (tile_names, 0..) |maybe, t| {
                            const name = maybe orelse continue;
                            if (mtl.Pipeline.init(device, nax_lib, name, false)) |pf| {
                                if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_tiles[t] = pf else pf.deinit();
                            } else |_| {}
                        }
                        // The same tile with the grouping's padded rows skipped .
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax48", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_skip_pad = pf else pf.deinit();
                        } else |_| {}
                        // The routed tile with the stream word read as one 4-byte load .
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax57", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_wide = pf else pf.deinit();
                        } else |_| {}
                        // The routed tile with the funnel path as the only path .
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax59", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_funnel_only = pf else pf.deinit();
                        } else |_| {}
                        // The 128-row routed tile: the funnel read on a four-window run and the grouping's padded
                        // rows skipped.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax49", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 256) prefill_nax_skip_pad128 = pf else pf.deinit();
                        } else |_| {}
                        // The 128-row routed tile with the thirty-two-row warp.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax54", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_warp32 = pf else pf.deinit();
                        } else |_| {}
                        // The dense prompt tile with the funnel read .
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax51", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 256) prefill_nax_dense_funnel = pf else pf.deinit();
                        } else |_| {}
                        // The funnel read in the routed prompt tile .
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax42", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_funnel = pf else pf.deinit();
                        } else |_| {}
                        // The cooperative read -- the block's words by one lane a word and a shuffle.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax52", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_coop = pf else pf.deinit();
                        } else |_| {}
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax53", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_coop_skip = pf else pf.deinit();
                        } else |_| {}
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax10", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_db = pf else pf.deinit();
                        } else |err| std.log.err("exl3: nax10 pipeline init failed: {s}", .{@errorName(err)});
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax18", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_db64 = pf else pf.deinit();
                        } else |err| std.log.err("exl3: nax18 pipeline init failed: {s}", .{@errorName(err)});
                        // The dense 128x64 prompt tile on the double-buffered body: the same k order and MMA order as
                        // the single-tile body, so the output is bit-identical.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax31", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_dense_db = pf else pf.deinit();
                        } else |err| std.log.err("exl3: nax31 pipeline init failed: {s}", .{@errorName(err)});
                        // The prompt-row int8 body. Only under the flag, so a default boot compiles and loads exactly
                        // what it did before.
                        if (std.c.getenv("TF_DSV41_I8_PREFILL") != null) {
                            if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_fp8", false)) |pf| {
                                if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) {
                                    prefill_fp8 = pf;
                                    std.log.info("exl3: prompt-row int8 body active (tf_exl3_matmul_prefill_fp8)", .{});
                                } else pf.deinit();
                            } else |err| std.log.err("exl3: fp8 prefill pipeline failed: {s}", .{@errorName(err)});
                            if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_fp8b", false)) |pf| {
                                if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) {
                                    prefill_fp8b = pf;
                                    std.log.info("exl3: prompt-row int8 direct body active (tf_exl3_matmul_prefill_fp8b)", .{});
                                } else pf.deinit();
                            } else |err| std.log.err("exl3: fp8b prefill pipeline failed: {s}", .{@errorName(err)});
                        }
                        // The dense 128x64 tile with a simdgroup that owns thirty-two rows: the NAXTile shape. The
                        // lane keeps eight states, so the stream words it reads cover twice the states, and the
                        // weight fragments are read once for two row groups. The values, the k order and the MMA
                        // order per output are the single-tile body's, so the output is bit-identical.
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax56", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_dense_rpg2 = pf else pf.deinit();
                        } else |err| std.log.err("exl3: nax56 pipeline init failed: {s}", .{@errorName(err)});
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matmul_prefill_nax19", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) prefill_nax_moe_db = pf else pf.deinit();
                        } else |err| std.log.err("exl3: nax19 pipeline init failed: {s}", .{@errorName(err)});
                        // TF_DSV41_INT8_BODY: the same skeleton, its tile and x block quantised per 16x16 fragment
                        // and multiplied as int8 x int8 on the tensor units. Loaded from its own source so both
                        // bodies stay in the tree; the wider bf16 tiles are dropped so every chunk uses it.
                        if (std.c.getenv("TF_DSV41_INT8_BODY") != null) {
                            if (frags.source(device, g, sources.core_exl3_projection_nax_i8)) |src8| {
                                defer g.free(src8);
                                if (mtl.Library.fromSource(device, src8, mtl.CompileOptions.mlx())) |lib8| {
                                    defer lib8.deinit();
                                    if (mtl.Pipeline.init(device, lib8, "tf_exl3_matmul_prefill_nax_i8", false)) |pf8| {
                                        if (pf8.simdWidth() == 32 and pf8.maxThreads() >= 128) {
                                            if (prefill_nax) |old| old.deinit();
                                            prefill_nax = pf8;
                                            for (&prefill_nax_tiles) |*t| if (t.*) |q| {
                                                q.deinit();
                                                t.* = null;
                                            };
                                            std.log.info("exl3: int8 routed body active (tf_exl3_matmul_prefill_nax_i8)", .{});
                                        } else {
                                            std.log.err("exl3: int8 body rejected: simdWidth={d} maxThreads={d}", .{ pf8.simdWidth(), pf8.maxThreads() });
                                            pf8.deinit();
                                        }
                                    } else |err| std.log.err("exl3: int8 body pipeline failed: {s}", .{@errorName(err)});
                                } else |err| std.log.err("exl3: int8 body library failed: {s}", .{@errorName(err)});
                            } else |err| std.log.err("exl3: int8 body source failed: {s}", .{@errorName(err)});
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
                        // TF_DSV41_EXL3_ROUTED_TABLE: the same decode bodies with the codebook read from the resident
                        // table. The engine binds the table at buffer 7 for every stage, and the table holds the
                        // format's own decoder's values, so the output is bit-identical.
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
                            // The same compact table body with the two-word funnel read. halves = 6 is the routed
                            // pack's width; the body falls back to the four-word path otherwise.
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
                            var warps: usize = fma_warps;
                            if (std.c.getenv("TF_DSV41_FMA_WARPS")) |v| {
                                if (std.fmt.parseInt(usize, std.mem.span(v), 10)) |w| {
                                    if (w == 1 or w == 2 or w == 4 or w == 8 or w == 16 or w == 32) warps = w;
                                } else |_| {}
                            }
                            // The FMA body's prefetch ring depth (TF_FMA_PF blocks in flight). TF_DSV41_FMA_PF=1 is
                            // the shipped path.
                            var fma_pf: usize = 1;
                            if (std.c.getenv("TF_DSV41_FMA_PF")) |v| {
                                if (std.fmt.parseInt(usize, std.mem.span(v), 10)) |d| {
                                    if (d == 1 or d == 2 or d == 4 or d == 8) fma_pf = d;
                                } else |_| {}
                            }
                            if (fma_pf != 1) std.log.info("exl3: FMA prefetch ring depth {d}", .{fma_pf});
                            if (std.fmt.allocPrint(g, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_PF {d}\n{s}", .{ warps, fma_pf, src })) |src_fma| {
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
                                    // The same body with the static state selection and the codebook read from a
                                    // 1024-entry threadgroup LUT.
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) |pl| {
                                        if (pl.simdWidth() == 32 and pl.maxThreads() >= 32 * warps) {
                                            decode_fma_static_lut = pl;
                                            std.log.info("exl3: register-only decode pipeline active (tf_exl3_matmul_decode_fma_static_lut, {d} warps)", .{warps});
                                        } else pl.deinit();
                                    }
                                    // The same body on the dense one-row path: a dense matrix has no expert
                                    // descriptor, so this variant reads the trellis from the buffer and the strides,
                                    // the width and the codebook from the args.
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_dense", false) catch null) |pd| {
                                        if (pd.simdWidth() == 32 and pd.maxThreads() >= 32 * warps) {
                                            decode_fma_dense = pd;
                                            std.log.info("exl3: dense register-only decode pipeline active (tf_exl3_matmul_decode_fma_dense, {d} warps)", .{warps});
                                        } else pd.deinit();
                                    }
                                    // The same body with two and four rows a threadgroup: the weight decode is shared
                                    // by the rows, and every row's sum is bit-identical.
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
                                    // The paired routed decode: the entries that name the same expert share one
                                    // weight decode.
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_static_pair", false) catch null) |pd| {
                                        if (pd.simdWidth() == 32 and pd.maxThreads() >= 32 * warps) {
                                            decode_fma_static_pair = pd;
                                            std.log.info("exl3: paired routed register-only decode pipeline active (tf_exl3_matmul_decode_fma_static_pair, {d} warps)", .{warps});
                                        } else pd.deinit();
                                    }
                                    if (mtl.Pipeline.init(device, libfma, "tf_exl3_matmul_decode_fma_static_pair_lut", false) catch null) |pd| {
                                        if (pd.simdWidth() == 32 and pd.maxThreads() >= 32 * warps) {
                                            decode_fma_static_pair_lut = pd;
                                            std.log.info("exl3: paired routed register-only decode pipeline active (tf_exl3_matmul_decode_fma_static_pair_lut, {d} warps)", .{warps});
                                        } else pd.deinit();
                                    }
                                } else std.log.err("exl3: fma decode library compile failed", .{});
                            } else |_| {}
                        }
                        if (mtl.Pipeline.init(device, nax_lib, "tf_exl3_matvec_nax", false)) |pf| {
                            if (pf.simdWidth() == 32 and pf.maxThreads() >= 128) {
                                matvec_nax = pf;
                                std.log.info("exl3: tensor-unit matvec pipeline active (tf_exl3_matvec_nax)", .{});
                            } else pf.deinit();
                        } else |err| std.log.err("exl3: matvec nax pipeline init failed: {s}", .{@errorName(err)});
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
        return .{ .rotate_in = ri, .matmul = mm, .rotate_out = ro, .swiglu_prerotate = sr, .dense = dn, .prefill = prefill, .prefill_fp8 = prefill_fp8, .prefill_fp8b = prefill_fp8b, .prefill_cast = prefill_cast, .reduce = reduce, .prefill_nax = prefill_nax, .prefill_nax_tiles = prefill_nax_tiles, .decode_nax = decode_nax, .decode_nax_c = decode_nax_c, .decode_nax_tab = decode_nax_tab, .decode_nax_ctab = decode_nax_ctab, .decode_nax_ctab_f = decode_nax_ctab_f, .decode_fma = decode_fma, .decode_fma_static = decode_fma_static, .decode_fma_static_lut = decode_fma_static_lut, .decode_fma_dense = decode_fma_dense, .decode_fma_dense_multi2 = decode_fma_dense_multi2, .decode_fma_dense_multi4 = decode_fma_dense_multi4, .decode_fma_static_pair = decode_fma_static_pair, .decode_fma_static_pair_lut = decode_fma_static_pair_lut, .fma_warps = fma_warps, .matvec_nax = matvec_nax, .int8_tile = int8_tile, .matmul_tab = if (table != null) mmt else null, .matmul_wide = if (table != null) mmw else null, .matmul_direct = if (table != null) mmd else null, .matmul_static = if (table != null) mms else null, .matmul_static_lut = if (table != null) mmsl else null, .codebook_table = table, .prefill_nax_db = prefill_nax_db, .prefill_nax_db64 = prefill_nax_db64, .prefill_nax_dense_db = prefill_nax_dense_db, .prefill_nax_dense_rpg2 = prefill_nax_dense_rpg2, .prefill_nax_moe_db = prefill_nax_moe_db, .prefill_nax_funnel = prefill_nax_funnel, .prefill_nax_skip_pad = prefill_nax_skip_pad, .prefill_nax_wide = prefill_nax_wide, .prefill_nax_funnel_only = prefill_nax_funnel_only, .prefill_nax_skip_pad128 = prefill_nax_skip_pad128, .prefill_nax_warp32 = prefill_nax_warp32, .prefill_nax_coop = prefill_nax_coop, .prefill_nax_coop_skip = prefill_nax_coop_skip, .prefill_nax_dense_funnel = prefill_nax_dense_funnel };
    }
    pub fn deinit(p: *Pipelines) void {
        p.rotate_in.deinit();
        p.matmul.deinit();
        p.rotate_out.deinit();
        if (p.norm_transform) |v| v.deinit();
        if (p.prefill_fp8) |v| v.deinit();
        if (p.prefill_fp8b) |v| v.deinit();
        if (p.prefill_cast) |v| v.deinit();
        p.swiglu_prerotate.deinit();
        p.dense.deinit();
        if (p.reduce) |pf| pf.deinit();
        if (p.codebook_table) |b| b.deinit();
        if (p.prefill) |pf| pf.deinit();
        if (p.prefill_nax) |pf| pf.deinit();
        for (p.prefill_nax_tiles) |maybe| if (maybe) |pf| pf.deinit();
        if (p.prefill_nax_skip_pad128) |pf| pf.deinit();
        if (p.prefill_nax_warp32) |pf| pf.deinit();
        if (p.prefill_nax_coop) |pf| pf.deinit();
        if (p.prefill_nax_coop_skip) |pf| pf.deinit();
        if (p.prefill_nax_dense_funnel) |pf| pf.deinit();
        if (p.prefill_nax_db) |pf| pf.deinit();
        if (p.prefill_nax_db64) |pf| pf.deinit();
        if (p.prefill_nax_dense_db) |pf| pf.deinit();
        if (p.prefill_nax_dense_rpg2) |pf| pf.deinit();
        if (p.prefill_nax_wide) |pf| pf.deinit();
        if (p.prefill_nax_funnel_only) |pf| pf.deinit();
        if (p.prefill_nax_moe_db) |pf| pf.deinit();
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
        if (p.decode_fma_static_pair) |pf| pf.deinit();
        if (p.decode_fma_static_pair_lut) |pf| pf.deinit();
        if (p.matvec_nax) |pf| pf.deinit();
        if (p.matmul_tab) |pf| pf.deinit();
        if (p.matmul_wide) |pf| pf.deinit();
        if (p.matmul_direct) |pf| pf.deinit();
        if (p.matmul_static) |pf| pf.deinit();
        if (p.matmul_static_lut) |pf| pf.deinit();
        if (p.int8_tile) |pf| pf.deinit();
    }
};

/// TF_DSV41_EXL3_NAX_MIN_ROWS (default 64) sets the row count at which the tensor-unit kernel is used;
/// TF_DSV41_EXL3_SHARED_MIN_ROWS (default 32) does the same for the shared prefill kernel. Raising them forces the
/// scalar path in the SAME binary, which is how the tensor-unit kernel is isolated from every other difference
/// between trees. The defaults reproduce the shipped behaviour exactly, so a build with these gates unset is
/// unchanged.
var nax_min_rows_cache: ?u32 = null;
var shared_min_rows_cache: ?u32 = null;
var nax_tile_cache: ?u32 = null;
/// Split count for the routed-expert decode matmuls. The order differs from the sequential path, but both canon
/// hashes are unchanged on the probes.
fn moeSplitCount() usize {
    return envRows("TF_DSV41_EXL3_MOE_SPLITS", 8, &moe_splits_cache);
}
var moe_splits_cache: ?u32 = null;
var splits_override: ?u32 = null;
var wide_splits_cache: ?u32 = null;
var moe_rows_cache: ?u32 = null;
var splits_override_set: bool = false;
var nax_cols_cache: ?u32 = null;
var nax_db_logged: bool = false;
var routed_logged: bool = false;
var dense_log_count: u32 = 0;
/// The log counter for the calls the eight-row dense gate moves.
var dense8_log_count: u32 = 0;
var matmul_tab_logged: bool = false;
var direct_log_count: u32 = 0;

/// TF_DSV41_EXL3_TABLE_DECODE: default on; `=0` restores the ALU codebook for a same-binary A/B. The table's values
/// are the format's own decoder's, so the default path's arithmetic is unchanged.
fn tableDecodeOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_TABLE_DECODE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_WIDE_DECODE: the 128-column one-row table body. `=0` restores the 16-column body for a same-binary
/// A/B; the outputs are bit-identical (identical at every split count).
fn wideDecodeOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_WIDE_DECODE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_DIRECT_DECODE: the direct-register one-row table body. `=0` restores the 128-column shared-tile body
/// for a same-binary A/B; the outputs are bit-identical (identical at every split count and shape).
fn directDecodeOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_DIRECT_DECODE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_FMA_LUT: the codebook read from a 1024-entry threadgroup LUT in the register-only routed decode
/// body. `=0` restores the resident table for a same-binary A/B; the outputs are bit-identical.
fn fmaLutOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_FMA_LUT") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// True when the indexed decode body that honours a descriptor's own output width is the one the stages will pick for
/// a rows <= 8 indexed call -- the register-only body. The attention's wq_a (n 1280) and wkv (n 512) batch into one
/// indexed call, so the caller gates on this: a fallback body that only knows the call's n would read a narrower
/// entry's stream past its last column.
pub fn indexedWidthBody(p: *const Pipelines) bool {
    if (p.decode_nax == null or p.decode_fma == null) return false;
    if (std.c.getenv("TF_DSV41_EXL3_SCALAR_EXPERTS") != null) return false;
    if (std.c.getenv("TF_DSV41_EXL3_FMA")) |v| {
        if (v[0] == '0') return false;
    }
    return true;
}

/// TF_DSV41_EXL3_LUT_DECODE: the codebook read from a 1024-entry threadgroup LUT in the static one-row body. `=0`
/// restores the resident table for a same-binary A/B; the outputs are bit-identical.
fn lutDecodeOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_LUT_DECODE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_FMA_STATIC: the static state selection in the register-only routed decode body. `=0` restores the
/// shipped cursor for a same-binary A/B; the outputs are bit-identical.
fn fmaStaticOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_FMA_STATIC") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_FMA_DENSE: the register-only body on the dense one-row path. `=0` restores the shipped 128-column
/// threadgroup-tile body for a same-binary A/B. The reduction class is the routed body's (the register-only body's),
/// not the dense body's, so the arm needed the FP64 reference for the dense one-row class before it could default on:
/// it stands (the register-only body is the more accurate of the two at the pack's widths 10 and 12, as it was
/// for the routed class), so it defaults on.
fn fmaDenseOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_FMA_DENSE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_STATIC_DECODE: the static state selection in the one-row body (`tf_exl3_matmul_wide_static_tab`).
/// `=0` restores the shipped dynamic cursor for a same-binary A/B; the outputs are bit-identical (the hash is
/// equal to the shipped body's at every shape and split count).
fn staticDecodeOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_STATIC_DECODE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}
var moe_cols_cache: ?u32 = null;
/// (instrument): a one-shot line the first time the paired body is selected, so an in-situ A/B can say whether the
/// call-site hint fired at all.
var pair_logged: bool = false;
var fma_log_count: usize = 0;
var dense_db_cache: ?u32 = null;

/// TF_DSV41_EXL3_DENSE_DB: the dense prompt attention projections on the double-buffered 128x64 body. Same k order
/// and MMA order as the shipped body, so the output is bit-identical. Default on: the outputs are identical on both
/// canons, and the 20-prompt band holds at 294/294. Set =0 to force the schedule back.
fn denseDb() bool {
    return envRows("TF_DSV41_EXL3_DENSE_DB", 1, &dense_db_cache) != 0;
}

var dense_rpg2_cache: ?u32 = null;

/// TF_DSV41_EXL3_DENSE_RPG2: the dense prompt tile on the thirty-two-row simdgroup body: the NAXTile shape. The lane
/// owns eight states instead of four, so the stream words it reads cover twice the states, and the weight fragments
/// are read once for two row groups, so the tile's threadgroup traffic per row halves. The values, their order, the k
/// order and the MMA order per output are the single-tile body's, so the output is bit-identical.
fn denseRpg2() bool {
    return envRows("TF_DSV41_EXL3_DENSE_RPG2", 0, &dense_rpg2_cache) != 0;
}

var moe_db_cache: ?u32 = null;

/// TF_DSV41_EXL3_FUNNEL_PREFILL: the routed prompt tile reads its states from the two-word funnel -- the structure
/// (one funnel shift a block, the eight windows at compile-time shifts) on our map and our ring. The values, their
/// order and the k order are the shipped body's, so the output is bit-identical (the hash is equal to the
/// shipped body's at the routed shape). `=0` restores the shipped cursor for a same-binary A/B.
/// TF_DSV41_EXL3_SKIP_PAD: the routed prompt tile skips the multiply of the grouping's padded rows (the padding rows
/// carry the sentinel expert id, so a warp reads its own sixteen route ids and skips). The engine's grouping pads
/// each expert's block to sixty-four rows, so a 3,619-token prompt's 21,714 routed rows are dispatched as 45,888 rows
/// -- 2.11 times the real work. The real rows' values, their order and the k order are the shipped body's, so the
/// reply is bit-identical. `=0` restores the shipped multiply for a same-binary A/B.
fn skipPadOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_SKIP_PAD") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_WIDE_WORDS: the routed prompt tile reads its stream word as one 4-byte load -- the read -- instead
/// of two 2-byte loads. The same bytes, so the output is bit-identical; the kernel falls back to the shipped read
/// when the tile's byte address is not 4-byte aligned.
fn wideWordsOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_WIDE_WORDS") orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_FUNNEL_ONLY: the routed prompt tile takes the funnel path as its only path, so the cursor arrays the
/// Fast path needs -- off, sh and rel -- are not live across the k loop. The shipped default (the funnel path is the
/// one the routed pack's width takes); `=0` restores the shipped body.
fn funnelOnlyOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_FUNNEL_ONLY") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_DENSE_FUNNEL: the dense prompt tile reads its states from the two-word funnel (the wide form's
/// structure, ) instead of the cursor's four-word read. The dense tile is 128 rows, so a lane owns four states and
/// the funnel reads a twenty-five-bit run -- two words. Bit-identical to the shipped body.
var dense_funnel8_cache: ?u32 = null;

fn denseFunnelOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_DENSE_FUNNEL") orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_DENSE_FUNNEL8 (, default OFF): the dense prompt tile's funnel read at halves = 8, the tap
/// projection's width. The dense tile's lane owns four states four bits apart, a 28-bit run, so the funnel's two
/// words cover it where the shipped body reads four. Default off: no engine path selects it; `=1` is the instrument.
fn denseFunnel8On() bool {
    return envRows("TF_DSV41_EXL3_DENSE_FUNNEL8", 0, &dense_funnel8_cache) != 0;
}

/// TF_DSV41_EXL3_COOP_PREFILL: the routed prompt tile reads its block's twenty-four words once a warp (one lane a
/// word) and takes the two it needs by a shuffle, instead of every lane reading them: the same words, the same
/// values, the same order (bit-identical). `=0` restores the per-lane read.
fn coopPrefillOn() bool {
    // The flag keeps the instrument.
    const v = std.c.getenv("TF_DSV41_EXL3_COOP_PREFILL") orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_WARP32: the 128-row routed tile with the thirty-two-row warp: a warp owns two sixteen-row groups, so
/// the block is decoded by one warp and the lane keeps eight states and eight bits a weight. `=0` restores the
/// four-state lane for the A/B.
fn warp32On() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_WARP32") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

fn funnelPrefillOn() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_FUNNEL_PREFILL") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_EXL3_MOE_DB: the routed 64x64 prompt tile on the double-buffered ALU body .
fn moeDb() bool {
    return envRows("TF_DSV41_EXL3_MOE_DB", 0, &moe_db_cache) != 0;
}

/// TF_DSV41_EXL3_SPLITS pins the one-row split count for the whole binary; unset keeps the per-shape default.
fn splitOverride() ?u32 {
    if (!splits_override_set) {
        splits_override_set = true;
        if (std.c.getenv("TF_DSV41_EXL3_SPLITS")) |v| splits_override = std.fmt.parseInt(u32, std.mem.span(v), 10) catch null;
    }
    return splits_override;
}

fn envRows(name: [:0]const u8, default: u32, cache: *?u32) u32 {
    if (cache.*) |v| return v;
    var out = default;
    if (std.c.getenv(name)) |v| out = std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
    cache.* = out;
    return out;
}

const Args = extern struct {
    k: u32,
    n: u32,
    rows: u32,
    halves: u32,
    codebook: u32,
    su_signs: u32,
    sv_signs: u32,
    has_bias: u32,
    output_f32: u32,
    input_ids: u32,
    output_ids: u32,
    routed: u32,
    expert_id: u32,
    repeat: u32,
    indexed: u32 = 0,
    experts: u32 = 0,
    splits: u32 = 1,
    x_stride: u64,
    y_stride: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
    /// The call site's pair hint. Set only where the duplicates are.
    pair: u32 = 0,
};

fn shape(k: usize, n: usize) !void {
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0 or
        k > std.math.maxInt(u32) or n > std.math.maxInt(u32)) return error.UnsupportedExl3Shape;
}
fn rotatedBytes(k: usize, rows: usize) !usize {
    const bytes = try std.math.mul(usize, try std.math.mul(usize, k, rows), 2);
    return (try std.math.add(usize, bytes, 15)) & ~@as(usize, 15);
}

/// Caller allocates this once for its largest window; scratch never holds a dequantized full weight matrix.
pub fn scratchBytes(k: usize, n: usize, rows: usize) !usize {
    try shape(k, n);
    if (rows == 0 or rows > std.math.maxInt(u32) - 7) return error.InvalidExl3Rows;
    var bytes = std.math.add(usize, try rotatedBytes(k, rows), try std.math.mul(usize, try std.math.mul(usize, n, rows), 4)) catch return error.UnsupportedExl3Shape;
    // The split dispatch's partials, one partial row a split a row of the call (it also covers the decode tile's
    // eight splits by eight rows).
    if (rows <= 8) bytes = std.math.add(usize, bytes, try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, one_row_splits, rows), n), 4)) catch return error.UnsupportedExl3Shape;
    // A decode tile of a few routed rows can split its k range too: room for every split's rows by n, and the fused
    // gate+up call carries two six-row slots a token, so the bound is the fused call's row count.
    if (rows <= moe_split_rows) bytes = std.math.add(usize, bytes, try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, moeSplitCount(), @max(rows, 8)), n), 4)) catch return error.UnsupportedExl3Shape;
    return bytes;
}

/// Split-K count for a one-row projection: one threadgroup a column tile leaves the GPU idle, so the k range is cut
/// into this many slices and the partials summed in split order.
pub const one_row_splits: usize = 16;

/// Routed rows the decode tile can carry in one call with its k range split: eight shipped rows, and the fused
/// gate+up call's two six-row slots a token out to a five-row window (sixty rows).
pub const moe_split_rows: usize = 64;

/// Split-K count for a one-row projection: narrow matrices otherwise get a handful of threadgroups.
pub fn splitCount(n: usize) usize {
    const tiles = @max(n / 16, 1);
    const want = @max(1024 / tiles, 1);
    return @min(want, 32);
}
fn check(v: View, bytes: usize, alignment: usize) !void {
    const len = v.buffer.length();
    if (v.offset % alignment != 0 or v.offset > len or bytes > len - v.offset) return error.InvalidExl3View;
}

fn disjoint(a: View, a_bytes: usize, b: View, b_bytes: usize) !void {
    if (a.buffer.id == b.buffer.id and a.offset < b.offset + b_bytes and b.offset < a.offset + a_bytes)
        return error.AliasedExl3View;
}

fn readonly(v: View, bytes: usize, y: View, y_bytes: usize, scratch: View, scratch_bytes: usize) !void {
    try disjoint(v, bytes, y, y_bytes);
    try disjoint(v, bytes, scratch, scratch_bytes);
}
fn rowsArgs(k: usize, n: usize, r: Rows, x: View, y: View, scratch: View) !Args {
    const scratch_size = try scratchBytes(k, n, r.count);
    if (r.x_stride < k or r.y_stride < n) return error.InvalidExl3Stride;
    const repeat = if (r.routes) |v| v.repeat else 1;
    if (repeat == 0 or repeat > std.math.maxInt(u32) or r.count % repeat != 0) return error.InvalidExl3Repeat;
    if ((r.input_ids != null and r.input_rows == 0) or
        (r.output_ids != null and r.output_rows == 0)) return error.InvalidExl3Capacity;
    const xrows = if (r.input_ids != null) r.input_rows else r.count / repeat;
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const xend = try std.math.add(usize, try std.math.mul(usize, xrows - 1, r.x_stride), k);
    const yend = try std.math.add(usize, try std.math.mul(usize, yrows - 1, r.y_stride), n);
    const xbytes = try std.math.mul(usize, xend, 2);
    const ybytes = try std.math.mul(usize, yend, if (r.output == .f32) @as(usize, 4) else 2);
    try check(x, xbytes, 2);
    try check(y, ybytes, if (r.output == .f32) @as(usize, 4) else 2);
    try check(scratch, scratch_size, 16);
    // A prerotated call's `x` is the scratch buffer's rotated base by construction, so only the write guards apply.
    if (r.prerotated) {
        try check(x, xbytes, 2);
        try disjoint(x, xbytes, y, ybytes);
    } else try readonly(x, xbytes, y, ybytes, scratch, scratch_size);
    try disjoint(y, ybytes, scratch, scratch_size);
    const id_bytes = try std.math.mul(usize, r.count, 4);
    for ([_]?View{ r.input_ids, r.output_ids, if (r.routes) |v| v.ids else null }) |optional| {
        if (optional) |v| {
            try check(v, id_bytes, 4);
            try readonly(v, id_bytes, y, ybytes, scratch, scratch_size);
        }
    }
    return .{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(r.count),
        .halves = 0,
        .codebook = 0,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = @intFromBool(r.output == .f32),
        .input_ids = @intFromBool(r.input_ids != null),
        .output_ids = @intFromBool(r.output_ids != null),
        .routed = @intFromBool(r.routes != null),
        .expert_id = if (r.routes) |v| v.expert_id else 0,
        .repeat = @intCast(repeat),
        .x_stride = r.x_stride,
        .y_stride = r.y_stride,
        .tile_k_stride = 0,
        .tile_n_stride = 0,
        .pair = @intFromBool(r.pair),
    };
}
/// The f32 sums' offset in a call's scratch -- the rotated region rounded to 16 bytes. The fold's caller reads them
/// with `encodeNormTransform`.
pub fn sumsOffset(k: usize, rows: usize) usize {
    return (@as(usize, k) * rows * 2 + 15) & ~@as(usize, 15);
}

pub const NormTransform = extern struct {
    n0: u32,
    n1: u32,
    k_next: u32,
    su_signs: u32,
    sv_signs0: u32,
    sv_signs1: u32,
    eps: f32,
};

/// THE ONE-ROW TRANSFORM FOLD. Reads the merged wq_a+wkv call's f32 sums (its dispatch ran with `no_epilogue`),
/// writes the output transform of both entries (qr 1280, raw 512), the q_norm of qr, and wq_b's rotated input into
/// the scratch base, all with the shipped kernels' arithmetic and order (see the metal kernel's header). The next
/// call runs with `prerotated`.
pub fn encodeNormTransform(p: *const Pipelines, e: mtl.ComputeEncoder, sums: View, sv0: View, sv1: View, weight: View, su_next: View, qr: View, raw: View, rotated: View, d: NormTransform) !void {
    const pipe = p.norm_transform orelse return error.UnsupportedExl3Shape;
    e.setPipeline(pipe);
    bind(e, sums, 0);
    bind(e, sv0, 1);
    bind(e, sv1, 2);
    bind(e, weight, 3);
    bind(e, su_next, 4);
    bind(e, qr, 5);
    bind(e, raw, 6);
    bind(e, rotated, 7);
    e.setValue(d, 8);
    e.dispatchGroups(.{ .width = 2, .height = 1, .depth = 1 }, .{ .width = 256 });
}

fn bind(e: mtl.ComputeEncoder, v: View, index: usize) void {
    e.setBuffer(v.buffer, v.offset, index);
}

/// No allocation or commit; retain all views until completion, disjoint from writes, with valid GPU IDs.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, m: Matrix, x: View, y: View, scratch: View, r: Rows) !void {
    const storage_bytes = try m.checkStorage();
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    try matrixWrites(m, storage_bytes, y, scratch, r);
    a.halves = m.width.halves;
    a.codebook = @intCast(@backingInt(m.codebook));
    a.su_signs = @intFromBool(m.suh_format == .packed_signs);
    a.sv_signs = @intFromBool(m.svh_format == .packed_signs);
    a.has_bias = @intFromBool(m.bias != null);
    a.tile_k_stride = try m.kStride();
    a.tile_n_stride = m.nStride();
    stages(p, e, a, m.trellis, m.suh, m.svh, m.bias orelse m.svh, scratch, x, y, scratch, r);
}

/// A kept-dense EXL3 matrix: a bf16 weight beside the input/output rotation scales, no trellis.
pub const DenseMatrix = struct {
    weight: View,
    suh: View,
    svh: View,
    bias: ?View = null,
    k: usize,
    n: usize,
};

/// Kept-dense: quantized rotate stages, trellis to dense, f32 in rotate_out sums; sequential k, as the fp64
/// reference.
pub fn encodeDense(p: *const Pipelines, e: mtl.ComputeEncoder, m: DenseMatrix, x: View, y: View, scratch: View, r: Rows) !void {
    try shape(m.k, m.n);
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    a.su_signs = 0;
    a.sv_signs = 0;
    a.has_bias = @intFromBool(m.bias != null);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
    const sums_bytes = try std.math.mul(usize, try std.math.mul(usize, m.n, r.count), 4);
    try readonly(m.weight, try std.math.mul(usize, try std.math.mul(usize, m.n, m.k), 2), y, 0, scratch, sums_bytes);
    try readonly(m.suh, m.k * 2, y, 0, scratch, sums_bytes);
    try readonly(m.svh, m.n * 2, y, 0, scratch, sums_bytes);
    if (m.bias) |b| try readonly(b, m.n * 2, y, 0, scratch, sums_bytes);
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, m.suh, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    e.dispatchGroups(.{ .width = a.k / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
    e.setPipeline(p.dense);
    bind(e, scratch, 0);
    bind(e, m.weight, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    e.dispatchThreads(.{ .width = a.n, .height = a.rows }, .{ .width = 64 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    e.dispatchGroups(.{ .width = a.n / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
}

/// The int8 tile path for one dense call: rotate in, the int8 tiles into `sums`, rotate out. The caller owns the
/// buffers; `sums` is cleared here because the kernel accumulates into it.
pub fn encodeInt8Tile(p: *const Pipelines, e: mtl.ComputeEncoder, m: Matrix, x: View, y: View, scratch: View, r: Rows) !void {
    if (p.int8_tile == null or p.codebook_table == null) return error.UnsupportedExl3Int8;
    const storage_bytes = try m.checkStorage();
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    a.halves = m.width.halves;
    a.codebook = @intCast(@backingInt(m.codebook));
    a.su_signs = @intFromBool(m.suh_format == .packed_signs);
    a.sv_signs = @intFromBool(m.svh_format == .packed_signs);
    a.has_bias = @intFromBool(m.bias != null);
    a.tile_k_stride = try m.kStride();
    a.tile_n_stride = m.nStride();
    a.splits = 1;
    try matrixWrites(m, storage_bytes, y, scratch, r);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, m.suh, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, r.input_ids orelse scratch, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    const rotate_width: usize = (a.k / 128 + 3) / 4;
    e.dispatchGroups(.{ .width = rotate_width, .height = a.rows }, .{ .width = 128 });
    e.barrier();
    e.setPipeline(p.int8_tile.?);
    bind(e, scratch, 0);
    bind(e, m.trellis, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, scratch, 5);
    e.setBuffer(p.codebook_table.?, 0, 7);
    e.dispatchGroups(.{ .width = (a.n + 15) / 16, .height = (a.rows + 15) / 16 }, .{ .width = 32 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, r.output_ids orelse scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    const rotate_out_width: usize = (a.n / 128 + 3) / 4;
    e.dispatchGroups(.{ .width = rotate_out_width, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}

/// The prompt-row int8 path for a dense projection. `direct` is a pre-rotated pack (the weight carries suh/H_K and
/// H_N/svh), so the call is one byte body into the caller's bf16 output; otherwise the shipped rotate_in and
/// rotate_out bracket the byte body exactly as the trellis path brackets its GEMM. The row scale is applied by the
/// body's own epilogue, so an int8 pack's scale never reaches rotate_out a second time.
pub fn encodePrefillInt8(p: *const Pipelines, e: mtl.ComputeEncoder, m: Matrix, side: Int8, x: View, y: View, scratch: View, r: Rows, direct: bool) !void {
    if (p.prefill_fp8 == null or p.prefill_fp8b == null) return error.UnsupportedExl3Int8;
    if (m.k != side.k or m.n != side.n) return error.InvalidExl3Stride;
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    a.halves = 0;
    a.codebook = 2;
    a.su_signs = 0;
    a.sv_signs = 0;
    a.has_bias = 0;
    a.tile_k_stride = (m.n / 16) * 16;
    a.tile_n_stride = 16;
    const weight_bytes = try std.math.mul(usize, m.k, m.n);
    try check(side.weight, weight_bytes, 16);
    try check(side.scales, try std.math.mul(usize, m.n, 4), 4);
    if (direct) {
        if (r.output != .bf16 or p.prefill_cast == null) return error.UnsupportedExl3Output;
        if (std.c.getenv("TF_DSV41_I8_DEBUG") != null) std.log.info("i8c direct rows={d} k={d} n={d} xs={d} ys={d} yoff={d}", .{ a.rows, a.k, a.n, a.x_stride, a.y_stride, y.offset });
        try readonly(side.weight, weight_bytes, y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
        try readonly(side.scales, try std.math.mul(usize, m.n, 4), y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
        const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
        e.setPipeline(p.prefill_fp8.?);
        bind(e, x, 0);
        bind(e, side.weight, 1);
        bind(e, sums, 2);
        e.setValue(a, 3);
        bind(e, side.scales, 4);
        e.dispatchGroups(.{ .width = (a.n + 63) / 64, .height = (a.rows + 127) / 128 }, .{ .width = 256 });
        e.barrier();
        e.setPipeline(p.prefill_cast.?);
        bind(e, sums, 0);
        bind(e, y, 1);
        e.setValue(a, 2);
        e.dispatchThreads(.{ .width = a.n, .height = a.rows }, .{ .width = 64 });
        e.barrier();
        return;
    }
    try readonly(side.weight, weight_bytes, y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
    try readonly(side.scales, try std.math.mul(usize, m.n, 4), y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, m.suh, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    e.dispatchGroups(.{ .width = a.k / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
    // The rotated activation is contiguous at k; the call x stride belongs to the raw input the rotation read.
    a.x_stride = m.k;
    e.setPipeline(p.prefill_fp8.?);
    bind(e, scratch, 0);
    bind(e, side.weight, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, side.scales, 4);
    e.dispatchGroups(.{ .width = (a.n + 63) / 64, .height = (a.rows + 127) / 128 }, .{ .width = 256 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    e.dispatchGroups(.{ .width = a.n / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
}

/// GPU routes choose caller-validated resident descriptors; grouped padding uses maxInt(u32).
pub fn encodeIndexed(p: *const Pipelines, e: mtl.ComputeEncoder, experts: Experts, x: View, y: View, scratch: View, r: Rows) !void {
    if (r.routes == null or experts.count == 0 or experts.count > std.math.maxInt(u32)) return error.InvalidExl3Experts;
    if (r.grouped and (r.input_ids == null or r.output_ids == null or r.count % 8 != 0))
        return error.InvalidExl3GroupedRows;
    try shape(experts.k, experts.n);
    try check(experts.descriptors, try std.math.mul(usize, experts.count, @sizeOf(ExpertDescriptor)), 8);
    var a = try rowsArgs(experts.k, experts.n, r, x, y, scratch);
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const ybytes = ((yrows - 1) * r.y_stride + experts.n) * (if (r.output == .f32) @as(usize, 4) else 2);
    try readonly(experts.descriptors, experts.count * @sizeOf(ExpertDescriptor), y, ybytes, scratch, try scratchBytes(experts.k, experts.n, r.count));
    a.indexed = if (r.grouped) 2 else 1;
    a.experts = @intCast(experts.count);
    stages(p, e, a, scratch, scratch, scratch, scratch, experts.descriptors, x, y, scratch, r);
}

fn matrixWrites(m: Matrix, storage_bytes: usize, y: View, scratch: View, r: Rows) !void {
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const ybytes = ((yrows - 1) * r.y_stride + m.n) * (if (r.output == .f32) @as(usize, 4) else 2);
    const scratch_bytes = try scratchBytes(m.k, m.n, r.count);
    try readonly(m.trellis, storage_bytes, y, ybytes, scratch, scratch_bytes);
    try readonly(m.suh, m.k * 2 / (if (m.suh_format == .packed_signs) @as(usize, 16) else 1), y, ybytes, scratch, scratch_bytes);
    try readonly(m.svh, m.n * 2 / (if (m.svh_format == .packed_signs) @as(usize, 16) else 1), y, ybytes, scratch, scratch_bytes);
    if (m.bias) |b| try readonly(b, m.n * 2, y, ybytes, scratch, scratch_bytes);
}

const ReduceArgs = extern struct { splits: u32, n: u32 };

fn stages(p: *const Pipelines, e: mtl.ComputeEncoder, args: Args, trellis: View, su: View, sv: View, bias: View, descriptors: View, x: View, y: View, scratch: View, r: Rows) void {
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
    const wide_splits: usize = envRows("TF_DSV41_EXL3_SPLITS_WIDE", 8, &wide_splits_cache);
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
    const fma_dense_multi4 = fma_dense_ok and multi_rows == 4 and a.rows >= 4 and (multi_ceil or a.rows % 4 == 0);
    const fma_dense_multi2 = fma_dense_ok and multi_rows >= 2 and a.rows >= 2 and (multi_ceil or a.rows % 2 == 0) and
        (multi_rows == 2 or !fma_dense_multi4);
    const fma_dense = fma_dense_ok and fmaDenseOn() and !fma_dense_multi4 and !fma_dense_multi2;
    // The default is therefore the shipped single-row body. TF_DSV41_EXL3_FMA_PAIR=1 forces the pair on every routed
    // call (the A/B arm); =2 honours the call-site hint alone (the instrument); unset or =0 is the shipped path.
    const pair_env = std.c.getenv("TF_DSV41_EXL3_FMA_PAIR");
    const pair_off = pair_env != null and pair_env.?[0] == '0' and pair_env.?[1] == 0;
    const pair_force = pair_env != null and !pair_off and pair_env.?[0] == '1' and pair_env.?[1] == 0;
    const pair_hint = pair_env != null and !pair_off and pair_env.?[0] == '2' and pair_env.?[1] == 0;
    const pair_on = pair_force or (pair_hint and a.pair != 0);
    const fma_pair_lut = pair_on and p.decode_fma_static_pair_lut != null and fma_static_lut;
    const fma_pair = pair_on and p.decode_fma_static_pair != null and fma_static and !fma_static_lut;
    if ((fma_pair_lut or fma_pair) and !pair_logged) {
        pair_logged = true;
        std.log.info("exl3: PAIRED body selected (rows {d}, indexed {d}, hint {d})", .{ a.rows, a.indexed, a.pair });
    }
    const splits: usize = if (fma or fma_dense or fma_dense_multi2 or fma_dense_multi4 or fma_pair or fma_pair_lut) 1 else if (a.indexed == 0 and a.routed == 0 and (a.rows == 1 or (row_invariant and a.rows <= 8)) and p.reduce != null)
        @as(usize, splitOverride() orelse @as(u32, @intCast(if (wide) wide_splits else one_row_splits)))
    else
        moe_splits;
    a.splits = @intCast(splits);
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
    // Tile shape: rows a threadgroup covers (a dense tile can span more rows because its weights are shared) and
    // columns it covers (a wider column tile re-reads the activation tile fewer times).
    const nax_tile: usize = if (a.indexed != 0) (if (r.tile_rows != 0) r.tile_rows else 64) else envRows("TF_DSV41_EXL3_NAX_TILE_ROWS", 128, &nax_tile_cache);
    // A 256-row tile wants the narrower column tile so the threadgroup stays at 512 threads. A routed chunk's column
    // tile is the inherited dense constant (64) unless TF_DSV41_EXL3_MOE_COLS says otherwise: a 128-column tile
    // halves the number of column tiles and therefore the activation fragment re-reads, at twice the decoded weight
    // tile and half the threadgroups. Exactness is unchanged (same 16-k block order, hash-identical).
    const moe_cols = if (a.indexed != 0) envRows("TF_DSV41_EXL3_MOE_COLS", 0, &moe_cols_cache) else 0;
    const nax_cols: usize = if (a.indexed != 0 and nax_tile >= 256) 64 else if (moe_cols != 0) moe_cols else envRows("TF_DSV41_EXL3_NAX_COLS", if (a.indexed != 0) 128 else 64, &nax_cols_cache);
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
        std.c.getenv("TF_DSV41_EXL3_TABLE") != null and a.k % 64 == 0 and a.rows >= envRows("TF_DSV41_EXL3_NAX_MIN_ROWS", 64, &nax_min_rows_cache);
    if (nax_db and !nax_db_logged) {
        nax_db_logged = true;
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
    const nax_min_rows: u32 = envRows("TF_DSV41_EXL3_NAX_MIN_ROWS", 64, &nax_min_rows_cache);
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
    if (nax_dense_db and !nax_db_logged) {
        nax_db_logged = true;
        std.log.info("exl3: nax31 dense double-buffered body selected rows={d} tile={d}", .{ a.rows, nax_tile });
    }
    const shared = !nax and !decode_nax and p.prefill != null and a.rows >= envRows("TF_DSV41_EXL3_SHARED_MIN_ROWS", 32, &shared_min_rows_cache) and a.indexed == 0 and a.routed == 0;
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
    if (a.rows == 1 and a.indexed == 0 and a.routed == 0 and direct_log_count < 10) {
        direct_log_count += 1;
        std.log.info("exl3: one-row body n={d} k={d} splits={d} direct={} wide={} static={} lut={} tab={} pd={} halves={d} rows={d} idx={d} rt={d} nmod={d} kmod={d} env={}", .{ a.n, a.k, splits, matmul_direct, matmul_wide, matmul_static, matmul_static_lut, matmul_tab, p.matmul_direct != null, a.halves, a.rows, a.indexed, a.routed, a.n % 128, a.k % 16, directDecodeOn() });
    }
    if (matmul_wide and !matmul_tab_logged) {
        matmul_tab_logged = true;
        std.log.info("exl3: wide table-decode body selected rows={d} n={d} k={d} splits={d}", .{ a.rows, a.n, a.k, splits });
    }
    if (matmul_tab and !matmul_tab_logged) {
        matmul_tab_logged = true;
        std.log.info("exl3: table-decode matmul body selected rows={d} n={d} k={d} splits={d}", .{ a.rows, a.n, a.k, splits });
    }
    e.setPipeline(if (fma_pair_lut) p.decode_fma_static_pair_lut.? else if (fma_pair) p.decode_fma_static_pair.? else if (fma_static_lut) p.decode_fma_static_lut.? else if (fma_static) p.decode_fma_static.? else if (fma) p.decode_fma.? else if (fma_dense_multi4) p.decode_fma_dense_multi4.? else if (fma_dense_multi2) p.decode_fma_dense_multi2.? else if (fma_dense) p.decode_fma_dense.? else if (matvec_nax) p.matvec_nax.? else if (decode_pipe) |dp| dp else if (matmul_static_lut) p.matmul_static_lut.? else if (matmul_static) p.matmul_static.? else if (matmul_direct) p.matmul_direct.? else if (matmul_wide) p.matmul_wide.? else if (matmul_tab) p.matmul_tab.? else if (nax_db) nax_db_pick.? else if (nax_dense_rpg2) p.prefill_nax_dense_rpg2.? else if (nax_dense_funnel) p.prefill_nax_dense_funnel.? else if (nax_dense_db) p.prefill_nax_dense_db.? else if (nax_moe_db) p.prefill_nax_moe_db.? else if (nax_coop) nax_coop_pick.? else if (nax_wide) nax_wide_pick.? else if (nax_funnel_only) nax_funnel_only_pick.? else if (nax_skip_pad) nax_skip_pad_pick.? else if (nax_funnel) nax_funnel_pick.? else if (nax) nax_pick.? else if (shared) p.prefill.? else p.matmul);
    bind(e, scratch, 0);
    bind(e, trellis, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, routes, 4);
    bind(e, descriptors, 5);
    bind(e, partials, 6);
    if (p.codebook_table) |t| e.setBuffer(t, 0, 7);
    // Rows a decode threadgroup takes: one a SIMDgroup (TF_DSV41_EXL3_MOE_ROWS selects one a threadgroup).
    const moe_rows: usize = if (decode_nax) (if (compact) 1 else envRows("TF_DSV41_EXL3_MOE_ROWS", 2, &moe_rows_cache)) else 4;
    const group_rows: usize = if (fma or fma_static or fma_static_lut or fma_pair or fma_pair_lut or fma_dense_multi2 or fma_dense_multi4) 1 else if (decode_nax) moe_rows else if (nax) nax_tile else if (shared) 32 else 8;
    // The geometry must follow the PIPELINE that was chosen: the static body is the wide body's (128 columns, 256
    // threads), so it is named before the direct body's 256-column geometry.
    const group_cols: usize = if (fma or fma_static or fma_static_lut or fma_dense) 16 else if (matmul_static or matmul_static_lut) 128 else if (matmul_direct) 256 else if (matvec_nax or matmul_wide) 128 else if (nax) nax_cols else if (decode_nax) 32 else 16;
    const height: usize = if (fma_dense_multi4) (a.rows + 3) / 4 else if (fma_dense_multi2) (a.rows + 1) / 2 else if (fma or fma_static or fma_static_lut or fma_dense or fma_pair or fma_pair_lut) a.rows else if (matvec_nax) 1 else if (splits > 1 and !decode_nax) splits else if (a.indexed == 1 and !decode_nax) a.rows else (@as(usize, a.rows) + group_rows - 1) / group_rows;
    // The warp32 tile runs four warps (a warp owns two sixteen-row groups), the other nax tiles one warp a
    // sixteen-row group.
    const nax_warp32: bool = nax and a.indexed != 0 and tile_id == 2 and warp32On() and nax_skip_pad_pick != null;
    const threads: usize = if (fma or fma_static or fma_static_lut or fma_dense or fma_dense_multi2 or fma_dense_multi4 or fma_pair or fma_pair_lut) 32 * p.fma_warps else if (nax) (if (nax_dense_rpg2 or nax_warp32) nax_tile / 32 * 32 * (nax_cols / 64) else nax_tile / 16 * 32 * (nax_cols / 64)) else if (decode_nax) 32 * moe_rows else if (matmul_static or matmul_static_lut) 256 else if (matmul_direct) 256 else if (matmul_wide) 256 else if (shared or matvec_nax) 128 else 32;
    // One split dispatch over a window: a row group a threadgroup (the tile spans eight rows), the splits in y.
    const split_groups: usize = if (splits > 1 and !decode_nax) (a.rows + 7) / 8 else 1;
    if ((fma or fma_static or fma_static_lut) and fma_log_count < 8) {
        fma_log_count += 1;
        std.log.info("exl3: routed decode body rows={d} n={d} k={d} halves={d} cols={d} grid={d}x{d}x{d} threads={d} static={} fma={} experts={d}", .{ a.rows, a.n, a.k, a.halves, group_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, if (fma or fma_static) 1 else if (decode_nax) moe_splits else split_groups, threads, fma_static, fma, a.experts });
    }

    if (a.indexed == 0 and a.rows >= 64 and dense_log_count < 14) {
        dense_log_count += 1;
        const dpick: []const u8 = if (nax_dense_rpg2) "nax56" else if (nax_dense_funnel) "nax51" else if (nax_dense_db) "nax31" else if (nax) "nax_tile" else if (shared) "shared" else "matmul";
        std.log.info("exl3: dense dispatch rows={d} n={d} k={d} tile={d} cols={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, nax_tile, nax_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, dpick });
    }
    // (the instrument): the dense calls the eight-row gate moves, named.
    if (dense8_on and a.indexed == 0 and a.routed == 0 and a.rows >= 8 and a.rows < 64 and dense8_log_count < 14) {
        dense8_log_count += 1;
        std.log.info("exl3: dense8 rows={d} n={d} k={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, if (nax) "nax_tile" else if (shared) "shared" else "matmul" });
    }
    if (a.indexed != 0 and !routed_logged) {
        routed_logged = true;
        const pick: []const u8 = if (nax_db) "nax10" else if (nax_dense_db) "nax31" else if (nax_moe_db) "nax19" else if (nax_coop) (if (nax_skip_pad) "nax53" else "nax52") else if (nax_wide) "nax57" else if (nax_funnel_only) "nax59" else if (nax_skip_pad) (if (nax_warp32) "nax54" else "nax48") else if (nax_funnel) "nax42" else if (nax) "nax_tile" else if (shared) "shared" else if (decode_nax) "decode" else "matmul";
        std.log.info("exl3: indexed dispatch rows={d} n={d} k={d} tile={d} cols={d} grid={d}x{d} threads={d} body={s}", .{ a.rows, a.n, a.k, nax_tile, nax_cols, (@as(usize, a.n) + group_cols - 1) / group_cols, height, threads, pick });
    }
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
    for (0..gemm_passes) |_| e.dispatchGroups(.{ .width = (@as(usize, a.n) + group_cols - 1) / group_cols, .height = height, .depth = if (fma or fma_static or fma_static_lut or fma_dense) 1 else if (decode_nax) moe_splits else split_groups }, .{ .width = threads });
    e.barrier();
    if (splits > 1) {
        e.setPipeline(p.reduce.?);
        bind(e, partials, 0);
        bind(e, sums, 1);
        e.setValue(ReduceArgs{ .splits = @intCast(splits), .n = @as(u32, @intCast(a.rows)) * a.n }, 2);
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
    // The Args record counts the defaulted fields and the pair hint.
    try std.testing.expectEqual(@as(usize, 112), @sizeOf(Args));
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
