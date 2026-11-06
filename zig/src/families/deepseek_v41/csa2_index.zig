//! Tiled GPU scores and bounded online heaps; the memory footprint does not grow with context length.
const std = @import("std");
const max_context = @import("kv_state.zig").max_context;
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("core").frags;
const View = @import("view.zig").View;
pub const score_tile_width = 4096;

/// ROUND-411: the indexer select block's REMOVAL mask (TF_DSV41_SKIP_SELECT). A doubling arm prices a
/// dispatch's MARGINAL cost -- the second copy runs on warm data, and the merge's second pass finds the
/// heap already full, so its pushes are rejected at once -- so the block's true cost needs the dispatch
/// REMOVED, not repeated. Bit 0 the clear, bit 1 the dense-key table, bit 2 the score, bit 3 the merge,
/// bit 4 the finish.
pub const skip_clear: u32 = 1;
pub const skip_keys: u32 = 2;
pub const skip_score: u32 = 4;
pub const skip_merge: u32 = 8;
pub const skip_finish: u32 = 16;
/// Warps a score threadgroup holds and keys each warp evaluates (TF_DSV41_INDEX_WARPS). A wider group is 2 % on a
/// prefill: the kernel is instruction-bound per key, so packing mostly saves dispatch. Four keys a warp, with the
/// query hoisted, measured slower than one — the extra state per lane costs more than the loads it saves.
pub const warps_per_group_default = 8;
pub const Params = extern struct {
    rows: u32,
    heads: u32 = 32,
    width: u32 = 128,
    q_stride: u32,
    weight_stride: u32,
    position: u32,
    ratio: u32,
    keys: u32,
    key_stride: u32 = 68,
    tile_start: u32 = 0,
    tile_width: u32 = score_tile_width,
    topk: u32 = 512,
    candidate_topk: u32 = 2048,
    block_size: u32 = 8,
    use_candidates: u32 = 0,
    make_candidates: u32 = 0,
    layer: u32 = 0,
    skip_mask: u32 = 0,
    /// ROUND-469: 0 = one tile a row (the heap path), 1 = tile-major, every tile resident (the decode select).
    score_layout: u32 = 0,
    scale: f32 = 0.015625,
    // ROUND-468: the metal struct is a hand-written mirror of this one. `layer`/`skip_mask` were added here
    // without that mirror, so the kernel's `scale` read `layer`'s bits and every score came out 0.0. The
    // size is asserted on both sides; a field added here without the kernel fails the build, not the reply.
    comptime {
        if (@sizeOf(Params) != 80) @compileError("Dsv41Index (index_exl3.metal) must mirror Params");
        // ROUND-473: and the field LIST, in order: a reorder or a same-size swap is caught by neither the
        // size nor the reply. The check parses the embedded kernel source at comptime.
        @import("metal/index_mirror.zig").check(Params, @import("kernel_sources").deepseek_v41_index, "Dsv41Index");
    }
    pub fn validate(p: Params) !void {
        if (p.rows == 0 or p.heads == 0 or p.heads > 32 or p.width == 0 or p.width > 128 or p.width % 32 != 0 or
            p.q_stride < p.heads * p.width or p.weight_stride < p.heads or
            (p.ratio != 1 and p.ratio != 2) or p.key_stride < p.width / 2 + p.width / 32 or
            p.topk != 512 or p.candidate_topk != 2048 or p.block_size != 8 or
            p.tile_width != score_tile_width or p.use_candidates > 1 or p.make_candidates > 1 or
            (p.use_candidates != 0 and p.make_candidates != 0) or
            @as(u64, p.position) + p.rows > max_context or p.keys > max_context or
            !std.math.isFinite(p.scale) or p.scale <= 0 or
            @as(u64, p.rows) * p.q_stride > std.math.maxInt(u32) or
            @as(u64, p.rows) * p.weight_stride > std.math.maxInt(u32) or
            @as(u64, p.keys) * p.key_stride > std.math.maxInt(u32))
            return error.InvalidIndexerShape;
    }
};
var warps_per_group_cache: ?u32 = null;
fn warpsPerGroup() u32 {
    if (warps_per_group_cache) |v| return v;
    var out: u32 = warps_per_group_default;
    if (std.c.getenv("TF_DSV41_INDEX_WARPS")) |v| {
        const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch out;
        if (n != 0 and n <= 32 and score_tile_width % n == 0) out = n;
    }
    warps_per_group_cache = out;
    return out;
}

/// Largest index-key count the dense table is sized for (context cap over the smallest ratio).
pub const max_dense_keys: usize = 65536;

pub const Scratch = struct {
    buffer: mtl.Buffer,
    rows: u32,
    /// ROUND-531: the selection's cache key, so a batch's several segments can each keep their picks.
    sel_layer: ?u32 = null,
    sel_position: u32 = 0,
    sel_rows: u32 = 0,
    pub fn bytes(rows: u32) usize {
        // The dense key table is a key-count-sized table, not a per-row one: outside the row multiplier.
        return @as(usize, rows) * (score_tile_width * 4 + 512 * 8 + 2048 * 8 + 512 * 4 + 528 + 32 * 68 +
            score_tile_width * 8 + 8) + max_dense_keys * 128 * 2;
    }
    pub fn init(device: mtl.Device, rows: u32) !Scratch {
        if (rows == 0 or rows > 16384) return error.InvalidIndexerScratch;
        const s = Scratch{ .buffer = try device.buffer(bytes(rows), mtl.ResourceOptions.shared), .rows = rows };
        // The filter's per-row compaction counter must start at zero; the insert restores it after use.
        const counter = s.count();
        const base: [*]u8 = @ptrCast(counter.buffer.contents());
        @memset(base[counter.offset..][0 .. @as(usize, rows) * 8], 0);
        return s;
    }
    pub fn deinit(s: *Scratch) void {
        s.buffer.deinit();
    }
    pub fn scores(s: Scratch) View {
        return .{ .buffer = s.buffer };
    }
    pub fn heap(s: Scratch) View {
        return s.scores().shifted(@as(usize, s.rows) * score_tile_width * 4);
    }
    pub fn candidates(s: Scratch) View {
        return s.heap().shifted(@as(usize, s.rows) * 512 * 8);
    }
    pub fn positions(s: Scratch) View {
        return s.candidates().shifted(@as(usize, s.rows) * 2048 * 8);
    }
    pub fn windowChunk(s: Scratch) View {
        return s.positions().shifted(@as(usize, s.rows) * 512 * 4);
    }
    pub fn queryPacked(s: Scratch) View {
        return s.windowChunk().shifted(@as(usize, s.rows) * 528);
    }
    /// The keys a tile's filter kept, a row's worth per row, and the per-row compaction counter.
    pub fn picks(s: Scratch) View {
        return s.queryPacked().shifted(@as(usize, s.rows) * 32 * 68);
    }
    pub fn count(s: Scratch) View {
        return s.picks().shifted(@as(usize, s.rows) * score_tile_width * 8);
    }
    /// The dense bf16 index-key table: the NAX scorer's input, one pass per layer.
    pub fn denseKeys(s: Scratch) View {
        return s.seedEnd().shifted(@as(usize, s.rows) * 4);
    }
    /// Where the seed dispatch stopped filling the heap, a row's worth per row.
    pub fn seedEnd(s: Scratch) View {
        return s.count().shifted(@as(usize, s.rows) * 4);
    }
};
pub const Pipelines = struct {
    score: mtl.Pipeline,
    clear: mtl.Pipeline,
    merge: mtl.Pipeline,
    keys_dense: mtl.Pipeline,
    seed: mtl.Pipeline,
    filter: mtl.Pipeline,
    insert: mtl.Pipeline,
    finish: mtl.Pipeline,
    /// ROUND-411: the finish's parallel form (a threadgroup a row).
    finish_par: mtl.Pipeline,
    /// ROUND-469: the decode path's whole selection in one dispatch (a threadgroup a row).
    select_one: mtl.Pipeline,
    /// ROUND-483: the same selection with float4 reads, a two-pass radix and the ties folded
    /// into the filter (TF_DSV41_INDEX_SELECT_V=0 is the shipped form).
    select_one_v: mtl.Pipeline,
    /// The identity selection for a row whose every visible compressed key is kept.
    identity: mtl.Pipeline,
    /// The tensor-unit score tile (TF_DSV41_INDEX_NAX): the scalar scorer's term, one 128-dim MMA a head
    /// over a 64-key by 64-row tile. Null when the GPU has no tensor units.
    nax_score: ?mtl.Pipeline = null,
    /// ROUND-470: the decode-shaped score -- the keys on the op M and the head queries on its N, so a
    /// one-row step no longer pays a 64-row prompt tile tensor-unit work. Same MMA chain and epilogue.
    row_score: ?mtl.Pipeline = null,
    /// ROUND-481: the keys themselves on M (four 16-key blocks, one a SIMDgroup) at one row, so every MMA
    /// row is live; the head terms are summed 0..31 in order, so a score is bit-identical to row_score's.
    keys_score: ?mtl.Pipeline = null,
    pub fn load(device: mtl.Device) !Pipelines {
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const gpa = std.heap.page_allocator;
        const joined = try std.mem.concat(gpa, u8, &.{ sources.deepseek_v41_kv, sources.deepseek_v41_index });
        defer gpa.free(joined);
        const text = try frags.source(device, gpa, joined);
        defer gpa.free(text);
        const lib = try mtl.Library.fromSource(device, text, options);
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.score = try mtl.Pipeline.init(device, lib, "dsv41_index_score", false);
        errdefer p.score.deinit();
        p.clear = try mtl.Pipeline.init(device, lib, "dsv41_index_clear", false);
        errdefer p.clear.deinit();
        p.merge = try mtl.Pipeline.init(device, lib, "dsv41_index_merge", false);
        errdefer p.merge.deinit();
        p.keys_dense = try mtl.Pipeline.init(device, lib, "dsv41_index_keys_dense", false);
        errdefer p.keys_dense.deinit();
        p.seed = try mtl.Pipeline.init(device, lib, "dsv41_index_seed", false);
        errdefer p.seed.deinit();
        p.filter = try mtl.Pipeline.init(device, lib, "dsv41_index_filter", false);
        errdefer p.filter.deinit();
        p.insert = try mtl.Pipeline.init(device, lib, "dsv41_index_insert", false);
        errdefer p.insert.deinit();
        p.finish = try mtl.Pipeline.init(device, lib, "dsv41_index_finish", false);
        p.finish_par = try mtl.Pipeline.init(device, lib, "dsv41_index_finish_par", false);
        errdefer p.finish_par.deinit();
        p.select_one = try mtl.Pipeline.init(device, lib, "dsv41_index_select", false);
        p.select_one_v = try mtl.Pipeline.init(device, lib, "dsv41_index_select_v", false);
        errdefer p.select_one_v.deinit();
        p.identity = try mtl.Pipeline.init(device, lib, "dsv41_index_identity", false);
        p.nax_score = mtl.Pipeline.init(device, lib, "dsv41_index_scores_nax", false) catch null;
        p.row_score = mtl.Pipeline.init(device, lib, "dsv41_index_scores_row", false) catch null;
        p.keys_score = mtl.Pipeline.init(device, lib, "dsv41_index_scores_keys", false) catch null;
        std.log.info("index: decode-shaped scorer {s}", .{if (p.row_score == null) "UNAVAILABLE" else "loaded"});
        return p;
    }
    pub fn deinit(p: *Pipelines) void {
        p.score.deinit();
        p.clear.deinit();
        p.merge.deinit();
        p.keys_dense.deinit();
        p.seed.deinit();
        p.filter.deinit();
        p.insert.deinit();
        p.finish.deinit();
        p.finish_par.deinit();
        p.select_one_v.deinit();
        p.identity.deinit();
        if (p.nax_score) |pf| pf.deinit();
        if (p.row_score) |pf| pf.deinit();
        if (p.keys_score) |pf| pf.deinit();
    }
    /// Q has already undergone BF16 RoPE and in-place FP4/E8M0 quantization; weights are BF16 projection output.
    pub fn encode(p: *const Pipelines, enc: mtl.ComputeEncoder, q: View, weights: View, keys: View, scratch: Scratch, args: Params) !void {
        try args.validate();
        const warps: u32 = warpsPerGroup();
        const keys_per_group: u32 = warps;
        if (args.rows > scratch.rows) return error.IndexerScratchExceeded;
        try (View{ .buffer = scratch.buffer }).require(Scratch.bytes(scratch.rows), 4);
        if (args.keys != 0) {
            try q.requireRows(args.rows, args.q_stride, @as(usize, args.heads) * args.width, 2);
            try weights.requireRows(args.rows, args.weight_stride, args.heads, 2);
            try keys.requireRows(args.keys, args.key_stride, args.width / 2 + args.width / 32, 1);
        }
        // ROUND-411: the block's removal mask (the caller keeps the first index-source layer's picks, so
        // the downstream attention reads the same keys and the delta is the removed dispatch's own cost).
        const mask: u32 = args.skip_mask;
        const count = if (args.use_candidates != 0) args.candidate_topk * args.block_size else args.keys;
        // Measurement only (TF_DSV41_DOUBLE_SCORE / _MERGE): a second identical pass prices the dispatch in
        // place. Both are idempotent in content, so the selection - and the canon reply - is unchanged.
        const score_passes: u32 = if (mask & skip_score != 0) 0 else if (std.c.getenv("TF_DSV41_DOUBLE_SCORE") != null) 2 else 1;
        const merge_passes: u32 = if (mask & skip_merge != 0) 0 else if (std.c.getenv("TF_DSV41_DOUBLE_MERGE") != null) 2 else 1;
        // The tensor-unit tile needs the dense key table to stage and the plain (non-candidate) slot map.
        const nax_ok = p.nax_score != null and args.use_candidates == 0 and args.width == 128 and args.heads == 32 and
            args.keys != 0 and args.ratio != 0 and std.c.getenv("TF_DSV41_NO_INDEX_NAX") == null;
        // ROUND-470: the decode-shaped score (keys on M, heads on N). It needs the dense key table and the
        // plain slot map; TF_DSV41_INDEX_ROW_SCORE=0 is the same-binary control.
        const row_env = std.c.getenv("TF_DSV41_INDEX_ROW_SCORE");
        const row_ok = p.row_score != null and (row_env == null or
            !std.mem.eql(u8, std.mem.span(row_env.?), "0")) and args.use_candidates == 0 and
            args.width == 128 and args.heads == 32 and args.keys != 0 and args.ratio != 0;
        // ROUND-481: the keys-on-M tile. Same inputs and slot map as row_score; TF_DSV41_INDEX_KEYS_SCORE=0
        // is the same-binary control. Only the one-dispatch decode path (rows <= 8) reaches it.
        const keys_env = std.c.getenv("TF_DSV41_INDEX_KEYS_SCORE");
        const keys_ok = p.keys_score != null and (keys_env == null or
            !std.mem.eql(u8, std.mem.span(keys_env.?), "0")) and args.use_candidates == 0 and
            args.width == 128 and args.heads == 32 and args.keys != 0 and args.ratio != 0 and args.rows <= 8;
        const dense_keys = (std.c.getenv("TF_DSV41_INDEX_KEYS") != null or nax_ok or row_ok or keys_ok) and args.keys != 0;
        if (dense_keys and mask & skip_keys == 0) {
            enc.setPipeline(p.keys_dense);
            for ([_]View{ keys, scratch.denseKeys() }, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 2);
            enc.dispatchThreads(mtl.Size.of(@as(usize, args.keys) * (args.width / 2), 1, 1), mtl.Size.of(256, 1, 1));
            enc.barrier();
            if (std.c.getenv("TF_DSV41_DOUBLE_KEYS") != null) {
                enc.setPipeline(p.keys_dense);
                for ([_]View{ keys, scratch.denseKeys() }, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
                enc.setValue(args, 2);
                enc.dispatchThreads(mtl.Size.of(@as(usize, args.keys) * (args.width / 2), 1, 1), mtl.Size.of(256, 1, 1));
                enc.barrier();
            }
        }
        // ROUND-469: at decode rows the heap path is 27 dispatches and ~1,000 one-thread pushes a source. Score
        // every tile into the tile-major layout (the scratch's score region is rows * 16 KB, only 16 KB live at
        // rows = 1) and let one threadgroup a row do the whole selection. TF_DSV41_INDEX_SELECT_ONE=0 is the
        // same-binary control; the removal arms (mask != 0) and the doubling arms stay on the heap path.
        const one_env = std.c.getenv("TF_DSV41_INDEX_SELECT_ONE");
        // ROUND 515: the one-dispatch select was decode-only (rows <= 8). The heap branch pays three
        // merge dispatches a score tile -- 172 tiles at a 21.9k context -- and that is ~10 ms a source
        // layer for a small chunk, where the one-shot path ranks the same scores (the same tile-major
        // array, the same per-row visibility) with one radix select a row. The row bound the buffer
        // actually imposes is the capacity check below, so it, not an arbitrary 8 -- governs; the env
        // restores the old limit for a same-binary A/B and TF_DSV41_INDEX_SELECT_ONE=0 the heap path.
        const one_rows: u32 = blk: {
            const raw = std.c.getenv("TF_DSV41_INDEX_SELECT_ONE_ROWS") orelse break :blk 16384;
            break :blk std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 8;
        };
        const one_shot = (one_env == null or !std.mem.eql(u8, std.mem.span(one_env.?), "0")) and
            args.rows <= one_rows and args.use_candidates == 0 and args.make_candidates == 0 and
            mask == 0 and score_passes == 1 and count != 0 and
            @as(u64, (count + score_tile_width - 1) / score_tile_width) * args.rows <= scratch.rows;
        if (one_shot) {
            // ROUND-542: at one row the tile-major score layout is the key-major one, so the three
            // 4,096-key launches are one -- the same kernel, the same 64-key threadgroups, tile_start 0,
            // the grid over every key. TF_DSV41_INDEX_SCORE_ONE=0 is the same-binary control.
            const score_one_env = std.c.getenv("TF_DSV41_INDEX_SCORE_ONE");
            const score_one = args.rows == 1 and keys_ok and
                (score_one_env == null or !std.mem.eql(u8, std.mem.span(score_one_env.?), "0"));
            if (score_one) {
                var tile = args;
                tile.tile_start = 0;
                tile.score_layout = 1;
                enc.setPipeline(p.keys_score.?);
                for ([_]View{ q, weights, scratch.denseKeys(), scratch.scores() }, 0..) |v, i|
                    enc.setBuffer(v.buffer, v.offset, i);
                enc.setValue(tile, 4);
                enc.dispatchGroups(mtl.Size.of((count + 63) / 64, 1, 1), mtl.Size.of(128, 1, 1));
                enc.barrier();
            } else {
                var shot: u32 = 0;
                while (shot < count) : (shot += score_tile_width) {
                    var tile = args;
                    tile.tile_start = shot;
                    tile.score_layout = 1;
                    const keys_here: u32 = @min(score_tile_width, count - shot);
                    if (keys_ok) {
                        enc.setPipeline(p.keys_score.?);
                        for ([_]View{ q, weights, scratch.denseKeys(), scratch.scores() }, 0..) |v, i|
                            enc.setBuffer(v.buffer, v.offset, i);
                        enc.setValue(tile, 4);
                        enc.dispatchGroups(mtl.Size.of((keys_here + 63) / 64, 1, 1), mtl.Size.of(128, 1, 1));
                        enc.barrier();
                    } else if (row_ok) {
                        enc.setPipeline(p.row_score.?);
                        for ([_]View{ q, weights, scratch.denseKeys(), scratch.scores() }, 0..) |v, i|
                            enc.setBuffer(v.buffer, v.offset, i);
                        enc.setValue(tile, 4);
                        enc.dispatchGroups(mtl.Size.of((keys_here + 63) / 64, (args.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
                        enc.barrier();
                    } else if (nax_ok) {
                        enc.setPipeline(p.nax_score.?);
                        for ([_]View{ q, weights, scratch.denseKeys(), scratch.scores() }, 0..) |v, i|
                            enc.setBuffer(v.buffer, v.offset, i);
                        enc.setValue(tile, 4);
                        enc.dispatchGroups(mtl.Size.of((keys_here + 63) / 64, (args.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
                        enc.barrier();
                    } else {
                        enc.setPipeline(p.score);
                        for ([_]View{ q, weights, keys, scratch.candidates(), scratch.scores() }, 0..) |v, i|
                            enc.setBuffer(v.buffer, v.offset, i);
                        enc.setValue(tile, 5);
                        enc.dispatchGroups(mtl.Size.of(score_tile_width / keys_per_group, args.rows, 1), mtl.Size.of(32 * warps, 1, 1));
                        enc.barrier();
                    }
                }
            }
            var one = args;
            one.score_layout = 1;
            const v_env = std.c.getenv("TF_DSV41_INDEX_SELECT_V");
            const fast_scan = v_env == null or !std.mem.eql(u8, std.mem.span(v_env.?), "0");
            enc.setPipeline(if (fast_scan) p.select_one_v else p.select_one);
            for ([_]View{ scratch.scores(), scratch.positions() }, 0..) |v, i|
                enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(one, 2);
            enc.dispatchGroups(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(256, 1, 1));
            enc.barrier();
            return;
        }
        // ROUND-542: the heap/candidate clear is paid only by the heap path, which reads those regions.
        // The one-shot select ranks the score array and writes positions; it never touches heap or
        // candidates, so the branch below returns before this point and the dispatch is not launched.
        if (mask & skip_clear == 0) {
            enc.setPipeline(p.clear);
            for ([_]View{ scratch.heap(), scratch.candidates() }, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 2);
            enc.dispatchThreads(mtl.Size.of(@max(args.topk, args.candidate_topk), args.rows, 1), mtl.Size.of(128, 1, 1));
            enc.barrier();
        }
        // ROUND-410: the indexer OTHER two dispatches, priced (the clear and the dense-key table). The
        // selection whole-block duplicate (TF_DSV41_DOUBLE_INDEX) reads far more than the score and the
        // merge, so the rest of the block needed names.
        if (mask & skip_clear == 0 and std.c.getenv("TF_DSV41_DOUBLE_CLEAR") != null) {
            enc.setPipeline(p.clear);
            for ([_]View{ scratch.heap(), scratch.candidates() }, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 2);
            enc.dispatchThreads(mtl.Size.of(@max(args.topk, args.candidate_topk), args.rows, 1), mtl.Size.of(128, 1, 1));
            enc.barrier();
        }
        var at: u32 = 0;
        while (at < count) : (at += score_tile_width) {
            var tile = args;
            tile.tile_start = at;
            const keys_here: u32 = @min(score_tile_width, count - at);
            for (0..score_passes) |_| {
                if (nax_ok) {
                    enc.setPipeline(p.nax_score.?);
                    for ([_]View{ q, weights, scratch.denseKeys(), scratch.scores() }, 0..) |v, i|
                        enc.setBuffer(v.buffer, v.offset, i);
                    enc.setValue(tile, 4);
                    enc.dispatchGroups(mtl.Size.of((keys_here + 63) / 64, (args.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
                    enc.barrier();
                    continue;
                }
                enc.setPipeline(p.score);
                for ([_]View{ q, weights, keys, scratch.candidates(), scratch.scores() }, 0..) |v, i|
                    enc.setBuffer(v.buffer, v.offset, i);
                enc.setValue(tile, 5);
                enc.dispatchGroups(mtl.Size.of(score_tile_width / keys_per_group, args.rows, 1), mtl.Size.of(32 * warps, 1, 1));
                enc.barrier();
            }
            for (0..merge_passes) |_| {
                if (std.c.getenv("TF_DSV41_SKIP_MERGE") != null) continue; // measurement: price this path
                // More keys than one tile is where the filter earns its atomics; a single tile fills an
                // empty heap, so every key qualifies and the parallel path only adds cost (measured).
                const parallel_merge = (args.keys > score_tile_width or
                    std.c.getenv("TF_DSV41_INDEX_PARALLEL_MERGE") != null) and
                    std.c.getenv("TF_DSV41_INDEX_SERIAL_MERGE") == null;
                if (!parallel_merge) {
                    enc.setPipeline(p.merge);
                    for ([_]View{ scratch.scores(), scratch.heap(), scratch.candidates() }, 0..) |v, i|
                        enc.setBuffer(v.buffer, v.offset, i);
                    enc.setValue(tile, 3);
                    enc.dispatchThreads(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(1, 1, 1));
                    enc.barrier();
                } else {
                    enc.setPipeline(p.seed);
                    for ([_]View{ scratch.scores(), scratch.heap(), scratch.candidates(), scratch.seedEnd() }, 0..) |v, i|
                        enc.setBuffer(v.buffer, v.offset, i);
                    enc.setValue(tile, 4);
                    enc.dispatchThreads(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(1, 1, 1));
                    enc.barrier();
                    enc.setPipeline(p.filter);
                    for ([_]View{ scratch.scores(), scratch.heap(), scratch.candidates(), scratch.picks(), scratch.seedEnd(), scratch.count() }, 0..) |v, i|
                        enc.setBuffer(v.buffer, v.offset, i);
                    enc.setValue(tile, 6);
                    enc.dispatchGroups(mtl.Size.of(score_tile_width / 256, args.rows, 1), mtl.Size.of(256, 1, 1));
                    enc.barrier();
                    enc.setPipeline(p.insert);
                    for ([_]View{ scratch.heap(), scratch.picks(), scratch.count(), scratch.candidates(), scratch.scores() }, 0..) |v, i|
                        enc.setBuffer(v.buffer, v.offset, i);
                    enc.setValue(tile, 5);
                    enc.dispatchThreads(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(1, 1, 1));
                    enc.barrier();
                }
            }
        }
        const finish_passes: u32 = if (mask & skip_finish != 0) 0 else if (std.c.getenv("TF_DSV41_DOUBLE_FINISH") != null) 2 else 1;
        // ROUND-411: the parallel finish (a threadgroup a row) -- the same positions array, the same
        // order. It needs the slot map to be the identity (the candidate path maps the slots to
        // positions the bitmap/sort would not know), so it is used only off the candidate path.
        const pf_flag = std.c.getenv("TF_DSV41_INDEX_PARALLEL_FINISH");
        const parallel_finish = !(pf_flag != null and std.mem.startsWith(u8, std.mem.span(pf_flag.?), "0")) and
            args.use_candidates == 0 and args.make_candidates == 0;
        for (0..finish_passes) |_| {
            enc.setPipeline(if (parallel_finish) p.finish_par else p.finish);
            for ([_]View{ scratch.heap(), scratch.candidates(), scratch.positions() }, 0..) |v, i|
                enc.setBuffer(v.buffer, v.offset, i);
            enc.setValue(args, 3);
            if (parallel_finish)
                enc.dispatchGroups(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(256, 1, 1))
            else
                enc.dispatchThreads(mtl.Size.of(args.rows, 1, 1), mtl.Size.of(1, 1, 1));
            enc.barrier();
        }
    }
    /// The picks when a row keeps every visible key: positions are the identity, -1 past the row's visible keys.
    pub fn encodeIdentity(p: *const Pipelines, enc: mtl.ComputeEncoder, scratch: Scratch, args: Params) void {
        enc.setPipeline(p.identity);
        enc.setBuffer(scratch.positions().buffer, scratch.positions().offset, 0);
        enc.setValue(args, 1);
        enc.dispatchThreads(mtl.Size.of(args.topk, args.rows, 1), mtl.Size.of(128, 1, 1));
    }
};
