//! Metal launches for Hy4-flash's absorbed MLA projections.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const weights = @import("mla_weights.zig");
const attention = @import("mla_attention.zig");
const rotary = @import("rope.zig");

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
const Args = extern struct { rows: u32, heads: u32 };

/// The BF16 absorb and expand-and-gate products on the tensor units, the same tiles at every row count.
pub const Nax = struct {
    absorb: mtl.Pipeline,
    expand: mtl.Pipeline,
    attend_split: mtl.Pipeline,
    combine: mtl.Pipeline,
    attend: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Nax {
        const text = try @import("lane_projection").frags.source(device, a, sources.hy4_flash_mla_nax);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var absorb = try mtl.Pipeline.init(device, lib, "tf_hy4_absorb_nax", false);
        errdefer absorb.deinit();
        var expand = try mtl.Pipeline.init(device, lib, "tf_hy4_expand_gated_nax", false);
        errdefer expand.deinit();
        var attend_split = try mtl.Pipeline.init(device, lib, "tf_hy4_attend_nax_split", false);
        errdefer attend_split.deinit();
        var combine = try mtl.Pipeline.init(device, lib, "tf_hy4_attend_combine", false);
        errdefer combine.deinit();
        return .{ .absorb = absorb, .expand = expand, .attend_split = attend_split, .combine = combine, .attend = try mtl.Pipeline.init(device, lib, "tf_hy4_attend_nax", false) };
    }

    pub fn deinit(self: Nax) void {
        self.absorb.deinit();
        self.expand.deinit();
        self.attend_split.deinit();
        self.combine.deinit();
        self.attend.deinit();
    }

    /// Flash attention over contiguous key ranges: out [rows][heads][512] from the absorbed queries and the cache.
    fn splits(shape: attention.Shape, has_parts: bool, live_keys: u32) u32 {
        _ = .{ shape, has_parts, live_keys };
        return 1; // row-invariant bits: a split count from the pass's widest row would change other rows' sums
    }

    fn flash(self: Nax, e: mtl.ComputeEncoder, b: Attention.Buffers, ranges: KeyRanges, shape: attention.Shape) void {
        const n = splits(shape, b.flash_parts != null, b.live_keys);
        e.setPipeline(if (n > 1) self.attend_split else self.attend);
        for ([_]Ref{ b.query, b.cache, ranges.ranges, ranges.tiles, b.sinks, b.output }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        const args = extern struct { rows: u32, heads: u32, scale: f32, pad: u32 }{ .rows = shape.rows, .heads = shape.heads, .scale = attention.score_scale, .pad = n };
        e.setValue(args, 6);
        if (n == 1) return e.dispatchGroups(mtl.Size.of(shape.heads, (shape.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        const parts = b.flash_parts.?;
        const ml: Ref = .{ .buffer = parts.buffer, .offset = parts.offset + @as(usize, max_splits) * 16 * shape.heads * weights.latent * 4 };
        e.setBuffer(parts.buffer, parts.offset, 7);
        e.setBuffer(ml.buffer, ml.offset, 8);
        e.dispatchGroups(mtl.Size.of(shape.heads, 1, n), mtl.Size.of(128, 1, 1));
        e.barrier();
        e.setPipeline(self.combine);
        for ([_]Ref{ parts, ml, b.sinks, b.output }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(args, 4);
        e.dispatchGroups(mtl.Size.of(shape.heads, shape.rows, 1), mtl.Size.of(128, 1, 1));
    }

    pub const max_splits = 8;

    /// Bytes of the split partials for `heads` heads (16 rows, max_splits runs, O then max and sum).
    pub fn partBytes(heads: u32) usize {
        return @as(usize, max_splits) * 16 * heads * (weights.latent + 2) * 4;
    }

    fn run(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, refs: []const Ref, rows: u32, heads: u32, col_blocks: u32) void {
        e.setPipeline(pipe);
        for (refs[0..3], 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(Args{ .rows = rows, .heads = heads }, 3);
        if (refs.len > 3) e.setBuffer(refs[3].buffer, refs[3].offset, 4);
        e.dispatchGroups(mtl.Size.of(col_blocks, heads, (rows + 15) / 16), mtl.Size.of(128, 1, 1));
    }
};

/// Each row's keys as one range of key indices [lo, hi) (u32 pairs), and each 32-key tile's first slot
pub const KeyRanges = struct { ranges: Ref, tiles: Ref };

pub const Projections = struct {
    absorb_pipeline: mtl.Pipeline,
    expand_pipeline: mtl.Pipeline,
    table_pipeline: mtl.Pipeline,
    rotate_pipeline: mtl.Pipeline,
    cache_pipeline: mtl.Pipeline,
    gate_pipeline: mtl.Pipeline,
    /// expand then gate in one launch (bit-equal to the two passes).
    expand_gated: mtl.Pipeline,
    /// On tensor units: the absorb and expand-and-gate products there instead.
    nax: ?Nax = null,

    pub fn load(device: mtl.Device) !Projections {
        const library = try mtl.Library.fromSource(device, sources.hy4_flash_mla_project, mtl.CompileOptions.mlx());
        defer library.deinit();
        var absorb_pipeline = try mtl.Pipeline.init(device, library, "tf_hy4_absorb_q", false);
        errdefer absorb_pipeline.deinit();
        var expand_pipeline = try mtl.Pipeline.init(device, library, "tf_hy4_expand_v", false);
        errdefer expand_pipeline.deinit();
        var table_pipeline = try mtl.Pipeline.init(device, library, "tf_hy4_rotary_table", false);
        errdefer table_pipeline.deinit();
        var rotate_pipeline = try mtl.Pipeline.init(device, library, "tf_hy4_rotate", false);
        errdefer rotate_pipeline.deinit();
        var cache_pipeline = try mtl.Pipeline.init(device, library, "tf_hy4_cache_keys", false);
        errdefer cache_pipeline.deinit();
        var expand_gated = try mtl.Pipeline.init(device, library, "tf_hy4_expand_v_gated", false);
        errdefer expand_gated.deinit();
        return .{
            .expand_gated = expand_gated,
            .absorb_pipeline = absorb_pipeline,
            .expand_pipeline = expand_pipeline,
            .table_pipeline = table_pipeline,
            .rotate_pipeline = rotate_pipeline,
            .cache_pipeline = cache_pipeline,
            .gate_pipeline = try mtl.Pipeline.init(device, library, "tf_hy4_gate_values", false),
        };
    }

    pub fn deinit(self: *Projections) void {
        self.absorb_pipeline.deinit();
        self.expand_pipeline.deinit();
        self.table_pipeline.deinit();
        self.rotate_pipeline.deinit();
        self.cache_pipeline.deinit();
        self.gate_pipeline.deinit();
        self.expand_gated.deinit();
        if (self.nax) |n| n.deinit();
        self.* = undefined;
    }

    fn encode(pipeline: mtl.Pipeline, encoder: mtl.ComputeEncoder, refs: [3]Ref, rows: u32, heads: u32, width: u32) !void {
        if (rows == 0 or heads == 0) return;
        const row_heads = try std.math.mul(u32, rows, heads);
        encoder.setPipeline(pipeline);
        for (refs, 0..) |r, i| encoder.setBuffer(r.buffer, r.offset, i);
        encoder.setValue(Args{ .rows = rows, .heads = heads }, refs.len);
        encoder.dispatchThreads(mtl.Size.of(width, row_heads, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn absorb(self: Projections, encoder: mtl.ComputeEncoder, query: Ref, wk: Ref, output: Ref, rows: u32, heads: u32) !void {
        try encode(self.absorb_pipeline, encoder, .{ query, wk, output }, rows, heads, weights.latent);
    }

    pub fn expand(self: Projections, encoder: mtl.ComputeEncoder, latent: Ref, wv: Ref, output: Ref, rows: u32, heads: u32) !void {
        try encode(self.expand_pipeline, encoder, .{ latent, wv, output }, rows, heads, weights.value_width);
    }

    /// Positions [row][3] i32, table [row][32][cos,sin] BF16. Order the table before the rotations that use it.
    pub fn rotaryTable(self: Projections, e: mtl.ComputeEncoder, positions: Ref, table: Ref, rows: u32, p: rotary.Parameters) void {
        if (rows == 0) return;
        e.setPipeline(self.table_pipeline);
        e.setBuffer(positions.buffer, positions.offset, 0);
        e.setBuffer(table.buffer, table.offset, 1);
        e.setValue(p, 2);
        e.dispatchThreads(mtl.Size.of(rotary.pairs, rows, 1), mtl.Size.of(rotary.pairs, 1, 1));
    }

    pub fn rotate(self: Projections, e: mtl.ComputeEncoder, input: Ref, table: Ref, output: Ref, shape: rotary.View) !void {
        try shape.check();
        e.setPipeline(self.rotate_pipeline);
        const buffers = [_]Ref{ input, table, output };
        for (buffers, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, buffers.len);
        e.dispatchThreads(mtl.Size.of(rotary.pairs, shape.heads, shape.rows), mtl.Size.of(rotary.pairs, 1, 1));
    }

    /// Normalize and rotate KV projection rows directly into the lane's assigned latent slots.
    pub fn cacheKeys(self: Projections, e: mtl.ComputeEncoder, input: Ref, gain: Ref, table: Ref, slots: Ref, cache: Ref, shape: attention.CacheWrite) !void {
        try shape.check();
        e.setPipeline(self.cache_pipeline);
        const buffers = [_]Ref{ input, gain, table, slots, cache };
        for (buffers, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, buffers.len);
        e.dispatchGroups(mtl.Size.of(shape.rows, 1, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn gate(self: Projections, e: mtl.ComputeEncoder, values: Ref, logits: Ref, rows: u32, heads: u32) !void {
        if (rows == 0) return;
        if (heads == 0 or heads > 64) return error.BadHy4GateShape;
        const columns = heads * weights.value_width;
        e.setPipeline(self.gate_pipeline);
        e.setBuffer(values.buffer, values.offset, 0);
        e.setBuffer(logits.buffer, logits.offset, 1);
        e.setValue(columns, 2);
        e.dispatchThreads(mtl.Size.of(columns, rows, 1), mtl.Size.of(128, 1, 1));
    }
};

pub const Attention = struct {
    chunks: mtl.Pipeline,
    merge: mtl.Pipeline,
    /// Scores for every key at once, then each chunk's softmax over them (bit-equal to `chunks`).
    scores: mtl.Pipeline,
    scored: mtl.Pipeline,

    pub const Buffers = struct {
        query: Ref,
        cache: Ref,
        slots: Ref,
        counts: Ref,
        sinks: Ref,
        scratch: Ref,
        output: Ref,
        /// [rows][heads][key_stride] floats; null runs the one-step chunks kernel.
        key_scores: ?Ref = null,
        /// Contiguous key ranges: the tensor units' flash kernel instead of the slot lists.
        ranges: ?KeyRanges = null,
        /// The flash kernel's split partials for decode passes (Nax.partBytes); null keeps one threadgroup a head.
        flash_parts: ?Ref = null,
        /// The widest row's live key count (0: unknown, the key stride sizes the split).
        live_keys: u32 = 0,
        /// A pass across 2,048 keys: rows before this one take the flash kernel over `ranges`
        lists_from: ?u32 = null,
    };

    pub fn load(device: mtl.Device) !Attention {
        const library = try mtl.Library.fromSource(device, sources.hy4_flash_mla_attention, mtl.CompileOptions.mlx());
        defer library.deinit();
        var chunks = try mtl.Pipeline.init(device, library, "tf_hy4_attend_chunks", false);
        errdefer chunks.deinit();
        var merge = try mtl.Pipeline.init(device, library, "tf_hy4_attend_merge", false);
        errdefer merge.deinit();
        var scores = try mtl.Pipeline.init(device, library, "tf_hy4_attend_scores", false);
        errdefer scores.deinit();
        return .{ .chunks = chunks, .merge = merge, .scores = scores, .scored = try mtl.Pipeline.init(device, library, "tf_hy4_attend_scored", false) };
    }

    pub fn deinit(self: *Attention) void {
        self.chunks.deinit();
        self.merge.deinit();
        self.scores.deinit();
        self.scored.deinit();
        self.* = undefined;
    }

    /// Caller orders the query/cache writers before this dispatch. Counts and physical slots must be valid.
    pub fn attend(self: Attention, e: mtl.ComputeEncoder, b: Buffers, shape: attention.Shape) !void {
        try shape.check();
        const length = b.scratch.buffer.length();
        if (b.scratch.offset > length or shape.scratchBytes() > length - b.scratch.offset) return error.ShortHy4AttentionScratch;
        if (b.key_scores) |ks| {
            if (ks.buffer.length() < ks.offset + @as(usize, shape.rows) * shape.heads * shape.key_stride * 4) return error.ShortHy4AttentionScratch;
            e.setPipeline(self.scores);
            const first = .{ b.query, b.cache, b.slots, b.counts, ks };
            inline for (first, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
            e.setValue(shape, first.len);
            e.setValue(attention.score_scale, first.len + 1);
            e.dispatchGroups(mtl.Size.of((shape.heads + 3) / 4, (shape.key_stride + 63) / 64, shape.rows), mtl.Size.of(128, 1, 1));
            e.barrier();
            e.setPipeline(self.scored);
            const second = .{ ks, b.cache, b.slots, b.counts, b.scratch };
            inline for (second, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
            e.setValue(shape, second.len);
            e.dispatchGroups(mtl.Size.of((shape.heads + 3) / 4, shape.chunks, shape.rows), mtl.Size.of(128, 1, 1));
        } else {
            e.setPipeline(self.chunks);
            const input = .{ b.query, b.cache, b.slots, b.counts, b.scratch };
            inline for (input, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
            e.setValue(shape, input.len);
            e.setValue(attention.score_scale, input.len + 1);
            e.dispatchGroups(mtl.Size.of((shape.heads + 3) / 4, shape.chunks, shape.rows), mtl.Size.of(128, 1, 1));
        }
        e.barrier();
        e.setPipeline(self.merge);
        const merge_buffers = .{ b.scratch, b.sinks, b.output };
        inline for (merge_buffers, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, merge_buffers.len);
        e.dispatchGroups(mtl.Size.of(shape.heads, shape.rows, 1), mtl.Size.of(128, 1, 1));
    }
};

pub const KvInput = struct {
    projection: Ref,
    gain: Ref,
    write_slots: Ref,
    capacity: u32,
    epsilon: f32,
};

/// Decode passes (at most 4 rows): absorb and expand-and-gate as GEMVs over the checkpoint's affine kv_b codes
pub const MlaPass = struct {
    query: Ref,
    key_weights: Ref,
    value_weights: Ref,
    value_logits: Ref,
    positions: Ref,
    rotary_table: Ref,
    kv: ?KvInput = null,
    attention: Attention.Buffers,
    values: Ref,
};

fn rowsPast(r: Ref, bytes: usize) Ref {
    return .{ .buffer = r.buffer, .offset = r.offset + bytes };
}

/// The affine projections feed this pass. The caller owns the KV leases and retained physical key list.
pub fn forward(project: Projections, attend: Attention, e: mtl.ComputeEncoder, b: MlaPass, shape: attention.Shape, p: rotary.Parameters) !void {
    try shape.check();
    if (b.kv) |kv| try (attention.CacheWrite{ .rows = shape.rows, .capacity = kv.capacity, .epsilon = kv.epsilon }).check();
    project.rotaryTable(e, b.positions, b.rotary_table, shape.rows, p);
    if (project.nax) |n|
        Nax.run(n.absorb, e, &.{ b.query, b.key_weights, b.attention.query }, shape.rows, shape.heads, weights.latent / 128)
    else
        try project.absorb(e, b.query, b.key_weights, b.attention.query, shape.rows, shape.heads);
    e.barrier();
    const input: Ref = .{ .buffer = b.query.buffer, .offset = b.query.offset + weights.nope * 2 };
    const output: Ref = .{
        .buffer = b.attention.query.buffer,
        .offset = b.attention.query.offset + weights.latent * 2,
    };
    try project.rotate(e, input, b.rotary_table, output, .{
        .rows = shape.rows,
        .heads = shape.heads,
        .input_stride = weights.query_stride,
        .output_stride = weights.absorbed_stride,
    });
    if (b.kv) |kv| try project.cacheKeys(e, kv.projection, kv.gain, b.rotary_table, kv.write_slots, b.attention.cache, .{
        .rows = shape.rows,
        .capacity = kv.capacity,
        .epsilon = kv.epsilon,
    });
    e.barrier();
    if (b.attention.lists_from) |k| {
        const n = project.nax orelse return error.Hy4ListSplitNeedsNax;
        const r = b.attention.ranges orelse return error.Hy4ListSplitNeedsRanges;
        if (k == 0 or k >= shape.rows) return error.BadHy4ListSplit;
        n.flash(e, b.attention, r, try attention.Shape.init(k, shape.heads, shape.key_stride));
        var rest = b.attention;
        rest.ranges = null;
        rest.lists_from = null;
        rest.query = rowsPast(b.attention.query, k * shape.heads * weights.absorbed_stride * 2);
        rest.slots = rowsPast(b.attention.slots, k * shape.key_stride * 4);
        rest.counts = rowsPast(b.attention.counts, k * 4);
        rest.output = rowsPast(b.attention.output, k * shape.heads * weights.latent * 2);
        try attend.attend(e, rest, try attention.Shape.init(shape.rows - k, shape.heads, shape.key_stride));
    } else if (project.nax) |n| if (b.attention.ranges) |r| {
        n.flash(e, b.attention, r, shape);
    } else try attend.attend(e, b.attention, shape) else try attend.attend(e, b.attention, shape);
    e.barrier();
    if (shape.heads > 64) return error.BadHy4GateShape;
    if (project.nax) |n| return Nax.run(n.expand, e, &.{ b.attention.output, b.value_weights, b.values, b.value_logits }, shape.rows, shape.heads, weights.value_width / 128);
    e.setPipeline(project.expand_gated);
    for ([_]Ref{ b.attention.output, b.value_weights, b.values }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    e.setValue(Args{ .rows = shape.rows, .heads = shape.heads }, 3);
    e.setBuffer(b.value_logits.buffer, b.value_logits.offset, 4);
    e.dispatchThreads(mtl.Size.of(weights.value_width, try std.math.mul(u32, shape.rows, shape.heads), 1), mtl.Size.of(32, 1, 1));
}
