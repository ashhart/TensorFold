//! Metal launches for Hy4-flash's absorbed MLA projections.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const weights = @import("mla_weights.zig");
const attention = @import("mla_attention.zig");
const rotary = @import("rope.zig");

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
const Args = extern struct { rows: u32, heads: u32 };

pub const Projections = struct {
    absorb_pipeline: mtl.Pipeline,
    expand_pipeline: mtl.Pipeline,
    table_pipeline: mtl.Pipeline,
    rotate_pipeline: mtl.Pipeline,
    cache_pipeline: mtl.Pipeline,
    gate_pipeline: mtl.Pipeline,

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
        return .{
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

    pub const Buffers = struct {
        query: Ref,
        cache: Ref,
        slots: Ref,
        counts: Ref,
        sinks: Ref,
        scratch: Ref,
        output: Ref,
    };

    pub fn load(device: mtl.Device) !Attention {
        const library = try mtl.Library.fromSource(device, sources.hy4_flash_mla_attention, mtl.CompileOptions.mlx());
        defer library.deinit();
        var chunks = try mtl.Pipeline.init(device, library, "tf_hy4_attend_chunks", false);
        errdefer chunks.deinit();
        return .{ .chunks = chunks, .merge = try mtl.Pipeline.init(device, library, "tf_hy4_attend_merge", false) };
    }

    pub fn deinit(self: *Attention) void {
        self.chunks.deinit();
        self.merge.deinit();
        self.* = undefined;
    }

    /// Caller orders the query/cache writers before this dispatch. Counts and physical slots must be valid.
    pub fn attend(self: Attention, e: mtl.ComputeEncoder, b: Buffers, shape: attention.Shape) !void {
        try shape.check();
        const length = b.scratch.buffer.length();
        if (b.scratch.offset > length or shape.scratchBytes() > length - b.scratch.offset) return error.ShortHy4AttentionScratch;
        e.setPipeline(self.chunks);
        const input = .{ b.query, b.cache, b.slots, b.counts, b.scratch };
        inline for (input, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, input.len);
        e.setValue(attention.score_scale, input.len + 1);
        e.dispatchGroups(mtl.Size.of((shape.heads + 3) / 4, shape.chunks, shape.rows), mtl.Size.of(128, 1, 1));
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

/// The affine projections feed this pass. The caller owns the KV leases and retained physical key list.
pub fn forward(project: Projections, attend: Attention, e: mtl.ComputeEncoder, b: MlaPass, shape: attention.Shape, p: rotary.Parameters) !void {
    try shape.check();
    if (b.kv) |kv| try (attention.CacheWrite{ .rows = shape.rows, .capacity = kv.capacity, .epsilon = kv.epsilon }).check();
    project.rotaryTable(e, b.positions, b.rotary_table, shape.rows, p);
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
    try attend.attend(e, b.attention, shape);
    e.barrier();
    try project.expand(e, b.attention.output, b.value_weights, b.values, shape.rows, shape.heads);
    e.barrier();
    try project.gate(e, b.values, b.value_logits, shape.rows, shape.heads);
}
