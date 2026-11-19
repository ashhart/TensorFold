//! DFly2 on Metal. The scheduler supplies the target KV and retained indices for each draft layer.
const std = @import("std");
const mtl = @import("metal");
const source = @import("kernel_sources");
const dense_ops = @import("draft_ops");
const ck = @import("loader.zig");
const model = @import("metal_model.zig");
const common = @import("metal_dense.zig");
const mla = @import("mla_metal.zig");
const weights = @import("model_weights.zig");
const geom = @import("mla_weights.zig");
const rope = @import("rope.zig");

pub const Ref = common.Ref;

const Linear = struct {
    w: mtl.Buffer,
    n: u32,
    k: u32,
    fn upload(b: *common.Buffers, t: @import("core").checkpoint.Tensor) !Linear {
        return .{ .w = try b.upload(t.bytes), .n = @intCast(t.shape[0]), .k = @intCast(t.shape[1]) };
    }
    fn run(p: Linear, ops: dense_ops.Ops, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32) !void {
        try ops.linear(e, dr(x), .{ .buf = p.w }, dr(y), .{
            .rows = rows,
            .n = p.n,
            .k = p.k,
            .x_stride = p.k,
            .w_stride = p.k,
            .y_stride = p.n,
        });
    }
};

fn reference(b: mtl.Buffer) Ref {
    return .{ .buffer = b };
}

fn dr(x: Ref) dense_ops.Ref {
    return .{ .buf = x.buffer, .off = x.offset };
}

fn offset(x: Ref, words: usize) Ref {
    return .{ .buffer = x.buffer, .offset = x.offset + words * @sizeOf(u16) };
}

const Feed = struct { gate: Linear, up: Linear, down: Linear };

pub const Block = struct {
    target_kv: u32,
    target_index: u32,
    heads: u32,
    rope: rope.Parameters,
    norm: mtl.Buffer,
    q_low: Linear,
    q_norm: mtl.Buffer,
    q_high: Linear,
    kv: Linear,
    kv_norm: mtl.Buffer,
    wk: mtl.Buffer,
    wv: mtl.Buffer,
    gate: Linear,
    o: Linear,
    o_norm: mtl.Buffer,
    sinks: mtl.Buffer,
    mlp_norm: mtl.Buffer,
    mlp: Feed,

    fn upload(b: *common.Buffers, host: *ck.Loaded, layer: u32) !Block {
        var weights_ = try weights.DraftLayer.read(b.a, host, layer, .{ .rank = 0, .world = 1 });
        defer weights_.deinit(b.a);
        const t = host.config.value.text_config;
        const heads = weights_.heads.count;
        return .{
            .target_kv = weights_.target_kv,
            .target_index = weights_.target_index,
            .heads = heads,
            .rope = try t.rotary(t.layer_types[weights_.target_kv]),
            .norm = try b.upload(weights_.input_norm.bytes),
            .q_low = try Linear.upload(b, weights_.query_low),
            .q_norm = try b.upload(weights_.query_norm.bytes),
            .q_high = try Linear.upload(b, weights_.query_high),
            .kv = try Linear.upload(b, weights_.kv_low),
            .kv_norm = try b.upload(weights_.kv_norm.bytes),
            .wk = try b.upload(std.mem.sliceAsBytes(weights_.absorbed.key)),
            .wv = try b.upload(std.mem.sliceAsBytes(weights_.absorbed.value)),
            .gate = try Linear.upload(b, weights_.gate),
            .o = try Linear.upload(b, weights_.output),
            .o_norm = try b.upload(weights_.output_norm.bytes),
            .sinks = try b.upload(std.mem.sliceAsBytes(weights_.sinks[0..heads])),
            .mlp_norm = try b.upload(weights_.mlp_norm.bytes),
            .mlp = .{
                .gate = try Linear.upload(b, weights_.mlp.gate),
                .up = try Linear.upload(b, weights_.mlp.up),
                .down = try Linear.upload(b, weights_.mlp.down),
            },
        };
    }
};

pub const Draft = struct {
    own: common.Buffers,
    blocks: []Block,
    width: u32,
    hidden: u32,
    epsilon: f32,
    anchor_norm: mtl.Buffer,
    hidden_norm: mtl.Buffer,
    fusion: Linear,
    fusion_gate: mtl.Buffer,
    final_norm: mtl.Buffer,
    correct_hidden: mtl.Buffer,
    correct_embed: mtl.Buffer,
    correction: Feed,
    fuse: mtl.Pipeline,
    residual: mtl.Pipeline,

    pub fn open(a: std.mem.Allocator, d: mtl.Device, host: *ck.Loaded) !Draft {
        var b: common.Buffers = .{ .a = a, .d = d };
        errdefer b.clear();
        const cfg = host.config.value;
        const t = cfg.text_config;
        const blocks = try a.alloc(Block, cfg.dfly_num_hidden_layers);
        errdefer a.free(blocks);
        for (blocks, 0..) |*layer, i| layer.* = try Block.upload(&b, host, @intCast(i));
        const w = try weights.DraftHead.read(host);
        const lib = try mtl.Library.fromSource(d, source.hy4_flash_ihc, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var fuse = try mtl.Pipeline.init(d, lib, "tf_hy4_draft_fuse", false);
        errdefer fuse.deinit();
        var out: Draft = .{
            .own = undefined,
            .blocks = blocks,
            .width = cfg.speculative_config.num_speculative_tokens + 1,
            .hidden = t.hidden_size,
            .epsilon = t.rms_norm_eps,
            .anchor_norm = try b.upload(w.anchor_norm.bytes),
            .hidden_norm = try b.upload(w.hidden_norm.bytes),
            .fusion = try Linear.upload(&b, w.fusion),
            .fusion_gate = try b.upload(w.fusion_gate.bytes),
            .final_norm = try b.upload(w.norm.bytes),
            .correct_hidden = try b.upload(w.correction_hidden_norm.bytes),
            .correct_embed = try b.upload(w.correction_embedding_norm.bytes),
            .correction = .{
                .gate = try Linear.upload(&b, w.correction.gate),
                .up = try Linear.upload(&b, w.correction.up),
                .down = try Linear.upload(&b, w.correction.down),
            },
            .fuse = fuse,
            .residual = try mtl.Pipeline.init(d, lib, "tf_hy4_draft_residual", false),
        };
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Draft) void {
        self.fuse.deinit();
        self.residual.deinit();
        self.own.a.free(self.blocks);
        self.own.clear();
    }

    fn norm(self: *const Draft, ops: dense_ops.Ops, e: mtl.ComputeEncoder, x: Ref, gain: mtl.Buffer, y: Ref, rows: u32, dim: u32, xs: u32, ys: u32) !void {
        try ops.norm(e, dr(x), .{ .buf = gain }, dr(y), .{
            .rows = rows,
            .heads = 1,
            .dim = dim,
            .x_stride = xs,
            .y_stride = ys,
            .eps = self.epsilon,
        });
    }

    fn add(self: *const Draft, e: mtl.ComputeEncoder, x: Ref, y: Ref, out: Ref, rows: u32, stride: u32) void {
        e.setPipeline(self.residual);
        for ([_]Ref{ x, y, out }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue([2]u32{ self.hidden, stride }, 3);
        e.dispatchThreads(mtl.Size.of(self.hidden, rows, 1), mtl.Size.of(128, 1, 1));
    }

    fn mlp(ops: dense_ops.Ops, e: mtl.ComputeEncoder, w: Feed, temp: *const common.Scratch, x: Ref, y: Ref, rows: u32) !void {
        const n = w.gate.n;
        if (n > temp.inner_capacity) return error.ShortHy4Scratch;
        try w.gate.run(ops, e, x, temp.gate, rows);
        try w.up.run(ops, e, x, temp.up, rows);
        e.barrier();
        try ops.swiglu(e, dr(temp.gate), dr(temp.up), dr(temp.inner), .{
            .rows = rows,
            .width = n,
            .gate_stride = n,
            .up_stride = n,
            .y_stride = n,
        });
        e.barrier();
        try w.down.run(ops, e, temp.inner, y, rows);
    }

    pub fn encode(self: *const Draft, target: *const model.Model, e: mtl.ComputeEncoder, s: *Frame, ids: Ref, anchors: Ref, lanes: u32, caches: []const model.LayerCache) !void {
        s.lanes = 0;
        if (lanes == 0 or lanes > s.capacity or caches.len != self.blocks.len) return error.BadHy4DraftFrame;
        const begin = s.streams[0];
        const rows = lanes * self.width;
        const h = self.hidden;
        const ops = target.common.dense;
        target.embedRows(e, ids, begin, rows);
        try self.norm(ops, e, anchors, self.hidden_norm, offset(s.concat, h), lanes, h, h, 2 * h);
        e.barrier();
        try self.norm(ops, e, begin, self.anchor_norm, s.concat, lanes, h, self.width * h, 2 * h);
        e.barrier();
        try self.fusion.run(ops, e, s.concat, s.delta, lanes);
        e.barrier();
        e.setPipeline(self.fuse);
        e.setBuffer(begin.buffer, begin.offset, 0);
        e.setBuffer(s.delta.buffer, s.delta.offset, 1);
        e.setBuffer(self.fusion_gate, 0, 2);
        e.setValue([2]u32{ h, self.width }, 3);
        e.dispatchThreads(mtl.Size.of(h, lanes, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        var x = begin;
        var y = s.streams[1];
        for (self.blocks, caches) |block, cache| {
            try s.common.bind(rows, block.heads, cache.key_stride);
            try self.encodeBlock(target, e, block, s, cache.rows, x, y);
            e.barrier();
            std.mem.swap(Ref, &x, &y);
        }
        try self.norm(ops, e, x, self.final_norm, s.normalized, rows, h, h, h);
        s.lanes = lanes;
    }

    fn encodeBlock(self: *const Draft, target: *const model.Model, e: mtl.ComputeEncoder, w: Block, s: *const Frame, cache: common.CacheRows, x: Ref, y: Ref) !void {
        const ops = target.common.dense;
        const v = &s.common;
        const rows = v.shape.rows;
        const h = self.hidden;
        try self.norm(ops, e, x, w.norm, v.normalized, rows, h, h, h);
        e.barrier();
        try w.q_low.run(ops, e, v.normalized, v.query_a, rows);
        try w.kv.run(ops, e, v.normalized, v.kv, rows);
        try w.gate.run(ops, e, v.normalized, v.value_logits, rows);
        e.barrier();
        try self.norm(ops, e, v.query_a, w.q_norm, v.query_norm, rows, w.q_low.n, w.q_low.n, w.q_low.n);
        e.barrier();
        try w.q_high.run(ops, e, v.query_norm, v.query, rows);
        e.barrier();
        try mla.forward(target.common.projections, target.common.attention, e, .{
            .query = v.query,
            .key_weights = reference(w.wk),
            .value_weights = reference(w.wv),
            .value_logits = v.value_logits,
            .positions = cache.positions,
            .rotary_table = v.rotary,
            .values = v.values,
            .kv = .{ .projection = v.kv, .gain = reference(w.kv_norm), .write_slots = cache.write_slots, .capacity = cache.capacity, .epsilon = self.epsilon },
            .attention = .{
                .query = v.absorbed,
                .cache = cache.cache,
                .slots = cache.retained,
                .counts = cache.counts,
                .sinks = reference(w.sinks),
                .scratch = v.attention,
                .output = v.latent,
            },
        }, v.shape, w.rope);
        e.barrier();
        try w.o.run(ops, e, v.values, v.branch, rows);
        e.barrier();
        try self.norm(ops, e, v.branch, w.o_norm, v.normalized, rows, h, h, h);
        e.barrier();
        self.add(e, x, v.normalized, s.after_attention, rows, h);
        e.barrier();
        try self.norm(ops, e, s.after_attention, w.mlp_norm, v.normalized, rows, h, h, h);
        e.barrier();
        try mlp(ops, e, w.mlp, v, v.normalized, v.branch, rows);
        e.barrier();
        self.add(e, s.after_attention, v.branch, y, rows, h);
    }

    /// Apply the hidden correction at the chosen draft column with the previous proposal's token.
    pub fn next(self: *const Draft, target: *const model.Model, e: mtl.ComputeEncoder, s: *const Frame, previous: Ref, column: u32) !Ref {
        if (column == 0 or column >= self.width or s.lanes == 0) return error.BadHy4DraftColumn;
        const ops = target.common.dense;
        const rows = s.lanes;
        const h = self.hidden;
        const draft = offset(s.normalized, @as(usize, column) * h);
        target.embedRows(e, previous, s.previous, rows);
        try self.norm(ops, e, draft, self.correct_hidden, s.concat, rows, h, h * self.width, 2 * h);
        e.barrier();
        try self.norm(ops, e, s.previous, self.correct_embed, offset(s.concat, h), rows, h, h, 2 * h);
        e.barrier();
        try mlp(ops, e, self.correction, &s.common, s.concat, s.delta, rows);
        e.barrier();
        self.add(e, draft, s.delta, s.previous, rows, h * self.width);
        e.barrier();
        try target.readout(e, s.previous, s.logits, rows);
        return s.logits;
    }
};

pub const Frame = struct {
    own: common.Buffers,
    capacity: u32,
    lanes: u32 = 0,
    common: common.Scratch,
    streams: [2]Ref,
    concat: Ref,
    delta: Ref,
    normalized: Ref,
    after_attention: Ref,
    previous: Ref,
    logits: Ref,

    pub fn init(a: std.mem.Allocator, d: mtl.Device, target: *const model.Model, draft: *const Draft, lanes: u32, keys: u32) !Frame {
        if (lanes == 0 or draft.blocks.len == 0) return error.BadHy4DraftFrame;
        const rows = try std.math.mul(u32, lanes, draft.width);
        const h = draft.hidden;
        const t = target.host.config.value.text_config;
        var max_inner = draft.correction.gate.n;
        var max_heads: u32 = 0;
        var target_layer: u32 = 0;
        for (draft.blocks) |block| {
            max_inner = @max(max_inner, block.mlp.gate.n);
            if (block.heads > max_heads) {
                max_heads = block.heads;
                target_layer = block.target_kv;
            }
        }
        var temp = try common.Scratch.init(a, d, &target.blocks[target_layer].attention, max_inner, keys, rows);
        errdefer temp.deinit();
        var b: common.Buffers = .{ .a = a, .d = d };
        errdefer b.clear();
        var out: Frame = .{
            .own = undefined,
            .capacity = lanes,
            .common = temp,
            .streams = .{ try b.rows(rows, h), try b.rows(rows, h) },
            .concat = try b.rows(lanes, 2 * h),
            .delta = try b.rows(lanes, h),
            .normalized = try b.rows(rows, h),
            .after_attention = try b.rows(rows, h),
            .previous = try b.rows(lanes, h),
            .logits = try b.rows(lanes, t.vocab_size),
        };
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Frame) void {
        self.common.deinit();
        self.own.clear();
    }
};
