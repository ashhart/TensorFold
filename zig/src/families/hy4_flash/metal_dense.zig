//! The native Metal attention and dense feed-forward layer passes for Hy4-flash.
const std = @import("std");
const mtl = @import("metal");
const affine = @import("row_projection");
const dense = @import("draft_ops");
const core = @import("core");
const cfg = @import("config.zig");
const ihc = @import("ihc.zig");
const ihc_gpu = @import("ihc_metal.zig");
const mla_gpu = @import("mla_metal.zig");
const mha = @import("mla_attention.zig");
const shape_ = @import("mla_weights.zig");
const weights = @import("model_weights.zig");
const rope = @import("rope.zig");
const gpu_qkv = @import("metal_tensor_qkv.zig");
const lane = @import("lane_projection");

pub const Ref = mla_gpu.Ref;

pub const Buffers = struct {
    a: std.mem.Allocator,
    d: mtl.Device,
    list: std.ArrayList(mtl.Buffer) = .empty,

    pub fn alloc(self: *Buffers, bytes: usize) !mtl.Buffer {
        const b = try self.d.buffer(bytes, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        try self.list.append(self.a, b);
        return b;
    }

    pub fn upload(self: *Buffers, bytes: []const u8) !mtl.Buffer {
        const b = try self.alloc(bytes.len);
        @memcpy(b.slice(u8, bytes.len), bytes);
        return b;
    }

    pub fn clear(self: *Buffers) void {
        for (self.list.items) |b| b.deinit();
        self.list.deinit(self.a);
    }

    pub fn matrix(self: *Buffers, w: core.quant.mlx.Host) !affine.Weights {
        const view: affine.Weights = .{
            .w = try self.upload(w.words.bytes),
            .scales = try self.upload(w.scales.bytes),
            .biases = try self.upload(w.biases.bytes),
            .n = w.words.shape[0],
            .k = w.scales.shape[1] * w.group,
            .bits = w.bits,
            .group = w.group,
        };
        try view.validate();
        return view;
    }

    pub fn rows(self: *Buffers, count: u32, columns: u32) !Ref {
        return .{ .buffer = try self.alloc(@as(usize, count) * columns * @sizeOf(u16)) };
    }
};

pub const Merge = struct {
    project: mtl.Buffer,
    bias: mtl.Buffer,
    scale: mtl.Buffer,
    shape: ihc.Shape,

    pub fn upload(b: *Buffers, w: ihc.Weights) !Merge {
        return .{
            .project = try b.upload(w.project.bytes),
            .bias = try b.upload(w.bias.bytes),
            .scale = try b.upload(w.scale.bytes),
            .shape = w.shape,
        };
    }
};

const InputWeights = union(enum) {
    portable: struct { q_a: affine.Weights, q_b: affine.Weights, kv_a: affine.Weights, gates: affine.Weights },
    tensor: struct { qkv: gpu_qkv.Packed, q_b: gpu_qkv.Packed, flavor: usize },
};

pub const Attention = struct {
    own: Buffers,
    hidden: u32,
    query_rank: u32,
    heads: u32,
    epsilon: f32,
    rope: rope.Parameters,
    merge: Merge,
    norm: mtl.Buffer,
    inputs: InputWeights,
    q_norm: mtl.Buffer,
    kv_norm: mtl.Buffer,
    o: affine.Weights,
    o_norm: mtl.Buffer,
    wk: mtl.Buffer,
    wv: mtl.Buffer,
    sinks: mtl.Buffer,

    pub fn upload(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, t: cfg.Text) !Attention {
        return uploadForModel(a, device, w, t, false);
    }

    pub fn uploadForModel(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, t: cfg.Text, use_tensors: bool) !Attention {
        if (w.attention.owns_output) return error.Hy4MetalAttentionNeedsFullHeadRange;
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const heads = w.attention.heads.count;
        const bits = w.project_query.bits;
        const inputs: InputWeights = if (use_tensors)
            .{ .tensor = .{
                .qkv = try gpu_qkv.Packed.upload(&b, &.{ w.project_query, w.project_kv, w.attention.gate }, gpu_qkv.inputLayout(t, bits, heads)),
                .q_b = try gpu_qkv.Packed.upload(&b, &.{w.attention.query}, gpu_qkv.queryLayout(t, bits, heads)),
                .flavor = gpu_qkv.flavor(t, heads),
            } }
        else
            .{ .portable = .{
                .q_a = try b.matrix(w.project_query),
                .q_b = try b.matrix(w.attention.query),
                .kv_a = try b.matrix(w.project_kv),
                .gates = try b.matrix(w.attention.gate),
            } };
        const out: Attention = .{
            .own = undefined,
            .hidden = t.hidden_size,
            .query_rank = t.q_lora_rank,
            .heads = heads,
            .epsilon = t.rms_norm_eps,
            .rope = w.position,
            .merge = try Merge.upload(&b, w.attention_hc),
            .norm = try b.upload(w.attention_norm.bytes),
            .inputs = inputs,
            .q_norm = try b.upload(w.norm_query.bytes),
            .kv_norm = try b.upload(w.norm_kv.bytes),
            .o = try b.matrix(w.attention.output),
            .o_norm = try b.upload(w.attention_output_norm.bytes),
            .wk = try b.upload(std.mem.sliceAsBytes(w.absorbed.key)),
            .wv = try b.upload(std.mem.sliceAsBytes(w.absorbed.value)),
            .sinks = try b.upload(std.mem.sliceAsBytes(w.sinks[0..heads])),
        };
        return withAllocation(out, b);
    }

    pub fn deinit(self: *Attention) void {
        self.own.clear();
        self.* = undefined;
    }
};

pub const Feed = struct {
    own: Buffers,
    merge: Merge,
    norm: mtl.Buffer,
    gate: affine.Weights,
    up: affine.Weights,
    down: affine.Weights,
    output_norm: mtl.Buffer,

    pub fn upload(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer) !Feed {
        if (w.feed != .dense) return error.Hy4DenseFeedRequired;
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const feed = w.feed.dense;
        const out: Feed = .{
            .own = undefined,
            .merge = try Merge.upload(&b, w.mlp_hc),
            .norm = try b.upload(w.mlp_norm.bytes),
            .gate = try b.matrix(feed.projections.gate),
            .up = try b.matrix(feed.projections.up),
            .down = try b.matrix(feed.projections.down),
            .output_norm = try b.upload(feed.norm.bytes),
        };
        return withAllocation(out, b);
    }

    pub fn deinit(self: *Feed) void {
        self.own.clear();
        self.* = undefined;
    }
};

fn withAllocation(out: anytype, b: Buffers) @TypeOf(out) {
    var owned = out;
    owned.own = b;
    return owned;
}

pub const Scratch = struct {
    own: Buffers,
    capacity: mha.Shape,
    shape: mha.Shape,
    inner_capacity: u32,
    merged: Ref,
    qkv: ?Ref = null,
    post: Ref,
    normalized: Ref,
    query_a: Ref,
    query_norm: Ref,
    query: Ref,
    kv: Ref,
    value_logits: Ref,
    values: Ref,
    absorbed: Ref,
    rotary: Ref,
    latent: Ref,
    attention: Ref,
    branch: Ref,
    gate: Ref,
    up: Ref,
    inner: Ref,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, layer: *const Attention, inner: u32, keys: u32, rows: u32) !Scratch {
        return initForModel(a, device, layer, inner, keys, rows, false);
    }

    pub fn initForModel(a: std.mem.Allocator, device: mtl.Device, layer: *const Attention, inner: u32, keys: u32, rows: u32, use_tensors: bool) !Scratch {
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const r = rows;
        const h = layer.hidden;
        const heads = layer.heads;
        const shape = try mha.Shape.init(r, heads, keys);
        const out: Scratch = .{
            .own = undefined,
            .capacity = shape,
            .shape = shape,
            .inner_capacity = inner,
            .merged = try b.rows(r, h),
            .qkv = if (use_tensors) try b.rows(r, layer.query_rank + shape_.absorbed_stride + heads * shape_.value_width) else null,
            .post = .{ .buffer = try b.alloc(@as(usize, r) * ihc.streams * @sizeOf(f32)) },
            .normalized = try b.rows(r, h),
            .query_a = try b.rows(r, layer.query_rank),
            .query_norm = try b.rows(r, layer.query_rank),
            .query = try b.rows(r, heads * shape_.query_stride),
            .kv = try b.rows(r, shape_.absorbed_stride),
            .value_logits = try b.rows(r, heads * shape_.value_width),
            .values = try b.rows(r, heads * shape_.value_width),
            .absorbed = try b.rows(r, heads * shape_.absorbed_stride),
            .rotary = try b.rows(r, shape_.rope),
            .latent = try b.rows(r, heads * shape_.latent),
            .attention = .{ .buffer = try b.alloc(shape.scratchBytes()) },
            .branch = try b.rows(r, h),
            .gate = try b.rows(r, inner),
            .up = try b.rows(r, inner),
            .inner = try b.rows(r, inner),
        };
        return withAllocation(out, b);
    }

    /// Keep one allocation at the widest layer and bind the current lane batch and head count before encoding.
    pub fn bind(self: *Scratch, rows: u32, heads: u32, keys: u32) !void {
        if (rows > self.capacity.rows or heads > self.capacity.heads or keys > self.capacity.key_stride)
            return error.ShortHy4Scratch;
        self.shape = try mha.Shape.init(rows, heads, keys);
    }

    pub fn deinit(self: *Scratch) void {
        self.own.clear();
        self.* = undefined;
    }
};

pub const CacheRows = struct {
    positions: Ref,
    write_slots: Ref,
    retained: Ref,
    counts: Ref,
    cache: Ref,
    capacity: u32,
};

pub const Kernels = struct {
    affine: affine.Pipelines,
    dense: dense.Ops,
    merge: ihc_gpu.Kernels,
    projections: mla_gpu.Projections,
    attention: mla_gpu.Attention,
    tensors: ?gpu_qkv.Kernels = null,

    pub fn load(device: mtl.Device) !Kernels {
        var row_project = try affine.Pipelines.load(device);
        errdefer row_project.deinit();
        const normalize = try dense.Ops.init(device);
        errdefer normalize.deinit();
        var merge = try ihc_gpu.Kernels.load(device);
        errdefer merge.deinit();
        var proj = try mla_gpu.Projections.load(device);
        errdefer proj.deinit();
        return .{
            .affine = row_project,
            .dense = normalize,
            .merge = merge,
            .projections = proj,
            .attention = try mla_gpu.Attention.load(device),
        };
    }

    pub fn deinit(self: *Kernels) void {
        if (self.tensors) |t| t.deinit();
        self.attention.deinit();
        self.projections.deinit();
        self.merge.deinit();
        self.dense.deinit();
        self.affine.deinit();
    }

    pub fn loadForModel(a: std.mem.Allocator, device: mtl.Device, t: cfg.Text, bits: u8) !Kernels {
        var out = try load(device);
        errdefer out.deinit();
        if (device.tensorUnits()) out.tensors = try gpu_qkv.Kernels.load(a, device, t, bits);
        return out;
    }

    pub fn project(self: *const Kernels, e: mtl.ComputeEncoder, w: affine.Weights, x: Ref, y: Ref, rows: u32) !void {
        const call = try affine.call(&self.affine, w, rows, false);
        e.setPipeline(call.pipeline);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(w.w, w.w_off, 1);
        e.setBuffer(w.scales, w.s_off, 2);
        e.setBuffer(w.biases, w.b_off, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(y.buffer, y.offset, 5);
        e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
    }

    pub fn dref(x: Ref) dense.Ref {
        return .{ .buf = x.buffer, .off = x.offset };
    }

    fn tref(x: Ref) lane.Ref {
        return .{ .buf = x.buffer, .off = x.offset };
    }

    pub fn norm(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, gain: mtl.Buffer, y: Ref, rows: u32, width: u32, epsilon: f32) !void {
        try self.dense.norm(e, dref(x), .{ .buf = gain }, dref(y), .{
            .rows = rows,
            .heads = 1,
            .dim = width,
            .x_stride = width,
            .y_stride = width,
            .eps = epsilon,
        });
    }

    pub fn collapse(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Merge, gain: mtl.Buffer, s: *const Scratch) !ihc.Shape {
        var shape = w.shape;
        shape.rows = s.shape.rows;
        try self.merge.collapse(e, .{
            .hidden = .{ .buffer = x.buffer, .offset = x.offset },
            .projection = .{ .buffer = w.project },
            .bias = .{ .buffer = w.bias },
            .scale = .{ .buffer = w.scale },
            .output = .{ .buffer = s.merged.buffer, .offset = s.merged.offset },
            .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
        }, shape);
        e.barrier();
        try self.norm(e, s.merged, gain, s.normalized, shape.rows, shape.width, shape.eps);
        e.barrier();
        return shape;
    }

    pub fn residual(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, branch: Ref, post: Ref, out: Ref, shape: ihc.Shape) !void {
        try self.merge.add(e, .{
            .residual = .{ .buffer = x.buffer, .offset = x.offset },
            .branch = .{ .buffer = branch.buffer, .offset = branch.offset },
            .post = .{ .buffer = post.buffer, .offset = post.offset },
            .output = .{ .buffer = out.buffer, .offset = out.offset },
        }, shape);
    }

    /// The caller supplies the retained list for this lane from its sliding or sparse index state.
    pub fn attend(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Attention, s: *const Scratch, cache: CacheRows, x: Ref, out: Ref) !void {
        if (s.shape.heads != w.heads) return error.Hy4WrongAttentionHeadCount;
        if (w.inputs == .tensor and (self.tensors == null or s.qkv == null)) return error.ShortHy4Scratch;
        const shape = try self.collapse(e, x, w.merge, w.norm, s);
        const rows = shape.rows;
        var q_a = s.query_a;
        var kv = s.kv;
        var gates = s.value_logits;
        switch (w.inputs) {
            .portable => |m| {
                try self.project(e, m.q_a, s.normalized, q_a, rows);
                try self.project(e, m.kv_a, s.normalized, kv, rows);
                try self.project(e, m.gates, s.normalized, gates, rows);
                e.barrier();
                try self.norm(e, q_a, w.q_norm, s.query_norm, rows, w.query_rank, w.epsilon);
                e.barrier();
                try self.project(e, m.q_b, s.query_norm, s.query, rows);
            },
            .tensor => |m| {
                const kernels = &self.tensors.?;
                const qkv = s.qkv.?;
                try kernels.inputs[m.flavor].encode(e, tref(s.normalized), m.qkv.w, m.qkv.sb, tref(qkv), rows);
                e.barrier();
                if (rows == 1) {
                    q_a = qkv;
                    kv = .{ .buffer = qkv.buffer, .offset = @as(usize, w.query_rank) * @sizeOf(u16) };
                    gates = .{ .buffer = qkv.buffer, .offset = (@as(usize, w.query_rank) + shape_.absorbed_stride) * @sizeOf(u16) };
                } else {
                    kernels.split(e, tref(qkv), tref(q_a), tref(kv), tref(gates), rows, w.query_rank, w.heads * shape_.value_width);
                    e.barrier();
                }
                try self.norm(e, q_a, w.q_norm, s.query_norm, rows, w.query_rank, w.epsilon);
                e.barrier();
                try kernels.queries[m.flavor].encode(e, tref(s.query_norm), m.q_b.w, m.q_b.sb, tref(s.query), rows);
            },
        }
        e.barrier();
        try mla_gpu.forward(self.projections, self.attention, e, .{
            .query = s.query,
            .key_weights = .{ .buffer = w.wk },
            .value_weights = .{ .buffer = w.wv },
            .value_logits = gates,
            .positions = cache.positions,
            .rotary_table = s.rotary,
            .values = s.values,
            .kv = .{ .projection = kv, .gain = .{ .buffer = w.kv_norm }, .write_slots = cache.write_slots, .capacity = cache.capacity, .epsilon = w.epsilon },
            .attention = .{
                .query = s.absorbed,
                .cache = cache.cache,
                .slots = cache.retained,
                .counts = cache.counts,
                .sinks = .{ .buffer = w.sinks },
                .scratch = s.attention,
                .output = s.latent,
            },
        }, s.shape, w.rope);
        e.barrier();
        try self.project(e, w.o, s.values, s.branch, rows);
        e.barrier();
        try self.norm(e, s.branch, w.o_norm, s.normalized, rows, w.hidden, w.epsilon);
        e.barrier();
        try self.residual(e, x, s.normalized, s.post, out, shape);
    }

    pub fn feed(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Feed, s: *const Scratch, x: Ref, out: Ref) !void {
        const inner: u32 = @intCast(w.gate.n);
        if (inner > s.inner_capacity) return error.ShortHy4Scratch;
        const shape = try self.collapse(e, x, w.merge, w.norm, s);
        try self.project(e, w.gate, s.normalized, s.gate, shape.rows);
        try self.project(e, w.up, s.normalized, s.up, shape.rows);
        e.barrier();
        try self.dense.swiglu(e, dref(s.gate), dref(s.up), dref(s.inner), .{
            .rows = shape.rows,
            .width = inner,
            .gate_stride = inner,
            .up_stride = inner,
            .y_stride = inner,
        });
        e.barrier();
        try self.project(e, w.down, s.inner, s.branch, shape.rows);
        e.barrier();
        try self.norm(e, s.branch, w.output_norm, s.normalized, shape.rows, shape.width, shape.eps);
        e.barrier();
        try self.residual(e, x, s.normalized, s.post, out, shape);
    }
};
