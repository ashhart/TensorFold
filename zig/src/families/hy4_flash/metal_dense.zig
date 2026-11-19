//! The native Metal attention and dense feed-forward layer passes for Hy4-flash.
const std = @import("std");
const mtl = @import("metal");
const affine = @import("../../core/row_projection.zig");
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
const lane = @import("../../core/lane_projection.zig");
const index = @import("metal_index.zig");

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

    /// A dense projection's weights: packed for the tensor units' lane projection when `tensors`
    pub fn linear(self: *Buffers, w: core.quant.mlx.Host, tensors: bool) !Linear {
        const n: u32 = @intCast(w.words.shape[0]);
        const k: u32 = @intCast(w.scales.shape[1] * w.group);
        if (tensors and n % 32 == 0 and w.group == 64 and (w.bits == 4 or w.bits == 8)) {
            const packed_ = try gpu_qkv.Packed.upload(self, &.{w}, gpu_qkv.denseLayout(w.bits, n, k));
            return .{ .tensor = .{ .w = packed_.w, .sb = packed_.sb, .n = n, .k = k } };
        }
        return .{ .portable = try self.matrix(w) };
    }

    pub fn rows(self: *Buffers, count: u32, columns: u32) !Ref {
        return .{ .buffer = try self.alloc(@as(usize, count) * columns * @sizeOf(u16)) };
    }
};

/// A dense affine projection [n, k]: the core row kernels, or the tensor units' lane projection over packed weights.
pub const Linear = union(enum) {
    portable: affine.Weights,
    tensor: struct { w: lane.Ref, sb: lane.Ref, n: u32, k: u32 },

    pub fn outputs(self: Linear) u32 {
        return switch (self) {
            .portable => |w| @intCast(w.n),
            .tensor => |t| t.n,
        };
    }
};

pub const Linears = @import("metal_linears.zig").Linears;

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
    o: Linear,
    o_norm: mtl.Buffer,
    wk: mtl.Buffer,
    wv: mtl.Buffer,
    bits: u8 = 4,
    sinks: mtl.Buffer,
    /// A sparse layer's compressed indexer (null on sliding layers).
    index: ?index.Weights = null,

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
            .o = try b.linear(w.attention.output, use_tensors),
            .o_norm = try b.upload(w.attention_output_norm.bytes),
            .wk = try b.upload(std.mem.sliceAsBytes(w.absorbed.key)),
            .wv = try b.upload(std.mem.sliceAsBytes(w.absorbed.value)),
            .bits = if (w.absorbed.packed_rows) |p| p.bits else 4,
            .sinks = try b.upload(std.mem.sliceAsBytes(w.sinks[0..heads])),
            .index = if (w.index) |*ix| try index.Weights.upload(&b, ix, use_tensors) else null,
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
    gate: Linear,
    up: Linear,
    down: Linear,
    output_norm: mtl.Buffer,

    pub fn upload(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer) !Feed {
        return uploadForModel(a, device, w, false);
    }

    pub fn uploadForModel(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, use_tensors: bool) !Feed {
        if (w.feed != .dense) return error.Hy4DenseFeedRequired;
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const feed = w.feed.dense;
        const out: Feed = .{
            .own = undefined,
            .merge = try Merge.upload(&b, w.mlp_hc),
            .norm = try b.upload(w.mlp_norm.bytes),
            .gate = try b.linear(feed.projections.gate, use_tensors),
            .up = try b.linear(feed.projections.up, use_tensors),
            .down = try b.linear(feed.projections.down, use_tensors),
            .output_norm = try b.upload(feed.norm.bytes),
        };
        return withAllocation(out, b);
    }

    pub fn deinit(self: *Feed) void {
        self.own.clear();
        self.* = undefined;
    }
};

pub fn withAllocation(out: anytype, b: Buffers) @TypeOf(out) {
    var owned = out;
    owned.own = b;
    return owned;
}

pub const Scratch = @import("metal_scratch.zig").Scratch;

pub const Rows = @import("metal_linears.zig").Rows;

fn twoStep(keys: u32) bool {
    return keys >= 1024;
}

pub const CacheRows = struct {
    positions: Ref,
    write_slots: Ref,
    retained: Ref,
    counts: Ref,
    cache: Ref,
    capacity: u32,
    /// When every row's keys are one contiguous run: their ranges and tiles, for the tensor units' flash kernel.
    ranges: ?mla_gpu.KeyRanges = null,
    /// The widest row's live key count this pass (0: unknown), which sizes the decode attention split.
    live_keys: u32 = 0,
    /// A sparse layer's indexer for this pass (null: it attends `retained`, the probe's contexts under 2,048).
    index: ?index.Pass = null,
};

pub const Kernels = struct {
    affine: affine.Pipelines,
    dense: dense.Ops,
    merge: ihc_gpu.Kernels,
    projections: mla_gpu.Projections,
    attention: mla_gpu.Attention,
    tensors: ?gpu_qkv.Kernels = null,
    rows: ?Rows = null,
    linears: ?Linears = null,

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
        if (self.linears) |*l| l.deinit();
        if (self.rows) |r| r.deinit();
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
        if (device.tensorUnits()) out.linears = try Linears.load(a, device, t, bits);
        out.rows = try Rows.load(a, device);
        if (device.tensorUnits()) out.projections.nax = try mla_gpu.Nax.load(a, device);
        return out;
    }

    pub fn project(self: *const Kernels, e: mtl.ComputeEncoder, w: affine.Weights, x: Ref, y: Ref, rows: u32) !void {
        var call = try affine.call(&self.affine, w, rows, false);
        if (self.rows) |r| if (r.choose(w, rows)) |pipe| {
            call.pipeline = pipe;
            call.groups = w.n / (r.rps * r.sg());
            call.threads = 32 * r.sg();
        };
        e.setPipeline(call.pipeline);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(w.w, w.w_off, 1);
        e.setBuffer(w.scales, w.s_off, 2);
        e.setBuffer(w.biases, w.b_off, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(y.buffer, y.offset, 5);
        e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
    }

    /// y = x W^T for a dense projection, on whichever kernels its weights were packed for.
    pub fn linear(self: *const Kernels, e: mtl.ComputeEncoder, w: Linear, x: Ref, y: Ref, rows: u32) !void {
        return self.linearsFrom(e, x, &.{.{ .w = w, .y = y }}, rows);
    }

    pub const Out = struct { w: Linear, y: Ref };

    /// Every projection of the same input x: on tensor units one group-sums pass, a barrier
    pub fn linearsFrom(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, outs: []const Out, rows: u32) !void {
        if (outs.len == 0) return;
        if (outs[0].w == .portable) {
            for (outs) |o| try self.project(e, o.w.portable, x, o.y, rows);
            return;
        }
        const all = &(self.linears orelse return error.Hy4LinearsMissing);
        const k = outs[0].w.tensor.k;
        // small passes: the kernels that add their own group sums (the same words, no sums pass or barrier)
        if (rows <= gpu_qkv.inline_rows) inline_sums: {
            for (outs) |o| if (all.findInline(o.w.tensor.n, o.w.tensor.k) == null) break :inline_sums;
            for (outs) |o| try all.findInline(o.w.tensor.n, o.w.tensor.k).?.encode(e, tref(x), o.w.tensor.w, o.w.tensor.sb, tref(o.y), rows);
            return;
        }
        const first = all.find(outs[0].w.tensor.n, k) orelse return error.Hy4LinearShapeMissing;
        const shared = first.reg and first.sums_pipe != null and rows <= Linears.sum_rows;
        const stride: u32 = 16 * ((rows + 15) / 16);
        if (shared) {
            e.setPipeline(first.sums_pipe.?);
            e.setBuffer(x.buffer, x.offset, 0);
            e.setValue(@as(i32, @intCast(rows)), 1);
            e.setBuffer(all.sums.?, 0, 2);
            e.setValue(@as(i32, @intCast(stride)), 3);
            e.dispatchThreads(.{ .width = stride * (k / 64) }, .{ .width = 256 });
            e.barrier();
        }
        for (outs) |o| {
            const m = o.w.tensor;
            if (m.k != k) return error.Hy4LinearInputsDiffer;
            const p = all.find(m.n, m.k) orelse return error.Hy4LinearShapeMissing;
            if (shared)
                try p.encodeSums(e, tref(x), m.w, m.sb, .{ .ref = .{ .buf = all.sums.? }, .stride = stride }, tref(o.y), rows)
            else
                try p.encode(e, tref(x), m.w, m.sb, tref(o.y), rows);
        }
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

    /// Where a sub-block sits between seams: `ready` when the previous tail already wrote this collapse's partials
    pub const Seam = struct { ready: bool = false, headed: bool = false, next: ?Merge = null, next_gain: ?mtl.Buffer = null };

    /// Decode passes fuse each tail with the next head.
    pub fn fusedSeams(rows: u32) bool {
        return rows <= ihc_gpu.Kernels.seam_rows;
    }

    /// The collapse and input norm in two launches at most (partials unless `ready`, then gates, row and norm).
    pub fn collapseSeam(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Merge, gain: mtl.Buffer, s: *const Scratch, ready: bool) !ihc.Shape {
        return self.collapseSeamFrom(e, x, w, gain, s, .{ .ready = ready });
    }

    pub fn collapseSeamFrom(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Merge, gain: mtl.Buffer, s: *const Scratch, seam: Seam) !ihc.Shape {
        const ready = seam.ready;
        var shape = w.shape;
        shape.rows = s.shape.rows;
        if (seam.headed) return shape;
        const part: ihc_gpu.Ref = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset };
        const hidden: ihc_gpu.Ref = .{ .buffer = x.buffer, .offset = x.offset };
        if (!ready) {
            try self.merge.seamPartials(e, hidden, .{ .buffer = w.project }, part, shape);
            e.barrier();
        }
        try self.merge.headPass(e, .{
            .hidden = hidden,
            .partials = part,
            .bias = .{ .buffer = w.bias },
            .scale = .{ .buffer = w.scale },
            .gain = .{ .buffer = gain },
            .collapsed = .{ .buffer = s.merged.buffer, .offset = s.merged.offset },
            .normalized = .{ .buffer = s.normalized.buffer, .offset = s.normalized.offset },
            .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
        }, shape, shape.eps);
        e.barrier();
        return shape;
    }

    /// A sub-block's end: its branch (two added, for the experts) through the output norm, mixed into `out`
    pub fn tail(self: *const Kernels, e: mtl.ComputeEncoder, branch: Ref, branch2: ?Ref, gain: mtl.Buffer, eps: f32, x: Ref, out: Ref, s: *const Scratch, shape: ihc.Shape, next: ?Merge) !void {
        return self.tailTo(e, branch, branch2, gain, eps, x, out, s, shape, .{ .next = next });
    }

    pub fn tailTo(self: *const Kernels, e: mtl.ComputeEncoder, branch: Ref, branch2: ?Ref, gain: mtl.Buffer, eps: f32, x: Ref, out: Ref, s: *const Scratch, shape: ihc.Shape, seam: Seam) !void {
        const next = seam.next;
        if (next) |n| if (seam.next_gain) |ng| if (fusedSeams(shape.rows) and shape.width == 12 * 256) {
            const r = ihc_gpu.Ref;
            return self.merge.seamPass(e, .{
                .branch = r{ .buffer = branch.buffer, .offset = branch.offset },
                .branch2 = if (branch2) |b2| r{ .buffer = b2.buffer, .offset = b2.offset } else null,
                .gain = .{ .buffer = gain },
                .residual = .{ .buffer = x.buffer, .offset = x.offset },
                .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
                .output = .{ .buffer = out.buffer, .offset = out.offset },
                .next = r{ .buffer = n.project },
                .partials = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset },
            }, .{
                .bias = .{ .buffer = n.bias },
                .scale = .{ .buffer = n.scale },
                .gain = .{ .buffer = ng },
                .collapsed = .{ .buffer = s.merged.buffer, .offset = s.merged.offset },
                .normalized = .{ .buffer = s.normalized.buffer, .offset = s.normalized.offset },
                .shape = n.shape,
                .barriers = .{ .buffer = s.seam_barriers.buffer, .offset = s.seam_barriers.offset },
            }, shape.rows, shape.width, eps);
        };
        const r = ihc_gpu.Ref;
        self.merge.tailPass(e, .{
            .branch = r{ .buffer = branch.buffer, .offset = branch.offset },
            .branch2 = if (branch2) |b2| r{ .buffer = b2.buffer, .offset = b2.offset } else null,
            .gain = .{ .buffer = gain },
            .residual = .{ .buffer = x.buffer, .offset = x.offset },
            .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
            .output = .{ .buffer = out.buffer, .offset = out.offset },
            .next = if (next) |n| r{ .buffer = n.project } else null,
            .partials = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset },
        }, shape.rows, shape.width, eps, if (next) |n| n.shape.gates else 8);
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
            .partials = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset },
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
        return self.attendSeam(e, w, s, cache, x, out, .{});
    }

    pub fn attendSeam(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Attention, s: *const Scratch, cache: CacheRows, x: Ref, out: Ref, seam: Seam) !void {
        if (s.shape.heads != w.heads) return error.Hy4WrongAttentionHeadCount;
        if (w.inputs == .tensor and (self.tensors == null or s.qkv == null)) return error.ShortHy4Scratch;
        const shape = try self.collapseSeamFrom(e, x, w.merge, w.norm, s, seam);
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
                try kernels.inputFor(m.flavor, rows).encode(e, tref(s.normalized), m.qkv.w, m.qkv.sb, tref(qkv), rows);
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
                try kernels.queryFor(m.flavor, rows).encode(e, tref(s.query_norm), m.q_b.w, m.q_b.sb, tref(s.query), rows);
            },
        }
        e.barrier();
        const ix_pass: ?index.Pass = if (cache.index) |p| (if (w.index != null) p else null) else null;
        if (ix_pass) |p| {
            if (p.scratch.rows != rows) return error.Hy4IndexRowsOutOfStep;
            try index.run(p, self, self.projections, e, &w.index.?, s.normalized, s.query_norm, cache.write_slots, cache.positions, s.rotary, w.rope, w.epsilon);
            e.barrier();
        }
        // past 2,048 keys a sparse layer attends each row's recalled slots
        const lists = if (ix_pass) |p| p.scratch.selects() else false;
        // rows of a pass across 2,048 keys that sit before it attend as they would in any other pass
        const lists_from: ?u32 = if (lists and cache.ranges != null and ix_pass.?.scratch.first < index.keys_a_row) index.keys_a_row - ix_pass.?.scratch.first else null;
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
                .slots = if (lists) ix_pass.?.cache.tokens else cache.retained,
                .counts = if (lists) ix_pass.?.cache.counts else cache.counts,
                .sinks = .{ .buffer = w.sinks },
                .scratch = s.attention,
                .output = s.latent,
                // two steps pay off from two chunks of keys; below that the one-step kernel is quicker
                .key_scores = if (twoStep(s.shape.key_stride)) s.key_scores else null,
                .ranges = if (lists and lists_from == null) null else cache.ranges,
                .lists_from = lists_from,
                .flash_parts = s.flash_parts,
                .live_keys = if (lists) index.keys_a_row else cache.live_keys,
            },
        }, s.shape, w.rope);
        e.barrier();
        try self.linear(e, w.o, s.values, s.branch, rows);
        e.barrier();
        try self.tailTo(e, s.branch, null, w.o_norm, w.epsilon, x, out, s, shape, seam);
    }

    pub fn feed(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Feed, s: *const Scratch, x: Ref, out: Ref) !void {
        return self.feedSeam(e, w, s, x, out, .{});
    }

    pub fn feedSeam(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Feed, s: *const Scratch, x: Ref, out: Ref, seam: Seam) !void {
        const inner: u32 = w.gate.outputs();
        if (inner > s.inner_capacity) return error.ShortHy4Scratch;
        const shape = try self.collapseSeamFrom(e, x, w.merge, w.norm, s, seam);
        try self.linearsFrom(e, s.normalized, &.{ .{ .w = w.gate, .y = s.gate }, .{ .w = w.up, .y = s.up } }, shape.rows);
        e.barrier();
        try self.dense.swiglu(e, dref(s.gate), dref(s.up), dref(s.inner), .{
            .rows = shape.rows,
            .width = inner,
            .gate_stride = inner,
            .up_stride = inner,
            .y_stride = inner,
        });
        e.barrier();
        try self.linear(e, w.down, s.inner, s.branch, shape.rows);
        e.barrier();
        try self.tailTo(e, s.branch, null, w.output_norm, shape.eps, x, out, s, shape, seam);
    }
};
