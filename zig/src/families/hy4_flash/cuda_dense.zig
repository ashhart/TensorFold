//! The native CUDA attention and dense feed-forward passes for Hy4-flash.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const proj = @import("cuda_projection.zig");
const ihc = @import("ihc.zig");
const hc_gpu = @import("ihc_cuda.zig");
const mla = @import("mla_cuda.zig");
const shape_ = @import("mla_attention.zig");
const dims = @import("mla_weights.zig");
const model = @import("model_weights.zig");
const rope = @import("rope.zig");

pub const Hyper = struct {
    w: u64,
    b: u64,
    scale: u64,
    shape: ihc.Shape,

    fn upload(a: *proj.Owned, w: ihc.Weights) !Hyper {
        return .{
            .w = try a.upload(w.project.bytes),
            .b = try a.upload(w.bias.bytes),
            .scale = try a.upload(w.scale.bytes),
            .shape = w.shape,
        };
    }
};

pub const Attention = struct {
    heads: u32,
    position: rope.Parameters,
    q_a: proj.Weight,
    q_norm: u64,
    q_b: proj.Weight,
    kv_a: proj.Weight,
    kv_norm: u64,
    gate: proj.Weight,
    output: proj.Weight,
    output_norm: u64,
    key: u64,
    value: u64,
    sinks: u64,
    norm: u64,
    merge: Hyper,
    needs_reduction: bool,

    pub fn upload(k: *const Kernels, a: *proj.Owned, s: cuda.Stream, w: *const model.TargetLayer) !Attention {
        const heads = w.attention.heads.count;
        return .{
            .heads = heads,
            .position = w.position,
            .q_a = try k.project.pack(a, s, w.project_query),
            .q_norm = try a.upload(w.norm_query.bytes),
            .q_b = try k.project.pack(a, s, w.attention.query),
            .kv_a = try k.project.pack(a, s, w.project_kv),
            .kv_norm = try a.upload(w.norm_kv.bytes),
            .gate = try k.project.pack(a, s, w.attention.gate),
            .output = try k.project.pack(a, s, w.attention.output),
            .output_norm = try a.upload(w.attention_output_norm.bytes),
            .key = try a.upload(std.mem.sliceAsBytes(w.absorbed.key)),
            .value = try a.upload(std.mem.sliceAsBytes(w.absorbed.value)),
            .sinks = try a.upload(std.mem.sliceAsBytes(w.sinks[0..heads])),
            .norm = try a.upload(w.attention_norm.bytes),
            .merge = try Hyper.upload(a, w.attention_hc),
            .needs_reduction = w.attention.owns_output,
        };
    }
};

pub const Feed = struct {
    gate: proj.Weight,
    up: proj.Weight,
    down: proj.Weight,
    output_norm: u64,
    input_norm: u64,
    merge: Hyper,

    pub fn upload(k: *const Kernels, a: *proj.Owned, s: cuda.Stream, w: *const model.TargetLayer) !Feed {
        if (w.feed != .dense) return error.Hy4DenseFeedRequired;
        const f = w.feed.dense;
        return .{
            .gate = try k.project.pack(a, s, f.projections.gate),
            .up = try k.project.pack(a, s, f.projections.up),
            .down = try k.project.pack(a, s, f.projections.down),
            .output_norm = try a.upload(f.norm.bytes),
            .input_norm = try a.upload(w.mlp_norm.bytes),
            .merge = try Hyper.upload(a, w.mlp_hc),
        };
    }
};

pub const Scratch = struct {
    cap: shape_.Shape,
    shape: shape_.Shape,
    inner: u32,
    collapsed: u64,
    post: u64,
    norm_input: u64,
    normalized: u64,
    q_low: u64,
    q_norm: u64,
    q: u64,
    q_absorbed: u64,
    kv: u64,
    gate: u64,
    values: u64,
    latent: u64,
    rotary: u64,
    part: u64,
    sums: u64,
    branch: u64,
    ffn_gate: u64,
    ffn_up: u64,
    act: u64,

    pub fn init(a: *proj.Owned, shape: shape_.Shape, hidden: u32, q_rank: u32, inner: u32) !Scratch {
        try shape.check();
        const rows = shape.rows;
        const heads = shape.heads;
        if (rows > proj.max_rows or hidden == 0 or q_rank == 0 or inner == 0) return error.ShortHy4CudaScratch;
        const max_width = @max(hidden, q_rank, inner, heads * dims.value_width);
        return .{
            .cap = shape,
            .shape = shape,
            .inner = inner,
            .collapsed = try bf16(a, rows, hidden),
            .post = try a.alloc(@as(usize, rows) * ihc.streams * 4),
            .norm_input = try bf16(a, rows, @max(hidden, q_rank)),
            .normalized = try bf16(a, rows, hidden),
            .q_low = try bf16(a, rows, q_rank),
            .q_norm = try bf16(a, rows, q_rank),
            .q = try bf16(a, rows, heads * dims.query_stride),
            .q_absorbed = try bf16(a, rows, heads * dims.absorbed_stride),
            .kv = try bf16(a, rows, dims.absorbed_stride),
            .gate = try bf16(a, rows, heads * dims.value_width),
            .values = try bf16(a, rows, heads * dims.value_width),
            .latent = try bf16(a, rows, heads * dims.latent),
            .rotary = try bf16(a, rows, dims.rope),
            .part = try a.alloc(shape.scratchBytes()),
            .sums = try a.alloc(@as(usize, rows) * max_width / 64 * 4),
            .branch = try bf16(a, rows, hidden),
            .ffn_gate = try bf16(a, rows, inner),
            .ffn_up = try bf16(a, rows, inner),
            .act = try bf16(a, rows, inner),
        };
    }

    pub fn bind(self: *Scratch, rows: u32, heads: u32, keys: u32) !void {
        if (rows > self.cap.rows or heads > self.cap.heads or keys > self.cap.key_stride) return error.ShortHy4CudaScratch;
        self.shape = try shape_.Shape.init(rows, heads, keys);
    }
};

fn bf16(a: *proj.Owned, rows: u32, width: u32) !u64 {
    return a.alloc(@as(usize, rows) * width * @sizeOf(u16));
}

pub const Cache = struct {
    positions: u64,
    slots: u64,
    retained: u64,
    counts: u64,
    data: u64,
    capacity: u32,
};

pub const Reduce = struct {
    ctx: *anyopaque,
    /// Reduce this rank's output projection before applying the checkpoint's output norm.
    call: *const fn (*anyopaque, cuda.Stream, u64, usize) anyerror!void,
};

pub const Kernels = struct {
    project: proj.Kernels,
    norm: cuda.norms.Norms,
    merge: hc_gpu.Kernels,
    mla_project: mla.Projections,
    mla_attend: mla.Attention,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        var project = try proj.Kernels.load(ctx);
        errdefer project.deinit();
        var norm = try cuda.norms.Norms.load(ctx.d);
        errdefer norm.unload();
        var merge = try hc_gpu.Kernels.load(ctx.d);
        errdefer merge.deinit();
        var m = try mla.Projections.load(ctx.d);
        errdefer m.deinit();
        return .{
            .project = project,
            .norm = norm,
            .merge = merge,
            .mla_project = m,
            .mla_attend = try mla.Attention.load(ctx.d),
        };
    }

    pub fn deinit(self: *Kernels) void {
        self.mla_attend.deinit();
        self.mla_project.deinit();
        self.merge.deinit();
        self.norm.unload();
        self.project.deinit();
    }

    fn normal(self: *const Kernels, s: cuda.Stream, x: u64, w: u64, y: u64, f: *const Scratch, width: usize, eps: f32) !void {
        try self.norm.addRmsTail(s, x, null, w, f.norm_input, y, null, f.shape.rows, width, eps, .{ .round = true });
    }

    fn collapse(self: *const Kernels, s: cuda.Stream, w: Hyper, norm: u64, f: *const Scratch, x: u64) !ihc.Shape {
        var shape = w.shape;
        shape.rows = f.shape.rows;
        try self.merge.collapse(s, .{ .hidden = x, .projection = w.w, .bias = w.b, .scale = w.scale, .output = f.collapsed, .post = f.post }, shape);
        try self.normal(s, f.collapsed, norm, f.normalized, f, shape.width, shape.eps);
        return shape;
    }

    pub fn attend(self: *const Kernels, s: cuda.Stream, w: *const Attention, f: *const Scratch, cache: Cache, reduction: ?Reduce, x: u64, out: u64) !void {
        if (f.shape.heads != w.heads or w.needs_reduction != (reduction != null)) return error.BadHy4Rank;
        const shape = try self.collapse(s, w.merge, w.norm, f, x);
        const r = shape.rows;
        try self.project.run(s, w.q_a, f.normalized, f.sums, f.q_low, r);
        try self.project.run(s, w.kv_a, f.normalized, f.sums, f.kv, r);
        try self.project.run(s, w.gate, f.normalized, f.sums, f.gate, r);
        try self.normal(s, f.q_low, w.q_norm, f.q_norm, f, w.q_b.inputs(), shape.eps);
        try self.project.run(s, w.q_b, f.q_norm, f.sums, f.q, r);
        try mla.forward(self.mla_project, self.mla_attend, s, .{
            .query = f.q,
            .key_weights = w.key,
            .value_weights = w.value,
            .value_logits = f.gate,
            .positions = cache.positions,
            .rotary_table = f.rotary,
            .values = f.values,
            .kv = .{ .projection = f.kv, .gain = w.kv_norm, .write_slots = cache.slots, .capacity = cache.capacity, .epsilon = shape.eps },
            .attention = .{
                .query = f.q_absorbed,
                .cache = cache.data,
                .slots = cache.retained,
                .counts = cache.counts,
                .sinks = w.sinks,
                .scratch = f.part,
                .scratch_bytes = f.cap.scratchBytes(),
                .output = f.latent,
            },
        }, f.shape, w.position);
        try self.project.run(s, w.output, f.values, f.sums, f.branch, r);
        if (reduction) |op| try op.call(op.ctx, s, f.branch, @as(usize, r) * shape.width);
        try self.normal(s, f.branch, w.output_norm, f.normalized, f, shape.width, shape.eps);
        try self.merge.add(s, .{ .residual = x, .branch = f.normalized, .post = f.post, .output = out }, shape);
    }

    pub fn feed(self: *const Kernels, s: cuda.Stream, w: *const Feed, f: *const Scratch, x: u64, out: u64) !void {
        const inner = w.down.inputs();
        if (inner > f.inner) return error.ShortHy4CudaScratch;
        const shape = try self.collapse(s, w.merge, w.input_norm, f, x);
        const r = shape.rows;
        try self.project.run(s, w.gate, f.normalized, f.sums, f.ffn_gate, r);
        try self.project.run(s, w.up, f.normalized, f.sums, f.ffn_up, r);
        try self.project.act.swigluRound(s, f.ffn_gate, inner, f.ffn_up, inner, f.act, null, r, inner);
        try self.project.run(s, w.down, f.act, f.sums, f.branch, r);
        try self.normal(s, f.branch, w.output_norm, f.normalized, f, shape.width, shape.eps);
        try self.merge.add(s, .{ .residual = x, .branch = f.normalized, .post = f.post, .output = out }, shape);
    }
};
