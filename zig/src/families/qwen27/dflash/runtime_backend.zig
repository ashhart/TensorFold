//! The draft adapter owns workspace and rolling caches while the target owns embedding, head and accepted state.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const op = @import("operators.zig");
const cfg = @import("config.zig");
const ck = @import("../checkpoint.zig");
const Target = @import("../model.zig").Model;
const callbacks = @import("runtime_callbacks.zig");
pub const Head = @import("runtime_head.zig").Head;
pub const Frame = @import("runtime_frame.zig").Frame;
pub const Ref = core.draft_ops.Ref;
const caches_impl = @import("caches.zig");
pub const Cache = caches_impl.Pair(mtl.Buffer);
pub fn newCaches(a: std.mem.Allocator, device: mtl.Device, c: cfg.Config, layers: usize) ![]Cache {
    return caches_impl.newCaches(mtl.Buffer, a, device, c, layers);
}
pub fn freeCaches(a: std.mem.Allocator, caches: []Cache) void {
    caches_impl.freeCaches(mtl.Buffer, a, caches);
}
const simd = core.row_projection.simd;
/// Chips without tensor units: a prepared q4 matrix in MLX layout on simdgroup-matrix tiles.
pub const Tiled = struct {
    w: mtl.Buffer,
    scales: mtl.Buffer,
    biases: mtl.Buffer,
    n: usize,
    k: usize,
    pub fn init(device: mtl.Device, p: core.affine4.Prepared) !Tiled {
        const w = try device.buffer(p.words.len * 4, options);
        errdefer w.deinit();
        const scales = try device.buffer(p.scales.len * 2, options);
        errdefer scales.deinit();
        const biases = try device.buffer(p.biases.len * 2, options);
        @memcpy(w.slice(u32, p.words.len), p.words);
        @memcpy(scales.slice(u16, p.scales.len), p.scales);
        @memcpy(biases.slice(u16, p.biases.len), p.biases);
        return .{ .w = w, .scales = scales, .biases = biases, .n = p.n, .k = p.k };
    }
    pub fn deinit(t: Tiled) void {
        t.w.deinit();
        t.scales.deinit();
        t.biases.deinit();
    }
    pub fn matrix(t: Tiled) simd.Matrix {
        return .{ .w = t.w, .scales = t.scales, .biases = t.biases, .n = t.n, .k = t.k };
    }
};
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
/// Ops a draft frame encodes before committing them, so the GPU starts while the CPU encodes the rest.
const part_ops = 40;
/// The first part is short, so the GPU starts while the rest is still being encoded.
const first_part_ops = 6;
pub const Backend = struct {
    allocator: std.mem.Allocator,
    residency: ?mtl.ResidencySet.Follow = null,
    target: *Target,
    checkpoint: *core.checkpoint_metal.Checkpoint,
    config: cfg.Config,
    primitives: core.draft_ops.Ops,
    attention: core.shared_attention.Attention,
    topk: core.bf16_topk_gpu.Ops,
    head: @import("runtime_head.zig").Head,
    frame: Frame,
    caches: []Cache,
    context: op.Context = .{ .cache = 1, .begin = 0, .end = 0 },
    batch: []const []Cache = &.{}, // a batched pass's streams' caches, by stream
    transaction: ?u64 = null,
    staged: ?u64 = null,
    appended: u32 = 0,
    appended_rows: u32 = 0,
    failed: bool = false,
    prepared: std.AutoHashMapUnmanaged(usize, core.affine4_lane.Adapter) = .empty,
    tiled: std.AutoHashMapUnmanaged(usize, Tiled) = .empty,
    simd: ?simd.Pipelines = null,
    pub fn init(a: std.mem.Allocator, target: *Target, checkpoint: *core.checkpoint_metal.Checkpoint, c: cfg.Config) !Backend {
        try c.check();
        const primitives = try core.draft_ops.Ops.init(target.device);
        errdefer primitives.deinit();
        const attention = try core.shared_attention.Attention.init(a, target.device, .{ .query_heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.head_dim, .rotary_dim = c.head_dim, .capacity = c.window + 16, .output = .bf16 });
        errdefer attention.deinit();
        const topk = try core.bf16_topk_gpu.Ops.init(target.device);
        errdefer topk.deinit();
        var head = try @import("runtime_head.zig").Head.init(a, target.device, target.weights.head);
        errdefer head.deinit();
        const caches = try newCaches(a, target.device, c, c.layers);
        return .{ .allocator = a, .target = target, .checkpoint = checkpoint, .config = c, .primitives = primitives, .attention = attention, .topk = topk, .head = head, .frame = .{ .allocator = a, .device = target.device, .queue = target.queue }, .caches = caches };
    }
    pub fn setResident(b: *Backend, hook: ?mtl.ResidencySet.Follow) void {
        @import("../residency.zig").setDraft(b, hook);
    }

    pub fn deinit(b: *Backend) void {
        b.setResident(null);
        b.frame.deinit();
        var prepared = b.prepared.valueIterator();
        while (prepared.next()) |adapter| adapter.deinit();
        b.prepared.deinit(b.allocator);
        var tiled = b.tiled.valueIterator();
        while (tiled.next()) |t| t.deinit();
        b.tiled.deinit(b.allocator);
        if (b.simd) |*p| p.deinit();
        freeCaches(b.allocator, b.caches);
        b.head.deinit();
        b.topk.deinit();
        b.attention.deinit();
        b.primitives.deinit();
    }
    /// Prepared q4 linears use tensor adapters or SIMD tiles, keyed by the graph weight address.
    pub fn prepare(b: *Backend) !void {
        const units = b.target.device.tensorUnits();
        const order = if (units) try core.lane_projection.RegOrder.init(b.target.device) else null;
        defer if (order) |o| o.deinit();
        if (!units) b.simd = try simd.Pipelines.load(b.target.device);
        var rows: std.ArrayList(u16) = .empty;
        defer rows.deinit(b.allocator);
        var kept = b.checkpoint.tensors.iterator();
        while (kept.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const t = entry.value_ptr.*;
            try b.prepareOne(order, @intFromPtr(t.buffer.contents() + t.offset), t.dtype, t.shape, t.host(u16));
        };
        var later = b.checkpoint.deferred.iterator();
        while (later.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const d = entry.value_ptr.*;
            try rows.resize(b.allocator, d.bytes / 2);
            try b.checkpoint.read(entry.key_ptr.*, 0, std.mem.sliceAsBytes(rows.items));
            try b.prepareOne(order, @intFromPtr((try b.checkpoint.view(entry.key_ptr.*)).ptr), d.dtype, d.shape, rows.items);
        };
        if ((if (units) b.prepared.count() else b.tiled.count()) != b.config.linearCount()) return error.DraftTensorCoverage;
    }
    /// One bf16 [n][k] linear's q4 copy under `key`: an M5 lane adapter, or (no `order`) the same bits on tiles.
    fn prepareOne(b: *Backend, order: ?core.lane_projection.RegOrder, key: usize, dtype: core.checkpoint_metal.DType, shape: @FieldType(core.checkpoint_metal.Tensor, "shape"), rows: []const u16) !void {
        if (dtype != .bf16 or shape[0] % 32 != 0 or shape[1] % 64 != 0) return error.DraftPreparationUnqualified;
        var prepared_words = try core.affine4.prepare(b.allocator, rows, shape[0], shape[1]);
        defer prepared_words.deinit();
        if (order) |o| {
            const adapter = try core.affine4_lane.Adapter.init(b.allocator, b.target.device, prepared_words, try @import("../projection.zig").split(shape[0], shape[1]), o);
            try @import("../residency.zig").putPrepared(b, key, adapter);
        } else {
            const t = try Tiled.init(b.target.device, prepared_words);
            try @import("../residency.zig").putTiled(b, key, t);
        }
    }
    /// The draft's linears: its rank-2 weights, which q4 drafts read only through their prepared copies.
    pub fn linear(name: []const u8, rank: usize) bool {
        return rank == 2 and std.mem.endsWith(u8, name, ".weight");
    }
    pub fn ops(b: *Backend) op.Ops {
        return .{ .ptr = b, .vtable = &callbacks.vtable };
    }
    pub fn source(b: *Backend) ck.Source {
        return .{ .ptr = b, .getFn = get };
    }
    /// A tensor's host view; a deferred linear's view only names it (its key), its bytes read once by prepare.
    fn get(ptr: *anyopaque, name: []const u8) !ck.Tensor {
        const b: *Backend = @ptrCast(@alignCast(ptr));
        const d = b.checkpoint.deferred.get(name) orelse return cpu(try b.checkpoint.get(name));
        var shape: @FieldType(ck.Tensor, "shape") = @splat(1);
        @memcpy(shape[0..d.rank], d.shape[0..d.rank]);
        return .{ .dtype = d.dtype, .rank = @intCast(d.rank), .shape = shape, .bytes = try b.checkpoint.view(name) };
    }
    pub fn cpu(t: core.checkpoint_metal.Tensor) ck.Tensor {
        var shape: @FieldType(ck.Tensor, "shape") = @splat(1);
        @memcpy(shape[0..t.rank], t.shape[0..t.rank]);
        return .{ .dtype = t.dtype, .rank = @intCast(t.rank), .shape = shape, .bytes = t.buffer.contents()[t.offset .. t.offset + t.bytes] };
    }
    pub fn weight(b: *Backend, t: ck.Tensor) !Ref {
        var it = b.checkpoint.tensors.valueIterator();
        while (it.next()) |candidate| if (candidate.buffer.contents() + candidate.offset == t.bytes.ptr and candidate.bytes == t.bytes.len and candidate.dtype == t.dtype) return .{ .buf = candidate.buffer, .off = candidate.offset };
        return error.UnownedDraftWeight;
    }
    pub fn encoder(b: *Backend, epoch: u64) !mtl.ComputeEncoder {
        if (b.failed) return error.DraftBackendFailed;
        try b.frame.require(epoch);
        b.frame.ops += 1;
        if (b.frame.ops == first_part_ops or b.frame.ops % part_ops == 0) try b.frame.part();
        return b.frame.encoder;
    }
    /// TF_Q27_PROFILE: the frame's passes join `trace`'s `drafter` class, and its spans keep an open frame's encoder.
    pub fn profile(b: *Backend, trace: ?*core.gpu_profile.Trace) void {
        b.frame.profiler = trace;
    }
    pub fn copy(b: *Backend, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32, width: u32) !void {
        try b.target.glue.unstack(e, .{ .buffer = x.buf, .offset = x.off }, .{ .buffer = y.buf, .offset = y.off }, .{ .rows = rows, .width = width, .stride = width, .offset = 0 });
        e.barrier();
    }
};
