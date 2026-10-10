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
pub const Cache = struct { keys: mtl.Buffer, values: mtl.Buffer };
const simd = core.row_projection.simd;
/// Chips without tensor units: a prepared q4 matrix in MLX layout on simdgroup-matrix tiles.
pub const Tiled = struct {
    w: mtl.Buffer,
    scales: mtl.Buffer,
    biases: mtl.Buffer,
    n: usize,
    k: usize,
    fn init(device: mtl.Device, p: core.affine4.Prepared) !Tiled {
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
    fn deinit(t: Tiled) void {
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
pub const Backend = struct {
    allocator: std.mem.Allocator,
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
        const caches = try a.alloc(Cache, c.layers);
        var made: usize = 0;
        errdefer {
            for (caches[0..made]) |cache| {
                cache.keys.deinit();
                cache.values.deinit();
            }
            a.free(caches);
        }
        for (caches) |*cache| {
            const bytes = @as(usize, c.window + 16) * c.kvWidth() * 2;
            const keys = try target.device.buffer(bytes, options);
            const values = target.device.buffer(bytes, options) catch |err| {
                keys.deinit();
                return err;
            };
            cache.* = .{ .keys = keys, .values = values };
            made += 1;
        }
        return .{ .allocator = a, .target = target, .checkpoint = checkpoint, .config = c, .primitives = primitives, .attention = attention, .topk = topk, .head = head, .frame = .{ .allocator = a, .device = target.device, .queue = target.queue }, .caches = caches };
    }
    pub fn deinit(b: *Backend) void {
        b.frame.deinit();
        var prepared = b.prepared.valueIterator();
        while (prepared.next()) |adapter| adapter.deinit();
        b.prepared.deinit(b.allocator);
        var tiled = b.tiled.valueIterator();
        while (tiled.next()) |t| t.deinit();
        b.tiled.deinit(b.allocator);
        if (b.simd) |*p| p.deinit();
        for (b.caches) |cache| {
            cache.keys.deinit();
            cache.values.deinit();
        }
        b.allocator.free(b.caches);
        b.head.deinit();
        b.topk.deinit();
        b.attention.deinit();
        b.primitives.deinit();
    }
    pub fn prepare(b: *Backend) !void {
        if (!b.target.device.tensorUnits()) return b.prepareTiled();
        const order = try core.lane_projection.RegOrder.init(b.target.device);
        defer order.deinit();
        var iterator = b.checkpoint.tensors.iterator();
        while (iterator.next()) |entry| {
            const tensor = entry.value_ptr.*;
            if (tensor.rank != 2 or !std.mem.endsWith(u8, entry.key_ptr.*, ".weight")) continue;
            if (tensor.dtype != .bf16 or tensor.shape[0] % 32 != 0 or tensor.shape[1] % 64 != 0) return error.DraftPreparationUnqualified;
            var prepared_words = try core.affine4.prepare(b.allocator, tensor.host(u16), tensor.shape[0], tensor.shape[1]);
            defer prepared_words.deinit();
            const sk = try @import("../projection.zig").split(tensor.shape[0], tensor.shape[1]);
            const adapter = try core.affine4_lane.Adapter.init(b.allocator, b.target.device, prepared_words, sk, order);
            const key = @intFromPtr(tensor.buffer.contents() + tensor.offset);
            b.prepared.put(b.allocator, key, adapter) catch |err| {
                adapter.deinit();
                return err;
            };
        }
        if (b.prepared.count() != b.config.linearCount()) return error.DraftTensorCoverage;
    }
    /// The same q4 words, scales and biases as the M5 path, on simdgroup-matrix tiles (every width the tiles' bits).
    fn prepareTiled(b: *Backend) !void {
        b.simd = try simd.Pipelines.load(b.target.device);
        var iterator = b.checkpoint.tensors.iterator();
        while (iterator.next()) |entry| {
            const tensor = entry.value_ptr.*;
            if (tensor.rank != 2 or !std.mem.endsWith(u8, entry.key_ptr.*, ".weight")) continue;
            if (tensor.dtype != .bf16 or tensor.shape[0] % 32 != 0 or tensor.shape[1] % 64 != 0) return error.DraftPreparationUnqualified;
            var prepared_words = try core.affine4.prepare(b.allocator, tensor.host(u16), tensor.shape[0], tensor.shape[1]);
            defer prepared_words.deinit();
            const t = try Tiled.init(b.target.device, prepared_words);
            b.tiled.put(b.allocator, @intFromPtr(tensor.buffer.contents() + tensor.offset), t) catch |err| {
                t.deinit();
                return err;
            };
        }
        if (b.tiled.count() != b.config.linearCount()) return error.DraftTensorCoverage;
    }
    pub fn ops(b: *Backend) op.Ops {
        return .{ .ptr = b, .vtable = &callbacks.vtable };
    }
    pub fn source(b: *Backend) ck.Source {
        return .{ .ptr = b, .getFn = get };
    }
    fn get(ptr: *anyopaque, name: []const u8) !ck.Tensor {
        const b: *Backend = @ptrCast(@alignCast(ptr));
        return cpu(try b.checkpoint.get(name));
    }
    pub fn cpu(t: core.checkpoint_metal.Tensor) ck.Tensor {
        var shape: [4]usize = @splat(1);
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
        if (b.frame.ops % part_ops == 0) try b.frame.part();
        return b.frame.encoder;
    }
    pub fn copy(b: *Backend, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32, width: u32) !void {
        var first: u32 = 0;
        while (first < rows) : (first += 128) {
            const off = @as(usize, first) * width * 2;
            try b.target.glue.unstack(e, .{ .buffer = x.buf, .offset = x.off + off }, .{ .buffer = y.buf, .offset = y.off + off }, .{ .rows = @min(rows - first, 128), .width = width, .stride = width, .offset = 0 });
            e.barrier();
        }
    }
};
