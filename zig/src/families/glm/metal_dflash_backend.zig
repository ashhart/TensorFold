//! GLM's DFlash2 draft adapter: the shared draft ops encoded over GLM's own kernels, frame and rolling caches.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const op = @import("../qwen27/dflash/operators.zig");
const draft_cfg = @import("../qwen27/dflash/config.zig");
const ck = @import("../qwen27/checkpoint.zig");
const frame_mod = @import("../qwen27/dflash/runtime_frame.zig");
const context = @import("metal_dflash_context.zig");
const fwd = @import("forward.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kernels = @import("kernels.zig");
const wide = @import("wide.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const dtopk = @import("metal_dflash_topk.zig");

pub const Frame = frame_mod.Frame;
pub const Ref = core.draft_ops.Ref;
const caches_impl = @import("../qwen27/dflash/caches.zig");
pub const Cache = caches_impl.Pair(mtl.Buffer);
pub fn newCaches(a: std.mem.Allocator, device: mtl.Device, c: draft_cfg.Config, layers: usize) ![]Cache {
    return caches_impl.newCaches(mtl.Buffer, a, device, c, layers);
}
pub fn freeCaches(a: std.mem.Allocator, caches: []Cache) void {
    caches_impl.freeCaches(mtl.Buffer, a, caches);
}

/// The GLM target the drafter borrows: its device, queue, kernels, weights and shape (one Mac, TP1).
pub const Target = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: *kernels.Kernels,
    w: *const wts.Weights,
    c: cfg.Config,
};

/// Ops a draft frame encodes before committing them (`first_part_ops` opens the first commit earlier).
const part_ops = 40;
/// The first part is short, so the GPU starts while the rest is still being encoded.
const first_part_ops = 6;
/// A drafter pass's times: host ticks at its commit, its wait's return and its lattice read; its GPU span (seconds).
pub const Stamps = struct {
    begin: u64 = 0,
    first: u64 = 0,
    commit: u64 = 0,
    waited: u64 = 0,
    read: u64 = 0,
    gpu: [2]f64 = .{ 0, 0 },

    /// GLM_WINDOWS' draft line: phases in ms after the last window returned (host) or ended (GPU span).
    pub fn logPhases(s: Stamps, r: u64, last_end: f64, enter: u64, ahead: u64, passes: u64, done: u64) void {
        const ms = struct {
            fn of(t: u64, from: u64) f64 {
                return if (t == 0 or t < from) -1 else @as(f64, @floatFromInt(t - from)) / 24e3;
            }
        }.of;
        const drafted = s.begin >= enter;
        std.log.info("glm draft: enter {d:.2} ahead {d:.2} begin {d:.2} commit {d:.2} gpu {d:.2}-{d:.2} waited {d:.2} read {d:.2} passes {d:.2} done {d:.2} ms start {d:.3} seen {d:.3} ms", .{
            ms(enter, r),                                                                    ms(ahead, r),
            if (drafted) ms(s.begin, r) else -1,                                             if (drafted) ms(s.commit, r) else -1,
            if (drafted) (s.gpu[0] - last_end) * 1e3 else -1,                                if (drafted) (s.gpu[1] - last_end) * 1e3 else -1,
            if (drafted) ms(s.waited, r) else -1,                                            if (drafted) ms(s.read, r) else -1,
            ms(passes, r),                                                                   ms(done, r),
            if (drafted) (s.gpu[0] - @as(f64, @floatFromInt(s.first)) / 24e6) * 1e3 else -1, if (drafted) (@as(f64, @floatFromInt(s.waited)) / 24e6 - s.gpu[1]) * 1e3 else -1,
        });
    }
};
fn ticks() u64 {
    return @import("sys").c.mach_absolute_time();
}
/// The frame's GPU span: its first command buffer's start to its last one's end.
fn span(f: *const Frame) [2]f64 {
    const first = if (f.part_count > 0) f.parts[0] else f.cb;
    return .{ first.gpuStart(), f.cb.gpuEnd() };
}
/// A block planned on the GPU (metal_dflash_follow): its ids, rope positions, mask and attention table.
pub const Device = struct { ids: Ref, positions: Ref, mask: Ref, keys: [8]@import("metal_dflash_attention.zig").Stream, streams: u32, mask_stride: u32 };
pub const Backend = struct {
    allocator: std.mem.Allocator,
    target: *Target,
    checkpoint: *core.checkpoint_metal.Checkpoint,
    config: draft_cfg.Config,
    primitives: core.draft_ops.Ops,
    attention: core.shared_attention.Attention,
    draft_attention: @import("metal_dflash_attention.zig").Pipes,
    topk: core.bf16_topk_gpu.Ops,
    draft_topk: dtopk.TopK,
    frame: Frame,
    queue: mtl.Queue, // the queue its passes commit to: the target's
    caches: []Cache,
    batch: []const []Cache = &.{}, // a batched pass's streams' caches, by stream
    batch_block: u32 = 0, // a batched pass's rows a stream: the row kernels' head takes each stream's mask rows
    context: op.Context = .{ .cache = 1, .begin = 0, .end = 0 },
    transaction: ?u64 = null,
    staged: ?u64 = null,
    appended: u32 = 0,
    appended_rows: u32 = 0,
    failed: bool = false,
    copies: ?@import("metal_dflash_prepared.zig").Copies = null, // --drafter-bits 4: the linears' copies
    stamps: Stamps = .{}, // the last pass's host and GPU times, for GLM_WINDOWS' draft line
    follow: ?Device = null, // while set, the block reads its ids, positions and attention mask from these buffers
    pub fn init(a: std.mem.Allocator, target: *Target, checkpoint: *core.checkpoint_metal.Checkpoint, c: draft_cfg.Config) !Backend {
        try c.check();
        if (target.c.tp != 1 or target.c.tp_rank != 0) return error.TargetBinding;
        // the drafter reads the target's own embedding and head: their widths' kernels load here once
        if (!wide.native(target.w.embed) or !wide.native(target.w.head)) try target.k.widen(a, target.device);
        const primitives = try core.draft_ops.Ops.init(target.device);
        errdefer primitives.deinit();
        const attention = try core.shared_attention.Attention.init(a, target.device, .{ .query_heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.head_dim, .capacity = c.window + 16, .output = .bf16 });
        errdefer attention.deinit();
        const draft_attention = try @import("metal_dflash_attention.zig").Pipes.init(a, target.device, .{ .heads = c.heads, .kv_heads = c.kv_heads, .dim = c.head_dim }, null);
        errdefer draft_attention.deinit();
        const topk = try core.bf16_topk_gpu.Ops.init(target.device);
        errdefer topk.deinit();
        const draft_topk = try dtopk.TopK.init(a, target.device, null);
        errdefer draft_topk.deinit();
        const caches = try newCaches(a, target.device, c, c.layers);
        return .{ .allocator = a, .target = target, .checkpoint = checkpoint, .config = c, .primitives = primitives, .attention = attention, .draft_attention = draft_attention, .topk = topk, .draft_topk = draft_topk, .frame = .{ .allocator = a, .device = target.device, .queue = target.queue }, .queue = target.queue, .caches = caches };
    }
    pub fn deinit(b: *Backend) void {
        if (b.copies) |*c| c.deinit();
        b.frame.deinit();
        freeCaches(b.allocator, b.caches);
        b.draft_topk.deinit();
        b.topk.deinit();
        b.draft_attention.deinit();
        b.attention.deinit();
        b.primitives.deinit();
    }
    /// The head the pass reads: the lightweight 4-bit draft head when present, else the target's output head.
    pub fn headWeights(b: *const Backend, _: op.View) wts.Affine {
        return b.target.w.draft_head orelse b.target.w.head;
    }
    pub fn ops(b: *Backend) op.Ops {
        return .{ .ptr = b, .vtable = &vtable };
    }
    pub fn source(b: *Backend) ck.Source {
        return .{ .ptr = b, .getFn = get };
    }
    /// A tensor's host view; a linear left in the file names its copy by its view's address and has no bytes.
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
        if (b.frame.ops == first_part_ops) b.stamps.first = ticks(); // the first part's commit, for GLM_WINDOWS
        if (b.frame.ops == first_part_ops or b.frame.ops % part_ops == 0) try b.frame.part();
        return b.frame.encoder;
    }
    /// Rows of bf16 through `copy_u32` (whole u32s: every width here is even), then a barrier.
    pub fn copy(b: *Backend, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32, width: u32) !void {
        const words: u32 = @intCast(@as(usize, rows) * width * 2 / 4);
        if (@as(usize, rows) * width * 2 % 4 != 0) return error.DraftOperatorShape;
        e.setPipeline(b.target.k.copy_u32);
        fwd.bind(e, 0, .{ .{ .buf = x.buf, .off = x.off }, .{ .buf = y.buf, .off = y.off } });
        e.setValue(words, 2);
        e.dispatchThreads(fwd.size(words, 1, 1), fwd.size(256, 1, 1));
        e.barrier();
    }
    /// The forward shim over the target's kernels and weights: embedding rows and the head's products.
    fn shim(b: *const Backend) fwd.Ctx {
        return .{ .k = b.target.k, .c = &b.target.c, .w = b.target.w, .s = undefined, .sc = undefined };
    }
};

fn backend(ptr: *anyopaque) *Backend {
    return @ptrCast(@alignCast(ptr));
}
fn begin(ptr: *anyopaque) !u64 {
    const b = backend(ptr);
    if (b.failed) return error.DraftBackendFailed;
    b.stamps = .{ .begin = ticks() };
    return b.frame.begin();
}
fn release(ptr: *anyopaque, epoch: u64) void {
    const b = backend(ptr);
    if (epoch == b.frame.epoch) b.frame.release();
}
fn project(ptr: *anyopaque, epoch: u64, x: op.View, w: ck.Tensor, mode: op.Mode) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, @intCast(w.shape[0]), .bf16);
    if (mode != .bf16_reference) {
        const copies = b.copies orelse return error.DraftPreparationUnqualified;
        return if (copies.project(e, w, x, try b.frame.ref(x), y, try b.frame.ref(y))) y else |err| err;
    }
    try b.primitives.linear(e, try b.frame.ref(x), try b.weight(w), try b.frame.ref(y), .{ .rows = x.rows, .n = y.width, .k = x.width, .x_stride = x.stride, .w_stride = x.width, .y_stride = y.stride });
    e.barrier();
    return y;
}
fn norm(ptr: *anyopaque, epoch: u64, x: op.View, p: op.Norm) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, x.width, .bf16);
    try b.primitives.norm(e, try b.frame.ref(x), try b.weight(p.weight), try b.frame.ref(y), .{ .rows = x.rows, .heads = p.heads, .dim = p.head_dim, .x_stride = x.stride, .y_stride = y.stride, .eps = p.eps });
    e.barrier();
    return y;
}
/// The target's own embedding table reads the ids: GLM's mask token embeds in bounds, so no mapping rides along.
fn embed(ptr: *anyopaque, epoch: u64, _: ck.Linear, ids: []const u32) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const r = if (b.follow) |f| f.ids else blk: {
        const tokens = try b.frame.ref(try b.frame.alloc(@intCast(ids.len), 1, .u32));
        @memcpy(tokens.buf.slice(u32, ids.len), ids);
        break :blk tokens;
    };
    const y = try b.frame.alloc(@intCast(ids.len), b.config.hidden, .bf16);
    const out = try b.frame.ref(y);
    var x = b.shim();
    fwd.embedRows(&x, e, .{ .buf = r.buf, .off = r.off }, .{ .buf = out.buf, .off = out.off }, @intCast(ids.len));
    e.barrier();
    return y;
}
fn conv(ptr: *anyopaque, epoch: u64, x: op.View, p: op.Conv) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, x.width, .bf16);
    try b.primitives.conv(e, try b.frame.ref(x), try b.frame.ref(p.dynamic), try b.weight(p.base), if (p.residual) |r| try b.frame.ref(r) else null, try b.frame.ref(y), .{ .rows = x.rows, .width = x.width, .group = p.group, .block = p.block, .branch = p.branch, .residual = @intFromBool(p.residual != null), .x_stride = x.stride, .dynamic_stride = p.dynamic.stride, .y_stride = y.stride });
    e.barrier();
    return y;
}
fn rope(ptr: *anyopaque, epoch: u64, x: op.View, p: op.Rotary) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const r = if (b.follow) |f| f.positions else blk: {
        const positions = try b.frame.ref(try b.frame.alloc(x.rows, 1, .u64));
        @memcpy(positions.buf.slice(u64, p.positions.len), p.positions);
        break :blk positions;
    };
    const y = try b.frame.alloc(x.rows, x.width, .bf16);
    try b.primitives.rope(e, try b.frame.ref(x), r, try b.frame.ref(y), .{ .rows = x.rows, .heads = p.heads, .dim = p.dim, .x_stride = x.stride, .y_stride = y.stride, .theta = @floatCast(p.theta) });
    e.barrier();
    return y;
}
fn swiglu(ptr: *anyopaque, epoch: u64, g: op.View, u: op.View) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(g.rows, g.width, .bf16);
    try b.primitives.swiglu(e, try b.frame.ref(g), try b.frame.ref(u), try b.frame.ref(y), .{ .rows = g.rows, .width = g.width, .gate_stride = g.stride, .up_stride = u.stride, .y_stride = y.stride });
    e.barrier();
    return y;
}
fn maskRows(ptr: *anyopaque, epoch: u64, x: op.View, first: u32, rows: u32) !op.View {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    if (first > x.rows or rows > x.rows - first) return error.DraftOperatorShape;
    const r = try b.frame.ref(x);
    return b.frame.add(.{ .buf = r.buf, .off = r.off + @as(usize, first) * x.stride * 2 }, rows, x.width, x.stride, .bf16);
}
/// Full-vocabulary logits through the target's own head: no span remap, so no token ids ride along.
fn head(ptr: *anyopaque, epoch: u64, x: op.View) !op.Head {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const hw = b.headWeights(x);
    const y = try b.frame.alloc(x.rows, hw.n, .bf16);
    const input = try b.frame.ref(x);
    const output = try b.frame.ref(y);
    if (blockHead(b, x)) |pipes| {
        const q = hw;
        const sums = try b.frame.ref(try b.frame.alloc(x.rows, q.k / 64, .f32));
        const offsets = try b.frame.ref(try b.frame.alloc(1, 1, .u32));
        std.mem.writeInt(i32, offsets.buf.contents()[offsets.off..][0..4], 0, .little);
        affine_mm.rowSums(e, pipes, 64, input, sums, x.rows, q.k);
        e.barrier();
        headGather(e, pipes, input, sums, q, offsets, output, x.rows);
        e.barrier();
        return .{ .logits = y, .token_ids = null };
    }
    var x_ctx = b.shim();
    if (b.batch_block == 0) {
        var s: u32 = 0;
        while (s < x.rows) {
            const step = @min(x.rows - s, 16);
            const at = @as(usize, s) * x.stride * 2;
            const to = @as(usize, s) * hw.n * 2;
            fwd.qmv(&x_ctx, e, b.target.k.qmv_head, .{ .buf = input.buf, .off = input.off + at }, hw, .{ .buf = output.buf, .off = output.off + to }, step);
            s += step;
        }
        e.barrier();
        return .{ .logits = y, .token_ids = null };
    }
    if (b.batch.len > 1 and b.batch_block > 1 and x.rows == @as(u32, @intCast(b.batch.len)) * (b.batch_block - 1)) {
        const step = b.batch_block - 1;
        var s: u32 = 0;
        while (s < x.rows) : (s += step) {
            const at = @as(usize, s) * x.stride * 2;
            const to = @as(usize, s) * hw.n * 2;
            fwd.qmv(&x_ctx, e, b.target.k.qmv_head, .{ .buf = input.buf, .off = input.off + at }, hw, .{ .buf = output.buf, .off = output.off + to }, step);
        }
        e.barrier();
        return .{ .logits = y, .token_ids = null };
    }
    // the row kernels take a lone stream's mask rows as its own pass does: a batch heads each stream's apart
    const block = if (b.batch_block > 1 and x.rows % b.batch_block == 0) b.batch_block else x.rows;
    const first: u32 = if (block == x.rows) 0 else 1;
    var s: u32 = 0;
    while (s < x.rows) : (s += block) {
        const at = (@as(usize, s) + first) * x.stride * 2;
        const to = (@as(usize, s) + first) * hw.n * 2;
        fwd.qmv(&x_ctx, e, b.target.k.qmv_head, .{ .buf = input.buf, .off = input.off + at }, hw, .{ .buf = output.buf, .off = output.off + to }, block - first);
    }
    e.barrier();
    return .{ .logits = y, .token_ids = null };
}
/// x through the head as one matrix: affine_mm's 32-row gather tile to 32 rows, else its 64-row tile (same bits).
pub fn headGather(e: mtl.ComputeEncoder, p: affine_mm.Pipes, x: Ref, sums: Ref, q: wts.Affine, offsets: Ref, y: Ref, rows: u32) void {
    const bm: u32 = if (rows <= 32) 32 else 64;
    e.setPipeline(p[if (bm == 32) 2 else 3]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(q.w.buf, q.w.off, 1);
    e.setBuffer(q.s.buf, q.s.off, 2);
    e.setBuffer(q.b.buf, q.b.off, 3);
    e.setBuffer(sums.buf, sums.off, 4);
    e.setValue(affine_mm.Args{ .rows = @intCast(rows), .n = @intCast(q.n), .k = @intCast(q.k), .experts = 1 }, 5);
    e.setBuffer(y.buf, y.off, 6);
    e.setBuffer(offsets.buf, offsets.off, 7);
    e.dispatchGroups(mtl.Size.of(q.n / 64, (rows + bm - 1) / bm, 1), mtl.Size.of(32, 2, 2));
}
/// Where the draft head runs: affine_mm's 4-bit or 8-bit g64 product on tensor units, else the decode row kernels.
pub const HeadPath = enum { block4, block8, rows };
pub fn headPath(tensor_units: bool, q: wts.Affine, contiguous: bool) HeadPath {
    if (!tensor_units or !contiguous or q.more != 0 or q.group != 64 or q.n % 64 != 0 or q.k % 64 != 0) return .rows;
    if (wide.native(q)) return .block4;
    return if (q.bits == 8) .block8 else .rows;
}
fn blockHead(b: *const Backend, x: op.View) ?affine_mm.Pipes {
    return switch (headPath(b.target.device.tensorUnits(), b.headWeights(x), x.stride == x.width)) {
        .block4 => b.target.k.mm_bf16,
        .block8 => b.target.k.wide.?.mm8,
        .rows => null,
    };
}
fn lattice(ptr: *anyopaque, epoch: u64, h: op.Head, projected: op.View, c: draft_cfg.Config) !op.Lattice {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    const output = try b.frame.alloc(h.logits.rows, @sizeOf(core.bf16_topk.Row), .u8);
    const logits = try b.frame.ref(h.logits);
    const out = try b.frame.ref(output);
    const p = core.bf16_topk.Params{ .rows = h.logits.rows, .vocab = h.logits.width, .k = c.topk, .stride = h.logits.stride };
    // the block rows' top k with core's bytes: a threadgroup a row, else core's spans and merge
    if (dtopk.takes(p)) try b.draft_topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = out.buf, .off = out.off }, p) else try b.topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = out.buf, .off = out.off }, p);
    b.frame.encoder.barrier();
    b.stamps.commit = ticks();
    b.frame.finish() catch |err| {
        b.failed = true;
        return err;
    };
    b.stamps.waited = ticks();
    b.stamps.gpu = span(&b.frame);
    defer b.stamps.read = ticks();
    return @import("../qwen27/dflash/runtime_lattice.zig").read(b.allocator, &b.frame, output, projected, c, h);
}
/// A pass's top k encoded with no wait: its rows' candidates, read once the frame's last buffer completes.
pub fn encodeTopk(b: *Backend, epoch: u64, h: op.Head, k: u32) !op.View {
    _ = try b.encoder(epoch);
    const output = try b.frame.alloc(h.logits.rows, @sizeOf(core.bf16_topk.Row), .u8);
    const logits = try b.frame.ref(h.logits);
    const out = try b.frame.ref(output);
    const p = core.bf16_topk.Params{ .rows = h.logits.rows, .vocab = h.logits.width, .k = k, .stride = h.logits.stride };
    if (dtopk.takes(p)) try b.draft_topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = out.buf, .off = out.off }, p) else try b.topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = out.buf, .off = out.off }, p);
    b.frame.encoder.barrier();
    return output;
}
/// A completed pass's lattices: a lone stream's headed rows, packed active rows, or full block rows.
pub fn readLattices(b: *Backend, h: op.Head, topk: op.View, projected: op.View, c: draft_cfg.Config, block: u32, out: []op.Lattice) !void {
    const read = @import("../qwen27/dflash/runtime_lattice.zig").read;
    if (out.len == 1 and h.logits.rows == block - 1) {
        out[0] = try read(b.allocator, &b.frame, topk, projected, c, h);
        return;
    }
    const packed_rows = out.len > 1 and h.logits.rows == out.len * (block - 1);
    if (!packed_rows and h.logits.rows != out.len * block) return error.DraftOperatorShape;
    const step: u32 = if (packed_rows) block - 1 else block;
    const skip: u32 = if (packed_rows) 0 else 1;
    var made: usize = 0;
    errdefer for (out[0..made]) |*l| l.deinit();
    for (out, 0..) |*l, k| {
        const first: u32 = @intCast(k * step + skip); // full-block layout skips each stream's anchor row
        const hs: op.Head = .{ .logits = try rowsOf(b, h.logits, first, block - 1), .token_ids = h.token_ids };
        l.* = try read(b.allocator, &b.frame, try rowsOf(b, topk, first, block - 1), try rowsOf(b, projected, first, block - 1), c, hs);
        made += 1;
    }
}
/// Commit the open draft frame, wait for completion, and record host/GPU stamp times.
pub fn finishStamped(b: *Backend) !void {
    b.stamps.commit = ticks();
    b.frame.finish() catch |err| {
        b.failed = true;
        return err;
    };
    b.stamps.waited = ticks();
    b.stamps.gpu = span(&b.frame);
}
/// Read stamp after lattice readback.
pub fn stampRead(b: *Backend) void {
    b.stamps.read = ticks();
}
/// Rows [first, first + rows) of `x`, an alias in the frame.
pub fn rowsOf(b: *Backend, x: op.View, first: u32, rows: u32) !op.View {
    if (first > x.rows or rows > x.rows - first) return error.DraftOperatorShape;
    const r = try b.frame.ref(x);
    return b.frame.add(.{ .buf = r.buf, .off = r.off + @as(usize, first) * x.stride * x.dtype.size() }, rows, x.width, x.stride, x.dtype);
}
/// A batched pass's head: one top-k over every stream's rows and one wait, then each stream's lattice past its anchor.
fn lattices(ptr: *anyopaque, epoch: u64, h: op.Head, projected: op.View, c: draft_cfg.Config, block: u32, out: []op.Lattice) !void {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    if (block < 2 or h.logits.rows != out.len * block or out.len > dtopk.max_rows / block) return error.DraftOperatorShape;
    const output = try b.frame.alloc(h.logits.rows, @sizeOf(core.bf16_topk.Row), .u8);
    const logits = try b.frame.ref(h.logits);
    const rows = try b.frame.ref(output);
    // a row's top k does not depend on the rows beside it: the anchors' rows are ranked and never read
    try b.draft_topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = rows.buf, .off = rows.off }, .{ .rows = h.logits.rows, .vocab = h.logits.width, .k = c.topk, .stride = h.logits.stride });
    b.frame.encoder.barrier();
    b.stamps.commit = ticks();
    b.frame.finish() catch |err| {
        b.failed = true;
        return err;
    };
    b.stamps.waited = ticks();
    b.stamps.gpu = span(&b.frame);
    defer b.stamps.read = ticks();
    var made: usize = 0;
    errdefer for (out[0..made]) |*l| l.deinit();
    for (out, 0..) |*l, k| {
        const first: u32 = @intCast(k * block + 1); // a stream's anchor row drafts nothing
        const hs: op.Head = .{ .logits = try rowsOf(b, h.logits, first, block - 1), .token_ids = h.token_ids };
        l.* = try @import("../qwen27/dflash/runtime_lattice.zig").read(b.allocator, &b.frame, try rowsOf(b, output, first, block - 1), try rowsOf(b, projected, first, block - 1), c, hs);
        made += 1;
    }
}
pub const vtable = op.Ops.VTable{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = context.attention, .swiglu = swiglu, .mask_rows = maskRows, .head = head, .lattice = lattice, .lattices = lattices, .context_begin = context.begin, .context_append = context.append, .context_commit = context.commit, .context_stage = context.stage, .context_abort = context.abort, .stream_append = context.streamAppend };
