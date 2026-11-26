//! A DFlash2 pass that follows a window on the GPU: its rows absorbed ahead, its block from the window's acceptance.
const std = @import("std");
const mtl = @import("metal");
const op = @import("../qwen27/dflash/operators.zig");
const execution = @import("../qwen27/dflash/execution.zig");
const selector = @import("../qwen27/dflash/selector.zig");
const model_mod = @import("metal_dflash_model.zig");
const Model = model_mod.Model;
const ahead = @import("metal_dflash_ahead.zig");
const follow = @import("metal_dflash_follow.zig");
const backend_mod = @import("metal_dflash_backend.zig");
const encode_mod = @import("metal_dflash_encode.zig");
const ring = @import("metal_dflash_context.zig").ring;

/// Where the block's acceptance comes from: the accept kernel's words on the GPU, or the host's (the checks' plan).
pub const Source = union(enum) { gpu: follow.Ref, host: []const u32 };

/// An encoded pass, read once its frame's last buffer completes.
pub const Pass = struct {
    streams: u32,
    block: u32,
    plan: follow.Plan,
    head: op.Head,
    projected: op.View,
    topk: op.View,
    windows: [8]follow.Window = undefined,
};

/// The streams' window rows absorbed ahead, then their blocks planned at their longest contexts, into `frame`.
pub fn encode(m: *Model, kernel: *const follow.Follow, frame: u64, aheads: []const ahead.Ahead, block: u32, src: Source) !Pass {
    const b = &m.backend;
    const c = m.graph.config;
    const n: u32 = @intCast(aheads.len);
    if (n == 0 or n > 8 or b.follow != null or n * block > 128) return error.BadDraftOptions;
    var caches: [8][]backend_mod.Cache = undefined;
    var windows: [8]follow.Window = undefined;
    var streams: [8]execution.Stream = undefined;
    for (aheads, 0..) |x, k| {
        caches[k] = x.context.caches;
        windows[k] = .{ .begin = x.context.session.context.begin, .first = x.first, .rows = x.rows };
        streams[k] = .{ .context = x.context.session.context, .anchor = 0 }; // read anchors from follow buffers
    }
    const p = try follow.plan(c, windows[0..n], block, ring(c));
    b.batch = caches[0..n];
    b.batch_block = if (n > 1) block else 0;
    defer {
        b.batch = &.{};
        b.batch_block = 0;
        b.follow = null;
    }
    try ahead.absorb(m, frame, aheads);
    const rows = p.args.rows;
    const ids = try b.frame.ref(try b.frame.alloc(rows, 1, .u32));
    const positions = try b.frame.ref(try b.frame.alloc(rows, 1, .u64));
    const mask = try b.frame.ref(try b.frame.alloc(rows, p.args.mask_stride, .u8));
    switch (src) {
        .gpu => |accepted| {
            const e = try b.encoder(frame);
            try kernel.encode(e, p, accepted, .{ .buf = ids.buf, .off = ids.off }, .{ .buf = positions.buf, .off = positions.off }, .{ .buf = mask.buf, .off = mask.off });
            e.barrier();
        },
        .host => |accepted| {
            if (accepted.len != 2 * n) return error.BadDraftOptions;
            const at = struct {
                fn of(comptime T: type, r: backend_mod.Ref, count: usize) []T {
                    return @as([*]T, @ptrCast(@alignCast(r.buf.contents() + r.off)))[0..count];
                }
            }.of;
            follow.fill(p, accepted, at(u32, ids, rows), at(u64, positions, rows), at(u8, mask, @as(usize, rows) * p.args.mask_stride));
        },
    }
    b.follow = .{ .ids = ids, .positions = positions, .mask = mask, .keys = p.keys, .streams = n, .mask_stride = p.args.mask_stride };
    const hidden = try encode_mod.encodeLayers(b.ops(), &m.graph, m.preparation, streams[0..n], frame, block);
    b.follow = null;
    const headed: u32 = n * (block - 1);
    const masks = if (n == 1) try b.ops().vtable.mask_rows(b.ops().ptr, frame, hidden, 1, block - 1) else blk: {
        const dst = try b.frame.alloc(headed, c.hidden, .bf16);
        const src_ref = try b.frame.ref(hidden);
        const dst_ref = try b.frame.ref(dst);
        const e = try b.encoder(frame);
        for (0..n) |k| try b.copy(e, .{ .buf = src_ref.buf, .off = src_ref.off + (k * block + 1) * hidden.stride * 2 }, .{ .buf = dst_ref.buf, .off = dst_ref.off + k * (block - 1) * dst.stride * 2 }, block - 1, c.hidden);
        break :blk dst;
    };
    const final = try b.ops().vtable.norm(b.ops().ptr, frame, masks, .{ .weight = m.graph.norm, .eps = c.eps, .heads = 1, .head_dim = c.hidden });
    const head = try b.ops().vtable.head(b.ops().ptr, frame, final);
    if (head.logits.handle == 0 or head.logits.rows != headed or head.logits.dtype != .bf16 or head.logits.width == 0 or head.logits.width > c.vocab or head.logits.stride < head.logits.width) return error.DraftOperatorShape;
    if (head.token_ids) |mapping| try mapping.check(1, head.logits.width, .u32) else if (head.logits.width != c.vocab) return error.DraftCandidateId;
    const projected = try b.ops().vtable.project(b.ops().ptr, frame, final, m.graph.hidden_projection, m.preparation.mode);
    const topk = try backend_mod.encodeTopk(b, frame, head, c.topk);
    return .{ .streams = n, .block = block, .plan = p, .head = head, .projected = projected, .topk = topk, .windows = windows };
}

/// A completed pass's lattices, one a stream, in the order its windows were given.
pub fn lattices(m: *Model, pass: *const Pass, out: []op.Lattice) !void {
    if (out.len != pass.streams) return error.BadDraftOptions;
    try backend_mod.readLattices(&m.backend, pass.head, pass.topk, pass.projected, m.graph.config, pass.block, out);
    for (out) |l| try l.check(m.graph.config);
}

/// Stream `k`'s tree from its lattice, its window having kept `kept` drafts and drawn `anchor`.
pub fn tree(m: *Model, pass: *const Pass, k: usize, lattice: op.Lattice, kept: u32, anchor: u32, options: selector.Options) !selector.Tree {
    const end = pass.windows[k].first + kept + 1;
    return selector.propose(m.allocator, &m.graph, lattice, anchor, end + 1, options);
}

/// A frame on a target window's open buffer and encoder: the pass encodes behind its work and commits nothing.
pub fn beginOn(m: *Model, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !u64 {
    if (m.backend.failed) return error.DraftBackendFailed;
    return m.backend.frame.beginOn(cb, e);
}

/// The window's buffer completed: the pass's lattices are readable until `release`.
pub fn landed(m: *Model) !void {
    try m.backend.frame.landed();
}

/// Encode then wait in a frame of the pass's own (the checks' entry); `release` lets the frame go once it is read.
pub fn run(m: *Model, kernel: *const follow.Follow, aheads: []const ahead.Ahead, block: u32, src: Source) !Pass {
    const ops = m.backend.ops();
    const frame = try ops.vtable.begin(ops.ptr);
    errdefer ops.vtable.release(ops.ptr, frame);
    const pass = try encode(m, kernel, frame, aheads, block, src);
    m.backend.frame.finish() catch |err| {
        m.backend.failed = true;
        return err;
    };
    return pass;
}

/// The pass's frame let go: its lattices are read.
pub fn release(m: *Model) void {
    const ops = m.backend.ops();
    ops.vtable.release(ops.ptr, m.backend.frame.epoch);
}
