//! The batched decode window: k slots' one-row attention, one dispatch a stage, state via a per-forward address table.
const std = @import("std");
const mtl = @import("metal");
const weights = @import("weights.zig");
const config = @import("config.zig");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const common = @import("attention_common.zig");
const View = common.View;
const Seg = common.Seg;
const Pipelines = common.Pipelines;
const Workspace = common.Workspace;
const RowsScratch = common.RowsScratch;
const tensor = common.tensor;
const splitScan = common.splitScan;

var enabled_cache: ?bool = null;

/// The slot phase runs batched by default (same arithmetic); TF_DSV41_BATCH_ATTN=0 restores the per-slot chain.
fn enabled() bool {
    if (enabled_cache == null) {
        enabled_cache = @import("sys").environment.read(bool, "TF_DSV41_BATCH_ATTN", true) catch unreachable;
    }
    return enabled_cache.?;
}

/// Whether this layer's slot phase can run batched: the tensor-unit scan, the bf16 kv norm, one-row segments.
pub fn usable(p: *const Pipelines, w: *const weights.Attention, work: *const Workspace, c: config.Config, segs: []const Seg) bool {
    if (!enabled() or segs.len > RowsScratch.slots or segs.len > work.sel_extra.len + 1) return false;
    if (!p.sparse.rowsReady(c.heads, c.head_dim) or w.kv_norm.dtype != .bf16 or c.head_dim != 512) return false;
    return c.index_topk <= 512 and c.rope_dim <= c.head_dim;
}

fn view(work: *const Workspace, at: usize) View {
    return .{ .buffer = work.rows.tab, .offset = at };
}

/// Writes the per-forward slot table when it changed: positions, live-buffer, selection and per-owner main addresses.
fn upload(p: *const Pipelines, work: *Workspace, segs: []const Seg) !void {
    var pos = std.mem.zeroes([RowsScratch.slots]u32);
    var addr = std.mem.zeroes([RowsScratch.slots * 4]u64);
    var mains = std.mem.zeroes([RowsScratch.slots * 4]u64);
    for (segs, 0..) |sg, s| {
        if (sg.p0 >= p.rotary.capacity) return error.InvalidRotaryShape;
        pos[s] = sg.p0;
        addr[4 * s] = sg.state.live.gpuAddress();
        const sel = work.selectionFor(s).positions();
        addr[4 * s + 2] = sel.buffer.gpuAddress() + sel.offset;
        for (kv.cache.owners, kv.cache.ratios, 0..) |owner, ratio, o| {
            const main = (try sg.state.source(owner, ratio)).main;
            mains[4 * s + o] = main.buffer.gpuAddress() + main.offset;
        }
    }
    var next: [RowsScratch.tab_bytes]u8 = undefined;
    @memcpy(next[RowsScratch.pos_at..][0..@sizeOf(@TypeOf(pos))], std.mem.asBytes(&pos));
    @memcpy(next[RowsScratch.addr_at..][0..@sizeOf(@TypeOf(addr))], std.mem.asBytes(&addr));
    @memcpy(next[RowsScratch.mains_at..][0..@sizeOf(@TypeOf(mains))], std.mem.asBytes(&mains));
    if (work.rows.uploaded_valid and std.mem.eql(u8, &work.rows.uploaded, &next)) return;
    @memcpy(work.rows.tab.contents()[0..RowsScratch.tab_bytes], &next);
    work.rows.uploaded = next;
    work.rows.uploaded_valid = true;
}

fn chunkView(work: *const Workspace) View {
    return .{ .buffer = work.rows.chunk, .offset = 0 };
}

/// Every slot's query rotated and its window row normed, rotated and packed, in two dispatches.
pub fn head(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, work: *Workspace, q: View, raw: View, c: config.Config, ratio: u32, segs: []const Seg) !void {
    try upload(p, work, segs);
    // The kernels reach these buffers by address, so the encoder must be told it uses them (and order them).
    for (segs, 0..) |sg, s| {
        e.useResource(sg.state.live, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.useResource(sg.state.storage, mtl.ResourceUsage.read);
        e.useResource(work.selectionFor(s).buffer, mtl.ResourceUsage.read);
    }
    const n: u32 = @intCast(segs.len);
    const spectrum: rope.Spectrum = if (ratio == 0) .normal else .compressed;
    const pos = view(work, RowsScratch.pos_at);
    p.rotary.encodeRows(e, q, pos, .{ .rows = n, .heads = c.heads, .width = c.head_dim, .rotary = c.rope_dim, .stride = c.heads * c.head_dim }, spectrum);
    const factor_offset: usize = if (ratio == 0) 0 else @as(usize, p.rotary.capacity) * p.rotary.rotary * @sizeOf(f32);
    const factors: View = .{ .buffer = p.rotary.factors, .offset = factor_offset };
    try p.cache.encodeGlueRows(e, raw, tensor(w.kv_norm), factors, chunkView(work), pos, .{
        .rows = n,
        .width = c.head_dim,
        .stride = c.head_dim,
        .packed_stride = kv.cache.ring_stride,
        .block = 32,
        .position = 0,
        .rotary = c.rope_dim,
        .write_y = 0,
        .eps = c.eps,
    });
}

/// Every slot's decode keys, scan and merge; then each slot's window row into its ring; then the output un-rotation.
pub fn tail(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, work: *Workspace, q: View, c: config.Config, layer: u32, ratio: u32, segs: []const Seg) !void {
    const n: u32 = @intCast(segs.len);
    const ring_off: u32 = @intCast((try segs[0].state.ring(layer)).offset);
    const owner: u32 = if (ratio == 0) 0 else @intCast(try kv.cache.ownerFor(layer, ratio));
    const pos = view(work, RowsScratch.pos_at);
    const addr = view(work, RowsScratch.addr_at);
    const dense: View = .{ .buffer = work.rows.dense, .offset = 0 };
    p.sparse.encodeRows(e, q, tensor(w.sink), chunkView(work), dense, work.splitPartials(), pos, addr, view(work, RowsScratch.mains_at), .{
        .rows = n,
        .q_stride = c.heads * c.head_dim,
        .ring_off = ring_off,
        .owner = owner,
        .ratio = ratio,
        .index_topk = c.index_topk,
        .splits = splitScan(),
    });
    e.barrier();
    p.cache.publishRows(e, chunkView(work), addr, pos, ring_off, n);
    const spectrum: rope.Spectrum = if (ratio == 0) .normal else .compressed;
    p.rotary.encodeRows(e, q, pos, .{ .rows = n, .heads = c.heads, .width = c.head_dim, .rotary = c.rope_dim, .stride = c.heads * c.head_dim, .inverse = 1 }, spectrum);
    e.barrier();
}
